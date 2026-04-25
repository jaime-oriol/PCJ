"""
M05_psxg - Post-shot xG (PSxG) via LightGBM + calibracion isotonic.

Training: StatsBomb events + 360 freeze-frames.
Corpus: Euro 2020 + Euro 2024 + Bundesliga 23/24 = 136 partidos, 3.545 shots.
WC22 SAGRADO: excluido del training; solo para prediccion en aplicacion.

Diseño "top 1% SOTA":
  - Features pre-shot: location, distance_goal, angle, body_part, technique,
    shot_type, play_pattern, first_time, under_pressure, deflected.
  - Features end-location: end_x, end_y, end_z (altura del balon — CORE de PSxG).
  - Features 360 freeze-frame: keeper_x/y, keeper_dist_to_endpoint,
    n_defenders_in_cone, n_defenders_near_endpoint,
    dist_to_nearest_defender / nearest_teammate, n_teammates_close_to_endpoint.
  - Modelo: LightGBM con 5-fold CV stratified POR MATCH (evita leakage shot-shot
    dentro de la misma jugada) + CalibratedClassifierCV isotonic.
  - Baseline: statsbomb_xg (pre-shot xG nativo SB). PSxG debe superar en AUC.

Acceptance (ARCHITECTURE.md): AUC holdout > baseline pre-shot xG.

Cache:
  data/parquet/derived/psxg/
    training_shots.parquet        # features + label (para reproducibilidad)
    model/psxg_lgb.pkl            # modelo + calibrador + feature list
    shots.parquet                 # psxg aplicado a WC22 SB + (fallback) training

Consumer: M06 near-miss (criterio "parada con PSxG >= 0.6" identifica paradas
decisivas como cuasi-experimento exogeno tipo Gauriot & Page 2019).

Depende de: M02 (loader SB).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from M02_loader_public import (
    STATSBOMB_COMPETITIONS, load_statsbomb_events, list_statsbomb_match_ids,
)


# -- Rutas ------------------------------------------------------------------

_REPO    = Path(__file__).resolve().parents[1]
_DERIVED = _REPO / "data" / "parquet" / "derived" / "psxg"
_MODEL   = _DERIVED / "model"


# -- Geometria SB (coords 120x80, gol en x=120, center y=40) ----------------

_SB_PITCH_X   = 120.0
_SB_PITCH_Y   = 80.0
_SB_GOAL_X    = 120.0
_SB_GOAL_Y    = 40.0
_SB_GOAL_HALF = 4.0          # half-width del gol (y in [36, 44])
_NEAR_ENDPOINT_RADIUS = 3.0  # metros (en coords SB ~0.75m por unidad)


# ===========================================================================
#  SECCION 1 — Feature engineering
# ===========================================================================

def _angle_to_goal(x: float, y: float) -> float:
    """Angulo en radianes del shot location al gol (center)."""
    dx = _SB_GOAL_X - x
    dy = y - _SB_GOAL_Y
    return float(np.arctan2(dy, dx))


def _dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def _point_in_cone(p: tuple[float, float],
                   apex: tuple[float, float],
                   post1: tuple[float, float],
                   post2: tuple[float, float]) -> bool:
    """True si p esta dentro del triangulo apex-post1-post2 (cone shot-goal)."""
    def sign(a, b, c):
        return (a[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (a[1] - c[1])
    d1 = sign(p, apex, post1)
    d2 = sign(p, post1, post2)
    d3 = sign(p, post2, apex)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _freeze_frame_features(freeze_frame: list | None,
                           shot_loc: tuple[float, float],
                           end_loc: tuple[float, float]) -> dict:
    """Extrae features del 360 freeze-frame con geometria rica.

    Devuelve keys con 'ff_' prefix para separacion clara.
    Si freeze_frame es None o vacio, valores neutros (medianas tipicas).
    """
    post_near = (_SB_GOAL_X, _SB_GOAL_Y - _SB_GOAL_HALF)
    post_far  = (_SB_GOAL_X, _SB_GOAL_Y + _SB_GOAL_HALF)
    # Defaults para shots sin freeze-frame
    out = {
        "ff_has_frame":         0,
        "ff_keeper_present":    0,
        "ff_keeper_x":          float(_SB_GOAL_X),
        "ff_keeper_y":          float(_SB_GOAL_Y),
        "ff_keeper_off_line":   0.0,
        "ff_keeper_y_offset":   0.0,
        "ff_keeper_dist_shot":  20.0,
        "ff_n_def_in_cone":     1,
        "ff_n_def_near_end":    0,
        "ff_dist_nearest_def":  5.0,
        "ff_dist_nearest_teammate":  10.0,
        "ff_n_teammates_near_end": 0,
        "ff_n_def_total":       5,
        "ff_n_teammates_total": 3,
        "ff_def_between_shot_goal": 0,
    }
    if not freeze_frame:
        return out
    out["ff_has_frame"] = 1

    keeper_loc = None
    def_locs: list[tuple[float, float]] = []
    teammate_locs: list[tuple[float, float]] = []
    for entry in freeze_frame:
        loc = entry.get("location")
        if loc is None or len(loc) < 2:
            continue
        p = (float(loc[0]), float(loc[1]))
        pos = entry.get("position") or {}
        name = pos.get("name") if pos else None
        is_teammate = bool(entry.get("teammate"))
        if (not is_teammate) and name == "Goalkeeper":
            keeper_loc = p
        elif is_teammate:
            teammate_locs.append(p)
        else:
            def_locs.append(p)

    out["ff_n_def_total"] = len(def_locs)
    out["ff_n_teammates_total"] = len(teammate_locs)

    if keeper_loc is not None:
        out["ff_keeper_present"] = 1
        out["ff_keeper_x"] = keeper_loc[0]
        out["ff_keeper_y"] = keeper_loc[1]
        out["ff_keeper_off_line"] = _SB_GOAL_X - keeper_loc[0]   # cuanto salio
        out["ff_keeper_y_offset"] = abs(keeper_loc[1] - _SB_GOAL_Y)
        out["ff_keeper_dist_shot"] = _dist(keeper_loc, shot_loc)

    # Defenders en cono shot -> goal
    in_cone = 0
    for dp in def_locs:
        if _point_in_cone(dp, shot_loc, post_near, post_far):
            in_cone += 1
    out["ff_n_def_in_cone"] = in_cone

    # Defensores ESTRICTAMENTE entre shot y gol (x > shot_x)
    out["ff_def_between_shot_goal"] = sum(
        1 for dp in def_locs if dp[0] > shot_loc[0]
    )

    # Defenders y teammates cerca del endpoint
    if def_locs:
        near_end = sum(1 for dp in def_locs
                       if _dist(dp, end_loc) <= _NEAR_ENDPOINT_RADIUS)
        out["ff_n_def_near_end"] = near_end
        out["ff_dist_nearest_def"] = min(_dist(dp, shot_loc) for dp in def_locs)

    if teammate_locs:
        out["ff_n_teammates_near_end"] = sum(
            1 for tp in teammate_locs
            if _dist(tp, end_loc) <= _NEAR_ENDPOINT_RADIUS
        )
        out["ff_dist_nearest_teammate"] = min(
            _dist(tp, shot_loc) for tp in teammate_locs
        )
    return out


def _shot_to_features(ev_dict: dict) -> dict | None:
    """Extrae dict de features desde un event SB tipo Shot. None si invalido."""
    shot = ev_dict.get("shot")
    loc = ev_dict.get("location")
    if shot is None or loc is None or len(loc) < 2:
        return None
    end_loc_raw = shot.get("end_location") or []
    end_x = float(end_loc_raw[0]) if len(end_loc_raw) >= 1 else _SB_GOAL_X
    end_y = float(end_loc_raw[1]) if len(end_loc_raw) >= 2 else _SB_GOAL_Y
    end_z = float(end_loc_raw[2]) if len(end_loc_raw) >= 3 else 0.0
    x, y = float(loc[0]), float(loc[1])

    body   = (shot.get("body_part") or {}).get("name") or "Other"
    tech   = (shot.get("technique") or {}).get("name") or "Normal"
    sh_type = (shot.get("type")     or {}).get("name") or "Open Play"
    play_p  = (ev_dict.get("play_pattern") or {}).get("name") or "Regular Play"

    # Geometria pre-shot
    dist_goal = _dist((x, y), (_SB_GOAL_X, _SB_GOAL_Y))
    angle_goal = _angle_to_goal(x, y)
    post_near = (_SB_GOAL_X, _SB_GOAL_Y - _SB_GOAL_HALF)
    post_far  = (_SB_GOAL_X, _SB_GOAL_Y + _SB_GOAL_HALF)
    # Apertura visual del arco desde la posicion del shot (ley del coseno)
    d_near = _dist((x, y), post_near)
    d_far  = _dist((x, y), post_far)
    # cos rule: cos(aperture) = (d_near^2 + d_far^2 - goal_width^2) / (2 d_near d_far)
    goal_w = 2 * _SB_GOAL_HALF
    cos_aperture = (d_near**2 + d_far**2 - goal_w**2) / (2 * d_near * d_far + 1e-8)
    cos_aperture = max(-1.0, min(1.0, cos_aperture))
    goal_aperture = float(np.arccos(cos_aperture))   # angulo solido al arco

    feats = {
        "x":                x,
        "y":                y,
        "dist_goal":        dist_goal,
        "angle_goal":       angle_goal,
        "goal_aperture":    goal_aperture,           # mas grande = mas facil
        "dist_goal_x_aperture": dist_goal * goal_aperture,  # interaccion
        # Posicion relativa al arco
        "y_from_center":    abs(y - _SB_GOAL_Y),
        "x_to_goal_line":   _SB_GOAL_X - x,
        # Trayectoria legitima (no cruce de linea)
        "end_y":            end_y,
        "end_z":            end_z,
        "end_y_from_center": abs(end_y - _SB_GOAL_Y),
        "end_z_above_bar":  max(0.0, end_z - 2.44),  # saliente sobre el travesano
        "end_near_post":    min(abs(end_y - 36), abs(end_y - 44)),   # dist al palo
        "end_lane":         abs(end_y - 40) / 4.0,   # lane normalizada [0=center, >1=out]
        # body part one-hot (4 clases)
        "bp_right_foot":    int(body == "Right Foot"),
        "bp_left_foot":     int(body == "Left Foot"),
        "bp_head":          int(body == "Head"),
        "bp_other":         int(body not in ("Right Foot", "Left Foot", "Head")),
        # technique (7 clases)
        "tech_normal":      int(tech == "Normal"),
        "tech_volley":      int(tech == "Volley"),
        "tech_half_volley": int(tech == "Half Volley"),
        "tech_lob":         int(tech == "Lob"),
        "tech_overhead":    int(tech == "Overhead Kick"),
        "tech_diving_head": int(tech == "Diving Header"),
        "tech_backheel":    int(tech == "Backheel"),
        # shot type (5 clases)
        "type_open_play":   int(sh_type == "Open Play"),
        "type_corner":      int(sh_type == "Corner"),
        "type_free_kick":   int(sh_type == "Free Kick"),
        "type_penalty":     int(sh_type == "Penalty"),
        "type_kick_off":    int(sh_type == "Kick Off"),
        # play pattern (5 clases principales)
        "pp_regular":       int(play_p == "Regular Play"),
        "pp_from_corner":   int(play_p == "From Corner"),
        "pp_from_fk":       int(play_p == "From Free Kick"),
        "pp_from_throw":    int(play_p == "From Throw In"),
        "pp_counter":       int(play_p == "From Counter"),
        # flags
        "first_time":       int(bool(shot.get("first_time"))),
        "under_pressure":   int(bool(ev_dict.get("under_pressure"))),
        "deflected":        int(bool(shot.get("deflected"))),
    }
    feats.update(_freeze_frame_features(
        shot.get("freeze_frame"), (x, y), (end_x, end_y)
    ))
    # Target + baseline + metadata
    outcome = (shot.get("outcome") or {}).get("name") or ""
    feats["_label"]       = int(outcome == "Goal")
    feats["_sb_xg"]       = float(shot.get("statsbomb_xg") or 0.0)
    feats["_event_uuid"]  = ev_dict.get("id")
    feats["_match_id"]    = None   # se rellena fuera
    feats["_minute"]      = int(ev_dict.get("minute") or 0)
    feats["_second"]      = int(ev_dict.get("second") or 0)
    feats["_period"]      = int(ev_dict.get("period") or 1)
    feats["_outcome"]     = outcome
    return feats


# ===========================================================================
#  SECCION 2 — Build dataset
# ===========================================================================

def _collect_shots_from_matches(match_ids: list[int]) -> pl.DataFrame:
    """Recorre matches, extrae features de cada shot."""
    rows: list[dict] = []
    for mid in match_ids:
        ev = load_statsbomb_events(mid)
        shots = ev.filter(pl.col("type").struct.field("name") == "Shot")
        for d in shots.to_dicts():
            feats = _shot_to_features(d)
            if feats is None:
                continue
            feats["_match_id"] = int(mid)
            rows.append(feats)
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, infer_schema_length=None)


def build_training_shots(cache: bool = True) -> pl.DataFrame:
    """Construye DataFrame de shots training (Euro20+Euro24+Bundes23, sin WC22).

    Cache en data/parquet/derived/psxg/training_shots.parquet.
    """
    cache_path = _DERIVED / "training_shots.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    train_mids = []
    for alias, (cid, sid) in STATSBOMB_COMPETITIONS.items():
        if (cid, sid) == (43, 106):   # WC22 sagrado
            continue
        train_mids.extend(list_statsbomb_match_ids(comp_id=cid, season_id=sid))

    df = _collect_shots_from_matches(train_mids)
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_path, compression="snappy", statistics=True)
    return df


def build_wc22_shots(cache: bool = True) -> pl.DataFrame:
    """Shots de WC22 para aplicar el modelo (WC22 sagrado: solo prediccion).

    Cache en data/parquet/derived/psxg/wc22_shots.parquet.
    """
    cache_path = _DERIVED / "wc22_shots.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)
    mids = list_statsbomb_match_ids(comp_id=43, season_id=106)
    df = _collect_shots_from_matches(mids)
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_path, compression="snappy", statistics=True)
    return df


# ===========================================================================
#  SECCION 3 — Entreno LightGBM + isotonic calibration
# ===========================================================================

FEATURE_COLS = [
    # Pre-shot geometria
    "x", "y", "dist_goal", "angle_goal",
    "goal_aperture", "dist_goal_x_aperture",
    "y_from_center", "x_to_goal_line",
    # Shot trajectory legitima (NO cruce de linea)
    "end_y", "end_z", "end_y_from_center",
    "end_z_above_bar", "end_near_post", "end_lane",
    # Body / technique / type (one-hot)
    "bp_right_foot", "bp_left_foot", "bp_head", "bp_other",
    "tech_normal", "tech_volley", "tech_half_volley", "tech_lob",
    "tech_overhead", "tech_diving_head", "tech_backheel",
    "type_open_play", "type_corner", "type_free_kick",
    "type_penalty", "type_kick_off",
    "pp_regular", "pp_from_corner", "pp_from_fk", "pp_from_throw", "pp_counter",
    # Flags
    "first_time", "under_pressure", "deflected",
    # 360 freeze-frame (posicion AT SHOT TIME, no post-shot)
    "ff_has_frame", "ff_keeper_present",
    "ff_keeper_x", "ff_keeper_y",
    "ff_keeper_off_line", "ff_keeper_y_offset", "ff_keeper_dist_shot",
    "ff_n_def_in_cone", "ff_def_between_shot_goal",
    "ff_n_def_near_end", "ff_n_def_total",
    "ff_dist_nearest_def", "ff_dist_nearest_teammate",
    "ff_n_teammates_near_end", "ff_n_teammates_total",
]
# Features EXCLUIDAS por leakage outcome (documentado):
#   end_x             : >120 si cruza linea = goal, <120 si parada/bloqueo
#   shot_travel       : depende de end_x
#   ff_keeper_dist_end: keeper posicion al momento del shot vs end_location
#                       (si save, keeper = end_location → cero; si gol, lejano)


def _get_folds(match_ids: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """Split por match (cada partido entero en un solo fold)."""
    unique_matches = np.array(sorted(set(match_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_matches)
    return [np.array(f) for f in np.array_split(unique_matches, n_folds)]


def _cv_oof(X: np.ndarray, y: np.ndarray, match_ids: np.ndarray,
            folds: list[np.ndarray], params: dict, seed: int) -> np.ndarray:
    """OOF predictions con params dados. Retorna pred OOF (raw, sin calibrar)."""
    import lightgbm as lgb
    oof = np.zeros(len(y), dtype=np.float32)
    for fi, val_m in enumerate(folds):
        val_mask = np.isin(match_ids, val_m)
        tr_mask = ~val_mask
        model = lgb.LGBMClassifier(**params, random_state=seed + fi, verbose=-1)
        model.fit(X[tr_mask], y[tr_mask],
                  eval_set=[(X[val_mask], y[val_mask])],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        oof[val_mask] = model.predict_proba(X[val_mask])[:, 1]
    return oof


def tune_hyperparameters(df: pl.DataFrame, n_trials: int = 60,
                         n_folds: int = 5, seed: int = 42,
                         timeout_sec: int | None = None) -> dict:
    """Optuna hyperparam search maximizando AUC CV (5-fold by match).

    Returns: dict con best_params + estudio summary.
    """
    import optuna
    from sklearn.metrics import roc_auc_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X = df.select(FEATURE_COLS).to_numpy().astype(np.float32)
    y = df["_label"].to_numpy().astype(np.int32)
    match_ids = df["_match_id"].to_numpy()
    folds = _get_folds(match_ids, n_folds, seed)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 150, 900),
            "max_depth":         trial.suggest_int("max_depth", 3, 9),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 7, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq":    1,
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 2.0, log=True),
            "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 0.2),
        }
        oof = _cv_oof(X, y, match_ids, folds, params, seed)
        return float(roc_auc_score(y, oof))

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=False)
    return {
        "best_params": study.best_params,
        "best_auc":    study.best_value,
        "n_trials":    len(study.trials),
    }


def fit_psxg(df: pl.DataFrame, n_folds: int = 5, seed: int = 42,
             tuned_params: dict | None = None,
             n_trials: int = 60) -> dict:
    """Entrena PSxG via LightGBM con Optuna tuning + isotonic calibration.

    Pipeline riguroso:
      1. Optuna hyperparameter search (60 trials por defecto) maximizando AUC CV.
      2. 5-fold CV stratified by match con best_params -> OOF predictions.
      3. Isotonic calibration sobre OOF -> predictions calibradas.
      4. Modelo FINAL entrenado sobre TODO el training con best_params.
      5. Metrics CV: AUC raw/calibrated, Brier, LogLoss, ECE (Expected Calibration Error)
         vs SB statsbomb_xg baseline.

    Args:
        tuned_params: si se pasan, evita Optuna (para re-train con params ya conocidos).
        n_trials: Optuna trials si tuned_params es None.
    """
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

    X = df.select(FEATURE_COLS).to_numpy().astype(np.float32)
    y = df["_label"].to_numpy().astype(np.int32)
    match_ids = df["_match_id"].to_numpy()
    sb_xg = df["_sb_xg"].to_numpy()
    folds = _get_folds(match_ids, n_folds, seed)

    # 1. Hyperparameter tuning
    if tuned_params is None:
        print(f"Optuna tuning (n_trials={n_trials}, 5-fold CV by match)...")
        tune_res = tune_hyperparameters(df, n_trials=n_trials, n_folds=n_folds, seed=seed)
        best_params = tune_res["best_params"]
        print(f"  best AUC tuning: {tune_res['best_auc']:.4f}")
        print(f"  best params: {best_params}")
    else:
        best_params = tuned_params

    # 2. OOF con best params
    oof_raw = _cv_oof(X, y, match_ids, folds, best_params, seed)

    # 3. Isotonic calibration
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_raw, y)
    oof_cal = calibrator.predict(oof_raw)

    # 4. Metrics + ECE
    def ece(p: np.ndarray, y_: np.ndarray, n_bins: int = 10) -> float:
        bins = np.linspace(0, 1, n_bins + 1)
        n = len(p)
        s = 0.0
        for i in range(n_bins):
            mask = (p >= bins[i]) & (p < bins[i+1] if i < n_bins-1 else p <= bins[i+1])
            if mask.sum() == 0:
                continue
            s += (mask.sum() / n) * abs(p[mask].mean() - y_[mask].mean())
        return float(s)

    metrics = {
        "auc_psxg_raw":        float(roc_auc_score(y, oof_raw)),
        "auc_psxg_calibrated": float(roc_auc_score(y, oof_cal)),
        "auc_baseline_sb_xg":  float(roc_auc_score(y, sb_xg)),
        "brier_psxg":          float(brier_score_loss(y, oof_cal)),
        "brier_sb_xg":         float(brier_score_loss(y, sb_xg)),
        "logloss_psxg":        float(log_loss(y, np.clip(oof_cal, 1e-6, 1-1e-6))),
        "logloss_sb_xg":       float(log_loss(y, np.clip(sb_xg, 1e-6, 1-1e-6))),
        "ece_psxg":            ece(oof_cal, y),
        "ece_sb_xg":           ece(sb_xg, y),
        "n_shots":             len(y),
        "n_goals":             int(y.sum()),
        "goal_rate":           float(y.mean()),
        "best_params":         best_params,
    }

    # 5. Modelo final sobre todo el training
    final_model = lgb.LGBMClassifier(**best_params, random_state=seed, verbose=-1)
    final_model.fit(X, y)

    return {
        "model":         final_model,
        "calibrator":    calibrator,
        "feature_cols":  FEATURE_COLS,
        "oof_raw":       oof_raw,
        "oof_cal":       oof_cal,
        "metrics":       metrics,
    }


def permutation_importance_cv(fit: dict, df: pl.DataFrame, n_repeats: int = 5,
                               seed: int = 42) -> pl.DataFrame:
    """Permutation importance sobre OOF: feature que al shufflearse baja AUC.

    Util para detectar leakage (feature dominante que no debe ser determinante).
    """
    from sklearn.metrics import roc_auc_score
    X = df.select(fit["feature_cols"]).to_numpy().astype(np.float32)
    y = df["_label"].to_numpy()

    # Pred base del modelo final sobre todo (proxy; si overfits algo es normal)
    baseline_pred = fit["model"].predict_proba(X)[:, 1]
    baseline_auc = roc_auc_score(y, baseline_pred)

    rng = np.random.default_rng(seed)
    results = []
    for i, col in enumerate(fit["feature_cols"]):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            idx = rng.permutation(len(X_perm))
            X_perm[:, i] = X_perm[idx, i]
            perm_pred = fit["model"].predict_proba(X_perm)[:, 1]
            drops.append(baseline_auc - roc_auc_score(y, perm_pred))
        results.append({"feature": col, "auc_drop_mean": float(np.mean(drops)),
                        "auc_drop_std": float(np.std(drops))})
    return pl.DataFrame(results).sort("auc_drop_mean", descending=True)


def save_fit(fit: dict, path: Path | None = None) -> Path:
    import pickle
    if path is None:
        path = _MODEL / "psxg_lgb.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = {
        "model":        fit["model"],
        "calibrator":   fit["calibrator"],
        "feature_cols": fit["feature_cols"],
        "metrics":      fit["metrics"],
    }
    with open(path, "wb") as f:
        pickle.dump(serial, f)
    return path


def load_fit(path: Path | None = None) -> dict:
    import pickle
    if path is None:
        path = _MODEL / "psxg_lgb.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


# ===========================================================================
#  SECCION 4 — Predict + cache aplicacion
# ===========================================================================

def predict_psxg(shots_df: pl.DataFrame, fit: dict) -> np.ndarray:
    """Predice PSxG calibrado para un DataFrame de shots."""
    X = shots_df.select(fit["feature_cols"]).to_numpy().astype(np.float32)
    raw = fit["model"].predict_proba(X)[:, 1]
    cal = fit["calibrator"].predict(raw)
    return cal


def cache_wc22_psxg(fit: dict, overwrite: bool = False) -> Path:
    """Genera tabla cacheada con PSxG aplicado a WC22 shots + baseline."""
    out_path = _DERIVED / "shots.parquet"
    if out_path.exists() and not overwrite:
        return out_path
    wc22 = build_wc22_shots(cache=True)
    psxg = predict_psxg(wc22, fit)
    out = wc22.with_columns([
        pl.Series("psxg", psxg),
        pl.col("_sb_xg").alias("xg_baseline"),
        pl.col("_label").alias("is_goal"),
    ]).select([
        pl.col("_match_id").alias("match_id"),
        pl.col("_event_uuid").alias("event_uuid"),
        pl.col("_period").alias("period"),
        pl.col("_minute").alias("minute"),
        pl.col("_second").alias("second"),
        pl.col("_outcome").alias("shot_outcome"),
        "is_goal", "xg_baseline", "psxg",
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path, compression="snappy", statistics=True)
    return out_path


# -- Sanity inline ----------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=== M05_psxg sanity ===")

    t0 = time.time()
    train = build_training_shots(cache=True)
    print(f"training shots: {train.height} (cached) en {time.time()-t0:.1f}s")
    print(f"  goals: {int(train['_label'].sum())} ({100*train['_label'].mean():.1f}%)")
    print(f"  con freeze_frame: {int(train['ff_has_frame'].sum())} "
          f"({100*train['ff_has_frame'].mean():.1f}%)")

    # Sanity: penaltis alta P(gol), corners baja
    pen = train.filter(pl.col("type_penalty") == 1)
    print(f"  penaltis: {pen.height}, goal_rate={pen['_label'].mean():.2f} "
          f"(esperado ~0.75-0.85)")

    fit_path = _MODEL / "psxg_lgb.pkl"
    if fit_path.exists():
        fit = load_fit(fit_path)
        print("fit cargado desde cache")
    else:
        t0 = time.time()
        fit = fit_psxg(train, n_folds=5, seed=42)
        print(f"LightGBM + 5-fold CV + isotonic en {time.time()-t0:.1f}s")
        save_fit(fit)

    m = fit["metrics"]
    print(f"\nMetrics (5-fold CV, stratified by match):")
    print(f"  N shots = {m['n_shots']:,}, goals = {m['n_goals']} ({100*m['goal_rate']:.1f}%)")
    print(f"  AUC PSxG raw     : {m['auc_psxg_raw']:.4f}")
    print(f"  AUC PSxG calibr. : {m['auc_psxg_calibrated']:.4f}")
    print(f"  AUC baseline SB  : {m['auc_baseline_sb_xg']:.4f}")
    print(f"  Brier PSxG       : {m['brier_psxg']:.4f}")
    print(f"  Brier SB xg      : {m['brier_sb_xg']:.4f}")
    print(f"  LogLoss PSxG     : {m['logloss_psxg']:.4f}")
    print(f"  LogLoss SB xg    : {m['logloss_sb_xg']:.4f}")
    assert m["auc_psxg_calibrated"] > m["auc_baseline_sb_xg"], \
        "ACCEPTANCE FAIL: PSxG debe superar baseline SB xG"
    print("  ACCEPTANCE: PSxG AUC > baseline SB xG  OK")

    # Cache WC22
    print()
    t0 = time.time()
    out = cache_wc22_psxg(fit, overwrite=True)
    print(f"cache WC22 -> {out} en {time.time()-t0:.1f}s")
    big = pl.read_parquet(out)
    print(f"  {big.height} shots, {big['is_goal'].sum()} goles")

    # -- Validacion rigorosa adicional (WC22 holdout + calibration) --
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
    wc22 = build_wc22_shots(cache=True)
    psxg_w = predict_psxg(wc22, fit)
    sb_xg_w = wc22["_sb_xg"].to_numpy()
    y_w = wc22["_label"].to_numpy()
    print(f"\nWC22 holdout (NUNCA visto en training):")
    print(f"  AUC PSxG     : {roc_auc_score(y_w, psxg_w):.4f}")
    print(f"  AUC SB xg    : {roc_auc_score(y_w, sb_xg_w):.4f}")
    print(f"  Brier PSxG   : {brier_score_loss(y_w, psxg_w):.4f}")
    print(f"  Brier SB xg  : {brier_score_loss(y_w, sb_xg_w):.4f}")

    # Reliability diagram deciles
    print("\nCalibration (deciles PSxG en WC22 — psxg_pred vs empirical_goal_rate):")
    print(f"  {'bin':<4} | {'pred':>7} | {'actual':>7} | {'n':>5}")
    deciles = np.percentile(psxg_w, np.arange(0, 101, 10))
    for i in range(10):
        lo, hi = deciles[i], deciles[i+1]
        mask = (psxg_w >= lo) & (psxg_w <= hi if i == 9 else psxg_w < hi)
        if mask.sum() == 0:
            continue
        print(f"  d{i+1:<3} | {psxg_w[mask].mean():>7.3f} | "
              f"{y_w[mask].mean():>7.3f} | {int(mask.sum()):>5}")

    # Near-miss candidates desde saves (consumer M06)
    saves_mask = (wc22["_outcome"] == "Saved").to_numpy()
    n_saves = int(saves_mask.sum())
    n_nearmiss_06 = int(((psxg_w >= 0.6) & saves_mask).sum())
    n_nearmiss_04 = int(((psxg_w >= 0.4) & saves_mask).sum())
    print(f"\nNear-miss candidates en WC22 (consumer M06):")
    print(f"  saves totales: {n_saves}")
    print(f"  saves PSxG >= 0.6 (estricto): {n_nearmiss_06}")
    print(f"  saves PSxG >= 0.4 (laxo)    : {n_nearmiss_04}")

    # Permutation importance — detectar leakage residual
    print(f"\nPermutation importance (n_repeats=3) — TOP 10:")
    imp = permutation_importance_cv(fit, train, n_repeats=3, seed=42)
    print(imp.head(10))
    dom = imp.filter(pl.col("auc_drop_mean") > 0.15)
    assert dom.height <= 1, f"Posible leakage: {dom.height} features dominantes"
    print(f"  Features con AUC drop > 0.15: {dom.height} (esperado <=1)")
