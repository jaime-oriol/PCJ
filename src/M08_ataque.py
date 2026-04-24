"""
M08_ataque - Canal Empuje Ofensivo via Atomic-VAEP (Decroos & Davis 2020).

Fase 2 PCJ, canal 1 de 4. Valora la produccion ofensiva on-ball por jugador.
Aisla la contribucion individual del rendimiento de sus companeros (atomic
version, mejor que VAEP clasico para DiD within-player).

Training corpus (WC22 EXCLUIDO, sagrado):
  - Euro 2020     (55, 43)  — 51 partidos
  - Euro 2024     (55, 282) — 51 partidos
  - Bundesliga 23 (9, 281)  — 34 partidos
  Total: 136 partidos -> ~60k-80k atomic actions.

Modelo: CatBoost via Z01_vaep.py (2 modelos: P(scores 10 acc), P(concedes 10 acc)).
Features: 148 cols atomic (actiontype_onehot, bodypart, location, polar, movement).

Pipeline:
  1. Load SB events via StatsBombLoader (socceraction nativo).
  2. Convert to SPADL -> Atomic SPADL (convert_to_atomic).
  3. Extract features + labels (compute_features, compute_labels).
  4. Optuna tuning (TPE, 3-fold CV by match) + train CatBoost 5-fold CV by match
     + isotonic calibration + final model sobre todo el training.
  5. Apply a WC22 atomic actions -> offensive_value per action.
  6. Aggregate: (match_id, player_id_sb, minute, score_atk_minute, n_actions).
  7. Map player_id_sb -> player_id_pff (nombre+equipo exacto, fallback last-name
     unico dentro del equipo).
  8. Aggregate per shock-window (pre/post -10/+10 min).

Output:
  data/parquet/derived/ataque/
    training_atomic.parquet      # atomic actions training (cached)
    wc22_atomic.parquet          # atomic actions WC22 (cached)
    model/vaep_atk_{scores,concedes}.cbm + vaep_atk_meta.pkl
    per_minute.parquet           # (sb_match_id, sb_player_id, minute, score_atk_minute, ...)
    per_shock_window.parquet     # (match_id, shock_id, pff_player_id, shock_type, pre/post)
    sb_to_pff_player_map.parquet # mapping explicito

Acceptance (ARCHITECTURE): distribucion score_atk por rol coherente (CFs > CBs).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# M01/M07 PFF-side
from M01_loader_pff import load_rosters
from M07_shocks import build_shocks_table
# Z01 VAEP infraestructura (CatBoost + socceraction)
import Z01_vaep as vaep_mod

# socceraction
import socceraction.spadl as spadl
from socceraction.data.statsbomb import StatsBombLoader
from socceraction.atomic.spadl import convert_to_atomic, add_names as atomic_add_names


# -- Rutas y constantes ----------------------------------------------------

_REPO        = Path(__file__).resolve().parents[1]
_DERIVED     = _REPO / "data" / "parquet" / "derived" / "ataque"
_MODEL_DIR   = _DERIVED / "model"
_SB_JSON_DIR = _REPO / "data" / "public" / "statsbomb" / "data"

# SB competition x season combos
TRAINING_COMPS = [
    ("Euro20",   55,  43),
    ("Euro24",   55, 282),
    ("Bundes23",  9, 281),
]
WC22_COMP = ("WC22", 43, 106)

SB_LOADER = StatsBombLoader(root=str(_SB_JSON_DIR), getter="local")


# ===========================================================================
#  SECCION 1 — Build Atomic SPADL (training + WC22)
# ===========================================================================

def _build_atomic_actions(comps: list[tuple], cache_name: str,
                          overwrite: bool = False) -> pd.DataFrame:
    """Load SB events de cada (comp_id, season_id) -> Atomic SPADL. Cache parquet."""
    cache_path = _DERIVED / f"{cache_name}_atomic.parquet"
    if cache_path.exists() and not overwrite:
        return pd.read_parquet(cache_path)

    all_atomic = []
    for alias, cid, sid in comps:
        games = SB_LOADER.games(competition_id=cid, season_id=sid)
        print(f"  [{alias}] {len(games)} partidos...", flush=True)
        for _, g in games.iterrows():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                events = SB_LOADER.events(game_id=int(g.game_id))
            try:
                actions = spadl.statsbomb.convert_to_actions(
                    events, home_team_id=int(g.home_team_id)
                )
                atomic = convert_to_atomic(actions)
                atomic["_competition"] = alias
                all_atomic.append(atomic)
            except Exception as e:
                print(f"    skip game {g.game_id}: {e}")
    df = pd.concat(all_atomic, ignore_index=True)
    df = atomic_add_names(df)
    _DERIVED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def build_training_atomic(overwrite: bool = False) -> pd.DataFrame:
    """Atomic SPADL training (Euro20+Euro24+Bundes23, sin WC22)."""
    return _build_atomic_actions(TRAINING_COMPS, "training", overwrite)


def build_wc22_atomic(overwrite: bool = False) -> pd.DataFrame:
    """Atomic SPADL WC22 (aplicacion)."""
    return _build_atomic_actions([WC22_COMP], "wc22", overwrite)


# ===========================================================================
#  SECCION 2 — Train atomic-VAEP via Z01
# ===========================================================================

def _get_match_folds(match_ids: np.ndarray, n_folds: int,
                      seed: int) -> list[np.ndarray]:
    """Split por game_id (cada match va entero a un fold)."""
    uniq = np.array(sorted(set(match_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    return [np.array(f) for f in np.array_split(uniq, n_folds)]


def tune_catboost_hparams(X_all: pd.DataFrame, y: np.ndarray,
                           match_ids: np.ndarray, n_trials: int = 30,
                           seed: int = 42) -> dict:
    """Optuna tuning para CatBoost con 3-fold CV by match (log-loss objective).

    Usamos 3 folds para que cada trial sea barato. Espacio de busqueda:
      depth [4, 8], learning_rate log[0.01, 0.2], l2_leaf_reg log[1, 20],
      bagging_temperature [0, 1], iterations 600 + early stopping.
    """
    import optuna
    from catboost import CatBoostClassifier
    from sklearn.metrics import log_loss

    folds_3 = _get_match_folds(match_ids, 3, seed)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            iterations=600,
            depth=trial.suggest_int("depth", 4, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
            eval_metric="Logloss", task_type="CPU",
            random_seed=seed, verbose=0,
        )
        oof = np.zeros(len(X_all), dtype=np.float32)
        for fi, val_m in enumerate(folds_3):
            val_mask = np.isin(match_ids, val_m)
            tr_mask  = ~val_mask
            m = CatBoostClassifier(**params)
            m.fit(X_all.iloc[tr_mask], y[tr_mask],
                  eval_set=(X_all.iloc[val_mask], y[val_mask]),
                  early_stopping_rounds=40, verbose=0)
            oof[val_mask] = m.predict_proba(X_all.iloc[val_mask])[:, 1]
        return float(log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6)))

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_vaep_model(atomic_df: pd.DataFrame,
                     n_folds: int = 5,
                     seed: int = 42,
                     tune: bool = True,
                     n_trials: int = 30) -> dict:
    """Entrena atomic-VAEP TOP RIGUROSO via CatBoost.

    Pipeline:
      1. Optuna tuning opcional (3-fold CV by match, log-loss).
      2. 5-fold CV stratified by match (evita leakage intra-partido).
      3. OOF predictions raw para AMBOS modelos (scores, concedes).
      4. Isotonic calibration sobre OOF -> calibrated OOF.
      5. Train AUC vs OOF AUC check (delta < 0.05 = no overfitting).
      6. Modelos FINALES entrenados sobre TODO el training.
    """
    from catboost import CatBoostClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

    X_all  = vaep_mod.compute_features(atomic_df, atomic=True, provider="statsbomb_atk")
    ys_all, yc_all = vaep_mod.compute_labels(atomic_df, atomic=True, provider="statsbomb_atk")
    ys = ys_all.values.ravel().astype(np.int32)
    yc = yc_all.values.ravel().astype(np.int32)
    match_ids = atomic_df["game_id"].to_numpy()
    folds = _get_match_folds(match_ids, n_folds, seed)

    # Hyperparam tuning (scores como objetivo; concedes usa mismos hparams -
    # task simetrica, evita duplicar coste de Optuna).
    if tune:
        print(f"  Optuna tuning ({n_trials} trials, 3-fold, scores objective)...",
              flush=True)
        best_hp = tune_catboost_hparams(X_all, ys, match_ids,
                                         n_trials=n_trials, seed=seed)
        print(f"  best hparams: {best_hp}", flush=True)
    else:
        best_hp = dict(depth=6, learning_rate=0.05,
                        l2_leaf_reg=3.0, bagging_temperature=0.5)

    cb_params = dict(
        iterations=800, **best_hp,
        eval_metric="Logloss", task_type="CPU",
        random_seed=seed, verbose=0,
    )

    oof_s = np.zeros(len(X_all), dtype=np.float32)
    oof_c = np.zeros(len(X_all), dtype=np.float32)

    print("  5-fold CV by match (scores + concedes)...", flush=True)
    for fi, val_m in enumerate(folds):
        val_mask = np.isin(match_ids, val_m)
        tr_mask = ~val_mask
        m_s = CatBoostClassifier(**{**cb_params, "random_seed": seed + fi})
        m_s.fit(X_all.iloc[tr_mask], ys[tr_mask],
                eval_set=(X_all.iloc[val_mask], ys[val_mask]),
                early_stopping_rounds=50, verbose=0)
        oof_s[val_mask] = m_s.predict_proba(X_all.iloc[val_mask])[:, 1]
        m_c = CatBoostClassifier(**{**cb_params, "random_seed": seed + fi + 100})
        m_c.fit(X_all.iloc[tr_mask], yc[tr_mask],
                eval_set=(X_all.iloc[val_mask], yc[val_mask]),
                early_stopping_rounds=50, verbose=0)
        oof_c[val_mask] = m_c.predict_proba(X_all.iloc[val_mask])[:, 1]
        print(f"    fold {fi+1}/{n_folds} done", flush=True)

    # Isotonic calibration sobre OOF
    cal_s = IsotonicRegression(out_of_bounds="clip"); cal_s.fit(oof_s, ys)
    cal_c = IsotonicRegression(out_of_bounds="clip"); cal_c.fit(oof_c, yc)
    oof_s_cal = cal_s.predict(oof_s)
    oof_c_cal = cal_c.predict(oof_c)

    # Modelos FINAL sobre todo (con early stopping sobre un random 10% val)
    print("  fitting FINAL models on all training...", flush=True)
    rng = np.random.default_rng(seed)
    val_idx = rng.choice(len(X_all), int(len(X_all)*0.1), replace=False)
    tr_idx = np.setdiff1d(np.arange(len(X_all)), val_idx)
    m_s_final = CatBoostClassifier(**cb_params)
    m_s_final.fit(X_all.iloc[tr_idx], ys[tr_idx],
                  eval_set=(X_all.iloc[val_idx], ys[val_idx]),
                  early_stopping_rounds=50, verbose=0)
    m_c_final = CatBoostClassifier(**{**cb_params, "random_seed": seed + 100})
    m_c_final.fit(X_all.iloc[tr_idx], yc[tr_idx],
                  eval_set=(X_all.iloc[val_idx], yc[val_idx]),
                  early_stopping_rounds=50, verbose=0)

    # Train AUC vs OOF AUC
    train_pred_s = m_s_final.predict_proba(X_all)[:, 1]
    train_pred_c = m_c_final.predict_proba(X_all)[:, 1]
    train_auc_s = roc_auc_score(ys, train_pred_s)
    train_auc_c = roc_auc_score(yc, train_pred_c)

    metrics = {
        "n_actions_total":       int(len(X_all)),
        "n_matches":             int(len(set(match_ids))),
        "auc_scores_oof_raw":    float(roc_auc_score(ys, oof_s)),
        "auc_scores_oof_cal":    float(roc_auc_score(ys, oof_s_cal)),
        "auc_concedes_oof_raw":  float(roc_auc_score(yc, oof_c)),
        "auc_concedes_oof_cal":  float(roc_auc_score(yc, oof_c_cal)),
        "auc_scores_train":      float(train_auc_s),
        "auc_concedes_train":    float(train_auc_c),
        "overfit_delta_scores":  float(train_auc_s - roc_auc_score(ys, oof_s)),
        "overfit_delta_concedes": float(train_auc_c - roc_auc_score(yc, oof_c)),
        "brier_scores_oof":      float(brier_score_loss(ys, oof_s_cal)),
        "brier_concedes_oof":    float(brier_score_loss(yc, oof_c_cal)),
        "logloss_scores":        float(log_loss(ys, np.clip(oof_s_cal, 1e-6, 1-1e-6))),
        "logloss_concedes":      float(log_loss(yc, np.clip(oof_c_cal, 1e-6, 1-1e-6))),
    }
    return {
        "model_s":      m_s_final,
        "model_c":      m_c_final,
        "cal_s":        cal_s,
        "cal_c":        cal_c,
        "metrics":      metrics,
        "oof_s_cal":    oof_s_cal,
        "oof_c_cal":    oof_c_cal,
    }


def save_models(fit: dict, path_prefix: Path | None = None) -> Path:
    """Guarda CatBoost models + calibradores isotonic + metrics."""
    import pickle
    if path_prefix is None:
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path_prefix = _MODEL_DIR / "vaep_atk"
    vaep_mod.save_models(fit["model_s"], fit["model_c"], str(path_prefix))
    # calibradores + metrics en pkl aparte
    with open(f"{path_prefix}_meta.pkl", "wb") as f:
        pickle.dump({
            "cal_s": fit.get("cal_s"),
            "cal_c": fit.get("cal_c"),
            "metrics": fit.get("metrics"),
        }, f)
    return path_prefix


def load_models(path_prefix: Path | None = None) -> dict:
    """Carga CatBoost + calibradores + metrics. Devuelve dict."""
    import pickle
    if path_prefix is None:
        path_prefix = _MODEL_DIR / "vaep_atk"
    m_s, m_c = vaep_mod.load_models(str(path_prefix))
    meta_path = Path(f"{path_prefix}_meta.pkl")
    meta = {}
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
    return {"model_s": m_s, "model_c": m_c, **meta}


# ===========================================================================
#  SECCION 3 — Apply a WC22 + aggregate per player-minute
# ===========================================================================

def apply_vaep_to_wc22(fit: dict,
                       wc22_atomic: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aplica VAEP calibrado a acciones WC22.

    Si fit tiene calibradores (cal_s, cal_c), se usa para P(scores)/P(concedes).
    """
    if wc22_atomic is None:
        wc22_atomic = build_wc22_atomic(overwrite=False)
    X = vaep_mod.compute_features(wc22_atomic, atomic=True, provider="statsbomb_wc22")
    # Schema consistency check vs training
    expected_cols = getattr(fit["model_s"], "feature_names_", None)
    if expected_cols is not None:
        assert list(X.columns) == list(expected_cols), (
            f"Feature mismatch train vs apply. extra={set(X.columns)-set(expected_cols)}; "
            f"missing={set(expected_cols)-set(X.columns)}"
        )
    # Raw predictions
    p_s = fit["model_s"].predict_proba(X)[:, 1]
    p_c = fit["model_c"].predict_proba(X)[:, 1]
    # Isotonic calibration si disponible
    if fit.get("cal_s") is not None:
        p_s = fit["cal_s"].predict(p_s)
    if fit.get("cal_c") is not None:
        p_c = fit["cal_c"].predict(p_c)
    values = vaep_mod._formula_mod(atomic=True).value(
        wc22_atomic.reset_index(drop=True),
        pd.Series(p_s), pd.Series(p_c),
    )
    wc22_atomic = wc22_atomic.copy().reset_index(drop=True)
    wc22_atomic["offensive_value"]  = values["offensive_value"].values
    wc22_atomic["defensive_value"]  = values["defensive_value"].values
    wc22_atomic["vaep_value"]       = values["vaep_value"].values
    return wc22_atomic


def aggregate_per_player_minute(wc22_with_vaep: pd.DataFrame,
                                 cache: bool = True) -> pl.DataFrame:
    """Agrega atomic-VAEP por (match_id, player_id_sb, minute).

    score_atk_minute = sum(offensive_value) de acciones ON-BALL del jugador.
    """
    cache_path = _DERIVED / "per_minute.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    df = pl.from_pandas(wc22_with_vaep[[
        "game_id", "period_id", "time_seconds", "team_id",
        "player_id", "offensive_value", "vaep_value",
    ]])
    df = df.with_columns([
        (pl.col("time_seconds") // 60
         + (pl.col("period_id") - 1) * 45).cast(pl.Int64).alias("minute"),
    ])
    df = df.filter(pl.col("player_id").is_not_null())

    agg = df.group_by(["game_id", "player_id", "minute"]).agg([
        pl.col("offensive_value").sum().alias("score_atk_minute"),
        pl.col("vaep_value").sum().alias("vaep_minute"),
        pl.len().alias("n_actions"),
    ]).rename({"game_id": "sb_match_id", "player_id": "sb_player_id"})

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        agg.write_parquet(cache_path, compression="snappy")
    return agg


# ===========================================================================
#  SECCION 4 — Mapping SB player_id <-> PFF player_id
# ===========================================================================

def build_sb_to_pff_player_map(cache: bool = True) -> pl.DataFrame:
    """Mapea SB player_id -> PFF player_id via (nombre, equipo) join.

    Para WC22: para cada (sb_player_id, sb_name, sb_team_name), busca
    el (pff_player_id, pff_name, pff_team_name) con mismo nombre exacto
    y mismo equipo (dentro del torneo WC22).
    """
    cache_path = _DERIVED / "sb_to_pff_player_map.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    # SB side: usar loader oficial (player_id, player_name) + M02 matches
    # (socceraction games no trae team_name, pero M02 load_statsbomb_matches si).
    from M02_loader_public import load_statsbomb_matches
    sb_matches = load_statsbomb_matches(comp_id=43, season_id=106)
    home_lookup = {int(r["match_id"]): (r["home_team_id"], r["home_team_name"])
                   for r in sb_matches.iter_rows(named=True)}
    away_lookup = {int(r["match_id"]): (r["away_team_id"], r["away_team_name"])
                   for r in sb_matches.iter_rows(named=True)}

    sb_rows = []
    games = SB_LOADER.games(competition_id=43, season_id=106)
    for _, g in games.iterrows():
        gid = int(g.game_id)
        players = SB_LOADER.players(game_id=gid)
        home_tid, home_tname = home_lookup.get(gid, (None, None))
        away_tid, away_tname = away_lookup.get(gid, (None, None))
        for _, p in players.iterrows():
            pid = int(p.team_id)
            team_name = home_tname if pid == home_tid else away_tname
            sb_rows.append({
                "sb_player_id":   int(p.player_id),
                "sb_player_name": p.player_name,
                "team_id_sb":     pid,
                "team_name":      team_name,
            })
    sb_df = pl.DataFrame(sb_rows).unique(subset=["sb_player_id"])

    # PFF side: rosters tiene (player_id, player_name, team_name)
    pff = load_rosters().select([
        pl.col("player_id").alias("pff_player_id"),
        pl.col("player_name").alias("pff_player_name"),
        pl.col("team_name"),
    ]).unique(subset=["pff_player_id"])

    # Normalizacion de nombres: NFKD (quita tildes) + lower + quita puntuacion
    # + colapsa espacios. Ademas derivamos last_token para fallback por apellido.
    import re
    import unicodedata
    _punct_re = re.compile(r"[^a-z0-9 ]+")
    _ws_re    = re.compile(r"\s+")

    def norm(s: str | None) -> str | None:
        if s is None:
            return None
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        s = _punct_re.sub(" ", s.lower())
        return _ws_re.sub(" ", s).strip()

    def last_token(s: str | None) -> str | None:
        if s is None:
            return None
        toks = s.split()
        return toks[-1] if toks else None

    sb_df = sb_df.with_columns([
        pl.col("sb_player_name").map_elements(norm, return_dtype=pl.String).alias("name_norm"),
    ])
    sb_df = sb_df.with_columns(
        pl.col("name_norm").map_elements(last_token, return_dtype=pl.String).alias("last_norm")
    )
    pff = pff.with_columns([
        pl.col("pff_player_name").map_elements(norm, return_dtype=pl.String).alias("name_norm"),
    ])
    pff = pff.with_columns(
        pl.col("name_norm").map_elements(last_token, return_dtype=pl.String).alias("last_norm")
    )

    # Pase 1: full (name_norm, team_name)
    mapping = sb_df.join(
        pff.select(["pff_player_id", "pff_player_name", "team_name", "name_norm"]),
        on=["name_norm", "team_name"], how="left",
    )

    # Pase 2: para SB sin pff_player_id, intentar (last_norm, team_name) si es unico
    #         dentro del equipo en el lado PFF (evita falsos positivos por apellidos
    #         comunes como "Silva" cuando hay varios Silva en la misma seleccion).
    last_unique = pff.group_by(["last_norm", "team_name"]).agg([
        pl.col("pff_player_id").first().alias("pff_id_by_last"),
        pl.len().alias("n_last"),
    ]).filter(pl.col("n_last") == 1)
    mapping = mapping.join(
        last_unique.select(["last_norm", "team_name", "pff_id_by_last"]),
        on=["last_norm", "team_name"], how="left",
    ).with_columns(
        pl.coalesce(["pff_player_id", "pff_id_by_last"]).alias("pff_player_id")
    ).drop("pff_id_by_last")

    mapping = mapping.with_columns([
        pl.col("sb_player_id").cast(pl.Int64),
        pl.col("pff_player_id").cast(pl.Int64, strict=False),
    ]).select(["sb_player_id", "sb_player_name", "team_name", "pff_player_id"])

    if cache:
        mapping.write_parquet(cache_path, compression="snappy")
    return mapping


# ===========================================================================
#  SECCION 5 — Aggregate per shock window (pre/post ±10 min)
# ===========================================================================

def _sb_to_pff_match_map() -> dict[int, int]:
    """Mapping SB match_id -> PFF match_id (mismo que M03)."""
    from M03_preprocess import _pff_to_sb_match_id
    return {v: k for k, v in _pff_to_sb_match_id().items()}


def aggregate_per_shock_window(per_minute: pl.DataFrame,
                                player_map: pl.DataFrame,
                                shocks: pl.DataFrame | None = None,
                                cache: bool = True) -> pl.DataFrame:
    """Por cada (shock, player), suma score_atk en pre/post windows.

    Args:
        per_minute: (sb_match_id, sb_player_id, minute, score_atk_minute, ...).
        player_map: (sb_player_id, pff_player_id) mapping.
        shocks: tabla de M07 (si None, se carga del cache).

    Returns: (match_id, shock_id, player_id_pff, shock_type,
              score_atk_pre, score_atk_post, n_actions_pre, n_actions_post).
    """
    cache_path = _DERIVED / "per_shock_window.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    if shocks is None:
        shocks = build_shocks_table(cache=True, overwrite=False)

    # Map SB match_id -> PFF match_id
    sb2pff = _sb_to_pff_match_map()
    per_min = per_minute.with_columns(
        pl.col("sb_match_id").replace_strict(sb2pff, default=None).alias("match_id")
    ).filter(pl.col("match_id").is_not_null())

    # Map sb_player_id -> pff_player_id (cast to ensure same dtype)
    per_min = per_min.with_columns(pl.col("sb_player_id").cast(pl.Int64))
    pm_cast = player_map.select(["sb_player_id", "pff_player_id"]).with_columns([
        pl.col("sb_player_id").cast(pl.Int64),
        pl.col("pff_player_id").cast(pl.Int64, strict=False),
    ])
    per_min = per_min.join(pm_cast, on="sb_player_id", how="left") \
                      .filter(pl.col("pff_player_id").is_not_null())

    # Join shocks + filter por ventana pre/post:
    # pre: minute in [window_pre_start//60, window_pre_end//60)
    # post: minute in [window_post_start//60, window_post_end//60]
    shocks_slim = shocks.select([
        "match_id", "shock_id", "player_id", "shock_type",
        "window_pre_start", "window_pre_end",
        "window_post_start", "window_post_end",
    ]).rename({"player_id": "pff_player_id"})

    # Join por (match_id, player_id) y luego compute pre/post aggs
    joined = shocks_slim.join(per_min, on=["match_id", "pff_player_id"], how="left")
    # minute en segundos = minute * 60 (approx)
    joined = joined.with_columns(
        (pl.col("minute") * 60).alias("min_sec")
    )
    # Calcular pre/post sums
    pre = joined.filter(
        (pl.col("min_sec") >= pl.col("window_pre_start")) &
        (pl.col("min_sec") < pl.col("window_pre_end"))
    ).group_by(["match_id","shock_id","pff_player_id","shock_type"]).agg([
        pl.col("score_atk_minute").sum().alias("score_atk_pre"),
        pl.col("n_actions").sum().alias("n_actions_pre"),
    ])
    post = joined.filter(
        (pl.col("min_sec") >= pl.col("window_post_start")) &
        (pl.col("min_sec") <= pl.col("window_post_end"))
    ).group_by(["match_id","shock_id","pff_player_id","shock_type"]).agg([
        pl.col("score_atk_minute").sum().alias("score_atk_post"),
        pl.col("n_actions").sum().alias("n_actions_post"),
    ])

    # Full list de (match_id, shock_id, player_id, shock_type) desde shocks
    base = shocks.select([
        "match_id", "shock_id", "player_id", "shock_type"
    ]).rename({"player_id": "pff_player_id"}).unique()

    out = base.join(pre,  on=["match_id","shock_id","pff_player_id","shock_type"], how="left") \
              .join(post, on=["match_id","shock_id","pff_player_id","shock_type"], how="left") \
              .with_columns([
                  pl.col("score_atk_pre").fill_null(0.0),
                  pl.col("score_atk_post").fill_null(0.0),
                  pl.col("n_actions_pre").fill_null(0),
                  pl.col("n_actions_post").fill_null(0),
              ])

    if cache:
        out.write_parquet(cache_path, compression="snappy")
    return out


# -- Sanity inline ---------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=== M08_ataque sanity ===")

    # 1. Build training atomic SPADL (cached)
    t0 = time.time()
    print("\n[1] Building atomic SPADL training (Euro20+Euro24+Bundes23)...")
    train_df = build_training_atomic(overwrite=False)
    print(f"  atomic actions training: {len(train_df):,} en {time.time()-t0:.1f}s")
    print(f"  type_name top 10: {train_df['type_name'].value_counts().head(10).to_dict()}")

    # 2. Build WC22 atomic SPADL (cached)
    t0 = time.time()
    print("\n[2] Building atomic SPADL WC22...")
    wc22_df = build_wc22_atomic(overwrite=False)
    print(f"  atomic actions WC22: {len(wc22_df):,} en {time.time()-t0:.1f}s")

    # 3. Train VAEP model (TOP RIGUROSO: 5-fold CV + isotonic + overfit check)
    model_prefix = _MODEL_DIR / "vaep_atk"
    meta_path = Path(f"{model_prefix}_meta.pkl")
    if (Path(f"{model_prefix}_scores.cbm").exists() and
        Path(f"{model_prefix}_concedes.cbm").exists() and
        meta_path.exists()):
        fit = load_models()
        print("\n[3] VAEP models + calibradores cargados desde cache")
    else:
        print("\n[3] Training CatBoost atomic-VAEP TOP (5-fold CV + isotonic)...")
        t0 = time.time()
        fit = train_vaep_model(train_df, n_folds=5, seed=42)
        print(f"  train completo en {time.time()-t0:.1f}s")
        save_models(fit)

    m = fit["metrics"]
    print(f"\nMetrics CV OOF + overfitting check:")
    print(f"  N actions total: {m['n_actions_total']:,} en {m['n_matches']} matches")
    print(f"  AUC scores OOF raw        : {m['auc_scores_oof_raw']:.4f}")
    print(f"  AUC scores OOF calibrated : {m['auc_scores_oof_cal']:.4f}")
    print(f"  AUC scores TRAIN (final)  : {m['auc_scores_train']:.4f}")
    print(f"  overfit delta scores      : {m['overfit_delta_scores']:+.4f}  "
          f"{'OK' if abs(m['overfit_delta_scores']) < 0.06 else 'CHECK'}")
    print(f"  AUC concedes OOF raw      : {m['auc_concedes_oof_raw']:.4f}")
    print(f"  AUC concedes OOF cal.     : {m['auc_concedes_oof_cal']:.4f}")
    print(f"  AUC concedes TRAIN (final): {m['auc_concedes_train']:.4f}")
    print(f"  overfit delta concedes    : {m['overfit_delta_concedes']:+.4f}  "
          f"{'OK' if abs(m['overfit_delta_concedes']) < 0.06 else 'CHECK'}")
    print(f"  Brier scores / concedes  : {m['brier_scores_oof']:.4f} / {m['brier_concedes_oof']:.4f}")

    # 4. Apply a WC22 (con calibracion)
    t0 = time.time()
    print("\n[4] Aplicando VAEP calibrado a WC22...")
    wc22_with_vaep = apply_vaep_to_wc22(fit, wc22_df)
    print(f"  VAEP applied en {time.time()-t0:.1f}s")
    print(f"  offensive_value range: [{wc22_with_vaep['offensive_value'].min():.3f}, "
          f"{wc22_with_vaep['offensive_value'].max():.3f}]")

    # 5. Aggregate per player-minute
    t0 = time.time()
    print("\n[5] Agregando per player-minute...")
    per_min = aggregate_per_player_minute(wc22_with_vaep, cache=True)
    print(f"  filas: {per_min.height:,} en {time.time()-t0:.1f}s")
    print(f"  cols: {per_min.columns}")
    print(f"  players unicos: {per_min['sb_player_id'].n_unique()}")

    # 6. Mapping SB -> PFF
    print("\n[6] Building SB -> PFF player mapping...")
    mapping = build_sb_to_pff_player_map(cache=True)
    mapped = mapping.filter(pl.col("pff_player_id").is_not_null()).height
    print(f"  mapping rows: {mapping.height}, mapped: {mapped} "
          f"({100*mapped/mapping.height:.1f}%)")

    # 7. Aggregate per shock window
    t0 = time.time()
    print("\n[7] Agregando per shock window...")
    per_shock = aggregate_per_shock_window(per_min, mapping, cache=True)
    print(f"  filas: {per_shock.height:,} en {time.time()-t0:.1f}s")
    print(f"  cols: {per_shock.columns}")
    # Sanity: score_atk_post - score_atk_pre por shock_type
    summary = per_shock.group_by("shock_type").agg([
        pl.col("score_atk_pre").mean().alias("mean_pre"),
        pl.col("score_atk_post").mean().alias("mean_post"),
        (pl.col("score_atk_post") - pl.col("score_atk_pre")).mean().alias("mean_delta"),
    ])
    print("  score_atk por shock_type:")
    print(summary)

    # Sanity acceptance: score_atk por rol (CFs > CBs)
    print("\n[8] Acceptance — distribucion score_atk por rol:")
    pm_cast = per_min.with_columns(pl.col("sb_player_id").cast(pl.Int64))
    map_cast = mapping.select(["sb_player_id","pff_player_id"]).with_columns([
        pl.col("sb_player_id").cast(pl.Int64),
        pl.col("pff_player_id").cast(pl.Int64, strict=False),
    ])
    pm_with_role = pm_cast.join(map_cast, on="sb_player_id", how="left")
    pm_with_role = pm_with_role.filter(pl.col("pff_player_id").is_not_null())
    pm_with_role = pm_with_role.join(
        load_rosters().select(["player_id","position_group"]).unique(subset=["player_id"]).rename({"player_id":"pff_player_id"}),
        on="pff_player_id", how="left",
    )
    by_role = pm_with_role.group_by("position_group").agg([
        pl.col("score_atk_minute").sum().alias("total"),
        pl.col("score_atk_minute").mean().alias("mean_per_minute"),
        pl.len().alias("n_minutes"),
    ]).sort("mean_per_minute", descending=True)
    print(by_role)
