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
  6. Aggregate: (sb_match_id, pff_match_id, sb_player_id, pff_player_id,
     period, minute_in_period, sec_abs, score_atk_minute, vaep_minute, n_actions).
  7. Map player_id_sb -> player_id_pff (cascada 5 pases: exact, tokens-subset
     enriquecido, Levenshtein per-token, difflib SequenceMatcher, manual overrides).
  8. Aggregate per shock-window (pre/post -10/+10 min).

Output:
  data/parquet/derived/ataque/
    training_atomic.parquet      # atomic actions training (cached)
    wc22_atomic.parquet          # atomic actions WC22 (cached)
    model/vaep_atk_{scores,concedes}.cbm + vaep_atk_meta.pkl
    per_minute.parquet           # ambos ids + period + minute_in_period + sec_abs
                                 #   + score_atk_v2_minute (= atomic-VAEP + un-xPass),
                                 #   score_atk_minute (legacy), unxpass_value_minute,
                                 #   vaep_minute, n_actions
    per_shock_window.parquet     # (pff_match_id, sb_match_id, shock_id,
                                 #   pff_player_id, sb_player_id, shock_type) +
                                 #   v2 pre/post + LOO + delta_relative + legacy
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
from M07_shocks import build_shocks_table, attach_team_loo

# socceraction + Z01 se importan lazy en las funciones que los usan
# (evita disparar el bug numpy 2.0 / pandera al solo cargar M08).
def _import_vaep_stack():
    """Lazy: socceraction + Z01_vaep solo cuando se entrena/aplica modelo."""
    import Z01_vaep as vaep_mod
    import socceraction.spadl as spadl
    from socceraction.data.statsbomb import StatsBombLoader
    from socceraction.atomic.spadl import convert_to_atomic, add_names as atomic_add_names
    return vaep_mod, spadl, StatsBombLoader, convert_to_atomic, atomic_add_names


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

_SB_LOADER_CACHE: object | None = None


def _catboost_task_type() -> str:
    """GPU si env CATBOOST_GPU=1 + CUDA disponible, else CPU.

    GPU acelera CatBoost ~3-5x para n_actions ~450k. Requiere catboost build
    con CUDA. Para pods RunPod con GPU, exportar CATBOOST_GPU=1 antes de
    correr el pipeline.
    """
    import os
    if os.environ.get("CATBOOST_GPU", "0") == "1":
        return "GPU"
    return "CPU"


def _get_sb_loader():
    """Lazy SB loader: solo se instancia cuando build_atomic_actions / mapping lo usa."""
    global _SB_LOADER_CACHE
    if _SB_LOADER_CACHE is None:
        _, _, StatsBombLoader, _, _ = _import_vaep_stack()
        _SB_LOADER_CACHE = StatsBombLoader(root=str(_SB_JSON_DIR), getter="local")
    return _SB_LOADER_CACHE


# ===========================================================================
#  SECCION 1 — Build Atomic SPADL (training + WC22)
# ===========================================================================

def _build_atomic_actions(comps: list[tuple], cache_name: str,
                          overwrite: bool = False) -> pd.DataFrame:
    """Load SB events de cada (comp_id, season_id) -> Atomic SPADL. Cache parquet."""
    cache_path = _DERIVED / f"{cache_name}_atomic.parquet"
    if cache_path.exists() and not overwrite:
        return pd.read_parquet(cache_path)

    _, spadl, _, convert_to_atomic, atomic_add_names = _import_vaep_stack()
    sb_loader = _get_sb_loader()

    all_atomic = []
    for alias, cid, sid in comps:
        games = sb_loader.games(competition_id=cid, season_id=sid)
        print(f"  [{alias}] {len(games)} partidos...", flush=True)
        for _, g in games.iterrows():
            # Suprime warnings de socceraction (xy_fidelity_version inferido +
            # FutureWarning de pandas chained-assignment dentro de spadl.statsbomb).
            # Cubre TODAS las llamadas socceraction (events + convert_to_actions
            # + convert_to_atomic), no solo events.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    events = sb_loader.events(game_id=int(g.game_id))
                    actions = spadl.statsbomb.convert_to_actions(
                        events, home_team_id=int(g.home_team_id),
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
            eval_metric="Logloss", task_type=_catboost_task_type(),
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

    vaep_mod, *_ = _import_vaep_stack()
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
        eval_metric="Logloss", task_type=_catboost_task_type(),
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
    vaep_mod, *_ = _import_vaep_stack()
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
    vaep_mod, *_ = _import_vaep_stack()
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
    vaep_mod, *_ = _import_vaep_stack()
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
    values = vaep_mod.formula_mod(atomic=True).value(
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
    """Agrega atomic-VAEP por (sb_match_id, sb_player_id, period, minute_in_period).

    Schema: incluye `pff_match_id`, `pff_player_id`, `sec_abs` (segundos
    absolutos desde inicio de partido) para que M12 DiD pueda alinear con
    M07 windows sin reconvertir time_seconds. minute_in_period es 0..44.
    """
    cache_path = _DERIVED / "per_minute.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    from M03_preprocess import sb_to_pff_match_id

    df = pl.from_pandas(wc22_with_vaep[[
        "game_id", "period_id", "time_seconds", "team_id",
        "player_id", "offensive_value", "vaep_value",
    ]])
    df = df.with_columns([
        (pl.col("time_seconds") // 60).cast(pl.Int64).alias("minute_in_period"),
        # sec_abs: SB period_id 1..5 con periodos de 45 min -> offset (p-1)*45*60.
        # Approx: para period 2 con stoppage de period 1 quedan 60s desfase
        # vs PFF sgc, aceptable porque M07 window son ±600s.
        ((pl.col("period_id") - 1) * 45 * 60
         + pl.col("time_seconds")).cast(pl.Int64).alias("sec_abs"),
    ])
    df = df.filter(pl.col("player_id").is_not_null())

    agg = df.group_by(
        ["game_id", "period_id", "player_id", "minute_in_period"],
    ).agg([
        pl.col("offensive_value").sum().alias("score_atk_minute"),
        pl.col("vaep_value").sum().alias("vaep_minute"),
        pl.col("sec_abs").min().alias("sec_abs"),
        pl.len().cast(pl.Int64).alias("n_actions"),
    ]).rename({
        "game_id":   "sb_match_id",
        "player_id": "sb_player_id",
        "period_id": "period",
    })

    # Anadir pff_match_id (M03 mapping) + pff_player_id (cascada SB→PFF)
    sb2pff_match = sb_to_pff_match_id()
    pmap = build_sb_to_pff_player_map(cache=True).select([
        pl.col("sb_player_id").cast(pl.Int64),
        pl.col("pff_player_id").cast(pl.Int64, strict=False),
    ])
    agg = agg.with_columns([
        pl.col("sb_match_id").cast(pl.Int64),
        pl.col("sb_player_id").cast(pl.Int64),
        pl.col("sb_match_id").replace_strict(sb2pff_match, default=None)
                              .alias("pff_match_id"),
    ]).join(pmap, on="sb_player_id", how="left")

    # un-xPass light (Robberechts 2023): creative decision rating
    unxp_path = _DERIVED.parent / "ataque" / "unxpass" / "per_minute.parquet"
    if unxp_path.exists():
        unxp = pl.read_parquet(unxp_path).select([
            "pff_match_id", "pff_player_id", "period", "minute_in_period",
            "unxpass_value_minute",
        ])
        agg = agg.join(
            unxp, on=["pff_match_id", "pff_player_id",
                       "period", "minute_in_period"],
            how="left",
        ).with_columns(pl.col("unxpass_value_minute").fill_null(0.0))
    else:
        agg = agg.with_columns(pl.lit(0.0).alias("unxpass_value_minute"))

    # Canal ataque v2 SOTA: atomic-VAEP (valor on-ball) + un-xPass (creative
    # decision residual). Captura tanto valor de la accion como
    # "decisiones inesperadas exitosas" (Robberechts 2023 KDD).
    agg = agg.with_columns(
        (pl.col("score_atk_minute") + pl.col("unxpass_value_minute"))
            .alias("score_atk_v2_minute")
    )

    # Orden final canonico: ids -> tiempo -> metricas
    agg = agg.select([
        "pff_match_id", "sb_match_id",
        "pff_player_id", "sb_player_id",
        "period", "minute_in_period", "sec_abs",
        "score_atk_v2_minute",                   # OUTCOME PRINCIPAL: VAEP + un-xPass
        "score_atk_minute",                      # legacy atomic-VAEP solo
        "unxpass_value_minute",                  # componente un-xPass
        "vaep_minute", "n_actions",
    ])

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
    sb_loader = _get_sb_loader()
    games = sb_loader.games(competition_id=43, season_id=106)
    for _, g in games.iterrows():
        gid = int(g.game_id)
        players = sb_loader.players(game_id=gid)
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
    # + colapsa espacios.
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

    sb_df = sb_df.with_columns(
        pl.col("sb_player_name").map_elements(norm, return_dtype=pl.String).alias("name_norm"),
    )
    pff = pff.with_columns(
        pl.col("pff_player_name").map_elements(norm, return_dtype=pl.String).alias("name_norm"),
    )

    # Pase 1: full (name_norm, team_name)
    mapping = sb_df.join(
        pff.select(["pff_player_id", "pff_player_name", "team_name", "name_norm"]),
        on=["name_norm", "team_name"], how="left",
    )

    # Enrichment: cargar firstName + lastName del CSV original (data_mundial/
    # players.csv) para cubrir casos donde PFF roster usa solo el nickname
    # (Rodri en lugar de Rodrigo Hernandez Cascante, Koke en lugar de Jorge
    # Resurreccion, etc). El SB nombre completo si lleva los apellidos.
    csv_path = _REPO / "data_mundial" / "players.csv"
    if csv_path.exists():
        players_csv = pl.read_csv(csv_path).select([
            pl.col("id").cast(pl.Int64).alias("pff_player_id"),
            "firstName", "lastName",
        ]).unique(subset=["pff_player_id"])
        pff = pff.join(players_csv, on="pff_player_id", how="left")
        # Combina nickname + firstName + lastName en tokens enriquecidos
        def combine_tokens(s_nick, s_first, s_last):
            parts = []
            for s in (s_nick, s_first, s_last):
                if s: parts.append(norm(s))
            return " ".join(parts)
        pff = pff.with_columns(
            pl.struct(["pff_player_name", "firstName", "lastName"]).map_elements(
                lambda r: combine_tokens(r["pff_player_name"], r["firstName"], r["lastName"]),
                return_dtype=pl.String,
            ).alias("name_norm_enriched")
        )
    else:
        pff = pff.with_columns(pl.col("name_norm").alias("name_norm_enriched"))

    # Pase 2: tokens-subset matching con name_norm_enriched (nickname +
    # firstName + lastName del CSV). Cubre:
    #   - Apellidos hispanos: SB "Pablo Sarabia García" → PFF "Pablo Sarabia"
    #   - Nicknames: SB "Rodrigo Hernández Cascante" → PFF "Rodri" (firstName
    #     "Rodrigo", lastName "Hernández" del CSV → tokens "rodri rodrigo
    #     hernandez")
    #   - Diacriticos: SB "Højbjerg" matches PFF tokens via NFKD strip ASCII.
    # Match si UNICO en equipo y al menos 2 tokens coinciden, o si todos los
    # PFF tokens estan en SB tokens. Evita falsos positivos one-token-share.
    pff_tokens = pff.with_columns(
        pl.col("name_norm_enriched").str.split(" ").alias("pff_tokens_e")
    ).select(["pff_player_id", "team_name", "pff_tokens_e", "name_norm_enriched"])
    unmapped_sb = mapping.filter(pl.col("pff_player_id").is_null()).select([
        "sb_player_id", "team_name", "name_norm",
    ]).with_columns(pl.col("name_norm").str.split(" ").alias("sb_tokens"))

    rows_pass2 = []
    if unmapped_sb.height > 0:
        for sb_row in unmapped_sb.iter_rows(named=True):
            cands = pff_tokens.filter(pl.col("team_name") == sb_row["team_name"])
            sb_set = set(sb_row["sb_tokens"]) - {""}
            matches = []
            for c in cands.iter_rows(named=True):
                pff_set = set(c["pff_tokens_e"]) - {""}
                if not pff_set:
                    continue
                inter = pff_set & sb_set
                # Match: PFF tokens ⊆ SB (caso "Pablo Sarabia" ⊆ "Pablo
                # Sarabia García"), O al menos 2 tokens significativos en
                # comun (cubre nicknames con firstName/lastName del CSV).
                if pff_set.issubset(sb_set) or len(inter) >= 2:
                    matches.append((c["pff_player_id"], c["name_norm_enriched"]))
            if len(matches) == 1:
                rows_pass2.append({
                    "sb_player_id": sb_row["sb_player_id"],
                    "pff_id_pass2": matches[0][0],
                })
    if rows_pass2:
        pass2_df = pl.DataFrame(rows_pass2, schema={
            "sb_player_id": pl.Int64, "pff_id_pass2": pl.Int64,
        })
        mapping = mapping.join(pass2_df, on="sb_player_id", how="left").with_columns(
            pl.coalesce(["pff_player_id", "pff_id_pass2"]).alias("pff_player_id")
        ).drop("pff_id_pass2")

    # Pase 3: fuzzy Levenshtein distance per-token. Cubre casos edge que
    # pase 2 perdio: diacriticos especiales (Mæhle, Højbjerg → NFKD strip),
    # nombres con caracteres no-latinos transliterados, abreviaciones (Maxi
    # vs Maximiliano). Match si: para CADA token PFF significativo (>=4
    # chars), existe un token SB con Levenshtein <= 2. UNICO en equipo.
    def lev(a: str, b: str, max_d: int = 2) -> int:
        if abs(len(a) - len(b)) > max_d: return max_d + 1
        if not a: return len(b)
        if not b: return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i] + [0] * len(b)
            for j, cb in enumerate(b, 1):
                cur[j] = min(cur[j-1] + 1, prev[j] + 1,
                              prev[j-1] + (0 if ca == cb else 1))
            prev = cur
            if min(prev) > max_d: return max_d + 1
        return prev[-1]

    unmapped_sb2 = mapping.filter(pl.col("pff_player_id").is_null()).select([
        "sb_player_id", "team_name", "name_norm",
    ]).with_columns(pl.col("name_norm").str.split(" ").alias("sb_tokens"))
    rows_pass3 = []
    if unmapped_sb2.height > 0:
        for sb_row in unmapped_sb2.iter_rows(named=True):
            cands = pff_tokens.filter(pl.col("team_name") == sb_row["team_name"])
            sb_set = [t for t in sb_row["sb_tokens"] if len(t) >= 3]
            matches = []
            for c in cands.iter_rows(named=True):
                pff_set = [t for t in c["pff_tokens_e"] if len(t) >= 4]
                if not pff_set:
                    continue
                # Para cada PFF token, hay un SB token con Levenshtein <=2?
                fuzzy_hits = sum(
                    1 for pt in pff_set
                    if any(lev(pt, st) <= 2 for st in sb_set)
                )
                if fuzzy_hits >= max(2, len(set(pff_set))):
                    matches.append((c["pff_player_id"], c["name_norm_enriched"]))
            unique_matches = list({m[0]: m for m in matches}.values())
            if len(unique_matches) == 1:
                rows_pass3.append({
                    "sb_player_id": sb_row["sb_player_id"],
                    "pff_id_pass3": unique_matches[0][0],
                })
    if rows_pass3:
        pass3_df = pl.DataFrame(rows_pass3, schema={
            "sb_player_id": pl.Int64, "pff_id_pass3": pl.Int64,
        })
        mapping = mapping.join(pass3_df, on="sb_player_id", how="left").with_columns(
            pl.coalesce(["pff_player_id", "pff_id_pass3"]).alias("pff_player_id")
        ).drop("pff_id_pass3")

    # Pase 4: difflib SequenceMatcher ratio sobre nombre enriquecido completo.
    # Cubre transliteraciones arabes ("Salman Mohammed Al Faraj" vs "Salman
    # Al-Faraj") y abreviaciones ("Phil" vs "Philip"). Match si ratio >= 0.55
    # Y es el UNICO PFF del equipo arriba de ese umbral.
    from difflib import SequenceMatcher
    unmapped_sb3 = mapping.filter(pl.col("pff_player_id").is_null()).select([
        "sb_player_id", "team_name", "name_norm",
    ])
    rows_pass4 = []
    if unmapped_sb3.height > 0:
        for sb_row in unmapped_sb3.iter_rows(named=True):
            cands = pff_tokens.filter(pl.col("team_name") == sb_row["team_name"])
            sb_name = sb_row["name_norm"]
            scored = []
            for c in cands.iter_rows(named=True):
                ratio = SequenceMatcher(None, sb_name, c["name_norm_enriched"]).ratio()
                if ratio >= 0.55:
                    scored.append((ratio, c["pff_player_id"]))
            scored.sort(reverse=True)
            # match si el TOP es claramente mejor que el siguiente (gap > 0.10)
            if scored and (len(scored) == 1 or scored[0][0] - scored[1][0] > 0.10):
                rows_pass4.append({
                    "sb_player_id": sb_row["sb_player_id"],
                    "pff_id_pass4": scored[0][1],
                })
    if rows_pass4:
        pass4_df = pl.DataFrame(rows_pass4, schema={
            "sb_player_id": pl.Int64, "pff_id_pass4": pl.Int64,
        })
        mapping = mapping.join(pass4_df, on="sb_player_id", how="left").with_columns(
            pl.coalesce(["pff_player_id", "pff_id_pass4"]).alias("pff_player_id")
        ).drop("pff_id_pass4")

    # Pase 5: manual overrides hardcoded para los casos edge donde ningun
    # algoritmo automatico llega (transliteraciones arabes con tokens
    # ambiguos, abreviaciones idiosincraticas). Validados manualmente
    # contra rosters PFF (data_mundial/players.csv + nombres oficiales WC22).
    # Cada entry justificada por inspeccion del nombre completo SB vs el
    # PFF roster del mismo equipo.
    MANUAL_OVERRIDES_SB_TO_PFF: dict[str, int] = {
        # Saudi Arabia (transliteracion "Al X" vs "Al-X"):
        "Hassan Mohammed Al-Tambakti":      13996,  # Hassan Tambakti
        "Salman Mohammed Al Faraj":         13999,  # Salman Al-Faraj
        "Salem Mohammed Al Dawsari":        13998,  # Salem Al-Dawsari (top-scorer KSA)
        "Mohammed Khalil Al Owais":         13987,  # Mohammed Al-Owais
        "Abdulelah Saad Hameed Al-Malki":   14004,  # Abdulelah Al-Malki
        "Nawaf Shaker Al Abid":             14000,  # Nawaf Al-Abed
        "Firas Tariq Nasser Al Albirakan":  14010,  # Firas Al-Buraikan
        "Mohammed Awad Khalifa Kanoo":      None,   # NO esta en PFF roster
        # Qatar:
        "Ali Assadalla Thaimn Qambar":      None,   # ambiguo, varios Ali
        "Hassan Khalid Al Heidos":          None,   # no tiene contrapartida clara
        "Abdulkarim Hassan Fadlalla":       None,
        "Mohammed Waed Abdulwahhab Al Bayati": None,
        "Boualem Khoukhi":                  None,
        "Karim Boudiaf":                    None,
        "Ahmed Alaa Eldin Abdelmotaal":     13964,  # Ahmed Alaaeldin
        # Hispanos/Portugues:
        "Alejandro Darío Gómez":            8420,   # Papu Gómez
        "Rúben Diogo Da Silva Neves":       240,    # Rúben Neves
        "Anssumane Fati":                   1529,   # Ansu Fati
        "Daniel Alves da Silva":            9330,   # Dani Alves
    }
    if MANUAL_OVERRIDES_SB_TO_PFF:
        manual_rows = [
            {"sb_player_name": k, "pff_id_manual": v}
            for k, v in MANUAL_OVERRIDES_SB_TO_PFF.items() if v is not None
        ]
        if manual_rows:
            manual_df = pl.DataFrame(manual_rows, schema={
                "sb_player_name": pl.String, "pff_id_manual": pl.Int64,
            })
            mapping = mapping.join(manual_df, on="sb_player_name", how="left").with_columns(
                pl.coalesce(["pff_player_id", "pff_id_manual"]).alias("pff_player_id")
            ).drop("pff_id_manual")

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

def aggregate_per_shock_window(per_minute: pl.DataFrame,
                                player_map: pl.DataFrame,
                                shocks: pl.DataFrame | None = None,
                                cache: bool = True) -> pl.DataFrame:
    """Por cada (shock, player), suma score_atk en pre/post windows.

    Schema: publica `pff_match_id`, `sb_match_id`, `pff_player_id`,
    `sb_player_id` y filtra por `sec_abs` real (no minute*60 sintetico).

    Args:
        per_minute: schema canonico M08.aggregate_per_player_minute, con
                    cols `pff_match_id`, `sb_match_id`, `pff_player_id`,
                    `sb_player_id`, `sec_abs`, `score_atk_minute`, `n_actions`.
        player_map: (sb_player_id, pff_player_id) mapping (no usado si
                    per_minute ya trae pff_player_id; param mantenido por
                    compatibilidad).
        shocks: tabla de M07 (si None, se carga del cache).
    """
    cache_path = _DERIVED / "per_shock_window.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    if shocks is None:
        shocks = build_shocks_table(cache=True, overwrite=False)

    per_min = per_minute.filter(
        pl.col("pff_match_id").is_not_null() &
        pl.col("pff_player_id").is_not_null()
    ).rename({"pff_match_id": "match_id"})

    shocks_slim = shocks.select([
        "match_id", "shock_id", "player_id", "shock_type",
        pl.col("period").alias("shock_period"),
        "window_pre_start", "window_pre_end",
        "window_post_start", "window_post_end",
    ]).rename({"player_id": "pff_player_id"})

    joined = shocks_slim.join(
        per_min, on=["match_id", "pff_player_id"], how="left",
    )

    # Filtra por sec_abs real Y period == shock_period. SB sec_abs = (p-1)*45*60
    # + time_seconds, asi que un evento period 1 minuto 47 stoppage (sec_abs=2820)
    # colisiona con un evento period 2 minuto 2 (tambien 2820). Sin filtrar
    # period, ~8% de eventos contaminan cross-period (medido empiricamente).
    # Pre [t-600, t), post (t, t+600].
    pre = joined.filter(
        (pl.col("sec_abs") >= pl.col("window_pre_start")) &
        (pl.col("sec_abs") < pl.col("window_pre_end")) &
        (pl.col("period") == pl.col("shock_period"))
    ).group_by(["match_id","shock_id","pff_player_id","shock_type"]).agg([
        pl.col("score_atk_v2_minute").sum().alias("score_atk_v2_pre"),
        pl.col("score_atk_minute").sum().alias("score_atk_pre"),
        pl.col("unxpass_value_minute").sum().alias("unxpass_pre"),
        pl.col("n_actions").sum().cast(pl.Int64).alias("n_actions_pre"),
    ])
    post = joined.filter(
        (pl.col("sec_abs") >= pl.col("window_post_start")) &
        (pl.col("sec_abs") <= pl.col("window_post_end")) &
        (pl.col("period") == pl.col("shock_period"))
    ).group_by(["match_id","shock_id","pff_player_id","shock_type"]).agg([
        pl.col("score_atk_v2_minute").sum().alias("score_atk_v2_post"),
        pl.col("score_atk_minute").sum().alias("score_atk_post"),
        pl.col("unxpass_value_minute").sum().alias("unxpass_post"),
        pl.col("n_actions").sum().cast(pl.Int64).alias("n_actions_post"),
    ])

    base = shocks.select([
        "match_id", "shock_id", "player_id", "shock_type"
    ]).rename({"player_id": "pff_player_id"}).unique()

    pff_to_sb_pl = player_map.select([
        pl.col("pff_player_id").cast(pl.Int64, strict=False),
        pl.col("sb_player_id").cast(pl.Int64),
    ]).filter(pl.col("pff_player_id").is_not_null()).unique(
        subset=["pff_player_id"], keep="first",
    )

    from M03_preprocess import pff_to_sb_match_id
    pff2sb_match = pff_to_sb_match_id()

    out = (
        base
        .join(pre,  on=["match_id","shock_id","pff_player_id","shock_type"],
              how="left")
        .join(post, on=["match_id","shock_id","pff_player_id","shock_type"],
              how="left")
        .with_columns([
            pl.col("score_atk_v2_pre").fill_null(0.0),
            pl.col("score_atk_v2_post").fill_null(0.0),
            pl.col("score_atk_pre").fill_null(0.0),
            pl.col("score_atk_post").fill_null(0.0),
            pl.col("unxpass_pre").fill_null(0.0),
            pl.col("unxpass_post").fill_null(0.0),
            pl.col("n_actions_pre").fill_null(0),
            pl.col("n_actions_post").fill_null(0),
        ])
    )

    pm_for_loo = per_minute.filter(
        pl.col("pff_match_id").is_not_null()
        & pl.col("pff_player_id").is_not_null()
    )

    # LOO outcome principal v2 (atomic-VAEP + un-xPass)
    loo_v2 = attach_team_loo(
        pm_for_loo, value_col="score_atk_v2_minute",
    ).rename({
        "score_atk_v2_minute_team_loo_pre":  "score_atk_v2_team_loo_pre",
        "score_atk_v2_minute_team_loo_post": "score_atk_v2_team_loo_post",
        "score_atk_v2_minute_relative_pre":  "score_atk_v2_relative_pre",
        "score_atk_v2_minute_relative_post": "score_atk_v2_relative_post",
        "score_atk_v2_minute_delta_player":  "score_atk_v2_delta_player",
        "score_atk_v2_minute_delta_team_loo":"score_atk_v2_delta_team_loo",
        "score_atk_v2_minute_delta_relative":"score_atk_v2_delta_relative",
    }).select([
        "match_id", "shock_id", "pff_player_id", "shock_type",
        "score_atk_v2_team_loo_pre", "score_atk_v2_team_loo_post",
        "score_atk_v2_relative_pre", "score_atk_v2_relative_post",
        "score_atk_v2_delta_player", "score_atk_v2_delta_team_loo",
        "score_atk_v2_delta_relative", "n_block",
    ])

    # LOO legacy atomic-VAEP (sensitivity)
    loo = attach_team_loo(
        pm_for_loo, value_col="score_atk_minute",
    ).rename({
        "score_atk_minute_team_loo_pre":  "score_atk_team_loo_pre",
        "score_atk_minute_team_loo_post": "score_atk_team_loo_post",
        "score_atk_minute_relative_pre":  "score_atk_relative_pre",
        "score_atk_minute_relative_post": "score_atk_relative_post",
        "score_atk_minute_delta_player":  "score_atk_delta_player",
        "score_atk_minute_delta_team_loo":"score_atk_delta_team_loo",
        "score_atk_minute_delta_relative":"score_atk_delta_relative",
    }).select([
        "match_id", "shock_id", "pff_player_id", "shock_type",
        "score_atk_team_loo_pre", "score_atk_team_loo_post",
        "score_atk_relative_pre", "score_atk_relative_post",
        "score_atk_delta_player", "score_atk_delta_team_loo",
        "score_atk_delta_relative",
    ])

    out = (
        out
        .join(loo_v2, on=["match_id","shock_id","pff_player_id","shock_type"],
              how="left")
        .join(loo, on=["match_id","shock_id","pff_player_id","shock_type"],
              how="left")
        .rename({"match_id": "pff_match_id"})
        .join(pff_to_sb_pl, on="pff_player_id", how="left")
        .with_columns(
            pl.col("pff_match_id").replace_strict(pff2sb_match, default=None)
                                    .alias("sb_match_id")
        )
        .select([
            "pff_match_id", "sb_match_id",
            "shock_id", "shock_type",
            "pff_player_id", "sb_player_id",
            # Outcome principal v2 SOTA: atomic-VAEP + un-xPass
            "score_atk_v2_pre", "score_atk_v2_post",
            "score_atk_v2_team_loo_pre", "score_atk_v2_team_loo_post",
            "score_atk_v2_relative_pre", "score_atk_v2_relative_post",
            "score_atk_v2_delta_player", "score_atk_v2_delta_team_loo",
            "score_atk_v2_delta_relative",
            # Legacy (sensitivity)
            "score_atk_pre", "score_atk_post",
            "score_atk_team_loo_pre", "score_atk_team_loo_post",
            "score_atk_relative_pre", "score_atk_relative_post",
            "score_atk_delta_player", "score_atk_delta_team_loo",
            "score_atk_delta_relative",
            "unxpass_pre", "unxpass_post",
            "n_actions_pre", "n_actions_post", "n_block",
        ])
    )

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

    # Sanity acceptance: score_atk por rol (CFs > CBs).
    # per_min ya trae pff_player_id desde aggregate_per_player_minute.
    print("\n[8] Acceptance — distribucion score_atk por rol:")
    pm_with_role = per_min.filter(pl.col("pff_player_id").is_not_null()).join(
        load_rosters().select(["player_id","position_group"])
                       .unique(subset=["player_id"])
                       .rename({"player_id":"pff_player_id"}),
        on="pff_player_id", how="left",
    )
    by_role = pm_with_role.group_by("position_group").agg([
        pl.col("score_atk_minute").sum().alias("total"),
        pl.col("score_atk_minute").mean().alias("mean_per_minute"),
        pl.len().alias("n_minutes"),
    ]).sort("mean_per_minute", descending=True)
    print(by_role)
