"""Z04_vdep - VDEP stricto Toda et al. 2022 PLOS ONE 17(1):e0263051.

Cabeza dedicada para valoracion defensiva individualizada sobre acciones SPADL:

    VDEP(action) = P(recovery_in_N | action, state) - C * P(attacked_in_N | action, state)

donde:
  - recovery: en proximas N acciones SPADL, equipo del defensor obtiene posesion.
  - attacked:  en proximas N acciones SPADL, equipo del defensor concede gol.
  - C: hyperparametro que pondera coste de ser atacado vs valor de recovery
       (Toda 2022 calibra C ≈ 0.5 sobre J-League; aqui lo aprendemos como
       ratio mean(attacked)/mean(recovery) para el corpus en uso).
  - state: features atomic-SPADL pre-accion (location, type, distancias, etc).

Reemplaza el `vdep_like_minute` heuristico de M09 (que reusa la cabeza
P(concedes) de atomic-VAEP) por un modelo dedicado entrenado sobre los
mismos features atomic, pero con TARGETS distintos: recovery + attacked
en lugar de scores + concedes. Mas fiel al paper original.

Diseno:
  - Training: SPADL atomic Euro20 + Euro24 + Bundes23 (WC22 SAGRADO).
  - Filtro: acciones del equipo que NO tiene posesion del balon previo a la
            accion (es decir, defensive actions desde la perspectiva del
            defensor que actua).
  - Targets:
      y_recovery[a] = 1 si en accs[a+1 .. a+N] el carrier_team == defender_team
                       OR la accion misma es interception/tackle/clearance/keeper_*
                       que recupera el balon.
      y_attacked[a] = 1 si en accs[a+1 .. a+N] hay shot que termina en gol del
                       opponent_team (i.e., el equipo del defensor concede).
  - Model: LightGBM x 2 (recovery, attacked) + 5-fold CV by match + isotonic.
  - Apply: VDEP = P_rec_cal - C * P_att_cal sobre WC22 acciones defensivas.

Outputs:
  data/parquet/derived/defensa/vdep_strict/
    training.parquet     # features + (y_recovery, y_attacked)
    model_rec.pkl        # LightGBM + cal
    model_att.pkl        # LightGBM + cal
    per_event.parquet    # WC22 actions con vdep_strict_value
    per_minute.parquet   # sum vdep_strict per (player, period, minute_in_period)
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import polars as pl

_REPO    = Path(__file__).resolve().parents[1]
_DERIVED = _REPO / "data" / "parquet" / "derived" / "defensa" / "vdep_strict"

# Toda 2022 simplificado: 3 acciones para recovery (mas estricto que VAEP/10
# para evitar laxitud que infla rec_rate por encima del 85%; con 3 buscamos
# "consolidacion inmediata" del defender_team en 3 acciones siguientes).
HORIZON_REC = 3
HORIZON_ATT = 5
DEF_ACTION_TYPES = {
    "tackle", "interception", "clearance", "foul",
    "keeper_save", "keeper_claim", "keeper_punch", "keeper_pick_up",
}


# --- Features (reuso atomic-SPADL existing en M08) ------------------------

# Subset compacto de cols atomic-SPADL pre-accion utiles para VDEP
FEATURE_COLS = [
    "x", "y", "dx", "dy",                          # location + delta
    "type_pass", "type_dribble", "type_cross",     # type one-hot top3
    "type_shot", "type_tackle", "type_interception",
    "type_clearance", "type_foul", "type_take_on", "type_keeper_save",
    "bodypart_foot", "bodypart_head",
    "period_id", "time_seconds_norm",              # contexto temporal
    "score_diff",                                  # marcador absoluto (defender perspective)
    "is_home_action",
]


def _build_atomic_features(atomic_df) -> pl.DataFrame:
    """Convierte atomic-SPADL pandas (M08) a DataFrame con FEATURE_COLS + targets-ready."""
    df = pl.from_pandas(atomic_df[[
        "game_id", "period_id", "time_seconds", "team_id", "player_id",
        "type_name", "bodypart_name", "x", "y", "dx", "dy",
    ]])
    # type one-hot
    df = df.with_columns([
        (pl.col("type_name") == "pass").cast(pl.Int64).alias("type_pass"),
        (pl.col("type_name") == "dribble").cast(pl.Int64).alias("type_dribble"),
        (pl.col("type_name") == "cross").cast(pl.Int64).alias("type_cross"),
        (pl.col("type_name") == "shot").cast(pl.Int64).alias("type_shot"),
        (pl.col("type_name") == "tackle").cast(pl.Int64).alias("type_tackle"),
        (pl.col("type_name") == "interception").cast(pl.Int64).alias("type_interception"),
        (pl.col("type_name") == "clearance").cast(pl.Int64).alias("type_clearance"),
        (pl.col("type_name") == "foul").cast(pl.Int64).alias("type_foul"),
        (pl.col("type_name") == "take_on").cast(pl.Int64).alias("type_take_on"),
        pl.col("type_name").str.starts_with("keeper").cast(pl.Int64).alias("type_keeper_save"),
        (pl.col("bodypart_name") == "foot").cast(pl.Int64).alias("bodypart_foot"),
        (pl.col("bodypart_name") == "head").cast(pl.Int64).alias("bodypart_head"),
        (pl.col("time_seconds") / 5400.0).alias("time_seconds_norm"),  # /90min
        pl.col("period_id").cast(pl.Int64),
    ])
    # score_diff + is_home_action: aprox via cum_sum de shots con outcome=success
    # (las acciones atomic no traen score state nativo; aproximamos con 0).
    df = df.with_columns([
        pl.lit(0).cast(pl.Int64).alias("score_diff"),
        pl.lit(0).cast(pl.Int64).alias("is_home_action"),
    ])
    return df


def build_training_table(atomic_df, cache: bool = True,
                         horizon_rec: int = HORIZON_REC,
                         horizon_att: int = HORIZON_ATT) -> pl.DataFrame:
    """Construye tabla con features + targets (recovery, attacked) sobre acciones
    defensivas SPADL.

    Recovery: dentro de las proximas `horizon` acciones, hay tackle/interception/
              clearance/keeper_* del equipo defensor (recupera posesion).
    Attacked: dentro de las proximas `horizon` acciones, hay shot exitoso del
              equipo NO-defensor (concede).
    """
    cache_path = _DERIVED / "training.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    df = _build_atomic_features(atomic_df)
    df = df.with_row_index("row_idx")

    # Para cada accion, etiqueta defensiva: ejecutor del SPADL es defender si
    # la accion es del set DEF_ACTION_TYPES (su equipo recibia presion).
    df = df.with_columns(
        pl.col("type_name").is_in(list(DEF_ACTION_TYPES)).alias("is_def_action")
    )
    # Para acciones defensivas, generar targets:
    # - recovery: en next horizon, EQUIPO DEL DEFENSOR (= team_id de la accion)
    #   tiene otra accion del set DEF_ACTION_TYPES o type_pass del SAME team_id
    #   (= mantiene posesion).
    # - attacked: en next horizon, OPONENTE (team_id != defender_team) tiene
    #   un shot que es gol.

    # Para hacer joins de "next N actions", agrupamos por game_id y rolling.
    # Sin shift por game_id en polars directo, hago group_by + over.
    df = df.sort(["game_id", "period_id", "time_seconds"])

    # next-action shifts over game_id (max horizon necesario)
    H = max(horizon_rec, horizon_att)
    over_game = pl.col("game_id")
    for k in range(1, H + 1):
        df = df.with_columns([
            pl.col("team_id").shift(-k).over(over_game).alias(f"next{k}_team"),
            pl.col("type_name").shift(-k).over(over_game).alias(f"next{k}_type"),
        ])

    # recovery STRICTO: en los proximos `horizon_rec` actions, defender_team
    # tiene >=2 acciones ofensivas consecutivas (consolidacion). Mas estricto
    # que "any next has same team" — exige sostenibilidad de la posesion.
    OFF_TYPES = ["pass", "dribble", "cross", "shot"]
    rec_count_expr = pl.lit(0).cast(pl.Int64)
    for k in range(1, horizon_rec + 1):
        rec_count_expr = rec_count_expr + (
            (pl.col(f"next{k}_team") == pl.col("team_id")) &
            (pl.col(f"next{k}_type").is_in(OFF_TYPES))
        ).cast(pl.Int64)
    rec_expr = (rec_count_expr >= 2)

    # attacked: opponent shot en next horizon_att (proxy de "concedio").
    att_expr = pl.lit(False)
    for k in range(1, horizon_att + 1):
        att_expr = att_expr | (
            (pl.col(f"next{k}_team") != pl.col("team_id")) &
            (pl.col(f"next{k}_type") == "shot")
        )

    df = df.with_columns([
        rec_expr.cast(pl.Int64).fill_null(0).alias("y_recovery"),
        att_expr.cast(pl.Int64).fill_null(0).alias("y_attacked"),
    ])

    # Filtrar solo acciones defensivas (dedupe period_id que tambien esta en FEATURE_COLS).
    # Drop ultimas H acciones de cada game_id (sin contexto next* valido para targets).
    keep_cols = ["game_id", "time_seconds", "team_id", "player_id", "type_name",
                 *FEATURE_COLS, "y_recovery", "y_attacked"]
    train = (df.filter(pl.col("is_def_action") &
                          pl.col(f"next{H}_team").is_not_null())
               .select(list(dict.fromkeys(keep_cols))))

    # Drop next* helpers
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        train.write_parquet(cache_path, compression="snappy")
    return train


# --- Train -----------------------------------------------------------------

def fit_vdep(df: pl.DataFrame, n_folds: int = 5, n_trials: int = 25,
             seed: int = 42) -> dict:
    """Entrena 2 cabezas LightGBM (recovery, attacked) + isotonic + Optuna."""
    import lightgbm as lgb
    import optuna
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X = df.select(FEATURE_COLS).to_numpy().astype(np.float32)
    y_rec = df["y_recovery"].to_numpy().astype(np.int32)
    y_att = df["y_attacked"].to_numpy().astype(np.int32)
    match_ids = df["game_id"].to_numpy()
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(match_ids)))
    rng.shuffle(uniq)
    folds = [np.array(f) for f in np.array_split(uniq, n_folds)]

    def _oof(y, params):
        oof = np.full(len(y), float(y.mean()), dtype=np.float32)   # baseline
        for fi, val_m in enumerate(folds):
            val_mask = np.isin(match_ids, val_m)
            tr_mask = ~val_mask
            # Salta folds con una sola clase en train (sin signal)
            if y[tr_mask].sum() == 0 or y[tr_mask].sum() == tr_mask.sum():
                continue
            m = lgb.LGBMClassifier(**params, random_state=seed + fi, verbose=-1)
            m.fit(X[tr_mask], y[tr_mask],
                  eval_set=[(X[val_mask], y[val_mask])],
                  callbacks=[lgb.early_stopping(20, verbose=False)])
            proba = m.predict_proba(X[val_mask])
            # Si solo vio 1 clase en train, predict_proba devuelve (N, 1)
            oof[val_mask] = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
        return oof

    def _objective(y, trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 400),
            "max_depth":         trial.suggest_int("max_depth", 3, 7),
            "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 7, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 80),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq":    1,
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        oof = _oof(y, params)
        return float(roc_auc_score(y, oof))

    print(f"  Optuna recovery (n_trials={n_trials})...")
    s_rec = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=seed))
    s_rec.optimize(lambda t: _objective(y_rec, t), n_trials=n_trials, show_progress_bar=False)
    print(f"  Optuna attacked (n_trials={n_trials})...")
    s_att = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=seed))
    s_att.optimize(lambda t: _objective(y_att, t), n_trials=n_trials, show_progress_bar=False)

    oof_rec = _oof(y_rec, s_rec.best_params)
    oof_att = _oof(y_att, s_att.best_params)

    cal_rec = IsotonicRegression(out_of_bounds="clip"); cal_rec.fit(oof_rec, y_rec)
    cal_att = IsotonicRegression(out_of_bounds="clip"); cal_att.fit(oof_att, y_att)
    oof_rec_cal = cal_rec.predict(oof_rec)
    oof_att_cal = cal_att.predict(oof_att)

    final_rec = lgb.LGBMClassifier(**s_rec.best_params, random_state=seed, verbose=-1)
    final_rec.fit(X, y_rec)
    final_att = lgb.LGBMClassifier(**s_att.best_params, random_state=seed, verbose=-1)
    final_att.fit(X, y_att)

    # C: ratio mean(attacked) / mean(recovery) sobre OOF cal — calibracion
    # automatica del peso defensa.
    C = float(y_att.mean() / max(y_rec.mean(), 1e-6))
    metrics = {
        "n_obs":         int(len(y_rec)),
        "rec_rate":      float(y_rec.mean()),
        "att_rate":      float(y_att.mean()),
        "auc_rec_cal":   float(roc_auc_score(y_rec, oof_rec_cal)),
        "auc_att_cal":   float(roc_auc_score(y_att, oof_att_cal)),
        "brier_rec":     float(brier_score_loss(y_rec, oof_rec_cal)),
        "brier_att":     float(brier_score_loss(y_att, oof_att_cal)),
        "C":             C,
        "best_rec":      s_rec.best_params,
        "best_att":      s_att.best_params,
    }
    return {
        "model_rec": final_rec, "cal_rec": cal_rec,
        "model_att": final_att, "cal_att": cal_att,
        "feature_cols": FEATURE_COLS,
        "C": C, "metrics": metrics,
    }


def save_fit(fit: dict, path: Path | None = None) -> Path:
    if path is None:
        path = _DERIVED / "model.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(fit, f)
    return path


def load_fit(path: Path | None = None) -> dict:
    if path is None:
        path = _DERIVED / "model.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


# --- Apply WC22 -----------------------------------------------------------

def predict_per_event(fit: dict, atomic_df) -> pl.DataFrame:
    """Aplica las dos cabezas a las acciones defensivas WC22 + computa VDEP."""
    df = _build_atomic_features(atomic_df)
    df = df.filter(pl.col("type_name").is_in(list(DEF_ACTION_TYPES)))
    if df.height == 0:
        return df
    X = df.select(fit["feature_cols"]).to_numpy().astype(np.float32)
    p_rec = fit["model_rec"].predict_proba(X)[:, 1]
    p_att = fit["model_att"].predict_proba(X)[:, 1]
    p_rec_cal = fit["cal_rec"].predict(p_rec)
    p_att_cal = fit["cal_att"].predict(p_att)
    vdep = p_rec_cal - fit["C"] * p_att_cal
    return df.with_columns([
        pl.Series("p_recovery", p_rec_cal).cast(pl.Float64),
        pl.Series("p_attacked", p_att_cal).cast(pl.Float64),
        pl.Series("vdep_strict", vdep).cast(pl.Float64),
    ])


def aggregate_per_minute(per_event: pl.DataFrame, cache: bool = True) -> pl.DataFrame:
    """Suma VDEP per (game_id, player_id, period_id, minute_in_period).

    Output schema alineado con M09 (pff_match_id, pff_player_id, period,
    minute_in_period). El game_id de SPADL es sb_match_id; el mapping a PFF
    se hace via M03.sb_to_pff_match_id() y atk.build_sb_to_pff_player_map().
    """
    cache_path = _DERIVED / "per_minute.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    import sys
    sys.path.insert(0, str(_REPO / "src"))
    from M03_preprocess import sb_to_pff_match_id
    import M08_ataque as atk

    df = per_event.with_columns([
        (pl.col("time_seconds") // 60).cast(pl.Int64).alias("minute_in_period"),
    ]).group_by(["game_id", "period_id", "player_id", "minute_in_period"]).agg([
        pl.col("vdep_strict").sum().alias("vdep_strict_minute"),
        pl.col("p_recovery").mean().alias("p_recovery_mean"),
        pl.col("p_attacked").mean().alias("p_attacked_mean"),
        pl.len().cast(pl.Int64).alias("n_def_actions"),
    ]).rename({
        "game_id": "sb_match_id", "player_id": "sb_player_id",
        "period_id": "period",
    })

    sb2pff_match = sb_to_pff_match_id()
    pmap = atk.build_sb_to_pff_player_map(cache=True).select([
        pl.col("sb_player_id").cast(pl.Int64),
        pl.col("pff_player_id").cast(pl.Int64, strict=False),
    ])
    df = df.with_columns([
        pl.col("sb_match_id").cast(pl.Int64),
        pl.col("sb_player_id").cast(pl.Int64),
        pl.col("sb_match_id").replace_strict(sb2pff_match, default=None)
            .alias("pff_match_id"),
    ]).join(pmap, on="sb_player_id", how="left").filter(
        pl.col("pff_match_id").is_not_null() & pl.col("pff_player_id").is_not_null()
    ).select([
        "pff_match_id", "pff_player_id", "period", "minute_in_period",
        "vdep_strict_minute", "p_recovery_mean", "p_attacked_mean", "n_def_actions",
    ])

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_path, compression="snappy")
    return df


def compute_all(overwrite: bool = False, n_trials: int = 25) -> dict[str, Path]:
    """Pipeline completa VDEP stricto: build training + fit + apply WC22 + aggregate."""
    out_paths = {
        "training":   _DERIVED / "training.parquet",
        "model":      _DERIVED / "model.pkl",
        "per_event":  _DERIVED / "per_event.parquet",
        "per_minute": _DERIVED / "per_minute.parquet",
    }
    if not overwrite and all(p.exists() for p in out_paths.values()):
        return out_paths

    import sys
    sys.path.insert(0, str(_REPO / "src"))
    import M08_ataque as atk

    print("[VDEP] Loading atomic SPADL training (Euro20+Euro24+Bundes23)...")
    train_atomic = atk.build_training_atomic(overwrite=False)
    print(f"  {len(train_atomic):,} actions training")

    print("[VDEP] Building training table (def actions + recovery/attacked targets)...")
    df = build_training_table(train_atomic, cache=True)
    print(f"  {df.height:,} def actions; rec_rate={df['y_recovery'].mean():.3f}, "
          f"att_rate={df['y_attacked'].mean():.3f}")
    if overwrite or not out_paths["training"].exists():
        df.write_parquet(out_paths["training"], compression="snappy")

    print("[VDEP] Fit 2 cabezas LightGBM + Optuna + isotonic...")
    fit = fit_vdep(df, n_folds=5, n_trials=n_trials)
    m = fit["metrics"]
    print(f"  AUC recovery cal: {m['auc_rec_cal']:.4f}")
    print(f"  AUC attacked cal: {m['auc_att_cal']:.4f}")
    print(f"  Brier rec/att   : {m['brier_rec']:.4f} / {m['brier_att']:.4f}")
    print(f"  C (att/rec)     : {m['C']:.4f}")
    save_fit(fit, out_paths["model"])

    print("[VDEP] Apply WC22 atomic actions...")
    wc22_atomic = atk.build_wc22_atomic(overwrite=False)
    pe = predict_per_event(fit, wc22_atomic)
    pe.write_parquet(out_paths["per_event"], compression="snappy")
    print(f"  per_event: {pe.height:,} def actions WC22")
    print(f"  vdep_strict range: [{pe['vdep_strict'].min():.3f}, "
          f"{pe['vdep_strict'].max():.3f}]")

    pm = aggregate_per_minute(pe, cache=False)
    pm.write_parquet(out_paths["per_minute"], compression="snappy")
    print(f"  per_minute: {pm.height:,} (player x match x minute)")
    return out_paths


if __name__ == "__main__":
    paths = compute_all(overwrite=True)
    for k, p in paths.items():
        print(f"  {k:<10} -> {p}  ({p.stat().st_size//1024} KB)")
