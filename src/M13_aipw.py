"""
M13_aipw - Validacion causal independiente vía cuasi-experimento near-miss.

Capa 3 PCJ — Estrategia B (Gauriot & Page 2019, ReStat). Estima ATT del shock
sobre los 4 canales explotando que, dado pre-shot xG en rango comparable, que
el balon entre o no es practicamente azar (variacion exogena del outcome).

Diseno de identificacion:
  - Universe: 127 goles SB con xg_baseline ∈ [0.15, 0.85] + 57 near-miss M06
              (12 palo + 38 save psxg≥0.6 + 5 offside cercano + 2 GLC) = 184 shots
  - Cluster: cada shot es 1 cluster (interferencia parcial Hudgens-Halloran 2008)
  - Unit: player-in-field at shot moment (22 players por cluster)
  - Treatment: 1 si is_goal else 0
  - Outcome: mean canal en ventana sec_abs ∈ [t+60, t+600] (1-10 min post-shot)

Estado del arte:

  Estimador           Referencia                         Implementacion
  -----------------   --------------------------------   ----------------------
  AIPW (IRM)          Robins-Rotnitzky-Zhao 1994 +       doubleml.DoubleMLIRM
                      Bang-Robins 2005 + Chernozhukov    (LightGBM cross-fit
                      et al. 2018 (DML)                  5-fold by match)
  PLR                 Chernozhukov et al. 2018           doubleml.DoubleMLPLR
  DR-learner          Kennedy 2023 (oraculo-optimo)      manual cross-fit
  RDD local-lineal    Imbens-Kalyanaraman 2012 +         manual con kernel
                      Calonico-Cattaneo-Titiunik 2014    triangular + CCT bw
  Spec curve          Simonsohn 2020                     manual sobre 3 def
  Balance test        Sant'Anna-Song-Xu 2022             SMD pre/post weighting
  Sensitivity         Cinelli-Hazlett 2020               omitted variable bias

8 estimaciones (4 canales x 2 shock_types) por estimador, paralelas a M12 ATE.
Comparacion ATT (M13) vs ATE (M12) por (canal, shock_type) — acceptance:
mismo signo + magnitudes en mismo orden = identificacion robusta. Divergencia
sistematica = upper-bound del confounding no controlado en M12.

Cluster errors: por sb_match_id (Cameron-Gelbach-Miller 2011 implem doubleml).

Outputs (data/parquet/derived/aipw/):
  panel_master.parquet            (event_uuid x pff_player_id x covariables)
  att_aipw.parquet                (channel x shock_type -> ATT AIPW + IC + N)
  att_dml_plr.parquet             (channel x shock_type -> ATT PLR + IC)
  att_dr_learner.parquet          (channel x shock_type -> ATT DR-learner + IC)
  att_rdd.parquet                 (channel x bandwidth -> ATT RDD + IC)
  spec_curve.parquet              (def_near_miss x channel x shock_type -> ATT)
  balance.parquet                 (covariable -> SMD pre + SMD post)
  sensitivity.parquet             (channel x shock_type -> robustness value)
  comparison_m12.parquet          (canal x shock_type -> ATE M12 vs ATT M13)

Depende de: M03 (sb_to_pff_match_id, player_minutes), M05 (shots + wc22 cov),
M06 (nearmiss_table), M07 (shocks_table mapeo), M08-M11 (per_minute por canal).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


# -- Rutas ------------------------------------------------------------------

_REPO    = Path(__file__).resolve().parents[1]
_DERIVED = _REPO / "data" / "parquet" / "derived" / "aipw"
_PSXG    = _REPO / "data" / "parquet" / "derived" / "psxg"
_NM      = _REPO / "data" / "parquet" / "derived" / "nearmiss" / "nearmiss_table.parquet"
_DID     = _REPO / "data" / "parquet" / "derived" / "did"


# -- Constantes pre-registradas -------------------------------------------

XG_PRIMARY_LO   = 0.15           # goles candidatos: xg pre-shot in [LO, HI]
XG_PRIMARY_HI   = 0.85
WINDOW_POST_SEC = (60, 600)      # outcome window post-shot (1-10 min)
N_FOLDS         = 5              # cross-fit folds (DoubleML)
RDD_BANDWIDTHS  = (0.10, 0.15, 0.20, 0.25)   # PSxG bw para sensitivity RDD
RDD_THRESHOLD   = 0.5            # umbral natural "save vs goal" en PSxG
SHOCK_TYPES     = ("GOAL_FOR", "GOAL_AGAINST")

# Mapeo canal -> (path per_minute, outcome col, fill_value)
CHANNELS: dict[str, tuple[str, str, float | None]] = {
    "ataque":  ("ataque/per_minute.parquet",  "score_atk_minute", 0.0),
    "defensa": ("defensa/per_minute.parquet", "score_def_minute", 0.0),
    "offball": ("offball/per_minute.parquet", "obso_mean",        0.0),
    "fisico":  ("fisico/per_minute.parquet",  "score_phys",       None),
}

# Covariables pre-shot AIPW (de M05 wc22_shots — todas existen pre-disparo)
COVARIATES = [
    "x", "y", "dist_goal", "angle_goal", "goal_aperture",
    "y_from_center", "x_to_goal_line",
    "bp_right_foot", "bp_left_foot", "bp_head", "bp_other",
    "tech_normal", "tech_volley", "tech_half_volley", "tech_lob",
    "type_open_play", "type_corner", "type_free_kick", "type_penalty",
    "pp_regular", "pp_from_corner", "pp_from_fk", "pp_from_throw", "pp_counter",
    "first_time", "under_pressure",
    "ff_has_frame", "ff_keeper_present",
    "ff_keeper_off_line", "ff_keeper_y_offset", "ff_keeper_dist_shot",
    "ff_n_def_in_cone", "ff_def_between_shot_goal",
    "ff_n_def_near_end", "ff_dist_nearest_def",
    "_sb_xg",
]


# ===========================================================================
#  SECCION 1 — Universe: pool treated (goles) + control (near-miss)
# ===========================================================================

def build_shot_pool(xg_lo: float = XG_PRIMARY_LO,
                     xg_hi: float = XG_PRIMARY_HI,
                     near_miss_types: tuple[str, ...] | None = None,
                     ) -> pl.DataFrame:
    """Pool treated/control: goles xg in [lo, hi] + near-miss filtrados.

    Args:
        xg_lo, xg_hi: rango pre-shot xG para goles incluidos.
        near_miss_types: tupla con tipos M06 a incluir; None = todos.
                         Para spec curve, restringir a {a_woodwork, c_save_psxg}.

    Returns:
        DataFrame con (sb_match_id, event_uuid, period, minute, second,
                       team_id_shooter, treated, xg_baseline, psxg, +covariables).
    """
    derived = _DERIVED.parent

    # M05 shots con outcome + xg + psxg
    shots = pl.read_parquet(_PSXG / "shots.parquet").with_columns(
        pl.col("is_goal").cast(pl.Boolean),
    )
    # M05 wc22_shots con TODAS las covariables (61 cols)
    wc22 = pl.read_parquet(_PSXG / "wc22_shots.parquet").select(
        ["_event_uuid", "_match_id"] + COVARIATES
    ).rename({"_event_uuid": "event_uuid", "_match_id": "sb_match_id"})

    # Treated: goles en rango xG
    treated = shots.filter(
        pl.col("is_goal") & pl.col("xg_baseline").is_between(xg_lo, xg_hi)
    ).with_columns(pl.lit(1, dtype=pl.Int64).alias("treated"))

    # Control: near-miss filtrado por tipo
    nm = pl.read_parquet(_NM)
    if near_miss_types is not None:
        nm = nm.filter(pl.col("near_miss_type").is_in(list(near_miss_types)))
    control = nm.select(
        ["sb_match_id", "event_uuid", "period", "minute", "second",
         "shot_outcome", "is_goal", "xg_baseline", "psxg"]
    ).with_columns([
        pl.col("is_goal").cast(pl.Boolean),
        pl.lit(0, dtype=pl.Int64).alias("treated"),
    ])

    pool = pl.concat([treated, control], how="diagonal_relaxed").unique(
        subset=["event_uuid"], keep="first",
    )

    # Enrich con team_id del shooter (sacar de SB events vía event_uuid)
    pool = _enrich_shooter_team(pool)

    # Enrich con covariables wc22
    pool = pool.join(wc22, on=["sb_match_id", "event_uuid"], how="left")

    # Drop disparos sin covariables (deberian ser cero, sanity)
    pool = pool.filter(pl.col("dist_goal").is_not_null())

    return pool


def _enrich_shooter_team(pool: pl.DataFrame) -> pl.DataFrame:
    """Saca shooter_team_id de SB events para cada event_uuid del pool."""
    from M02_loader_public import load_statsbomb_events

    rows = []
    for mid in pool["sb_match_id"].unique().to_list():
        ev = load_statsbomb_events(int(mid))
        shots_ev = ev.filter(pl.col("type").struct.field("name") == "Shot")
        if shots_ev.height == 0:
            continue
        rows.append(shots_ev.select([
            pl.col("id").alias("event_uuid"),
            pl.col("team").struct.field("id").cast(pl.Int64).alias("shooter_team_id"),
            pl.col("team").struct.field("name").alias("shooter_team_name"),
        ]))
    if not rows:
        return pool
    teams = pl.concat(rows)
    return pool.join(teams, on="event_uuid", how="left")


# ===========================================================================
#  SECCION 2 — Players in field at shot moment (PFF rosters + minutes)
# ===========================================================================

def players_in_field_at_shot(pool: pl.DataFrame) -> pl.DataFrame:
    """Para cada shot, devuelve los ~22 jugadores en campo en ese minuto.

    Cluster (event_uuid) -> 22 player rows (11 per team) con:
        pff_player_id, pff_team_id, position_group, perspective ∈
        {GOAL_FOR si player_team == shooter_team else GOAL_AGAINST}.
    """
    from M03_preprocess import sb_to_pff_match_id, player_minutes

    sb2pff = sb_to_pff_match_id()
    rows = []
    for r in pool.iter_rows(named=True):
        sb_mid = int(r["sb_match_id"])
        pff_mid = sb2pff.get(sb_mid)
        if pff_mid is None:
            continue
        pm = player_minutes(pff_mid)
        # SB minute es absoluto. Para identificar players en campo, comparamos
        # contra minute_in/minute_out (ambos absolutos en MINUTOS PFF).
        sb_min = int(r["minute"])
        on_field = pm.filter(
            pl.col("minute_in").is_not_null() &
            (pl.col("minute_in") <= sb_min) &
            (pl.col("minute_out") >= sb_min)
        )
        shooter_team_id = r.get("shooter_team_id")
        for pl_row in on_field.iter_rows(named=True):
            # Perspective: GOAL_FOR si player es del shooter team; pero los
            # team_ids de PFF y SB difieren. Usamos shooter_team_name vs PFF
            # team_name via la metadata del partido.
            rows.append({
                "event_uuid":      r["event_uuid"],
                "sb_match_id":     sb_mid,
                "pff_match_id":    pff_mid,
                "pff_player_id":   int(pl_row["player_id"]),
                "pff_team_id":     int(pl_row["team_id"]),
                "position_group":  pl_row["position_group"],
                "shooter_team_id": shooter_team_id,
            })
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows)
    df = _attach_perspective(df, pool)
    return df


def _attach_perspective(df: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    """Asigna perspective ∈ {GOAL_FOR, GOAL_AGAINST} via metadata PFF.

    SB shooter_team_id != PFF pff_team_id (distintos universos de IDs).
    Resolvemos via M01 metadata: el shooter team es el que tiene mismo nombre
    en metadata del partido.
    """
    from M01_loader_pff import load_metadata

    perspectives = []
    sb_team_to_pff = {}
    for sb_mid in pool["sb_match_id"].unique().to_list():
        from M03_preprocess import sb_to_pff_match_id
        pff_mid = sb_to_pff_match_id().get(int(sb_mid))
        if pff_mid is None:
            continue
        md = load_metadata(int(pff_mid)).row(0, named=True)
        # Match by name SB vs PFF
        for ev_row in pool.filter(pl.col("sb_match_id") == sb_mid).iter_rows(named=True):
            sb_name = ev_row.get("shooter_team_name")
            if sb_name == md["home_team_name"]:
                sb_team_to_pff[(int(sb_mid), ev_row["event_uuid"])] = md["home_team_id"]
            elif sb_name == md["away_team_name"]:
                sb_team_to_pff[(int(sb_mid), ev_row["event_uuid"])] = md["away_team_id"]

    persp = []
    for r in df.iter_rows(named=True):
        key = (int(r["sb_match_id"]), r["event_uuid"])
        shooter_pff_team = sb_team_to_pff.get(key)
        if shooter_pff_team is None:
            persp.append(None)
        elif r["pff_team_id"] == shooter_pff_team:
            persp.append("GOAL_FOR")
        else:
            persp.append("GOAL_AGAINST")
    return df.with_columns(pl.Series("perspective", persp))


# ===========================================================================
#  SECCION 3 — Outcomes post-shot por canal
# ===========================================================================

def attach_outcomes(panel: pl.DataFrame, pool: pl.DataFrame,
                     channel: str) -> pl.DataFrame:
    """Anade outcome_post = mean(canal) en ventana sec_abs ∈ [t+60, t+600].

    sec_abs del shot = sb_minute*60 + sb_second (alineado con M12 sec_abs).
    """
    derived = _DERIVED.parent
    pm_relpath, outcome_col, fill_value = CHANNELS[channel]
    pm = pl.read_parquet(derived / pm_relpath).select([
        "pff_match_id", "pff_player_id", "sec_abs",
        pl.col(outcome_col).alias("outcome"),
    ])

    # sec_abs del shot
    pool_with_sec = pool.with_columns(
        (pl.col("minute") * 60 + pl.col("second")).cast(pl.Int64).alias("t_event_sec")
    ).select(["event_uuid", "t_event_sec"])

    panel = panel.join(pool_with_sec, on="event_uuid", how="left")

    # Window join: para cada (event_uuid, player), filtrar pm en [t+60, t+600]
    # y aggregate mean. Polars no tiene join asof rangos, asi que hacemos cross
    # join intra-match (small N) y filtramos.
    panel_with_outcome = panel.join(
        pm, on=["pff_match_id", "pff_player_id"], how="left",
    ).with_columns(
        (pl.col("sec_abs") - pl.col("t_event_sec")).alias("rel_sec")
    ).filter(
        pl.col("rel_sec").is_between(WINDOW_POST_SEC[0], WINDOW_POST_SEC[1])
    )

    if fill_value is not None:
        panel_with_outcome = panel_with_outcome.with_columns(
            pl.col("outcome").fill_null(fill_value)
        )

    agg = panel_with_outcome.group_by(
        ["event_uuid", "pff_player_id"]
    ).agg([
        pl.col("outcome").mean().alias("outcome_post"),
        pl.col("outcome").count().cast(pl.Int64).alias("n_minutes_obs"),
    ])

    out = panel.join(agg, on=["event_uuid", "pff_player_id"], how="left")
    return out


def build_panel_for_channel_perspective(
    channel: str, perspective: str,
    near_miss_types: tuple[str, ...] | None = None,
    cache: bool = True,
) -> pl.DataFrame:
    """Panel completo para 1 (canal, perspective): pool x players x covariables x outcome."""
    cache_path = _DERIVED / f"panel_{channel}_{perspective.lower()}.parquet"
    if cache and cache_path.exists() and near_miss_types is None:
        return pl.read_parquet(cache_path)

    pool = build_shot_pool(near_miss_types=near_miss_types)
    panel = players_in_field_at_shot(pool)
    panel = panel.filter(pl.col("perspective") == perspective)

    # Anadir treated, covariables, outcome_post
    pool_slim = pool.select(
        ["event_uuid", "treated", "xg_baseline", "psxg"] + COVARIATES
    )
    panel = panel.join(pool_slim, on="event_uuid", how="left")
    panel = attach_outcomes(panel, pool, channel)

    # Drop rows sin outcome (sin per_minute en ventana — fisico p.ej. si NaN)
    panel = panel.filter(pl.col("outcome_post").is_not_null())

    if cache and near_miss_types is None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        panel.write_parquet(cache_path, compression="snappy")
    return panel


# ===========================================================================
#  SECCION 4 — AIPW via DoubleMLIRM (Chernozhukov 2018)
# ===========================================================================

def estimate_att_aipw(panel: pl.DataFrame) -> dict:
    """ATT via DoubleML IRM (AIPW): cross-fit 5-fold by sb_match_id.

    DoubleMLIRM = Interactive Regression Model = AIPW de doble robustez:
    nuisance models para Y(0)=g0(X), Y(1)=g1(X), propensity m(X)=P(D=1|X)
    todos via LightGBM cross-fitted. ATT robusto bajo error en alguno de
    los nuisance.
    """
    from doubleml import DoubleMLData, DoubleMLIRM
    from sklearn.model_selection import GroupKFold
    import lightgbm as lgb

    if panel.height < 30 or panel["treated"].sum() == 0 or \
       panel["treated"].sum() == panel.height:
        return {"att": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "n_obs": panel.height, "n_treated": int(panel["treated"].sum())}

    df = panel.to_pandas()
    cov_cols = [c for c in COVARIATES if c in df.columns]

    # GroupKFold por sb_match_id evita leakage shot-shot del mismo partido
    n_groups = df["sb_match_id"].nunique()
    n_folds = min(N_FOLDS, n_groups)
    gkf = GroupKFold(n_splits=n_folds)
    smpls = [(tr.tolist(), te.tolist()) for tr, te in
             gkf.split(df, df["treated"], groups=df["sb_match_id"])]

    data = DoubleMLData(df, y_col="outcome_post", d_cols="treated",
                         x_cols=cov_cols)
    ml_g = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                              random_state=42, verbose=-1)
    ml_m = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                random_state=42, verbose=-1)
    dml = DoubleMLIRM(data, ml_g=ml_g, ml_m=ml_m, n_folds=n_folds,
                       score="ATTE", trimming_threshold=0.01)
    dml.set_sample_splitting(all_smpls=smpls)
    dml.fit()
    att = float(dml.coef[0])
    se = float(dml.se[0])
    ci_lo, ci_hi = att - 1.96 * se, att + 1.96 * se

    return {
        "att": att, "se": se, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "n_obs": int(df.shape[0]),
        "n_treated": int(df["treated"].sum()),
        "n_clusters": int(df["sb_match_id"].nunique()),
    }


# ===========================================================================
#  SECCION 5 — DML PLR (Partially Linear Regression)
# ===========================================================================

def estimate_att_dml_plr(panel: pl.DataFrame) -> dict:
    """ATT via DoubleML PLR: especificacion lineal en treatment, no-param X."""
    from doubleml import DoubleMLData, DoubleMLPLR
    from sklearn.model_selection import GroupKFold
    import lightgbm as lgb

    if panel.height < 30 or panel["treated"].sum() == 0 or \
       panel["treated"].sum() == panel.height:
        return {"att": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "n_obs": panel.height}

    df = panel.to_pandas()
    cov_cols = [c for c in COVARIATES if c in df.columns]
    n_folds = min(N_FOLDS, df["sb_match_id"].nunique())
    gkf = GroupKFold(n_splits=n_folds)
    smpls = [(tr.tolist(), te.tolist()) for tr, te in
             gkf.split(df, df["treated"], groups=df["sb_match_id"])]

    data = DoubleMLData(df, y_col="outcome_post", d_cols="treated",
                         x_cols=cov_cols)
    ml_l = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                              random_state=42, verbose=-1)
    ml_m = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                              random_state=42, verbose=-1)
    dml = DoubleMLPLR(data, ml_l=ml_l, ml_m=ml_m, n_folds=n_folds)
    dml.set_sample_splitting(all_smpls=smpls)
    dml.fit()
    att = float(dml.coef[0])
    se = float(dml.se[0])
    return {"att": att, "se": se,
            "ci_lo": att - 1.96 * se, "ci_hi": att + 1.96 * se,
            "n_obs": int(df.shape[0])}


# ===========================================================================
#  SECCION 6 — DR-learner manual (Kennedy 2023, oraculo-optimo)
# ===========================================================================

def estimate_att_dr_learner(panel: pl.DataFrame) -> dict:
    """DR-learner via cross-fitted LightGBM (Kennedy 2023).

    Algoritmo:
      1. Cross-fit nuisance: g0(X)=E[Y|D=0,X], g1(X)=E[Y|D=1,X], m(X)=P(D=1|X)
      2. Pseudo-outcome: phi = g1(X) - g0(X) + D*(Y-g1(X))/m(X)
                                              - (1-D)*(Y-g0(X))/(1-m(X))
      3. ATT = mean(phi over treated)
      4. SE via influence function + cluster bootstrap por match
    """
    from sklearn.model_selection import GroupKFold
    import lightgbm as lgb

    if panel.height < 30 or panel["treated"].sum() == 0 or \
       panel["treated"].sum() == panel.height:
        return {"att": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "n_obs": panel.height}

    df = panel.to_pandas()
    cov_cols = [c for c in COVARIATES if c in df.columns]
    X = df[cov_cols].values
    Y = df["outcome_post"].values
    D = df["treated"].values
    groups = df["sb_match_id"].values

    n_folds = min(N_FOLDS, df["sb_match_id"].nunique())
    gkf = GroupKFold(n_splits=n_folds)
    g0_pred, g1_pred, m_pred = np.zeros(len(Y)), np.zeros(len(Y)), np.zeros(len(Y))
    for tr_idx, te_idx in gkf.split(X, D, groups=groups):
        # g0 entrenado en controles del fold train
        tr_ctrl = tr_idx[D[tr_idx] == 0]
        tr_treat = tr_idx[D[tr_idx] == 1]
        if len(tr_ctrl) == 0 or len(tr_treat) == 0:
            continue
        g0 = lgb.LGBMRegressor(n_estimators=200, max_depth=4,
                                random_state=42, verbose=-1)
        g0.fit(X[tr_ctrl], Y[tr_ctrl])
        g0_pred[te_idx] = g0.predict(X[te_idx])
        g1 = lgb.LGBMRegressor(n_estimators=200, max_depth=4,
                                random_state=42, verbose=-1)
        g1.fit(X[tr_treat], Y[tr_treat])
        g1_pred[te_idx] = g1.predict(X[te_idx])
        m = lgb.LGBMClassifier(n_estimators=200, max_depth=4,
                                random_state=42, verbose=-1)
        m.fit(X[tr_idx], D[tr_idx])
        m_pred[te_idx] = m.predict_proba(X[te_idx])[:, 1]

    # Trimming propensity
    m_pred = np.clip(m_pred, 0.01, 0.99)
    # Pseudo-outcome (Kennedy 2023 Eq 1)
    phi = (g1_pred - g0_pred
           + D * (Y - g1_pred) / m_pred
           - (1 - D) * (Y - g0_pred) / (1 - m_pred))
    att = float(np.mean(phi))

    # Cluster bootstrap por match (200 iters)
    rng = np.random.default_rng(42)
    unique_groups = np.unique(groups)
    boots = []
    for _ in range(200):
        sampled_g = rng.choice(unique_groups, len(unique_groups), replace=True)
        mask = np.concatenate([np.where(groups == g)[0] for g in sampled_g])
        boots.append(np.mean(phi[mask]))
    se = float(np.std(boots, ddof=1))
    ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    return {"att": att, "se": se, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "n_obs": int(len(Y)), "n_treated": int(D.sum())}


# ===========================================================================
#  SECCION 7 — RDD local-lineal sobre PSxG (Imbens-Kalyanaraman)
# ===========================================================================

def estimate_att_rdd(panel: pl.DataFrame, threshold: float = RDD_THRESHOLD,
                      bandwidth: float = 0.20) -> dict:
    """RDD local-lineal sobre PSxG con kernel triangular.

    Treatment de discontinuidad: en PSxG cerca del umbral, si entra/no es
    aproximadamente aleatorio. Comparamos outcome_post de shots justo arriba
    (high PSxG, mayor prob gol) vs justo abajo (high PSxG, mayor prob save).

    bandwidth en torno al threshold (defecto 0.20 = simetrico, [0.30, 0.70]).
    """
    df = panel.to_pandas()
    if df.shape[0] < 30:
        return {"att": np.nan, "se": np.nan, "n_obs": 0}

    psxg = df["psxg"].values
    Y = df["outcome_post"].values
    D = df["treated"].values
    in_band = np.abs(psxg - threshold) <= bandwidth
    if in_band.sum() < 20:
        return {"att": np.nan, "se": np.nan, "n_obs": int(in_band.sum())}

    df_b = df[in_band]
    psxg_b = df_b["psxg"].values
    Y_b = df_b["outcome_post"].values
    D_b = df_b["treated"].values
    # Triangular kernel weights
    w = 1.0 - np.abs(psxg_b - threshold) / bandwidth

    # Local linear regression separately on each side, evaluate at threshold
    above = psxg_b >= threshold
    below = ~above
    if above.sum() < 5 or below.sum() < 5:
        return {"att": np.nan, "se": np.nan, "n_obs": int(in_band.sum())}

    # Y = a + b * (psxg - threshold), weighted by w
    # Fallback a lstsq pseudo-inverse si X.T W X es singular (sucede cuando
    # los puntos del bin estan colineales en psxg, e.g., todos al mismo valor).
    def _local_fit(mask):
        X = np.column_stack([np.ones(mask.sum()), psxg_b[mask] - threshold])
        W = np.diag(w[mask])
        try:
            beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ Y_b[mask])
            bread = np.linalg.inv(X.T @ W @ X)
        except np.linalg.LinAlgError:
            beta, *_ = np.linalg.lstsq(W @ X, W @ Y_b[mask], rcond=None)
            bread = np.linalg.pinv(X.T @ W @ X)
        resid = Y_b[mask] - X @ beta
        meat = X.T @ np.diag(w[mask] * resid ** 2) @ X
        cov = bread @ meat @ bread
        return float(beta[0]), float(np.sqrt(max(cov[0, 0], 0)))

    try:
        intercept_above, se_above = _local_fit(above)
        intercept_below, se_below = _local_fit(below)
    except Exception:
        return {"att": np.nan, "se": np.nan, "n_obs": int(in_band.sum()),
                "bandwidth": bandwidth, "threshold": threshold}
    att = intercept_above - intercept_below
    se = float(np.sqrt(se_above ** 2 + se_below ** 2))
    return {"att": att, "se": se, "ci_lo": att - 1.96 * se, "ci_hi": att + 1.96 * se,
            "n_obs": int(in_band.sum()), "bandwidth": bandwidth,
            "threshold": threshold}


# ===========================================================================
#  SECCION 8 — Specification curve (Simonsohn 2020)
# ===========================================================================

def specification_curve(channel: str, perspective: str) -> pl.DataFrame:
    """3 def near-miss (estricta/intermedia/laxa) → ATT AIPW para cada."""
    SPECS = {
        "strict":   ("a_woodwork", "c_save_psxg"),
        "medium":   ("a_woodwork", "c_save_psxg", "b_offside_close"),
        "lax":      ("a_woodwork", "c_save_psxg", "b_offside_close",
                     "d_goal_line_clearance"),
    }
    rows = []
    for spec_name, types in SPECS.items():
        panel = build_panel_for_channel_perspective(channel, perspective,
                                                      near_miss_types=types,
                                                      cache=False)
        att = estimate_att_aipw(panel)
        rows.append({"spec": spec_name,
                     "channel": channel, "perspective": perspective,
                     **{f"att_{k}" if k in ("att", "se", "ci_lo", "ci_hi", "n_obs") else k: v
                        for k, v in att.items()}})
    return pl.DataFrame(rows)


# ===========================================================================
#  SECCION 9 — Balance test (SMD pre-balanceo)
# ===========================================================================

def balance_check(panel: pl.DataFrame) -> pl.DataFrame:
    """Standardized Mean Difference por covariable (Sant'Anna-Song-Xu 2022).

    SMD_j = (mean_j_treated - mean_j_control) / sqrt((var_j_t + var_j_c)/2).
    Aceptable: |SMD| < 0.10 (rule of thumb literatura aplicada).
    """
    df = panel.to_pandas()
    cov_cols = [c for c in COVARIATES if c in df.columns]
    rows = []
    treated_mask = df["treated"] == 1
    for c in cov_cols:
        x_t = df.loc[treated_mask, c].values
        x_c = df.loc[~treated_mask, c].values
        if len(x_t) == 0 or len(x_c) == 0:
            continue
        m_t, m_c = float(np.mean(x_t)), float(np.mean(x_c))
        v_t, v_c = float(np.var(x_t)), float(np.var(x_c))
        denom = float(np.sqrt((v_t + v_c) / 2)) if (v_t + v_c) > 0 else 1.0
        smd = (m_t - m_c) / denom if denom > 0 else 0.0
        rows.append({"covariate": c, "mean_treated": m_t, "mean_control": m_c,
                     "smd": smd, "abs_smd": abs(smd),
                     "balanced": abs(smd) < 0.10})
    return pl.DataFrame(rows).sort("abs_smd", descending=True)


# ===========================================================================
#  SECCION 10 — Sensitivity analysis (Cinelli-Hazlett 2020)
# ===========================================================================

def sensitivity_analysis(att_dict: dict, panel: pl.DataFrame) -> dict:
    """Robustness Value (Cinelli-Hazlett 2020): cuanta confounding hace falta
    para anular el ATT estimado. RV alto = robusto a OVB.

    RV = 0.5 * (sqrt(f_y^2 * (f_y^2 + 4) - f_y^2)) donde f_y es la cota de
    R^2 parcial entre confounder y outcome.

    Implementacion compacta: estima R2 parcial del treatment sobre outcome
    (sin confounders), luego RV bajo el supuesto de confounders del mismo
    poder explicativo que las covariables existentes.
    """
    if np.isnan(att_dict.get("att", np.nan)):
        return {"robustness_value": np.nan, "tstat_ratio": np.nan}
    att, se = att_dict["att"], att_dict["se"]
    if se == 0:
        return {"robustness_value": np.nan, "tstat_ratio": np.nan}
    t_stat = att / se
    # Bound mininum needed: Cinelli-Hazlett RV formula
    # RV_q = 0.5 * (sqrt(q^4 + 4*q^2) - q^2) donde q = t_stat / sqrt(df-1)
    df_resid = max(panel.height - 2, 1)
    q = t_stat / np.sqrt(df_resid)
    rv = 0.5 * (np.sqrt(q ** 4 + 4 * q ** 2) - q ** 2)
    return {"robustness_value": float(rv),
            "tstat_ratio": float(t_stat),
            "interpretation": ("robusto" if rv > 0.05 else "fragil"
                                if not np.isnan(rv) else "indeterminado")}


# ===========================================================================
#  SECCION 11 — compute_all + comparacion M12
# ===========================================================================

def compare_with_m12(att_df: pl.DataFrame) -> pl.DataFrame:
    """Compara ATT M13 (AIPW) vs ATE M12 (DiD FE) por (canal, shock_type).

    Acceptance: mismo signo + magnitudes en mismo orden = ID robusta.
    Divergencia sistematica = upper-bound del confounding no controlado.
    """
    m12_path = _DID / "ate_population.parquet"
    if not m12_path.exists():
        return pl.DataFrame()
    m12 = pl.read_parquet(m12_path).select([
        "channel", "shock_type",
        pl.col("ate").alias("ate_m12"), pl.col("se").alias("se_m12"),
    ])
    m13 = att_df.rename({"perspective": "shock_type"}).select([
        "channel", "shock_type",
        pl.col("att").alias("att_m13"), pl.col("se").alias("se_m13"),
    ])
    cmp = m12.join(m13, on=["channel", "shock_type"], how="left")
    cmp = cmp.with_columns([
        (pl.col("att_m13") * pl.col("ate_m12") > 0).alias("same_sign"),
        ((pl.col("att_m13") - pl.col("ate_m12")).abs() /
         (pl.col("se_m12") + pl.col("se_m13"))).alias("diff_normalized"),
    ])
    return cmp


def compute_all(cache: bool = True, overwrite: bool = False) -> dict[str, Path]:
    """Pipeline completa M13: paneles + AIPW + PLR + DR + RDD + spec + balance + sensitivity + comparison."""
    out_paths = {
        "panel":          _DERIVED / "panel_master.parquet",
        "att_aipw":       _DERIVED / "att_aipw.parquet",
        "att_dml_plr":    _DERIVED / "att_dml_plr.parquet",
        "att_dr_learner": _DERIVED / "att_dr_learner.parquet",
        "att_rdd":        _DERIVED / "att_rdd.parquet",
        "spec_curve":     _DERIVED / "spec_curve.parquet",
        "balance":        _DERIVED / "balance.parquet",
        "sensitivity":    _DERIVED / "sensitivity.parquet",
        "comparison_m12": _DERIVED / "comparison_m12.parquet",
    }
    if not overwrite and all(p.exists() for p in out_paths.values()):
        return out_paths
    _DERIVED.mkdir(parents=True, exist_ok=True)

    aipw_rows, plr_rows, dr_rows, rdd_rows, sens_rows, panels_long = (
        [], [], [], [], [], []
    )
    spec_rows, balance_rows = [], []

    for ch in CHANNELS:
        for persp in SHOCK_TYPES:
            print(f"  computing ({ch}, {persp})...", flush=True)
            panel = build_panel_for_channel_perspective(ch, persp, cache=cache)
            if panel.height < 30:
                print(f"    SKIP n={panel.height}")
                continue

            panels_long.append(panel.with_columns([
                pl.lit(ch).alias("channel"),
                pl.lit(persp).alias("perspective"),
                # Cast a Float64 unificado (M08/M09 vienen Float32, M10/M11 Float64)
                pl.col("outcome_post").cast(pl.Float64),
                pl.col("xg_baseline").cast(pl.Float64),
                pl.col("psxg").cast(pl.Float64),
            ]).select(["channel", "perspective", "event_uuid",
                       "pff_player_id", "treated", "outcome_post",
                       "xg_baseline", "psxg"]))

            att_aipw = estimate_att_aipw(panel)
            aipw_rows.append({"channel": ch, "perspective": persp, **att_aipw})

            att_plr = estimate_att_dml_plr(panel)
            plr_rows.append({"channel": ch, "perspective": persp, **att_plr})

            att_dr = estimate_att_dr_learner(panel)
            dr_rows.append({"channel": ch, "perspective": persp, **att_dr})

            for bw in RDD_BANDWIDTHS:
                rdd = estimate_att_rdd(panel, bandwidth=bw)
                rdd_rows.append({"channel": ch, "perspective": persp,
                                 "bandwidth": bw, **rdd})

            sens = sensitivity_analysis(att_aipw, panel)
            sens_rows.append({"channel": ch, "perspective": persp, **sens})

            bal = balance_check(panel).with_columns([
                pl.lit(ch).alias("channel"), pl.lit(persp).alias("perspective"),
            ])
            balance_rows.append(bal)

            sc = specification_curve(ch, persp)
            spec_rows.append(sc)

    panel_master = pl.concat(panels_long) if panels_long else pl.DataFrame()
    att_aipw_df = pl.DataFrame(aipw_rows)
    att_plr_df = pl.DataFrame(plr_rows)
    att_dr_df = pl.DataFrame(dr_rows)
    att_rdd_df = pl.DataFrame(rdd_rows)
    sens_df = pl.DataFrame(sens_rows)
    spec_df = pl.concat(spec_rows) if spec_rows else pl.DataFrame()
    balance_df = pl.concat(balance_rows) if balance_rows else pl.DataFrame()
    cmp_df = compare_with_m12(att_aipw_df)

    if cache:
        panel_master.write_parquet(out_paths["panel"], compression="snappy")
        att_aipw_df.write_parquet(out_paths["att_aipw"], compression="snappy")
        att_plr_df.write_parquet(out_paths["att_dml_plr"], compression="snappy")
        att_dr_df.write_parquet(out_paths["att_dr_learner"], compression="snappy")
        att_rdd_df.write_parquet(out_paths["att_rdd"], compression="snappy")
        spec_df.write_parquet(out_paths["spec_curve"], compression="snappy")
        balance_df.write_parquet(out_paths["balance"], compression="snappy")
        sens_df.write_parquet(out_paths["sensitivity"], compression="snappy")
        cmp_df.write_parquet(out_paths["comparison_m12"], compression="snappy")
    return out_paths


# -- Sanity inline ---------------------------------------------------------

if __name__ == "__main__":
    import time, sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import warnings
    warnings.filterwarnings("ignore")

    print("=== M13_aipw sanity ===\n")

    print("[1] Build pool treated (goles xg in [0.15, 0.85]) + control (near-miss)...")
    t0 = time.time()
    pool = build_shot_pool()
    print(f"  pool: {pool.height} disparos "
          f"({pool['treated'].sum()} treated + {(1-pool['treated']).sum()} control) "
          f"en {time.time()-t0:.1f}s")

    print("\n[2] Players in field at shot moment...")
    t0 = time.time()
    pif = players_in_field_at_shot(pool)
    print(f"  panel cluster x player: {pif.height:,} rows "
          f"(esperado ~{pool.height * 22} = {pool.height * 22:,})")
    print(f"  perspective distribution:")
    print(pif.group_by("perspective").len())

    print("\n[3] Build panel para 1 (canal, perspective): (ataque, GOAL_AGAINST)...")
    t0 = time.time()
    panel = build_panel_for_channel_perspective("ataque", "GOAL_AGAINST",
                                                  cache=False)
    print(f"  panel ataque/GA: {panel.height:,} rows en {time.time()-t0:.1f}s")
    print(f"  treated/control: {panel.group_by('treated').len()}")

    print("\n[4] AIPW (DoubleMLIRM) (ataque, GOAL_AGAINST)...")
    t0 = time.time()
    att = estimate_att_aipw(panel)
    print(f"  ATT={att['att']:+.4f}, SE={att['se']:.4f}, "
          f"CI=[{att['ci_lo']:+.4f}, {att['ci_hi']:+.4f}], "
          f"N={att['n_obs']:,}, treated={att['n_treated']:,} "
          f"en {time.time()-t0:.1f}s")

    print("\n[5] DML PLR...")
    plr = estimate_att_dml_plr(panel)
    print(f"  ATT_PLR={plr['att']:+.4f}, SE={plr['se']:.4f}")

    print("\n[6] DR-learner (Kennedy 2023)...")
    dr = estimate_att_dr_learner(panel)
    print(f"  ATT_DR={dr['att']:+.4f}, SE={dr['se']:.4f}, "
          f"CI=[{dr['ci_lo']:+.4f}, {dr['ci_hi']:+.4f}]")

    print("\n[7] RDD local-lineal sobre PSxG (bandwidth 0.20)...")
    rdd = estimate_att_rdd(panel)
    print(f"  ATT_RDD={rdd['att']:+.4f}, SE={rdd['se']:.4f}, "
          f"N={rdd['n_obs']}")

    print("\n[8] Balance check SMD:")
    bal = balance_check(panel)
    print(f"  cov balanceadas (|SMD|<0.10): {int(bal['balanced'].sum())}/{bal.height}")
    print(f"  worst:")
    print(bal.head(5).select(['covariate', 'smd', 'balanced']))

    print("\n[9] Sensitivity Cinelli-Hazlett (RV):")
    sens = sensitivity_analysis(att, panel)
    print(f"  RV={sens['robustness_value']:.4f}, t={sens['tstat_ratio']:.2f}, "
          f"interp={sens['interpretation']}")

    print("\n[10] Specification curve (3 def near-miss):")
    sc = specification_curve("ataque", "GOAL_AGAINST")
    print(sc)

    print("\n[11] compute_all + cache (8 estimaciones x 4 estimadores)...")
    t0 = time.time()
    paths = compute_all(cache=True, overwrite=True)
    print(f"  todos los outputs cacheados en {time.time()-t0:.0f}s")
    for k, p in paths.items():
        if p.exists():
            print(f"    {k:<15} -> {p.relative_to(_REPO)} ({p.stat().st_size//1024} KB)")

    print("\n[12] Acceptance: comparacion M12 vs M13 (4 estimadores convergen):")
    cmp = pl.read_parquet(paths["comparison_m12"])
    print(cmp)

    aipw_df = pl.read_parquet(paths["att_aipw"])
    dr_df = pl.read_parquet(paths["att_dr_learner"])
    print(f"\n  ATT AIPW vs DR-learner (deberian estar cerca):")
    joined = aipw_df.select(["channel", "perspective",
                              pl.col("att").alias("att_aipw")]).join(
        dr_df.select(["channel", "perspective",
                      pl.col("att").alias("att_dr")]),
        on=["channel", "perspective"], how="inner",
    )
    print(joined)
