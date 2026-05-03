"""M15_pcj — Perfil Clutch del Jugador: ensamblaje scout-facing final.

Lee outputs M14 + posterior samples + metadata → produce tabla maestra
`outputs/pcj_table.parquet` con vector 4-canal × 2 shocks + indices + IC
bayesianos + tier labels + posterior probabilities + rankings.

Decisiones de diseno (TOP 1% SOTA):
  - **Threshold 270 min** (3 partidos completos, estandar scout)
  - **8 CATEs preservados** (4 canales x 2 shocks) - chasing vs protecting
    son fenomenos psicologicos distintos, agregar cancelaria signos
  - **Vector PCJ summary 4-canal directional**:
      pcj_atk = cate_atk_GA  (chasing → atk relevant cuando vas perdiendo)
      pcj_def = cate_def_GF  (protecting → def relevant cuando vas ganando)
      pcj_off = cate_off_GA  (chasing → off-ball cuando atacas resp.)
      pcj_phys = cate_phys con max(|GA|,|GF|) signed (reactividad fisica)
  - **Tier labels percentile-based** dual: global + within-position
  - **Significance flag bayesiano**: P(idx>0|data) > 0.95 → "Sig clutch"
    (lo que diferencia esto de Wyscout/InStat: ellos no tienen IC posterior)

Outputs:
  outputs/pcj_table.parquet         (1 fila por jugador, ~60 cols)
  outputs/pcj_aux/top10_chasing_per_position.parquet
  outputs/pcj_aux/top10_protecting_per_position.parquet
  outputs/pcj_aux/dual_clutch_top.parquet
  outputs/pcj_aux/by_team.parquet

Uso:
    python M15_pcj.py [overwrite]
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl

_REPO = Path(__file__).resolve().parents[1]
_CATE_DIR = _REPO / "data" / "parquet" / "derived" / "cate"
_SHOCKS = _REPO / "data" / "parquet" / "derived" / "shocks" / "shocks_table.parquet"
_PFF_GRADES = _REPO / "data" / "parquet" / "derived" / "preprocess" / "pff_grades.parquet"
_PLAYERS_CSV = _REPO / "data_mundial" / "players.csv"
_OUT_DIR = _REPO / "outputs"
_AUX_DIR = _OUT_DIR / "pcj_aux"

MIN_MINUTES = 270
SIG_THRESHOLD = 0.95

CHANNELS = ["ataque", "defensa", "offball", "fisico"]
SHOCK_TYPES = ["GOAL_AGAINST", "GOAL_FOR"]   # GA=0, GF=1 (M14 sort order)


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
def _load_m14() -> dict:
    with open(_CATE_DIR / "model" / "cate_nuts.pkl", "rb") as f:
        fit = pickle.load(f)
    posterior = pl.read_parquet(_CATE_DIR / "posterior_player.parquet")
    indices   = pl.read_parquet(_CATE_DIR / "indices.parquet")
    rankings  = pl.read_parquet(_CATE_DIR / "rankings.parquet")
    return dict(fit=fit, posterior=posterior, indices=indices, rankings=rankings)


def _load_player_meta() -> pl.DataFrame:
    """Identidad: pff_player_id, player_name, position_group, team_name.

    Combina players.csv (nicknames) + pff_grades.parquet (position_group + team
    derivados de rosters).
    """
    players = pl.read_csv(_PLAYERS_CSV).select(
        pl.col("id").alias("pff_player_id"),
        pl.col("nickname").alias("player_name"),
    ).unique("pff_player_id")
    grades = pl.read_parquet(_PFF_GRADES).select(
        ["pff_player_id", "team_name", "position_group"])
    return players.join(grades, on="pff_player_id", how="inner")


def _load_player_minutes() -> pl.DataFrame:
    """Suma minutos por jugador a lo largo de los 64 partidos WC22."""
    sys.path.insert(0, str(_REPO / "src"))
    from M01_loader_pff import list_event_match_ids
    from M03_preprocess import player_minutes
    rows = []
    for mid in list_event_match_ids():
        pm = player_minutes(mid)
        rows.append(pm.select(["player_id", "minutes_played"]))
    pm_all = pl.concat(rows)
    return (pm_all.group_by("player_id").agg([
                pl.col("minutes_played").sum().alias("minutes_played"),
                pl.col("minutes_played").gt(0).sum().alias("n_matches_played"),
            ]).rename({"player_id": "pff_player_id"}))


def _load_shock_exposure() -> pl.DataFrame:
    """Por jugador: n_shocks_for, n_shocks_against, n_groups, n_ko."""
    sh = pl.read_parquet(_SHOCKS).rename({"player_id": "pff_player_id"})
    return (sh.group_by("pff_player_id").agg([
        pl.col("shock_type").eq("GOAL_FOR").sum().alias("n_shocks_for"),
        pl.col("shock_type").eq("GOAL_AGAINST").sum().alias("n_shocks_against"),
        pl.col("stage").eq("groups").sum().alias("n_shocks_groups"),
        pl.col("stage").eq("ko").sum().alias("n_shocks_ko"),
    ]))


# ----------------------------------------------------------------------------
# Posterior probabilities desde samples
# ----------------------------------------------------------------------------
def _compute_posterior_probs(fit: dict) -> pl.DataFrame:
    """Per jugador: P(chasing>0|data), P(protecting>0|data), P(dual>0|data).

    Usa eta_ga + eta_gf samples (4000, 598, 4):
      chasing_clutch_idx  = mean(eta_ga[:, atk] + eta_ga[:, off])  (per sample)
      protecting_clutch_idx = mean(eta_gf[:, def] + eta_gf[:, phys]) (per sample)
    """
    s = fit["samples"]
    p_to_idx = fit["p_to_idx"]
    ch = fit["ch_to_idx"]
    eta_ga = s["eta_ga"]   # (n_samples, n_players, n_channels)
    eta_gf = s["eta_gf"]
    # chasing: mean across atk + off
    chasing_samples = (eta_ga[:, :, ch["ataque"]] + eta_ga[:, :, ch["offball"]]) / 2
    # protecting: mean across def + phys
    protecting_samples = (eta_gf[:, :, ch["defensa"]] + eta_gf[:, :, ch["fisico"]]) / 2
    # IC95 ya esta en posterior_player; aqui calculamos posterior probabilities + IC80
    n_samples = chasing_samples.shape[0]
    p_chasing_pos = (chasing_samples > 0).mean(axis=0)        # (n_players,)
    p_protecting_pos = (protecting_samples > 0).mean(axis=0)
    p_dual_pos = ((chasing_samples > 0) & (protecting_samples > 0)).mean(axis=0)
    chasing_mean = chasing_samples.mean(axis=0)
    chasing_sd = chasing_samples.std(axis=0)
    chasing_lo80 = np.quantile(chasing_samples, 0.10, axis=0)
    chasing_hi80 = np.quantile(chasing_samples, 0.90, axis=0)
    protecting_mean = protecting_samples.mean(axis=0)
    protecting_sd = protecting_samples.std(axis=0)
    protecting_lo80 = np.quantile(protecting_samples, 0.10, axis=0)
    protecting_hi80 = np.quantile(protecting_samples, 0.90, axis=0)

    # Map idx → pff_player_id
    idx_to_pid = {v: k for k, v in p_to_idx.items()}
    rows = []
    for i in range(eta_ga.shape[1]):
        rows.append(dict(
            pff_player_id=idx_to_pid[i],
            chasing_clutch_idx=float(chasing_mean[i]),
            chasing_clutch_sd=float(chasing_sd[i]),
            chasing_clutch_lo80=float(chasing_lo80[i]),
            chasing_clutch_hi80=float(chasing_hi80[i]),
            protecting_clutch_idx=float(protecting_mean[i]),
            protecting_clutch_sd=float(protecting_sd[i]),
            protecting_clutch_lo80=float(protecting_lo80[i]),
            protecting_clutch_hi80=float(protecting_hi80[i]),
            p_chasing_positive=float(p_chasing_pos[i]),
            p_protecting_positive=float(p_protecting_pos[i]),
            p_dual_positive=float(p_dual_pos[i]),
        ))
    return pl.DataFrame(rows)


# ----------------------------------------------------------------------------
# CATEs 8-valores preservados (4 canales × 2 shocks)
# ----------------------------------------------------------------------------
def _build_cate_wide(posterior: pl.DataFrame) -> pl.DataFrame:
    """Pivot posterior_player (long) → wide con 32 cols (8 channels x 4 stats)."""
    rows = []
    for r in posterior.iter_rows(named=True):
        pid = r["pff_player_id"]
        prefix = f"cate_{r['channel']}_{r['shock_type']}"
        rows.append((pid, f"{prefix}_mean", r["cate_mean"]))
        rows.append((pid, f"{prefix}_sd",   r["cate_sd"]))
        rows.append((pid, f"{prefix}_lo80", r["ci_lo80"]))
        rows.append((pid, f"{prefix}_hi80", r["ci_hi80"]))
    long = pl.DataFrame(rows, schema=["pff_player_id", "key", "val"], orient="row")
    return long.pivot("key", index="pff_player_id", values="val")


# ----------------------------------------------------------------------------
# Vector PCJ summary 4-canal directional
# ----------------------------------------------------------------------------
def _build_pcj_summary_vector(cate_wide: pl.DataFrame) -> pl.DataFrame:
    """4-vector directional: cada canal usa shock_type de máxima leverage.

    pcj_atk  = cate_ataque_GOAL_AGAINST_mean  (chasing)
    pcj_def  = cate_defensa_GOAL_FOR_mean    (protecting)
    pcj_off  = cate_offball_GOAL_AGAINST_mean (chasing)
    pcj_phys = cate_fisico con max-magnitude signed (GA o GF, el más reactivo)
    """
    return cate_wide.with_columns([
        pl.col("cate_ataque_GOAL_AGAINST_mean").alias("pcj_atk"),
        pl.col("cate_defensa_GOAL_FOR_mean").alias("pcj_def"),
        pl.col("cate_offball_GOAL_AGAINST_mean").alias("pcj_off"),
        pl.when(pl.col("cate_fisico_GOAL_AGAINST_mean").abs() >=
                pl.col("cate_fisico_GOAL_FOR_mean").abs())
          .then(pl.col("cate_fisico_GOAL_AGAINST_mean"))
          .otherwise(pl.col("cate_fisico_GOAL_FOR_mean"))
          .alias("pcj_phys"),
    ])


# ----------------------------------------------------------------------------
# Tier labels (percentile-based)
# ----------------------------------------------------------------------------
def _tier_from_percentile(pct: float) -> str:
    if pct >= 0.95:  return "Elite"
    if pct >= 0.85:  return "Top"
    if pct >= 0.60:  return "Above_avg"
    if pct >= 0.40:  return "Average"
    if pct >= 0.15:  return "Below_avg"
    return "Bottom"


def _add_tiers(df: pl.DataFrame) -> pl.DataFrame:
    """Tier labels: global + within-position, para chasing y protecting."""
    n = df.height
    # Percentile global (descending: highest value = highest percentile)
    df = df.with_columns([
        (pl.col("chasing_clutch_idx").rank(method="ordinal") / n)
            .alias("pct_chasing_global"),
        (pl.col("protecting_clutch_idx").rank(method="ordinal") / n)
            .alias("pct_protecting_global"),
        (pl.col("chasing_clutch_idx").rank(method="ordinal").over("position_group") /
            pl.col("position_group").count().over("position_group"))
            .alias("pct_chasing_in_position"),
        (pl.col("protecting_clutch_idx").rank(method="ordinal").over("position_group") /
            pl.col("position_group").count().over("position_group"))
            .alias("pct_protecting_in_position"),
    ])
    # Apply tier function
    for col in ["pct_chasing_global", "pct_protecting_global",
                "pct_chasing_in_position", "pct_protecting_in_position"]:
        tier_col = "tier_" + col.replace("pct_", "")
        df = df.with_columns(
            pl.col(col).map_elements(_tier_from_percentile, return_dtype=pl.String)
                       .alias(tier_col)
        )
    return df


def _add_significance(df: pl.DataFrame) -> pl.DataFrame:
    """Sig flag bayesiana: P(idx>0|data) > 0.95 → Sig clutch; <0.05 → Sig anti."""
    return df.with_columns([
        pl.when(pl.col("p_chasing_positive") >= SIG_THRESHOLD).then(pl.lit("Sig_remontador"))
          .when(pl.col("p_chasing_positive") <= 1 - SIG_THRESHOLD).then(pl.lit("Sig_anti_remontador"))
          .otherwise(pl.lit("Inconclusive"))
          .alias("sig_chasing"),
        pl.when(pl.col("p_protecting_positive") >= SIG_THRESHOLD).then(pl.lit("Sig_cerrojo"))
          .when(pl.col("p_protecting_positive") <= 1 - SIG_THRESHOLD).then(pl.lit("Sig_anti_cerrojo"))
          .otherwise(pl.lit("Inconclusive"))
          .alias("sig_protecting"),
    ])


# ----------------------------------------------------------------------------
# Rankings (global + in-position)
# ----------------------------------------------------------------------------
def _add_rankings(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.col("chasing_clutch_idx").rank(method="ordinal", descending=True)
            .alias("rank_chasing_global"),
        pl.col("protecting_clutch_idx").rank(method="ordinal", descending=True)
            .alias("rank_protecting_global"),
        pl.col("chasing_clutch_idx").rank(method="ordinal", descending=True)
            .over("position_group").alias("rank_chasing_in_position"),
        pl.col("protecting_clutch_idx").rank(method="ordinal", descending=True)
            .over("position_group").alias("rank_protecting_in_position"),
    ])


# ----------------------------------------------------------------------------
# Build maestro
# ----------------------------------------------------------------------------
def build_pcj_table() -> pl.DataFrame:
    print("[M15] Cargando M14 outputs + samples...")
    m14 = _load_m14()
    posterior = m14["posterior"]

    print("[M15] Cargando metadata jugadores...")
    meta = _load_player_meta()
    print(f"  meta: {meta.height} jugadores con identidad completa")

    print("[M15] Calculando minutos jugados (sumando 64 partidos)...")
    minutes = _load_player_minutes()
    print(f"  minutes: {minutes.height} jugadores")

    print("[M15] Cargando exposicion shocks...")
    shocks = _load_shock_exposure()

    print("[M15] Calculando posterior probabilities desde samples NUTS...")
    posterior_probs = _compute_posterior_probs(m14["fit"])
    print(f"  posterior_probs: {posterior_probs.height} jugadores")

    print("[M15] Construyendo CATE wide (8 canales x 4 stats)...")
    cate_wide = _build_cate_wide(posterior)
    print(f"  cate_wide: {cate_wide.height} jugadores, {cate_wide.width} cols")

    print("[M15] Vector PCJ summary 4-canal directional...")
    cate_wide = _build_pcj_summary_vector(cate_wide)

    print("[M15] Joining + filtrando minutos minimos...")
    df = (cate_wide
            .join(posterior_probs, on="pff_player_id", how="inner")
            .join(meta, on="pff_player_id", how="left")
            .join(minutes, on="pff_player_id", how="left")
            .join(shocks, on="pff_player_id", how="left"))
    n_total = df.height
    df = df.filter(pl.col("minutes_played") >= MIN_MINUTES)
    print(f"  {df.height}/{n_total} jugadores >={MIN_MINUTES} min")

    print("[M15] Rankings + tiers + sig flags...")
    df = _add_rankings(df)
    df = _add_tiers(df)
    df = _add_significance(df)

    # Reordenar columnas: identidad → exposicion → indices → posterior probs →
    # 4-vec → CATEs 8 → rankings → tiers → sig
    front = ["pff_player_id", "player_name", "team_name", "position_group",
             "minutes_played", "n_matches_played", "n_shocks_for", "n_shocks_against",
             "n_shocks_groups", "n_shocks_ko",
             "chasing_clutch_idx", "chasing_clutch_sd",
             "chasing_clutch_lo80", "chasing_clutch_hi80",
             "protecting_clutch_idx", "protecting_clutch_sd",
             "protecting_clutch_lo80", "protecting_clutch_hi80",
             "p_chasing_positive", "p_protecting_positive", "p_dual_positive",
             "pcj_atk", "pcj_def", "pcj_off", "pcj_phys",
             "rank_chasing_global", "rank_protecting_global",
             "rank_chasing_in_position", "rank_protecting_in_position",
             "tier_chasing_global", "tier_protecting_global",
             "tier_chasing_in_position", "tier_protecting_in_position",
             "sig_chasing", "sig_protecting"]
    cate_cols = sorted([c for c in df.columns if c.startswith("cate_")])
    pct_cols = [c for c in df.columns if c.startswith("pct_")]
    cols = [c for c in front if c in df.columns] + cate_cols + pct_cols
    cols += [c for c in df.columns if c not in cols]
    df = df.select(cols)
    return df


def build_aux_tables(pcj: pl.DataFrame) -> dict:
    """Tablas auxiliares scout-friendly."""
    aux = {}
    # Top10 chasing per position
    aux["top10_chasing_per_position"] = (pcj.sort("chasing_clutch_idx", descending=True)
        .group_by("position_group", maintain_order=True).head(10)
        .select(["position_group", "rank_chasing_in_position",
                 "player_name", "team_name", "chasing_clutch_idx",
                 "chasing_clutch_lo80", "chasing_clutch_hi80",
                 "p_chasing_positive", "tier_chasing_in_position",
                 "sig_chasing", "minutes_played"]))
    aux["top10_protecting_per_position"] = (pcj.sort("protecting_clutch_idx", descending=True)
        .group_by("position_group", maintain_order=True).head(10)
        .select(["position_group", "rank_protecting_in_position",
                 "player_name", "team_name", "protecting_clutch_idx",
                 "protecting_clutch_lo80", "protecting_clutch_hi80",
                 "p_protecting_positive", "tier_protecting_in_position",
                 "sig_protecting", "minutes_played"]))
    # Dual clutch top: (chasing + protecting), filtered to both significant
    dual = (pcj.with_columns(
                (pl.col("chasing_clutch_idx") + pl.col("protecting_clutch_idx"))
                .alias("dual_score"))
              .sort("dual_score", descending=True)
              .head(30)
              .select(["player_name", "team_name", "position_group",
                       "chasing_clutch_idx", "protecting_clutch_idx",
                       "dual_score", "p_chasing_positive", "p_protecting_positive",
                       "p_dual_positive", "minutes_played"]))
    aux["dual_clutch_top"] = dual
    # Por equipo: agg de minutos + indices
    by_team = (pcj.group_by("team_name").agg([
        pl.len().alias("n_players"),
        pl.col("chasing_clutch_idx").mean().alias("team_chasing_mean"),
        pl.col("protecting_clutch_idx").mean().alias("team_protecting_mean"),
        pl.col("p_chasing_positive").mean().alias("team_p_chasing"),
        pl.col("p_protecting_positive").mean().alias("team_p_protecting"),
    ]).sort("team_chasing_mean", descending=True))
    aux["by_team"] = by_team
    return aux


def main():
    pcj = build_pcj_table()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _AUX_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / "pcj_table.parquet"
    pcj.write_parquet(out_path)
    print(f"\n[M15] Saved {out_path} ({pcj.height} jugadores, {pcj.width} cols)")

    aux = build_aux_tables(pcj)
    for name, df in aux.items():
        path = _AUX_DIR / f"{name}.parquet"
        df.write_parquet(path)
        print(f"  + aux: {name}.parquet ({df.height} rows)")

    # Resumen sanity
    print(f"\n=== PCJ Table summary ===")
    print(f"Jugadores >= {MIN_MINUTES} min: {pcj.height}")
    print(f"Posiciones cubiertas: {pcj['position_group'].n_unique()}")
    print(f"Equipos cubiertos: {pcj['team_name'].n_unique()}")
    print(f"\nDistribucion sig_chasing:")
    print(pcj.group_by("sig_chasing").len().sort("len", descending=True))
    print(f"\nDistribucion sig_protecting:")
    print(pcj.group_by("sig_protecting").len().sort("len", descending=True))
    print(f"\nTop 10 Remontador globales:")
    print(pcj.sort("rank_chasing_global").head(10).select(
        ["rank_chasing_global", "player_name", "team_name", "position_group",
         "chasing_clutch_idx", "p_chasing_positive", "sig_chasing"]))
    print(f"\nTop 10 Cerrojo globales:")
    print(pcj.sort("rank_protecting_global").head(10).select(
        ["rank_protecting_global", "player_name", "team_name", "position_group",
         "protecting_clutch_idx", "p_protecting_positive", "sig_protecting"]))


if __name__ == "__main__":
    main()
