"""
M09_defensa - Canal Solidez Defensiva.

Fase 2 PCJ, canal 2 de 4. Valora la contribucion defensiva individual por
jugador-minuto combinando on-ball (VAEP) con off-ball (tracking PFF 25 Hz).

Reutiliza el modelo atomic-VAEP entrenado en M08 (CatBoost 5-fold CV +
Optuna + isotonic). defensive_value(action) mide cuanto REDUCE
P(encajar_en_10_acciones) la accion del defensor (formula atomic-VAEP).

Cuatro sub-canales agregados per (match, player, minute):
  1. score_def_minute       : sum(defensive_value) sobre TODAS las acciones on-ball.
  2. vdep_minute            : sum(defensive_value) FILTRADO a acciones defensivas
                              (tackle, interception, clearance, foul, keeper_*).
                              Equivalente a VDEP (Toda 2022 PLOS ONE) sin entrenar
                              modelo separado - misma cabeza p(concedes)
                              condicionada a acciones defensivas.
  3. def_third_pct          : fraccion de frames en el tercio defensivo propio
                              durante posesion rival (bloque bajo).
  4. press_intensity_frames : # frames-jugador a <= 3 m del balon durante posesion
                              rival (aprox. Bekkers 2024 arXiv:2501.04712).

Output:
  data/parquet/derived/defensa/
    def_third_context.parquet    # (pff_match_id, player_id, minute,
                                 #  def_third_pct, press_intensity_frames,
                                 #  oppo_possession_frames)
    per_minute.parquet           # sb_match_id + pff_match_id + sb/pff_player_id +
                                 #  minute + score_def_minute + vdep_minute +
                                 #  n_def_actions + n_actions_total +
                                 #  def_third_pct + press_intensity_frames +
                                 #  oppo_possession_frames
    per_shock_window.parquet     # (match_id, shock_id, pff_player_id, shock_type,
                                 #  score_def_{pre,post}, vdep_{pre,post},
                                 #  n_def_actions_{pre,post}, press_frames_{pre,post})

Acceptance (ARCHITECTURE): distribucion score_def por rol coherente
(CBs y DMs > CFs); GK score_def positivo por saves etc.

Depende de: M08 (modelo VAEP + atomic SPADL WC22 + mapping SB->PFF),
M01 (tracking PFF + rosters), M03 (attacking_direction, SB<->PFF match map),
M07 (shocks table).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from M01_loader_pff import (
    load_rosters, load_metadata, scan_tracking, list_event_match_ids,
)
from M03_preprocess import attacking_direction, _pff_to_sb_match_id
from M07_shocks import build_shocks_table
import M08_ataque as atk


# -- Rutas ------------------------------------------------------------------

_REPO    = Path(__file__).resolve().parents[1]
_DERIVED = _REPO / "data" / "parquet" / "derived" / "defensa"


# Tipos de accion atomic-SPADL que son defensivas (Decroos & Davis 2020).
_DEF_ACTION_TYPES = {
    "tackle", "interception", "clearance",
    "foul",                # defensa agresiva
    "keeper_save",         # paradas del portero
    "keeper_claim",
    "keeper_punch",
    "keeper_pick_up",
}

# Umbral (metros) para "presionar": defensor a <= radio del balon.
_PRESS_RADIUS_M    = 3.0
# Hysteresis de posesion: frames de debounce para home_ball flips espurios
# (corner, lateral). 2 frames @ 25 Hz = 80 ms, suficiente para descartar
# transiciones instantaneas sin perder recoveries reales.
_POSS_HYSTERESIS_F = 2


# ===========================================================================
#  SECCION 0 — Contexto defensivo off-ball via tracking PFF (bloque bajo)
# ===========================================================================

_DEF_CTX_SCHEMA = {
    "pff_match_id": pl.Int64, "player_id": pl.Int64, "minute": pl.Int64,
    "def_third_pct": pl.Float64, "press_intensity_frames": pl.Int64,
    "oppo_possession_frames": pl.Int64,
}


def _smooth_possession(home_ball: np.ndarray, k: int = _POSS_HYSTERESIS_F) -> np.ndarray:
    """Debounce hysteresis: require k consecutive frames of same value to flip.

    Implementacion O(n) sin copia: recorre la secuencia y mantiene ultimo valor
    estable; un flip solo se acepta tras k frames del nuevo valor.
    """
    n = len(home_ball)
    if n == 0 or k <= 1:
        return home_ball.astype(bool)
    out = np.empty(n, dtype=bool)
    last_stable = bool(home_ball[0])
    run = 0
    current = last_stable
    for i in range(n):
        v = bool(home_ball[i])
        if v == current:
            run += 1
        else:
            current = v
            run = 1
        if run >= k:
            last_stable = current
        out[i] = last_stable
    return out


def _def_third_pct_match(match_id: int) -> pl.DataFrame:
    """Contexto off-ball por jugador-minuto via tracking PFF 25 Hz (vectorizado).

    Tres metricas derivadas en 1 pasada polars (explode + joins, sin loops py):
      - def_third_frames        : jugador en tercio defensivo durante posesion rival.
      - press_intensity_frames  : jugador a <= PRESS_RADIUS_M del balon durante
                                    posesion rival (aprox. Bekkers 2024).
      - oppo_possession_frames  : denominador (frames en posesion rival).

    Posesion usa home_ball con hysteresis _POSS_HYSTERESIS_F (filtra flips
    espurios en corner/lateral). Coords PFF (metros, centro 0,0).
    """
    md = load_metadata(match_id).row(0, named=True)
    home_id = md["home_team_id"]
    away_id = md["away_team_id"]
    pitch_length = float(md.get("pitch_length") or 105.0)
    def_third_thr = -pitch_length / 6.0
    fps = float(md.get("fps") or 25.0)
    frames_per_min = fps * 60.0
    press_r2 = _PRESS_RADIUS_M ** 2

    # Lookup (team_id, shirt) -> player_id desde rosters
    ro = load_rosters(match_id).select(["team_id", "player_id", "shirt_number"]) \
        .filter(pl.col("shirt_number").is_not_null()) \
        .with_columns(pl.col("shirt_number").cast(pl.Int64, strict=False)
                        .alias("jersey_int"))

    # Lookup (team_id, period) -> direction ('R'|'L') -> sign (+1|-1)
    dir_df = attacking_direction(match_id).with_columns(
        pl.when(pl.col("direction") == "R").then(1.0).otherwise(-1.0).alias("def_sign")
    ).select(["team_id", "period", "def_sign"])

    frames = scan_tracking(match_id).select([
        pl.col("frameNum"),
        pl.col("period"),
        pl.col("game_event").struct.field("home_ball").alias("home_has_ball"),
        pl.col("homePlayersSmoothed").alias("home_players"),
        pl.col("awayPlayersSmoothed").alias("away_players"),
        pl.col("ball").list.first().struct.field("x").alias("bx"),
        pl.col("ball").list.first().struct.field("y").alias("by"),
    ]).filter(pl.col("home_has_ball").is_not_null()).collect()

    if frames.height == 0:
        return pl.DataFrame(schema=_DEF_CTX_SCHEMA)

    # Hysteresis sobre home_ball
    hb = _smooth_possession(frames["home_has_ball"].to_numpy().astype(bool),
                             _POSS_HYSTERESIS_F)
    frames = frames.with_columns(pl.Series("home_has_ball_smooth", hb))

    # Proceso cada side como DF separado (defendiendo = side no tiene balon)
    def _side_frame(players_col: str, side_team_id: int, defending_mask: pl.Expr) -> pl.DataFrame:
        sel = frames.filter(defending_mask).select([
            "frameNum", "period", "bx", "by",
            pl.col(players_col).alias("players"),
        ]).explode("players").filter(pl.col("players").is_not_null()).with_columns([
            pl.col("players").struct.field("x").alias("x"),
            pl.col("players").struct.field("y").alias("y"),
            pl.col("players").struct.field("jerseyNum")
                              .cast(pl.Int64, strict=False).alias("jersey_int"),
            pl.lit(side_team_id, dtype=pl.Int64).alias("team_id"),
        ]).filter(pl.col("x").is_not_null() & pl.col("jersey_int").is_not_null())
        return sel

    home_def = _side_frame("home_players", home_id, ~pl.col("home_has_ball_smooth"))
    away_def = _side_frame("away_players", away_id,  pl.col("home_has_ball_smooth"))
    all_def = pl.concat([home_def, away_def]) if (home_def.height + away_def.height) > 0 \
              else home_def  # schema vacio consistente

    if all_def.height == 0:
        return pl.DataFrame(schema=_DEF_CTX_SCHEMA)

    # Joins: jersey -> player_id, direction -> sign
    all_def = all_def.join(
        ro.select(["team_id", "jersey_int", "player_id"]),
        on=["team_id", "jersey_int"], how="inner",
    ).join(dir_df, on=["team_id", "period"], how="left")

    # Metricas por frame-jugador
    agg = all_def.with_columns([
        (pl.col("frameNum") / frames_per_min).cast(pl.Int64).alias("minute"),
        ((pl.col("def_sign") * pl.col("x")) < def_third_thr).alias("in_def_third"),
        (((pl.col("x") - pl.col("bx")) ** 2
          + (pl.col("y") - pl.col("by")) ** 2) <= press_r2).alias("pressing"),
    ]).group_by(["player_id", "minute"]).agg([
        pl.col("in_def_third").sum().alias("def_third_frames"),
        pl.col("pressing").sum().alias("press_intensity_frames"),
        pl.len().alias("oppo_possession_frames"),
    ]).with_columns([
        (pl.col("def_third_frames") / pl.col("oppo_possession_frames"))
         .alias("def_third_pct"),
        pl.lit(match_id).cast(pl.Int64).alias("pff_match_id"),
    ]).select(list(_DEF_CTX_SCHEMA.keys()))
    return agg


def build_def_third_all(cache: bool = True) -> pl.DataFrame:
    """Agrega def_third_pct + press_intensity_frames para los 64 partidos WC22."""
    cache_path = _DERIVED / "def_third_context.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)
    import time
    dfs = []
    t0 = time.time()
    for i, mid in enumerate(list_event_match_ids()):
        try:
            dfs.append(_def_third_pct_match(mid))
        except Exception as e:
            print(f"  skip {mid}: {e}")
        if (i+1) % 10 == 0:
            print(f"  {i+1}/64 en {time.time()-t0:.1f}s", flush=True)
    out = pl.concat(dfs) if dfs else pl.DataFrame(schema=_DEF_CTX_SCHEMA)
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(cache_path, compression="snappy")
    return out


# ===========================================================================
#  SECCION 1 — Aggregation per player-minute
# ===========================================================================

def aggregate_per_player_minute(cache: bool = True) -> pl.DataFrame:
    """Agrega defensive_value por (match_id, player_id_sb, minute) + n_def_actions.

    Reusa wc22_with_vaep del pipeline M08 (aplica VAEP calibrado a WC22 atomic).
    """
    cache_path = _DERIVED / "per_minute.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    # Load M08 fit + WC22 atomic + apply VAEP (calibrado)
    fit = atk.load_models()
    wc22_atomic = atk.build_wc22_atomic(overwrite=False)
    wc22_with_vaep = atk.apply_vaep_to_wc22(fit, wc22_atomic)

    # Agregar por (match_id, player, minute)
    df = pl.from_pandas(wc22_with_vaep[[
        "game_id", "period_id", "time_seconds", "team_id",
        "player_id", "type_name", "defensive_value", "vaep_value",
    ]])
    df = df.with_columns([
        (pl.col("time_seconds") // 60
         + (pl.col("period_id") - 1) * 45).cast(pl.Int64).alias("minute"),
        pl.col("type_name").is_in(list(_DEF_ACTION_TYPES)).alias("is_def_action"),
    ]).filter(pl.col("player_id").is_not_null())

    # vdep_contrib = defensive_value solo en acciones defensivas (VDEP puro Toda 2022)
    df = df.with_columns(
        pl.when(pl.col("is_def_action")).then(pl.col("defensive_value"))
          .otherwise(0.0).alias("vdep_contrib")
    )

    agg = df.group_by(["game_id", "player_id", "minute"]).agg([
        pl.col("defensive_value").sum().alias("score_def_minute"),
        pl.col("vdep_contrib").sum().alias("vdep_minute"),
        pl.col("is_def_action").sum().alias("n_def_actions"),
        pl.len().alias("n_actions_total"),
    ]).rename({"game_id": "sb_match_id", "player_id": "sb_player_id"})

    # Join con CONTEXTO off-ball (def_third_pct via tracking PFF).
    # Necesita mapeo sb_player_id -> pff_player_id y sb_match_id -> pff_match_id.
    sb2pff = {v: k for k, v in _pff_to_sb_match_id().items()}
    player_map = atk.build_sb_to_pff_player_map(cache=True).select([
        "sb_player_id", "pff_player_id",
    ]).with_columns([
        pl.col("sb_player_id").cast(pl.Int64),
        pl.col("pff_player_id").cast(pl.Int64, strict=False),
    ])

    agg = agg.with_columns([
        pl.col("sb_player_id").cast(pl.Int64),
        pl.col("sb_match_id").replace_strict(sb2pff, default=None).alias("pff_match_id"),
    ]).join(player_map, on="sb_player_id", how="left")

    def_ctx = build_def_third_all(cache=True)
    if def_ctx.height > 0:
        def_ctx_cast = def_ctx.with_columns([
            pl.col("pff_match_id").cast(pl.Int64),
            pl.col("player_id").cast(pl.Int64).alias("pff_player_id"),
            pl.col("minute").cast(pl.Int64),
        ]).select(["pff_match_id", "pff_player_id", "minute",
                   "def_third_pct", "press_intensity_frames",
                   "oppo_possession_frames"])
        agg = agg.join(def_ctx_cast,
                        on=["pff_match_id", "pff_player_id", "minute"], how="left")
    else:
        agg = agg.with_columns([
            pl.lit(None, dtype=pl.Float64).alias("def_third_pct"),
            pl.lit(None, dtype=pl.Int64).alias("press_intensity_frames"),
            pl.lit(None, dtype=pl.Int64).alias("oppo_possession_frames"),
        ])

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        agg.write_parquet(cache_path, compression="snappy")
    return agg


# ===========================================================================
#  SECCION 2 — Aggregation per shock window
# ===========================================================================

def aggregate_per_shock_window(cache: bool = True) -> pl.DataFrame:
    """Por cada (shock, player), suma score_def y n_def_actions en pre/post."""
    cache_path = _DERIVED / "per_shock_window.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    per_min = aggregate_per_player_minute(cache=True)
    player_map = atk.build_sb_to_pff_player_map(cache=True)
    shocks = build_shocks_table(cache=True, overwrite=False)

    # Map sb_match_id -> pff_match_id
    sb2pff = {v: k for k, v in _pff_to_sb_match_id().items()}

    per_min = per_min.with_columns([
        pl.col("sb_match_id").replace_strict(sb2pff, default=None).alias("match_id"),
        pl.col("sb_player_id").cast(pl.Int64),
    ]).filter(pl.col("match_id").is_not_null())

    pm_cast = player_map.select(["sb_player_id", "pff_player_id"]).with_columns([
        pl.col("sb_player_id").cast(pl.Int64),
        pl.col("pff_player_id").cast(pl.Int64, strict=False),
    ])
    per_min = per_min.join(pm_cast, on="sb_player_id", how="left") \
                      .filter(pl.col("pff_player_id").is_not_null())

    # Join con shocks + filter windows
    shocks_slim = shocks.select([
        "match_id", "shock_id", "player_id", "shock_type",
        "window_pre_start", "window_pre_end",
        "window_post_start", "window_post_end",
    ]).rename({"player_id": "pff_player_id"})

    joined = shocks_slim.join(per_min, on=["match_id", "pff_player_id"], how="left") \
                        .with_columns((pl.col("minute") * 60).alias("min_sec"))

    pre = joined.filter(
        (pl.col("min_sec") >= pl.col("window_pre_start")) &
        (pl.col("min_sec") < pl.col("window_pre_end"))
    ).group_by(["match_id","shock_id","pff_player_id","shock_type"]).agg([
        pl.col("score_def_minute").sum().alias("score_def_pre"),
        pl.col("vdep_minute").sum().alias("vdep_pre"),
        pl.col("n_def_actions").sum().alias("n_def_actions_pre"),
        pl.col("press_intensity_frames").sum().alias("press_frames_pre"),
    ])
    post = joined.filter(
        (pl.col("min_sec") >= pl.col("window_post_start")) &
        (pl.col("min_sec") <= pl.col("window_post_end"))
    ).group_by(["match_id","shock_id","pff_player_id","shock_type"]).agg([
        pl.col("score_def_minute").sum().alias("score_def_post"),
        pl.col("vdep_minute").sum().alias("vdep_post"),
        pl.col("n_def_actions").sum().alias("n_def_actions_post"),
        pl.col("press_intensity_frames").sum().alias("press_frames_post"),
    ])

    base = shocks.select([
        "match_id", "shock_id", "player_id", "shock_type"
    ]).rename({"player_id": "pff_player_id"}).unique()

    out = base.join(pre,  on=["match_id","shock_id","pff_player_id","shock_type"], how="left") \
              .join(post, on=["match_id","shock_id","pff_player_id","shock_type"], how="left") \
              .with_columns([
                  pl.col("score_def_pre").fill_null(0.0),
                  pl.col("score_def_post").fill_null(0.0),
                  pl.col("vdep_pre").fill_null(0.0),
                  pl.col("vdep_post").fill_null(0.0),
                  pl.col("n_def_actions_pre").fill_null(0),
                  pl.col("n_def_actions_post").fill_null(0),
                  pl.col("press_frames_pre").fill_null(0),
                  pl.col("press_frames_post").fill_null(0),
              ])

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(cache_path, compression="snappy")
    return out


# -- Sanity inline ---------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=== M09_defensa sanity ===")

    t0 = time.time()
    print("\n[1] Aggregating score_def + context def_third via tracking PFF...")
    per_min = aggregate_per_player_minute(cache=True)
    print(f"  filas: {per_min.height:,} en {time.time()-t0:.1f}s")
    print(f"  cols: {per_min.columns}")
    print(f"  score_def range : [{per_min['score_def_minute'].min():.3f}, "
          f"{per_min['score_def_minute'].max():.3f}]")
    if "def_third_pct" in per_min.columns:
        ctx_valid = per_min.filter(pl.col("def_third_pct").is_not_null())
        print(f"  def_third_pct valido: {ctx_valid.height}/{per_min.height} "
              f"({100*ctx_valid.height/per_min.height:.1f}%)")
        if ctx_valid.height > 0:
            print(f"  def_third_pct range: [{ctx_valid['def_third_pct'].min():.3f}, "
                  f"{ctx_valid['def_third_pct'].max():.3f}]")

    # Acceptance: distribucion por rol (CBs + DMs > CFs)
    print("\n[2] Acceptance — score_def por rol (CBs/DMs > CFs):")
    player_map = atk.build_sb_to_pff_player_map(cache=True)
    pm_cast = per_min.with_columns(pl.col("sb_player_id").cast(pl.Int64))
    map_cast = player_map.select(["sb_player_id","pff_player_id"]).with_columns([
        pl.col("sb_player_id").cast(pl.Int64),
        pl.col("pff_player_id").cast(pl.Int64, strict=False),
    ])
    pm_roles = pm_cast.join(map_cast, on="sb_player_id", how="left") \
                      .filter(pl.col("pff_player_id").is_not_null())
    ro = load_rosters().select(["player_id","position_group"]) \
                       .unique(subset=["player_id"]) \
                       .rename({"player_id": "pff_player_id"})
    pm_roles = pm_roles.join(ro, on="pff_player_id", how="left")
    by_role = pm_roles.group_by("position_group").agg([
        pl.col("score_def_minute").mean().alias("mean_def_per_min"),
        pl.col("n_def_actions").mean().alias("mean_n_def"),
        pl.len().alias("n_minutes"),
    ]).sort("mean_def_per_min", descending=True)
    print(by_role)

    # [3] Shock-window aggregation
    t0 = time.time()
    print("\n[3] Aggregating per shock window...")
    per_shock = aggregate_per_shock_window(cache=True)
    print(f"  filas: {per_shock.height:,} en {time.time()-t0:.1f}s")
    summary = per_shock.group_by("shock_type").agg([
        pl.col("score_def_pre").mean().alias("mean_pre"),
        pl.col("score_def_post").mean().alias("mean_post"),
        (pl.col("score_def_post") - pl.col("score_def_pre")).mean().alias("mean_delta"),
    ])
    print("  score_def por shock_type:")
    print(summary)
