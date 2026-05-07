"""
M09_defensa - Canal Solidez Defensiva.

Fase 2 PCJ, canal 2 de 4. Valora la contribucion defensiva individual por
jugador-minuto combinando on-ball (VAEP) con off-ball (tracking PFF 25 Hz).

Reutiliza el modelo atomic-VAEP entrenado en M08 (CatBoost 5-fold CV +
Optuna + isotonic). defensive_value(action) mide cuanto REDUCE
P(encajar_en_10_acciones) la accion del defensor (formula atomic-VAEP).

Cuatro sub-canales agregados per (match, player, minute):
  1. score_def_minute       : sum(defensive_value) sobre TODAS las acciones on-ball.
  2. vdep_like_minute       : sum(defensive_value) FILTRADO a acciones defensivas
                              (tackle, interception, clearance, foul, keeper_*).
                              **VDEP-LIKE, no VDEP stricto.** Toda et al. 2022
                              (PLOS ONE) entrena una cabeza dedicada
                              P(recovery) − C·P(attacked) con XGBoost; aqui
                              reusamos la cabeza P(concedes) de atomic-VAEP M08
                              condicionada al subset de acciones defensivas.
                              Equivalente bajo mismo horizonte (10 acciones) y
                              corpus, pero NO es VDEP fiel a la spec original.
                              Renombrado de `vdep_minute` -> `vdep_like_minute`
                              para no afirmar el metodo Toda explicitamente.
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
    load_events,
)
from M03_preprocess import (
    attacking_direction, pff_to_sb_match_id, sb_to_pff_match_id,
)
from M07_shocks import build_shocks_table, attach_team_loo
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
        pl.col("ballsSmoothed").struct.field("x").alias("bx"),
        pl.col("ballsSmoothed").struct.field("y").alias("by"),
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
        pl.col("in_def_third").sum().cast(pl.Int64).alias("def_third_frames"),
        pl.col("pressing").sum().cast(pl.Int64).alias("press_intensity_frames"),
        pl.len().cast(pl.Int64).alias("oppo_possession_frames"),
    ]).with_columns([
        (pl.col("def_third_frames") / pl.col("oppo_possession_frames"))
         .alias("def_third_pct"),
        pl.lit(match_id).cast(pl.Int64).alias("pff_match_id"),
    ]).select(list(_DEF_CTX_SCHEMA.keys()))
    return agg


_PRESS_VALUE_SCHEMA = {
    "pff_match_id": pl.Int64, "pff_player_id": pl.Int64,
    "period": pl.Int64, "minute_in_period": pl.Int64,
    "press_value_minute": pl.Float64,
}

# Pesos PFF press types (Maejima 2024 attribution + Lee 2025 exPress weighting)
# A: attempted (intento, sin contacto efectivo)
# L: passing-lane (cierra opcion de pase, presion sin contacto directo)
# P: player-pressured (presion EFECTIVA — el carrier acuso la presion)
_PRESS_TYPE_WEIGHT = {"A": 0.5, "L": 1.0, "P": 2.0}
_BAD_TOUCH_BOOST = 1.5    # multiplica si initialTouchType ∈ {M, B}: la presion ROMPIO el touch


def _pff_press_value_per_minute(match_id: int) -> pl.DataFrame:
    """Maejima light: atribuye press_value al defensor que APLICA la presion.

    Fuente: PFF initialTouch.initialPressurePlayerId — 100% cobertura cuando
    initialPressureType ∈ {A, L, P}. Pesos por type + boost si la presion
    rompe el touch (Lee et al. 2025 exPress-style: press exitosa = touch fallido).

    Aprovecha que PFF ya etiqueta el defensor que aplica la presion frame-level
    (no necesitamos nearest-defender computation, ya esta resuelto por el
    proveedor). Diferencia con score_def_minute: este NO requiere accion
    defensiva SPADL explicita — captura el credito por presion off-ball.
    """
    ev = load_events(match_id)
    it = pl.col("initialTouch").struct
    ge = pl.col("gameEvents").struct
    flat = ev.select([
        ge.field("period").alias("period"),
        ge.field("startGameClock").alias("sgc"),
        it.field("initialPressureType").alias("press_type"),
        it.field("initialPressurePlayerId").cast(pl.Int64).alias("press_player_id"),
        it.field("initialTouchType").alias("touch_type"),
    ]).filter(
        pl.col("press_type").is_in(["A", "L", "P"]) &
        pl.col("press_player_id").is_not_null()
    )
    if flat.height == 0:
        return pl.DataFrame(schema=_PRESS_VALUE_SCHEMA)

    # period_start_sgc: minute_in_period robusto vs convencion PFF
    p_start = (ev.select([
        ge.field("period").alias("period"),
        ge.field("startGameClock").alias("sgc"),
    ]).group_by("period").agg(pl.col("sgc").min().alias("p_start")))

    flat = flat.join(p_start, on="period", how="left").with_columns([
        ((pl.col("sgc") - pl.col("p_start")) // 60).cast(pl.Int64).alias("minute_in_period"),
        pl.col("press_type").replace_strict(_PRESS_TYPE_WEIGHT, default=0.0)
            .cast(pl.Float64).alias("base_w"),
        pl.col("touch_type").is_in(["M", "B"]).cast(pl.Float64).alias("bad_touch"),
    ]).with_columns(
        (pl.col("base_w")
         * pl.when(pl.col("bad_touch") > 0).then(_BAD_TOUCH_BOOST).otherwise(1.0))
            .alias("w")
    )

    return (flat.group_by(["period", "minute_in_period",
                              pl.col("press_player_id").alias("pff_player_id")])
                 .agg(pl.col("w").sum().alias("press_value_minute"))
                 .with_columns(pl.lit(match_id, dtype=pl.Int64).alias("pff_match_id"))
                 .select(list(_PRESS_VALUE_SCHEMA.keys())))


def build_press_value_all(cache: bool = True) -> pl.DataFrame:
    """Agrega press_value para los 64 partidos WC22."""
    cache_path = _DERIVED / "press_value.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)
    dfs = []
    for mid in list_event_match_ids():
        try:
            dfs.append(_pff_press_value_per_minute(mid))
        except Exception as e:
            print(f"  skip {mid}: {e}")
    out = pl.concat(dfs) if dfs else pl.DataFrame(schema=_PRESS_VALUE_SCHEMA)
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(cache_path, compression="snappy")
    return out


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
        (pl.col("time_seconds") // 60).cast(pl.Int64).alias("minute_in_period"),
        ((pl.col("period_id") - 1) * 45 * 60
         + pl.col("time_seconds")).cast(pl.Int64).alias("sec_abs"),
        pl.col("type_name").is_in(list(_DEF_ACTION_TYPES)).alias("is_def_action"),
    ]).filter(pl.col("player_id").is_not_null())

    # vdep_contrib: defensive_value condicionado a acciones defensivas. Reusa
    # la cabeza P(concedes) de M08 atomic-VAEP — NO es VDEP de Toda 2022 stricto
    # (que entrena cabeza P(recovery) − C·P(attacked) separada). Equivalente
    # bajo mismo horizonte y corpus; renombrar como "vdep_like" mas honesto.
    df = df.with_columns(
        pl.when(pl.col("is_def_action")).then(pl.col("defensive_value"))
          .otherwise(0.0).alias("vdep_contrib")
    )

    agg = df.group_by(
        ["game_id", "period_id", "player_id", "minute_in_period"],
    ).agg([
        pl.col("defensive_value").sum().alias("score_def_minute"),
        pl.col("vdep_contrib").sum().alias("vdep_like_minute"),
        pl.col("sec_abs").min().alias("sec_abs"),
        pl.col("is_def_action").sum().cast(pl.Int64).alias("n_def_actions"),
        pl.len().cast(pl.Int64).alias("n_actions_total"),
    ]).rename({
        "game_id":   "sb_match_id",
        "player_id": "sb_player_id",
        "period_id": "period",
    })

    # Mapping ids via APIs publicas (X1+X2)
    sb2pff = sb_to_pff_match_id()
    player_map = atk.build_sb_to_pff_player_map(cache=True).select([
        pl.col("sb_player_id").cast(pl.Int64),
        pl.col("pff_player_id").cast(pl.Int64, strict=False),
    ])

    agg = agg.with_columns([
        pl.col("sb_match_id").cast(pl.Int64),
        pl.col("sb_player_id").cast(pl.Int64),
        pl.col("sb_match_id").replace_strict(sb2pff, default=None).alias("pff_match_id"),
    ]).join(player_map, on="sb_player_id", how="left")

    # Maejima light: press_value via PFF initialPressurePlayerId (peso fijo)
    press_v = build_press_value_all(cache=True)
    if press_v.height > 0:
        agg = agg.join(
            press_v, on=["pff_match_id", "pff_player_id",
                          "period", "minute_in_period"],
            how="left",
        ).with_columns(pl.col("press_value_minute").fill_null(0.0))
    else:
        agg = agg.with_columns(pl.lit(0.0).alias("press_value_minute"))

    # exPress (Lee et al. 2025): cabeza calibrada P(recovery<5s|press_event)
    xpress_path = _DERIVED / "xpress" / "per_minute.parquet"
    if xpress_path.exists():
        xp = pl.read_parquet(xpress_path).select([
            "pff_match_id", "pff_player_id", "period", "minute_in_period",
            "xpress_value_minute",
        ])
        agg = agg.join(
            xp, on=["pff_match_id", "pff_player_id",
                     "period", "minute_in_period"],
            how="left",
        ).with_columns(pl.col("xpress_value_minute").fill_null(0.0))
    else:
        agg = agg.with_columns(pl.lit(0.0).alias("xpress_value_minute"))

    def_ctx = build_def_third_all(cache=True)
    if def_ctx.height > 0:
        # def_ctx publica `minute` period-relative -> renombrar a minute_in_period
        # para alinear con el schema X3 estandarizado.
        def_ctx_cast = def_ctx.with_columns([
            pl.col("pff_match_id").cast(pl.Int64),
            pl.col("player_id").cast(pl.Int64).alias("pff_player_id"),
            pl.col("minute").cast(pl.Int64).alias("minute_in_period"),
        ]).select(["pff_match_id", "pff_player_id", "minute_in_period",
                   "def_third_pct", "press_intensity_frames",
                   "oppo_possession_frames"])
        agg = agg.join(def_ctx_cast,
                        on=["pff_match_id", "pff_player_id", "minute_in_period"],
                        how="left")
    else:
        agg = agg.with_columns([
            pl.lit(None, dtype=pl.Float64).alias("def_third_pct"),
            pl.lit(None, dtype=pl.Int64).alias("press_intensity_frames"),
            pl.lit(None, dtype=pl.Int64).alias("oppo_possession_frames"),
        ])

    # Canal defensa SOTA v2: vdep_like (defensive_value sobre acciones defensivas
    # SPADL — Toda 2022 -like) + xpress_value (Lee 2025 P(recovery<5s|press)
    # calibrado). Captura tanto pressing alto como bloque bajo, alineado con
    # la propuesta §Solidez Defensiva. score_def_minute (legacy: defensive_value
    # sobre TODAS las acciones) preservado como sensitivity.
    agg = agg.with_columns(
        (pl.col("vdep_like_minute") + pl.col("xpress_value_minute"))
            .alias("score_def_v2_minute")
    )

    # Schema canonico: ids -> tiempo -> metricas -> contexto off-ball
    agg = agg.select([
        "pff_match_id", "sb_match_id",
        "pff_player_id", "sb_player_id",
        "period", "minute_in_period", "sec_abs",
        "score_def_v2_minute",                  # OUTCOME PRINCIPAL canal defensa
        "score_def_minute", "vdep_like_minute",  # legacy + componente
        "press_value_minute", "xpress_value_minute",  # sensitivity
        "n_def_actions", "n_actions_total",
        "def_third_pct", "press_intensity_frames", "oppo_possession_frames",
    ])

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        agg.write_parquet(cache_path, compression="snappy")
    return agg


# ===========================================================================
#  SECCION 2 — Aggregation per shock window
# ===========================================================================

def aggregate_per_shock_window(cache: bool = True) -> pl.DataFrame:
    """Por cada (shock, player), suma score_def y n_def_actions en pre/post.

    Schema (X3): pff_match_id + sb_match_id + pff_player_id + sb_player_id.
    Filtra por sec_abs real (no minute*60 sintetico). vdep_like_minute en
    lugar de vdep_minute para honestidad semantica (no es VDEP de Toda 2022
    stricto, ver docstring de modulo).
    """
    cache_path = _DERIVED / "per_shock_window.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    per_min = aggregate_per_player_minute(cache=True).rename(
        {"pff_match_id": "match_id"}
    ).filter(pl.col("pff_player_id").is_not_null())

    shocks = build_shocks_table(cache=True, overwrite=False)
    shocks_slim = shocks.select([
        "match_id", "shock_id", "player_id", "shock_type",
        pl.col("period").alias("shock_period"),
        "window_pre_start", "window_pre_end",
        "window_post_start", "window_post_end",
    ]).rename({"player_id": "pff_player_id"})

    joined = shocks_slim.join(
        per_min, on=["match_id", "pff_player_id"], how="left",
    )

    # period == shock_period evita contaminacion cross-period (PFF sgc usa
    # convencion period-displayed-clock, ~8% de eventos colisionarian sin filtro).
    pre = joined.filter(
        (pl.col("sec_abs") >= pl.col("window_pre_start")) &
        (pl.col("sec_abs") < pl.col("window_pre_end")) &
        (pl.col("period") == pl.col("shock_period"))
    ).group_by(["match_id","shock_id","pff_player_id","shock_type"]).agg([
        pl.col("score_def_v2_minute").sum().alias("score_def_v2_pre"),
        pl.col("score_def_minute").sum().alias("score_def_pre"),
        pl.col("vdep_like_minute").sum().alias("vdep_like_pre"),
        pl.col("n_def_actions").sum().cast(pl.Int64).alias("n_def_actions_pre"),
        pl.col("press_intensity_frames").sum().cast(pl.Int64)
            .alias("press_frames_pre"),
    ])
    post = joined.filter(
        (pl.col("sec_abs") >= pl.col("window_post_start")) &
        (pl.col("sec_abs") <= pl.col("window_post_end")) &
        (pl.col("period") == pl.col("shock_period"))
    ).group_by(["match_id","shock_id","pff_player_id","shock_type"]).agg([
        pl.col("score_def_v2_minute").sum().alias("score_def_v2_post"),
        pl.col("score_def_minute").sum().alias("score_def_post"),
        pl.col("vdep_like_minute").sum().alias("vdep_like_post"),
        pl.col("n_def_actions").sum().cast(pl.Int64).alias("n_def_actions_post"),
        pl.col("press_intensity_frames").sum().cast(pl.Int64)
            .alias("press_frames_post"),
    ])

    base = shocks.select([
        "match_id", "shock_id", "player_id", "shock_type"
    ]).rename({"player_id": "pff_player_id"}).unique()

    pff_to_sb_pl = atk.build_sb_to_pff_player_map(cache=True).select([
        pl.col("pff_player_id").cast(pl.Int64, strict=False),
        pl.col("sb_player_id").cast(pl.Int64),
    ]).filter(pl.col("pff_player_id").is_not_null()).unique(
        subset=["pff_player_id"], keep="first",
    )
    pff2sb_match = pff_to_sb_match_id()

    out = (
        base
        .join(pre,  on=["match_id","shock_id","pff_player_id","shock_type"],
              how="left")
        .join(post, on=["match_id","shock_id","pff_player_id","shock_type"],
              how="left")
        .with_columns([
            pl.col("score_def_v2_pre").fill_null(0.0),
            pl.col("score_def_v2_post").fill_null(0.0),
            pl.col("score_def_pre").fill_null(0.0),
            pl.col("score_def_post").fill_null(0.0),
            pl.col("vdep_like_pre").fill_null(0.0),
            pl.col("vdep_like_post").fill_null(0.0),
            pl.col("n_def_actions_pre").fill_null(0),
            pl.col("n_def_actions_post").fill_null(0),
            pl.col("press_frames_pre").fill_null(0),
            pl.col("press_frames_post").fill_null(0),
        ])
    )

    pm_for_loo = aggregate_per_player_minute(cache=True).filter(
        pl.col("pff_match_id").is_not_null()
        & pl.col("pff_player_id").is_not_null()
    )

    # LOO outcome principal (canal defensa v2 SOTA: vdep_like + xpress_value)
    loo_v2 = attach_team_loo(
        pm_for_loo, value_col="score_def_v2_minute",
    ).rename({
        "score_def_v2_minute_team_loo_pre":  "score_def_v2_team_loo_pre",
        "score_def_v2_minute_team_loo_post": "score_def_v2_team_loo_post",
        "score_def_v2_minute_relative_pre":  "score_def_v2_relative_pre",
        "score_def_v2_minute_relative_post": "score_def_v2_relative_post",
        "score_def_v2_minute_delta_player":  "score_def_v2_delta_player",
        "score_def_v2_minute_delta_team_loo":"score_def_v2_delta_team_loo",
        "score_def_v2_minute_delta_relative":"score_def_v2_delta_relative",
    }).select([
        "match_id", "shock_id", "pff_player_id", "shock_type",
        "score_def_v2_team_loo_pre", "score_def_v2_team_loo_post",
        "score_def_v2_relative_pre", "score_def_v2_relative_post",
        "score_def_v2_delta_player", "score_def_v2_delta_team_loo",
        "score_def_v2_delta_relative", "n_block",
    ])

    # LOO legacy score_def_minute (sensitivity)
    loo = attach_team_loo(
        pm_for_loo, value_col="score_def_minute",
    ).rename({
        "score_def_minute_team_loo_pre":  "score_def_team_loo_pre",
        "score_def_minute_team_loo_post": "score_def_team_loo_post",
        "score_def_minute_relative_pre":  "score_def_relative_pre",
        "score_def_minute_relative_post": "score_def_relative_post",
        "score_def_minute_delta_player":  "score_def_delta_player",
        "score_def_minute_delta_team_loo":"score_def_delta_team_loo",
        "score_def_minute_delta_relative":"score_def_delta_relative",
    }).select([
        "match_id", "shock_id", "pff_player_id", "shock_type",
        "score_def_team_loo_pre", "score_def_team_loo_post",
        "score_def_relative_pre", "score_def_relative_post",
        "score_def_delta_player", "score_def_delta_team_loo",
        "score_def_delta_relative",
    ])

    # LOO sobre press_value_minute (Maejima light, sensitivity para M14)
    loo_press = attach_team_loo(
        pm_for_loo, value_col="press_value_minute",
    ).rename({
        "press_value_minute_team_loo_pre":  "press_value_team_loo_pre",
        "press_value_minute_team_loo_post": "press_value_team_loo_post",
        "press_value_minute_delta_player":  "press_value_delta_player",
        "press_value_minute_delta_team_loo":"press_value_delta_team_loo",
        "press_value_minute_delta_relative":"press_value_delta_relative",
    }).select([
        "match_id", "shock_id", "pff_player_id", "shock_type",
        "press_value_team_loo_pre", "press_value_team_loo_post",
        "press_value_delta_player", "press_value_delta_team_loo",
        "press_value_delta_relative",
    ])

    out = (
        out
        .join(loo_v2, on=["match_id","shock_id","pff_player_id","shock_type"],
              how="left")
        .join(loo, on=["match_id","shock_id","pff_player_id","shock_type"],
              how="left")
        .join(loo_press, on=["match_id","shock_id","pff_player_id","shock_type"],
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
            # Outcome principal canal defensa SOTA v2 (vdep_like + xpress_value)
            "score_def_v2_pre", "score_def_v2_post",
            "score_def_v2_team_loo_pre", "score_def_v2_team_loo_post",
            "score_def_v2_relative_pre", "score_def_v2_relative_post",
            "score_def_v2_delta_player", "score_def_v2_delta_team_loo",
            "score_def_v2_delta_relative",
            # Legacy score_def_minute (sensitivity)
            "score_def_pre", "score_def_post",
            "score_def_team_loo_pre", "score_def_team_loo_post",
            "score_def_relative_pre", "score_def_relative_post",
            "score_def_delta_player", "score_def_delta_team_loo",
            "score_def_delta_relative",
            # Maejima light press_value (sensitivity)
            "press_value_team_loo_pre", "press_value_team_loo_post",
            "press_value_delta_player", "press_value_delta_team_loo",
            "press_value_delta_relative",
            "vdep_like_pre", "vdep_like_post",
            "n_def_actions_pre", "n_def_actions_post",
            "press_frames_pre", "press_frames_post",
            "n_block",
        ])
    )

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

    # Acceptance: distribucion por rol (CBs + DMs > CFs).
    # per_min ya trae pff_player_id (schema X3 estandarizado).
    print("\n[2] Acceptance — score_def por rol (CBs/DMs > CFs):")
    ro = load_rosters().select(["player_id","position_group"]) \
                       .unique(subset=["player_id"]) \
                       .rename({"player_id": "pff_player_id"})
    pm_roles = per_min.filter(pl.col("pff_player_id").is_not_null()) \
                       .join(ro, on="pff_player_id", how="left")
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
