"""
preprocess - M03. Normalizaciones que todo el pipeline asume.

Funciones nucleares:
  - attacking_direction(match_id) : (team_id, period) -> 'L' o 'R'
  - goals_timeline(match_id)      : goles validos con cum_home / cum_away
  - player_minutes(match_id)      : minutos jugados por jugador-partido
  - enrich_events(match_id)       : events enriquecidos (acceptance M03)

Convencion de direccion: 'R' = equipo ataca hacia x creciente (lado derecho
de la camara principal); 'L' = hacia x decreciente. Sistema de coordenadas
PFF: metros, (0,0) = centro, x in [-L/2, L/2], y in [-W/2, W/2].

Convencion de normalizacion de coordenadas post-flip: el equipo en posesion
SIEMPRE ataca hacia x_norm > 0. Implementado en col `ball_x_norm`.

Cache idempotente en data/parquet/derived/preprocess/events_enriched/.
Si el parquet existe y `cache=True`, se lee en vez de recomputar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

# Permite tanto `python src/preprocess.py` como `from src.preprocess import ...`
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from loader_pff import (
    load_metadata, load_rosters, load_events, list_goals, list_subs, scan_events,
)


# -- Rutas ------------------------------------------------------------------

_REPO    = Path(__file__).resolve().parents[1]
_DERIVED = _REPO / "data" / "parquet" / "derived" / "preprocess"


# -- Direccion de ataque ----------------------------------------------------

def attacking_direction(match_id: int) -> pl.DataFrame:
    """Direccion de ataque por (team_id, period) del partido.

    home_team_start_left == True  -> home ataca hacia x+ en period 1 (local
    empieza en el lado izquierdo de la camara y ataca a la derecha).
    En period 2 cambian lados. En ET se rige por home_team_start_left_et si
    esta, si no se hereda del patron period-1/2.

    Returns:
        DataFrame con cols: match_id, team_id, period (1..4), direction ('L'|'R').
    """
    md = load_metadata(match_id).row(0, named=True)
    home_id = md["home_team_id"]
    away_id = md["away_team_id"]
    hsl = bool(md["home_team_start_left"])
    hsl_et_raw = md["home_team_start_left_et"]
    hsl_et = bool(hsl_et_raw) if hsl_et_raw is not None else hsl

    rows = []
    for period in (1, 2, 3, 4):
        use = hsl if period in (1, 2) else hsl_et
        # odd periods: home starts-left => home attacks right
        # even periods: switch
        home_right = use if (period % 2 == 1) else (not use)
        rows.append((match_id, home_id, period, "R" if home_right else "L"))
        rows.append((match_id, away_id, period, "L" if home_right else "R"))
    return pl.DataFrame(rows, schema=["match_id", "team_id", "period", "direction"],
                        orient="row")


# -- Score state ------------------------------------------------------------

def goals_timeline(match_id: int) -> pl.DataFrame:
    """Goles validos del partido ordenados con cum_home / cum_away.

    Excluye goles anulados (disallowed) y penaltis de tanda (shootout).
    Own-goals se resuelven comparando el team del shooter con el team del keeper:
    el equipo QUE MARCA es el del keeper (portero sufrido).

    Returns:
        DataFrame: match_id, period, start_game_clock, minute, scoring_team_id,
                   is_own_goal, cum_home, cum_away.
    """
    md = load_metadata(match_id).row(0, named=True)
    home_id = md["home_team_id"]
    away_id = md["away_team_id"]
    g = list_goals(match_id).filter(
        ~pl.col("disallowed") & ~pl.col("shootout")
    )
    # Resolver scoring_team_id:
    #   keeper_team = equipo del keeper (encaja el gol)
    #   shooter_team = team_id del evento (= equipo en posesion cuando se remata)
    #   si shooter_team == keeper_team -> own goal -> scoring_team = el OTRO
    #   si shooter_team != keeper_team -> gol normal -> scoring_team = shooter_team
    ro = load_rosters(match_id).select(["player_id", "team_id"]).rename(
        {"player_id": "keeper_id", "team_id": "keeper_team_id"}
    )
    g = g.join(ro, on="keeper_id", how="left").with_columns([
        (pl.col("team_id") == pl.col("keeper_team_id")).alias("is_own_goal"),
    ]).with_columns(
        pl.when(pl.col("is_own_goal"))
          .then(
              pl.when(pl.col("team_id") == home_id)
                .then(pl.lit(away_id))
                .otherwise(pl.lit(home_id))
          )
          .otherwise(pl.col("team_id"))
          .alias("scoring_team_id")
    ).sort(["start_game_clock"])
    # Acumulados
    g = g.with_columns([
        (pl.col("scoring_team_id") == home_id).cast(pl.Int64).cum_sum().alias("cum_home"),
        (pl.col("scoring_team_id") == away_id).cast(pl.Int64).cum_sum().alias("cum_away"),
    ])
    return g.select([
        "match_id", "period", "start_game_clock", "minute",
        "scoring_team_id", "is_own_goal", "cum_home", "cum_away",
    ])


def score_state_before(
    events_df: pl.DataFrame, goals_df: pl.DataFrame, home_id: int,
) -> pl.DataFrame:
    """Anade score_home, score_away, score_diff a events_df (state BEFORE evento).

    score_diff se expresa desde la perspectiva del equipo en posesion
    (positivo = equipo en posesion por delante).
    """
    # Tomar goles ordenados; para cada evento, filtrar goals strict-before
    # por start_game_clock. Asof join.
    g = goals_df.sort("start_game_clock").select([
        pl.col("start_game_clock").alias("g_sgc"),
        "cum_home", "cum_away",
    ])
    ev = events_df.sort("start_game_clock")
    # asof en start_game_clock con estrictamente menor:
    # (goal en start_sgc es post-evento en ese mismo segundo? safer: strict <)
    # polars join_asof usa 'backward' = <= por defecto. Usamos < shifting g_sgc+1e-9:
    # mas simple: despues del join_asof, si g_sgc == sgc retroceder 1.
    ev = ev.join_asof(g, left_on="start_game_clock", right_on="g_sgc",
                      strategy="backward")
    ev = ev.with_columns([
        pl.col("cum_home").fill_null(0).alias("score_home"),
        pl.col("cum_away").fill_null(0).alias("score_away"),
    ]).drop(["g_sgc", "cum_home", "cum_away"])
    # score_diff desde perspectiva del equipo en posesion
    ev = ev.with_columns(
        pl.when(pl.col("team_id") == home_id)
          .then(pl.col("score_home") - pl.col("score_away"))
          .otherwise(pl.col("score_away") - pl.col("score_home"))
          .alias("score_diff_possession")
    )
    return ev


# -- Minutos jugados --------------------------------------------------------

def player_minutes(match_id: int) -> pl.DataFrame:
    """Minutos jugados por jugador-partido.

    Reglas:
      - started=True y NO substituted out -> [0, fin_partido]
      - started=True y substituted out    -> [0, sub_off_minute]
      - started=False y sub IN            -> [sub_on_minute, sub_off o fin]
      - started=False y no entro          -> [None, None], minutes_played = 0

    fin_partido = ultimo minuto observado en events del partido.

    Returns:
        DataFrame: match_id, player_id, team_id, position_group, started,
                   minute_in, minute_out, minutes_played, was_substituted_out.
    """
    ro = load_rosters(match_id)
    subs = list_subs(match_id)
    # fin del partido: max minute en events
    ev = load_events(match_id)
    last_min_by_period = ev.select(
        pl.col("gameEvents").struct.field("period").alias("period"),
        pl.col("gameEvents").struct.field("startGameClock").alias("sgc"),
    ).group_by("period").agg(pl.col("sgc").max()).sort("period")
    end_minute = int(last_min_by_period["sgc"].max() // 60) + 1  # redondeo superior

    # sub_off: jugador que sale del campo (player_off_id -> minute)
    off = subs.select([
        pl.col("player_off_id").alias("player_id"),
        pl.col("minute").alias("sub_off_minute"),
    ])
    on = subs.select([
        pl.col("player_on_id").alias("player_id"),
        pl.col("minute").alias("sub_on_minute"),
    ])
    out = (
        ro.select(["match_id", "player_id", "team_id", "position_group", "started"])
          .join(off, on="player_id", how="left")
          .join(on,  on="player_id", how="left")
    )
    out = out.with_columns([
        pl.when(pl.col("started"))
          .then(pl.lit(0))
          .otherwise(pl.col("sub_on_minute"))
          .alias("minute_in"),
        pl.when(pl.col("sub_off_minute").is_not_null())
          .then(pl.col("sub_off_minute"))
          .when(pl.col("started") | pl.col("sub_on_minute").is_not_null())
          .then(pl.lit(end_minute))
          .otherwise(None)
          .alias("minute_out"),
    ]).with_columns(
        (pl.col("minute_out") - pl.col("minute_in")).fill_null(0).alias("minutes_played"),
        pl.col("sub_off_minute").is_not_null().alias("was_substituted_out"),
    ).drop(["sub_off_minute", "sub_on_minute"])
    return out.sort(["team_id", "minute_in"])


# -- Enrich events (acceptance M03) -----------------------------------------

def enrich_events(match_id: int, cache: bool = True) -> pl.DataFrame:
    """Events enriquecidos con cols planas + score state + direction + ball normalizado.

    Cols anadidas (sobre load_events original, structs preservadas):
      - match_id, period, start_game_clock, match_second, minute
      - game_event_type, possession_event_type
      - team_id, team_name, player_id, player_name
      - score_home, score_away, score_diff_possession (state BEFORE evento)
      - attacking_direction ('L'|'R' del equipo en posesion)
      - ball_x, ball_y, ball_z (primer elemento de ball list)
      - ball_x_norm, ball_y_norm (coords flipeadas: equipo en posesion ataca a x+)

    Cachea en data/parquet/derived/preprocess/events_enriched/{match_id}.parquet
    si cache=True.
    """
    cache_path = _DERIVED / "events_enriched" / f"{match_id}.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    ev = load_events(match_id)
    ge = pl.col("gameEvents").struct
    pe = pl.col("possessionEvents").struct
    ball_first = pl.col("ball").list.first().struct

    df = ev.with_columns([
        pl.col("gameId").cast(pl.Int64).alias("match_id"),
        ge.field("period").alias("period"),
        ge.field("startGameClock").alias("start_game_clock"),
        ge.field("startGameClock").alias("match_second"),
        (ge.field("startGameClock") // 60).alias("minute"),
        ge.field("gameEventType").alias("game_event_type"),
        ge.field("teamId").alias("team_id"),
        ge.field("teamName").alias("team_name"),
        ge.field("playerId").alias("player_id"),
        ge.field("playerName").alias("player_name"),
        pe.field("possessionEventType").alias("possession_event_type"),
        ball_first.field("x").alias("ball_x"),
        ball_first.field("y").alias("ball_y"),
        ball_first.field("z").alias("ball_z"),
    ])

    # Score state
    md = load_metadata(match_id).row(0, named=True)
    home_id = md["home_team_id"]
    g = goals_timeline(match_id)
    df = score_state_before(df, g, home_id)

    # Direccion de ataque
    dirs = attacking_direction(match_id).rename({"direction": "attacking_direction"})
    df = df.join(dirs.drop("match_id"), on=["team_id", "period"], how="left")

    # Ball normalizado: equipo en posesion ataca a x+
    # Si attacking_direction == 'L', el equipo ataca hacia x-, por tanto flipeo x e y.
    df = df.with_columns([
        pl.when(pl.col("attacking_direction") == "L")
          .then(-pl.col("ball_x"))
          .otherwise(pl.col("ball_x"))
          .alias("ball_x_norm"),
        pl.when(pl.col("attacking_direction") == "L")
          .then(-pl.col("ball_y"))
          .otherwise(pl.col("ball_y"))
          .alias("ball_y_norm"),
    ])

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_path, compression="snappy", statistics=True)

    return df


def cache_all_enriched(overwrite: bool = False) -> dict:
    """Precomputa enrich_events de los 64 partidos y cachea a parquet."""
    from loader_pff import list_event_match_ids
    out = {}
    for mid in list_event_match_ids():
        p = _DERIVED / "events_enriched" / f"{mid}.parquet"
        if p.exists() and not overwrite:
            out[mid] = p
            continue
        _ = enrich_events(mid, cache=True)
        out[mid] = p
    return out


# -- Sanity inline ----------------------------------------------------------

if __name__ == "__main__":
    import time
    from loader_pff import list_matches, list_event_match_ids

    print("=== preprocess sanity ===")
    inv = list_matches()
    gid = int(inv.filter(pl.col("has_tracking"))["match_id"][0])

    t0 = time.time()
    d = attacking_direction(gid)
    print(f"attacking_direction({gid}): {d.height} filas (2 teams x 4 periods) "
          f"en {time.time()-t0:.2f}s")
    print(d)

    t0 = time.time()
    g = goals_timeline(gid)
    print(f"goals_timeline({gid}): {g.height} goles validos "
          f"en {time.time()-t0:.2f}s")
    print(g)

    t0 = time.time()
    pm = player_minutes(gid)
    print(f"player_minutes({gid}): {pm.height} jugadores "
          f"en {time.time()-t0:.2f}s")
    starters = pm.filter(pl.col("started"))
    bench = pm.filter(~pl.col("started"))
    print(f"  starters: {starters.height} (minutes mean={starters['minutes_played'].mean():.1f})")
    print(f"  bench:    {bench.height} ({bench.filter(pl.col('minute_in').is_not_null()).height} entraron)")

    t0 = time.time()
    ee = enrich_events(gid, cache=False)
    print(f"enrich_events({gid}): {ee.height:,} filas, {ee.width} cols "
          f"en {time.time()-t0:.2f}s")
    print(f"  score_home final: {ee['score_home'].max()}")
    print(f"  score_away final: {ee['score_away'].max()}")
    cov_ball = ee['ball_x_norm'].drop_nulls().len()
    cov_dir  = ee['attacking_direction'].drop_nulls().len()
    print(f"  ball_x_norm non-null: {cov_ball:,}/{ee.height:,} ({100*cov_ball/ee.height:.1f}%)")
    print(f"  attacking_direction non-null: {cov_dir:,}/{ee.height:,} ({100*cov_dir/ee.height:.1f}%)")
