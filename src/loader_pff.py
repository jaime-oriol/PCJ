"""
loader_pff - M01. API de lectura de los parquets PFF.

I/O puro + normalizacion de tipos + vistas derivadas (goles, disparos, subs).
Todo lo que M02-M16 necesitan para tocar PFF sin parsear JSON ni lidiar con
inconsistencias del proveedor.

Filosofia:
  - events y catalogos: eager (baratos, 69 MB total).
  - tracking: SIEMPRE lazy (3.8 GB, 8.7M frames). Un partido tiene 150-200k
    frames; concatenar los 47 revienta RAM.
  - Vistas derivadas (list_goals, list_shots, list_subs) filtran y aplanan
    structs — NO transforman. Score state, direccion de ataque y minutos
    jugados van en M03 preprocess.

Normalizaciones obligatorias (PFF es inconsistente entre ficheros):
  - match_id : Int64 en todas partes (metadata.id viene String, tracking.gameRefId
               viene Float64, events.gameId ya es Int64).
  - player_id / team_id : Int64 (rosters los trae String dentro de struct).
  - shirt_number : Int64 (rosters lo trae String).

Uso rapido:
    from src.loader_pff import (
        list_matches, load_events, scan_tracking,
        list_goals, list_shots, list_subs,
    )

    inv    = list_matches()           # 64 filas con has_tracking
    ev     = load_events(10502)       # eventos 1 partido, structs intactas
    tr     = scan_tracking(10502)     # LazyFrame de tracking
    goals  = list_goals()             # todos los goles del torneo
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl


# -- Rutas ------------------------------------------------------------------

_REPO         = Path(__file__).resolve().parents[1]
_PARQUET      = _REPO / "data" / "parquet"
_PFF          = _PARQUET / "pff"
_EVENTS_DIR   = _PFF / "events"
_TRACKING_DIR = _PFF / "tracking"
_METADATA     = _PFF / "metadata.parquet"
_ROSTERS      = _PFF / "rosters.parquet"


# -- Descubrimiento ---------------------------------------------------------

def list_event_match_ids() -> list[int]:
    """IDs de los 64 partidos con event data."""
    return sorted(int(f.stem) for f in _EVENTS_DIR.glob("*.parquet"))


def list_tracking_match_ids() -> list[int]:
    """IDs de los 47 partidos con tracking data."""
    return sorted(int(f.stem) for f in _TRACKING_DIR.glob("*.parquet"))


def list_matches() -> pl.DataFrame:
    """Inventario: match_id, date, week, equipos, pitch, fps, has_tracking."""
    md = load_metadata()
    tr_ids = set(list_tracking_match_ids())
    return md.select([
        "match_id", "date", "week",
        "home_team_id", "home_team_name",
        "away_team_id", "away_team_name",
        "pitch_length", "pitch_width",
        "fps", "home_team_start_left", "home_team_start_left_et",
        pl.col("match_id").is_in(list(tr_ids)).alias("has_tracking"),
    ]).sort("match_id")


# -- Raw: metadata / rosters ------------------------------------------------

def load_metadata(match_id: int | None = None) -> pl.DataFrame:
    """Metadata con match_id y team ids en Int64, pitch y flags de direccion planos.

    Campos derivados sobre el raw:
      match_id, home_team_id, home_team_name, away_team_id, away_team_name,
      pitch_length, pitch_width, home_team_start_left, home_team_start_left_et.
    Resto de cols raw se preservan (startPeriod1, endPeriod1, videoUrl, etc.).
    """
    df = pl.read_parquet(_METADATA)
    df = df.with_columns([
        pl.col("id").cast(pl.Int64).alias("match_id"),
        pl.col("homeTeam").struct.field("id").cast(pl.Int64).alias("home_team_id"),
        pl.col("homeTeam").struct.field("name").alias("home_team_name"),
        pl.col("awayTeam").struct.field("id").cast(pl.Int64).alias("away_team_id"),
        pl.col("awayTeam").struct.field("name").alias("away_team_name"),
        pl.col("stadium").struct.field("pitches").list.first().struct.field("length").alias("pitch_length"),
        pl.col("stadium").struct.field("pitches").list.first().struct.field("width").alias("pitch_width"),
        pl.col("homeTeamStartLeft").alias("home_team_start_left"),
        pl.col("homeTeamStartLeftExtraTime").alias("home_team_start_left_et"),
    ])
    if match_id is not None:
        df = df.filter(pl.col("match_id") == match_id)
    return df


def load_rosters(match_id: int | None = None) -> pl.DataFrame:
    """Rosters con player/team unnested; player_id, team_id, shirt_number en Int64."""
    df = pl.read_parquet(_ROSTERS)
    df = df.with_columns([
        pl.col("player").struct.field("id").cast(pl.Int64).alias("player_id"),
        pl.col("player").struct.field("nickname").alias("player_name"),
        pl.col("team").struct.field("id").cast(pl.Int64).alias("team_id"),
        pl.col("team").struct.field("name").alias("team_name"),
        pl.col("shirtNumber").cast(pl.Int64, strict=False).alias("shirt_number"),
        pl.col("positionGroupType").alias("position_group"),
    ]).select([
        "match_id", "team_id", "team_name",
        "player_id", "player_name", "position_group", "shirt_number", "started",
    ])
    if match_id is not None:
        df = df.filter(pl.col("match_id") == match_id)
    return df.sort(["match_id", "team_id", "shirt_number"])


# -- Raw: events ------------------------------------------------------------

def load_events(match_id: int) -> pl.DataFrame:
    """Events de 1 partido. Structs (gameEvents, possessionEvents, grades, ...) preservadas."""
    path = _EVENTS_DIR / f"{match_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"sin events para match_id={match_id}")
    return pl.read_parquet(path)


def scan_events(match_ids: Iterable[int] | None = None) -> pl.LazyFrame:
    """LazyFrame multi-partido; diagonal_relaxed tolera cols opcionales entre partidos."""
    ids = list(match_ids) if match_ids else list_event_match_ids()
    files = [_EVENTS_DIR / f"{mid}.parquet" for mid in ids]
    if not files:
        raise FileNotFoundError("no hay events en disco")
    return pl.concat([pl.scan_parquet(f) for f in files], how="diagonal_relaxed")


# -- Raw: tracking ----------------------------------------------------------

def scan_tracking(match_id: int) -> pl.LazyFrame:
    """LazyFrame de tracking de 1 partido. NO concatenar varios (3.8 GB totales).

    Incluye match_id como Int64 (el raw trae gameRefId Float64).
    """
    path = _TRACKING_DIR / f"{match_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"sin tracking para match_id={match_id}")
    return pl.scan_parquet(path).with_columns(
        pl.col("gameRefId").cast(pl.Int64).alias("match_id"),
    )


# -- Vistas derivadas -------------------------------------------------------

def _flatten_events(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Expone cols planas de gameEvents + possessionEvents necesarias en las vistas."""
    return lf.with_columns([
        pl.col("gameId").cast(pl.Int64).alias("match_id"),
        pl.col("gameEvents").struct.field("gameEventType").alias("game_event_type"),
        pl.col("gameEvents").struct.field("period").alias("period"),
        pl.col("gameEvents").struct.field("startGameClock").alias("start_game_clock"),
        pl.col("gameEvents").struct.field("teamId").alias("team_id"),
        pl.col("gameEvents").struct.field("teamName").alias("team_name"),
        pl.col("gameEvents").struct.field("homeTeam").alias("home_team_in_possession"),
        pl.col("gameEvents").struct.field("playerId").alias("player_id"),
        pl.col("gameEvents").struct.field("playerName").alias("player_name"),
        pl.col("possessionEvents").struct.field("possessionEventType").alias("possession_event_type"),
        pl.col("possessionEvents").struct.field("gameClock").alias("game_clock"),
    ]).with_columns(
        (pl.col("start_game_clock") // 60).alias("minute"),
    )


def list_goals(match_id: int | None = None) -> pl.DataFrame:
    """Goles (shotOutcomeType == 'G') del torneo o un partido.

    Captura goles deliberados E involuntarios (pases/despejes/rebotes que
    acaban en gol tambien rellenan los campos del shot, ver EVENT_DATA_SPEC §4.2).
    NO filtra nada — expone los crudos con flags para que M03/M07 decida:
      - disallowed : True si el evento fue anulado (VAR).
      - shootout   : True si es penal de tanda (period==4 & start_game_clock>7200).
      - setpiece_type : 'P' para penaltis (tanda o en juego).

    Team es el que REMATA. Para is_own_goal comparar con equipo del keeper (M03).
    """
    ids = [match_id] if match_id is not None else list_event_match_ids()
    lf = _flatten_events(scan_events(ids)).with_columns([
        pl.col("possessionEvents").struct.field("shotOutcomeType").alias("shot_outcome"),
        pl.col("possessionEvents").struct.field("shooterPlayerId").alias("shooter_id"),
        pl.col("possessionEvents").struct.field("shooterPlayerName").alias("shooter_name"),
        pl.col("possessionEvents").struct.field("keeperPlayerId").alias("keeper_id"),
        pl.col("possessionEvents").struct.field("keeperPlayerName").alias("keeper_name"),
        pl.col("possessionEvents").struct.field("bodyType").alias("body_part"),
        pl.col("gameEvents").struct.field("setpieceType").alias("setpiece_type"),
        pl.col("gameEvents").struct.field("initialNonEvent").alias("disallowed"),
    ]).with_columns(
        ((pl.col("period") == 4) & (pl.col("start_game_clock") > 7200)).alias("shootout"),
    )
    return lf.filter(pl.col("shot_outcome") == "G").select([
        "match_id", "period", "start_game_clock", "minute",
        "team_id", "team_name",
        "shooter_id", "shooter_name",
        "keeper_id", "keeper_name",
        "body_part", "setpiece_type",
        "possession_event_type",
        "disallowed", "shootout",
    ]).sort(["match_id", "start_game_clock"]).collect()


def list_shots(match_id: int | None = None) -> pl.DataFrame:
    """Disparos deliberados (possession_event_type == 'SH').

    Incluye shotOutcomeType, shotType, naturaleza, cuerpo, keeper, flags
    saveable/badParry. Base para M05 PSxG y M06 near-miss.
    """
    ids = [match_id] if match_id is not None else list_event_match_ids()
    lf = _flatten_events(scan_events(ids)).with_columns([
        pl.col("possessionEvents").struct.field("shooterPlayerId").alias("shooter_id"),
        pl.col("possessionEvents").struct.field("shooterPlayerName").alias("shooter_name"),
        pl.col("possessionEvents").struct.field("shotType").alias("shot_type"),
        pl.col("possessionEvents").struct.field("shotNatureType").alias("shot_nature"),
        pl.col("possessionEvents").struct.field("shotInitialHeightType").alias("shot_height"),
        pl.col("possessionEvents").struct.field("shotOutcomeType").alias("shot_outcome"),
        pl.col("possessionEvents").struct.field("bodyMovementType").alias("body_movement"),
        pl.col("possessionEvents").struct.field("ballMoving").alias("ball_moving"),
        pl.col("possessionEvents").struct.field("bodyType").alias("body_part"),
        pl.col("possessionEvents").struct.field("keeperPlayerId").alias("keeper_id"),
        pl.col("possessionEvents").struct.field("saveable").alias("saveable"),
        pl.col("possessionEvents").struct.field("badParry").alias("bad_parry"),
        pl.col("gameEvents").struct.field("setpieceType").alias("setpiece_type"),
    ])
    return lf.filter(pl.col("possession_event_type") == "SH").select([
        "match_id", "period", "start_game_clock", "minute",
        "team_id", "team_name",
        "shooter_id", "shooter_name",
        "shot_type", "shot_nature", "shot_height", "shot_outcome",
        "body_part", "body_movement", "ball_moving",
        "keeper_id", "saveable", "bad_parry",
        "setpiece_type",
    ]).sort(["match_id", "start_game_clock"]).collect()


def list_subs(match_id: int | None = None) -> pl.DataFrame:
    """Sustituciones (gameEventType == 'SUB'). Base para M03 minutos jugados."""
    ids = [match_id] if match_id is not None else list_event_match_ids()
    lf = scan_events(ids)
    ge = pl.col("gameEvents").struct
    return lf.filter(ge.field("gameEventType") == "SUB").select([
        pl.col("gameId").cast(pl.Int64).alias("match_id"),
        ge.field("period").alias("period"),
        ge.field("startGameClock").alias("start_game_clock"),
        (ge.field("startGameClock") // 60).alias("minute"),
        ge.field("teamId").alias("team_id"),
        ge.field("teamName").alias("team_name"),
        ge.field("playerOffId").alias("player_off_id"),
        ge.field("playerOffName").alias("player_off_name"),
        ge.field("playerOnId").alias("player_on_id"),
        ge.field("playerOnName").alias("player_on_name"),
        ge.field("subType").alias("sub_type"),
    ]).sort(["match_id", "start_game_clock"]).collect()


# -- Sanity inline ----------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=== loader_pff sanity ===")

    t0 = time.time()
    inv = list_matches()
    n_tr = int(inv["has_tracking"].sum())
    print(f"inventario: {inv.height} partidos ({n_tr} con tracking) "
          f"en {time.time()-t0:.2f}s")

    t0 = time.time()
    ro = load_rosters()
    print(f"rosters   : {ro.height:,} filas, {ro['player_id'].n_unique()} "
          f"jugadores unicos, {ro['team_id'].n_unique()} equipos "
          f"en {time.time()-t0:.2f}s")

    gid = int(inv.filter(pl.col("has_tracking"))["match_id"][0])
    t0 = time.time()
    ev = load_events(gid)
    n_frames = scan_tracking(gid).select(pl.len()).collect().item()
    print(f"partido {gid}: {ev.height:,} events, {n_frames:,} frames "
          f"en {time.time()-t0:.2f}s")

    t0 = time.time()
    goals = list_goals()
    shots = list_shots()
    subs = list_subs()
    n_valid = goals.filter(~pl.col("disallowed") & ~pl.col("shootout")).height
    n_shootout = int(goals["shootout"].sum())
    n_disallowed = int(goals["disallowed"].sum())
    print(f"torneo    : {goals.height} goles brutos "
          f"({n_valid} validos + {n_shootout} tanda + {n_disallowed} anulados), "
          f"{shots.height:,} shots, {subs.height} subs "
          f"en {time.time()-t0:.1f}s")

    # Sanity dobles: jugadores unicos en goals y shots deben ser subset de rosters
    gs_pids = set(goals["shooter_id"].drop_nulls().to_list())
    sh_pids = set(shots["shooter_id"].drop_nulls().to_list())
    ro_pids = set(ro["player_id"].to_list())
    print(f"  shooters en goles subset de rosters: "
          f"{gs_pids.issubset(ro_pids)} ({len(gs_pids - ro_pids)} huerfanos)")
    print(f"  shooters en shots subset de rosters: "
          f"{sh_pids.issubset(ro_pids)} ({len(sh_pids - ro_pids)} huerfanos)")
