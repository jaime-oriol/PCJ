"""
loader_public - M02. API de lectura de los parquets publicos (Wyscout + StatsBomb).

polars nativo, sin socceraction. I/O puro + normalizacion de tipos + helpers
por competicion. Sustituye funcionalmente a `src/loaders.py` (que queda solo
si algun training externo exige la API socceraction).

Wyscout 2017/18 — corpus base de training (Big 5 + WC18 + Euro16):
    7 parquets de events (uno por competicion), 1.941 matches, 3.25M eventos,
    3.603 jugadores, 142 equipos, 208 coaches, 46.897 rankings.

StatsBomb open subset (200 partidos para PSxG + cross-val):
    WC22 (64) + Euro24 (51) + Euro20 (51) + Bundes23/24 (34), 753k eventos,
    653k freeze-frames 360, 400 lineups (2 por partido).

Uso rapido:
    from src.loader_public import (
        scan_wyscout_events, load_wyscout_matches,
        list_statsbomb_competitions, load_statsbomb_events, load_statsbomb_360,
    )

    wc = scan_wyscout_events("World_Cup").collect()     # 101.759 eventos
    cmps = list_statsbomb_competitions()                # 4 torneos con n_matches
    ev = load_statsbomb_events(3857256)                 # WC22 partido
    ff = load_statsbomb_360(3857256)                    # freeze frames
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl


# -- Rutas ------------------------------------------------------------------

_REPO    = Path(__file__).resolve().parents[1]
_PARQUET = _REPO / "data" / "parquet"
_WYSCOUT = _PARQUET / "wyscout"
_SB      = _PARQUET / "statsbomb"


# -- Constantes -------------------------------------------------------------

# Wyscout: 5 ligas + Euro 16 + WC 18. Las 7 unicas competiciones del open dataset.
WYSCOUT_COMPETITIONS: tuple[str, ...] = (
    "England", "France", "Germany", "Italy", "Spain",
    "European_Championship", "World_Cup",
)

# StatsBomb: alias humano -> (competition_id, season_id).
# Las 4 unicas competiciones que sobreviven al filtro de este TFM.
STATSBOMB_COMPETITIONS: dict[str, tuple[int, int]] = {
    "WC22":     (43, 106),
    "Euro20":   (55,  43),
    "Euro24":   (55, 282),
    "Bundes23": (9,  281),
}


# ---------------------------------------------------------------------------
#  WYSCOUT
# ---------------------------------------------------------------------------

def list_wyscout_competitions() -> list[str]:
    """Nombres de las 7 competiciones disponibles."""
    return list(WYSCOUT_COMPETITIONS)


def load_wyscout_matches(competition: str | None = None) -> pl.DataFrame:
    """Partidos Wyscout con col 'competition' normalizada. 1.941 filas totales."""
    df = pl.read_parquet(_WYSCOUT / "matches.parquet")
    if competition is not None:
        _check_wyscout_competition(competition)
        df = df.filter(pl.col("competition") == competition)
    return df


def load_wyscout_players() -> pl.DataFrame:
    """Catalogo de jugadores Wyscout (3.603 filas)."""
    return pl.read_parquet(_WYSCOUT / "players.parquet")


def load_wyscout_teams() -> pl.DataFrame:
    """Catalogo de equipos Wyscout (142 filas)."""
    return pl.read_parquet(_WYSCOUT / "teams.parquet")


def load_wyscout_coaches() -> pl.DataFrame:
    """Catalogo de coaches Wyscout (208 filas)."""
    return pl.read_parquet(_WYSCOUT / "coaches.parquet")


def load_wyscout_playerank() -> pl.DataFrame:
    """Rankings ML PlayerRank de Wyscout (46.897 filas)."""
    return pl.read_parquet(_WYSCOUT / "playerank.parquet")


def scan_wyscout_events(competition: str | None = None) -> pl.LazyFrame:
    """LazyFrame de eventos: 1 competicion o concat de las 7 (3.25M eventos total).

    Schema: eventId, subEventName, tags (List[Struct{id}]), playerId, positions
    (List[Struct{x,y}] en 0-100), matchId, eventName, teamId, matchPeriod,
    eventSec, subEventId, id.
    """
    if competition is not None:
        _check_wyscout_competition(competition)
        return pl.scan_parquet(_WYSCOUT / f"events_{competition}.parquet")
    files = [_WYSCOUT / f"events_{c}.parquet" for c in WYSCOUT_COMPETITIONS]
    return pl.concat([pl.scan_parquet(f) for f in files], how="diagonal_relaxed")


def _check_wyscout_competition(name: str) -> None:
    if name not in WYSCOUT_COMPETITIONS:
        raise ValueError(f"competition '{name}' no existe; usa una de {WYSCOUT_COMPETITIONS}")


# ---------------------------------------------------------------------------
#  STATSBOMB
# ---------------------------------------------------------------------------

def list_statsbomb_competitions() -> pl.DataFrame:
    """Torneos efectivamente disponibles (filtrados), con n_matches por torneo.

    Devuelve los 4: WC22, Euro20, Euro24, Bundes23/24. El catalogo competitions.parquet
    trae los 75 del dump publico; aqui nos limitamos a los que tienen matches locales.
    """
    m = load_statsbomb_matches()
    return (
        m.group_by(["competition_id", "competition_name", "season_id", "season_name"])
         .len().rename({"len": "n_matches"})
         .sort("n_matches", descending=True)
    )


def list_statsbomb_match_ids(
    comp_id: int | None = None, season_id: int | None = None,
) -> list[int]:
    """IDs de partidos; filtra por (comp_id, season_id) si se dan."""
    df = load_statsbomb_matches(comp_id, season_id)
    return sorted(df["match_id"].to_list())


def load_statsbomb_matches(
    comp_id: int | None = None, season_id: int | None = None,
) -> pl.DataFrame:
    """200 partidos con cols planas (competition, season, home_team, away_team desempaquetados)."""
    df = pl.read_parquet(_SB / "matches.parquet").with_columns([
        pl.col("competition").struct.field("competition_id").alias("competition_id"),
        pl.col("competition").struct.field("competition_name").alias("competition_name"),
        pl.col("competition").struct.field("country_name").alias("competition_country"),
        pl.col("season").struct.field("season_id").alias("season_id"),
        pl.col("season").struct.field("season_name").alias("season_name"),
        pl.col("home_team").struct.field("home_team_id").alias("home_team_id"),
        pl.col("home_team").struct.field("home_team_name").alias("home_team_name"),
        pl.col("away_team").struct.field("away_team_id").alias("away_team_id"),
        pl.col("away_team").struct.field("away_team_name").alias("away_team_name"),
        pl.col("competition_stage").struct.field("name").alias("stage"),
    ])
    if comp_id is not None:
        df = df.filter(pl.col("competition_id") == comp_id)
    if season_id is not None:
        df = df.filter(pl.col("season_id") == season_id)
    return df.sort(["competition_id", "season_id", "match_id"])


def load_statsbomb_events(match_id: int) -> pl.DataFrame:
    """Events de 1 partido StatsBomb (structs preservadas: pass, shot, goalkeeper, ...)."""
    path = _SB / f"events/{match_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"sin events StatsBomb para match_id={match_id}")
    return pl.read_parquet(path)


def scan_statsbomb_events(match_ids: Iterable[int] | None = None) -> pl.LazyFrame:
    """LazyFrame multi-partido de events; diagonal_relaxed tolera cols opcionales."""
    ids = list(match_ids) if match_ids else list_statsbomb_match_ids()
    files = [_SB / f"events/{mid}.parquet" for mid in ids]
    if not files:
        raise FileNotFoundError("no hay events StatsBomb en disco")
    return pl.concat([pl.scan_parquet(f) for f in files], how="diagonal_relaxed")


def load_statsbomb_lineups(match_id: int) -> pl.DataFrame:
    """Lineups de 1 partido (2 filas: home + away, con lineup list anidada)."""
    path = _SB / f"lineups/{match_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"sin lineups para match_id={match_id}")
    return pl.read_parquet(path)


def load_statsbomb_360(match_id: int) -> pl.DataFrame:
    """Freeze-frames 360 de 1 partido (event_uuid, visible_area, freeze_frame)."""
    path = _SB / f"freeze_frames/{match_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"sin freeze_frames para match_id={match_id}")
    return pl.read_parquet(path)


# -- Sanity inline ----------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=== loader_public sanity ===")

    # --- Wyscout ---
    t0 = time.time()
    comps_w = list_wyscout_competitions()
    wm = load_wyscout_matches()
    wp = load_wyscout_players()
    wt = load_wyscout_teams()
    wpr = load_wyscout_playerank()
    print(f"wyscout catalogos: {len(comps_w)} comps, {wm.height:,} matches, "
          f"{wp.height:,} players, {wt.height} teams, {wpr.height:,} playerank "
          f"en {time.time()-t0:.2f}s")

    t0 = time.time()
    ev_wc = scan_wyscout_events("World_Cup").collect()
    print(f"wyscout events 'World_Cup': {ev_wc.height:,} filas "
          f"en {time.time()-t0:.2f}s")

    t0 = time.time()
    n_all = scan_wyscout_events().select(pl.len()).collect().item()
    print(f"wyscout events totales (lazy concat): {n_all:,} "
          f"en {time.time()-t0:.2f}s")

    # --- StatsBomb ---
    t0 = time.time()
    cmps = list_statsbomb_competitions()
    mids = list_statsbomb_match_ids()
    print(f"statsbomb: {cmps.height} torneos, {len(mids)} matches "
          f"en {time.time()-t0:.2f}s")
    print(cmps)

    wc_mid = list_statsbomb_match_ids(comp_id=43, season_id=106)[0]
    t0 = time.time()
    ev = load_statsbomb_events(wc_mid)
    li = load_statsbomb_lineups(wc_mid)
    ff = load_statsbomb_360(wc_mid)
    print(f"WC22 match {wc_mid}: {ev.height:,} events, {li.height} lineups, "
          f"{ff.height:,} freeze-frames en {time.time()-t0:.2f}s")

    t0 = time.time()
    n_sb = scan_statsbomb_events().select(pl.len()).collect().item()
    print(f"statsbomb events totales (lazy concat 200): {n_sb:,} "
          f"en {time.time()-t0:.2f}s")
