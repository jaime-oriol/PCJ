"""
M07_shocks - Deteccion de shocks emocionales (goles) + ventanas ±10 min por jugador.

Un shock = gol valido (excluye disallowed + tandas penaltis; ya filtrado
por M03 goals_timeline). Por cada shock genera filas por jugador en campo
con perspectiva GOAL_FOR (equipo que marca) o GOAL_AGAINST (equipo que encaja).

Ventanas:
  pre   = [t - 600s, t)   -- 10 min antes del gol
  post  = (t, t + 600s]   -- 10 min despues

Flags (propuesta §M07):
  truncated_pre  : ventana pre recortada por inicio partido / transicion periodo
  truncated_post : ventana post recortada por fin partido / transicion periodo
  overlap_flag   : otro shock (goal) en ±10 min del mismo partido
  sub_in_window  : jugador entra (minute_in) o sale (minute_out) DENTRO de
                   la ventana pre [t-600, t) o post (t, t+600]. Politica:
                     - Solo flagea, NO excluye filas. M12 DiD/M13 AIPW pueden
                       censurar (IPCW Robins 1994) o excluir cuando flag=True.
                     - No marca jugadores que estaban en campo y salieron POR
                       el shock como reaccion tactica del entrenador (la
                       sustitucion-respuesta no es selection bias en pre, pero
                       SI puede serlo en post si fue causada por el resultado).
                       Capturable downstream con sub_off_minute > t (dentro
                       de window_post).
  et_flag        : shock en tiempo extra (period 3 o 4)

Convencion de tiempos:
  t_event_seconds, window_*_start/end estan en `start_game_clock` ABSOLUTO
  PFF (segundos desde inicio del partido). M03.goals_timeline ya resuelve el
  sgc real PFF (no sintetizado m*60+s SB), critico para alinear stoppage.

Output: data/parquet/derived/shocks/shocks_table.parquet

Acceptance (ARCHITECTURE): ~172 shocks-gol totales, tabla larga
~(175 goles val) x ~22 jugadores en campo = ~3.800-4.000 filas player-level,
distribuida en GOAL_FOR / GOAL_AGAINST aproximadamente 50/50.

Depende de: M01 (events/metadata), M03 (goals_timeline, player_minutes).
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from M01_loader_pff import list_event_match_ids, load_events
from M03_preprocess import goals_timeline, player_minutes


# -- Rutas ------------------------------------------------------------------

_REPO    = Path(__file__).resolve().parents[1]
_DERIVED = _REPO / "data" / "parquet" / "derived" / "shocks"


# -- Constantes pre-registradas -------------------------------------------

WINDOW_SECONDS = 600       # ±10 min
OVERLAP_SECONDS = 600      # otro shock dentro de ±10 min = overlap


# ===========================================================================
#  SECCION 1 — Helpers
# ===========================================================================

def _match_end_seconds(match_id: int) -> int:
    """Ultimo start_game_clock del partido (fin real del juego tras stoppage)."""
    ev = load_events(match_id)
    return int(
        ev.select(
            pl.col("gameEvents").struct.field("startGameClock").max()
        ).item()
    )


def _period_boundaries(match_id: int) -> dict[int, tuple[int, int]]:
    """Inicio y fin (en start_game_clock seconds) de cada periodo observado."""
    ev = load_events(match_id)
    b = ev.select([
        pl.col("gameEvents").struct.field("period").alias("period"),
        pl.col("gameEvents").struct.field("startGameClock").alias("sgc"),
    ]).group_by("period").agg([
        pl.col("sgc").min().alias("start"),
        pl.col("sgc").max().alias("end"),
    ])
    return {int(r["period"]): (int(r["start"]), int(r["end"]))
            for r in b.iter_rows(named=True)}


def _truncated_by_period(t_event: int, period: int,
                         period_bounds: dict[int, tuple[int, int]],
                         direction: str) -> tuple[int, bool]:
    """Ajusta la ventana pre/post si cruza un cambio de periodo o inicio/fin.

    direction='pre': devuelve (start_truncado, truncated_flag)
    direction='post': devuelve (end_truncado, truncated_flag)

    Regla: la ventana se contiene dentro del mismo periodo del shock. Si se
    sale del periodo, se recorta al inicio/fin del periodo.
    """
    p_start, p_end = period_bounds[period]
    if direction == "pre":
        ideal_start = t_event - WINDOW_SECONDS
        if ideal_start < p_start:
            return p_start, True
        return ideal_start, False
    else:   # post
        ideal_end = t_event + WINDOW_SECONDS
        if ideal_end > p_end:
            return p_end, True
        return ideal_end, False


# ===========================================================================
#  SECCION 2 — Build shocks table
# ===========================================================================

def build_shocks_table(cache: bool = True,
                       overwrite: bool = False) -> pl.DataFrame:
    """Construye la tabla larga de shocks por jugador-shock.

    Cada gol -> multiples filas (una por jugador en campo), con perspectiva
    GOAL_FOR / GOAL_AGAINST + ventanas + flags.
    """
    cache_path = _DERIVED / "shocks_table.parquet"
    if cache and cache_path.exists() and not overwrite:
        return pl.read_parquet(cache_path)

    all_rows = []
    shock_id_counter = 0
    n_goals_total = 0

    for mid in list_event_match_ids():
        goals = goals_timeline(mid)
        if goals.height == 0:
            continue
        pm = player_minutes(mid)
        p_bounds = _period_boundaries(mid)

        # Pre-compute overlap table: for each goal, does another goal fall in ±10min?
        goal_times = goals["start_game_clock"].to_list()

        for i, g in enumerate(goals.iter_rows(named=True)):
            shock_id_counter += 1
            n_goals_total += 1
            t = int(g["start_game_clock"])
            p = int(g["period"])
            minute_goal = int(g["minute"])
            scoring_team_id = int(g["scoring_team_id"])

            # ventana ajustada a periodo
            pre_start, trunc_pre = _truncated_by_period(t, p, p_bounds, "pre")
            post_end, trunc_post = _truncated_by_period(t, p, p_bounds, "post")

            # overlap: otro gol en ±10 min (excluyendo el actual)
            overlap_flag = any(
                abs(other_t - t) <= OVERLAP_SECONDS and other_t != t
                for other_t in goal_times
            )

            et_flag = p in (3, 4)

            # Jugadores en campo al minuto del gol
            # minute_in/minute_out de player_minutes (en MINUTOS)
            minute_threshold = minute_goal
            on_field = pm.filter(
                pl.col("minute_in").is_not_null() &
                (pl.col("minute_in") <= minute_threshold) &
                (pl.col("minute_out") >= minute_threshold)
            )

            for pl_row in on_field.iter_rows(named=True):
                player_team_id = int(pl_row["team_id"])
                shock_type = ("GOAL_FOR"
                              if player_team_id == scoring_team_id
                              else "GOAL_AGAINST")
                # Sub in window: jugador entra o sale dentro de pre [t-600, t) o post (t, t+600]
                m_in = int(pl_row["minute_in"])
                m_out = int(pl_row["minute_out"])
                win_pre_min = (pre_start // 60, t // 60)
                win_post_min = (t // 60, post_end // 60)
                sub_in_window = (
                    (win_pre_min[0] <= m_in <= win_pre_min[1]) or
                    (win_post_min[0] <= m_out <= win_post_min[1])
                )

                all_rows.append({
                    "match_id":           mid,
                    "shock_id":           shock_id_counter,
                    "t_event_seconds":    t,
                    "period":             p,
                    "minute":             minute_goal,
                    "scoring_team_id":    scoring_team_id,
                    "is_own_goal":        bool(g["is_own_goal"]),
                    "player_id":          int(pl_row["player_id"]),
                    "player_team_id":     player_team_id,
                    "position_group":     pl_row["position_group"],
                    "shock_type":         shock_type,
                    "window_pre_start":   pre_start,
                    "window_pre_end":     t,
                    "window_post_start":  t,
                    "window_post_end":    post_end,
                    "truncated_pre":      trunc_pre,
                    "truncated_post":     trunc_post,
                    "overlap_flag":       overlap_flag,
                    "sub_in_window":      sub_in_window,
                    "et_flag":            et_flag,
                    "minute_in":          m_in,
                    "minute_out":         m_out,
                })

    df = pl.DataFrame(all_rows)
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_path, compression="snappy", statistics=True)
    return df


def summary_shocks(df: pl.DataFrame) -> pl.DataFrame:
    """Resumen por tipo + flags."""
    return df.group_by("shock_type").agg([
        pl.len().alias("n_rows"),
        pl.col("shock_id").n_unique().alias("n_shocks"),
        pl.col("truncated_pre").sum().alias("n_trunc_pre"),
        pl.col("truncated_post").sum().alias("n_trunc_post"),
        pl.col("overlap_flag").sum().alias("n_overlap"),
        pl.col("sub_in_window").sum().alias("n_sub"),
        pl.col("et_flag").sum().alias("n_et"),
    ]).sort("shock_type")


# -- Sanity inline ---------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=== M07_shocks sanity ===")
    t0 = time.time()
    df = build_shocks_table(cache=True, overwrite=True)
    print(f"shocks table built en {time.time()-t0:.1f}s")
    print(f"  total filas player-shock: {df.height:,}")
    print(f"  shocks unicos (goles validos): {df['shock_id'].n_unique()}")

    print()
    print("Resumen por shock_type:")
    print(summary_shocks(df))

    # Sanity: rate on-field per goal (~22 players)
    per_shock = df.group_by("shock_id").len()
    print(f"\nJugadores en campo por shock: mean={per_shock['len'].mean():.1f} "
          f"(esperado ~22), min={per_shock['len'].min()}, max={per_shock['len'].max()}")

    # Sanity: GOAL_FOR vs GOAL_AGAINST balance (~50/50)
    by_type = df.group_by("shock_type").len().to_dicts()
    tot_for = next(x["len"] for x in by_type if x["shock_type"] == "GOAL_FOR")
    tot_ag  = next(x["len"] for x in by_type if x["shock_type"] == "GOAL_AGAINST")
    print(f"  GOAL_FOR: {tot_for}, GOAL_AGAINST: {tot_ag}, "
          f"ratio for/ag = {tot_for/tot_ag:.2f} (esperado ~1.0)")

    # ET y overlap sanity
    et_shocks = df.filter(pl.col("et_flag"))["shock_id"].n_unique()
    print(f"\nShocks en ET (prorroga): {et_shocks}")
    overlap = df.filter(pl.col("overlap_flag"))["shock_id"].n_unique()
    print(f"Shocks con overlap (otro gol ±10 min): {overlap}")
    trunc = df.filter(pl.col("truncated_pre") | pl.col("truncated_post"))["shock_id"].n_unique()
    print(f"Shocks con ventana truncada (inicio/fin periodo): {trunc}")

    # Muestra
    print()
    print("Muestra 5 filas:")
    print(df.head(5).select(["match_id","shock_id","minute","shock_type",
                             "player_id","truncated_pre","truncated_post",
                             "overlap_flag","sub_in_window","et_flag"]))

    # Acceptance check
    assert df["shock_id"].n_unique() >= 150, "Muy pocos shocks (<150)"
    assert df["shock_id"].n_unique() <= 250, "Demasiados shocks (>250)"
    print("\nACCEPTANCE: ~172 shocks-gol totales  OK")
