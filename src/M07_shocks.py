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
~(172 goles val) x ~22 jugadores en campo = ~3.800-4.000 filas player-level,
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

from M01_loader_pff import (
    list_event_match_ids, load_events, load_metadata, list_goals,
)
from M03_preprocess import (
    goals_timeline, player_minutes, week_index_continuous,
)


# -- Rutas ------------------------------------------------------------------

_REPO    = Path(__file__).resolve().parents[1]
_DERIVED = _REPO / "data" / "parquet" / "derived" / "shocks"


# -- Constantes pre-registradas -------------------------------------------

WINDOW_SECONDS = 600       # ±10 min
OVERLAP_SECONDS = 600      # otro shock dentro de ±10 min = overlap


# ===========================================================================
#  SECCION 1 — Helpers
# ===========================================================================

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
    """Construye la tabla larga de shocks por jugador-shock + tabla auxiliar
    `shocks_team_members.parquet` con la composicion del bloque por shock.

    Cada gol -> multiples filas (una por jugador en campo), con perspectiva
    GOAL_FOR / GOAL_AGAINST + ventanas + flags + moduladores continuos
    (minute_norm, week_idx_norm, score_diff_post) + n_teammates_in_field.

    `shocks_team_members.parquet` (1 fila por (shock_id, team_id, player_id)):
    feeder de leave-one-out para M08-M11 (Δ_player_relative al bloque).
    """
    cache_path = _DERIVED / "shocks_table.parquet"
    members_path = _DERIVED / "shocks_team_members.parquet"
    if (cache and cache_path.exists() and members_path.exists()
            and not overwrite):
        return pl.read_parquet(cache_path)

    all_rows = []
    team_member_rows: list[dict] = []
    shock_id_counter = 0

    # Stage map: week 1-3 = groups (48 partidos), 4-8 = ko (16 partidos)
    md = load_metadata().select(["id", "week"])
    week_map = {int(r["id"]): int(r["week"]) for r in md.iter_rows(named=True)}

    # Leverage + elim_prox map: M04 WP per_minute per (match, minute).
    # leverage = sensibilidad WP a un gol mas (pivotal moment).
    # elim_prox_home/away = P(equipo NO clasifica) — la "proximidad de irse a casa"
    # propuesta_final.md:27. Antes elim_prox NO se propagaba (gap auditoria).
    wp_path = _DERIVED.parent / "wp" / "per_minute.parquet"
    leverage_map: dict[tuple[int, int], float] = {}
    elim_prox_home_map: dict[tuple[int, int], float] = {}
    elim_prox_away_map: dict[tuple[int, int], float] = {}
    if wp_path.exists():
        wp = pl.read_parquet(wp_path).select(
            ["match_id", "minute", "leverage", "elim_prox_home", "elim_prox_away"])
        for r in wp.iter_rows(named=True):
            key = (int(r["match_id"]), int(r["minute"]))
            leverage_map[key] = float(r["leverage"] or 0.0)
            elim_prox_home_map[key] = float(r["elim_prox_home"] or 0.0)
            elim_prox_away_map[key] = float(r["elim_prox_away"] or 0.0)

    for mid in list_event_match_ids():
        goals = goals_timeline(mid)
        if goals.height == 0:
            continue
        pm = player_minutes(mid)
        p_bounds = _period_boundaries(mid)
        match_week = week_map.get(mid)
        match_stage = "groups" if (match_week is not None and match_week <= 3) else "ko"
        # Cache home_team_id para resolver elim_prox desde perspectiva del jugador
        match_md = load_metadata(mid).row(0, named=True)
        home_team_id = int(match_md["home_team_id"])

        # Pre-compute overlap table: for each goal, does another goal fall in ±10min?
        goal_times = goals["start_game_clock"].to_list()

        # Modulador continuo "fase del torneo" (week_idx ∈ [0,1])
        match_week_idx_norm = week_index_continuous(mid)

        for i, g in enumerate(goals.iter_rows(named=True)):
            shock_id_counter += 1
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

            # Marcador POST-shock: cum_home/away del propio gol ya incluyen
            # ese gol via cum_sum de goals_timeline. NO recomputar.
            sh_post = int(g["cum_home"])
            sa_post = int(g["cum_away"])

            # Modulador continuo: minuto normalizado a [0, ~1.33] (ET=hasta 120/90)
            minute_norm = minute_goal / 90.0

            # On-field al minuto del gol. minute_in/out de player_minutes en MINUTOS.
            on_field = pm.filter(
                pl.col("minute_in").is_not_null() &
                (pl.col("minute_in") <= minute_goal) &
                (pl.col("minute_out") >= minute_goal)
            )

            # Bloque por team: composicion del LOO downstream
            scoring_team_block = on_field.filter(
                pl.col("team_id") == scoring_team_id
            )["player_id"].to_list()
            other_team_block = on_field.filter(
                pl.col("team_id") != scoring_team_id
            )["player_id"].to_list()

            # Persistir composicion del bloque por (shock, team, perspective)
            for pid in scoring_team_block:
                team_member_rows.append({
                    "match_id": mid, "shock_id": shock_id_counter,
                    "team_id": scoring_team_id, "perspective": "GOAL_FOR",
                    "player_id": int(pid),
                })
            # team_id del rival: cualquier jugador del other_team_block tiene el mismo
            other_team_id = (int(on_field.filter(
                pl.col("team_id") != scoring_team_id
            )["team_id"][0]) if other_team_block else None)
            if other_team_id is not None:
                for pid in other_team_block:
                    team_member_rows.append({
                        "match_id": mid, "shock_id": shock_id_counter,
                        "team_id": other_team_id, "perspective": "GOAL_AGAINST",
                        "player_id": int(pid),
                    })

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

                shock_leverage = leverage_map.get((mid, minute_goal), 0.0)
                # Player-perspective elim_prox: del equipo del jugador
                ep_home = elim_prox_home_map.get((mid, minute_goal), 0.0)
                ep_away = elim_prox_away_map.get((mid, minute_goal), 0.0)
                ep_player = ep_home if player_team_id == home_team_id else ep_away

                # score_diff_post desde la perspectiva del jugador
                score_diff_post = (
                    (sh_post - sa_post)
                    if player_team_id == home_team_id
                    else (sa_post - sh_post)
                )
                # n_teammates_in_field: companeros del bloque del focal (excluido el focal)
                team_block_size = (
                    len(scoring_team_block)
                    if player_team_id == scoring_team_id
                    else len(other_team_block)
                )
                n_teammates_in_field = max(0, team_block_size - 1)

                all_rows.append({
                    "match_id":           mid,
                    "shock_id":           shock_id_counter,
                    "stage":              match_stage,
                    "match_week":         match_week,
                    "week_idx_norm":      match_week_idx_norm,   # modulador continuo fase
                    "leverage_at_shock":  shock_leverage,
                    "elim_prox_home_at_shock": ep_home,
                    "elim_prox_away_at_shock": ep_away,
                    "elim_prox_player_at_shock": ep_player,    # perspectiva jugador
                    "t_event_seconds":    t,
                    "period":             p,
                    "minute":             minute_goal,
                    "minute_norm":        minute_norm,             # modulador continuo
                    "score_home_post":    sh_post,
                    "score_away_post":    sa_post,
                    "score_diff_post":    score_diff_post,         # modulador continuo (player-pov)
                    "n_teammates_in_field": n_teammates_in_field,
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
    members_df = pl.DataFrame(team_member_rows, schema={
        "match_id": pl.Int64, "shock_id": pl.Int64, "team_id": pl.Int64,
        "perspective": pl.String, "player_id": pl.Int64,
    })
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_path, compression="snappy", statistics=True)
        members_path.parent.mkdir(parents=True, exist_ok=True)
        members_df.write_parquet(members_path, compression="snappy", statistics=True)
    return df


def load_team_members(cache: bool = True) -> pl.DataFrame:
    """`shocks_team_members.parquet` con la composicion del bloque por shock.

    Schema: (match_id, shock_id, team_id, perspective, player_id).
    Generado por `build_shocks_table`. Feeder de leave-one-out en M08-M11.
    """
    p = _DERIVED / "shocks_team_members.parquet"
    if not p.exists() and cache:
        build_shocks_table(cache=True, overwrite=False)
    return pl.read_parquet(p)


def attach_team_loo(per_minute: pl.DataFrame, value_col: str,
                     match_col: str = "pff_match_id",
                     player_col: str = "pff_player_id") -> pl.DataFrame:
    """Para cada (shock, focal_player), agrega:
       value_pre, value_post           (suma del focal en ventana ±10 min)
       value_team_loo_pre, _post       (mean per-teammate excluyendo al focal)
       value_relative_pre, _post       (focal - team_loo, mismo sentido)
       value_delta_relative            ((post - pre)_focal - (post - pre)_team_loo)

    Mecanica:
      1. Per (shock, member): suma value_col del MIEMBRO del bloque en ventana.
      2. Per (shock, perspective): team_total = SUM members; n_block = count.
      3. Per (shock, focal): team_loo = (team_total - focal_sum) / (n_block - 1).

    Args:
        per_minute: DataFrame con cols [match_col, player_col, period, sec_abs,
                    value_col]. M08-M11 lo emiten con esos nombres.
        value_col:  Nombre de la col a agregar (e.g., 'score_atk_minute').

    Returns:
        DataFrame por (match_id, shock_id, focal_player_id, shock_type) con
        las 5 cols nuevas. Filtra shocks_table igual que el wrapper actual.
    """
    shocks = build_shocks_table(cache=True, overwrite=False)
    members = load_team_members(cache=True)

    # 1) Para cada miembro del bloque, suma de value_col en ventana pre y post
    sh_slim = shocks.select([
        "match_id", "shock_id", "shock_type",
        "window_pre_start", "window_pre_end",
        "window_post_start", "window_post_end",
        pl.col("period").alias("shock_period"),
    ]).unique(subset=["match_id", "shock_id"])

    pm = per_minute.rename({match_col: "match_id", player_col: "member_player_id"})

    # join shocks <- members <- per_minute via (match_id, member_player_id)
    member_with_window = members.rename({"player_id": "member_player_id"}).join(
        sh_slim, on=["match_id", "shock_id"], how="left"
    )
    joined = member_with_window.join(
        pm.select(["match_id", "member_player_id", "period", "sec_abs", value_col]),
        on=["match_id", "member_player_id"], how="left",
    )
    pre_member = joined.filter(
        (pl.col("sec_abs") >= pl.col("window_pre_start")) &
        (pl.col("sec_abs") < pl.col("window_pre_end")) &
        (pl.col("period") == pl.col("shock_period"))
    ).group_by(["match_id", "shock_id", "perspective", "team_id",
                 "member_player_id"]).agg(
        pl.col(value_col).sum().alias("member_pre")
    )
    post_member = joined.filter(
        (pl.col("sec_abs") >= pl.col("window_post_start")) &
        (pl.col("sec_abs") <= pl.col("window_post_end")) &
        (pl.col("period") == pl.col("shock_period"))
    ).group_by(["match_id", "shock_id", "perspective", "team_id",
                 "member_player_id"]).agg(
        pl.col(value_col).sum().alias("member_post")
    )
    member_full = (members.rename({"player_id": "member_player_id"})
                   .join(pre_member,
                         on=["match_id", "shock_id", "perspective",
                              "team_id", "member_player_id"], how="left")
                   .join(post_member,
                         on=["match_id", "shock_id", "perspective",
                              "team_id", "member_player_id"], how="left"))
    member_full = member_full.with_columns([
        pl.col("member_pre").fill_null(0.0),
        pl.col("member_post").fill_null(0.0),
    ])

    # 2) Per (shock, perspective): team_total + n_block
    team_total = member_full.group_by(["match_id", "shock_id", "perspective"]).agg([
        pl.col("member_pre").sum().alias("team_total_pre"),
        pl.col("member_post").sum().alias("team_total_post"),
        pl.col("member_player_id").n_unique().alias("n_block"),
    ])

    # 3) Per (shock, focal) — focal_pre/post (= member_pre/post del focal) + LOO
    focal = (shocks.select([
        "match_id", "shock_id", "shock_type",
        pl.col("player_id").alias("focal_player_id"),
        pl.col("player_team_id").alias("focal_team_id"),
    ]).unique(subset=["match_id", "shock_id", "focal_player_id"]))

    # focal_pre/post desde member_full (focal_player_id == member_player_id)
    focal_self = member_full.select([
        "match_id", "shock_id", "perspective",
        pl.col("member_player_id").alias("focal_player_id"),
        pl.col("member_pre").alias(f"{value_col}_pre"),
        pl.col("member_post").alias(f"{value_col}_post"),
    ])
    # join: focal por shock_type == perspective
    focal_with_self = (focal
        .with_columns(
            pl.when(pl.col("shock_type") == "GOAL_FOR").then(pl.lit("GOAL_FOR"))
              .otherwise(pl.lit("GOAL_AGAINST")).alias("perspective")
        )
        .join(focal_self,
              on=["match_id", "shock_id", "perspective", "focal_player_id"],
              how="left")
        .join(team_total, on=["match_id", "shock_id", "perspective"], how="left"))

    out = focal_with_self.with_columns([
        pl.col(f"{value_col}_pre").fill_null(0.0),
        pl.col(f"{value_col}_post").fill_null(0.0),
        pl.col("team_total_pre").fill_null(0.0),
        pl.col("team_total_post").fill_null(0.0),
        pl.col("n_block").fill_null(0).cast(pl.Int64),
    ]).with_columns([
        pl.when(pl.col("n_block") > 1)
          .then((pl.col("team_total_pre") - pl.col(f"{value_col}_pre"))
                / (pl.col("n_block") - 1))
          .otherwise(pl.lit(None, dtype=pl.Float64))
          .alias(f"{value_col}_team_loo_pre"),
        pl.when(pl.col("n_block") > 1)
          .then((pl.col("team_total_post") - pl.col(f"{value_col}_post"))
                / (pl.col("n_block") - 1))
          .otherwise(pl.lit(None, dtype=pl.Float64))
          .alias(f"{value_col}_team_loo_post"),
    ]).with_columns([
        (pl.col(f"{value_col}_pre") - pl.col(f"{value_col}_team_loo_pre"))
            .alias(f"{value_col}_relative_pre"),
        (pl.col(f"{value_col}_post") - pl.col(f"{value_col}_team_loo_post"))
            .alias(f"{value_col}_relative_post"),
    ]).with_columns([
        ((pl.col(f"{value_col}_post") - pl.col(f"{value_col}_pre"))
         - (pl.col(f"{value_col}_team_loo_post")
            - pl.col(f"{value_col}_team_loo_pre")))
            .alias(f"{value_col}_delta_relative"),
        ((pl.col(f"{value_col}_post") - pl.col(f"{value_col}_pre")))
            .alias(f"{value_col}_delta_player"),
        ((pl.col(f"{value_col}_team_loo_post")
          - pl.col(f"{value_col}_team_loo_pre")))
            .alias(f"{value_col}_delta_team_loo"),
    ]).select([
        "match_id", "shock_id", "shock_type",
        pl.col("focal_player_id").alias("pff_player_id"),
        f"{value_col}_pre", f"{value_col}_post",
        f"{value_col}_team_loo_pre", f"{value_col}_team_loo_post",
        f"{value_col}_relative_pre", f"{value_col}_relative_post",
        f"{value_col}_delta_player", f"{value_col}_delta_team_loo",
        f"{value_col}_delta_relative",
        pl.col("n_block"),
    ])
    return out


def compute_team_loo_at_minute(per_minute: pl.DataFrame, value_col: str,
                                 minute_col: str = "minute_abs_join",
                                 minutes_grid: pl.DataFrame | None = None,
                                 fill_value: float | None = 0.0,
                                 ) -> pl.DataFrame:
    """LOO a granularidad MINUTO: para cada (shock, perspective, focal, minute)
    devuelve `team_total`, `n_block`, `outcome_team_loo`, `outcome_relative`.

    Mecanica:
      1. cross-product (shock × member × minute_grid) → cobertura completa,
         miembros sin actividad cuentan como `fill_value` (0 default — un
         minuto sin accion del miembro contribuye 0 al team_total).
      2. Per (shock, perspective, minute): SUM members + count distinct.
      3. Per (shock, perspective, minute, focal): team_loo = (team_total -
         focal_outcome) / (n_block - 1).

    Args:
        per_minute: cols [pff_match_id, pff_player_id, minute_col, value_col].
        minutes_grid: opcional, DataFrame con [pff_match_id, shock_id, minute_col]
                      restringiendo el grid (e.g., bins de event-study). Si None,
                      cross con TODOS los minutos donde algun miembro del bloque
                      tiene fila — equivalente a la cobertura natural.
        fill_value: relleno para miembros sin actividad en ese minuto (0 para
                    canales aditivos; None = drop esos miembros).

    Returns:
        DataFrame columns: pff_match_id, shock_id, perspective, focal_player_id,
        minute_col, team_total, n_block, outcome_team_loo, outcome_relative.
    """
    members = load_team_members(cache=True).rename({
        "match_id": "pff_match_id", "player_id": "member_player_id",
    })
    pm = per_minute.rename({"pff_player_id": "member_player_id"}).select([
        "pff_match_id", "member_player_id", minute_col, value_col,
    ])

    # Grid de (shock × minute) sobre el que hacer LOO
    if minutes_grid is None:
        # Cobertura natural: minutos donde algun miembro de algun bloque tiene fila.
        minutes_grid = (members.join(pm, on=["pff_match_id", "member_player_id"],
                                       how="inner")
                        .select(["pff_match_id", "shock_id", minute_col])
                        .unique())

    # cross members × grid: cada miembro aparece para cada minuto del grid del shock
    members_grid = members.join(
        minutes_grid, on=["pff_match_id", "shock_id"], how="inner"
    )
    members_pm = members_grid.join(
        pm, on=["pff_match_id", "member_player_id", minute_col], how="left"
    )
    if fill_value is not None:
        members_pm = members_pm.with_columns(
            pl.col(value_col).fill_null(fill_value)
        )
    else:
        members_pm = members_pm.filter(pl.col(value_col).is_not_null())

    team_aggs = members_pm.group_by(
        ["pff_match_id", "shock_id", "perspective", minute_col]
    ).agg([
        pl.col(value_col).sum().alias("team_total"),
        pl.col("member_player_id").n_unique().alias("n_block"),
    ])

    # focal_outcome = el outcome del propio focal en ese minute (= member_pre/post
    # cuando el focal es el member). Lo sacamos del members_pm pre-aggregado.
    focal_self = members_pm.select([
        "pff_match_id", "shock_id", "perspective", minute_col,
        pl.col("member_player_id").alias("focal_player_id"),
        pl.col(value_col).alias("outcome_focal"),
    ])
    out = focal_self.join(
        team_aggs,
        on=["pff_match_id", "shock_id", "perspective", minute_col], how="left",
    ).with_columns([
        pl.when(pl.col("n_block") > 1)
          .then((pl.col("team_total") - pl.col("outcome_focal"))
                / (pl.col("n_block") - 1))
          .otherwise(pl.lit(None, dtype=pl.Float64))
          .alias("outcome_team_loo"),
    ]).with_columns(
        (pl.col("outcome_focal") - pl.col("outcome_team_loo"))
          .alias("outcome_relative")
    )
    return out


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
