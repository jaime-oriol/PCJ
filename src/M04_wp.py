"""
M04_wp - Win Probability bayesiana tiempo-variables + leverage.

Reimplementacion Robberechts, Van Haaren & Davis (2021, KDD) en numpyro:
  - Ordered-logistic 3-class {H, D, A} con intercepts tiempo-variables
  - Random-walk prior sobre los 90 time-bins (smoothness temporal)
  - SVI (AutoNormal) para inferencia rapida CPU
  - Features: score_diff, red_diff, elo_diff, comp_tier, shots_diff_recent
    (ventana rolling 10 min, proxy momentum ofensivo cross-dataset)

Training cross-dataset (excluyendo WC22 sagrado):
  - Wyscout 2017/18 Big 5 + Euro16 + WC18  : 1.941 partidos
  - StatsBomb Euro20 + Euro24 + Bundes23/24:   136 partidos
  Total: 2.077 partidos x 90 bins = ~187k samples.

Cobertura completa 0-90 + ET + penaltis:
  - Regulacion 0-90 : modelo bayesiano entrenado.
  - ET (90-120)    : Poisson goal-rate empirico sobre subset ET.
  - Penaltis tanda : Tijms (2019) formula cerrada.

Calibracion: temperature scaling sobre 48 partidos WC22 fase de grupos
(NO sobre los 16 KO, que son sagrados para test final).

Output: tabla `data/parquet/derived/wp/per_minute.parquet` con
(match_id, minute, wp_home, wp_draw, wp_away, leverage, elim_prox).

Depende de M01 (events+metadata PFF), M02 (training Wyscout+StatsBomb), M03
(goals_timeline PFF para evaluacion).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from M01_loader_pff import load_metadata as pff_meta, list_event_match_ids
from M02_loader_public import (
    STATSBOMB_COMPETITIONS,
    scan_wyscout_events, load_wyscout_matches,
    load_statsbomb_matches, load_statsbomb_events, list_statsbomb_match_ids,
)
from M03_preprocess import goals_timeline as pff_goals_timeline


# -- Rutas ------------------------------------------------------------------

_REPO    = Path(__file__).resolve().parents[1]
_DERIVED = _REPO / "data" / "parquet" / "derived" / "wp"
_CACHE   = _DERIVED / "training"      # training matrix + elo (cache)
_MODEL   = _DERIVED / "model"         # SVI params + temperature


# -- Constantes -------------------------------------------------------------

N_BINS          = 90       # 1 bin por minuto de regulacion
REG_MINUTES     = 90
ET_MINUTES      = 30
ELO_INIT        = 1500.0
ELO_K           = 20.0

# Mapeo periodo Wyscout -> offset de minuto absoluto
_WYSCOUT_PERIOD_OFFSET = {"1H": 0, "2H": 45, "E1": 90, "E2": 105, "P": 120}


# ===========================================================================
#  SECCION 1 — Extraccion de timelines (goles + rojas) cross-dataset
# ===========================================================================

def _wyscout_goals_timeline() -> pl.DataFrame:
    """Timeline de goles Wyscout: (match_id, minute_abs, scoring_team_id, is_own_goal).

    Dos fuentes de goles:
      (A) Goles normales: eventName in (Shot, Free Kick) + tag 101
          scoring_team = event.teamId
      (B) Own goals: CUALQUIER event con tag 102
          scoring_team = OPP del event.teamId (el equipo que comete propia
          es el event_team; el beneficiario es el rival).
          Dedupe por (matchId, minute_abs, teamId) para evitar que PFF marque
          tag 102 en varios eventos de la misma jugada.

    Para derivar el OPP en (B), join con metadata matches por matchId + side.
    """
    ev = scan_wyscout_events().with_columns(
        pl.col("tags").list.eval(pl.element().struct.field("id")).alias("tag_ids"),
        pl.col("matchPeriod").replace_strict(_WYSCOUT_PERIOD_OFFSET, default=0)
                              .cast(pl.Int64).alias("period_offset"),
        (pl.col("eventSec") / 60.0).alias("sec_min"),
    ).with_columns(
        (pl.col("period_offset") + pl.col("sec_min")).cast(pl.Int64).alias("minute_abs")
    )

    # (A) goles normales
    g_norm = ev.filter(
        pl.col("eventName").is_in(["Shot", "Free Kick"]) &
        pl.col("tag_ids").list.contains(101)
    ).select([
        pl.col("matchId").alias("match_id"),
        pl.col("teamId").alias("scoring_team_id"),
        "minute_abs",
    ]).with_columns(pl.lit(False).alias("is_own_goal"))

    # (B) own-goals: evento con tag 102. Scoring_team es el OPP.
    og = ev.filter(pl.col("tag_ids").list.contains(102)).select([
        pl.col("matchId").alias("match_id"),
        pl.col("teamId").alias("event_team_id"),
        "minute_abs",
    ]).collect().unique(subset=["match_id", "minute_abs", "event_team_id"])

    # Mapear event_team_id a OPP: para cada matchId sacar los 2 teamIds
    teams_per_match = scan_wyscout_events().select(["matchId","teamId"]).unique().collect()
    teams_per_match = teams_per_match.group_by("matchId").agg(
        pl.col("teamId").alias("teams")
    ).rename({"matchId": "match_id"})
    og = og.join(teams_per_match, on="match_id", how="left").with_columns(
        pl.struct(["teams", "event_team_id"]).map_elements(
            lambda r: [t for t in r["teams"] if t != r["event_team_id"]][0]
                      if r["teams"] is not None and len(r["teams"]) >= 2 else None,
            return_dtype=pl.Int64,
        ).alias("scoring_team_id")
    ).filter(pl.col("scoring_team_id").is_not_null()).select([
        "match_id", "scoring_team_id", "minute_abs",
    ]).with_columns(pl.lit(True).alias("is_own_goal"))

    out = pl.concat([g_norm.collect(), og], how="diagonal_relaxed")
    return out.with_columns(pl.lit("wyscout").alias("source"))


def _wyscout_reds_timeline() -> pl.DataFrame:
    """Timeline rojas Wyscout: (match_id, minute_abs, team_id). Tags 1701/1703."""
    ev = scan_wyscout_events().with_columns(
        pl.col("tags").list.eval(pl.element().struct.field("id")).alias("tag_ids")
    )
    r = ev.filter(
        pl.col("tag_ids").list.contains(1701) | pl.col("tag_ids").list.contains(1703)
    ).with_columns([
        pl.col("matchPeriod").replace_strict(_WYSCOUT_PERIOD_OFFSET, default=0)
                              .cast(pl.Int64).alias("period_offset"),
        (pl.col("eventSec") / 60.0).alias("sec_min"),
    ]).with_columns(
        (pl.col("period_offset") + pl.col("sec_min")).cast(pl.Int64).alias("minute_abs")
    ).select([
        pl.col("matchId").alias("match_id"),
        pl.col("teamId").alias("team_id"),
        "minute_abs",
    ])
    return r.collect().with_columns(pl.lit("wyscout").alias("source"))


def _statsbomb_goals_timeline(match_ids: list[int]) -> pl.DataFrame:
    """Timeline goles StatsBomb: shot.outcome.name == 'Goal' u 'Own Goal For'."""
    rows = []
    for mid in match_ids:
        ev = load_statsbomb_events(mid)
        # Goal normal
        if "shot" in ev.columns:
            g = ev.filter(
                pl.col("shot").struct.field("outcome").struct.field("name") == "Goal"
            ).select([
                pl.lit(mid).cast(pl.Int64).alias("match_id"),
                pl.col("team").struct.field("id").alias("scoring_team_id"),
                ((pl.col("period") - 1) * 45 + pl.col("minute"))
                .cast(pl.Int64).alias("minute_abs"),
            ])
            rows.append(g)
        # Own Goal For (type.name == 'Own Goal For'): team que BENEFICIA
        og = ev.filter(
            pl.col("type").struct.field("name") == "Own Goal For"
        ).select([
            pl.lit(mid).cast(pl.Int64).alias("match_id"),
            pl.col("team").struct.field("id").alias("scoring_team_id"),
            ((pl.col("period") - 1) * 45 + pl.col("minute")).cast(pl.Int64).alias("minute_abs"),
        ])
        rows.append(og)
    out = pl.concat([r for r in rows if r.height > 0], how="diagonal_relaxed") \
        if rows else pl.DataFrame(schema={"match_id": pl.Int64, "scoring_team_id": pl.Int64, "minute_abs": pl.Int64})
    return out.with_columns(pl.lit("statsbomb").alias("source"))


def _wyscout_shots_timeline() -> pl.DataFrame:
    """Timeline de shots Wyscout: (match_id, minute_abs, team_id).

    Shots = eventName in (Shot, Free Kick). Incluye shots que NO son gol.
    Usado como proxy de momentum ofensivo en la ventana reciente.
    """
    ev = scan_wyscout_events().with_columns(
        pl.col("matchPeriod").replace_strict(_WYSCOUT_PERIOD_OFFSET, default=0)
                              .cast(pl.Int64).alias("period_offset"),
        (pl.col("eventSec") / 60.0).alias("sec_min"),
    ).with_columns(
        (pl.col("period_offset") + pl.col("sec_min")).cast(pl.Int64).alias("minute_abs")
    )
    s = ev.filter(pl.col("eventName").is_in(["Shot", "Free Kick"])).select([
        pl.col("matchId").alias("match_id"),
        pl.col("teamId").alias("team_id"),
        "minute_abs",
    ]).collect().with_columns(pl.lit("wyscout").alias("source"))
    return s


def _statsbomb_shots_timeline(match_ids: list[int]) -> pl.DataFrame:
    """Timeline de shots StatsBomb: type.name == 'Shot'."""
    rows = []
    for mid in match_ids:
        ev = load_statsbomb_events(mid)
        shots = ev.filter(pl.col("type").struct.field("name") == "Shot")
        if shots.height == 0:
            continue
        s = shots.select([
            pl.lit(mid).cast(pl.Int64).alias("match_id"),
            pl.col("team").struct.field("id").alias("team_id"),
            ((pl.col("period") - 1) * 45 + pl.col("minute")).cast(pl.Int64).alias("minute_abs"),
        ])
        rows.append(s)
    if not rows:
        return pl.DataFrame(schema={"match_id": pl.Int64, "team_id": pl.Int64,
                                      "minute_abs": pl.Int64})
    return pl.concat(rows, how="diagonal_relaxed").with_columns(
        pl.lit("statsbomb").alias("source")
    )


def _statsbomb_reds_timeline(match_ids: list[int]) -> pl.DataFrame:
    """Timeline rojas StatsBomb: foul_committed.card + bad_behaviour.card.

    Defensivo: sub-struct 'card' solo existe si algun jugador recibio tarjeta
    en ese partido. Chequeamos fields internos del Struct via schema.
    """
    rows = []
    for mid in match_ids:
        ev = load_statsbomb_events(mid)
        card_exprs = []
        for col in ("foul_committed", "bad_behaviour"):
            if col not in ev.columns:
                continue
            inner = ev.schema[col]
            if inner is None or not hasattr(inner, "fields"):
                continue
            inner_names = {f.name for f in inner.fields}
            if "card" not in inner_names:
                continue
            card_exprs.append(
                pl.col(col).struct.field("card").struct.field("name").alias(f"card_{col}")
            )
        if not card_exprs:
            continue
        df = ev.select(
            [pl.lit(mid).cast(pl.Int64).alias("match_id"),
             pl.col("team").struct.field("id").alias("team_id"),
             ((pl.col("period") - 1) * 45 + pl.col("minute")).cast(pl.Int64).alias("minute_abs")]
            + card_exprs
        )
        mask = pl.lit(False)
        for c in df.columns:
            if c.startswith("card_"):
                mask = mask | df[c].is_in(["Red Card", "Second Yellow"])
        r = df.filter(mask).select(["match_id", "team_id", "minute_abs"])
        if r.height > 0:
            rows.append(r)
    if not rows:
        return pl.DataFrame(schema={"match_id": pl.Int64, "team_id": pl.Int64, "minute_abs": pl.Int64})
    out = pl.concat(rows, how="diagonal_relaxed")
    return out.with_columns(pl.lit("statsbomb").alias("source"))


# ===========================================================================
#  SECCION 2 — Metadata unificada (home/away teams + scores) cross-dataset
# ===========================================================================

def _wyscout_match_meta() -> pl.DataFrame:
    """Metadata unificada de matches Wyscout: home/away teams + scores + date.

    Parsea teamsData para sacar (home_team_id, away_team_id, score_home_90,
    score_away_90, score_home_et, score_away_et) via map_elements.
    """
    wm = load_wyscout_matches()

    def _parse(td: dict) -> dict:
        home_id = away_id = 0
        sh = sa = she = sae = 0
        for tid_s, data in td.items():
            if data is None:
                continue
            side = data.get("side")
            tid = int(tid_s)
            sc90 = int(data.get("score") or 0)
            scet = int(data.get("scoreET") or 0)
            if side == "home":
                home_id, sh, she = tid, sc90, scet
            elif side == "away":
                away_id, sa, sae = tid, sc90, scet
        return {
            "home_team_id": home_id, "away_team_id": away_id,
            "score_home_90": sh, "score_away_90": sa,
            "score_home_et": she, "score_away_et": sae,
        }

    parsed_schema = pl.Struct({
        "home_team_id": pl.Int64, "away_team_id": pl.Int64,
        "score_home_90": pl.Int64, "score_away_90": pl.Int64,
        "score_home_et": pl.Int64, "score_away_et": pl.Int64,
    })
    parsed = wm.select("teamsData").with_columns(
        pl.col("teamsData").map_elements(_parse, return_dtype=parsed_schema).alias("p")
    ).unnest("p").drop("teamsData")
    out = pl.concat([wm.select(["wyId", "dateutc", "competition", "duration"]), parsed], how="horizontal")
    return out.rename({"wyId": "match_id", "dateutc": "match_date"}).with_columns([
        pl.lit("wyscout").alias("source"),
        # tier competicion: 1=top5, 2=Euro, 3=WC, 4=Bundes
        pl.col("competition").map_elements(
            lambda c: 1 if c in {"England","France","Germany","Italy","Spain"}
                       else 2 if c == "European_Championship"
                       else 3 if c == "World_Cup" else 4,
            return_dtype=pl.Int64,
        ).alias("comp_tier"),
    ])


def _statsbomb_match_meta() -> pl.DataFrame:
    """Metadata unificada StatsBomb (excluyendo WC22)."""
    m = load_statsbomb_matches()
    m = m.filter(~((pl.col("competition_id") == 43) & (pl.col("season_id") == 106)))
    # Para 90-min score no lo tenemos directo; usamos home_score/away_score como final
    # y asumimos duration=Regular (NO es ET-aware en StatsBomb catalog).
    # Ventaja: si un partido fue a ET, su score "final" incluye ET. Corregimos con
    # timeline: contamos goles con minute_abs > 90 como ET-goals.
    out = m.select([
        pl.col("match_id"),
        pl.col("match_date"),
        pl.col("competition_name").alias("competition"),
        pl.col("home_team_id"), pl.col("away_team_id"),
        pl.col("home_score").alias("score_home_final"),
        pl.col("away_score").alias("score_away_final"),
    ]).with_columns([
        pl.lit("statsbomb").alias("source"),
        pl.when(pl.col("competition") == "1. Bundesliga").then(4)
         .otherwise(2).alias("comp_tier"),   # Bundes=4, Euros=2
    ])
    return out


def _unify_match_meta(wy: pl.DataFrame, sb: pl.DataFrame,
                     goals_all: pl.DataFrame) -> pl.DataFrame:
    """Unifica metadata Wyscout + StatsBomb con scores_90 derivados del timeline."""
    # StatsBomb: derivar score_home_90 y score_away_90 contando goles minute_abs <= 90
    g90 = goals_all.filter(pl.col("minute_abs") <= REG_MINUTES).group_by(
        ["match_id", "scoring_team_id"]
    ).len().rename({"len": "n_goals_90"})

    # score_home_90 / score_away_90 desde g90 join con home_team_id/away_team_id
    sb_aug = sb.join(
        g90.rename({"scoring_team_id": "home_team_id", "n_goals_90": "sh90"}),
        on=["match_id", "home_team_id"], how="left"
    ).join(
        g90.rename({"scoring_team_id": "away_team_id", "n_goals_90": "sa90"}),
        on=["match_id", "away_team_id"], how="left"
    ).with_columns([
        pl.col("sh90").fill_null(0).alias("score_home_90"),
        pl.col("sa90").fill_null(0).alias("score_away_90"),
    ]).drop(["sh90", "sa90", "score_home_final", "score_away_final"])

    # Alinear Wyscout a mismo esquema
    wy_aug = wy.select([
        "match_id", "match_date", "competition",
        "home_team_id", "away_team_id",
        "score_home_90", "score_away_90",
        "source", "comp_tier",
    ])
    casts = [
        pl.col("score_home_90").cast(pl.Int64),
        pl.col("score_away_90").cast(pl.Int64),
        pl.col("comp_tier").cast(pl.Int64),
    ]
    sb_aug = sb_aug.select(wy_aug.columns).with_columns(casts)
    wy_aug = wy_aug.with_columns(casts)
    return pl.concat([wy_aug, sb_aug])


# ===========================================================================
#  SECCION 3 — Elo dinamico bottom-up
# ===========================================================================

def compute_elo(meta: pl.DataFrame, k: float = ELO_K,
                home_adv: float = 100.0) -> pl.DataFrame:
    """Elo dinamico K=20 sobre los matches ordenados por fecha.

    Inicializa cada equipo en 1500. Actualiza con cada partido segun resultado
    final (home_90 vs away_90, sin ET para ser comparables al WP-regulation).
    Devuelve meta + cols: elo_home_pre, elo_away_pre (valor ANTES del partido).
    """
    df = meta.sort("match_date").to_dicts()
    elo: dict[int, float] = {}
    out_rows = []
    for r in df:
        h, a = r["home_team_id"], r["away_team_id"]
        eh = elo.get(h, ELO_INIT)
        ea = elo.get(a, ELO_INIT)
        exp_h = 1.0 / (1.0 + 10 ** ((ea - (eh + home_adv)) / 400.0))
        sh, sa = r["score_home_90"], r["score_away_90"]
        if sh > sa:   actual_h = 1.0
        elif sh < sa: actual_h = 0.0
        else:         actual_h = 0.5
        eh_new = eh + k * (actual_h - exp_h)
        ea_new = ea + k * ((1 - actual_h) - (1 - exp_h))
        out_rows.append({**r, "elo_home_pre": eh, "elo_away_pre": ea})
        elo[h], elo[a] = eh_new, ea_new
    return pl.DataFrame(out_rows)


# ===========================================================================
#  SECCION 4 — Training matrix (match_id x minute_bin 1..90)
# ===========================================================================

_SHOTS_WINDOW = 10   # minutos de la ventana rolling para shots_diff_recent


def _counts_per_match_minute(events: pl.DataFrame, meta: pl.DataFrame,
                             team_col: str) -> np.ndarray:
    """Para cada (match, minute_bin 1..90), cuenta eventos del home y del away.

    Devuelve ndarray (n_matches, 90, 2) int32: [:, :, 0]=home, [:, :, 1]=away.
    Este formato permite cum_sum y rolling sum via operaciones numpy
    vectorizadas (mas barato que joins cross × bins × events).
    """
    if events.height == 0:
        return np.zeros((meta.height, REG_MINUTES, 2), dtype=np.int32)

    joined = events.join(
        meta.select(["match_id", "home_team_id", "away_team_id"]),
        on="match_id", how="inner",
    ).with_columns(
        pl.when(pl.col(team_col) == pl.col("home_team_id")).then(0)
          .when(pl.col(team_col) == pl.col("away_team_id")).then(1)
          .otherwise(None)
          .alias("side_idx"),
        pl.col("minute_abs").clip(1, REG_MINUTES).alias("bin"),
    ).filter(pl.col("side_idx").is_not_null()).select(["match_id", "bin", "side_idx"])

    mid_to_row = {m: i for i, m in enumerate(meta["match_id"].to_list())}
    counts = np.zeros((len(mid_to_row), REG_MINUTES, 2), dtype=np.int32)
    for mid, b, s in zip(joined["match_id"].to_list(),
                          joined["bin"].to_list(),
                          joined["side_idx"].to_list()):
        row = mid_to_row.get(mid)
        if row is not None:
            counts[row, b - 1, s] += 1
    return counts


def build_training_matrix(cache: bool = True) -> tuple[pl.DataFrame, dict]:
    """Construye matrix (match_id, minute, [features], y).

    Features siguen Robberechts et al. 2021:
      score_diff, red_diff, elo_diff, comp_tier, shots_diff_recent.

    y: target multinomial {0=H, 1=D, 2=A} al final de los 90 minutos regulares.
    Vectorizacion: counts por (match × minute × side) como ndarray, cum_sum
    para goles/rojas, rolling-10 para shots. Cacheable a parquet.
    """
    cache_path = _CACHE / "training_matrix.parquet"
    if cache and cache_path.exists():
        df = pl.read_parquet(cache_path)
        return df, {"n_samples": df.height, "cached": True}

    # 1. Timelines cross-dataset
    g_wy = _wyscout_goals_timeline()
    r_wy = _wyscout_reds_timeline()
    sh_wy = _wyscout_shots_timeline()
    sb_mids = [mid for cid, sid in STATSBOMB_COMPETITIONS.values()
               if (cid, sid) != (43, 106)
               for mid in list_statsbomb_match_ids(comp_id=cid, season_id=sid)]
    g_sb = _statsbomb_goals_timeline(sb_mids)
    r_sb = _statsbomb_reds_timeline(sb_mids)
    sh_sb = _statsbomb_shots_timeline(sb_mids)

    goals_all = pl.concat([g_wy, g_sb], how="diagonal_relaxed")
    reds_all  = pl.concat([r_wy, r_sb], how="diagonal_relaxed")
    shots_all = pl.concat([sh_wy, sh_sb], how="diagonal_relaxed")

    # 2. Metadata unificada
    wy_meta = _wyscout_match_meta()
    sb_meta = _statsbomb_match_meta()
    meta = _unify_match_meta(wy_meta, sb_meta, goals_all)

    # 3. Elo
    meta = compute_elo(meta)

    # 4. Target: H / D / A al minuto 90
    meta = meta.with_columns(
        pl.when(pl.col("score_home_90") > pl.col("score_away_90")).then(pl.lit(0))
          .when(pl.col("score_home_90") < pl.col("score_away_90")).then(pl.lit(2))
          .otherwise(pl.lit(1))
          .alias("y")
    )

    # 5. Counts por (match × minute × side) — ndarray compacto
    goals_c = _counts_per_match_minute(goals_all, meta, "scoring_team_id")
    reds_c  = _counts_per_match_minute(reds_all,  meta, "team_id")
    shots_c = _counts_per_match_minute(shots_all, meta, "team_id")

    # Cumulativos para goles/rojas (shape match × bin × side)
    #  - "eventos en minuto m" se asignan al bin m. El score al INICIO del bin
    #    m es el cum hasta m-1. Usamos cumsum shifted: pad 1 delante y recorta.
    gc_cum = np.cumsum(goals_c, axis=1)                          # count al final del bin
    rc_cum = np.cumsum(reds_c,  axis=1)
    # Score/red "antes de" el bin m = count en bins < m -> shift +1 en eje minute
    gc_pre = np.concatenate([np.zeros_like(gc_cum[:, :1, :]),
                              gc_cum[:, :-1, :]], axis=1)
    rc_pre = np.concatenate([np.zeros_like(rc_cum[:, :1, :]),
                              rc_cum[:, :-1, :]], axis=1)

    # Rolling shots_diff_recent: sum de shots en los W minutos anteriores al bin
    # (ventana abierta por la derecha: [bin-W, bin-1]). Eficiente via diff de cumsum.
    shots_cum = np.cumsum(shots_c, axis=1)
    shots_pre = np.concatenate([np.zeros_like(shots_cum[:, :1, :]),
                                 shots_cum[:, :-1, :]], axis=1)       # <=bin-1
    w = _SHOTS_WINDOW
    shots_before_window = np.concatenate([
        np.zeros_like(shots_cum[:, :w, :]),
        shots_cum[:, :REG_MINUTES - w, :],
    ], axis=1)                                                        # <=bin-1-w
    shots_win = shots_pre - shots_before_window                       # ventana (bin-w..bin-1)

    n_m = meta.height
    match_ids = np.repeat(meta["match_id"].to_numpy(), REG_MINUTES)
    minute_bin = np.tile(np.arange(1, REG_MINUTES + 1, dtype=np.int64), n_m)

    score_diff = (gc_pre[:, :, 0] - gc_pre[:, :, 1]).reshape(-1).astype(np.int64)
    red_diff   = (rc_pre[:, :, 0] - rc_pre[:, :, 1]).reshape(-1).astype(np.int64)
    shots_diff = (shots_win[:, :, 0] - shots_win[:, :, 1]).reshape(-1).astype(np.int64)

    elo_diff = (meta["elo_home_pre"].to_numpy()
                - meta["elo_away_pre"].to_numpy())
    comp_tier = meta["comp_tier"].to_numpy()
    y = meta["y"].to_numpy()

    X = pl.DataFrame({
        "match_id":           match_ids,
        "minute_bin":         minute_bin,
        "score_diff":         score_diff,
        "red_diff":           red_diff,
        "elo_diff":           np.repeat(elo_diff, REG_MINUTES).astype(np.float64),
        "comp_tier":          np.repeat(comp_tier, REG_MINUTES).astype(np.int64),
        "shots_diff_recent":  shots_diff,
        "y":                  np.repeat(y, REG_MINUTES).astype(np.int64),
    })

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        X.write_parquet(cache_path, compression="snappy", statistics=True)

    return X, {"n_samples": X.height, "n_matches": meta.height, "cached": False}


# ===========================================================================
#  SECCION 5 — Modelo bayesiano numpyro (ordered-logistic time-variables)
# ===========================================================================

def _wp_model(X_feat, minute_bin, y=None, n_bins: int = N_BINS):
    """Bayesian ordered-logistic 3-class con intercepts tiempo-variables.

    P(Y <= k | x, t) = sigmoid(-(linear(x,b) + alpha_k[t]))
    Con alpha_H[t] < alpha_D[t]. Random walk prior sobre alpha's en t.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    n_feat = X_feat.shape[1]

    # Coefs globales sobre features (compartidos todos los bins)
    beta = numpyro.sample("beta", dist.Normal(0.0, 1.0).expand([n_feat]).to_event(1))

    # Random-walk prior para alpha[t]: alpha_delta[t] ~ N(0, sigma_rw)
    sigma_rw = numpyro.sample("sigma_rw", dist.HalfNormal(0.5))
    alpha_H0 = numpyro.sample("alpha_H0", dist.Normal(0.0, 2.0))
    alpha_D0 = numpyro.sample("alpha_D0", dist.Normal(1.0, 2.0))

    dH = numpyro.sample("dH", dist.Normal(0.0, 1.0).expand([n_bins-1]).to_event(1)) * sigma_rw

    alphaH_t = jnp.concatenate([jnp.array([alpha_H0]), alpha_H0 + jnp.cumsum(dH)])
    gap_base = jnp.exp(alpha_D0 - alpha_H0)      # gap constante > 0 asegura alpha_D > alpha_H
    alphaD_t = alphaH_t + gap_base

    linpred = jnp.dot(X_feat, beta)               # (n,)
    aH = alphaH_t[minute_bin]
    aD = alphaD_t[minute_bin]
    # Logits acumulativos: P(Y <= 0 | ..) = sigmoid(aH - linpred)
    #                      P(Y <= 1 | ..) = sigmoid(aD - linpred)
    pH = 1.0 / (1.0 + jnp.exp(-(aH - linpred)))
    pD = 1.0 / (1.0 + jnp.exp(-(aD - linpred)))
    pA = 1.0 - pD
    pDraw = pD - pH
    probs = jnp.stack([pH, pDraw, pA], axis=-1).clip(1e-6, 1 - 1e-6)
    probs = probs / probs.sum(axis=-1, keepdims=True)

    with numpyro.plate("N", X_feat.shape[0]):
        numpyro.sample("obs", dist.Categorical(probs=probs), obs=y)


def fit_wp(X: pl.DataFrame, n_steps: int = 3000, seed: int = 0) -> dict:
    """Entrena WP bayesiano via SVI. Devuelve dict con params posterior."""
    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal

    feat_cols = ["score_diff", "red_diff", "elo_diff", "comp_tier",
                 "shots_diff_recent"]
    # Estandarizar (media 0, std 1) para eficiencia SVI
    feat_stats = {}
    X_np = X.select(feat_cols).to_numpy().astype(np.float32)
    for i, c in enumerate(feat_cols):
        mu = float(X_np[:, i].mean())
        sd = float(X_np[:, i].std() + 1e-8)
        X_np[:, i] = (X_np[:, i] - mu) / sd
        feat_stats[c] = (mu, sd)

    minute_bin_np = (X["minute_bin"].to_numpy() - 1).astype(np.int32)  # 0..89
    y_np = X["y"].to_numpy().astype(np.int32)

    guide = AutoNormal(_wp_model)
    svi = SVI(_wp_model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    state = svi.init(jax.random.PRNGKey(seed),
                     jnp.asarray(X_np), jnp.asarray(minute_bin_np), jnp.asarray(y_np),
                     n_bins=N_BINS)
    for i in range(n_steps):
        state, loss = svi.update(state, jnp.asarray(X_np),
                                 jnp.asarray(minute_bin_np), jnp.asarray(y_np),
                                 n_bins=N_BINS)
        if i % 500 == 0:
            print(f"  step {i:5d}  elbo_loss={float(loss):.2f}")
    params = svi.get_params(state)
    return {"params": params, "feat_stats": feat_stats, "feat_cols": feat_cols}


def _derive_alphas(params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extrae alpha_H_t, alpha_D_t (shape (N_BINS,)) desde params SVI."""
    aH0 = float(params["alpha_H0_auto_loc"])
    aD0 = float(params["alpha_D0_auto_loc"])
    sigma = float(np.exp(params["sigma_rw_auto_loc"]))
    dH = np.array(params["dH_auto_loc"], dtype=np.float32) * sigma
    alphaH_t = np.concatenate([[aH0], aH0 + np.cumsum(dH)])
    gap = float(np.exp(aD0 - aH0))
    alphaD_t = alphaH_t + gap
    return alphaH_t, alphaD_t


def predict_wp(feat_vector: dict, minute: int, fit_result: dict) -> tuple[float, float, float]:
    """Predice (p_home_win, p_draw, p_away_win) dada feature vector + minuto (1..90).

    Aplica temperature scaling si fit_result["temperature"] esta presente.
    """
    params = fit_result["params"]
    fs = fit_result["feat_stats"]
    fc = fit_result["feat_cols"]

    x = np.array([(feat_vector[c] - fs[c][0]) / fs[c][1] for c in fc], dtype=np.float32)
    beta = np.array(params["beta_auto_loc"], dtype=np.float32)
    alphaH_t, alphaD_t = _derive_alphas(params)

    t = max(0, min(minute - 1, N_BINS - 1))
    linpred = float(x @ beta)
    pH = 1.0 / (1.0 + np.exp(-(alphaH_t[t] - linpred)))
    pD = 1.0 / (1.0 + np.exp(-(alphaD_t[t] - linpred)))
    pA = 1.0 - pD
    pDraw = pD - pH
    probs = np.array([[pH, pDraw, pA]]).clip(1e-6, 1 - 1e-6)
    probs = probs / probs.sum(axis=1, keepdims=True)
    T = fit_result.get("temperature")
    probs = _apply_temperature(probs, T) if T is not None else probs
    return float(probs[0, 0]), float(probs[0, 1]), float(probs[0, 2])


def predict_wp_batch(X_df: pl.DataFrame, fit_result: dict) -> np.ndarray:
    """Predicciones vectorizadas (N, 3) con T aplicada si existe en fit_result."""
    params = fit_result["params"]
    fs = fit_result["feat_stats"]
    fc = fit_result["feat_cols"]
    beta = np.array(params["beta_auto_loc"], dtype=np.float32)
    alphaH_t, alphaD_t = _derive_alphas(params)

    X_np = X_df.select(fc).to_numpy().astype(np.float32)
    for i, c in enumerate(fc):
        X_np[:, i] = (X_np[:, i] - fs[c][0]) / fs[c][1]
    t_idx = np.clip(X_df["minute_bin"].to_numpy() - 1, 0, N_BINS - 1).astype(np.int32)
    linpred = X_np @ beta
    pH = 1.0 / (1.0 + np.exp(-(alphaH_t[t_idx] - linpred)))
    pD = 1.0 / (1.0 + np.exp(-(alphaD_t[t_idx] - linpred)))
    pA = 1.0 - pD
    pDraw = pD - pH
    probs = np.stack([pH, pDraw, pA], axis=1).clip(1e-6, 1 - 1e-6)
    probs = probs / probs.sum(axis=1, keepdims=True)
    T = fit_result.get("temperature")
    return _apply_temperature(probs, T) if T is not None else probs


def save_fit(fit_result: dict, path: Path | None = None) -> Path:
    """Serializa fit + T a disco (pickle)."""
    import pickle
    if path is None:
        path = _MODEL / "wp_regulation.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = {
        "params": {k: np.array(v) for k, v in fit_result["params"].items()},
        "feat_stats": fit_result["feat_stats"],
        "feat_cols": fit_result["feat_cols"],
        "temperature": fit_result.get("temperature"),
    }
    with open(path, "wb") as f:
        pickle.dump(serial, f)
    return path


def load_fit(path: Path | None = None) -> dict:
    """Deserializa fit guardado (incluye T si se calibro)."""
    import pickle
    if path is None:
        path = _MODEL / "wp_regulation.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


# -- Validacion Brier -------------------------------------------------------

def train_val_split(X: pl.DataFrame, val_frac: float = 0.2,
                    seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split por match_id (evita leakage entre samples del mismo partido)."""
    mids = np.array(X["match_id"].unique().to_list())
    rng = np.random.default_rng(seed)
    val_ids = set(rng.choice(mids, int(len(mids) * val_frac), replace=False).tolist())
    return (X.filter(~pl.col("match_id").is_in(list(val_ids))),
            X.filter(pl.col("match_id").is_in(list(val_ids))))


def brier_3class(probs: np.ndarray, y: np.ndarray) -> float:
    """Brier score multinomial 3 clases: mean sum (p_k - y_k_onehot)^2."""
    y_oh = np.zeros_like(probs)
    y_oh[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - y_oh) ** 2, axis=1)))


def evaluate_brier_per_minute(X_val: pl.DataFrame, fit_result: dict) -> pl.DataFrame:
    """Brier por bin de minuto sobre val set (incluye T si fit_result lo trae)."""
    probs = predict_wp_batch(X_val, fit_result)
    y = X_val["y"].to_numpy().astype(np.int32)
    out = X_val.select("minute_bin").with_columns([
        pl.Series("p_H", probs[:, 0]),
        pl.Series("p_D", probs[:, 1]),
        pl.Series("p_A", probs[:, 2]),
        pl.Series("y", y),
    ])
    out = out.with_columns(
        ((pl.col("p_H") - (pl.col("y") == 0).cast(pl.Float64)) ** 2
         + (pl.col("p_D") - (pl.col("y") == 1).cast(pl.Float64)) ** 2
         + (pl.col("p_A") - (pl.col("y") == 2).cast(pl.Float64)) ** 2).alias("brier_row")
    )
    return out.group_by("minute_bin").agg([
        pl.col("brier_row").mean().alias("brier"),
        pl.col("p_H").mean().alias("mean_pH"),
        pl.len().alias("n"),
    ]).sort("minute_bin")


# -- Temperature scaling ----------------------------------------------------

def _pff_shots_timeline_match(match_id: int, home_id: int) -> tuple[list[int], list[int]]:
    """Devuelve (minutos_shots_home, minutos_shots_away) en [0,90] para 1 match PFF."""
    from M01_loader_pff import list_shots
    s = list_shots(match_id).filter(pl.col("minute") < REG_MINUTES)
    sh = s.filter(pl.col("team_id") == home_id)["minute"].to_list()
    sa = s.filter(pl.col("team_id") != home_id)["minute"].to_list()
    return sh, sa


def _shots_recent_at(shots_minutes: list[int], m: int, w: int = _SHOTS_WINDOW) -> int:
    """Cuenta shots en la ventana [m-w, m-1]."""
    return sum(1 for x in shots_minutes if m - w <= x <= m - 1)


def build_wc22_groups_calib_matrix() -> pl.DataFrame:
    """Matrix de calibracion: 48 partidos fase grupos WC22 x 90 bins.

    Los 48 partidos de grupos NO son sagrados (solo los 16 KO lo son para
    el test final del PCJ). Los usamos para recalibrar T via temperature
    scaling. match_id de KO en PFF son los 10502-10517 (16 consecutivos);
    el resto son fase de grupos.
    """
    all_mids = list_event_match_ids()
    group_mids = [m for m in all_mids if m < 10500]
    rows = []
    for mid in group_mids:
        md = pff_meta(mid).row(0, named=True)
        home_id = md["home_team_id"]
        g = pff_goals_timeline(mid)   # excluye disallowed + shootout
        gh = g.filter(pl.col("scoring_team_id") == home_id)["minute"].to_list()
        ga = g.filter(pl.col("scoring_team_id") != home_id)["minute"].to_list()
        sh_pff, sa_pff = _pff_shots_timeline_match(mid, home_id)
        sh_90 = sum(1 for x in gh if x < REG_MINUTES)
        sa_90 = sum(1 for x in ga if x < REG_MINUTES)
        y = 0 if sh_90 > sa_90 else (2 if sh_90 < sa_90 else 1)
        for m in range(1, REG_MINUTES + 1):
            sd = sum(1 for x in gh if x < m) - sum(1 for x in ga if x < m)
            sdr = _shots_recent_at(sh_pff, m) - _shots_recent_at(sa_pff, m)
            rows.append({
                "match_id": mid, "minute_bin": m,
                "score_diff": sd, "red_diff": 0,
                "elo_diff": 0.0, "comp_tier": 3,
                "shots_diff_recent": sdr,
                "y": y,
            })
    return pl.DataFrame(rows)


def _apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """Aplica temperature scaling a una matriz (N, 3) de probabilidades."""
    if T is None or abs(T - 1.0) < 1e-6:
        return probs
    log_p = np.log(probs.clip(1e-10))
    scaled = log_p / T
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(fit_result: dict, X_calib: pl.DataFrame) -> float:
    """Optimiza T minimizando NLL sobre X_calib via scipy.

    Busca T en [0.3, 5.0]. Devuelve T optimo.
    """
    from scipy.optimize import minimize_scalar
    probs = predict_wp_batch(X_calib, fit_result)
    y = X_calib["y"].to_numpy().astype(np.int32)

    def nll(T):
        p_cal = _apply_temperature(probs, float(T))
        return -float(np.mean(np.log(p_cal[np.arange(len(y)), y].clip(1e-10))))

    res = minimize_scalar(nll, bounds=(0.3, 5.0), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x)


def mean_brier(X: pl.DataFrame, fit_result: dict) -> float:
    """Brier medio 3-class sobre el DataFrame dado."""
    probs = predict_wp_batch(X, fit_result)
    y = X["y"].to_numpy().astype(np.int32)
    return brier_3class(probs, y)


# ===========================================================================
#  SECCION 6 — ET (Poisson goal-rate) + Penaltis (Tijms 2019)
# ===========================================================================

def et_goal_rate_empirical() -> tuple[float, float]:
    """Tasa empirica goals/min en ET desde partidos con duration ExtraTime.

    Usa Wyscout duration=ExtraTime (3) + StatsBomb Euro20/Euro24 con ET (~15).
    Devuelve (lambda_home_per_min, lambda_away_per_min). Si el corpus es
    pequeno, devuelve tasas empiricas WC22-compatibles (0.025/0.020).
    """
    # Wyscout: duration ExtraTime (3 partidos). Contar scoreET per team.
    wm = load_wyscout_matches().filter(pl.col("duration") == "ExtraTime")
    total_et_goals = 0
    n_et_matches = 0
    for row in wm.iter_rows(named=True):
        td = row["teamsData"]
        for tid, data in td.items():
            if data is not None:
                total_et_goals += int(data.get("scoreET") or 0)
        n_et_matches += 1
    if n_et_matches < 5:
        # Fallback FiveThirtyEight-style empirical rates
        return 0.025, 0.020
    avg_goals_per_et = total_et_goals / n_et_matches
    rate = (avg_goals_per_et / 2) / ET_MINUTES  # split 50/50 home/away
    return rate, rate


def shootout_probability(p_home_scores: float = 0.75,
                         p_away_scores: float = 0.73) -> float:
    """P(home gana tanda) via Tijms (2019) — simulacion Monte Carlo con
    formato 5-tiros + sudden death. Devuelve P(home_wins).
    """
    rng = np.random.default_rng(42)
    n_sim = 10_000
    wins = 0
    for _ in range(n_sim):
        h = a = 0
        # 5 rondas regulares
        for i in range(5):
            h += int(rng.random() < p_home_scores)
            a += int(rng.random() < p_away_scores)
            # Early stop si uno no puede alcanzar al otro
            remaining = 5 - (i + 1)
            if h - a > remaining: break
            if a - h > remaining: break
        if h != a:
            wins += int(h > a)
            continue
        # Sudden death
        while True:
            ho = int(rng.random() < p_home_scores)
            ao = int(rng.random() < p_away_scores)
            if ho != ao:
                wins += int(ho > ao)
                break
    return wins / n_sim


# ===========================================================================
#  SECCION 7 — API publica compute_wp_per_minute + cache
# ===========================================================================

def _pff_goals_home_away(match_id: int) -> pl.DataFrame:
    """Goles PFF con (minute, scoring_side H/A) para compute WP."""
    md = pff_meta(match_id).row(0, named=True)
    home_id = md["home_team_id"]
    g = pff_goals_timeline(match_id)
    if g.height == 0:
        return pl.DataFrame(schema={"minute": pl.Int64, "scoring_side": pl.String})
    return g.select([
        pl.col("minute"),
        pl.when(pl.col("scoring_team_id") == home_id).then(pl.lit("H")).otherwise(pl.lit("A"))
          .alias("scoring_side"),
    ])


def _wp_et_poisson_batch(score_diff_90: np.ndarray, minute_et: np.ndarray,
                         lam_h: float, lam_a: float,
                         prob_shootout_home: float,
                         max_g: int = 6) -> np.ndarray:
    """Vectorizado: WP durante ET via Poisson outer product para N minutos.

    Args:
        score_diff_90 : (N,) score_diff antes de ET en cada instante.
        minute_et     : (N,) minuto dentro de ET (1..30).
    Returns:
        (N, 3) wp_home, wp_draw, wp_away. Draw siempre 0 (tanda resuelve).
    """
    from math import factorial
    minutes_left = np.clip(ET_MINUTES - minute_et, 0, ET_MINUTES).astype(np.float64)
    mean_h = lam_h * minutes_left                                  # (N,)
    mean_a = lam_a * minutes_left                                  # (N,)
    ks = np.arange(max_g)
    fact = np.array([factorial(k) for k in ks], dtype=np.float64)

    # P(k goles) truncado Poisson: (N, max_g)
    p_gh = np.exp(-mean_h[:, None]) * (mean_h[:, None] ** ks[None, :]) / fact[None, :]
    p_ga = np.exp(-mean_a[:, None]) * (mean_a[:, None] ** ks[None, :]) / fact[None, :]
    p_gh /= p_gh.sum(axis=1, keepdims=True)
    p_ga /= p_ga.sum(axis=1, keepdims=True)

    # joint (N, max_g, max_g): joint[i, h, a] = p_gh[i,h] * p_ga[i,a]
    joint = p_gh[:, :, None] * p_ga[:, None, :]
    diff  = ks[None, :, None] - ks[None, None, :]                   # (1, max_g, max_g)
    final = score_diff_90[:, None, None] + diff                     # (N, max_g, max_g)
    p_H_reg = np.where(final > 0, joint, 0.0).sum(axis=(1, 2))
    p_A_reg = np.where(final < 0, joint, 0.0).sum(axis=(1, 2))
    p_D_reg = 1.0 - p_H_reg - p_A_reg

    p_H = p_H_reg + p_D_reg * prob_shootout_home
    p_A = p_A_reg + p_D_reg * (1.0 - prob_shootout_home)
    p_D = np.zeros_like(p_H)
    return np.stack([p_H, p_D, p_A], axis=1)


def compute_wp_per_minute(match_id: int, fit_result: dict,
                          elo_diff: float = 0.0,
                          comp_tier: int = 3,
                          lam_et_h: float | None = None,
                          lam_et_a: float | None = None,
                          prob_shootout_home: float = 0.5,
                          group_ctx: dict | None = None,
                          n_sim_groups: int = 1500) -> pl.DataFrame:
    """WP por minuto para 1 partido PFF: 0-90 bayesiano + 91-120 Poisson ET.

    Al final de ET (si empate persiste) aplica prob_shootout_home.

    Args:
        match_id: ID PFF.
        fit_result: fit bayesiano entrenado (de fit_wp).
        elo_diff: elo_home - elo_away (0.0 por defecto, equipos comparables).
        comp_tier: 3 = WC.
        lam_et_h/lam_et_a: tasas ET por minuto. Si None, deriva de empirical.
        prob_shootout_home: P(home gana tanda). 0.5 default (simetrico).

    Returns:
        DataFrame (match_id, minute 1..120, wp_home, wp_draw, wp_away, leverage,
        elim_prox, score_diff, phase) con phase in {regulation, extra_time}.
    """
    if lam_et_h is None or lam_et_a is None:
        lh, la = et_goal_rate_empirical()
        lam_et_h = lh if lam_et_h is None else lam_et_h
        lam_et_a = la if lam_et_a is None else lam_et_a

    goals = _pff_goals_home_away(match_id)
    gh = goals.filter(pl.col("scoring_side") == "H")["minute"].to_list()
    ga = goals.filter(pl.col("scoring_side") == "A")["minute"].to_list()

    # Shots PFF para shots_diff_recent
    md = pff_meta(match_id).row(0, named=True)
    sh_pff, sa_pff = _pff_shots_timeline_match(match_id, md["home_team_id"])

    went_to_et = md["home_team_start_left_et"] is not None

    # --- Regulacion 1..90 vectorizado ---
    mins_reg = np.arange(1, REG_MINUTES + 1)
    gh_arr = np.array(gh, dtype=np.int64)
    ga_arr = np.array(ga, dtype=np.int64)
    sd_reg = np.array([(gh_arr < m).sum() - (ga_arr < m).sum()
                       for m in mins_reg], dtype=np.int64)
    sdr_reg = np.array([_shots_recent_at(sh_pff, int(m))
                         - _shots_recent_at(sa_pff, int(m)) for m in mins_reg],
                        dtype=np.int64)
    X_reg = pl.DataFrame({
        "minute_bin":        mins_reg,
        "score_diff":        sd_reg,
        "red_diff":          np.zeros(REG_MINUTES, dtype=np.int64),
        "elo_diff":          np.full(REG_MINUTES, elo_diff, dtype=np.float64),
        "comp_tier":         np.full(REG_MINUTES, comp_tier, dtype=np.int64),
        "shots_diff_recent": sdr_reg,
    })
    probs_reg = predict_wp_batch(X_reg, fit_result)
    probs_plus = predict_wp_batch(
        X_reg.with_columns((pl.col("score_diff") + 1).alias("score_diff")),
        fit_result,
    )
    lev_reg = np.abs(probs_plus[:, 0] - probs_reg[:, 0])

    reg_df = pl.DataFrame({
        "match_id":   np.full(REG_MINUTES, match_id, dtype=np.int64),
        "minute":     mins_reg,
        "wp_home":    probs_reg[:, 0],
        "wp_draw":    probs_reg[:, 1],
        "wp_away":    probs_reg[:, 2],
        "leverage":   lev_reg,
        "score_diff": sd_reg,
        "phase":      ["regulation"] * REG_MINUTES,
    })

    if went_to_et:
        mins_et = np.arange(REG_MINUTES + 1, REG_MINUTES + ET_MINUTES + 1)
        minute_et = mins_et - REG_MINUTES
        sd_et = np.array([(gh_arr < m).sum() - (ga_arr < m).sum()
                          for m in mins_et], dtype=np.int64)
        probs_et = _wp_et_poisson_batch(sd_et, minute_et,
                                         lam_et_h, lam_et_a, prob_shootout_home)
        probs_et_plus = _wp_et_poisson_batch(sd_et + 1, minute_et,
                                              lam_et_h, lam_et_a, prob_shootout_home)
        lev_et = np.abs(probs_et_plus[:, 0] - probs_et[:, 0])
        et_df = pl.DataFrame({
            "match_id":   np.full(ET_MINUTES, match_id, dtype=np.int64),
            "minute":     mins_et,
            "wp_home":    probs_et[:, 0],
            "wp_draw":    probs_et[:, 1],
            "wp_away":    probs_et[:, 2],
            "leverage":   lev_et,
            "score_diff": sd_et,
            "phase":      ["extra_time"] * ET_MINUTES,
        })
        df = pl.concat([reg_df, et_df])
    else:
        df = reg_df

    # Elimination proximity:
    #  - Partidos de GRUPOS (match_id < 10500): Monte Carlo del grupo, considera
    #    standings reales, partidos simultaneos J3 y goles restantes a simular.
    #  - Partidos KO (match_id >= 10500): formula analitica P(no ganar).
    is_group_stage = (group_ctx is not None
                      and match_id in group_ctx["match_to_group"])

    if is_group_stage:
        ep = _compute_group_elim_prox_for_match(
            match_id, group_ctx, n_sim=n_sim_groups
        )
        # Para minutos de ET (no aplicable en grupos) rellena con NaN-like.
        df = df.join(ep, on="minute", how="left")
    else:
        df = df.with_columns([
            (pl.col("wp_away") + 0.5 * pl.col("wp_draw")).alias("elim_prox_home"),
            (pl.col("wp_home") + 0.5 * pl.col("wp_draw")).alias("elim_prox_away"),
        ])
    return df


def cache_all_wp(fit_result: dict, overwrite: bool = False,
                 group_ctx: dict | None = None) -> Path:
    """Aplica compute_wp_per_minute a los 64 PFF y persiste tabla unificada.

    Si group_ctx se pasa, los 48 partidos de fase de grupos reciben elim_prox
    via Monte Carlo del grupo (considerando simultaneos J3 y standings en vivo).
    Los 16 KO usan formula analitica simple (1 - wp_ganar).
    """
    out_path = _DERIVED / "per_minute.parquet"
    if out_path.exists() and not overwrite:
        return out_path
    if group_ctx is None:
        group_ctx = build_wc22_group_context()
    dfs = []
    for mid in list_event_match_ids():
        dfs.append(compute_wp_per_minute(mid, fit_result, group_ctx=group_ctx))
    big = pl.concat(dfs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    big.write_parquet(out_path, compression="snappy", statistics=True)
    return out_path


# ===========================================================================
#  SECCION 8 — Elimination proximity via Monte Carlo del grupo (WC22)
# ===========================================================================
#
#  Operacional: para cada (match_id, minute) de fase de grupos, P(equipo
#  clasifica como top-2 del grupo) se estima por simulacion. Tiene en cuenta:
#    - Resultados FINALES de partidos completados antes de la hora actual.
#    - Score PARCIAL del propio partido en curso.
#    - Score PARCIAL del partido SIMULTANEO (jornada 3, misma fecha+hora).
#    - Simulacion Poisson de los goles restantes en partidos en vivo + futuros.
#    - Criterios FIFA: pts -> GD -> GF. Head-to-head se ignora (edge case raro;
#      documentado como limitacion menor).
#  elim_prox_{home,away} = 1 - P(clasifica).
#
#  Caso icónico cubierto: ESP-JPN 2022-12-01 19:00 con GER-CRC en paralelo
#  (ambos grupo E). En el min 70, ESP perdia 1-2 y GER ganaba 2-1 a CRC:
#  P(ESP clasifica) cae a ~0.3, P(JPN clasifica) sube a ~0.8, P(GER) a ~0.55.

_WC22_GROUPS: dict[str, list[str]] = {
    "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
    "B": ["England", "Iran", "United States", "Wales"],
    "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
    "D": ["France", "Australia", "Denmark", "Tunisia"],
    "E": ["Spain", "Costa Rica", "Germany", "Japan"],
    "F": ["Belgium", "Canada", "Morocco", "Croatia"],
    "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
    "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
}


def build_wc22_group_context() -> dict:
    """Construye contexto fijo del torneo para simular eliminacion por grupos.

    Returns dict:
      - team_id_to_group : {int -> 'A'..'H'}
      - team_id_to_name  : {int -> str}
      - group_teams      : {'A'..'H' -> [team_id, team_id, team_id, team_id]}
      - group_matches    : {'A'..'H' -> [{match_id, home_id, away_id,
                                         date, home_final, away_final, week}, ...]}
      - match_to_group   : {match_id -> 'A'..'H'}
      - match_dates      : {match_id -> str ISO date}
      - goals_by_match   : {match_id -> list[(minute, 'H'|'A')]}
    """
    md = pff_meta()
    name_to_group = {name: g for g, team_names in _WC22_GROUPS.items()
                     for name in team_names}
    team_id_to_name: dict[int, str] = {}
    team_id_to_group: dict[int, str] = {}
    for r in md.iter_rows(named=True):
        team_id_to_name[r["home_team_id"]] = r["home_team_name"]
        team_id_to_name[r["away_team_id"]] = r["away_team_name"]
        for tid, tname in [(r["home_team_id"], r["home_team_name"]),
                            (r["away_team_id"], r["away_team_name"])]:
            if tname in name_to_group:
                team_id_to_group[tid] = name_to_group[tname]

    group_teams: dict[str, list[int]] = {g: [] for g in _WC22_GROUPS}
    for tid, g in team_id_to_group.items():
        if tid not in group_teams[g]:
            group_teams[g].append(tid)
    for g in group_teams:
        group_teams[g].sort()   # orden determinista

    # Resultados FINALES de los 48 partidos de fase grupos (week 1-3):
    # usamos M03 goals_timeline (SB como ground truth, ya resuelve own-goals).
    group_matches: dict[str, list[dict]] = {g: [] for g in _WC22_GROUPS}
    match_to_group: dict[int, str] = {}
    match_dates: dict[int, str] = {}
    goals_by_match: dict[int, list[tuple[int, str]]] = {}

    md_filtered = md.filter(pl.col("match_id").is_in(
        [mid for mid in md["match_id"].to_list() if mid < 10500]
    ))

    for r in md_filtered.iter_rows(named=True):
        mid = r["match_id"]
        home_id = r["home_team_id"]
        away_id = r["away_team_id"]
        g = pff_goals_timeline(mid)
        # Goles: para H/A perspectiva (lado de home team)
        home_final = int(g.filter(pl.col("scoring_team_id") == home_id).height)
        away_final = int(g.filter(pl.col("scoring_team_id") == away_id).height)
        grp = team_id_to_group.get(home_id)
        if grp is None:
            continue   # KO o no-grupo
        match_to_group[mid] = grp
        match_dates[mid] = r["date"]
        group_matches[grp].append({
            "match_id": mid, "home_id": home_id, "away_id": away_id,
            "date": r["date"], "home_final": home_final, "away_final": away_final,
            "week": r["week"],
        })
        # timeline de goles con side H/A para score_at_minute lookups
        goals_by_match[mid] = [
            (int(row["minute"]),
             "H" if row["scoring_team_id"] == home_id else "A")
            for row in g.iter_rows(named=True)
        ]

    # Orden cronologico dentro de cada grupo
    for g in group_matches:
        group_matches[g].sort(key=lambda m: m["date"])

    return {
        "team_id_to_group": team_id_to_group,
        "team_id_to_name": team_id_to_name,
        "group_teams": group_teams,
        "group_matches": group_matches,
        "match_to_group": match_to_group,
        "match_dates": match_dates,
        "goals_by_match": goals_by_match,
    }


def _score_at_minute(match_id: int, minute: int, group_ctx: dict) -> tuple[int, int]:
    """(score_home, score_away) del partido `match_id` en el minuto `minute`."""
    goals = group_ctx["goals_by_match"].get(match_id, [])
    sh = sum(1 for (m, s) in goals if m < minute and s == "H")
    sa = sum(1 for (m, s) in goals if m < minute and s == "A")
    return sh, sa


# Tasa empirica goles/min fase de grupos WC22 (se computa on-demand).
_WC22_LAM_CACHE: dict[str, float] = {}


def _wc22_group_goal_rate(group_ctx: dict) -> float:
    """goles/min/equipo empirico de los 48 partidos de fase grupos WC22."""
    if "rate" in _WC22_LAM_CACHE:
        return _WC22_LAM_CACHE["rate"]
    total_goals = 0
    n_matches = 0
    for g in group_ctx["group_matches"].values():
        for m in g:
            total_goals += m["home_final"] + m["away_final"]
            n_matches += 1
    # rate por equipo por minuto: total_goles / (n_matches * 90 * 2)
    rate = total_goals / (n_matches * REG_MINUTES * 2) if n_matches > 0 else 0.015
    _WC22_LAM_CACHE["rate"] = rate
    return rate


def p_qualifies(team_id: int, match_id: int, minute: int,
                score_now_home: int, score_now_away: int,
                group_ctx: dict,
                n_sim: int = 2000, seed: int | None = None,
                rng: np.random.Generator | None = None) -> float:
    """P(team_id clasifica top-2 del grupo) via Monte Carlo del grupo entero.

    Args:
        team_id         : equipo a evaluar.
        match_id        : partido en curso (contiene al team_id).
        minute          : minuto actual (1..90).
        score_now_home  : goles del equipo local del match_id al minuto.
        score_now_away  : goles del equipo visitante al minuto.
        group_ctx       : estructura de build_wc22_group_context().
        n_sim           : simulaciones Monte Carlo (default 2000).
        rng             : numpy Generator (para evitar overhead init).

    Returns:
        Float en [0, 1]. 1.0 = clasifica seguro; 0.0 = eliminado.
    """
    group = group_ctx["team_id_to_group"][team_id]
    teams_in_group = group_ctx["group_teams"][group]
    team_to_idx = {t: i for i, t in enumerate(teams_in_group)}
    my_idx = team_to_idx[team_id]

    current_date = group_ctx["match_dates"][match_id]
    lam = _wc22_group_goal_rate(group_ctx)
    if rng is None:
        rng = np.random.default_rng(seed)

    # Partials: uno por cada uno de los 6 partidos del grupo
    # (h_idx, a_idx, base_h, base_a, min_to_sim)
    partials = []
    for m in group_ctx["group_matches"][group]:
        h_idx = team_to_idx[m["home_id"]]
        a_idx = team_to_idx[m["away_id"]]
        if m["match_id"] == match_id:
            partials.append((h_idx, a_idx,
                             int(score_now_home), int(score_now_away),
                             REG_MINUTES - minute))
        elif m["date"] < current_date:
            partials.append((h_idx, a_idx,
                             m["home_final"], m["away_final"], 0))
        elif m["date"] == current_date:
            # Simultaneo J3: score parcial en ese mismo minuto
            sh, sa = _score_at_minute(m["match_id"], minute, group_ctx)
            partials.append((h_idx, a_idx, sh, sa, REG_MINUTES - minute))
        else:
            partials.append((h_idx, a_idx, 0, 0, REG_MINUTES))   # futuro

    n_m = len(partials)
    bases = np.array([[p[2], p[3]] for p in partials], dtype=np.int32)  # (6, 2)
    mins_left = np.array([p[4] for p in partials], dtype=np.int32)     # (6,)
    h_arr = np.array([p[0] for p in partials], dtype=np.int32)
    a_arr = np.array([p[1] for p in partials], dtype=np.int32)

    # Extras: (n_sim, 6, 2)
    lams = (mins_left * lam).astype(np.float32)                         # (6,)
    lams_3d = np.broadcast_to(lams[None, :, None], (n_sim, n_m, 2))
    extras = rng.poisson(lams_3d)
    finals = bases[None, :, :] + extras                                  # (n_sim, 6, 2)
    fh = finals[:, :, 0]; fa = finals[:, :, 1]

    # Tabla por sim
    pts = np.zeros((n_sim, 4), dtype=np.int32)
    gd  = np.zeros((n_sim, 4), dtype=np.int32)
    gf  = np.zeros((n_sim, 4), dtype=np.int32)
    home_win = fh > fa; away_win = fa > fh; draw = ~(home_win | away_win)
    for mi in range(n_m):
        h, a = h_arr[mi], a_arr[mi]
        pts[home_win[:, mi], h] += 3
        pts[away_win[:, mi], a] += 3
        pts[draw[:, mi], h] += 1
        pts[draw[:, mi], a] += 1
        gd[:, h] += fh[:, mi] - fa[:, mi]
        gd[:, a] += fa[:, mi] - fh[:, mi]
        gf[:, h] += fh[:, mi]
        gf[:, a] += fa[:, mi]

    # Ranking: score = pts*10000 + (gd+50)*100 + (gf+50) para single-sort desc
    score = pts * 10000 + (gd + 50) * 100 + (gf + 50)                    # (n_sim, 4)
    order = np.argsort(-score, axis=1)                                   # desc
    in_top2 = (order[:, 0] == my_idx) | (order[:, 1] == my_idx)
    return float(in_top2.mean())


# ===========================================================================
#  Integracion en compute_wp_per_minute (override para grupos)
# ===========================================================================

def _compute_group_elim_prox_for_match(match_id: int, group_ctx: dict,
                                       n_sim: int = 1500,
                                       seed: int = 42) -> pl.DataFrame:
    """Devuelve (minute, elim_prox_home, elim_prox_away) para los 90 min del partido.

    Solo para partidos de fase de grupos WC22 (match_id en group_ctx).
    """
    grp = group_ctx["match_to_group"][match_id]
    # home_id y away_id del partido:
    m_entry = next(m for m in group_ctx["group_matches"][grp]
                   if m["match_id"] == match_id)
    home_id = m_entry["home_id"]
    away_id = m_entry["away_id"]
    goals = group_ctx["goals_by_match"][match_id]
    rng = np.random.default_rng(seed)

    rows = []
    for m in range(1, REG_MINUTES + 1):
        sh = sum(1 for (mm, s) in goals if mm < m and s == "H")
        sa = sum(1 for (mm, s) in goals if mm < m and s == "A")
        p_home = p_qualifies(home_id, match_id, m, sh, sa, group_ctx,
                             n_sim=n_sim, rng=rng)
        p_away = p_qualifies(away_id, match_id, m, sh, sa, group_ctx,
                             n_sim=n_sim, rng=rng)
        rows.append({"minute": m,
                     "elim_prox_home": 1.0 - p_home,
                     "elim_prox_away": 1.0 - p_away})
    return pl.DataFrame(rows)


# -- Sanity inline ---------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=== M04_wp sanity ===")

    # 1. Training matrix
    t0 = time.time()
    X, info = build_training_matrix(cache=True)
    print(f"training matrix: {info} en {time.time()-t0:.1f}s")
    print(f"  y distrib: {X.group_by('y').len().sort('y').to_dicts()}")
    print(f"  score_diff         range: [{X['score_diff'].min()}, {X['score_diff'].max()}]")
    print(f"  red_diff           range: [{X['red_diff'].min()}, {X['red_diff'].max()}]")
    print(f"  elo_diff           range: [{X['elo_diff'].min():.0f}, {X['elo_diff'].max():.0f}]")
    print(f"  shots_diff_recent  range: [{X['shots_diff_recent'].min()}, "
          f"{X['shots_diff_recent'].max()}]  mean_abs={X['shots_diff_recent'].abs().mean():.2f}")

    # 2. Split train/val + fit
    X_tr, X_val = train_val_split(X, val_frac=0.2, seed=42)
    print(f"\ntrain={X_tr.height:,} val={X_val.height:,}")

    fit_path = _MODEL / "wp_regulation.pkl"
    if fit_path.exists():
        fit = load_fit(fit_path)
        print("fit cargado desde cache")
    else:
        t0 = time.time()
        fit = fit_wp(X_tr, n_steps=3000)
        print(f"SVI fit en {time.time()-t0:.1f}s")
        save_fit(fit)

    # 3. Brier pre-calibration
    brier_pre = mean_brier(X_val, fit)
    print(f"\nBrier pre-calib sobre val: {brier_pre:.4f}")

    # 4. Temperature scaling sobre 48 partidos grupos WC22
    t0 = time.time()
    X_calib = build_wc22_groups_calib_matrix()
    T_opt = fit_temperature(fit, X_calib)
    fit["temperature"] = T_opt
    save_fit(fit)  # Persiste T
    print(f"calib matrix: {X_calib.height} samples ({X_calib['match_id'].n_unique()} partidos) "
          f"en {time.time()-t0:.1f}s")
    print(f"T optimo: {T_opt:.4f}")

    # 5. Brier post-calibration
    brier_post = mean_brier(X_val, fit)
    print(f"Brier post-calib sobre val: {brier_post:.4f} (delta {brier_post-brier_pre:+.4f})")
    brier_df = evaluate_brier_per_minute(X_val, fit)
    tenbin = brier_df.with_columns((pl.col("minute_bin") // 10 * 10).alias("bin10"))
    print("brier post-calib por 10-min bins:")
    print(tenbin.group_by("bin10").agg([pl.col("brier").mean(), pl.col("n").sum()]).sort("bin10"))

    # 6. Predict sanity checks (post-calib)
    print()
    for (sd, minute, elo, tier, label) in [
        (0,  1,   0.0, 1, "0-0 min1 liga"),
        (1, 89, 100.0, 3, "1-0 min89 WC fav"),
        (-1,89, -100.0,3, "0-1 min89 WC debil"),
        (0, 45,   0.0, 3, "0-0 HT neutral"),
    ]:
        p = predict_wp({"score_diff": sd, "red_diff": 0, "elo_diff": elo,
                        "comp_tier": tier, "shots_diff_recent": 0},
                        minute, fit)
        print(f"  {label:<25} H={p[0]:.3f} D={p[1]:.3f} A={p[2]:.3f}")

    # 7. compute_wp_per_minute ET sanity (NED-ARG 10511)
    mid_et = 10511
    print(f"\ncompute_wp_per_minute({mid_et}) [NED-ARG con ET + penaltis]:")
    wp = compute_wp_per_minute(mid_et, fit)
    print(f"  filas: {wp.height}, phases: {sorted(wp['phase'].unique().to_list())}")
    key_minutes = wp.filter(pl.col("minute").is_in([1, 45, 83, 90, 105, 120]))
    print(key_minutes.select(["minute","phase","score_diff","wp_home","wp_draw","wp_away","leverage"]))

    # 8. Cache all 64 PFF
    print()
    t0 = time.time()
    out = cache_all_wp(fit, overwrite=True)
    print(f"cache_all_wp -> {out} en {time.time()-t0:.1f}s")
    big = pl.read_parquet(out)
    print(f"  total filas: {big.height:,}  matches cacheados: {big['match_id'].n_unique()}/64")
    # Sanity: todos los 64 con filas, al menos 90 bins cada uno
    per_match = big.group_by("match_id").len().sort("len")
    print(f"  bins por partido min/max: [{per_match['len'].min()}, {per_match['len'].max()}]")
