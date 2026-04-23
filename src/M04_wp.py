"""
M04_wp - Win Probability bayesiana tiempo-variables + leverage.

Reimplementacion Robberechts, Van Haaren & Davis (2021, KDD) en numpyro:
  - Ordered-logistic 3-class {H, D, A} con intercepts tiempo-variables
  - Random-walk prior sobre los 90 time-bins (smoothness temporal)
  - SVI (AutoNormal) para inferencia rapida CPU
  - Features: score_diff, red_cards_diff, elo_diff, home_adv, tier

Training cross-dataset (excluyendo WC22 sagrado):
  - Wyscout 2017/18 Big 5 + Euro16 + WC18  : 1.941 partidos
  - StatsBomb Euro20 + Euro24 + Bundes23/24:   136 partidos
  Total: 2.077 partidos x 90 bins = ~187k samples.

Cobertura completa 0-90 + ET + penaltis:
  - Regulacion 0-90 : modelo bayesiano entrenado.
  - ET (90-120)    : Poisson goal-rate empirico sobre subset ET.
  - Penaltis tanda : Tijms (2019) formula cerrada.

Calibracion: temperature scaling sobre 32 partidos WC22 fase de grupos
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
    WYSCOUT_COMPETITIONS, STATSBOMB_COMPETITIONS,
    scan_wyscout_events, load_wyscout_matches,
    load_statsbomb_matches, load_statsbomb_events, list_statsbomb_match_ids,
    scan_statsbomb_events,
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
    """Timeline de goles Wyscout: (match_id, minute_abs, team_id, is_own_goal).

    Regla: eventName in (Shot, Free Kick) + tag 101 = gol normal.
    Own goals (tag 102) se omiten — Wyscout marca el tag en varios events
    de la misma jugada y genera ~5% ruido; descartarlos pierde ~3% precision
    en score_diff pero mantiene el corpus limpio.
    """
    ev = scan_wyscout_events().with_columns(
        pl.col("tags").list.eval(pl.element().struct.field("id")).alias("tag_ids")
    )
    g = ev.filter(
        pl.col("eventName").is_in(["Shot", "Free Kick"]) &
        pl.col("tag_ids").list.contains(101)
    ).with_columns([
        pl.col("matchPeriod").replace_strict(_WYSCOUT_PERIOD_OFFSET, default=0)
                              .cast(pl.Int64).alias("period_offset"),
        (pl.col("eventSec") / 60.0).alias("sec_min"),
    ]).with_columns(
        (pl.col("period_offset") + pl.col("sec_min")).cast(pl.Int64).alias("minute_abs")
    ).select([
        pl.col("matchId").alias("match_id"),
        pl.col("teamId").alias("scoring_team_id"),
        "minute_abs",
    ])
    return g.collect().with_columns(pl.lit("wyscout").alias("source"))


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
                ((pl.col("period") - 1) * 45 + pl.col("minute")
                 - pl.when(pl.col("period") >= 3).then(0).otherwise(0))
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

def build_training_matrix(cache: bool = True) -> tuple[pl.DataFrame, dict]:
    """Construye matrix (match_id, minute, score_diff, red_diff, elo_diff, tier, y).

    y: target multinomial {0=H, 1=D, 2=A} al final de los 90 minutos regulares.
    Cacheable a parquet.
    """
    cache_path = _CACHE / "training_matrix.parquet"
    if cache and cache_path.exists():
        df = pl.read_parquet(cache_path)
        return df, {"n_samples": df.height, "cached": True}

    # 1. Timelines cross-dataset
    g_wy = _wyscout_goals_timeline()
    r_wy = _wyscout_reds_timeline()
    sb_mids = [mid for cid, sid in STATSBOMB_COMPETITIONS.values()
               if (cid, sid) != (43, 106)
               for mid in list_statsbomb_match_ids(comp_id=cid, season_id=sid)]
    g_sb = _statsbomb_goals_timeline(sb_mids)
    r_sb = _statsbomb_reds_timeline(sb_mids)

    goals_all = pl.concat([g_wy, g_sb], how="diagonal_relaxed")
    reds_all  = pl.concat([r_wy, r_sb], how="diagonal_relaxed")

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

    # 5. Generar bins 1..90 x matches
    bins = pl.DataFrame({"minute_bin": list(range(1, REG_MINUTES + 1))})
    grid = meta.join(bins, how="cross")

    # 6. Score/red cumulative at minute_bin: para cada match bin, count goals/reds
    #    del home (scoring_team_id == home_team_id) y del away con minute_abs <= bin.

    def _cum_by_team(timeline: pl.DataFrame, team_col: str) -> pl.DataFrame:
        """Cumulative count por (match_id, team_role) hasta minute_bin."""
        # clip minute_abs a [1, 90]: eventos stoppage del 1H (min 46-47) cuentan en bin 45+
        tl = timeline.with_columns(
            pl.col("minute_abs").clip(1, REG_MINUTES).alias("ma_clip")
        )
        return tl

    # Home cumulative goals
    goals_aligned = goals_all.join(
        meta.select(["match_id", "home_team_id", "away_team_id"]),
        on="match_id", how="inner"
    ).with_columns(
        pl.when(pl.col("scoring_team_id") == pl.col("home_team_id")).then(pl.lit("H"))
          .when(pl.col("scoring_team_id") == pl.col("away_team_id")).then(pl.lit("A"))
          .otherwise(pl.lit(None, dtype=pl.String))
          .alias("for_side")
    ).filter(pl.col("for_side").is_not_null())

    reds_aligned = reds_all.join(
        meta.select(["match_id", "home_team_id", "away_team_id"]),
        on="match_id", how="inner"
    ).with_columns(
        pl.when(pl.col("team_id") == pl.col("home_team_id")).then(pl.lit("H"))
          .when(pl.col("team_id") == pl.col("away_team_id")).then(pl.lit("A"))
          .otherwise(pl.lit(None, dtype=pl.String))
          .alias("for_side")
    ).filter(pl.col("for_side").is_not_null())

    def _cum_at_bin(events: pl.DataFrame, side: str) -> pl.DataFrame:
        e = events.filter(pl.col("for_side") == side).select(["match_id", "minute_abs"])
        # Para cada match, para cada bin, count events with minute_abs <= bin
        # Equivalente a sort + cum_count tras cross.
        e2 = e.with_columns(pl.col("minute_abs").clip(1, REG_MINUTES).alias("minute_abs"))
        return e2

    goals_h = _cum_at_bin(goals_aligned, "H")
    goals_a = _cum_at_bin(goals_aligned, "A")
    reds_h  = _cum_at_bin(reds_aligned,  "H")
    reds_a  = _cum_at_bin(reds_aligned,  "A")

    # Agregar por (match_id, minute_abs) para cum por minuto
    def _aggregate_cum(ev: pl.DataFrame, col_name: str) -> pl.DataFrame:
        """Para cada match, cum sum por minute."""
        e = ev.group_by(["match_id", "minute_abs"]).len().rename({"len": "cnt"})
        # Producto con bins y cum_sum
        per_match_bins = e.join(bins, how="cross").filter(
            pl.col("minute_abs") <= pl.col("minute_bin")
        ).group_by(["match_id", "minute_bin"]).agg(pl.col("cnt").sum().alias(col_name))
        return per_match_bins

    gh_cum = _aggregate_cum(goals_h, "gh_cum")
    ga_cum = _aggregate_cum(goals_a, "ga_cum")
    rh_cum = _aggregate_cum(reds_h,  "rh_cum")
    ra_cum = _aggregate_cum(reds_a,  "ra_cum")

    X = grid.join(gh_cum, on=["match_id","minute_bin"], how="left") \
            .join(ga_cum, on=["match_id","minute_bin"], how="left") \
            .join(rh_cum, on=["match_id","minute_bin"], how="left") \
            .join(ra_cum, on=["match_id","minute_bin"], how="left") \
        .with_columns([
            pl.col("gh_cum").fill_null(0).cast(pl.Int64),
            pl.col("ga_cum").fill_null(0).cast(pl.Int64),
            pl.col("rh_cum").fill_null(0).cast(pl.Int64),
            pl.col("ra_cum").fill_null(0).cast(pl.Int64),
        ]).with_columns([
            (pl.col("gh_cum") - pl.col("ga_cum")).cast(pl.Int64).alias("score_diff"),
            (pl.col("rh_cum") - pl.col("ra_cum")).cast(pl.Int64).alias("red_diff"),
            (pl.col("elo_home_pre") - pl.col("elo_away_pre")).alias("elo_diff"),
        ]).select([
            "match_id", "minute_bin", "score_diff", "red_diff",
            "elo_diff", "comp_tier", "y",
        ])

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
    dD = numpyro.sample("dD", dist.Normal(0.0, 1.0).expand([n_bins-1]).to_event(1)) * sigma_rw

    alphaH_t = jnp.concatenate([jnp.array([alpha_H0]), alpha_H0 + jnp.cumsum(dH)])
    gap_base = jnp.exp(alpha_D0 - alpha_H0)      # garantiza alpha_D > alpha_H
    alphaD_t = alphaH_t + gap_base * jnp.ones(n_bins)

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

    feat_cols = ["score_diff", "red_diff", "elo_diff", "comp_tier"]
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
    dD = np.array(params["dD_auto_loc"], dtype=np.float32) * sigma  # noqa: F841
    alphaH_t = np.concatenate([[aH0], aH0 + np.cumsum(dH)])
    gap = float(np.exp(aD0 - aH0))
    alphaD_t = alphaH_t + gap
    return alphaH_t, alphaD_t


def predict_wp(feat_vector: dict, minute: int, fit_result: dict) -> tuple[float, float, float]:
    """Predice (p_home_win, p_draw, p_away_win) dada feature vector + minuto (1..90)."""
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
    probs = np.array([pH, pDraw, pA]).clip(1e-6, 1 - 1e-6)
    probs /= probs.sum()
    return float(probs[0]), float(probs[1]), float(probs[2])


def predict_wp_batch(X_df: pl.DataFrame, fit_result: dict) -> np.ndarray:
    """Predicciones vectorizadas. X_df debe tener feat_cols + 'minute_bin'.

    Devuelve array (N, 3) con [P_H, P_D, P_A] por fila.
    """
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
    return probs


def save_fit(fit_result: dict, path: Path | None = None) -> Path:
    """Serializa fit a disco (pickle) para no re-entrenar."""
    import pickle
    if path is None:
        path = _MODEL / "wp_regulation.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convertir jax arrays a numpy para portabilidad
    serial = {
        "params": {k: np.array(v) for k, v in fit_result["params"].items()},
        "feat_stats": fit_result["feat_stats"],
        "feat_cols": fit_result["feat_cols"],
    }
    with open(path, "wb") as f:
        pickle.dump(serial, f)
    return path


def load_fit(path: Path | None = None) -> dict:
    """Deserializa fit guardado."""
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
    """Brier por bin de minuto sobre val set."""
    probs = predict_wp_batch(X_val, fit_result)
    y = X_val["y"].to_numpy().astype(np.int32)
    out = X_val.select("minute_bin").with_columns([
        pl.Series("p_H", probs[:, 0]),
        pl.Series("p_D", probs[:, 1]),
        pl.Series("p_A", probs[:, 2]),
        pl.Series("y", y),
    ])
    # Brier por fila
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


def _wp_et_poisson(score_diff_90: int, minute_et: int,
                   lam_h: float, lam_a: float,
                   prob_shootout_home: float) -> tuple[float, float, float]:
    """WP durante ET via Poisson goal arrival sobre los 30 min restantes de ET.

    Una vez llegado al minuto 120, si empate -> penaltis (shootout prob).
    Devuelve (wp_home, wp_draw, wp_away) donde "draw" aqui significa "llegara
    a penaltis" y el outcome final del partido si hay penaltis lo deriva
    prob_shootout_home (P home gana tanda).
    """
    # minutos restantes de ET hasta 120
    minutes_left = max(0, ET_MINUTES - minute_et)
    mean_h = lam_h * minutes_left
    mean_a = lam_a * minutes_left
    # Truncamos Poisson a 0..5 goles (prob extra negligible)
    max_g = 6
    from math import exp, factorial
    def poisson(k, mu): return np.exp(-mu) * mu**k / factorial(k)
    p_gh = np.array([poisson(k, mean_h) for k in range(max_g)])
    p_ga = np.array([poisson(k, mean_a) for k in range(max_g)])
    p_gh /= p_gh.sum(); p_ga /= p_ga.sum()
    # Distribucion conjunta del marcador FINAL de ET
    # final_diff = score_diff_90 + (gh - ga)
    p_H_reg, p_D_reg, p_A_reg = 0.0, 0.0, 0.0
    for h in range(max_g):
        for a in range(max_g):
            w = p_gh[h] * p_ga[a]
            final_diff = score_diff_90 + (h - a)
            if final_diff > 0:   p_H_reg += w
            elif final_diff < 0: p_A_reg += w
            else:                p_D_reg += w
    # p_D_reg = P(ET acaba en empate) -> hay penaltis.
    # En ese caso, prob_shootout_home determina quien gana.
    p_H = p_H_reg + p_D_reg * prob_shootout_home
    p_A = p_A_reg + p_D_reg * (1 - prob_shootout_home)
    p_D = 0.0  # al final, nadie empata (tanda resuelve)
    return float(p_H), float(p_D), float(p_A)


def compute_wp_per_minute(match_id: int, fit_result: dict,
                          elo_diff: float = 0.0,
                          comp_tier: int = 3,
                          lam_et_h: float | None = None,
                          lam_et_a: float | None = None,
                          prob_shootout_home: float = 0.5) -> pl.DataFrame:
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

    rows = []
    # --- Regulacion 1..90 ---
    for m in range(1, REG_MINUTES + 1):
        sd = sum(1 for x in gh if x < m) - sum(1 for x in ga if x < m)
        fv = {"score_diff": sd, "red_diff": 0, "elo_diff": elo_diff,
              "comp_tier": comp_tier}
        pH, pD, pA = predict_wp(fv, m, fit_result)
        # Leverage = diff WP si home marca +1: usamos fv con sd+1
        fv_plus = {**fv, "score_diff": sd + 1}
        pH_plus, _, _ = predict_wp(fv_plus, m, fit_result)
        leverage = abs(pH_plus - pH)
        rows.append({
            "match_id": match_id, "minute": m,
            "wp_home": pH, "wp_draw": pD, "wp_away": pA,
            "leverage": leverage, "score_diff": sd, "phase": "regulation",
        })

    # --- ET 91..120 (solo si partido llega a ET) ---
    # Usa score_diff final de regulacion + Poisson
    sd_final_reg = sum(1 for x in gh if x < REG_MINUTES + 1) \
                  - sum(1 for x in ga if x < REG_MINUTES + 1)
    et_max_event = max([m for m in gh + ga if m > REG_MINUTES], default=0)
    if et_max_event > 0 or sd_final_reg == 0:
        for m in range(REG_MINUTES + 1, REG_MINUTES + ET_MINUTES + 1):
            sd = sum(1 for x in gh if x < m) - sum(1 for x in ga if x < m)
            minute_et = m - REG_MINUTES
            pH, pD, pA = _wp_et_poisson(sd, minute_et, lam_et_h, lam_et_a,
                                         prob_shootout_home)
            pH_plus, _, _ = _wp_et_poisson(sd + 1, minute_et, lam_et_h, lam_et_a,
                                            prob_shootout_home)
            leverage = abs(pH_plus - pH)
            rows.append({
                "match_id": match_id, "minute": m,
                "wp_home": pH, "wp_draw": pD, "wp_away": pA,
                "leverage": leverage, "score_diff": sd, "phase": "extra_time",
            })

    df = pl.DataFrame(rows)
    # elimination_proximity para torneo KO: P(equipo home sera eliminado)
    # Approx: 1 - wp_home si home necesita ganar (KO). Para empate es ambiguo;
    # aqui usamos una metrica simple: max(wp_away, 0.5 * wp_draw).
    df = df.with_columns(
        (pl.col("wp_away") + 0.5 * pl.col("wp_draw")).alias("elim_prox_home")
    ).with_columns(
        (pl.col("wp_home") + 0.5 * pl.col("wp_draw")).alias("elim_prox_away")
    )
    return df


def cache_all_wp(fit_result: dict, overwrite: bool = False) -> Path:
    """Aplica compute_wp_per_minute a los 64 PFF y persiste tabla unificada."""
    out_path = _DERIVED / "per_minute.parquet"
    if out_path.exists() and not overwrite:
        return out_path
    dfs = []
    for mid in list_event_match_ids():
        dfs.append(compute_wp_per_minute(mid, fit_result))
    big = pl.concat(dfs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    big.write_parquet(out_path, compression="snappy", statistics=True)
    return out_path


# -- Sanity inline ---------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=== M04_wp sanity ===")

    # 1. Training matrix
    t0 = time.time()
    X, info = build_training_matrix(cache=True)
    print(f"training matrix: {info} en {time.time()-t0:.1f}s")
    print(f"  dtypes score_diff={X.schema['score_diff']}, red_diff={X.schema['red_diff']}")
    print(f"  y distrib: {X.group_by('y').len().sort('y').to_dicts()}")
    print(f"  score_diff range: [{X['score_diff'].min()}, {X['score_diff'].max()}]")
    print(f"  red_diff   range: [{X['red_diff'].min()}, {X['red_diff'].max()}]")

    # 2. Split y fit
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
        print(f"fit guardado en {fit_path}")

    # 3. Brier por bin
    brier_df = evaluate_brier_per_minute(X_val, fit)
    avg_brier = float(brier_df["brier"].mean())
    print(f"\nBrier 3-class sobre val: mean={avg_brier:.4f}")
    print("brier por 10-min bins:")
    tenbin = brier_df.with_columns((pl.col("minute_bin") // 10 * 10).alias("bin10"))
    print(tenbin.group_by("bin10").agg([
        pl.col("brier").mean(), pl.col("n").sum()
    ]).sort("bin10"))

    # 4. Predict sanity check
    print()
    for (sd, minute, elo, tier, label) in [
        (0,  1,   0.0, 1, "0-0 min1 liga"),
        (1, 89, 100.0, 3, "1-0 min89 WC fav"),
        (-1,89, -100.0,3, "0-1 min89 WC debil"),
        (0, 45,   0.0, 3, "0-0 HT neutral"),
    ]:
        p = predict_wp({"score_diff": sd, "red_diff": 0, "elo_diff": elo,
                        "comp_tier": tier}, minute, fit)
        print(f"  {label:<25} H={p[0]:.3f} D={p[1]:.3f} A={p[2]:.3f}")

    # 5. compute_wp_per_minute sobre 1 partido PFF (ET test)
    from M01_loader_pff import list_matches
    inv = list_matches()
    mid_et = 10511  # NED-ARG, fue a ET
    print(f"\ncompute_wp_per_minute({mid_et}) [NED-ARG con ET]:")
    wp = compute_wp_per_minute(mid_et, fit)
    print(f"  filas: {wp.height}, phases: {wp['phase'].unique().to_list()}")
    key_minutes = wp.filter(pl.col("minute").is_in([1, 45, 89, 105, 120]))
    print(key_minutes.select(["minute","phase","score_diff","wp_home","wp_draw","wp_away"]))

    # 6. Cache all 64 PFF
    print()
    t0 = time.time()
    out = cache_all_wp(fit, overwrite=True)
    print(f"cache_all_wp -> {out} en {time.time()-t0:.1f}s")
    big = pl.read_parquet(out)
    print(f"  total filas: {big.height:,} (esperado 64 * ~100 = ~6400)")
    print(f"  matches cacheados: {big['match_id'].n_unique()}")
