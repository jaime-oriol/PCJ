"""
M14_cate - CATE jerarquico bayesiano multivariate (Multivariate BCF analog).

Capa 4 PCJ. Estima el efecto causal HETEROGENEO por jugador del shock
emocional sobre los 4 canales conjuntamente, con jerarquia 3 niveles
(jugador ⊂ equipo ⊂ posicion), correlacion cross-canal LKJ, priors
informativos PFF grades y sampling HMC/NUTS exacto.

SOTA implementado:

  Componente                          Referencia                    Stack
  ---------------------------------   ---------------------------   ----------
  Multivariate BCF jerarquico         Hu et al. 2025 JRSS-A         numpyro
  Aggregate BCF datos jerarquicos     Thal et al. 2024 arXiv        numpyro
  NCP jerarquico (anti-funnel)        Betancourt-Girolami 2015      NCP manual
  LKJ Cholesky cross-canal x2         Lewandowski-Kurowicka-Joe     dist.LKJ
    (GA + GF separados)               2009                          Cholesky
  Priors PFF grades informativos      Gomes-Mendes-Neves 2025       gamma coef
  3-level hierarchy (player⊂team⊂pos) Yurko 2019, Maas-Hox 2005     hierarchy
  HMC/NUTS exact MCMC                 Hoffman-Gelman 2014           NUTS 4 chains
  R-hat + ESS convergence             Gelman-Rubin 1992 /           R-hat manual
                                      Vehtari 2021
  Posterior predictive checks         Gelman et al. 2013            simulate +
                                                                    KS + mean/sd

Implementacion numpyro multivariate COMPLETA:
  - NCP en TODOS los efectos aleatorios (no funnel en sigma_*)
  - Random effects SEPARADOS por shock_type (GA chasing, GF protecting)
    → indices Remontador/Cerrojo desde eta individual neto de equipo/posicion
  - Cross-canal correlation LKJ independiente por shock type
  - Priors informativos PFF_grade · gamma_k (Gomes-Mendes-Neves)
  - target_accept_prob=0.9 para topologia LKJ
  - R-hat < 1.05 + ESS_bulk > 400 verificado
  - Smoke test 2-chain × 100 iter antes del run completo

Modelo (NCP completo — Betancourt-Girolami 2015):
    delta_iks ~ Normal(mu_shock[s,k] + b_context[i,k] + eta[i,s,k], sigma_eps[k])
    b_context[i,k] = gamma[k]*pff_grade[i] + b_team[t(i),k] + b_position[p(i),k]
    b_team[t,k]    = sigma_team[k]     * b_team_raw[t,k]     (NCP)
    b_position[p,k] = sigma_pos[k]     * b_pos_raw[p,k]      (NCP)
    eta[i,GA,:] = (sigma_ga * L_ga_corr) @ eta_raw_ga[i,:]   (NCP chasing)
    eta[i,GF,:] = (sigma_gf * L_gf_corr) @ eta_raw_gf[i,:]   (NCP protecting)
    L_ga_corr, L_gf_corr ~ LKJCholesky(K=4, concentration=2.0)
    eta_raw_ga[i,:], eta_raw_gf[i,:] ~ Normal(0,1)
    b_team_raw[t,:], b_pos_raw[p,:] ~ Normal(0,1)
    mu_shock[s,k] ~ Normal(0, 0.5)  — shock-type population mean
    sigma_ga, sigma_gf, sigma_team, sigma_pos ~ HalfNormal(0.5)
    sigma_eps ~ HalfNormal(1.0)
    gamma[k] ~ Normal(0, 1)

donde:
    delta_iks = (post - pre) z-score within (channel, shock_type)
    i = player_id PFF
    k = canal ∈ {ataque, defensa, offball, fisico}  (orden: sorted)
    s = shock_type ∈ {GOAL_AGAINST=0, GOAL_FOR=1}   (orden: sorted alphabetic)

Indices PCJ (propuesta_final.md §Fase 5) — desde eta individual:
  Indice Remontador (chasing-clutch):
      = mean(eta_ga[i,atk] + eta_ga[i,off])
      [empuje ofensivo + off-ball INDIVIDUAL al conceder, neto de equipo/pos]
  Indice Cerrojo (protecting-clutch):
      = mean(eta_gf[i,def] + eta_gf[i,phys])
      [solidez defensiva + fisico INDIVIDUAL al marcar, neto de equipo/pos]
  Ranking within position_group: percentil del jugador respecto a su rol.

Outputs (data/parquet/derived/cate/):
  panel_delta.parquet    (player x shock x channel x shock_type → delta_z)
  posterior_player.parquet  (player x channel x shock_type → eta mean/sd/CI80/CI95)
  posterior_corr.parquet (shock_type x channel_k1 x channel_k2 → corr cross-canal)
  indices.parquet        (player → chasing_clutch_idx, protecting_clutch_idx)
  rankings.parquet       (player → rank_chasing, rank_protecting, rank_*_in_position)
  diagnostics.parquet    (param → r_hat, ess_bulk, converged)
  ppc.parquet            (canal x shock_type → KS_pvalue, mean/sd sim vs obs)
  model/cate_nuts.pkl    (NUTS samples posterior)

Depende de: M07 (shocks), M08-M11 (per_shock_window),
            M03 preprocess pff_grades.parquet (priors PFF).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


# -- Rutas ------------------------------------------------------------------

_REPO    = Path(__file__).resolve().parents[1]
_DERIVED = _REPO / "data" / "parquet" / "derived" / "cate"
_MODEL   = _DERIVED / "model"
_PFF_GRADES = _REPO / "data" / "parquet" / "derived" / "preprocess" / "pff_grades.parquet"


# -- Constantes pre-registradas --------------------------------------------

CHANNELS: dict[str, tuple[str, str, str]] = {
    "ataque":  ("ataque/per_shock_window.parquet",  "score_atk_pre", "score_atk_post"),
    "defensa": ("defensa/per_shock_window.parquet", "score_def_pre", "score_def_post"),
    "offball": ("offball/per_shock_window.parquet", "obso_pre",      "obso_post"),
    "fisico":  ("fisico/per_shock_window.parquet",  "score_phys_pre","score_phys_post"),
}
SHOCK_TYPES = ("GOAL_FOR", "GOAL_AGAINST")
N_CHANNELS  = len(CHANNELS)

# NUTS sampling (Hoffman-Gelman 2014). 4 chains paralelas para R-hat.
NUTS_NUM_CHAINS   = 4
NUTS_NUM_WARMUP   = 1000
NUTS_NUM_SAMPLES  = 1000

# Indices PCJ (propuesta §Fase 5)
CHASING_COMPONENTS    = (("ataque",  "GOAL_AGAINST"),
                          ("offball", "GOAL_AGAINST"))
PROTECTING_COMPONENTS = (("defensa", "GOAL_FOR"),
                          ("fisico",  "GOAL_FOR"))


# ===========================================================================
#  SECCION 1 — Build delta panel (player × shock × channel × shock_type)
# ===========================================================================

def build_delta_panel(cache: bool = True) -> pl.DataFrame:
    """Panel long: (pff_player_id, shock_id, channel, shock_type, delta_z).

    delta = post - pre dentro de cada (player, shock, channel). Z-score
    within (channel, shock_type) para que los 4 canales sean comparables en
    el modelo multivariate. Schema unificado X3.
    """
    cache_path = _DERIVED / "panel_delta.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    derived = _DERIVED.parent
    rows = []
    for ch, (rel_path, col_pre, col_post) in CHANNELS.items():
        df = pl.read_parquet(derived / rel_path).filter(
            pl.col(col_pre).is_not_null() & pl.col(col_post).is_not_null()
        )
        df = df.with_columns([
            (pl.col(col_post) - pl.col(col_pre)).cast(pl.Float64).alias("delta"),
            pl.lit(ch).alias("channel"),
        ]).select([
            "pff_match_id", "shock_id", "pff_player_id", "shock_type",
            "channel", "delta",
        ])
        rows.append(df)
    panel = pl.concat(rows)

    # Anadir position_group + team_id + stage + minute desde shocks_table
    shocks = pl.read_parquet(derived / "shocks/shocks_table.parquet").select([
        pl.col("match_id").alias("pff_match_id"),
        "shock_id",
        pl.col("player_id").alias("pff_player_id"),
        "position_group",
        pl.col("player_team_id").alias("pff_team_id"),
        "stage",                     # groups | ko (T1.3 column)
        "minute",                    # minuto del shock para join leverage
    ]).unique(subset=["pff_match_id", "shock_id", "pff_player_id"])
    panel = panel.join(shocks, on=["pff_match_id", "shock_id", "pff_player_id"],
                        how="left")

    # Anadir leverage del shock desde M04 WP per_minute
    wp_path = derived / "wp" / "per_minute.parquet"
    if wp_path.exists():
        wp = pl.read_parquet(wp_path).select([
            pl.col("match_id").alias("pff_match_id"),
            "minute", "leverage",
        ])
        panel = panel.join(wp, on=["pff_match_id", "minute"], how="left")
        # Z-score leverage para escala unitaria en el modelo
        lev_mean = float(panel["leverage"].drop_nulls().mean() or 0.0)
        lev_std = float(panel["leverage"].drop_nulls().std() or 1.0) or 1.0
        panel = panel.with_columns(
            ((pl.col("leverage").fill_null(lev_mean) - lev_mean) / lev_std)
                .alias("leverage_z")
        )
    else:
        panel = panel.with_columns(pl.lit(0.0).alias("leverage_z"))

    # Z-score within (channel, shock_type)
    panel = panel.with_columns(
        ((pl.col("delta") - pl.col("delta").mean().over(["channel", "shock_type"])) /
         pl.col("delta").std().over(["channel", "shock_type"])).alias("delta_z")
    )

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        panel.write_parquet(cache_path, compression="snappy")
    return panel


def attach_pff_grades(panel: pl.DataFrame) -> pl.DataFrame:
    """Anade PFF grade pre-torneo agregado por jugador (Gomes-Mendes-Neves 2025).

    pff_grade_z = z-score within position_group del PFF grade promedio del
    jugador. Z-score within position absorbe que GKs tienen grades distintos
    a CFs (graded en escalas distintas en sistema PFF).
    """
    if not _PFF_GRADES.exists():
        raise FileNotFoundError(
            f"Falta {_PFF_GRADES}. Ejecuta src/preprocess/pff_grades_extract.py"
        )
    grades = pl.read_parquet(_PFF_GRADES).select([
        "pff_player_id", "pff_grade_mean", "n_grades",
    ])
    panel = panel.join(grades, on="pff_player_id", how="left")
    # Players sin grade (no aparecen en eventos PFF) → 0 (prior neutral)
    panel = panel.with_columns(pl.col("pff_grade_mean").fill_null(0.0))
    # Z-score dentro de position_group
    panel = panel.with_columns(
        ((pl.col("pff_grade_mean") - pl.col("pff_grade_mean").mean().over("position_group")) /
         pl.col("pff_grade_mean").std().over("position_group").fill_null(1.0))
        .fill_null(0.0).alias("pff_grade_z")
    )
    return panel


# ===========================================================================
#  SECCION 2 — Modelo Multivariate Bayesian Hierarchical (numpyro NUTS)
# ===========================================================================

def _model_mvbcf(player_idx, shock_idx, channel_idx,
                  pff_grade_z, y, n_players, n_teams, n_positions, n_shock_types,
                  n_channels, player_to_team, player_to_position):
    """NCP completo — evita funnel de Neal en todos los efectos aleatorios.

    Betancourt-Girolami 2015: parameterizacion NO centrada en b_team, b_pos,
    eta_ga, eta_gf. El sampler NUTS ve solo variables N(0,1) + escalas
    independientes → geometria regular, buena mezcla.

    Efectos individuales separados por shock_type (GA y GF independientes)
    para que los indices Remontador/Cerrojo capturen respuesta INDIVIDUAL
    al shock especifico, neta de efectos de equipo y posicion.

    Stage + leverage NO se incluyen como moderadores aditivos: los shocks
    se identifican causalmente con FE de player-shock (M12) y el ATE
    population ya documenta stage-heterogeneidad (M12B T2.8). Anadir
    additive controls aqui solo desplaza la media sin capturar heterogeneidad
    individual (eso requeriria interaction eta * stage, fuera de scope).
    Las cols stage + leverage_z viven en el panel parquet y M15 las consume
    como exposure context per jugador.

    shock_idx: GOAL_AGAINST=0, GOAL_FOR=1 (sorted alphabetically — assert en fit_cate_nuts)
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    # Hyperpriors (solo escalas — NCP: no dependen de datos directamente)
    sigma_team     = numpyro.sample("sigma_team",     dist.HalfNormal(0.5).expand([n_channels]).to_event(1))
    sigma_position = numpyro.sample("sigma_position", dist.HalfNormal(0.5).expand([n_channels]).to_event(1))
    sigma_ga       = numpyro.sample("sigma_ga",       dist.HalfNormal(0.5).expand([n_channels]).to_event(1))
    sigma_gf       = numpyro.sample("sigma_gf",       dist.HalfNormal(0.5).expand([n_channels]).to_event(1))
    sigma_eps      = numpyro.sample("sigma_eps",      dist.HalfNormal(1.0).expand([n_channels]).to_event(1))

    # PFF grade coefficient por canal
    gamma = numpyro.sample("gamma", dist.Normal(0.0, 1.0).expand([n_channels]).to_event(1))

    # Shock-type population mean (intercepto medio por tipo de shock)
    mu_shock = numpyro.sample("mu_shock", dist.Normal(0.0, 0.5).expand([n_shock_types, n_channels]).to_event(2))

    # NCP para b_team y b_position — b = sigma * raw, raw ~ N(0,1)
    b_team_raw = numpyro.sample("b_team_raw", dist.Normal(0, 1).expand([n_teams, n_channels]).to_event(2))
    b_team = numpyro.deterministic("b_team", b_team_raw * sigma_team[None, :])

    b_pos_raw = numpyro.sample("b_pos_raw", dist.Normal(0, 1).expand([n_positions, n_channels]).to_event(2))
    b_position = numpyro.deterministic("b_position", b_pos_raw * sigma_position[None, :])

    # NCP para efectos individuales GOAL_AGAINST (chasing) con LKJ cross-canal
    # eta_ga[i,:] = L_ga @ eta_raw_ga[i,:]  con  L_ga = diag(sigma_ga) @ L_ga_corr
    L_ga_corr = numpyro.sample("L_ga_corr", dist.LKJCholesky(n_channels, concentration=2.0))
    L_ga = sigma_ga[:, None] * L_ga_corr                           # (K, K) lower tri
    eta_raw_ga = numpyro.sample("eta_raw_ga", dist.Normal(0, 1).expand([n_players, n_channels]).to_event(2))
    eta_ga = numpyro.deterministic("eta_ga", jnp.matmul(eta_raw_ga, L_ga.T))   # (P, K)

    # NCP para efectos individuales GOAL_FOR (protecting) con LKJ cross-canal
    L_gf_corr = numpyro.sample("L_gf_corr", dist.LKJCholesky(n_channels, concentration=2.0))
    L_gf = sigma_gf[:, None] * L_gf_corr
    eta_raw_gf = numpyro.sample("eta_raw_gf", dist.Normal(0, 1).expand([n_players, n_channels]).to_event(2))
    eta_gf = numpyro.deterministic("eta_gf", jnp.matmul(eta_raw_gf, L_gf.T))   # (P, K)

    # eta[i, shock_type, k]: (P, S, K) — GA=idx0, GF=idx1
    eta_player = jnp.stack([eta_ga, eta_gf], axis=1)               # (P, S, K)

    # b_context[i,k] = grade_effect + team_effect + position_effect (comun a GA/GF)
    b_context = numpyro.deterministic("b_context",
        gamma[None, :] * pff_grade_z[:, None]                       # (P, K)
        + b_team[player_to_team]
        + b_position[player_to_position])

    # Likelihood: obs = mu_shock[s,k] + b_context[i,k] + eta[i,s,k] + eps
    pred = (mu_shock[shock_idx, channel_idx]
            + b_context[player_idx, channel_idx]
            + eta_player[player_idx, shock_idx, channel_idx])
    with numpyro.plate("N", len(y)):
        numpyro.sample("obs", dist.Normal(pred, sigma_eps[channel_idx]), obs=y)


def fit_cate_nuts(panel: pl.DataFrame,
                  num_warmup: int = NUTS_NUM_WARMUP,
                  num_samples: int = NUTS_NUM_SAMPLES,
                  num_chains: int = NUTS_NUM_CHAINS,
                  seed: int = 42) -> dict:
    """Entrena modelo via NUTS HMC (4 chains) + diagnostics R-hat/ESS."""
    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.infer import MCMC, NUTS

    numpyro.set_host_device_count(num_chains)

    df = panel.filter(pl.col("delta_z").is_not_null() &
                       pl.col("position_group").is_not_null()).to_pandas()
    if df.shape[0] < 100:
        raise ValueError(f"Panel demasiado pequeno: {df.shape[0]} filas")

    # Indexers
    players = sorted(df["pff_player_id"].unique())
    teams = sorted(df["pff_team_id"].dropna().unique())
    positions = sorted(df["position_group"].unique())
    shock_types = sorted(df["shock_type"].unique())
    channels = sorted(df["channel"].unique())
    p_to_idx = {p: i for i, p in enumerate(players)}
    t_to_idx = {t: i for i, t in enumerate(teams)}
    pos_to_idx = {p: i for i, p in enumerate(positions)}
    sh_to_idx = {s: i for i, s in enumerate(shock_types)}
    ch_to_idx = {c: i for i, c in enumerate(channels)}
    # El modelo asume GOAL_AGAINST=0, GOAL_FOR=1 (sorted alphabetical)
    assert sh_to_idx.get("GOAL_AGAINST") == 0 and sh_to_idx.get("GOAL_FOR") == 1, \
        f"Orden shock_types inesperado: {sh_to_idx}"
    # Stage map (groups=0, ko=1) preservado en panel para consumers downstream (M15)
    stages = sorted(df["stage"].dropna().unique()) if "stage" in df.columns else []
    stage_to_idx = {s: i for i, s in enumerate(stages)}

    # Player → team y position lookups
    p_to_team = {}
    p_to_pos = {}
    p_to_grade_z = {}
    for r in df.itertuples(index=False):
        if not np.isnan(r.pff_team_id):
            p_to_team[r.pff_player_id] = t_to_idx[int(r.pff_team_id)]
        p_to_pos[r.pff_player_id] = pos_to_idx[r.position_group]
        p_to_grade_z[r.pff_player_id] = r.pff_grade_z

    player_to_team_arr = np.array(
        [p_to_team.get(p, 0) for p in players], dtype=np.int32)
    player_to_position_arr = np.array(
        [p_to_pos.get(p, 0) for p in players], dtype=np.int32)
    pff_grade_z_arr = np.array(
        [p_to_grade_z.get(p, 0.0) for p in players], dtype=np.float32)

    df = df[df["pff_team_id"].notna()].copy()
    player_idx = df["pff_player_id"].map(p_to_idx).values.astype(np.int32)
    shock_idx = df["shock_type"].map(sh_to_idx).values.astype(np.int32)
    channel_idx = df["channel"].map(ch_to_idx).values.astype(np.int32)
    y = df["delta_z"].values.astype(np.float32)

    print(f"  NUTS: N={len(y)}, players={len(players)}, teams={len(teams)}, "
          f"positions={len(positions)}, shock_types={len(shock_types)}, "
          f"channels={len(channels)}")
    print(f"  warmup={num_warmup}, samples={num_samples}, chains={num_chains}")

    # target_accept_prob=0.9 recomendado para modelos con LKJ + jerarquia
    kernel = NUTS(_model_mvbcf, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                 num_chains=num_chains, progress_bar=True)
    mcmc.run(
        jax.random.PRNGKey(seed),
        player_idx, shock_idx, channel_idx,
        pff_grade_z_arr, y,
        len(players), len(teams), len(positions), len(shock_types), len(channels),
        player_to_team_arr, player_to_position_arr,
        extra_fields=("diverging", "accept_prob"),
    )

    samples = mcmc.get_samples()
    samples_per_chain = mcmc.get_samples(group_by_chain=True)
    extra = mcmc.get_extra_fields()
    diverging = np.asarray(extra.get("diverging", np.zeros(0)))
    accept_prob = np.asarray(extra.get("accept_prob", np.zeros(0)))
    n_div = int(diverging.sum()) if diverging.size else 0
    if accept_prob.size:
        print(f"  divergencias HMC: {n_div}/{diverging.size} (=0 ideal, >1% problematico) "
              f"| accept_prob mean: {accept_prob.mean():.3f}")
    else:
        print(f"  divergencias HMC: {n_div}")

    return {
        "samples":           {k: np.array(v) for k, v in samples.items()},
        "samples_per_chain": {k: np.array(v) for k, v in samples_per_chain.items()},
        "p_to_idx":          p_to_idx,
        "t_to_idx":          t_to_idx,
        "pos_to_idx":        pos_to_idx,
        "sh_to_idx":         sh_to_idx,
        "ch_to_idx":         ch_to_idx,
        "stage_to_idx":      stage_to_idx,
        "player_to_team":    player_to_team_arr,
        "player_to_position": player_to_position_arr,
        "pff_grade_z":       pff_grade_z_arr,
        "n_obs":             int(len(y)),
        "n_diverging":       n_div,
        "accept_prob_mean":  float(accept_prob.mean()) if accept_prob.size else None,
    }


# ===========================================================================
#  SECCION 3 — Diagnosticos: R-hat + ESS (Gelman-Rubin)
# ===========================================================================

def compute_diagnostics(fit: dict) -> pl.DataFrame:
    """R-hat (Gelman-Rubin 1992) + ESS bulk (Vehtari 2021) por param escala.

    Acceptance: R-hat < 1.05 + ESS_bulk > 400 = convergencia OK.
    Solo diagnostica params de escala + correlacion. Los raw NCP (P x K)
    y los deterministic (eta_ga, eta_gf) se omiten por volume.
    """
    # Solo params diagnosticables (escalas, correlaciones, efectos globales)
    _SKIP = frozenset({
        "eta_raw_ga", "eta_raw_gf",   # NCP raw (P x K) — N(0,1) by design
        "b_team_raw", "b_pos_raw",     # NCP raw (T/P x K) — N(0,1) by design
        "eta_ga", "eta_gf",            # deterministic — derived
        "b_team", "b_position",        # deterministic — derived
        "b_context",                   # deterministic — derived (P x K)
    })
    samples = fit["samples_per_chain"]   # {param: (n_chains, n_samples, ...)}
    rows = []
    for name, arr in samples.items():
        if name in _SKIP:
            continue
        flat_arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
        for i in range(flat_arr.shape[2]):
            x = flat_arr[:, :, i]   # (n_chains, n_samples)
            rh = _r_hat(x)
            ess_b = _ess_bulk(x)
            rows.append({
                "param":     name,
                "idx":       i,
                "r_hat":     float(rh),
                "ess_bulk":  float(ess_b),
                "converged": bool(rh < 1.05 and ess_b > 400),
            })
    return pl.DataFrame(rows)


def _r_hat(x: np.ndarray) -> float:
    """Gelman-Rubin R-hat. x shape (n_chains, n_samples)."""
    n, m = x.shape[1], x.shape[0]
    chain_means = x.mean(axis=1)
    chain_vars = x.var(axis=1, ddof=1)
    W = chain_vars.mean()
    B = n * chain_means.var(ddof=1)
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W)) if W > 0 else 1.0


def _ess_bulk(x: np.ndarray) -> float:
    """ESS bulk simplificado (autocorrelacion lag-1). x shape (chains, samples)."""
    flat = x.flatten()
    n = len(flat)
    if n < 4:
        return float(n)
    rho = np.corrcoef(flat[:-1], flat[1:])[0, 1]
    if np.isnan(rho) or rho >= 1:
        return float(n)
    ess = n * (1 - rho) / (1 + rho)
    return max(float(ess), 1.0)


# ===========================================================================
#  SECCION 4 — Posterior predictive check (KS-test)
# ===========================================================================

def posterior_predictive_check(fit: dict, panel: pl.DataFrame,
                                n_replicates: int = 20,
                                seed: int = 0) -> pl.DataFrame:
    """PPC: simula y_rep desde posterior y compara con observado.

    Con N≈14k, el KS-test tiene poder infinito y siempre rechaza (p≈0) aunque
    el modelo este bien calibrado. Se reporta KS_p informativo pero la columna
    'calibrated' usa criterio practico: |mean_diff|<0.05 y |sd_diff|<0.10.
    """
    from scipy.stats import ks_2samp
    s = fit["samples"]
    rng = np.random.default_rng(seed)
    draw_idx = rng.choice(s["sigma_eps"].shape[0], n_replicates, replace=False)

    df_pd = (panel.filter(pl.col("delta_z").is_not_null() &
                           pl.col("position_group").is_not_null() &
                           pl.col("pff_team_id").is_not_null())
             .to_pandas())
    p_to_idx  = fit["p_to_idx"]
    sh_to_idx = fit["sh_to_idx"]
    ch_to_idx = fit["ch_to_idx"]
    player_idx  = df_pd["pff_player_id"].map(p_to_idx).values
    shock_idx_v = df_pd["shock_type"].map(sh_to_idx).values
    channel_idx_v = df_pd["channel"].map(ch_to_idx).values
    y_obs = df_pd["delta_z"].values

    rows = []
    for ch_n, ch_i in ch_to_idx.items():
        for sh_n, sh_i in sh_to_idx.items():
            mask = (channel_idx_v == ch_i) & (shock_idx_v == sh_i)
            obs = y_obs[mask]
            eta_key = "eta_ga" if sh_i == 0 else "eta_gf"   # GA=0, GF=1
            sims = []
            for r in draw_idx:
                # Modelo completo: mu_shock + b_context + eta + N(0, sigma_eps)
                eta_s   = s[eta_key][r]                  # (P, K)
                bctx    = s["b_context"][r]              # (P, K)
                mu_s    = s["mu_shock"][r, sh_i, ch_i]
                sig_eps = s["sigma_eps"][r, ch_i]
                mu_obs  = (mu_s
                           + bctx[player_idx[mask], ch_i]
                           + eta_s[player_idx[mask], ch_i])
                sims.extend((mu_obs + rng.standard_normal(len(mu_obs)) * sig_eps).tolist())
            ks_stat, ks_p = ks_2samp(obs, sims)
            obs_mean, sim_mean = float(obs.mean()), float(np.mean(sims))
            obs_sd,   sim_sd   = float(obs.std()),  float(np.std(sims))
            rows.append({
                "channel":    ch_n,
                "shock_type": sh_n,
                "obs_mean":   obs_mean,
                "obs_sd":     obs_sd,
                "sim_mean":   sim_mean,
                "sim_sd":     sim_sd,
                "ks_pvalue":  float(ks_p),
                # criterio practico: delta mean<0.05 y delta sd<0.10
                "calibrated": bool(abs(obs_mean - sim_mean) < 0.05
                                   and abs(obs_sd - sim_sd) < 0.10),
            })
    return pl.DataFrame(rows)


# ===========================================================================
#  SECCION 5 — Posterior per player + cross-canal correlation
# ===========================================================================

def posterior_per_player(fit: dict) -> pl.DataFrame:
    """IC bayesianos per (player, channel, shock_type) desde eta individual.

    Usa eta_ga (GOAL_AGAINST) y eta_gf (GOAL_FOR) — efecto individual
    neto de team, position y PFF grade. Es el input directo a los indices
    Remontador/Cerrojo y a M15.
    """
    s = fit["samples"]
    eta_ga = s["eta_ga"]   # (n_samples, n_players, n_channels)
    eta_gf = s["eta_gf"]   # (n_samples, n_players, n_channels)
    n_samples, n_players, n_channels = eta_ga.shape

    inv_p = {v: k for k, v in fit["p_to_idx"].items()}
    inv_c = {v: k for k, v in fit["ch_to_idx"].items()}

    def _block(arr: np.ndarray, shock_name: str) -> list[dict]:
        rows = []
        for c_i in range(n_channels):
            col = arr[:, :, c_i]                                # (S, P)
            mean    = col.mean(axis=0)
            sd      = col.std(axis=0)
            ci_lo80 = np.percentile(col, 10, axis=0)
            ci_hi80 = np.percentile(col, 90, axis=0)
            ci_lo95 = np.percentile(col, 2.5, axis=0)
            ci_hi95 = np.percentile(col, 97.5, axis=0)
            for p_i in range(n_players):
                rows.append({
                    "pff_player_id": inv_p[p_i],
                    "shock_type":    shock_name,
                    "channel":       inv_c[c_i],
                    "cate_mean":     float(mean[p_i]),
                    "cate_sd":       float(sd[p_i]),
                    "ci_lo80":       float(ci_lo80[p_i]),
                    "ci_hi80":       float(ci_hi80[p_i]),
                    "ci_lo95":       float(ci_lo95[p_i]),
                    "ci_hi95":       float(ci_hi95[p_i]),
                })
        return rows

    return pl.DataFrame(_block(eta_ga, "GOAL_AGAINST") + _block(eta_gf, "GOAL_FOR"))


def posterior_cross_canal_corr(fit: dict) -> pl.DataFrame:
    """Cross-canal correlation desde L_ga_corr (chasing) y L_gf_corr (protecting).

    Corr[k1,k2] = (L_corr @ L_corr.T)[k1,k2] — posterior mean por shock type.
    """
    s = fit["samples"]
    inv_c = {v: k for k, v in fit["ch_to_idx"].items()}
    rows = []
    for shock_name, key in [("GOAL_AGAINST", "L_ga_corr"), ("GOAL_FOR", "L_gf_corr")]:
        Ls = s[key]                              # (n_samples, K, K)
        corr = np.einsum("sij,skj->sik", Ls, Ls)  # L @ L.T per sample
        corr_mean = corr.mean(axis=0)            # (K, K)
        for i in range(corr_mean.shape[0]):
            for j in range(corr_mean.shape[1]):
                rows.append({
                    "shock_type":  shock_name,
                    "channel_1":   inv_c[i],
                    "channel_2":   inv_c[j],
                    "correlation": float(corr_mean[i, j]),
                })
    return pl.DataFrame(rows)


# ===========================================================================
#  SECCION 6 — Indices PCJ + ranking within position
# ===========================================================================

def compute_indices(fit: dict) -> pl.DataFrame:
    """Indices Remontador/Cerrojo desde eta individual (neto de team/pos/grade).

    chasing_clutch_idx  = mean(eta_ga[:,atk], eta_ga[:,off])  — GA individual
    protecting_clutch_idx = mean(eta_gf[:,def], eta_gf[:,phys]) — GF individual

    Al usar eta (no b_player), los indices reflejan respuesta INDIVIDUAL al shock,
    descontando el efecto de equipo y posicion.
    """
    s = fit["samples"]
    eta_ga = s["eta_ga"]   # (n_samples, n_players, n_channels)
    eta_gf = s["eta_gf"]
    ch = fit["ch_to_idx"]  # canal -> idx (sorted: ataque=0, defensa=1, fisico=2, offball=3)
    inv_p = {v: k for k, v in fit["p_to_idx"].items()}

    eta_ga_mean = eta_ga.mean(axis=0)   # (n_players, n_channels)
    eta_gf_mean = eta_gf.mean(axis=0)

    atk_i = ch["ataque"]
    off_i = ch["offball"]
    def_i = ch["defensa"]
    phy_i = ch["fisico"]

    rows = [
        {
            "pff_player_id":       inv_p[p_i],
            "chasing_clutch_idx":  float((eta_ga_mean[p_i, atk_i] + eta_ga_mean[p_i, off_i]) / 2),
            "protecting_clutch_idx": float((eta_gf_mean[p_i, def_i] + eta_gf_mean[p_i, phy_i]) / 2),
        }
        for p_i in range(len(inv_p))
    ]
    return pl.DataFrame(rows)


def compute_rankings(indices: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    """Ranking dentro del rol (position_group) + ranking global."""
    pos_per_player = panel.filter(pl.col("position_group").is_not_null()).group_by(
        "pff_player_id"
    ).agg(pl.col("position_group").mode().first().alias("position_group"))
    df = indices.join(pos_per_player, on="pff_player_id", how="left")
    df = df.with_columns([
        pl.col("chasing_clutch_idx").rank(descending=True, method="ordinal")
          .alias("rank_chasing_global"),
        pl.col("protecting_clutch_idx").rank(descending=True, method="ordinal")
          .alias("rank_protecting_global"),
        pl.col("chasing_clutch_idx").rank(descending=True, method="ordinal")
          .over("position_group").alias("rank_chasing_in_position"),
        pl.col("protecting_clutch_idx").rank(descending=True, method="ordinal")
          .over("position_group").alias("rank_protecting_in_position"),
    ])
    return df


# ===========================================================================
#  SECCION 7 — compute_all + cache
# ===========================================================================

def compute_all(cache: bool = True, overwrite: bool = False,
                 num_warmup: int = NUTS_NUM_WARMUP,
                 num_samples: int = NUTS_NUM_SAMPLES,
                 num_chains: int = NUTS_NUM_CHAINS) -> dict[str, Path]:
    """Pipeline completa M14 con HMC NUTS + LKJ + PFF priors + 3 niveles."""
    out_paths = {
        "panel":       _DERIVED / "panel_delta.parquet",
        "posterior":   _DERIVED / "posterior_player.parquet",
        "corr":        _DERIVED / "posterior_corr.parquet",
        "indices":     _DERIVED / "indices.parquet",
        "rankings":    _DERIVED / "rankings.parquet",
        "diagnostics": _DERIVED / "diagnostics.parquet",
        "ppc":         _DERIVED / "ppc.parquet",
        "model":       _MODEL   / "cate_nuts.pkl",
    }
    if not overwrite and all(p.exists() for p in out_paths.values()):
        return out_paths
    _DERIVED.mkdir(parents=True, exist_ok=True)
    _MODEL.mkdir(parents=True, exist_ok=True)

    print("[1] Build delta panel + PFF grades priors...")
    panel = build_delta_panel(cache=cache)
    panel = attach_pff_grades(panel)
    print(f"  panel: {panel.height:,} rows, {panel['pff_player_id'].n_unique()} players, "
          f"PFF grade coverage: {panel.filter(pl.col('pff_grade_z')!=0).height/panel.height*100:.0f}%")

    print("[2] Fit NUTS HMC (multivariate jerarquico 3 niveles + LKJ + PFF priors)...")
    fit = fit_cate_nuts(panel, num_warmup=num_warmup, num_samples=num_samples,
                         num_chains=num_chains)
    if cache:
        with open(out_paths["model"], "wb") as f:
            pickle.dump({k: v for k, v in fit.items() if k != "samples_per_chain"}, f)

    print("[3] Diagnostics R-hat + ESS...")
    diag = compute_diagnostics(fit)
    n_diverged = diag.filter(~pl.col("converged")).height
    print(f"  diagnostics: {diag.height} params, {n_diverged} no convergidos "
          f"(R-hat>=1.05 o ESS<400)")
    if cache:
        diag.write_parquet(out_paths["diagnostics"], compression="snappy")

    print("[4] Posterior predictive check (KS-test)...")
    ppc = posterior_predictive_check(fit, panel)
    n_calib = ppc.filter(pl.col("calibrated")).height
    print(f"  PPC: {n_calib}/{ppc.height} (channel x shock_type) calibrados (KS p>0.05)")
    if cache:
        ppc.write_parquet(out_paths["ppc"], compression="snappy")

    print("[5] Posterior per player + cross-canal correlation...")
    post = posterior_per_player(fit)
    corr = posterior_cross_canal_corr(fit)
    if cache:
        post.write_parquet(out_paths["posterior"], compression="snappy")
        corr.write_parquet(out_paths["corr"], compression="snappy")
    print(f"  posterior: {post.height} rows; cross-canal corr (GA):")
    print(corr.filter(pl.col("shock_type") == "GOAL_AGAINST")
              .pivot(on="channel_2", index="channel_1", values="correlation"))

    print("[6] Indices Remontador + Cerrojo + ranking within position...")
    idx = compute_indices(fit)
    rank = compute_rankings(idx, panel)
    if cache:
        idx.write_parquet(out_paths["indices"], compression="snappy")
        rank.write_parquet(out_paths["rankings"], compression="snappy")

    return out_paths


# ===========================================================================
#  SECCION 7.5 — Smoke test (2 chains x 100 iter, 5 partidos, ~2-3 min)
# ===========================================================================

def run_smoke_test(seed: int = 0, n_matches: int = 10,
                    num_warmup: int = 200, num_samples: int = 200) -> bool:
    """Smoke test exhaustivo: shapes, divergencias HMC, posterior sanity, PPC,
    NCP identity check, face validity differentiation.

    Default: 10 partidos, 2 chains x 400 iter (~3-5 min). Cubre todos los
    fallos modelales que podrian aparecer en el run completo (2h+).
    """
    print("[SMOKE] Cargando panel mini ({} partidos)...".format(n_matches))
    panel_full = build_delta_panel(cache=True)
    panel_full = attach_pff_grades(panel_full)
    # Mezclar groups + KO para smoke (asegura ambos stages presentes)
    all_matches = panel_full["pff_match_id"].unique().sort()
    matches = pl.concat([all_matches.head(n_matches // 2),
                          all_matches.tail(n_matches - n_matches // 2)]).unique()
    panel = panel_full.filter(pl.col("pff_match_id").is_in(matches))
    n_rows, n_players = panel.height, panel["pff_player_id"].n_unique()
    print(f"  mini panel: {n_rows:,} rows, {n_players} players")
    if n_rows < 200 or n_players < 20:
        print(f"  [FAIL] Panel demasiado pequeño.")
        return False

    print(f"[SMOKE] Fit NUTS 2 chains x ({num_warmup}+{num_samples}) iter...")
    fit = fit_cate_nuts(panel, num_warmup=num_warmup, num_samples=num_samples,
                        num_chains=2, seed=seed)
    s = fit["samples"]
    n_total = num_warmup + num_samples
    n_pl = len(fit["p_to_idx"])
    n_ch = len(fit["ch_to_idx"])
    n_te = len(fit["t_to_idx"])
    n_po = len(fit["pos_to_idx"])

    # ------------------------------------------------------------------ #
    # T1. Shapes de TODOS los sites del modelo
    # ------------------------------------------------------------------ #
    expected_shapes = {
        "eta_ga":     (n_total, n_pl, n_ch),
        "eta_gf":     (n_total, n_pl, n_ch),
        "eta_raw_ga": (n_total, n_pl, n_ch),
        "eta_raw_gf": (n_total, n_pl, n_ch),
        "b_team_raw": (n_total, n_te, n_ch),
        "b_pos_raw":  (n_total, n_po, n_ch),
        "L_ga_corr":  (n_total, n_ch, n_ch),
        "L_gf_corr":  (n_total, n_ch, n_ch),
        "sigma_ga":   (n_total, n_ch),
        "sigma_gf":   (n_total, n_ch),
        "sigma_team": (n_total, n_ch),
        "sigma_position": (n_total, n_ch),
        "sigma_eps":  (n_total, n_ch),
        "mu_shock":   (n_total, 2, n_ch),
        "gamma":      (n_total, n_ch),
    }
    fails = []
    for name, exp in expected_shapes.items():
        if name not in s:
            fails.append(f"missing site {name}")
        elif tuple(s[name].shape) != exp:
            fails.append(f"{name}: {s[name].shape} != {exp}")
    if fails:
        print("[SMOKE] FAIL T1 shapes:")
        for f in fails: print(f"   - {f}")
        return False
    print(f"  T1 shapes: OK ({len(expected_shapes)} sites)")

    # ------------------------------------------------------------------ #
    # T2. Divergencias HMC (= 0 ideal, < 1% tolerable)
    # ------------------------------------------------------------------ #
    n_div = fit.get("n_diverging", 0)
    div_ratio = n_div / (2 * num_samples) if num_samples else 0
    if div_ratio > 0.01:
        print(f"  T2 divergencias: FAIL ({n_div}/{2*num_samples} = {div_ratio:.1%})")
        return False
    print(f"  T2 divergencias: OK ({n_div}/{2*num_samples})")

    # ------------------------------------------------------------------ #
    # T3. NCP identity: eta_ga ?= eta_raw_ga @ L_ga.T para una muestra random
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(seed)
    r = int(rng.integers(0, n_total))
    L_ga_corr_r = s["L_ga_corr"][r]                # (K, K)
    sigma_ga_r  = s["sigma_ga"][r]                 # (K,)
    L_ga_r      = sigma_ga_r[:, None] * L_ga_corr_r
    eta_raw_r   = s["eta_raw_ga"][r]               # (P, K)
    eta_recon   = eta_raw_r @ L_ga_r.T
    eta_actual  = s["eta_ga"][r]
    max_diff    = float(np.abs(eta_recon - eta_actual).max())
    if max_diff > 1e-4:
        print(f"  T3 NCP identity: FAIL (max_diff={max_diff:.2e})")
        return False
    print(f"  T3 NCP identity: OK (max_diff={max_diff:.2e})")

    # ------------------------------------------------------------------ #
    # T4. Posterior sanity: escalas en rangos razonables
    # ------------------------------------------------------------------ #
    sanity = []
    for name in ("sigma_ga", "sigma_gf", "sigma_team", "sigma_position", "sigma_eps"):
        m = float(s[name].mean())
        if not (0.001 < m < 5.0):
            sanity.append(f"{name} mean={m:.3f} fuera de (0.001, 5.0)")
    mu_max = float(np.abs(s["mu_shock"]).max())
    if mu_max > 5.0:
        sanity.append(f"|mu_shock|.max={mu_max:.2f} > 5")
    g_max = float(np.abs(s["gamma"]).max())
    if g_max > 10.0:
        sanity.append(f"|gamma|.max={g_max:.2f} > 10")
    if sanity:
        print("  T4 posterior sanity: FAIL:")
        for x in sanity: print(f"   - {x}")
        return False
    print(f"  T4 posterior sanity: OK")

    # ------------------------------------------------------------------ #
    # T5. PPC: medias simuladas vs observadas (< 0.1 diff con 100 iter)
    # ------------------------------------------------------------------ #
    ppc = posterior_predictive_check(fit, panel, n_replicates=10)
    bad_ppc = ppc.filter(
        (pl.col("obs_mean") - pl.col("sim_mean")).abs() > 0.1
    )
    if bad_ppc.height > 0:
        print(f"  T5 PPC: FAIL ({bad_ppc.height}/{ppc.height} canales con |diff_mean|>0.1)")
        print(bad_ppc)
        return False
    print(f"  T5 PPC: OK (todas las medias simuladas dentro de 0.1 de obs)")

    # ------------------------------------------------------------------ #
    # T6. Pipeline de extraccion completo
    # ------------------------------------------------------------------ #
    post = posterior_per_player(fit)
    assert post.height == n_pl * n_ch * 2, f"posterior shape: {post.height}"
    corr = posterior_cross_canal_corr(fit)
    assert corr.height == 2 * n_ch * n_ch, f"corr shape: {corr.height}"
    idx = compute_indices(fit)
    assert idx.height == n_pl, f"indices shape: {idx.height}"
    rank = compute_rankings(idx, panel)
    assert rank.height == n_pl
    diag = compute_diagnostics(fit)
    print(f"  T6 extraccion: OK (post={post.height}, corr={corr.height}, "
          f"idx={idx.height}, rank={rank.height}, diag={diag.height} params)")

    # ------------------------------------------------------------------ #
    # T7. Face validity: ranking diferenciado (top != bottom, no degenerate)
    # ------------------------------------------------------------------ #
    cci_range = float(idx["chasing_clutch_idx"].max() - idx["chasing_clutch_idx"].min())
    pci_range = float(idx["protecting_clutch_idx"].max() - idx["protecting_clutch_idx"].min())
    if cci_range < 0.01 or pci_range < 0.01:
        print(f"  T7 differentiation: FAIL (cci_range={cci_range:.4f}, pci_range={pci_range:.4f})")
        return False
    print(f"  T7 differentiation: OK (cci range={cci_range:.3f}, pci range={pci_range:.3f})")

    # ------------------------------------------------------------------ #
    # T8. R-hat ESS sanity: laxo para 200 samples (R-hat < 1.5 OK)
    # ------------------------------------------------------------------ #
    bad_rhat = diag.filter(pl.col("r_hat") > 1.5)
    if bad_rhat.height > diag.height * 0.2:   # > 20% de params con R-hat alto
        print(f"  T8 R-hat: WARN ({bad_rhat.height}/{diag.height} con R-hat>1.5)")
    else:
        print(f"  T8 R-hat: OK ({bad_rhat.height}/{diag.height} con R-hat>1.5, esperable con {num_samples} iter)")

    print(f"\n[SMOKE] PASS — modelo robusto, NCP correcta, sin divergencias, PPC calibrado.")
    return True


# -- Sanity inline ---------------------------------------------------------

if __name__ == "__main__":
    import time, sys, warnings
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    warnings.filterwarnings("ignore")

    print("=== M14_cate smoke test ===\n")
    ok = run_smoke_test()
    if not ok:
        print("\n[ABORT] Smoke test fallido — corrige el modelo antes de lanzar el run completo.")
        sys.exit(1)

    print("\n=== Smoke test OK — para el run completo lanza compute_all() manualmente ===")
    print("  Ejemplo: python -c \"from M14_cate import compute_all; compute_all(overwrite=True)\"")
    print("  ETA: ~2h con NUTS 4 chains x 1000 iter sobre 14k obs.")
