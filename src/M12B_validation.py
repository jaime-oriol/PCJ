"""M12 validation suite SOTA: placebo + power + naive baseline + window sensitivity.

Outputs en data/parquet/derived/did_validation/ — consumidos por M15 (power_analysis)
y M16 paper. Complementa M12_did.py con tests robustez NO cubiertos en su pipeline.

Tests implementados:
- placebo_test.parquet: permutation 1000 iter (within player-shock outcome shuffle).
  Bajo H0 los pre/post son intercambiables. Devuelve p-empirico, z-score, IC95% null.
- power_analysis.parquet: bootstrap MDE@80%, effective_n via ICC-correction,
  posterior power para el real_ate observado.
- baseline_naive.parquet: post-pre simple (within-player) vs M12 DiD ATE, z-test
  de diferencia. Magnitud relativa demuestra valor de la correccion DiD.
- window_sensitivity.parquet: re-estima ATE con ventanas +-3/5/7/10/15 minutos.
- stage_stratified.parquet: ATE separado por stage (groups vs ko).

Lee M12 panels (`panel_{ch}.parquet`) que ya traen el outcome SOTA canonico
(score_atk_v2 + un-xPass; score_def_v4 = vdep_strict + xpress + maejima;
c_obso_mean; score_phys). Para window_sensitivity construye panel propio
desde `per_minute.parquet` con los mismos outcome cols SOTA.

Uso:
    python M12B_validation.py [overwrite]
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

_REPO = Path(__file__).resolve().parents[1]
_PANEL_DIR = _REPO / "data" / "parquet" / "derived" / "did"
_OUT_DIR   = _REPO / "data" / "parquet" / "derived" / "did_validation"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = ["ataque", "defensa", "offball", "fisico"]
SHOCK_TYPES = ["GOAL_FOR", "GOAL_AGAINST"]
N_PERM = 1000
N_BOOT = 1000
SEED = 42

# Outcome col SOTA por canal (per_minute) — DEBE coincidir con M12.CHANNELS
# para que window_sensitivity sea sensitivity coherente del ATE canonico.
_OUTCOME_COL = {
    "ataque":  "score_atk_v2_minute",   # atomic-VAEP + un-xPass
    "defensa": "score_def_v4_minute",   # vdep_strict + xpress + maejima
    "offball": "c_obso_mean",            # counterfactual Teranishi 2022
    "fisico":  "score_phys",             # residual z-score multivariate
}
_PER_MIN_DIR = _REPO / "data" / "parquet" / "derived"


# ----------------------------------------------------------------------------
# Estimadores
# ----------------------------------------------------------------------------
def _within_diff_per_ps(panel: pl.DataFrame) -> tuple[np.ndarray, dict]:
    """Devuelve array de diffs (post - pre) por (player, shock).

    Drops player-shock con missing data en pre o post (no balanced).
    """
    df = panel.drop_nulls("outcome").to_pandas()
    per_ps = (df.groupby(["pff_player_id", "shock_id", "post"])["outcome"]
                .mean().unstack("post"))
    per_ps = per_ps.dropna()
    diffs = (per_ps[1] - per_ps[0]).to_numpy()
    info = dict(n_player_shock=len(diffs),
                n_unique_shocks=df["shock_id"].nunique(),
                n_unique_players=df["pff_player_id"].nunique())
    return diffs, info


def _within_ate(diffs: np.ndarray) -> dict:
    """ATE within-player: mean(diff) ± SE/CI 95%."""
    n = len(diffs)
    ate = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(n))
    return dict(
        ate=ate, se=se,
        ci_lo=ate - 1.96 * se, ci_hi=ate + 1.96 * se,
        t_stat=ate / se if se > 0 else 0.0,
        p_value=2 * (1 - stats.norm.cdf(abs(ate / se))) if se > 0 else 1.0,
    )


# ----------------------------------------------------------------------------
# Baseline naive vs M12 DiD
# ----------------------------------------------------------------------------
def baseline_naive() -> pl.DataFrame:
    rows = []
    m12 = (pl.read_parquet(_PANEL_DIR / "ate_population.parquet")
             .select(["channel", "shock_type", "ate", "se", "ci_lo", "ci_hi"])
             .rename({"ate": "m12_ate", "se": "m12_se",
                      "ci_lo": "m12_ci_lo", "ci_hi": "m12_ci_hi"}))
    for ch in CHANNELS:
        panel = pl.read_parquet(_PANEL_DIR / f"panel_{ch}.parquet")
        for sh in SHOCK_TYPES:
            sub = panel.filter(pl.col("shock_type") == sh)
            diffs, info = _within_diff_per_ps(sub)
            est = _within_ate(diffs)
            # Pooled (ignora player FE)
            df = sub.drop_nulls("outcome").to_pandas()
            pooled = (df[df.post == 1].outcome.mean() -
                      df[df.post == 0].outcome.mean())
            rows.append(dict(channel=ch, shock_type=sh,
                             naive_pooled_diff=float(pooled),
                             naive_within_ate=est["ate"],
                             naive_within_se=est["se"],
                             naive_within_ci_lo=est["ci_lo"],
                             naive_within_ci_hi=est["ci_hi"],
                             naive_within_p=est["p_value"],
                             **info))
    naive = pl.DataFrame(rows)
    out = naive.join(m12, on=["channel", "shock_type"], how="left")
    out = out.with_columns([
        ((pl.col("m12_ate") - pl.col("naive_within_ate")).abs() /
            (pl.col("naive_within_se").pow(2) + pl.col("m12_se").pow(2)).sqrt())
            .alias("z_diff_did_vs_naive"),
        (pl.col("m12_ate") / pl.col("naive_within_ate"))
            .alias("ratio_did_naive"),
    ])
    out.write_parquet(_OUT_DIR / "baseline_naive.parquet")
    print(f"[naive] Saved baseline_naive.parquet ({out.height} rows)")
    return out


# ----------------------------------------------------------------------------
# Placebo test (Fisher exact via within-series permutation)
# ----------------------------------------------------------------------------
def placebo_test(n_perm: int = N_PERM) -> pl.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for ch in CHANNELS:
        panel = pl.read_parquet(_PANEL_DIR / f"panel_{ch}.parquet")
        for sh in SHOCK_TYPES:
            sub = panel.filter(pl.col("shock_type") == sh).drop_nulls("outcome")
            diffs_real, info = _within_diff_per_ps(sub)
            est_real = _within_ate(diffs_real)

            # Construir matriz (n_ps, 21) — solo player-shocks balanced full ±10
            df = sub.to_pandas()
            df = df.drop_duplicates(["pff_player_id", "shock_id", "relative_min"])
            ps_groups = df.groupby(["pff_player_id", "shock_id"])
            outcomes_list = []
            for _, g in ps_groups:
                g = g.sort_values("relative_min")
                rm = g["relative_min"].to_numpy()
                if len(g) != 21 or rm.min() != -10 or rm.max() != 10:
                    continue
                if g["outcome"].isna().any():
                    continue
                outcomes_list.append(g["outcome"].to_numpy())
            if not outcomes_list:
                rows.append(dict(channel=ch, shock_type=sh,
                                 real_ate=est_real["ate"], n_player_shock=0,
                                 placebo_mean=np.nan, placebo_sd=np.nan,
                                 placebo_q025=np.nan, placebo_q975=np.nan,
                                 z_score_vs_placebo=np.nan,
                                 p_emp_2sided=np.nan, p_emp_1sided=np.nan,
                                 real_in_placebo_95ci=False))
                continue
            outcomes_mat = np.stack(outcomes_list)             # (N_ps, 21)
            # relative_min = arange(-10, 11): index 10 = minute 0 (shock); pre = 0..9, post = 11..20
            post_mask = np.zeros(21, dtype=bool); post_mask[11:] = True
            pre_mask = np.zeros(21, dtype=bool); pre_mask[:10] = True

            # Permutation: para cada iter, shuffle outcomes within row
            placebo_ates = np.zeros(n_perm)
            for k in range(n_perm):
                # vectorized shuffle: argsort(random)
                shuffled_idx = rng.random(outcomes_mat.shape).argsort(axis=1)
                shuffled = np.take_along_axis(outcomes_mat, shuffled_idx, axis=1)
                pre_mean = shuffled[:, pre_mask].mean(axis=1)
                post_mean = shuffled[:, post_mask].mean(axis=1)
                placebo_ates[k] = (post_mean - pre_mean).mean()

            real_ate = est_real["ate"]
            placebo_mean = float(placebo_ates.mean())
            placebo_sd = float(placebo_ates.std(ddof=1))
            z_score = (real_ate - placebo_mean) / placebo_sd if placebo_sd > 0 else 0.0
            # p-empírico Fisher: prob de placebo |t| >= |real|
            p_2s = float((np.abs(placebo_ates - placebo_mean) >=
                          np.abs(real_ate - placebo_mean)).mean())
            p_1s = (float((placebo_ates <= real_ate).mean()) if real_ate < 0
                    else float((placebo_ates >= real_ate).mean()))
            q025 = float(np.quantile(placebo_ates, 0.025))
            q975 = float(np.quantile(placebo_ates, 0.975))

            rows.append(dict(
                channel=ch, shock_type=sh,
                real_ate=real_ate, n_player_shock=outcomes_mat.shape[0],
                placebo_mean=placebo_mean, placebo_sd=placebo_sd,
                placebo_q025=q025, placebo_q975=q975,
                z_score_vs_placebo=float(z_score),
                p_emp_2sided=p_2s, p_emp_1sided=p_1s,
                real_in_placebo_95ci=bool(q025 <= real_ate <= q975),
            ))
    out = pl.DataFrame(rows)
    out.write_parquet(_OUT_DIR / "placebo_test.parquet")
    print(f"[placebo] Saved placebo_test.parquet ({out.height} rows, "
          f"n_perm={n_perm})")
    return out


# ----------------------------------------------------------------------------
# Power analysis (bootstrap MDE + effective_n + observed power)
# ----------------------------------------------------------------------------
def _icc_one_way(diffs: np.ndarray, cluster_ids: np.ndarray) -> float:
    """ICC one-way ANOVA: ratio of between-cluster variance to total."""
    df = pl.DataFrame({"y": diffs, "g": cluster_ids})
    n = len(diffs)
    grand = float(df["y"].mean())
    cluster_means = df.group_by("g").agg(
        pl.col("y").mean().alias("m"), pl.len().alias("k"))
    ms_between = float((cluster_means["k"] *
                       (cluster_means["m"] - grand) ** 2).sum() /
                       max(1, cluster_means.height - 1))
    cluster_var = (df.join(cluster_means, on="g")
                     .with_columns((pl.col("y") - pl.col("m")).pow(2)
                                   .alias("dev2")))
    ms_within = float(cluster_var["dev2"].sum() /
                      max(1, n - cluster_means.height))
    k_bar = n / cluster_means.height
    icc = (ms_between - ms_within) / (ms_between + (k_bar - 1) * ms_within)
    return max(0.0, min(1.0, icc))


def power_analysis(n_boot: int = N_BOOT) -> pl.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for ch in CHANNELS:
        panel = pl.read_parquet(_PANEL_DIR / f"panel_{ch}.parquet")
        for sh in SHOCK_TYPES:
            sub = panel.filter(pl.col("shock_type") == sh).drop_nulls("outcome")
            df = sub.to_pandas()
            per_ps = (df.groupby(["pff_player_id", "shock_id"])
                        .agg(pre=("outcome", lambda x: x[df.loc[x.index, "post"] == 0].mean()),
                             post=("outcome", lambda x: x[df.loc[x.index, "post"] == 1].mean()))
                        .dropna())
            per_ps["diff"] = per_ps["post"] - per_ps["pre"]
            diffs = per_ps["diff"].to_numpy()
            cluster_ids = per_ps.index.get_level_values("pff_player_id").to_numpy()
            n = len(diffs)
            sigma = float(diffs.std(ddof=1))
            ate_real = float(diffs.mean())
            # ICC correction
            icc = _icc_one_way(diffs, cluster_ids)
            unique_clusters = len(np.unique(cluster_ids))
            k_bar = n / unique_clusters
            deff = 1 + (k_bar - 1) * icc                          # design effect
            n_eff = n / deff
            # MDE@80% (alpha=0.05 2-sided, power=0.8 => z_alpha + z_beta = 2.80)
            mde_naive = 2.80 * sigma / np.sqrt(n)
            mde_eff = 2.80 * sigma / np.sqrt(n_eff)
            # Bootstrap power para el ate observado
            boot_t = np.zeros(n_boot)
            for k in range(n_boot):
                idx = rng.integers(0, n, n)
                bd = diffs[idx]
                boot_t[k] = bd.mean() / (bd.std(ddof=1) / np.sqrt(n)) if bd.std() > 0 else 0
            power_obs = float((np.abs(boot_t) > 1.96).mean())
            # Para detectar 0.05 SD effect, what N needed?
            n_for_05sd = (2.80 / 0.05) ** 2 * deff  # using sigma=1 (z-score canal)
            rows.append(dict(
                channel=ch, shock_type=sh,
                n_player_shock=n, n_unique_clusters=unique_clusters,
                ate_observed=ate_real, sigma_diff=sigma,
                icc=icc, design_effect=deff, n_effective=n_eff,
                mde80_naive=mde_naive, mde80_effective=mde_eff,
                power_observed=power_obs,
                n_needed_for_0_05sd=n_for_05sd,
                ate_in_sd_units=ate_real / sigma,
            ))
    out = pl.DataFrame(rows)
    out.write_parquet(_OUT_DIR / "power_analysis.parquet")
    print(f"[power] Saved power_analysis.parquet ({out.height} rows, "
          f"n_boot={n_boot})")
    return out


# ----------------------------------------------------------------------------
# Window sensitivity ±3/5/7/10/15 min con panel extendido desde per_minute
# ----------------------------------------------------------------------------
def _build_extended_window_panel(channel: str, window: int) -> pl.DataFrame:
    """Reconstruye panel con ventana arbitraria desde per_minute + shocks_table.

    relative_min en [-window, +window], excluyendo minuto 0 (shock).
    """
    pm = pl.read_parquet(_PER_MIN_DIR / channel / "per_minute.parquet")
    out_col = _OUTCOME_COL[channel]
    # Convertir a minute_global (período + minuto_in_period)
    pm = pm.with_columns(
        ((pl.col("period") - 1) * 45 + pl.col("minute_in_period"))
            .alias("minute_global"))
    pm = pm.select(["pff_match_id", "pff_player_id", "minute_global",
                    pl.col(out_col).alias("outcome")])
    # shocks
    sh = pl.read_parquet(_REPO / "data" / "parquet" / "derived" / "shocks" /
                         "shocks_table.parquet")
    # Filtrar limpios M12 (no truncated, no overlap, no sub_in_window) — mismo filtro que M12
    sh_clean = sh.filter(
        (~pl.col("truncated_pre")) & (~pl.col("truncated_post")) &
        (~pl.col("overlap_flag")) & (~pl.col("sub_in_window"))
    ).rename({"match_id": "pff_match_id", "player_id": "pff_player_id"})
    sh_clean = sh_clean.select(["pff_match_id", "shock_id", "shock_type",
                                "pff_player_id", "position_group", "minute"])
    # Cross-join con relative_min in [-window, +window] excluyendo 0
    rels = pl.DataFrame({"relative_min":
                         [r for r in range(-window, window + 1) if r != 0]})
    sh_expanded = sh_clean.join(rels, how="cross").with_columns(
        (pl.col("minute") + pl.col("relative_min")).alias("minute_global"))
    panel = sh_expanded.join(
        pm, on=["pff_match_id", "pff_player_id", "minute_global"],
        how="left").with_columns(
        (pl.col("relative_min") > 0).cast(pl.Int64).alias("post"))
    return panel


def window_sensitivity() -> pl.DataFrame:
    rows = []
    windows = [3, 5, 7, 10, 15]
    for ch in CHANNELS:
        for w in windows:
            panel = _build_extended_window_panel(ch, w)
            for sh in SHOCK_TYPES:
                sub = panel.filter(pl.col("shock_type") == sh)
                # Drop ps con outcome incompleto (al menos 1 pre + 1 post)
                diffs, info = _within_diff_per_ps(sub)
                est = _within_ate(diffs)
                rows.append(dict(channel=ch, shock_type=sh, window_min=w,
                                 ate=est["ate"], se=est["se"],
                                 ci_lo=est["ci_lo"], ci_hi=est["ci_hi"],
                                 t_stat=est["t_stat"], p_value=est["p_value"],
                                 **info))
    out = pl.DataFrame(rows)
    out.write_parquet(_OUT_DIR / "window_sensitivity.parquet")
    print(f"[window] Saved window_sensitivity.parquet ({out.height} rows; "
          f"windows {windows})")
    return out


# ----------------------------------------------------------------------------
# Multiple test correction (Benjamini-Hochberg FDR)
# ----------------------------------------------------------------------------
def _bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> tuple:
    """Benjamini-Hochberg FDR adjustment. Returns (p_adj, reject_h0)."""
    n = len(pvals)
    order = np.argsort(pvals)
    p_sorted = pvals[order]
    # adjusted = min(p[k] * n / (k+1), 1)
    adj_sorted = np.minimum.accumulate((p_sorted * n / np.arange(1, n + 1))[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0, 1)
    p_adj = np.empty(n)
    p_adj[order] = adj_sorted
    return p_adj, p_adj < alpha


def add_multiple_test_correction() -> pl.DataFrame:
    """Lee placebo + naive p-values, aplica BH-FDR, persiste."""
    placebo = pl.read_parquet(_OUT_DIR / "placebo_test.parquet")
    naive = pl.read_parquet(_OUT_DIR / "baseline_naive.parquet")
    # FDR sobre 8 hipótesis (4 channels × 2 shock_types) para cada test
    p_placebo = placebo["p_emp_2sided"].to_numpy()
    p_adj_placebo, sig_placebo = _bh_fdr(p_placebo)
    p_naive = naive["naive_within_p"].to_numpy()
    p_adj_naive, sig_naive = _bh_fdr(p_naive)
    placebo = placebo.with_columns([
        pl.Series("p_placebo_bh_fdr", p_adj_placebo),
        pl.Series("sig_placebo_bh_fdr", sig_placebo),
    ])
    naive = naive.with_columns([
        pl.Series("p_naive_bh_fdr", p_adj_naive),
        pl.Series("sig_naive_bh_fdr", sig_naive),
    ])
    placebo.write_parquet(_OUT_DIR / "placebo_test.parquet")
    naive.write_parquet(_OUT_DIR / "baseline_naive.parquet")
    print(f"[FDR] Aplicado BH-FDR a placebo + naive (n={len(p_placebo)} hyp)")
    return placebo


# ----------------------------------------------------------------------------
# Stage-stratified ATE (groups vs KO)
# ----------------------------------------------------------------------------
def stage_stratified_ate() -> pl.DataFrame:
    """Estima ATE separado por stage (groups vs KO).

    Hipotesis: shocks en KO tienen efecto mayor por mayor leverage (eliminacion).
    """
    # Cargar shocks_table con stage
    sh = pl.read_parquet(_REPO / "data" / "parquet" / "derived" / "shocks" /
                         "shocks_table.parquet").select(["shock_id", "stage"])
    rows = []
    for ch in CHANNELS:
        panel = pl.read_parquet(_PANEL_DIR / f"panel_{ch}.parquet")
        panel = panel.join(sh, on="shock_id", how="left")
        for sh_type in SHOCK_TYPES:
            for stage in ["groups", "ko", "all"]:
                sub = panel.filter(pl.col("shock_type") == sh_type)
                if stage != "all":
                    sub = sub.filter(pl.col("stage") == stage)
                diffs, info = _within_diff_per_ps(sub)
                if len(diffs) == 0:
                    continue
                est = _within_ate(diffs)
                rows.append(dict(channel=ch, shock_type=sh_type, stage=stage,
                                 ate=est["ate"], se=est["se"],
                                 ci_lo=est["ci_lo"], ci_hi=est["ci_hi"],
                                 t_stat=est["t_stat"], p_value=est["p_value"],
                                 **info))
    out = pl.DataFrame(rows)
    out.write_parquet(_OUT_DIR / "stage_stratified.parquet")
    print(f"[stage] Saved stage_stratified.parquet ({out.height} rows)")
    return out


# ----------------------------------------------------------------------------
def main(overwrite: bool = False):
    print("=== M12 validation suite SOTA ===\n")
    naive = baseline_naive()
    print(naive.select(["channel", "shock_type", "naive_within_ate", "m12_ate",
                        "ratio_did_naive", "z_diff_did_vs_naive"]))
    print()
    placebo = placebo_test()
    print(placebo.select(["channel", "shock_type", "real_ate",
                          "placebo_mean", "z_score_vs_placebo",
                          "p_emp_2sided", "real_in_placebo_95ci"]))
    print()
    power = power_analysis()
    print(power.select(["channel", "shock_type", "ate_in_sd_units",
                        "icc", "n_effective", "mde80_effective",
                        "power_observed"]))
    print()
    window = window_sensitivity()
    print(window.pivot(on="window_min", index=["channel", "shock_type"],
                       values="ate"))
    print()
    add_multiple_test_correction()
    placebo_after = pl.read_parquet(_OUT_DIR / "placebo_test.parquet")
    print(placebo_after.select(["channel", "shock_type", "real_ate",
                                "p_emp_2sided", "p_placebo_bh_fdr",
                                "sig_placebo_bh_fdr"]))
    print()
    stage = stage_stratified_ate()
    print(stage.pivot(on="stage", index=["channel", "shock_type"],
                      values="ate"))
    print("\n=== Validation suite OK — outputs en data/parquet/derived/did_validation/ ===")


if __name__ == "__main__":
    overwrite = "overwrite" in sys.argv
    main(overwrite=overwrite)
