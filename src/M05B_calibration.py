"""M05B PSxG calibration diagnostics SOTA.

Toma el modelo + calibrador entrenado por M05 y produce:
- calibration_curve.parquet : (bin, pred_mean, frac_positive, n_shots) sobre OOF + WC22 holdout
- brier_decomposition.parquet : Brier = Reliability - Resolution + Uncertainty (Murphy 1973)
- calibration_metrics.parquet : ECE, MCE, Brier, AUC sobre OOF + WC22 holdout
- isotonic_curve.parquet : (raw_pred, calibrated_pred) del isotonic mapping

Outputs en data/parquet/derived/psxg/calibration/ — calibration plots para
analisis de la cabeza PSxG (M05).
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import KFold

_REPO = Path(__file__).resolve().parents[1]
_PSXG_DIR = _REPO / "data" / "parquet" / "derived" / "psxg"
_OUT_DIR = _PSXG_DIR / "calibration"
_OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_model() -> tuple:
    with open(_PSXG_DIR / "model" / "psxg_lgb.pkl", "rb") as f:
        fit = pickle.load(f)
    return fit


def _calibration_curve(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pl.DataFrame:
    """Calibration curve via cuantil-binning (más robusta que equal-width).

    Devuelve DataFrame: (bin, n, pred_mean, frac_positive, ece_contribution).
    """
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(p, quantiles))
    rows = []
    n = len(p)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if mask.sum() == 0:
            continue
        pm = float(p[mask].mean())
        fp = float(y[mask].mean())
        rows.append(dict(bin=i, lo=float(lo), hi=float(hi),
                         n=int(mask.sum()), pred_mean=pm, frac_positive=fp,
                         abs_dev=abs(pm - fp),
                         ece_contribution=(mask.sum() / n) * abs(pm - fp)))
    return pl.DataFrame(rows)


def _brier_decomposition(p: np.ndarray, y: np.ndarray,
                          n_bins: int = 10) -> dict:
    """Murphy (1973) decomposition: BS = Reliability - Resolution + Uncertainty.

    - Reliability (REL): expected miscalibration. Lower is better.
    - Resolution (RES): how much predictions discriminate vs base rate. Higher better.
    - Uncertainty (UNC): irreducible variance = p_bar*(1-p_bar). Constant for given y.
    """
    bs = float(brier_score_loss(y, p))
    p_bar = float(y.mean())
    unc = p_bar * (1 - p_bar)
    rel = 0.0
    res = 0.0
    n = len(p)
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(p, quantiles))
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi) if i == len(edges) - 2 else \
               (p >= lo) & (p < hi)
        nk = mask.sum()
        if nk == 0:
            continue
        ok = float(y[mask].mean())     # observed freq in bin
        pk = float(p[mask].mean())     # predicted mean in bin
        rel += (nk / n) * (pk - ok) ** 2
        res += (nk / n) * (ok - p_bar) ** 2
    return dict(brier=bs, reliability=rel, resolution=res,
                uncertainty=unc, residual=bs - (rel - res + unc))


def _ece_mce(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> tuple:
    """Expected + Maximum Calibration Error (uniform bins)."""
    bins = np.linspace(0, 1, n_bins + 1)
    n = len(p)
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        dev = abs(p[mask].mean() - y[mask].mean())
        ece += (mask.sum() / n) * dev
        mce = max(mce, dev)
    return float(ece), float(mce)


def compute_all() -> None:
    fit = _load_model()
    feature_cols = fit["feature_cols"]
    model = fit["model"]
    calibrator = fit["calibrator"]

    # --- OOF (training) ---
    train = pl.read_parquet(_PSXG_DIR / "training_shots.parquet")
    X_tr = train.select(feature_cols).to_numpy().astype(np.float32)
    y_tr = train["_label"].to_numpy()
    sb_xg_tr = train["_sb_xg"].to_numpy()

    # Recompute OOF (5-fold CV) via the same approach as M05
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_raw = np.zeros(len(y_tr))
    import lightgbm as lgb
    best_params = {k: v for k, v in fit["metrics"]["best_params"].items()}
    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_tr)):
        m = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof_raw[val_idx] = m.predict_proba(X_tr[val_idx])[:, 1]
    oof_cal = calibrator.predict(oof_raw)

    # --- WC22 holdout ---
    wc22 = pl.read_parquet(_PSXG_DIR / "wc22_shots.parquet")
    X_wc = wc22.select(feature_cols).to_numpy().astype(np.float32)
    y_wc = wc22["_label"].to_numpy()
    sb_xg_wc = wc22["_sb_xg"].to_numpy()
    pred_raw_wc = model.predict_proba(X_wc)[:, 1]
    pred_cal_wc = calibrator.predict(pred_raw_wc)

    # --- Calibration curves ---
    curves = []
    for label, p, y in [
        ("oof_psxg_calibrated", oof_cal, y_tr),
        ("oof_psxg_raw",        oof_raw, y_tr),
        ("oof_sb_xg",           sb_xg_tr, y_tr),
        ("wc22_psxg_calibrated", pred_cal_wc, y_wc),
        ("wc22_psxg_raw",        pred_raw_wc, y_wc),
        ("wc22_sb_xg",           sb_xg_wc, y_wc),
    ]:
        c = _calibration_curve(p, y).with_columns(pl.lit(label).alias("model"))
        curves.append(c)
    curve_df = pl.concat(curves, how="diagonal")
    curve_df.write_parquet(_OUT_DIR / "calibration_curve.parquet")
    print(f"[curve] Saved calibration_curve.parquet ({curve_df.height} rows)")

    # --- Metrics combined ---
    rows = []
    for label, p, y in [
        ("oof_psxg_calibrated", oof_cal, y_tr),
        ("oof_psxg_raw",        oof_raw, y_tr),
        ("oof_sb_xg",           sb_xg_tr, y_tr),
        ("wc22_psxg_calibrated", pred_cal_wc, y_wc),
        ("wc22_psxg_raw",        pred_raw_wc, y_wc),
        ("wc22_sb_xg",           sb_xg_wc, y_wc),
    ]:
        ece, mce = _ece_mce(p, y)
        decomp = _brier_decomposition(p, y)
        rows.append(dict(
            model=label, n=len(y), n_pos=int(y.sum()), pos_rate=float(y.mean()),
            auc=float(roc_auc_score(y, p)),
            brier=decomp["brier"],
            reliability=decomp["reliability"],
            resolution=decomp["resolution"],
            uncertainty=decomp["uncertainty"],
            log_loss=float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
            ece=ece, mce=mce,
        ))
    metrics = pl.DataFrame(rows)
    metrics.write_parquet(_OUT_DIR / "calibration_metrics.parquet")
    print(f"[metrics] Saved calibration_metrics.parquet ({metrics.height} rows)")
    print(metrics.select(["model", "n", "auc", "brier", "ece",
                          "reliability", "resolution"]))

    # --- Brier decomposition table ---
    decomp_df = metrics.select(["model", "brier", "reliability",
                                "resolution", "uncertainty"])
    decomp_df.write_parquet(_OUT_DIR / "brier_decomposition.parquet")
    print(f"[decomp] Saved brier_decomposition.parquet ({decomp_df.height} rows)")

    # --- Isotonic mapping curve ---
    iso_x = np.linspace(0, 1, 1000)
    iso_y = calibrator.predict(iso_x)
    iso_df = pl.DataFrame(dict(raw_pred=iso_x, calibrated_pred=iso_y))
    iso_df.write_parquet(_OUT_DIR / "isotonic_curve.parquet")
    print(f"[iso] Saved isotonic_curve.parquet ({iso_df.height} rows)")


if __name__ == "__main__":
    compute_all()
