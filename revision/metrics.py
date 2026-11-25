"""Evaluation metrics, calibration, subject-level bootstrap and paired tests.

Positive class = ALL (label 1). Class-specific metrics are reported for ALL
(sensitivity, PPV, F1_ALL) and HEM (specificity, NPV, F1_HEM); macro averages
are reported separately and labelled as such.
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, roc_curve


def _div(a, b):
    return float(a) / float(b) if b else float("nan")


def ece_score(y, p, n_bins=15):
    """Expected calibration error of the predicted (top-label) class, equal-width bins."""
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_table(y, p, n_bins=10):
    """Fraction of positives vs mean predicted P(ALL) per bin (for the reliability diagram)."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            rows.append({"bin_lo": lo, "bin_hi": hi, "n": int(m.sum()),
                         "mean_pred": float(p[m].mean()), "frac_pos": float(y[m].mean())})
    return pd.DataFrame(rows)


def binary_metrics(y, p, thr=0.5):
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    yhat = (p >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    sens, spec = _div(tp, tp + fn), _div(tn, tn + fp)
    ppv, npv = _div(tp, tp + fp), _div(tn, tn + fn)
    f1_all = _div(2 * ppv * sens, ppv + sens) if tp else 0.0
    f1_hem = _div(2 * npv * spec, npv + spec) if tn else 0.0
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    try:
        auc = roc_auc_score(y, p)
    except ValueError:
        auc = float("nan")
    return {
        "n": len(y), "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "accuracy": _div(tp + tn, len(y)), "balanced_accuracy": (sens + spec) / 2,
        "sensitivity_ALL": sens, "specificity": spec, "PPV_ALL": ppv, "NPV": npv,
        "F1_ALL": f1_all, "F1_HEM": f1_hem,
        "macro_precision": (ppv + npv) / 2, "macro_recall": (sens + spec) / 2,
        "macro_F1": (f1_all + f1_hem) / 2, "MCC": float(mcc), "AUC": float(auc),
        "Brier": float(np.mean((p - y) ** 2)), "ECE": ece_score(y, p),
    }


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def _subject_resample(df, rng):
    """Resample subjects with replacement, within each class, keeping the
    number of ALL and HEM subjects fixed. Returns row indices."""
    parts = []
    for _, g in df.groupby("label"):
        subs = g["subject"].unique()
        pick = rng.choice(subs, size=len(subs), replace=True)
        groups = g.groupby("subject").indices
        parts.extend(g.index.to_numpy()[groups[s]] for s in pick)
    return np.concatenate(parts)


def subject_bootstrap_ci(df, prob_col="p", thr=0.5, n_boot=2000, seed=0, alpha=0.05):
    """Percentile CIs for every metric, resampling SUBJECTS (cluster bootstrap)."""
    rng = np.random.default_rng(seed)
    point = binary_metrics(df["label"], df[prob_col], thr)
    draws = []
    for _ in range(n_boot):
        idx = _subject_resample(df, rng)
        d = df.loc[idx]
        draws.append(binary_metrics(d["label"], d[prob_col], thr))
    draws = pd.DataFrame(draws)
    out = {}
    for k, v in point.items():
        if k in ("n", "TP", "FN", "FP", "TN"):
            out[k] = v
            continue
        lo, hi = np.nanpercentile(draws[k], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[k] = v
        out[k + "_lo"], out[k + "_hi"] = float(lo), float(hi)
    return out


def paired_comparison(df, col_a, col_b, thr=0.5, metrics=("accuracy", "MCC", "macro_F1", "sensitivity_ALL", "ECE"),
                      n_boot=2000, seed=0):
    """Model A vs model B on the SAME images: exact McNemar test on accuracy and
    a paired subject-level bootstrap CI of the difference (A - B) for each metric."""
    y = df["label"].to_numpy()
    ca = ((df[col_a].to_numpy() >= thr).astype(int) == y)
    cb = ((df[col_b].to_numpy() >= thr).astype(int) == y)
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p_mcnemar = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    rng = np.random.default_rng(seed)
    point = {m: binary_metrics(y, df[col_a], thr)[m] - binary_metrics(y, df[col_b], thr)[m] for m in metrics}
    draws = {m: [] for m in metrics}
    for _ in range(n_boot):
        d = df.loc[_subject_resample(df, rng)]
        ma, mb = binary_metrics(d["label"], d[col_a], thr), binary_metrics(d["label"], d[col_b], thr)
        for m in metrics:
            draws[m].append(ma[m] - mb[m])
    out = {"A": col_a, "B": col_b, "A_only_correct": b, "B_only_correct": c, "mcnemar_p": float(p_mcnemar)}
    for m in metrics:
        lo, hi = np.nanpercentile(draws[m], [2.5, 97.5])
        out[f"delta_{m}"], out[f"delta_{m}_lo"], out[f"delta_{m}_hi"] = point[m], float(lo), float(hi)
    return out


def subject_level(df, prob_col="p", thr=0.5):
    """Exploratory subject-level read-out: a subject is called ALL when the mean
    P(ALL) over its cells is >= thr. (C-NMC subjects carry a single label.)"""
    g = df.groupby("subject").agg(label=("label", "first"), mean_p=(prob_col, "mean"),
                                  n_cells=(prob_col, "size"),
                                  frac_cells_ALL=(prob_col, lambda s: float((s >= thr).mean())))
    g["pred"] = (g["mean_p"] >= thr).astype(int)
    return g.reset_index()
