"""Quantify colour/illumination differences between the C-NMC training phase
(source) and the preliminary-test phase (target). No GPU needed.

Per image (cell pixels only; C-NMC cells are segmented on a black background):
mean and SD of CIE L*, a*, b* and the cell-area fraction.

Reported:
  * per phase and class: mean +- SD of each feature
  * standardised mean difference (Cohen's d) source vs target, within class
  * a "phase classifier": logistic regression predicting the phase from these
    7 colour features, subject-grouped 5-fold CV, ROC AUC. AUC near 0.5 means
    the phases are not separable by colour statistics; a high AUC quantifies a
    colour shift.

Usage:
  python revision/color_stats.py --data-root DATA
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import load_index

FEATS = ["L_mean", "a_mean", "b_mean", "L_sd", "a_sd", "b_sd", "area_frac"]


def rgb_to_lab(rgb):
    rgb = rgb / 255.0
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)


def image_features(path, size=225):
    img = np.asarray(Image.open(path).convert("RGB").resize((size, size)), dtype=np.float64)
    fg = img.max(-1) > 10
    if fg.sum() < 50:
        fg = np.ones_like(fg)
    lab = rgb_to_lab(img)[fg]
    return dict(zip(FEATS, [*lab.mean(0), *lab.std(0), float(fg.mean())]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits", default="revision/splits")
    ap.add_argument("--out", default="revision/results/tables")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    src = load_index(Path(a.splits) / "source_index.csv").assign(phase="source")
    tgt = load_index(Path(a.splits) / "target_index.csv").assign(phase="target")
    df = pd.concat([src, tgt], ignore_index=True)
    feats = pd.DataFrame([image_features(Path(a.data_root) / p) for p in df["path"]])
    df = pd.concat([df[["name", "subject", "label", "phase"]], feats], axis=1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "color_features_per_image.csv", index=False)

    summary = df.groupby(["phase", "label"])[FEATS].agg(["mean", "std"])
    summary.to_csv(out / "color_summary.csv")

    d_rows = []
    for label, g in df.groupby("label"):
        s, t = g[g.phase == "source"], g[g.phase == "target"]
        for f in FEATS:
            pooled = np.sqrt((s[f].var(ddof=1) + t[f].var(ddof=1)) / 2)
            d_rows.append({"label": label, "feature": f, "cohens_d": (t[f].mean() - s[f].mean()) / pooled})
    pd.DataFrame(d_rows).to_csv(out / "color_cohens_d.csv", index=False)

    y = (df.phase == "target").astype(int).to_numpy()
    strat = y * 2 + df.label.to_numpy()
    oof = np.zeros(len(df))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=a.seed).split(df, strat, groups=df.subject):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        clf.fit(df.loc[tr, FEATS], y[tr])
        oof[te] = clf.predict_proba(df.loc[te, FEATS])[:, 1]
    res = {"phase_classifier_auc_all": roc_auc_score(y, oof)}
    for label in (0, 1):
        m = df.label.to_numpy() == label
        res[f"phase_classifier_auc_label{label}"] = roc_auc_score(y[m], oof[m])
    pd.Series(res).to_csv(out / "color_phase_classifier.csv")
    print(pd.Series(res))


if __name__ == "__main__":
    main()
