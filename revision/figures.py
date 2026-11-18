"""Figures for the revised manuscript, drawn only from logged results.

  fig_adaptation_curves.png  training vs adaptation-validation loss/accuracy
                             (the validation data are the adapt_val subjects,
                             never the test subjects); dotted line = epoch kept
                             by early stopping
  fig_source_cv_curves.png   training vs inner-validation loss per CV fold
  fig_confusion.png          confusion matrices of the hybrid model, adapted,
                             with and without TTA, summed over repeats
  fig_reliability.png        reliability diagram, hybrid adapted, +-TTA
  fig_ablation.png           independently trained models vs stream masking

Usage:
  python revision/figures.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 10, "font.family": "serif"})


def adaptation_curves(target, out, arch="hybrid"):
    runs = pd.read_csv(target / "adaptation_runs.csv")
    files = sorted(target.glob(f"history_r*_{arch}.csv"))
    if not files:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for f in files:
        r = int(f.stem.split("_")[1][1:])
        h = pd.read_csv(f)
        best = runs[(runs.repeat == r) & (runs.arch == arch)].best_epoch.iloc[0]
        l, = a1.plot(h.epoch, h.train_loss, "-", lw=1, label=f"repeat {r} train")
        a1.plot(h.epoch, h.val_loss, "--", lw=1, color=l.get_color(), label=f"repeat {r} adapt-val")
        a1.axvline(best, color=l.get_color(), ls=":", lw=0.8)
        a2.plot(h.epoch, h.train_acc, "-", lw=1, color=l.get_color())
        a2.plot(h.epoch, h.val_acc, "--", lw=1, color=l.get_color())
    a1.set(xlabel="Epoch", ylabel="Loss")
    a2.set(xlabel="Epoch", ylabel="Accuracy")
    a1.legend(fontsize=7, ncol=2)
    for a in (a1, a2):
        a.grid(alpha=0.3, ls=":")
    plt.tight_layout()
    plt.savefig(out / "fig_adaptation_curves.png", dpi=300)
    plt.close()


def source_curves(source, out, arch="hybrid"):
    files = sorted((source / arch).glob("fold*_history.csv"))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for f in files:
        h = pd.read_csv(f)
        l, = ax.plot(h.epoch, h.train_loss, "-", lw=1, label=f"{f.stem.split('_')[0]} train")
        ax.plot(h.epoch, h.val_loss, "--", lw=1, color=l.get_color(), label=f"{f.stem.split('_')[0]} inner-val")
    ax.set(xlabel="Epoch", ylabel="Loss")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, ls=":")
    plt.tight_layout()
    plt.savefig(out / "fig_source_cv_curves.png", dpi=300)
    plt.close()


def confusion(tables, out):
    cm = pd.read_csv(tables / "confusion_matrices.csv")
    sel = cm[(cm.arch == "hybrid") & (cm.stage == "adapted") & (cm.masking == "none")]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, tta in zip(axes, (0, 1)):
        s = sel[sel.tta == tta][["TN", "FP", "FN", "TP"]].sum()
        m = np.array([[s.TN, s.FP], [s.FN, s.TP]])
        ax.imshow(m, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(m[i, j]), ha="center", va="center",
                        color="white" if m[i, j] > m.max() / 2 else "black", fontsize=12, weight="bold")
        ax.set_xticks([0, 1], ["HEM", "ALL"])
        ax.set_yticks([0, 1], ["HEM", "ALL"])
        ax.set(xlabel="Predicted", ylabel="True", title=f"Hybrid, adapted, {'with' if tta else 'without'} TTA")
    plt.tight_layout()
    plt.savefig(out / "fig_confusion.png", dpi=300)
    plt.close()


def reliability(tables, out):
    rel = pd.read_csv(tables / "reliability.csv")
    sel = rel[(rel.arch == "hybrid") & (rel.stage == "adapted")]
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="perfect calibration")
    for tta, g in sel.groupby("tta"):
        g = g.groupby(["bin_lo"]).apply(lambda d: pd.Series({
            "mean_pred": np.average(d.mean_pred, weights=d.n), "frac_pos": np.average(d.frac_pos, weights=d.n)}))
        ax.plot(g.mean_pred, g.frac_pos, "o-", label=f"{'with' if tta else 'without'} TTA")
    ax.set(xlabel="Mean predicted P(ALL)", ylabel="Observed fraction ALL", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, ls=":")
    plt.tight_layout()
    plt.savefig(out / "fig_reliability.png", dpi=300)
    plt.close()


def ablation(tables, out):
    per = pd.read_csv(tables / "target_metrics_per_repeat.csv")
    sel = per[(per.tta == 0)]
    labels, vals, errs = [], [], []
    for stage in ("zero_shot", "adapted"):
        for arch, masking, name in [("cnn", "none", "EfficientNet-B0 (trained alone)"),
                                    ("vit", "none", "ViT-B/16 (trained alone)"),
                                    ("hybrid", "cnn_stream_only", "Hybrid, ViT stream masked"),
                                    ("hybrid", "vit_stream_only", "Hybrid, CNN stream masked"),
                                    ("hybrid", "none", "Hybrid")]:
            g = sel[(sel.stage == stage) & (sel.arch == arch) & (sel.masking == masking)]["MCC"]
            if len(g):
                labels.append(f"{stage.replace('_', '-')}: {name}")
                vals.append(g.mean())
                errs.append(g.std(ddof=1) if len(g) > 1 else 0)
    fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(labels) + 1))
    ax.barh(range(len(labels)), vals, xerr=errs, color="#8da0cb")
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("MCC (mean $\\pm$ SD over repeats)")
    ax.grid(axis="x", alpha=0.3, ls=":")
    plt.tight_layout()
    plt.savefig(out / "fig_ablation.png", dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="revision/results/source")
    ap.add_argument("--target", default="revision/results/target")
    ap.add_argument("--tables", default="revision/results/tables")
    ap.add_argument("--out", default="revision/results/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    adaptation_curves(Path(a.target), out)
    source_curves(Path(a.source), out)
    confusion(Path(a.tables), out)
    reliability(Path(a.tables), out)
    ablation(Path(a.tables), out)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
