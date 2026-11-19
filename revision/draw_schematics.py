"""Draw the two schematic figures of the revised manuscript (no data needed).

  fig_protocol.png      subject-level evaluation protocol
  fig_architecture.png  dual-stream model and single-backbone baselines

Usage: python revision/draw_schematics.py --out revision/results/figures
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

C_SRC, C_TGT, C_TEST, C_MODEL, C_GREY = "#dbe9f6", "#fde9d4", "#f8d7da", "#e2f0d9", "#eeeeee"


def box(ax, x, y, w, h, text, fc, ec="#333333", fs=9, bold=False, lw=1.0, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            weight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1, text=None, style="-|>", ls="-", color="#333333"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=12,
                                 lw=1.1, ls=ls, color=color))
    if text:
        ax.text((x0 + x1) / 2 + 0.08, (y0 + y1) / 2, text, fontsize=7.5, va="center", color=color)


def protocol(out):
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    # Stage labels
    for y, t in [(5.75, "Stage 1  Source phase (C-NMC training phase)"),
                 (3.55, "Stage 2  Target phase split (C-NMC preliminary-test phase), repeated 5 times"),
                 (1.55, "Stage 3  Evaluation on held-out test subjects")]:
        ax.text(0.05, y, t, fontsize=9.5, weight="bold")

    # Stage 1
    box(ax, 0.1, 4.3, 2.3, 1.2, "73 subjects\n(47 ALL / 26 HEM)\n10,661 cells", C_SRC, bold=True)
    box(ax, 2.9, 4.3, 3.2, 1.2, "Subject-level 5-fold CV\n(StratifiedGroupKFold)\ninner validation = 10% of training\nsubjects -> early stopping", C_SRC)
    box(ax, 6.6, 4.3, 3.3, 1.2, "Final source model\ntrained on all 73 subjects\nfor median CV epoch\n(no fold selection)", C_MODEL, bold=False)
    arrow(ax, 2.4, 4.9, 2.9, 4.9)
    arrow(ax, 6.1, 4.9, 6.6, 4.9)
    ax.text(4.5, 4.12, "out-of-fold predictions -> CV metrics, secondary threshold (Youden J)",
            fontsize=7.5, ha="center", style="italic")

    # Stage 2
    box(ax, 0.1, 2.1, 2.3, 1.2, "28 subjects\n(13 ALL / 15 HEM)\n1,867 cells", C_TGT, bold=True)
    box(ax, 2.9, 2.1, 2.2, 1.2, "Adaptation-train\n4 subjects\n(2 ALL / 2 HEM)", C_TGT)
    box(ax, 5.3, 2.1, 2.0, 1.2, "Adaptation-val\n2 subjects\n(1 ALL / 1 HEM)\nearly stopping only", C_TGT, fs=8.5)
    box(ax, 7.5, 2.1, 2.4, 1.2, "Held-out test\n22 subjects\n(10 ALL / 12 HEM)\nnever used for selection", C_TEST, bold=True, fs=8.5)
    arrow(ax, 2.4, 2.7, 2.9, 2.7)

    # Stage 3 (both evaluations start from the final source model of Stage 1)
    box(ax, 2.9, 0.15, 4.4, 1.2, "Adapted: start from final source model,\nfine-tune on adaptation-train,\nstop on adaptation-val loss\n(with / without TTA)", C_MODEL, fs=8.5)
    box(ax, 7.5, 0.15, 2.4, 1.2, "Zero-shot: final source\nmodel, no update\n(with / without TTA)", C_MODEL, fs=8.5)
    arrow(ax, 4.0, 2.1, 4.0, 1.35, ls="--")
    arrow(ax, 6.3, 2.1, 6.3, 1.35, ls="--")
    arrow(ax, 8.7, 2.1, 8.7, 1.35, color="#b71c1c")
    arrow(ax, 7.9, 2.1, 7.1, 1.35, color="#b71c1c")
    ax.text(0.15, 0.25, "dashed: training /\nearly stopping\nred: final evaluation\nonly", fontsize=7.5, color="#555555")
    ax.text(9.95, -0.12, "Same pipeline for the dual-stream model, EfficientNet-B0 alone and ViT-B/16 alone",
            ha="right", fontsize=7.5, style="italic")
    plt.savefig(out / "fig_protocol.png", dpi=300, bbox_inches="tight")
    plt.close()


def architecture(out):
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    box(ax, 0.1, 2.3, 1.6, 1.0, "Cell image\n3 x 224 x 224", C_GREY, bold=True)
    box(ax, 2.2, 3.6, 2.6, 1.1, "EfficientNet-B0\nImageNet init.,\nall layers trained\nglobal avg. pool", C_SRC, fs=8.5)
    box(ax, 2.2, 0.9, 2.6, 1.1, "ViT-B/16\nImageNet init.,\nall layers trained\n[CLS] token", C_TGT, fs=8.5)
    arrow(ax, 1.7, 2.9, 2.2, 4.1)
    arrow(ax, 1.7, 2.7, 2.2, 1.45)
    box(ax, 5.1, 3.75, 1.1, 0.8, r"$v_{cnn}$" + "\n1280", C_SRC)
    box(ax, 5.1, 1.05, 1.1, 0.8, r"$v_{vit}$" + "\n768", C_TGT)
    arrow(ax, 4.8, 4.15, 5.1, 4.15)
    arrow(ax, 4.8, 1.45, 5.1, 1.45)
    box(ax, 6.55, 2.35, 1.0, 0.9, "concat\n2048", C_MODEL, bold=True)
    arrow(ax, 6.2, 4.15, 6.95, 3.25)
    arrow(ax, 6.2, 1.45, 6.95, 2.35)
    box(ax, 7.8, 1.6, 1.15, 2.4, "MLP head\n\nLinear 512\nBN, ReLU\nDropout 0.4\n\nLinear 128\nBN, ReLU\n\nLinear 2\nsoftmax\n-> P(ALL)", C_MODEL, fs=7.5)
    arrow(ax, 7.55, 2.8, 7.8, 2.8)
    ax.text(4.5, 5.35, "Dual-stream model", ha="center", fontsize=11, weight="bold")
    ax.text(4.5, 0.25,
            "Baselines: the same head on EfficientNet-B0 alone (1280 -> 512 -> 128 -> 2) or ViT-B/16 alone "
            "(768 -> 512 -> 128 -> 2), trained independently.\n"
            "Stream masking (secondary analysis): one feature vector of the trained dual-stream model set to zero at inference.",
            ha="center", fontsize=7.5, style="italic")
    plt.savefig(out / "fig_architecture.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="revision/results/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    protocol(out)
    architecture(out)
    print(f"Schematics written to {out}")


if __name__ == "__main__":
    main()
