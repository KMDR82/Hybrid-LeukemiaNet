"""Write the LaTeX tables of the revised manuscript from the result CSV files.

  python revision/make_tex_tables.py --results revision/results --splits revision/splits --out revision/results/tex

Outputs (one .tex fragment per table, each a bare tabular):
  paired_summary.tex   paired comparisons summarised over the five target splits
  paired_full.tex      every split (supplement)
  compute.tex          computational cost
  split_counts.tex     subjects and cells of every target split (supplement)
  epochs.tex           selected epochs of every source fold and target split (supplement)
  target_ci.tex        per-split metrics with subject-bootstrap CIs (supplement)
  color.tex            L*a*b* statistics by phase and class (supplement)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

NAME = {"hybrid": "Dual-stream", "cnn": "EfficientNet-B0", "vit": "ViT-B/16"}


def f3(x):
    return "--" if pd.isna(x) else f"{x:.3f}"


def fp(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def paired(res, out):
    c = pd.read_csv(res / "tables" / "paired_comparisons.csv")
    rows = []
    order = []
    for stage, lab in (("zero_shot", "Zero-shot"), ("adapted", "Adapted")):
        order += [(f"hybrid_{stage}_0", f"cnn_{stage}_0", f"{lab}: dual-stream $-$ EfficientNet-B0"),
                  (f"hybrid_{stage}_0", f"vit_{stage}_0", f"{lab}: dual-stream $-$ ViT-B/16"),
                  (f"hybrid_{stage}_1", f"hybrid_{stage}_0", f"{lab}: dual-stream, TTA $-$ no TTA")]
    for a in ("hybrid", "cnn", "vit"):
        order.append((f"{a}_adapted_0", f"{a}_zero_shot_0", f"{NAME[a]}: adapted $-$ zero-shot"))
    for a, b, lab in order:
        g = c[(c.A == a) & (c.B == b)]
        if g.empty:
            continue
        up = int((g.delta_MCC_lo > 0).sum())
        dn = int((g.delta_MCC_hi < 0).sum())
        rows.append(f"{lab} & {g.delta_MCC.mean():+.3f} & [{g.delta_MCC.min():+.3f}, {g.delta_MCC.max():+.3f}] & "
                    f"{up} / {dn} / {len(g) - up - dn} & {g.delta_sensitivity_ALL.mean():+.3f} & "
                    f"{g.delta_ECE.mean():+.3f} & {fp(g.mcnemar_p.median())} \\\\")
    head = ("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
            "Comparison (A $-$ B) & mean $\\Delta$MCC & range over splits & CI $>$0 / $<$0 / incl.\\ 0 & "
            "mean $\\Delta$Sens. & mean $\\Delta$ECE & median McNemar $p$ \\\\ \\midrule\n")
    (out / "paired_summary.tex").write_text(head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    full = []
    for a, b, lab in order:
        for _, r in c[(c.A == a) & (c.B == b)].iterrows():
            full.append(f"{lab} & {int(r["repeat"]) + 1} & {r.delta_MCC:+.3f} [{r.delta_MCC_lo:+.3f}, {r.delta_MCC_hi:+.3f}] & "
                        f"{r.delta_accuracy:+.3f} [{r.delta_accuracy_lo:+.3f}, {r.delta_accuracy_hi:+.3f}] & "
                        f"{r.delta_ECE:+.3f} [{r.delta_ECE_lo:+.3f}, {r.delta_ECE_hi:+.3f}] & "
                        f"{int(r.A_only_correct)} / {int(r.B_only_correct)} & {fp(r.mcnemar_p)} \\\\")
    head = ("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
            "Comparison (A $-$ B) & Split & $\\Delta$MCC [95\\% CI] & $\\Delta$Accuracy [95\\% CI] & "
            "$\\Delta$ECE [95\\% CI] & A only / B only correct & McNemar $p$ \\\\ \\midrule\n")
    (out / "paired_full.tex").write_text(head + "\n".join(full) + "\n\\bottomrule\n\\end{tabular}\n")


def compute(res, out):
    d = pd.read_csv(res / "tables" / "compute_cost.csv").set_index("arch")
    rows = []
    for a in ("cnn", "vit", "hybrid"):
        r = d.loc[a]
        rows.append(f"{NAME[a]} & {r.params_total_M:.1f} & {r.params_trainable_M:.1f} & {r.GFLOPs_per_image_torch_counter:.2f} & "
                    f"{r.peak_mem_MB_infer_b32:.0f} & {r.peak_mem_MB_train_b32:.0f} & "
                    f"{r.latency_ms_b1_k1:.1f} & {r.latency_ms_b1_k3:.1f} & "
                    f"{r.ms_per_image_b32_k1:.2f} & {r.ms_per_image_b32_k3:.2f} \\\\")
    head = ("\\begin{tabular}{@{}lccccccccc@{}}\n\\toprule\n"
            " & \\multicolumn{2}{c}{Parameters (M)} & GFLOPs & \\multicolumn{2}{c}{Peak memory (MB), batch 32} & "
            "\\multicolumn{2}{c}{ms / image, batch 1} & \\multicolumn{2}{c}{ms / image, batch 32} \\\\\n"
            "Model & total & trainable & per image & inference & training & $K{=}1$ & TTA $K{=}3$ & $K{=}1$ & TTA $K{=}3$ \\\\ \\midrule\n")
    (out / "compute.tex").write_text(head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def split_counts(splits, out):
    s = json.loads((splits / "split_summary.json").read_text())["target_repeats"]
    rows = []
    for r in sorted(s, key=int):
        cells = []
        for role in ("adapt_train", "adapt_val", "test"):
            x = s[r][role]
            cells.append(f"{x['images_ALL']} / {x['images_HEM']}")
        ids = ", ".join(s[r]["adapt_train"]["subject_ids"]) + "; " + ", ".join(s[r]["adapt_val"]["subject_ids"])
        rows.append(f"{int(r) + 1} & {' & '.join(cells)} & {ids} \\\\")
    head = ("\\begin{tabular}{@{}lcccl@{}}\n\\toprule\n"
            "Split & Adapt.-train cells (ALL / HEM) & Adapt.-val.\\ cells & Test cells & "
            "Adapt.-train subjects; adapt.-val.\\ subjects \\\\ \\midrule\n")
    (out / "split_counts.tex").write_text(head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def epochs(res, out):
    rows = []
    for a in ("cnn", "vit", "hybrid"):
        f = pd.read_csv(res / "source" / a / "cv_folds.csv")
        info = json.loads((res / "source" / a / "final_model_info.json").read_text())
        rows.append(f"{NAME[a]} & source CV & {', '.join(str(int(e)) for e in f.best_epoch)} & "
                    f"final source model: {info['final_epochs']} epochs \\\\")
    t = pd.read_csv(res / "target" / "adaptation_runs.csv")
    for a in ("cnn", "vit", "hybrid"):
        g = t[t.arch == a].sort_values("repeat")
        rows.append(f"{NAME[a]} & adaptation & {', '.join(str(int(e)) for e in g.best_epoch)} & "
                    f"median {int(np.median(g.best_epoch))} \\\\")
    head = ("\\begin{tabular}{@{}llll@{}}\n\\toprule\n"
            "Model & Stage & Restored epoch per fold / split & Note \\\\ \\midrule\n")
    (out / "epochs.tex").write_text(head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def target_ci(res, out):
    d = pd.read_csv(res / "tables" / "target_metrics_per_repeat.csv")
    d = d[(d.masking == "none")].sort_values(["arch", "stage", "tta", "repeat"])
    rows = []
    for _, r in d.iterrows():
        cell = lambda m: f"{f3(r[m])} [{f3(r[m + '_lo'])}, {f3(r[m + '_hi'])}]"
        rows.append(f"{NAME[r.arch]} & {r.stage.replace('_', '-')} & {'yes' if r.tta else 'no'} & {int(r["repeat"]) + 1} & "
                    f"{cell('accuracy')} & {cell('sensitivity_ALL')} & {cell('specificity')} & {cell('MCC')} & {cell('AUC')} \\\\")
    head = ("\\begin{tabular}{@{}llllccccc@{}}\n\\toprule\n"
            "Model & Stage & TTA & Split & Accuracy & Sensitivity (ALL) & Specificity & MCC & AUROC \\\\ \\midrule\n")
    (out / "target_ci.tex").write_text(head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def color(res, out):
    cs = pd.read_csv(res / "tables" / "color_summary.csv", header=[0, 1], index_col=[0, 1])
    cd = pd.read_csv(res / "tables" / "color_cohens_d.csv").pivot(index="feature", columns="label", values="cohens_d")
    lab = {"L_mean": "mean L*", "a_mean": "mean a*", "b_mean": "mean b*", "L_sd": "SD L*", "a_sd": "SD a*",
           "b_sd": "SD b*", "area_frac": "cell area fraction"}
    rows = []
    for f, name in lab.items():
        nd = 3 if f == "area_frac" else 2
        c = lambda ph, l: f"{cs.loc[(ph, l), (f, 'mean')]:.{nd}f} $\\pm$ {cs.loc[(ph, l), (f, 'std')]:.{nd}f}"
        rows.append(f"{name} & {c('source', 0)} & {c('target', 0)} & {cd.loc[f, 0]:+.2f} & "
                    f"{c('source', 1)} & {c('target', 1)} & {cd.loc[f, 1]:+.2f} \\\\")
    head = ("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n & \\multicolumn{3}{c}{HEM} & \\multicolumn{3}{c}{ALL} \\\\\n"
            "Feature & source & target & $d$ & source & target & $d$ \\\\ \\midrule\n")
    (out / "color.tex").write_text(head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="revision/results")
    ap.add_argument("--splits", default="revision/splits")
    ap.add_argument("--out", default="revision/results/tex")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paired(res, out)
    compute(res, out)
    split_counts(Path(a.splits), out)
    epochs(res, out)
    target_ci(res, out)
    color(res, out)
    print(f"LaTeX tables written to {out}")


if __name__ == "__main__":
    main()
