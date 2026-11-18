"""Stage 3: metrics, confidence intervals, paired tests, calibration and
subject-level read-out, written as CSV and LaTeX tables for the manuscript.

Inputs : revision/results/source/<arch>/cv_folds.csv, cv_summary.json
         revision/results/target/predictions.csv, adaptation_runs.csv
Outputs: revision/results/tables/*.csv and *.tex

Primary threshold: 0.5. Secondary threshold: Youden J from the SOURCE
out-of-fold predictions (never from target data).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import binary_metrics, paired_comparison, reliability_table, subject_bootstrap_ci, subject_level

MAIN = ["accuracy", "sensitivity_ALL", "specificity", "PPV_ALL", "NPV", "F1_ALL", "F1_HEM",
        "macro_F1", "MCC", "AUC", "ECE", "Brier"]
ARCH_NAME = {"hybrid": "EfficientNet-B0 + ViT-B/16", "cnn": "EfficientNet-B0", "vit": "ViT-B/16"}


def fmt(m, sd=None, lo=None, hi=None):
    s = f"{m:.3f}"
    if sd is not None and not np.isnan(sd):
        s += f" $\\pm$ {sd:.3f}"
    if lo is not None:
        s += f" [{lo:.3f}, {hi:.3f}]"
    return s


def source_tables(src_dir, out):
    rows = []
    for arch in ("hybrid", "cnn", "vit"):
        f = Path(src_dir) / arch / "cv_folds.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d.insert(0, "arch", arch)
        rows.append(d)
    if not rows:
        return
    d = pd.concat(rows)
    d.to_csv(out / "source_cv_folds.csv", index=False)
    agg = d.groupby("arch")[MAIN + ["best_epoch"]].agg(["mean", "std"])
    agg.to_csv(out / "source_cv_summary.csv")
    lines = ["\\begin{tabular}{l" + "c" * 6 + "}", "\\toprule",
             "Model & Accuracy & Sensitivity (ALL) & Specificity & Macro-F1 & MCC & AUC \\\\", "\\midrule"]
    for arch in agg.index:
        a = agg.loc[arch]
        lines.append(ARCH_NAME[arch] + " & " + " & ".join(
            fmt(a[(m, "mean")], a[(m, "std")]) for m in
            ["accuracy", "sensitivity_ALL", "specificity", "macro_F1", "MCC", "AUC"]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "source_cv_table.tex").write_text("\n".join(lines))


def target_tables(pred, runs, thr_secondary, out, n_boot):
    keys = ["arch", "stage", "tta", "masking"]
    per_rep = []
    for (arch, stage, tta, masking, r), g in pred.groupby(keys + ["repeat"]):
        g = g.reset_index(drop=True)
        ci = subject_bootstrap_ci(g, "p", 0.5, n_boot=n_boot, seed=int(r))
        sec = binary_metrics(g["label"], g["p"], thr_secondary.get(arch, 0.5))
        per_rep.append({"arch": arch, "stage": stage, "tta": tta, "masking": masking, "repeat": r,
                        **ci, **{f"youden_{k}": v for k, v in sec.items() if k in MAIN}})
    per_rep = pd.DataFrame(per_rep)
    per_rep.to_csv(out / "target_metrics_per_repeat.csv", index=False)

    summ = per_rep.groupby(keys)[MAIN].agg(["mean", "std"])
    summ.to_csv(out / "target_metrics_summary.csv")

    # Main table: mean +- SD across repeats; CI = mean of per-repeat subject-bootstrap CI limits
    lines = ["\\begin{tabular}{llc" + "c" * 7 + "}", "\\toprule",
             "Model & Stage & TTA & Accuracy & Sens. (ALL) & Spec. & PPV & NPV & Macro-F1 & MCC \\\\",
             "\\midrule"]
    for (arch, stage, tta, masking), g in per_rep[per_rep.masking == "none"].groupby(keys):
        cells = []
        for m in ["accuracy", "sensitivity_ALL", "specificity", "PPV_ALL", "NPV", "macro_F1", "MCC"]:
            cells.append(fmt(g[m].mean(), g[m].std(ddof=1)))
        lines.append(f"{ARCH_NAME[arch]} & {stage.replace('_', '-')} & {'yes' if tta else 'no'} & "
                     + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "target_main_table.tex").write_text("\n".join(lines))

    # Confusion matrices (summed over repeats, and per repeat)
    cm = per_rep[["arch", "stage", "tta", "masking", "repeat", "TP", "FN", "FP", "TN"]]
    cm.to_csv(out / "confusion_matrices.csv", index=False)

    # Stream masking (secondary analysis)
    mask = per_rep[(per_rep.arch == "hybrid") & (per_rep.tta == 0)].groupby(["stage", "masking"])[MAIN].agg(["mean", "std"])
    mask.to_csv(out / "stream_masking.csv")

    # Calibration (reliability) tables
    rel = []
    for (arch, stage, tta), g in pred[pred.masking == "none"].groupby(["arch", "stage", "tta"]):
        t = reliability_table(g["label"], g["p"])
        t["arch"], t["stage"], t["tta"] = arch, stage, tta
        rel.append(t)
    pd.concat(rel).to_csv(out / "reliability.csv", index=False)

    # Adaptation epochs (early stopping) per run
    if runs is not None:
        runs.to_csv(out / "adaptation_runs.csv", index=False)
    return per_rep


def paired_tests(pred, out, n_boot):
    base = pred[pred.masking == "none"]
    wide = base.pivot_table(index=["repeat", "name", "subject", "label"], columns=["arch", "stage", "tta"],
                            values="p").reset_index()
    wide.columns = ["_".join(str(c) for c in col if c != "") if isinstance(col, tuple) else col
                    for col in wide.columns]
    comps = []
    plan = []
    for stage in ("zero_shot", "adapted"):
        plan += [(f"hybrid_{stage}_0", f"cnn_{stage}_0"), (f"hybrid_{stage}_0", f"vit_{stage}_0"),
                 (f"hybrid_{stage}_1", f"hybrid_{stage}_0")]
    plan += [("hybrid_adapted_0", "hybrid_zero_shot_0")]
    for r, g in wide.groupby("repeat"):
        g = g.reset_index(drop=True)
        for a, b in plan:
            if a in g and b in g:
                comps.append({"repeat": r, **paired_comparison(g, a, b, n_boot=n_boot, seed=int(r))})
    comps = pd.DataFrame(comps)
    comps.to_csv(out / "paired_comparisons.csv", index=False)
    return comps


def subject_readout(pred, out):
    rows = []
    for (arch, stage, tta, r), g in pred[pred.masking == "none"].groupby(["arch", "stage", "tta", "repeat"]):
        s = subject_level(g)
        s["arch"], s["stage"], s["tta"], s["repeat"] = arch, stage, tta, r
        rows.append(s)
    s = pd.concat(rows)
    s.to_csv(out / "subject_level.csv", index=False)
    acc = s.assign(correct=(s.pred == s.label)).groupby(["arch", "stage", "tta", "repeat"])["correct"].mean()
    acc.groupby(["arch", "stage", "tta"]).agg(["mean", "std"]).to_csv(out / "subject_level_accuracy.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="revision/results/source")
    ap.add_argument("--target", default="revision/results/target")
    ap.add_argument("--out", default="revision/results/tables")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    source_tables(a.source, out)
    thr = {}
    for arch in ("hybrid", "cnn", "vit"):
        f = Path(a.source) / arch / "cv_summary.json"
        if f.exists():
            thr[arch] = json.loads(f.read_text())["youden_threshold_from_oof"]
    pred = pd.read_csv(Path(a.target) / "predictions.csv", dtype={"subject": str})
    runs_f = Path(a.target) / "adaptation_runs.csv"
    runs = pd.read_csv(runs_f) if runs_f.exists() else None
    target_tables(pred, runs, thr, out, a.n_boot)
    paired_tests(pred, out, a.n_boot)
    subject_readout(pred, out)
    print(f"Tables written to {out}")


if __name__ == "__main__":
    main()
