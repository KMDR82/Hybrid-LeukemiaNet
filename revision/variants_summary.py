"""Summarise the pre-specified improvement variants (run_variants.py).

Variants (target-phase predictions, one directory each):
  base_full      original pipeline (results/target)
  base_head      original source models, head-only class-balanced adaptation (B)
  coloraug_full  colour/scale augmentation + colour standardisation (A), full adaptation
  coloraug_head  A + B
Late fusion (E) is added to every variant as arch "late_fusion": the mean of the
EfficientNet-B0 and ViT-B/16 probabilities of the same image and condition.

All variants and comparisons are fixed in advance and all are reported; no
variant is chosen by its test-subject performance.

  python revision/variants_summary.py --results revision/results
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import binary_metrics, paired_comparison

VARIANTS = {"base_full": "target", "base_head": "variants/target_base_head",
            "coloraug_full": "variants/target_coloraug_full", "coloraug_head": "variants/target_coloraug_head"}
SOURCES = {"base": "source", "coloraug": "variants/source_coloraug"}
METRICS = ["accuracy", "sensitivity_ALL", "specificity", "macro_F1", "MCC", "AUC", "ECE", "Brier"]


def load(res):
    out = []
    for v, d in VARIANTS.items():
        f = res / d / "predictions.csv"
        if not f.exists():
            print(f"missing {f}, variant {v} skipped")
            continue
        p = pd.read_csv(f, dtype={"subject": str})
        p = p[p.masking == "none"].drop(columns="masking")
        key = ["repeat", "stage", "tta", "name", "subject", "label"]
        w = p.pivot_table(index=key, columns="arch", values="p").reset_index()
        if {"cnn", "vit"} <= set(w.columns):
            lf = w[key].assign(arch="late_fusion", p=(w["cnn"] + w["vit"]) / 2)
            p = pd.concat([p, lf], ignore_index=True)
        out.append(p.assign(variant=v))
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="revision/results")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    res = Path(a.results)
    out = res / "variants" / "summary"
    out.mkdir(parents=True, exist_ok=True)

    # source cross-validation of each source variant
    rows = []
    for sv, d in SOURCES.items():
        for arch in ("hybrid", "cnn", "vit"):
            f = res / d / arch / "cv_summary.json"
            if f.exists():
                s = json.loads(f.read_text())["cv_summary"]
                rows.append({"source_variant": sv, "arch": arch,
                             **{f"{m}_mean": s[m]["mean"] for m in METRICS if m in s},
                             **{f"{m}_sd": s[m]["sd"] for m in METRICS if m in s}})
    pd.DataFrame(rows).to_csv(out / "source_cv_variants.csv", index=False)

    pred = load(res)
    per = []
    for (v, arch, st, tta, r), g in pred.groupby(["variant", "arch", "stage", "tta", "repeat"]):
        per.append({"variant": v, "arch": arch, "stage": st, "tta": tta, "repeat": r,
                    **{m: binary_metrics(g["label"], g["p"])[m] for m in METRICS}})
    per = pd.DataFrame(per)
    per.to_csv(out / "variants_metrics_per_repeat.csv", index=False)
    summ = per.groupby(["variant", "arch", "stage", "tta"])[METRICS].agg(["mean", "std"])
    summ.to_csv(out / "variants_summary.csv")

    # pre-specified paired comparisons (no TTA), A - B on the same test cells
    plan = []
    for arch in ("hybrid", "cnn", "vit", "late_fusion"):
        for st in ("zero_shot", "adapted"):
            plan.append(("A: colour aug. + standardisation", ("coloraug_full", arch, st), ("base_full", arch, st)))
        plan.append(("B: head-only balanced adaptation", ("base_head", arch, "adapted"), ("base_full", arch, "adapted")))
        plan.append(("B on A", ("coloraug_head", arch, "adapted"), ("coloraug_full", arch, "adapted")))
    for v in VARIANTS:
        for st in ("zero_shot", "adapted"):
            plan.append(("E: late fusion vs dual-stream", (v, "late_fusion", st), (v, "hybrid", st)))
            plan.append(("E: late fusion vs EfficientNet-B0", (v, "late_fusion", st), (v, "cnn", st)))
    base = pred[pred.tta == 0]
    comps = []
    for label, A, B in plan:
        ga = base[(base.variant == A[0]) & (base.arch == A[1]) & (base.stage == A[2])]
        gb = base[(base.variant == B[0]) & (base.arch == B[1]) & (base.stage == B[2])]
        if ga.empty or gb.empty:
            continue
        m = ga.merge(gb, on=["repeat", "name", "subject", "label"], suffixes=("_a", "_b"))
        for r, g in m.groupby("repeat"):
            g = g.reset_index(drop=True)
            c = paired_comparison(g, "p_a", "p_b", n_boot=a.n_boot, seed=int(r))
            comps.append({"comparison": label, "A": "/".join(map(str, A)), "B": "/".join(map(str, B)),
                          "repeat": r, **{k: v for k, v in c.items() if k not in ("A", "B")}})
    comps = pd.DataFrame(comps)
    comps.to_csv(out / "variants_paired.csv", index=False)
    if not comps.empty:
        g = comps.groupby(["comparison", "A", "B"], sort=False)
        ps = g.agg(mean_dMCC=("delta_MCC", "mean"), min_dMCC=("delta_MCC", "min"), max_dMCC=("delta_MCC", "max"),
                   ci_above0=("delta_MCC_lo", lambda s: int((s > 0).sum())),
                   ci_below0=("delta_MCC_hi", lambda s: int((s < 0).sum())),
                   mean_dSens=("delta_sensitivity_ALL", "mean"), mean_dECE=("delta_ECE", "mean"),
                   n_splits=("repeat", "count")).reset_index()
        ps.to_csv(out / "variants_paired_summary.csv", index=False)
        print(ps.round(3).to_string())
    print(summ.xs(0, level="tta")[[("MCC", "mean"), ("MCC", "std"), ("AUC", "mean")]].round(3).to_string())
    print(f"Variant summary written to {out}")


if __name__ == "__main__":
    main()
