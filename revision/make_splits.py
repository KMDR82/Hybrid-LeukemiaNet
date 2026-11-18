"""Build subject-level split files for the C-NMC 2019 data (Kaggle mirror).

Outputs (in --out):
  source_index.csv   every training-phase image with subject ID, label and CV fold
  target_index.csv   every preliminary-test-phase image with subject ID and label
  target_splits.csv  for each repeat, the role of every target subject:
                     adapt_train / adapt_val / test
  split_summary.json subject and image counts per split (reported in the paper)

The split files are committed to the repository so that every number in the
paper can be reproduced with exactly the same subjects.

Usage:
  python revision/make_splits.py --data-root /path/to/C-NMC_Leukemia --out revision/splits
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common import parse_cnmc_name, save_json

EXPECTED = {"source_subjects": 73, "source_images": 10661,
            "target_subjects": 28, "target_images": 1867}


def build_source_index(root):
    root = Path(root)
    files = sorted(p for p in Path(root).rglob("*.bmp") if "training_data" in p.parts)
    if not files:
        raise SystemExit(f"No training_data/*.bmp found under {root}")
    rows = []
    for p in files:
        subject, label = parse_cnmc_name(p.name)
        rows.append({"path": str(p.relative_to(root)), "name": p.name, "subject": subject, "label": label,
                     "organiser_fold": next((x for x in p.parts if x.startswith("fold_")), "")})
    return pd.DataFrame(rows)


def build_target_index(root):
    root = Path(root)
    csvs = [p for p in Path(root).rglob("*.csv") if "prelim" in p.name.lower()]
    if len(csvs) != 1:
        raise SystemExit(f"Expected one preliminary-phase label CSV, found: {csvs}")
    lab = pd.read_csv(csvs[0])
    img_dir = next(p for p in Path(root).rglob("C-NMC_test_prelim_phase_data") if p.is_dir())
    rows = []
    for _, r in lab.iterrows():
        subject, label_from_name = parse_cnmc_name(r["Patient_ID"])
        if int(r["labels"]) != label_from_name:
            raise SystemExit(f"Label mismatch for {r['Patient_ID']}: csv={r['labels']}")
        rows.append({"path": str((img_dir / r["new_names"]).relative_to(root)), "name": r["new_names"],
                     "original_name": r["Patient_ID"], "subject": subject, "label": label_from_name})
    return pd.DataFrame(rows)


def assign_source_folds(df, n_folds, seed):
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df = df.copy()
    df["fold"] = -1
    for k, (_, idx) in enumerate(sgkf.split(df, df["label"], groups=df["subject"])):
        df.loc[df.index[idx], "fold"] = k
    return df


def target_repeats(df, n_repeats, n_train, n_val, seed):
    """n_train / n_val = subjects PER CLASS in the adaptation-train / adaptation-val parts."""
    subj = df.groupby("subject")["label"].first().reset_index()
    rows = []
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        for label in (0, 1):
            ids = subj.loc[subj.label == label, "subject"].to_numpy()
            ids = rng.permutation(ids)
            for i, s in enumerate(ids):
                role = "adapt_train" if i < n_train else ("adapt_val" if i < n_train + n_val else "test")
                rows.append({"repeat": r, "subject": s, "label": label, "role": role})
    return pd.DataFrame(rows)


def summarise(src, tgt, splits):
    def counts(d):
        return {"subjects_ALL": int(d.loc[d.label == 1, "subject"].nunique()),
                "subjects_HEM": int(d.loc[d.label == 0, "subject"].nunique()),
                "images_ALL": int((d.label == 1).sum()), "images_HEM": int((d.label == 0).sum())}
    out = {"source_total": counts(src),
           "source_folds": {int(k): counts(g) for k, g in src.groupby("fold")},
           "target_total": counts(tgt), "target_repeats": {}}
    for r, g in splits.groupby("repeat"):
        out["target_repeats"][int(r)] = {}
        for role, gg in g.groupby("role"):
            d = tgt[tgt.subject.isin(gg.subject)]
            c = counts(d)
            c["subject_ids"] = sorted(gg.subject.tolist())
            out["target_repeats"][int(r)][role] = c
    return out


def check_disjoint(src, tgt, splits):
    shared = set(src.subject) & set(tgt.subject)
    if shared:
        print(f"WARNING: subjects present in both phases: {sorted(shared)}")
    for k in sorted(src.fold.unique()):
        tr = set(src.loc[src.fold != k, "subject"])
        va = set(src.loc[src.fold == k, "subject"])
        assert not tr & va, f"subject leakage in source fold {k}"
    for r, g in splits.groupby("repeat"):
        roles = [set(g.loc[g.role == x, "subject"]) for x in ("adapt_train", "adapt_val", "test")]
        assert not (roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2]), \
            f"subject leakage in target repeat {r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="revision/splits")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--adapt-train-per-class", type=int, default=2)
    ap.add_argument("--adapt-val-per-class", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-unexpected-counts", action="store_true")
    a = ap.parse_args()

    src = assign_source_folds(build_source_index(a.data_root), a.n_folds, a.seed)
    tgt = build_target_index(a.data_root)
    splits = target_repeats(tgt, a.n_repeats, a.adapt_train_per_class, a.adapt_val_per_class, a.seed)
    check_disjoint(src, tgt, splits)

    found = {"source_subjects": src.subject.nunique(), "source_images": len(src),
             "target_subjects": tgt.subject.nunique(), "target_images": len(tgt)}
    print("Counts:", found)
    if found != EXPECTED and not a.allow_unexpected_counts:
        raise SystemExit(f"Counts differ from the published C-NMC figures {EXPECTED}. "
                         "Check the data root, or pass --allow-unexpected-counts.")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # Paths are stored relative to --data-root so the files are portable.
    src.to_csv(out / "source_index.csv", index=False)
    tgt.to_csv(out / "target_index.csv", index=False)
    splits.to_csv(out / "target_splits.csv", index=False)
    save_json(summarise(src, tgt, splits), out / "split_summary.json")
    print(f"Wrote split files to {out}")


if __name__ == "__main__":
    main()
