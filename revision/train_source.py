"""Stage 1: source-phase training (C-NMC training phase, 73 subjects).

1. Subject-level 5-fold cross-validation (folds from make_splits.py).
   In each fold, ~10% of the TRAINING subjects (stratified by class) form an
   inner validation set used only for early stopping; the held-out fold is
   used only for reporting. Out-of-fold (OOF) probabilities are saved.
2. The final source model is trained on ALL 73 subjects for a fixed number of
   epochs = median of the early-stopped epochs across folds (decided before
   any target data are seen). No "best fold" is selected.
3. A secondary decision threshold (Youden J) is computed from the OOF
   probabilities only. The primary threshold is 0.5.

Usage:
  python revision/train_source.py --data-root DATA --arch hybrid
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import get_device, load_index, save_json, seed_worker, set_seed
from data import CellDataset, eval_transform, train_transform
from engine import class_weights, fit, predict
from metrics import binary_metrics, youden_threshold
from models import LeukemiaNet, count_parameters


def inner_val_subjects(df, frac, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _, g in df.groupby("label"):
        subs = rng.permutation(g["subject"].unique())
        out.extend(subs[: max(1, int(round(frac * len(subs))))])
    return set(out)


def loader(df, root, tf, args, shuffle):
    g = torch.Generator()
    g.manual_seed(args.seed)
    return DataLoader(CellDataset(df, root, tf), batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.workers, pin_memory=True, worker_init_fn=seed_worker,
                      generator=g, drop_last=shuffle and len(df) > args.batch_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits", default="revision/splits")
    ap.add_argument("--arch", default="hybrid", choices=["hybrid", "cnn", "vit"])
    ap.add_argument("--out", default="revision/results/source")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min-delta", type=float, default=1e-3)
    ap.add_argument("--inner-val-frac", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true", help="smoke tests only")
    ap.add_argument("--skip-cv", action="store_true", help="reuse saved CV results, train final model only")
    args = ap.parse_args()

    device = get_device()
    out = Path(args.out) / args.arch
    out.mkdir(parents=True, exist_ok=True)
    src = load_index(Path(args.splits) / "source_index.csv")
    folds = sorted(src["fold"].unique())

    if not args.skip_cv:
        fold_rows, oof = [], []
        for k in folds:
            done_f, oof_f = out / f"fold{k}_result.json", out / f"fold{k}_oof.csv"
            if done_f.exists() and oof_f.exists():  # resume after a crash
                import json
                fold_rows.append(json.loads(done_f.read_text()))
                oof.append(pd.read_csv(oof_f, dtype={"subject": str}))
                print(f"[{args.arch}] fold {k}: already finished, skipped")
                continue
            set_seed(args.seed + int(k))
            train_all, test = src[src.fold != k], src[src.fold == k]
            iv = inner_val_subjects(train_all, args.inner_val_frac, args.seed + int(k))
            train, val = train_all[~train_all.subject.isin(iv)], train_all[train_all.subject.isin(iv)]
            print(f"[{args.arch}] fold {k}: train {train.subject.nunique()} subj / {len(train)} img, "
                  f"inner-val {val.subject.nunique()} / {len(val)}, test {test.subject.nunique()} / {len(test)}")
            model = LeukemiaNet(args.arch, pretrained=not args.no_pretrained).to(device)
            hist, best_epoch, secs = fit(
                model, loader(train, args.data_root, train_transform(), args, True), device,
                lr=args.lr, weight_decay=args.weight_decay, max_epochs=args.max_epochs,
                val_loader=loader(val, args.data_root, eval_transform(), args, False),
                patience=args.patience, min_delta=args.min_delta,
                weights=class_weights(train.label, device))
            pd.DataFrame(hist).to_csv(out / f"fold{k}_history.csv", index=False)
            p, y, i = predict(model, loader(test, args.data_root, eval_transform(), args, False), device)
            part = test.iloc[i][["name", "subject", "label"]].assign(p=p, fold=k)
            oof.append(part)
            m = binary_metrics(y, p)
            fold_rows.append({"fold": int(k), "best_epoch": best_epoch, "epochs_run": len(hist),
                              "train_seconds": secs, "test_subjects": test.subject.nunique(),
                              "test_images": len(test), **m})
            part.to_csv(oof_f, index=False)
            save_json(fold_rows[-1], done_f)
            del model
            torch.cuda.empty_cache()
        folds_df = pd.DataFrame(fold_rows)
        folds_df.to_csv(out / "cv_folds.csv", index=False)
        oof = pd.concat(oof)
        oof.to_csv(out / "cv_oof_predictions.csv", index=False)
        summary = {c: {"mean": float(folds_df[c].mean()), "sd": float(folds_df[c].std(ddof=1))}
                   for c in folds_df.columns if c not in ("fold",) and folds_df[c].dtype != object}
        final_epochs = int(np.median(folds_df["best_epoch"]))
        thr = youden_threshold(oof["label"], oof["p"])
        save_json({"arch": args.arch, "cv_summary": summary, "final_epochs": final_epochs,
                   "youden_threshold_from_oof": thr, "args": vars(args)}, out / "cv_summary.json")
    else:
        import json
        with open(out / "cv_summary.json") as f:
            final_epochs = json.load(f)["final_epochs"]

    if (out / "final_source_model.pt").exists():
        print(f"[{args.arch}] final source model already exists, skipped")
        return
    print(f"[{args.arch}] final source model on all {src.subject.nunique()} subjects, {final_epochs} epochs")
    set_seed(args.seed)
    model = LeukemiaNet(args.arch, pretrained=not args.no_pretrained).to(device)
    hist, _, secs = fit(model, loader(src, args.data_root, train_transform(), args, True), device,
                        lr=args.lr, weight_decay=args.weight_decay, max_epochs=final_epochs,
                        weights=class_weights(src.label, device))
    pd.DataFrame(hist).to_csv(out / "final_history.csv", index=False)
    total, trainable = count_parameters(model)
    save_json({"final_epochs": final_epochs, "train_seconds": secs, "params_total": total,
               "params_trainable": trainable}, out / "final_model_info.json")
    torch.save(model.state_dict(), out / "final_source_model.pt")


if __name__ == "__main__":
    main()
