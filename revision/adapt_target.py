"""Stage 2: limited target-phase adaptation and held-out testing
(C-NMC preliminary-test phase, 28 subjects).

For every repeat r of the subject-level target split (make_splits.py):
  adapt_train subjects -> fine-tuning data
  adapt_val subjects   -> early stopping only (validation loss)
  test subjects        -> evaluated once per condition; never used for any
                          decision (epoch, checkpoint, threshold)

Conditions written to predictions.csv (one row per test image):
  stage    zero_shot | adapted
  tta      0 | 1      (same weights with and without TTA -> paired comparison)
  masking  none | cnn_stream_only | vit_stream_only   (hybrid, zero-shot and adapted)

Usage:
  python revision/adapt_target.py --data-root DATA --arch hybrid cnn vit

Variant B (pre-specified): --mode head trains only the classifier head (both
backbones frozen, in eval mode) at lr 5e-4; --balanced draws adaptation-train
cells with class-balanced sampling (WeightedRandomSampler) instead of using a
class-weighted loss. The source variant (augmentation, colour standardisation)
is read from <source>/variant.json.
"""
import json
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from common import get_device, load_index, save_json, seed_worker, set_seed
from data import CellDataset, eval_transform, train_transform, tta_transforms
from engine import class_weights, fit, predict
from models import LeukemiaNet


def make_loader(df, root, tf, bs, workers, seed, shuffle, balanced=False):
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = None
    if balanced:
        counts = df["label"].value_counts()
        w = df["label"].map(lambda c: 1.0 / counts[c]).to_numpy()
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(df),
                                        replacement=True, generator=g)
    return DataLoader(CellDataset(df, root, tf), batch_size=bs, shuffle=shuffle and sampler is None,
                      sampler=sampler, num_workers=workers, pin_memory=True, worker_init_fn=seed_worker,
                      generator=g, drop_last=shuffle and len(df) > bs)


def evaluate_all(model, arch, test, args, stage, repeat):
    rows = []
    maskings = ["none"] + (["cnn_stream_only", "vit_stream_only"] if arch == "hybrid" else [])
    for masking in maskings:
        model.mask_cnn = masking == "vit_stream_only"
        model.mask_vit = masking == "cnn_stream_only"
        for tta in (0, 1):
            if masking != "none" and tta:
                continue
            tf = tta_transforms(args.std) if tta else eval_transform(args.std)
            bs = max(1, args.batch_size // 3) if tta else args.batch_size
            p, _, i = predict(model, make_loader(test, args.data_root, tf, bs, args.workers, args.seed, False),
                              args.device, tta=bool(tta))
            part = test.iloc[i][["name", "subject", "label"]].copy()
            part["p"] = p
            part["repeat"], part["arch"], part["stage"], part["tta"], part["masking"] = repeat, arch, stage, tta, masking
            rows.append(part)
    model.mask_cnn = model.mask_vit = False
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits", default="revision/splits")
    ap.add_argument("--source", default="revision/results/source")
    ap.add_argument("--out", default="revision/results/target")
    ap.add_argument("--arch", nargs="+", default=["hybrid", "cnn", "vit"])
    ap.add_argument("--repeats", type=int, nargs="*", default=None, help="default: all")
    ap.add_argument("--lr", type=float, default=None, help="default 5e-5 (full) or 5e-4 (head)")
    ap.add_argument("--mode", default="full", choices=["full", "head"])
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min-delta", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true", help="smoke tests only")
    args = ap.parse_args()
    args.device = get_device()
    if args.lr is None:
        args.lr = 5e-4 if args.mode == "head" else 5e-5
    vf = Path(args.source) / "variant.json"
    variant = json.loads(vf.read_text()) if vf.exists() else {"aug": "basic", "standardize": False}
    args.aug = variant["aug"]
    args.std = None
    if variant.get("standardize"):
        ref = json.loads((Path(args.source) / "color_reference.json").read_text())
        args.std = (ref["mean"], ref["sd"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tgt = load_index(Path(args.splits) / "target_index.csv")
    splits = load_index(Path(args.splits) / "target_splits.csv")
    repeats = args.repeats if args.repeats else sorted(splits["repeat"].unique())

    preds, runs = [], []
    done = set()
    if (out / "adaptation_runs.csv").exists() and (out / "predictions.csv").exists():  # resume after a crash
        old_runs = pd.read_csv(out / "adaptation_runs.csv")
        old_pred = pd.read_csv(out / "predictions.csv", dtype={"subject": str})
        done = set(zip(old_runs.repeat.astype(int), old_runs.arch))
        runs = old_runs.to_dict("records")
        keep = [(int(r), a) in done for r, a in zip(old_pred.repeat, old_pred.arch)]
        preds = [old_pred[keep]]
        print(f"Resuming: {len(done)} finished runs found, they will be skipped")
    for r in repeats:
        sp = splits[splits.repeat == r]
        role = dict(zip(sp.subject, sp.role))
        tgt_r = tgt.assign(role=tgt.subject.map(role))
        a_train, a_val, test = (tgt_r[tgt_r.role == x].reset_index(drop=True)
                                for x in ("adapt_train", "adapt_val", "test"))
        for arch in args.arch:
            if (int(r), arch) in done:
                continue
            set_seed(args.seed + int(r))
            print(f"repeat {r} | {arch}: adapt-train {a_train.subject.nunique()} subj/{len(a_train)} img, "
                  f"adapt-val {a_val.subject.nunique()}/{len(a_val)}, test {test.subject.nunique()}/{len(test)}")
            # backbone weights come from the source checkpoint, so no ImageNet download is needed
            model = LeukemiaNet(arch, pretrained=False).to(args.device)
            state = torch.load(Path(args.source) / arch / "final_source_model.pt", map_location=args.device)
            model.load_state_dict(state)
            preds += evaluate_all(model, arch, test, args, "zero_shot", r)

            if args.mode == "head":
                model.set_head_only(True)
            hist, best_epoch, secs = fit(
                model, make_loader(a_train, args.data_root, train_transform(args.aug, args.std), args.batch_size,
                                   args.workers, args.seed + int(r), True, balanced=args.balanced), args.device,
                lr=args.lr, weight_decay=args.weight_decay, max_epochs=args.max_epochs,
                val_loader=make_loader(a_val, args.data_root, eval_transform(args.std), args.batch_size,
                                       args.workers, args.seed, False),
                patience=args.patience, min_delta=args.min_delta,
                weights=None if args.balanced else class_weights(a_train.label, args.device))
            model.set_head_only(False)
            pd.DataFrame(hist).to_csv(out / f"history_r{r}_{arch}.csv", index=False)
            runs.append({"repeat": int(r), "arch": arch, "best_epoch": best_epoch, "epochs_run": len(hist),
                         "train_seconds": secs,
                         "adapt_train_subjects": a_train.subject.nunique(), "adapt_train_images": len(a_train),
                         "adapt_val_subjects": a_val.subject.nunique(), "adapt_val_images": len(a_val),
                         "test_subjects": test.subject.nunique(), "test_images": len(test)})
            preds += evaluate_all(model, arch, test, args, "adapted", r)
            if int(r) == int(repeats[0]):  # kept for the explainability figure (explain.py)
                torch.save(model.state_dict(), out / f"adapted_r{r}_{arch}.pt")
            del model
            torch.cuda.empty_cache()
            # write after every run so an interrupted job keeps its results
            pd.concat(preds).to_csv(out / "predictions.csv", index=False)
            pd.DataFrame(runs).to_csv(out / "adaptation_runs.csv", index=False)
    save_json({k: (str(v) if k == "device" else v) for k, v in vars(args).items()}, out / "adapt_args.json")


if __name__ == "__main__":
    main()
