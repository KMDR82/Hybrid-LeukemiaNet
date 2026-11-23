"""Pre-specified improvement variants, run after run_all.py (overnight, with resume).

  python revision/run_variants.py --data-root "D:/data/C-NMC_Leukemia"

Needs the finished results of run_all.py (revision/results/source/*/final_source_model.pt
and revision/results/target/predictions.csv). Same splits, seeds and early-stopping
rules as the main pipeline; test subjects are never used for any decision.

Stages
  1 source_coloraug_{cnn,vit,hybrid}  (A) source CV + final model with strong colour/scale
                                       augmentation and cell-wise colour standardisation
  2 target_base_head                   (B) original source models, head-only class-balanced adaptation
  3 target_coloraug_full               (A) full adaptation of the A source models
  4 target_coloraug_head               (A + B)
  5 variants_summary                   all variants + late fusion (E), paired comparisons

Outputs in revision/results/variants/. Log: revision/results/variants/run_variants.log
"""
import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
VAR = RES / "variants"


def log(msg):
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    VAR.mkdir(parents=True, exist_ok=True)
    with open(VAR / "run_variants.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(name, args, done_marker):
    if done_marker is not None and Path(done_marker).exists():
        log(f"SKIP {name} (already done: {done_marker})")
        return
    cmd = [sys.executable, "-u", str(HERE / args[0])] + [str(a) for a in args[1:]]
    log(f"START {name}: {' '.join(cmd)}")
    t0 = dt.datetime.now()
    with open(VAR / f"{name}.log", "a", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             cwd=HERE.parent, bufsize=1, encoding="utf-8", errors="replace")
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        p.wait()
    if p.returncode != 0:
        log(f"FAILED {name} (exit {p.returncode}); see revision/results/variants/{name}.log. "
            "Re-run the same command to resume.")
        sys.exit(p.returncode)
    log(f"DONE {name} in {dt.datetime.now() - t0}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--source-batch", type=int, default=32)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--no-pretrained", action="store_true", help="smoke tests only")
    a = ap.parse_args()
    splits = HERE / "splits"
    w = ["--workers", a.workers]
    npt = ["--no-pretrained"] if a.no_pretrained else []
    for need in [RES / "source" / x / "final_source_model.pt" for x in ("cnn", "vit", "hybrid")] + \
                [RES / "target" / "predictions.csv"]:
        if not need.exists():
            log(f"Missing {need}: run revision/run_all.py first.")
            sys.exit(1)

    src_a = VAR / "source_coloraug"
    stages = []
    for arch in ("cnn", "vit", "hybrid"):
        stages.append((f"source_coloraug_{arch}",
                       ["train_source.py", "--data-root", a.data_root, "--splits", splits, "--arch", arch,
                        "--out", src_a, "--batch-size", a.source_batch, "--aug", "strong", "--standardize"] + w + npt,
                       src_a / arch / "final_source_model.pt"))
    for name, source, mode in (("target_base_head", RES / "source", "head"),
                               ("target_coloraug_full", src_a, "full"),
                               ("target_coloraug_head", src_a, "head")):
        extra = ["--mode", "head", "--balanced"] if mode == "head" else []
        stages.append((name, ["adapt_target.py", "--data-root", a.data_root, "--splits", splits,
                              "--source", source, "--out", VAR / name] + extra + w, None))
    stages.append(("variants_summary", ["variants_summary.py", "--results", RES], None))

    log("=" * 60)
    log(f"Variants start, data root = {a.data_root}")
    for name, args, marker in stages:
        if name in a.skip:
            log(f"SKIP {name} (--skip)")
            continue
        run(name, args, marker)
    log("ALL VARIANTS FINISHED. Results are in revision/results/variants.")


if __name__ == "__main__":
    main()
