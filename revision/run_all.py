"""Run the whole revised pipeline end to end, overnight, with resume.

  python revision/run_all.py --data-root "D:/data/C-NMC_Leukemia"

Every stage writes its outputs to revision/results/ and is skipped on a
re-run if it already finished, so after a crash or power cut simply run the
same command again. A full log is appended to revision/results/run_all.log.

Stages
  1 make_splits      subject-level split files (seconds)
  2 color_stats      colour shift between phases, CPU (minutes)
  3 profile_compute  params, GFLOPs, memory, latency on this GPU (minutes)
  4 train_source     5-fold subject-level CV + final model, for cnn, vit, hybrid (longest)
  5 adapt_target     5 repeats x 3 models, zero-shot and adapted, +-TTA
  6 analyze          tables with subject-level bootstrap CIs and paired tests
  7 figures          figures from the logged results
  8 explain          Grad-CAM / attention-rollout figure
  9 tex_tables       LaTeX tables for the manuscript and supplement
"""
import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def log(msg):
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / "run_all.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(name, args, done_marker):
    if done_marker is not None and Path(done_marker).exists():
        log(f"SKIP {name} (already done: {done_marker})")
        return
    cmd = [sys.executable, "-u", str(HERE / args[0])] + [str(a) for a in args[1:]]
    log(f"START {name}: {' '.join(cmd)}")
    t0 = dt.datetime.now()
    with open(RES / f"{name}.log", "a", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             cwd=HERE.parent, bufsize=1, encoding="utf-8", errors="replace")
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        p.wait()
    if p.returncode != 0:
        log(f"FAILED {name} (exit {p.returncode}); see revision/results/{name}.log. Re-run the same command to resume.")
        sys.exit(p.returncode)
    log(f"DONE {name} in {dt.datetime.now() - t0}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="folder that contains training_data and validation_data")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers (use 0 if Windows gives errors)")
    ap.add_argument("--source-batch", type=int, default=32, help="lower to 16 if CUDA out of memory")
    ap.add_argument("--skip", nargs="*", default=[], help="stage names to skip")
    a = ap.parse_args()
    splits = HERE / "splits"
    w = ["--workers", a.workers]

    stages = [
        ("make_splits", ["make_splits.py", "--data-root", a.data_root, "--out", splits], splits / "split_summary.json"),
        ("color_stats", ["color_stats.py", "--data-root", a.data_root, "--splits", splits,
                         "--out", RES / "tables"], RES / "tables" / "color_phase_classifier.csv"),
        ("profile_compute", ["profile_compute.py", "--out", RES / "tables"], RES / "tables" / "compute_cost.csv"),
    ]
    for arch in ("cnn", "vit", "hybrid"):
        stages.append((f"train_source_{arch}", ["train_source.py", "--data-root", a.data_root, "--splits", splits,
                                                "--arch", arch, "--out", RES / "source",
                                                "--batch-size", a.source_batch] + w,
                       RES / "source" / arch / "final_source_model.pt"))
    stages += [
        ("adapt_target", ["adapt_target.py", "--data-root", a.data_root, "--splits", splits,
                          "--source", RES / "source", "--out", RES / "target"] + w, None),
        ("analyze", ["analyze.py", "--source", RES / "source", "--target", RES / "target",
                     "--out", RES / "tables"], None),
        ("figures", ["figures.py", "--source", RES / "source", "--target", RES / "target",
                     "--tables", RES / "tables", "--out", RES / "figures"], None),
        ("explain", ["explain.py", "--data-root", a.data_root, "--splits", splits, "--target", RES / "target",
                     "--out", RES / "figures"], None),
        ("tex_tables", ["make_tex_tables.py", "--results", RES, "--splits", splits, "--out", RES / "tex"], None),
    ]
    log("=" * 60)
    log(f"Pipeline start, data root = {a.data_root}")
    try:
        import torch
        log(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}"
            + (f", GPU: {torch.cuda.get_device_name()}" if torch.cuda.is_available() else ""))
        if not torch.cuda.is_available():
            log("WARNING: no CUDA GPU found; training will be extremely slow")
    except ImportError:
        log("torch is not installed; see revision/README.md")
        sys.exit(1)
    for name, args, marker in stages:
        if name in a.skip:
            log(f"SKIP {name} (--skip)")
            continue
        run(name, args, marker)
    log("ALL STAGES FINISHED. Results are in revision/results.")


if __name__ == "__main__":
    main()
