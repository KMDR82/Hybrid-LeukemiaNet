"""Computational cost of the three models on the same GPU and settings.

Reports per model: total / trainable parameters, GFLOPs per 224x224 image
(forward pass, counted with torch.utils.flop_counter), peak GPU memory during
inference and during one training step, and inference latency (batch 1 and
batch 32, with and without K=3 TTA). Mean and SD over timed repetitions after
warm-up. Training time per epoch is read from the training histories.

Usage:
  python revision/profile_compute.py --out revision/results/tables
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from common import get_device
from models import LeukemiaNet, count_parameters


def gflops(model, device):
    x = torch.randn(1, 3, 224, 224, device=device)
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 1e9  # FLOPs (multiply and add counted separately)


def latency(model, device, batch, k_views, reps, warmup, amp):
    model.eval()
    x = torch.randn(batch * k_views, 3, 224, 224, device=device)
    times = []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
        for i in range(warmup + reps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - t) * 1000)
    return float(np.mean(times)), float(np.std(times, ddof=1))


def peak_memory(fn, device):
    if device.type != "cuda":
        return float("nan")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="revision/results/tables")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--train-batch", type=int, default=32)
    ap.add_argument("--no-amp", action="store_true")
    a = ap.parse_args()
    device = get_device()
    amp = (not a.no_amp) and device.type == "cuda"
    rows = []
    for arch in ("cnn", "vit", "hybrid"):
        model = LeukemiaNet(arch, pretrained=False).to(device)
        total, trainable = count_parameters(model)
        row = {"arch": arch, "params_total_M": total / 1e6, "params_trainable_M": trainable / 1e6,
               "GFLOPs_per_image_torch_counter": gflops(model, device), "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
               "amp_fp16": amp}
        for bs in (1, 32):
            for k in (1, 3):
                m, s = latency(model, device, bs, k, a.reps, a.warmup, amp)
                row[f"latency_ms_b{bs}_k{k}"] = m
                row[f"latency_ms_b{bs}_k{k}_sd"] = s
                row[f"ms_per_image_b{bs}_k{k}"] = m / bs

        def infer():
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                model.eval()
                model(torch.randn(32, 3, 224, 224, device=device))

        def train_step():
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                loss = model(torch.randn(a.train_batch, 3, 224, 224, device=device)).float().sum()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        row["peak_mem_MB_infer_b32"] = peak_memory(infer, device)
        row["peak_mem_MB_train_b%d" % a.train_batch] = peak_memory(train_step, device)
        rows.append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "compute_cost.csv", index=False)
    print(df.T.to_string())


if __name__ == "__main__":
    main()
