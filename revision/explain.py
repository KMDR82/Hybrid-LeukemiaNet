"""Explainability figure for the adapted hybrid model (repeat 0).

For deterministically chosen test cells (the most confident TP, TN, FP and FN
examples) it draws: the input cell, Grad-CAM of the EfficientNet-B0 stream
(last convolutional block, gradient of the ALL logit of the full hybrid model),
and attention rollout of the ViT-B/16 stream (Abnar & Zuidema, 2020).

These maps show where each stream is sensitive; they are a qualitative aid
and do not by themselves prove that the model uses the morphological
criteria of haematologists.

Usage:
  python revision/explain.py --data-root DATA
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from common import get_device, load_index
from data import eval_transform
from models import LeukemiaNet


def grad_cam(model, x, target=1):
    feats = {}

    def hook(_, __, out):
        out.retain_grad()
        feats["a"] = out

    h = model.cnn.features.register_forward_hook(hook)
    model.zero_grad()
    logits = model(x)
    logits[:, target].sum().backward()
    h.remove()
    a = feats["a"]
    w = a.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()


@torch.no_grad()
def attention_rollout(model, x):
    attns, hooks = [], []
    for layer in model.vit.encoder.layers:
        def pre(mod, args, kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = True
            return args, kwargs

        def post(mod, args, out):
            attns.append(out[1])
        hooks.append(layer.self_attention.register_forward_pre_hook(pre, with_kwargs=True))
        hooks.append(layer.self_attention.register_forward_hook(post))
    model(x)
    for h in hooks:
        h.remove()
    n = attns[0].shape[-1]
    roll = torch.eye(n, device=x.device)
    for a in attns:
        a = 0.5 * a[0] + 0.5 * torch.eye(n, device=x.device)
        a = a / a.sum(-1, keepdim=True)
        roll = a @ roll
    m = roll[0, 1:]
    side = int(np.sqrt(m.numel()))
    m = m.reshape(1, 1, side, side)
    m = F.interpolate(m, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    m = m - m.min()
    return (m / (m.max() + 1e-8)).cpu().numpy()


def pick_examples(pred, per_group):
    p = pred[(pred.arch == "hybrid") & (pred.stage == "adapted") & (pred.tta == 0) & (pred.masking == "none")]
    p = p[p.repeat == p.repeat.min()].copy()
    p["yhat"] = (p.p >= 0.5).astype(int)
    p["conf"] = np.where(p.yhat == 1, p.p, 1 - p.p)
    groups = {"TP": (1, 1), "TN": (0, 0), "FP": (0, 1), "FN": (1, 0)}
    out = []
    for g, (y, yh) in groups.items():
        sel = p[(p.label == y) & (p.yhat == yh)].sort_values(["conf", "name"], ascending=[False, True]).head(per_group)
        out.append(sel.assign(group=g))
    return pd.concat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits", default="revision/splits")
    ap.add_argument("--target", default="revision/results/target")
    ap.add_argument("--out", default="revision/results/figures")
    ap.add_argument("--per-group", type=int, default=2)
    a = ap.parse_args()
    device = get_device()
    pred = pd.read_csv(Path(a.target) / "predictions.csv", dtype={"subject": str})
    ex = pick_examples(pred, a.per_group)
    r = int(pred.repeat.min())
    model = LeukemiaNet("hybrid", pretrained=False).to(device)
    model.load_state_dict(torch.load(Path(a.target) / f"adapted_r{r}_hybrid.pt", map_location=device))
    model.eval()
    tgt = load_index(Path(a.splits) / "target_index.csv").set_index("name")
    tf = eval_transform()

    fig, axes = plt.subplots(len(ex), 3, figsize=(7.5, 2.5 * len(ex)), squeeze=False)
    for row, (_, e) in enumerate(ex.iterrows()):
        img = Image.open(Path(a.data_root) / tgt.loc[e["name"], "path"]).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        cam = grad_cam(model, x.clone().requires_grad_(False))
        roll = attention_rollout(model, x)
        shown = np.asarray(img.resize((224, 224)))
        axes[row, 0].imshow(shown)
        axes[row, 0].set_title(f"{e['group']}  P(ALL)={e['p']:.2f}", fontsize=9)
        axes[row, 1].imshow(shown)
        axes[row, 1].imshow(cam, cmap="jet", alpha=0.45)
        axes[row, 2].imshow(shown)
        axes[row, 2].imshow(roll, cmap="jet", alpha=0.45)
        for c in range(3):
            axes[row, c].axis("off")
    axes[0, 1].set_title("Grad-CAM (EfficientNet-B0 stream)\n" + axes[0, 1].get_title(), fontsize=9)
    axes[0, 2].set_title("Attention rollout (ViT-B/16 stream)", fontsize=9)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out / "explainability.png", dpi=300, bbox_inches="tight")
    ex.to_csv(out / "explainability_examples.csv", index=False)
    print(f"Saved {out / 'explainability.png'}")


if __name__ == "__main__":
    main()
