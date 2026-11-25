"""Training and inference loops shared by the source and target scripts."""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import EarlyStopping


def class_weights(labels, device):
    """Inverse-frequency class weights for the cross-entropy loss (class imbalance)."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2).astype(float)
    w = counts.sum() / (2.0 * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32, device=device)


def _amp(device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    tot_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with _amp(device):
                logits = model(x)
                loss = criterion(logits.float(), y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            tot_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
    return tot_loss / n, correct / n


def fit(model, train_loader, device, *, lr, weight_decay, max_epochs, val_loader=None,
        patience=5, min_delta=1e-3, weights=None, log=print):
    """Train with AdamW at a constant learning rate.

    With ``val_loader``: early stopping on validation loss (patience, min_delta),
    and the weights of the best epoch are restored.
    Without it: exactly ``max_epochs`` epochs (used for the final source model,
    whose epoch count is fixed in advance from cross-validation).
    Returns (history, best_epoch, train_seconds).
    """
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    stopper = EarlyStopping(patience, min_delta) if val_loader is not None else None
    history, t0 = [], time.time()
    for epoch in range(1, max_epochs + 1):
        tl, ta = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        row = {"epoch": epoch, "train_loss": tl, "train_acc": ta}
        if val_loader is not None:
            vl, va = run_epoch(model, val_loader, criterion, device)
            row.update(val_loss=vl, val_acc=va)
        history.append(row)
        log(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if stopper is not None and stopper.step(row["val_loss"], epoch, model):
            break
    best_epoch = max_epochs
    if stopper is not None:
        model.load_state_dict(stopper.best_state)
        best_epoch = stopper.best_epoch
    return history, best_epoch, time.time() - t0


@torch.no_grad()
def predict(model, loader, device, tta=False):
    """Returns (p_all, labels, row_indices). With tta=True the loader yields
    (B, K, 3, H, W) and the K softmax outputs are averaged."""
    model.eval()
    probs, labels, idx = [], [], []
    for x, y, i in loader:
        x = x.to(device, non_blocking=True)
        with _amp(device):
            if tta:
                b, k = x.shape[:2]
                p = F.softmax(model(x.flatten(0, 1)).float(), 1).view(b, k, -1).mean(1)
            else:
                p = F.softmax(model(x).float(), 1)
        probs.append(p[:, 1].cpu().numpy())
        labels.append(np.asarray(y))
        idx.append(np.asarray(i))
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(idx)
