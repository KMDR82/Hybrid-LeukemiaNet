"""Shared utilities for the revised, subject-level evaluation protocol.

Everything that decides a split, an epoch or a threshold lives here or in the
scripts that import it, so that no decision can depend on the held-out test set.
"""
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Positive class throughout the study: ALL (malignant) = 1, HEM (normal) = 0.
POSITIVE_LABEL = 1
CLASS_NAMES = {0: "HEM", 1: "ALL"}

# Original C-NMC file names: UID_<P>_<N>_<C>_all.bmp (ALL subject P) and
# UID_H<S>_<N>_<C>_hem.bmp (healthy subject S). The subject ID is the token
# after "UID_" ("57" or "H49").
_SUBJECT_RE = re.compile(r"UID_(H?\d+)_\d+_\d+_(all|hem)", re.IGNORECASE)


def parse_cnmc_name(name):
    """Return (subject_id, label) parsed from an original C-NMC file name."""
    m = _SUBJECT_RE.search(os.path.basename(str(name)))
    if m is None:
        raise ValueError(f"Not an original C-NMC file name: {name!r}")
    subject, cls = m.group(1), m.group(2).lower()
    return subject, 1 if cls == "all" else 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load_index(path):
    df = pd.read_csv(path, dtype={"subject": str})
    return df


class EarlyStopping:
    """Stops when the monitored validation loss has not improved by more than
    ``min_delta`` for ``patience`` consecutive epochs. Keeps the best state."""

    def __init__(self, patience=5, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.best_epoch = 0
        self.bad_epochs = 0
        self.best_state = None

    def step(self, value, epoch, model):
        if value < self.best - self.min_delta:
            self.best = value
            self.best_epoch = epoch
            self.bad_epochs = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience
