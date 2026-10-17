"""Shared PyTorch MLP plus local training and evaluation helpers."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
from torch import nn

N_FEATURES = 39
N_CLASSES = 34


class MLP(nn.Module):
    """39 -> 128 -> 64 -> 34 feed-forward network (ReLU). Outputs raw logits."""

    def __init__(self, n_features: int = N_FEATURES, n_classes: int = N_CLASSES, hidden: tuple[int, int] = (128, 64)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_local(model: nn.Module, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, lr: float, seed: int) -> float:
    """Train in place on local data. Returns mean training loss over the last epoch."""
    gen = torch.Generator().manual_seed(seed)
    Xt, yt = torch.from_numpy(np.ascontiguousarray(X)), torch.from_numpy(np.ascontiguousarray(y))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    last = float("nan")
    for _ in range(epochs):
        perm = torch.randperm(len(yt), generator=gen)
        total, seen = 0.0, 0
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[b]), yt[b])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss. Check scaling, learning rate and labels.")
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
            seen += len(b)
        last = total / seen
    return last


@torch.no_grad()
def evaluate_model(model: nn.Module, X: np.ndarray, y: np.ndarray, n_classes: int = N_CLASSES, batch_size: int = 8192) -> dict:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    total, preds = 0.0, []
    for i in range(0, len(yt), batch_size):
        out = model(Xt[i:i + batch_size])
        total += loss_fn(out, yt[i:i + batch_size]).item()
        preds.append(out.argmax(1).numpy())
    pred = np.concatenate(preds)
    labels = list(range(n_classes))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    return {
        "loss": total / len(y),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", labels=labels, zero_division=0)),
        "per_class": {"precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist(), "support": s.tolist()},
    }
