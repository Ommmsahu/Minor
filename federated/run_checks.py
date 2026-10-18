"""Integrity checks applied after Flower returns. A run is only reported as complete if all pass."""
from __future__ import annotations

import math
from collections import Counter

VAL_KEYS = ("val_loss", "val_accuracy", "val_balanced_accuracy", "val_macro_f1", "val_weighted_f1")


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_run_integrity(cfg, client_records: list[dict], server_records: list[dict],
                        expected_sizes: list[int] | None = None, n_classes: int = 34) -> list[str]:
    """Return a list of problems. An empty list means the run is complete and consistent."""
    errors: list[str] = []
    rounds, clients = range(1, cfg.rounds + 1), range(cfg.num_clients)

    for rec in client_records:
        r, c = rec.get("round"), rec.get("client_id")
        if r not in rounds:
            errors.append(f"client record with unexpected round {r!r}")
        if c not in clients:
            errors.append(f"client record with unexpected client_id {c!r}")
        if rec.get("run_id") not in (None, cfg.run_id):
            errors.append(f"client record from another run: {rec.get('run_id')!r}")
        if not _finite(rec.get("train_loss")) or not _finite(rec.get("train_seconds")):
            errors.append(f"non-finite loss/time for round {r} client {c}")
        n = rec.get("num_examples")
        if not isinstance(n, int) or n <= 0:
            errors.append(f"invalid num_examples for round {r} client {c}: {n!r}")
        elif expected_sizes is not None and c in clients and n != expected_sizes[c]:
            errors.append(f"round {r} client {c} trained on {n} rows but its partition has {expected_sizes[c]}")

    counts = Counter((r.get("round"), r.get("client_id")) for r in client_records)
    for (r, c), n in sorted(counts.items(), key=str):
        if n > 1:
            errors.append(f"duplicate client update: round {r} client {c} appears {n} times")
    for r in rounds:
        for c in clients:
            if counts.get((r, c), 0) == 0:
                errors.append(f"missing client update: round {r} client {c}")

    scounts = Counter(s.get("round") for s in server_records)
    for r in range(0, cfg.rounds + 1):
        if scounts.get(r, 0) == 0:
            errors.append(f"missing global validation record for round {r}")
        elif scounts[r] > 1:
            errors.append(f"duplicate global validation record for round {r}")
    for s in server_records:
        if s.get("round") not in range(0, cfg.rounds + 1):
            errors.append(f"validation record with unexpected round {s.get('round')!r}")
        if s.get("run_id") not in (None, cfg.run_id):
            errors.append(f"validation record from another run: {s.get('run_id')!r}")
        bad = [k for k in VAL_KEYS if not _finite(s.get(k))]
        if bad:
            errors.append(f"non-finite or missing validation metrics in round {s.get('round')}: {bad}")
        if len(s.get("per_class", {})) != n_classes:
            errors.append(f"round {s.get('round')} per-class metrics cover {len(s.get('per_class', {}))} classes, expected {n_classes}")
    return errors


def select_final_record(server_records: list[dict]) -> dict:
    """The final validation record is the one with the highest round number, whatever the list order."""
    if not server_records:
        raise ValueError("No server validation records.")
    return max(server_records, key=lambda r: r["round"])
