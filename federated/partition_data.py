"""IID and non-IID (Dirichlet label-skew) partitions of the training rows across clients.

Run:  python -m federated.partition_data
Creates:
  output/fl_cache/partition_<type>_seed<seed>.npz   (row indices per client, not committed)
  reports/client_partition_summary.csv              (rows and distribution summary per client)
  reports/client_class_counts_<type>.csv            (class x client count matrix)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from federated import data_utils as du

N_CLIENTS = 5


def iid_partition(y: np.ndarray, n_clients: int = N_CLIENTS, seed: int = 42) -> list[np.ndarray]:
    """Stratified IID split: each class is shuffled and dealt out in near-equal chunks."""
    rng = np.random.default_rng(seed)
    parts: list[list[np.ndarray]] = [[] for _ in range(n_clients)]
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        for k, chunk in enumerate(np.array_split(idx, n_clients)):
            parts[k].append(chunk)
    return [np.sort(np.concatenate(p)) for p in parts]


def dirichlet_partition(
    y: np.ndarray,
    n_clients: int = N_CLIENTS,
    alpha: float = 0.5,
    seed: int = 42,
    min_size: int = 1000,
    max_tries: int = 200,
) -> list[np.ndarray]:
    """Label-wise Dirichlet split. For every class, client shares ~ Dirichlet(alpha).

    Small alpha = stronger skew. The whole draw is repeated until every client holds at
    least `min_size` rows, so no client ends up empty or too small to train.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    class_idx = {c: np.flatnonzero(y == c) for c in classes}
    for _ in range(max_tries):
        parts: list[list[np.ndarray]] = [[] for _ in range(n_clients)]
        for c in classes:
            idx = class_idx[c].copy()
            rng.shuffle(idx)
            p = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].append(chunk)
        result = [np.sort(np.concatenate(p)) for p in parts]
        if min(len(r) for r in result) >= min_size:
            return result
    raise RuntimeError(f"No Dirichlet draw gave every client >= {min_size} rows. Raise alpha or lower min_size.")


def check_partition(parts: list[np.ndarray], n_rows: int) -> None:
    """Every row must be assigned to exactly one client."""
    allidx = np.concatenate(parts)
    if len(allidx) != n_rows:
        raise AssertionError(f"Assigned {len(allidx)} rows but source has {n_rows}.")
    if len(np.unique(allidx)) != n_rows:
        raise AssertionError("Some rows are assigned to more than one client.")


def class_count_matrix(y: np.ndarray, parts: list[np.ndarray], n_classes: int) -> np.ndarray:
    return np.stack([np.bincount(y[p], minlength=n_classes) for p in parts], axis=1)  # classes x clients


def summarize(y: np.ndarray, parts: list[np.ndarray], ptype: str, seed: int, alpha: float | None, label_names: list[str]) -> pd.DataFrame:
    n_classes = len(label_names)
    counts = class_count_matrix(y, parts, n_classes)
    global_p = counts.sum(axis=1) / counts.sum()
    rows = []
    for k in range(counts.shape[1]):
        col = counts[:, k]
        p = col / col.sum()
        top = int(np.argmax(col))
        rows.append({
            "partition_type": ptype, "seed": seed, "dirichlet_alpha": alpha if alpha is not None else "",
            "client_id": k, "rows": int(col.sum()), "classes_present": int((col > 0).sum()),
            "top_class": label_names[top], "top_class_share": round(float(p[top]), 4),
            "tv_distance_to_global": round(float(0.5 * np.abs(p - global_p).sum()), 4),
        })
    return pd.DataFrame(rows)


def partition_path(cache_dir: Path, ptype: str, seed: int) -> Path:
    return Path(cache_dir) / f"partition_{ptype}_seed{seed}.npz"


def load_partition(cache_dir: Path, ptype: str, seed: int, client_id: int) -> np.ndarray:
    with np.load(partition_path(cache_dir, ptype, seed)) as z:
        return z[f"client_{client_id}"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default=str(du.DEFAULT_TRAIN_CSV))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--clients", type=int, default=N_CLIENTS)
    ap.add_argument("--min-size", type=int, default=1000)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--force", action="store_true", help="rebuild the preprocessed cache")
    a = ap.parse_args()

    cfg = du.prepare_cache(Path(a.train_csv), du.CACHE_DIR, a.seed, a.val_fraction, a.force)
    print(f"Schema OK: {cfg['n_features']} features, {cfg['n_classes']} classes, "
          f"{cfg['train_rows']} train rows, {cfg['val_rows']} validation rows (from train.csv only).")
    _, y = du.load_cache_arrays(du.CACHE_DIR)
    label_names = [n for n, _ in sorted(cfg["label_map"].items(), key=lambda kv: kv[1])]

    summaries = []
    for ptype in ("iid", "noniid"):
        if ptype == "iid":
            parts, alpha = iid_partition(y, a.clients, a.seed), None
        else:
            parts, alpha = dirichlet_partition(y, a.clients, a.alpha, a.seed, a.min_size), a.alpha
        check_partition(parts, len(y))
        np.savez(partition_path(du.CACHE_DIR, ptype, a.seed), **{f"client_{k}": p for k, p in enumerate(parts)})
        counts = class_count_matrix(y, parts, len(label_names))
        cm = pd.DataFrame(counts, index=label_names, columns=[f"client_{k}" for k in range(a.clients)])
        cm.index.name = "class"
        cm.to_csv(du.REPORTS_DIR / f"client_class_counts_{ptype}.csv")
        s = summarize(y, parts, ptype, a.seed, alpha, label_names)
        summaries.append(s)
        print(f"\n{ptype.upper()} partition (seed={a.seed}" + (f", alpha={alpha}" if alpha else "") + ")")
        print(s[["client_id", "rows", "classes_present", "top_class", "top_class_share", "tv_distance_to_global"]].to_string(index=False))
    out = pd.concat(summaries, ignore_index=True)
    out.to_csv(du.REPORTS_DIR / "client_partition_summary.csv", index=False)
    print(f"\nSaved reports/client_partition_summary.csv and per-type class count matrices.")


if __name__ == "__main__":
    main()
