"""IID and non-IID (Dirichlet label-skew) partitions of the client-training rows.

Run:  python -m federated.partition_data
Creates:
  output/fl_cache/partition_<type>_seed<seed>.npz        row indices per client (not committed)
  output/fl_cache/partition_<type>_seed<seed>_meta.json  seed, alpha, client count, sizes, cache id
  reports/client_partition_summary.csv                   rows and distribution summary per client
  reports/client_class_counts_<type>.csv                 class x client count matrix
Partitions index only the client-training arrays. Validation rows are never included.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from federated import data_utils as du
from federated.io_utils import atomic_write_json

N_CLIENTS = 5


class PartitionError(ValueError):
    """A partition is invalid, stale or does not match the cache or run configuration."""


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
    y: np.ndarray, n_clients: int = N_CLIENTS, alpha: float = 0.5, seed: int = 42,
    min_size: int = 1000, max_tries: int = 200,
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


def check_partition(parts: list[np.ndarray], n_rows: int, n_clients: int | None = None, min_size: int = 1) -> None:
    """Every client-training row must be assigned to exactly one valid client."""
    if n_clients is not None and len(parts) != n_clients:
        raise PartitionError(f"Expected {n_clients} clients, found {len(parts)}.")
    for k, p in enumerate(parts):
        if p.ndim != 1 or not np.issubdtype(p.dtype, np.integer):
            raise PartitionError(f"Client {k} indices must be a 1-D integer array.")
        if len(p) < min_size:
            raise PartitionError(f"Client {k} has {len(p)} rows, below the minimum of {min_size}.")
        if len(p) and (p.min() < 0 or p.max() >= n_rows):
            raise PartitionError(f"Client {k} has row indices outside [0, {n_rows - 1}].")
    allidx = np.concatenate(parts)
    if len(allidx) != n_rows:
        raise PartitionError(f"Assigned {len(allidx)} rows but there are {n_rows} client-training rows.")
    if len(np.unique(allidx)) != n_rows:
        raise PartitionError("Some rows are assigned to more than one client (or some are missing).")


def check_partition_vs_cache(parts: list[np.ndarray], cache_dir: Path) -> None:
    """Map client rows back to train.csv row numbers and confirm none are validation rows."""
    train_src = np.load(Path(cache_dir) / "train_source_idx.npy")
    val_src = np.load(Path(cache_dir) / "val_source_idx.npy")
    src = train_src[np.concatenate(parts)]
    if np.intersect1d(src, val_src).size:
        raise PartitionError("A client partition contains validation rows.")
    if len(np.unique(src)) != len(train_src):
        raise PartitionError("Client partitions do not cover the client-training rows exactly once.")


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


def partition_meta_path(cache_dir: Path, ptype: str, seed: int) -> Path:
    return Path(cache_dir) / f"partition_{ptype}_seed{seed}_meta.json"


def save_partition(cache_dir: Path, ptype: str, seed: int, parts: list[np.ndarray], alpha: float | None, min_size: int) -> dict:
    meta = {
        "partition_type": ptype, "seed": seed, "alpha": alpha, "n_clients": len(parts),
        "n_rows": int(sum(len(p) for p in parts)), "min_size": min_size,
        "client_sizes": [int(len(p)) for p in parts], "cache_id": du.load_cache_meta(cache_dir)["cache_id"],
        "method": "stratified IID deal" if ptype == "iid" else "label-wise Dirichlet",
    }
    np.savez(partition_path(cache_dir, ptype, seed), **{f"client_{k}": p for k, p in enumerate(parts)})
    atomic_write_json(partition_meta_path(cache_dir, ptype, seed), meta)
    return meta


def load_partition(cache_dir: Path, ptype: str, seed: int, client_id: int) -> np.ndarray:
    with np.load(partition_path(cache_dir, ptype, seed)) as z:
        return z[f"client_{client_id}"]


def verify_partition_files(cache_dir: Path, ptype: str, seed: int, n_clients: int, alpha: float | None, cache_meta: dict) -> dict:
    """Confirm the saved partition exists, matches the run settings and the active cache. Returns its metadata."""
    pp, mp = partition_path(cache_dir, ptype, seed), partition_meta_path(cache_dir, ptype, seed)
    hint = "Re-run: python -m federated.partition_data"
    if not pp.exists() or not mp.exists():
        raise PartitionError(f"Partition files for {ptype}/seed {seed} are missing or have no metadata. {hint}")
    meta = json.loads(mp.read_text())
    if meta["cache_id"] != cache_meta["cache_id"]:
        raise PartitionError(f"Partition was built from a different cache ({meta['cache_id']} vs {cache_meta['cache_id']}). {hint}")
    if meta["n_clients"] != n_clients:
        raise PartitionError(f"Partition has {meta['n_clients']} clients but the run asks for {n_clients}. {hint} --clients {n_clients}")
    if meta["seed"] != seed or meta["partition_type"] != ptype:
        raise PartitionError(f"Partition metadata (type {meta['partition_type']}, seed {meta['seed']}) does not match the run. {hint}")
    if ptype == "noniid" and meta["alpha"] != alpha:
        raise PartitionError(f"Partition was built with alpha={meta['alpha']} but the run asks for alpha={alpha}. {hint} --alpha {alpha}")
    with np.load(pp) as z:
        expected_keys = {f"client_{k}" for k in range(n_clients)}
        if set(z.files) != expected_keys:
            raise PartitionError(f"Partition file holds clients {sorted(z.files)}, expected {sorted(expected_keys)}.")
        parts = [z[f"client_{k}"] for k in range(n_clients)]
    check_partition(parts, cache_meta["train_rows"], n_clients, min_size=1)
    if [len(p) for p in parts] != meta["client_sizes"]:
        raise PartitionError("Client sizes in the partition file differ from its metadata.")
    check_partition_vs_cache(parts, cache_dir)
    return meta


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
    if a.clients < 1 or not np.isfinite(a.alpha) or a.alpha <= 0 or a.min_size < 1:
        raise SystemExit("Invalid arguments: clients and min-size must be >= 1 and alpha must be > 0.")

    cfg = du.prepare_cache(Path(a.train_csv), du.CACHE_DIR, a.seed, a.val_fraction, a.force)
    cache_meta = du.verify_cache(du.CACHE_DIR)
    print(f"Schema OK: {cfg['n_features']} features, {cfg['n_classes']} classes, "
          f"{cfg['train_rows']} train rows, {cfg['val_rows']} validation rows (from train.csv only). cache_id={cache_meta['cache_id']}")
    _, y = du.load_cache_arrays(du.CACHE_DIR)
    label_names = [n for n, _ in sorted(cfg["label_map"].items(), key=lambda kv: kv[1])]

    summaries = []
    for ptype in ("iid", "noniid"):
        if ptype == "iid":
            parts, alpha = iid_partition(y, a.clients, a.seed), None
        else:
            parts, alpha = dirichlet_partition(y, a.clients, a.alpha, a.seed, a.min_size), a.alpha
        check_partition(parts, len(y), a.clients, min_size=a.min_size if ptype == "noniid" else 1)
        check_partition_vs_cache(parts, du.CACHE_DIR)
        save_partition(du.CACHE_DIR, ptype, a.seed, parts, alpha, a.min_size)
        verify_partition_files(du.CACHE_DIR, ptype, a.seed, a.clients, alpha, cache_meta)
        counts = class_count_matrix(y, parts, len(label_names))
        cm = pd.DataFrame(counts, index=label_names, columns=[f"client_{k}" for k in range(a.clients)])
        cm.index.name = "class"
        du.REPORTS_DIR.mkdir(exist_ok=True)
        cm.to_csv(du.REPORTS_DIR / f"client_class_counts_{ptype}.csv")
        s = summarize(y, parts, ptype, a.seed, alpha, label_names)
        summaries.append(s)
        print(f"\n{ptype.upper()} partition (seed={a.seed}" + (f", alpha={alpha}" if alpha else "") + ") integrity OK")
        print(s[["client_id", "rows", "classes_present", "top_class", "top_class_share", "tv_distance_to_global"]].to_string(index=False))
    pd.concat(summaries, ignore_index=True).to_csv(du.REPORTS_DIR / "client_partition_summary.csv", index=False)
    print("\nSaved reports/client_partition_summary.csv and per-type class count matrices.")


if __name__ == "__main__":
    main()
