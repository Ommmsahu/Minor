"""Data validation, label mapping, cache identity and preprocessing for GreenFed-IDS FL.

Only train.csv is ever read. Any attempt to read a file named like test.csv is refused.
Validation rows are a stratified hold-out taken from train.csv itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from federated.io_utils import atomic_save_npy, atomic_write_json, sha256_file, short_hash

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CSV = REPO_ROOT / "output" / "splits" / "train.csv"
CACHE_DIR = REPO_ROOT / "output" / "fl_cache"
REPORTS_DIR = REPO_ROOT / "reports"
LOGS_DIR = REPO_ROOT / "logs"

TARGET_COL = "Label"
N_FEATURES = 39
N_CLASSES = 34
CLIP_VALUE = 10.0  # standardized features are clipped to [-10, 10] for training stability
SCHEMA_VERSION = 2

# Trusted feature schema: exact names in the exact order used for training.
TRUSTED_FEATURES: list[str] = [
    "Header_Length", "Protocol Type", "Time_To_Live", "Rate", "fin_flag_number", "syn_flag_number",
    "rst_flag_number", "psh_flag_number", "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP",
    "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC", "Tot sum", "Min",
    "Max", "AVG", "Std", "Tot size", "IAT", "Number", "Variance",
]
assert len(TRUSTED_FEATURES) == N_FEATURES and len(set(TRUSTED_FEATURES)) == N_FEATURES

CACHE_FILES = ("X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy", "train_source_idx.npy", "val_source_idx.npy")
META_NAME = "cache_meta.json"


class CacheError(ValueError):
    """The preprocessed cache is missing, stale or inconsistent."""


def preprocess_config() -> dict:
    return {"scaler": "standard", "clip": CLIP_VALUE, "dtype": "float32", "scaler_fit_on": "client-training rows only"}


def reject_test_file(path: Path) -> None:
    name = Path(path).name.lower()
    if name == "test.csv" or name.startswith("test"):
        raise PermissionError(f"Refusing to read '{path}': the held-out test file must not be used here.")


def validate_frame(
    df: pd.DataFrame,
    expected_features: list[str] | None = None,
    n_classes: int = N_CLASSES,
    allow_reorder: bool = False,
) -> list[str]:
    """Validate schema and values. Returns the trusted ordered feature list or raises ValueError."""
    expected = list(TRUSTED_FEATURES if expected_features is None else expected_features)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")
    cols = [c for c in df.columns if c != TARGET_COL]
    if len(set(cols)) != len(cols):
        raise ValueError("Duplicate column names found.")
    if cols != expected:
        missing, extra = sorted(set(expected) - set(cols)), sorted(set(cols) - set(expected))
        if missing or extra:
            raise ValueError(f"Feature names differ from the trusted schema. Missing: {missing}. Unexpected: {extra}.")
        if not allow_reorder:
            raise ValueError("Feature columns are in a different order than the trusted schema. "
                             "Fix the file or pass allow_reorder=True to reorder explicitly.")
    non_numeric = [c for c in expected if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns: {non_numeric}")
    values = df[expected].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"Found {int((~np.isfinite(values)).sum())} missing or non-finite feature values.")
    if df[TARGET_COL].isna().any():
        raise ValueError("Missing values in Label column.")
    found = df[TARGET_COL].nunique()
    if found != n_classes:
        raise ValueError(f"Expected {n_classes} classes, found {found}.")
    return expected


def build_label_map(labels: pd.Series) -> dict[str, int]:
    """One global label -> index map. Sorted so it is identical on every run."""
    return {name: i for i, name in enumerate(sorted(labels.astype(str).unique()))}


def read_train_csv(csv_path: Path, nrows: int | None = None) -> pd.DataFrame:
    reject_test_file(csv_path)
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    dtypes = {c: "float32" for c in header if c != TARGET_COL}
    return pd.read_csv(csv_path, dtype=dtypes, nrows=nrows)


def file_identity(path: Path) -> dict:
    p = Path(path).resolve()
    st = p.stat()
    return {"path": str(p), "name": p.name, "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha256_file(p)}


def _expected_meta(csv_id: dict, seed: int, val_fraction: float) -> dict:
    return {
        "schema_version": SCHEMA_VERSION, "csv_size_bytes": csv_id["size_bytes"], "csv_sha256": csv_id["sha256"],
        "seed": seed, "val_fraction": val_fraction, "preprocessing": preprocess_config(),
        "feature_names": TRUSTED_FEATURES, "n_features": N_FEATURES, "n_classes": N_CLASSES,
    }


def load_cache_meta(cache_dir: Path = CACHE_DIR) -> dict:
    p = Path(cache_dir) / META_NAME
    if not p.exists():
        raise CacheError(f"{p} not found. Run: python -m federated.partition_data")
    return json.loads(p.read_text())


def verify_cache(cache_dir: Path = CACHE_DIR) -> dict:
    """Check the cache is internally consistent. Returns its metadata or raises CacheError."""
    cache_dir = Path(cache_dir)
    meta = load_cache_meta(cache_dir)
    for name in CACHE_FILES:
        if not (cache_dir / name).exists():
            raise CacheError(f"Cache file {name} is missing. Rebuild with: python -m federated.partition_data --force")
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise CacheError(f"Cache schema version {meta.get('schema_version')} != {SCHEMA_VERSION}.")
    if meta["feature_names"] != TRUSTED_FEATURES:
        raise CacheError("Cached feature names/order differ from the trusted schema.")
    if sorted(meta["label_map"].values()) != list(range(N_CLASSES)) or len(meta["label_map"]) != N_CLASSES:
        raise CacheError(f"Cached label map does not hold exactly {N_CLASSES} classes numbered 0..{N_CLASSES - 1}.")
    Xt, yt = np.load(cache_dir / "X_train.npy"), np.load(cache_dir / "y_train.npy")
    Xv, yv = np.load(cache_dir / "X_val.npy"), np.load(cache_dir / "y_val.npy")
    ts, vs = np.load(cache_dir / "train_source_idx.npy"), np.load(cache_dir / "val_source_idx.npy")
    for name, X, y, n in (("train", Xt, yt, meta["train_rows"]), ("val", Xv, yv, meta["val_rows"])):
        if X.ndim != 2 or X.shape != (n, N_FEATURES) or X.dtype != np.float32:
            raise CacheError(f"X_{name} has shape {X.shape} dtype {X.dtype}; expected ({n}, {N_FEATURES}) float32.")
        if y.shape != (n,) or y.dtype != np.int64:
            raise CacheError(f"y_{name} has shape {y.shape} dtype {y.dtype}; expected ({n},) int64.")
        if not np.isfinite(X).all():
            raise CacheError(f"X_{name} contains non-finite values.")
        if y.min() < 0 or y.max() >= N_CLASSES:
            raise CacheError(f"y_{name} has labels outside [0, {N_CLASSES - 1}].")
        if len(np.unique(y)) != N_CLASSES:
            raise CacheError(f"y_{name} does not contain all {N_CLASSES} classes.")
    if len(ts) != meta["train_rows"] or len(vs) != meta["val_rows"]:
        raise CacheError("Source index arrays do not match the cached row counts.")
    if np.intersect1d(ts, vs).size:
        raise CacheError("Training and validation rows overlap.")
    if len(np.unique(ts)) != len(ts) or len(np.unique(vs)) != len(vs):
        raise CacheError("Duplicate source row indices in the cache.")
    if len(ts) + len(vs) != meta["source_rows"] or (len(ts) and (min(ts.min(), vs.min()) != 0 or max(ts.max(), vs.max()) != meta["source_rows"] - 1)):
        raise CacheError("Train + validation rows do not cover the source rows exactly once.")
    return meta


def cache_is_current(csv_path: Path, cache_dir: Path, seed: int, val_fraction: float) -> tuple[bool, str]:
    """True only if the stored metadata matches the current CSV and settings."""
    cache_dir = Path(cache_dir)
    if not (cache_dir / META_NAME).exists():
        return False, "no cache metadata"
    meta = json.loads((cache_dir / META_NAME).read_text())
    csv_id = file_identity(csv_path)
    want = _expected_meta(csv_id, seed, val_fraction)
    for key, value in want.items():
        if meta.get(key) != value:
            return False, f"'{key}' changed"
    try:
        verify_cache(cache_dir)
    except (CacheError, FileNotFoundError, ValueError, KeyError) as exc:
        return False, f"cache invalid: {exc}"
    return True, "metadata matches"


def prepare_cache(
    csv_path: Path = DEFAULT_TRAIN_CSV,
    cache_dir: Path = CACHE_DIR,
    seed: int = 42,
    val_fraction: float = 0.10,
    force: bool = False,
    reports_dir: Path = REPORTS_DIR,
    allow_reorder: bool = False,
) -> dict:
    """Validate train.csv, split off a validation set, standardize, save arrays plus metadata.

    The cache is reused only if its metadata matches the current CSV (size and SHA-256),
    seed, validation fraction, preprocessing settings and feature schema. Otherwise it is rebuilt.
    The scaler uses statistics pooled over all clients, which is a simulation shortcut.
    """
    csv_path, cache_dir, reports_dir = Path(csv_path), Path(cache_dir), Path(reports_dir)
    reject_test_file(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Place train.csv at output/splits/train.csv.")
    cfg_path = reports_dir / "schema_config.json"
    if not force:
        ok, reason = cache_is_current(csv_path, cache_dir, seed, val_fraction)
        if ok and cfg_path.exists():
            print(f"Cache reused ({reason}).")
            return json.loads(cfg_path.read_text())
        print(f"Rebuilding cache ({reason if not ok else 'schema_config.json missing'}).")
    else:
        print("Rebuilding cache (--force).")

    csv_id = file_identity(csv_path)
    df = read_train_csv(csv_path)
    feature_cols = validate_frame(df, allow_reorder=allow_reorder)
    reordered = [c for c in df.columns if c != TARGET_COL] != feature_cols
    label_map = build_label_map(df[TARGET_COL])
    y = df[TARGET_COL].astype(str).map(label_map).to_numpy(dtype=np.int64)
    X = df[feature_cols].to_numpy(dtype=np.float32)
    del df

    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(idx, test_size=val_fraction, stratify=y, random_state=seed)
    train_idx.sort()
    val_idx.sort()

    mean = X[train_idx].mean(axis=0, dtype=np.float64)
    std = X[train_idx].std(axis=0, dtype=np.float64)
    std[std == 0] = 1.0  # constant columns stay at zero after centering

    def transform(a: np.ndarray) -> np.ndarray:
        z = (a - mean.astype(np.float32)) / std.astype(np.float32)
        return np.clip(z, -CLIP_VALUE, CLIP_VALUE).astype(np.float32)

    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / META_NAME).unlink(missing_ok=True)  # an interrupted rebuild must not look valid
    atomic_save_npy(cache_dir / "X_train.npy", transform(X[train_idx]))
    atomic_save_npy(cache_dir / "y_train.npy", y[train_idx])
    atomic_save_npy(cache_dir / "X_val.npy", transform(X[val_idx]))
    atomic_save_npy(cache_dir / "y_val.npy", y[val_idx])
    atomic_save_npy(cache_dir / "train_source_idx.npy", train_idx.astype(np.int64))
    atomic_save_npy(cache_dir / "val_source_idx.npy", val_idx.astype(np.int64))

    cfg = {
        "feature_columns": feature_cols, "label_map": label_map, "n_features": len(feature_cols),
        "n_classes": len(label_map), "source_rows": int(len(y)), "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)), "val_fraction": val_fraction, "seed": seed, "clip_value": CLIP_VALUE,
        "scaler_mean": mean.tolist(), "scaler_std": std.tolist(),
        "note": "Scaler fit on training rows only (validation rows and test.csv excluded).",
    }
    meta = {
        **_expected_meta(csv_id, seed, val_fraction),
        "csv_path": csv_id["path"], "csv_mtime_ns": csv_id["mtime_ns"], "label_map": label_map,
        "source_rows": cfg["source_rows"], "train_rows": cfg["train_rows"], "val_rows": cfg["val_rows"],
        "columns_reordered_from_source": bool(reordered),
        "validation_source": f"stratified hold-out of {csv_id['name']} only",
        "arrays": {"X_train": [cfg["train_rows"], N_FEATURES], "X_val": [cfg["val_rows"], N_FEATURES]},
    }
    meta["cache_id"] = short_hash({k: meta[k] for k in ("csv_sha256", "seed", "val_fraction", "preprocessing", "feature_names", "label_map", "schema_version")})
    reports_dir.mkdir(exist_ok=True)
    atomic_write_json(cfg_path, cfg)
    atomic_write_json(cache_dir / META_NAME, meta)  # written last: its presence marks a complete cache
    verify_cache(cache_dir)
    return cfg


def load_cache_arrays(cache_dir: Path = CACHE_DIR, mmap: bool = True):
    mode = "r" if mmap else None
    X = np.load(Path(cache_dir) / "X_train.npy", mmap_mode=mode)
    y = np.load(Path(cache_dir) / "y_train.npy")
    return X, y


def load_val_arrays(cache_dir: Path = CACHE_DIR):
    return np.load(Path(cache_dir) / "X_val.npy"), np.load(Path(cache_dir) / "y_val.npy")


def load_schema_config(reports_dir: Path = REPORTS_DIR) -> dict:
    return json.loads((Path(reports_dir) / "schema_config.json").read_text())
