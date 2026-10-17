"""Data validation, label mapping and one-time preprocessing for GreenFed-IDS FL.

Only train.csv is read here. test.csv is never opened by any file in this folder.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CSV = REPO_ROOT / "output" / "splits" / "train.csv"
CACHE_DIR = REPO_ROOT / "output" / "fl_cache"
REPORTS_DIR = REPO_ROOT / "reports"
LOGS_DIR = REPO_ROOT / "logs"

TARGET_COL = "Label"
N_FEATURES = 39
N_CLASSES = 34
CLIP_VALUE = 10.0  # standardized features are clipped to [-10, 10] for training stability


def validate_frame(df: pd.DataFrame, n_features: int = N_FEATURES, n_classes: int = N_CLASSES) -> list[str]:
    """Check schema and values. Returns the ordered feature column list or raises ValueError."""
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {list(df.columns)[:5]}...")
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    if len(feature_cols) != n_features:
        raise ValueError(f"Expected {n_features} feature columns, found {len(feature_cols)}.")
    if len(set(feature_cols)) != len(feature_cols):
        raise ValueError("Duplicate feature column names found.")
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns: {non_numeric}")
    values = df[feature_cols].to_numpy()
    if not np.isfinite(values).all():
        bad = int((~np.isfinite(values)).sum())
        raise ValueError(f"Found {bad} missing or non-finite feature values.")
    if df[TARGET_COL].isna().any():
        raise ValueError("Missing values in Label column.")
    found = df[TARGET_COL].nunique()
    if found != n_classes:
        raise ValueError(f"Expected {n_classes} classes, found {found}.")
    return feature_cols


def build_label_map(labels: pd.Series) -> dict[str, int]:
    """One global label -> index map. Sorted so it is identical on every run."""
    return {name: i for i, name in enumerate(sorted(labels.astype(str).unique()))}


def read_train_csv(csv_path: Path, nrows: int | None = None) -> pd.DataFrame:
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    dtypes = {c: "float32" for c in header if c != TARGET_COL}
    return pd.read_csv(csv_path, dtype=dtypes, nrows=nrows)


def prepare_cache(
    csv_path: Path = DEFAULT_TRAIN_CSV,
    cache_dir: Path = CACHE_DIR,
    seed: int = 42,
    val_fraction: float = 0.10,
    force: bool = False,
) -> dict:
    """Validate train.csv, split off a validation set, standardize, save .npy arrays.

    The scaler is fit on the training portion only (after the validation rows are removed).
    It uses statistics pooled over all clients, which is a simulation shortcut: a real
    deployment could not compute them centrally.
    """
    cache_dir = Path(cache_dir)
    cfg_path = REPORTS_DIR / "schema_config.json"
    if (cache_dir / "X_train.npy").exists() and cfg_path.exists() and not force:
        return json.loads(cfg_path.read_text())

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Place train.csv at output/splits/train.csv.")
    df = read_train_csv(csv_path)
    feature_cols = validate_frame(df)
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
    np.save(cache_dir / "X_train.npy", transform(X[train_idx]))
    np.save(cache_dir / "y_train.npy", y[train_idx])
    np.save(cache_dir / "X_val.npy", transform(X[val_idx]))
    np.save(cache_dir / "y_val.npy", y[val_idx])

    cfg = {
        "feature_columns": feature_cols,
        "label_map": label_map,
        "n_features": len(feature_cols),
        "n_classes": len(label_map),
        "source_rows": int(len(y)),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "val_fraction": val_fraction,
        "seed": seed,
        "clip_value": CLIP_VALUE,
        "scaler_mean": mean.tolist(),
        "scaler_std": std.tolist(),
        "note": "Scaler fit on training rows only (validation rows and test.csv excluded).",
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return cfg


def load_cache_arrays(cache_dir: Path = CACHE_DIR, mmap: bool = True):
    mode = "r" if mmap else None
    X = np.load(Path(cache_dir) / "X_train.npy", mmap_mode=mode)
    y = np.load(Path(cache_dir) / "y_train.npy")
    return X, y


def load_val_arrays(cache_dir: Path = CACHE_DIR):
    return np.load(Path(cache_dir) / "X_val.npy"), np.load(Path(cache_dir) / "y_val.npy")


def load_schema_config() -> dict:
    return json.loads((REPORTS_DIR / "schema_config.json").read_text())
