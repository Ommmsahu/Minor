
from pathlib import Path
import gc
import time

import joblib
import numpy as np
import pandas as pd
import psutil

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------
BASE = Path(__file__).resolve().parent

TRAIN_PATH = BASE / "output" / "splits" / "train.csv"
TEST_PATH = BASE / "output" / "splits" / "test.csv"

MODEL_DIR = BASE / "models"
REPORT_DIR = BASE / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "greenfed_random_forest.joblib"

N_TREES = 100
MAX_DEPTH = 18
MAX_LEAF_NODES = 2048
MAX_SAMPLES = 0.5
N_JOBS = 2

# --------------------------------------------------
# 2. Check files and available RAM
# --------------------------------------------------
if not TRAIN_PATH.is_file() or not TEST_PATH.is_file():
    raise FileNotFoundError(
        "Training or testing CSV not found. "
        "Check output/splits/."
    )

memory = psutil.virtual_memory()
print(f"Available RAM: {memory.available / 1024**3:.2f} GB")

if memory.available < 3 * 1024**3:
    raise MemoryError(
        "Less than 3 GB RAM is available. "
        "Close other applications and rerun the script."
    )

start_time = time.time()

# --------------------------------------------------
# 3. Load the training data
# --------------------------------------------------
print("\nLoading training data...")

header = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()

if "Label" not in header:
    raise ValueError("Label column is missing.")

feature_columns = [c for c in header if c != "Label"]
dtype_map = {c: np.float32 for c in feature_columns}

train_df = pd.read_csv(
    TRAIN_PATH,
    dtype=dtype_map,
    low_memory=True,
)

y_train = train_df["Label"].astype(str).str.strip()
X_train = train_df[feature_columns].to_numpy(dtype=np.float32)

del train_df
gc.collect()

print(f"Training rows: {len(y_train):,}")
print(f"Features: {X_train.shape[1]}")
print(f"Training labels: {y_train.nunique()}")

if not np.isfinite(X_train).all():
    raise ValueError("Non-finite values found in training features.")

classes = np.sort(y_train.unique())

if len(classes) != 34:
    raise ValueError(
        f"Expected 34 classes, found {len(classes)}."
    )

# --------------------------------------------------
# 4. Train Random Forest
# --------------------------------------------------
print("\nTraining Random Forest...")
print("This may take some time. Please keep the laptop plugged in.")

model = RandomForestClassifier(
    n_estimators=N_TREES,
    max_depth=MAX_DEPTH,
    max_leaf_nodes=MAX_LEAF_NODES,
    max_samples=MAX_SAMPLES,
    class_weight="balanced_subsample",
    n_jobs=N_JOBS,
    random_state=42,
    bootstrap=True,
    verbose=1,
)

model.fit(X_train, y_train)

print("\nRandom Forest training completed.")
print(f"Elapsed time: {(time.time() - start_time) / 60:.1f} minutes")

# --------------------------------------------------
# 5. Load and evaluate the held-out test set
# --------------------------------------------------
print("\nLoading test data...")

test_df = pd.read_csv(
    TEST_PATH,
    dtype=dtype_map,
    low_memory=True,
)

y_test = test_df["Label"].astype(str).str.strip()
X_test = test_df[feature_columns].to_numpy(dtype=np.float32)

del test_df
gc.collect()

if not np.isfinite(X_test).all():
    raise ValueError("Non-finite values found in test features.")

print("Predicting test labels...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
macro_f1 = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)
weighted_f1 = f1_score(
    y_test, y_pred, average="weighted", zero_division=0
)

print("\n" + "=" * 55)
print("GREENFED-IDS: RANDOM FOREST BASELINE")
print("=" * 55)
print(f"Test rows: {len(y_test):,}")
print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced accuracy: {balanced_acc:.4f}")
print(f"Macro-F1:          {macro_f1:.4f}")
print(f"Weighted-F1:       {weighted_f1:.4f}")

# Per-class metrics
report = classification_report(
    y_test,
    y_pred,
    labels=classes,
    zero_division=0,
    output_dict=True,
)

report_path = REPORT_DIR / "random_forest_classification_report.csv"
pd.DataFrame(report).transpose().to_csv(report_path)

# Confusion matrix
matrix = confusion_matrix(y_test, y_pred, labels=classes)

matrix_path = REPORT_DIR / "random_forest_confusion_matrix.csv"
pd.DataFrame(
    matrix, index=classes, columns=classes
).to_csv(matrix_path, index_label="Actual_Label")

# Save model and feature metadata
joblib.dump(
    {
        "model": model,
        "feature_columns": feature_columns,
        "classes": classes,
        "configuration": {
            "n_estimators": N_TREES,
            "max_depth": MAX_DEPTH,
            "max_leaf_nodes": MAX_LEAF_NODES,
            "max_samples": MAX_SAMPLES,
            "class_weight": "balanced_subsample",
            "random_state": 42,
        },
    },
    MODEL_PATH,
)

print("\nSaved files:")
print(f"Model: {MODEL_PATH}")
print(f"Classification report: {report_path}")
print(f"Confusion matrix: {matrix_path}")
print(f"Total elapsed time: {(time.time() - start_time) / 60:.1f} minutes")
print("Original datasets and split files were not modified.")
