
from pathlib import Path
import csv
import hashlib
import json
from collections import Counter

BASE = Path(__file__).resolve().parent
INPUTS = [
    BASE / "output" / "Merged01_clean.csv",
    BASE / "output" / "Merged02_clean.csv",
]

SPLIT_DIR = BASE / "output" / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = SPLIT_DIR / "train.csv"
TEST_PATH = SPLIT_DIR / "test.csv"

LABEL_COLUMN = "Label"
TEST_BUCKETS = 5  # Approximately 20% test, 80% train

def feature_hash(row, label_index):
    features = [
        value.strip()
        for index, value in enumerate(row)
        if index != label_index
    ]
    text = json.dumps(
        features, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

train_rows = 0
test_rows = 0
train_labels = Counter()
test_labels = Counter()
train_groups = set()
test_groups = set()
expected_header = None

with TRAIN_PATH.open("w", newline="", encoding="utf-8") as train_file, \
     TEST_PATH.open("w", newline="", encoding="utf-8") as test_file:

    train_writer = csv.writer(train_file)
    test_writer = csv.writer(test_file)

    for path in INPUTS:
        print(f"Splitting {path.name}...")

        with path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.reader(file)
            header = next(reader, None)

            if header is None or LABEL_COLUMN not in header:
                raise ValueError(f"Invalid dataset header: {path}")

            if expected_header is None:
                expected_header = header
                train_writer.writerow(header)
                test_writer.writerow(header)
            elif header != expected_header:
                raise ValueError("Dataset columns do not match.")

            label_index = header.index(LABEL_COLUMN)

            for row in reader:
                if len(row) != len(header):
                    raise ValueError(f"Malformed row in {path.name}")

                digest = feature_hash(row, label_index)
                bucket = int(digest[:16], 16) % TEST_BUCKETS
                label = row[label_index].strip()

                if bucket == 0:
                    test_writer.writerow(row)
                    test_rows += 1
                    test_labels[label] += 1
                    test_groups.add(digest)
                else:
                    train_writer.writerow(row)
                    train_rows += 1
                    train_labels[label] += 1
                    train_groups.add(digest)

                if (train_rows + test_rows) % 200000 == 0:
                    print(
                        f"Processed {train_rows + test_rows:,} rows"
                    )

print("\nSplit completed.")
print(f"Training rows: {train_rows:,}")
print(f"Testing rows:  {test_rows:,}")
print(f"Training labels: {len(train_labels)}")
print(f"Testing labels:  {len(test_labels)}")
print(f"Training feature groups: {len(train_groups):,}")
print(f"Testing feature groups:  {len(test_groups):,}")
print(f"Groups shared across partitions: "
      f"{len(train_groups & test_groups):,}")
print(f"Training file: {TRAIN_PATH}")
print(f"Testing file:  {TEST_PATH}")
print("Input datasets were not modified.")
