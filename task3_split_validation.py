
from pathlib import Path
import csv
from collections import Counter

BASE = Path(__file__).resolve().parent
SPLITS = {
    "Train": BASE / "output" / "splits" / "train.csv",
    "Test": BASE / "output" / "splits" / "test.csv",
}

counts_by_split = {}
total_by_split = {}

for split_name, path in SPLITS.items():
    if not path.is_file():
        raise FileNotFoundError(path)

    counts = Counter()
    total = 0

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if "Label" not in (reader.fieldnames or []):
            raise ValueError(f"Label column missing in {path}")

        for row in reader:
            label = (row["Label"] or "").strip()
            if not label:
                raise ValueError(f"Blank label found in {path}")
            counts[label] += 1
            total += 1

    counts_by_split[split_name] = counts
    total_by_split[split_name] = total

all_labels = sorted(
    set(counts_by_split["Train"]) | set(counts_by_split["Test"])
)

print(f"{'Label':32} {'Train':>10} {'Test':>10} {'Test %':>9}")
print("-" * 65)

for label in all_labels:
    train_n = counts_by_split["Train"][label]
    test_n = counts_by_split["Test"][label]
    test_pct = test_n / total_by_split["Test"] * 100
    print(f"{label:32} {train_n:10,} {test_n:10,} {test_pct:8.4f}%")

print("\nSummary")
for name in SPLITS:
    print(
        f"{name}: {total_by_split[name]:,} rows, "
        f"{len(counts_by_split[name])} labels"
    )

print("\nValidation complete. Split files were not modified.")
