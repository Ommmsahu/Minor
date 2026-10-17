
from pathlib import Path
import csv
from collections import Counter

PROJECT_DIR = Path(__file__).resolve().parent

DATASETS = {
    "Merged01": PROJECT_DIR / "output" / "Merged01_clean.csv",
    "Merged02": PROJECT_DIR / "output" / "Merged02_clean.csv",
}

for name, path in DATASETS.items():
    print("\n" + "=" * 55)
    print(f"Label Audit: {name}")
    print("=" * 55)

    if not path.is_file():
        print(f"File not found: {path}")
        continue

    label_counts = Counter()
    missing_labels = 0
    total_rows = 0

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if "Label" not in (reader.fieldnames or []):
            print("ERROR: Label column not found.")
            continue

        for row in reader:
            total_rows += 1
            label = row["Label"]

            if label is None or not label.strip():
                missing_labels += 1
            else:
                label_counts[label.strip()] += 1

    print(f"Total rows: {total_rows:,}")
    print(f"Number of distinct labels: {len(label_counts)}")
    print(f"Missing/blank labels: {missing_labels}")

    print("\nRecords per label:")
    for label, count in sorted(label_counts.items()):
        percentage = count / total_rows * 100 if total_rows else 0
        print(f"{label}: {count:,} ({percentage:.4f}%)")

print("\nLabel audit completed. No data was modified.")
