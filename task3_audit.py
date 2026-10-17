
from pathlib import Path
import csv

# Project folder = folder containing this Python script
PROJECT_DIR = Path(__file__).resolve().parent

# Update these paths if your files are stored elsewhere
DATASETS = {
    "Merged01": PROJECT_DIR / "output" / "Merged01_clean.csv",
    "Merged02": PROJECT_DIR / "output" / "Merged02_clean.csv",
}

EXPECTED_COLUMNS = 40
EXPECTED_LABEL_COLUMN = "Label"

print("=" * 60)
print("GreenFed-IDS | Task 3: Dataset Audit")
print("=" * 60)

for name, path in DATASETS.items():
    print(f"\nDataset: {name}")
    print(f"Path: {path}")

    if not path.is_file():
        print("STATUS: FILE NOT FOUND")
        continue

    print(f"File size: {path.stat().st_size / (1024**3):.2f} GB")

    # Read only the header and count rows without loading data into RAM
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        header = next(reader, None)

        if header is None:
            print("STATUS: EMPTY FILE")
            continue

        row_count = sum(1 for _ in reader)

    print(f"Columns: {len(header)}")
    print(f"Data rows: {row_count:,}")
    print(f"Label column present: {EXPECTED_LABEL_COLUMN in header}")
    print(f"Column names unique: {len(header) == len(set(header))}")
    print(f"Expected 40 columns: {len(header) == EXPECTED_COLUMNS}")

    if len(header) != len(set(header)):
        duplicates = sorted({
            col for col in header if header.count(col) > 1
        })
        print(f"Repeated column names: {duplicates}")

    if EXPECTED_LABEL_COLUMN in header:
        print("STATUS: BASIC CHECK COMPLETE")
    else:
        print("STATUS: CHECK FAILED - Label column missing")

print("\nAudit finished. No data was modified.")
