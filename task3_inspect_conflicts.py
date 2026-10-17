
from pathlib import Path
import csv

PROJECT_DIR = Path(__file__).resolve().parent
REPORT_PATH = PROJECT_DIR / "reports" / "conflicting_feature_labels.csv"

with REPORT_PATH.open(
    "r", newline="", encoding="utf-8-sig"
) as file:
    reader = csv.DictReader(file)
    examples = []

    for row in reader:
        examples.append(row)
        if len(examples) == 10:
            break

print("First 10 conflicting feature combinations:\n")

for i, row in enumerate(examples, start=1):
    print(f"{i}. Hash: {row['feature_hash']}")
    print(f"   Labels: {row['conflicting_labels']}\n")
