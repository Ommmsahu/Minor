
from pathlib import Path
import csv
import hashlib
import json
import sqlite3

PROJECT_DIR = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

DATASETS = [
    PROJECT_DIR / "output" / "Merged01_clean.csv",
    PROJECT_DIR / "output" / "Merged02_clean.csv",
]

DB_PATH = REPORT_DIR / "task3_conflict_audit.sqlite"
RESULT_PATH = REPORT_DIR / "conflicting_feature_labels.csv"

# A separate audit database avoids keeping all records in RAM.
if DB_PATH.exists():
    DB_PATH.unlink()

connection = sqlite3.connect(DB_PATH)
connection.execute("""
    CREATE TABLE feature_labels (
        feature_hash TEXT NOT NULL,
        label TEXT NOT NULL,
        PRIMARY KEY (feature_hash, label)
    )
""")

insert_sql = """
    INSERT OR IGNORE INTO feature_labels (feature_hash, label)
    VALUES (?, ?)
"""

try:
    expected_header = None
    batch = []
    total_rows = 0

    for path in DATASETS:
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")

        print(f"Auditing: {path.name}")

        with path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.reader(file)
            header = next(reader, None)

            if header is None or "Label" not in header:
                raise ValueError(f"Invalid header in {path.name}")

            if expected_header is None:
                expected_header = header
            elif header != expected_header:
                raise ValueError(
                    "Column names or order differ between datasets."
                )

            label_index = header.index("Label")

            for row in reader:
                if len(row) != len(header):
                    continue

                label = row[label_index].strip()
                features = [
                    value.strip()
                    for index, value in enumerate(row)
                    if index != label_index
                ]

                # Hash feature values only; exclude the target label.
                feature_text = json.dumps(
                    features, ensure_ascii=False, separators=(",", ":")
                )
                feature_hash = hashlib.sha256(
                    feature_text.encode("utf-8")
                ).hexdigest()

                batch.append((feature_hash, label))
                total_rows += 1

                if len(batch) >= 5000:
                    connection.executemany(insert_sql, batch)
                    connection.commit()
                    batch.clear()

                if total_rows % 100000 == 0:
                    print(f"Rows checked: {total_rows:,}")

    if batch:
        connection.executemany(insert_sql, batch)
        connection.commit()

    print("\nChecking for feature combinations with multiple labels...")

    query = """
        SELECT feature_hash, GROUP_CONCAT(label, ' | ') AS labels
        FROM feature_labels
        GROUP BY feature_hash
        HAVING COUNT(*) > 1
    """

    conflicts = connection.execute(query).fetchall()

    with RESULT_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["feature_hash", "conflicting_labels"])
        writer.writerows(conflicts)

    print(f"\nTotal rows scanned: {total_rows:,}")
    print(f"Conflicting feature combinations: {len(conflicts):,}")
    print(f"Detailed report: {RESULT_PATH}")
    print("Original cleaned datasets were not modified.")

finally:
    connection.close()
