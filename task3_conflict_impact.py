
from pathlib import Path
import csv
import sqlite3

BASE = Path(__file__).resolve().parent
DATASETS = [
    BASE / "output" / "Merged01_clean.csv",
    BASE / "output" / "Merged02_clean.csv",
]
DB = BASE / "reports" / "task3_conflict_audit.sqlite"

con = sqlite3.connect(DB)

# Store the number of original rows for each feature-label combination.
con.execute("""
    DROP TABLE IF EXISTS row_counts
""")
con.execute("""
    CREATE TABLE row_counts (
        feature_hash TEXT,
        label TEXT,
        row_count INTEGER,
        PRIMARY KEY (feature_hash, label)
    )
""")

# Recompute feature hashes using the same procedure as the previous audit.
import hashlib
import json

insert = """
    INSERT INTO row_counts VALUES (?, ?, 1)
    ON CONFLICT(feature_hash, label)
    DO UPDATE SET row_count = row_count + 1
"""

try:
    for path in DATASETS:
        print("Counting rows in", path.name)
        batch = []

        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)
            label_index = header.index("Label")

            for row in reader:
                if len(row) != len(header):
                    continue

                label = row[label_index].strip()
                features = [
                    value.strip()
                    for i, value in enumerate(row)
                    if i != label_index
                ]
                text = json.dumps(
                    features, ensure_ascii=False, separators=(",", ":")
                )
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                batch.append((digest, label))

                if len(batch) >= 5000:
                    con.executemany(insert, batch)
                    con.commit()
                    batch.clear()

        if batch:
            con.executemany(insert, batch)
            con.commit()

    # Count total row occurrences belonging to conflicting feature groups.
    result = con.execute("""
        WITH conflict_groups AS (
            SELECT feature_hash
            FROM row_counts
            GROUP BY feature_hash
            HAVING COUNT(DISTINCT label) > 1
        )
        SELECT
            COUNT(DISTINCT r.feature_hash),
            SUM(r.row_count)
        FROM row_counts r
        JOIN conflict_groups c USING (feature_hash)
    """).fetchone()

    print("\nConflicting feature combinations:", f"{result[0]:,}")
    print("Original row occurrences in those groups:", f"{result[1]:,}")

    print("\nMost frequent conflicting feature-label groups:")
    rows = con.execute("""
        SELECT r.feature_hash, r.label, r.row_count
        FROM row_counts r
        JOIN (
            SELECT feature_hash
            FROM row_counts
            GROUP BY feature_hash
            HAVING COUNT(DISTINCT label) > 1
        ) c USING (feature_hash)
        ORDER BY r.row_count DESC
        LIMIT 20
    """).fetchall()

    for digest, label, count in rows:
        print(f"{digest[:16]}... | {label} | {count:,} rows")

    print("\nOriginal CSV files were not modified.")

finally:
    con.close()
