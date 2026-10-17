
from pathlib import Path
from collections import Counter
from itertools import combinations, groupby
import sqlite3

PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "reports" / "task3_conflict_audit.sqlite"

if not DB_PATH.exists():
    raise FileNotFoundError(
        f"Audit database not found: {DB_PATH}\n"
        "Run task3_conflict_audit.py first."
    )

pair_counts = Counter()
conflict_group_sizes = Counter()
total_feature_groups = 0
conflicting_groups = 0

with sqlite3.connect(DB_PATH) as connection:
    cursor = connection.execute("""
        SELECT feature_hash, label
        FROM feature_labels
        ORDER BY feature_hash, label
    """)

    for feature_hash, group in groupby(cursor, key=lambda row: row[0]):
        labels = [row[1] for row in group]
        total_feature_groups += 1

        if len(labels) > 1:
            conflicting_groups += 1
            conflict_group_sizes[len(labels)] += 1

            # Count each distinct label pair once per feature combination.
            for pair in combinations(labels, 2):
                pair_counts[pair] += 1

print("=" * 65)
print("GreenFed-IDS: Conflicting Label Summary")
print("=" * 65)
print(f"Distinct feature combinations: {total_feature_groups:,}")
print(f"Combinations with multiple labels: {conflicting_groups:,}")

print("\nConflicting groups by number of distinct labels:")
for size, count in sorted(conflict_group_sizes.items()):
    print(f"{size} labels: {count:,} feature combinations")

print("\nTop 20 conflicting label pairs:")
for (label_a, label_b), count in pair_counts.most_common(20):
    print(f"{label_a} <-> {label_b}: {count:,} feature combinations")

print("\nAudit complete. No dataset records were modified.")
