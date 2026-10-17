
from pathlib import Path
import sqlite3

BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "reports" / "task3_conflict_audit.sqlite"

if not DB_PATH.exists():
    raise FileNotFoundError(f"Audit database not found: {DB_PATH}")

with sqlite3.connect(DB_PATH) as con:
    query = """
        WITH group_labels AS (
            SELECT
                feature_hash,
                COUNT(DISTINCT label) AS number_of_labels
            FROM row_counts
            GROUP BY feature_hash
        )
        SELECT
            r.label,
            SUM(r.row_count) AS original_rows,
            COUNT(DISTINCT r.feature_hash) AS feature_groups,
            SUM(
                CASE WHEN g.number_of_labels > 1
                     THEN 1 ELSE 0 END
            ) AS groups_with_conflicts
        FROM row_counts r
        JOIN group_labels g USING (feature_hash)
        GROUP BY r.label
        ORDER BY feature_groups ASC
    """

    results = con.execute(query).fetchall()

print("Label-wise group readiness")
print("-" * 88)
print(f"{'Label':32} {'Rows':>10} {'Groups':>10} {'Conflict groups':>17}")
print("-" * 88)

for label, rows, groups, conflicts in results:
    print(f"{label:32} {rows:10,} {groups:10,} {conflicts:17,}")

print("\nAudit complete. No CSV data was changed.")
