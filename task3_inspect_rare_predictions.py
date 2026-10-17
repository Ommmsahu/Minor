
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
PATH = BASE / "reports" / "random_forest_confusion_matrix.csv"

cm = pd.read_csv(PATH, index_col=0)

rare_classes = [
    "BACKDOOR_MALWARE",
    "RECON-PINGSWEEP",
    "UPLOADING_ATTACK",
]

for label in rare_classes:
    if label not in cm.index:
        print(f"\nLabel not found: {label}")
        continue

    actual_counts = cm.loc[label]
    total = actual_counts.sum()
    correct = actual_counts[label] if label in cm.columns else 0

    print(f"\n{'=' * 60}")
    print(f"Actual class: {label}")
    print(f"Test examples: {total}")
    print(f"Correctly predicted: {correct}")

    wrong = actual_counts.drop(
        labels=[label], errors="ignore"
    ).sort_values(ascending=False)

    wrong = wrong[wrong > 0].head(10)

    if wrong.empty:
        print("No incorrect predictions recorded.")
    else:
        print("Most common incorrect predictions:")
        for predicted_label, count in wrong.items():
            print(f"  Predicted as {predicted_label}: {int(count)}")

print("\nInspection complete. No files were modified.")
