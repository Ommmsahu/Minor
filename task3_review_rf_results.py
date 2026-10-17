
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
REPORT = BASE / "reports" / "random_forest_classification_report.csv"

df = pd.read_csv(REPORT, index_col=0)

# Keep only individual class rows, excluding aggregate averages.
summary = df.loc[
    ~df.index.isin(["accuracy", "macro avg", "weighted avg", "micro avg"]),
    ["precision", "recall", "f1-score", "support"]
].copy()

summary = summary[
    summary.index.notna()
    & (summary.index.astype(str).str.strip() != "")
]

summary["support"] = summary["support"].astype(int)
summary = summary.sort_values("f1-score")

print("\n10 classes with the lowest F1:")
print(summary.head(10).round(4).to_string())

print("\n10 classes with the highest F1:")
print(summary.tail(10).sort_values("f1-score", ascending=False)
      .round(4).to_string())

print("\nClasses with fewer than 50 test examples:")
rare = summary[summary["support"] < 50]
print(rare.round(4).to_string() if not rare.empty else "None")

print("\nFull per-class review complete.")
