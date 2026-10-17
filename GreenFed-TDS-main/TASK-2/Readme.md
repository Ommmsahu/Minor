# Task 2: Dataset Preprocessing

## Objective
Clean and validate Merged01.csv and Merged02.csv.

## Input Datasets
- Merged01.csv: 712,311 rows and 40 columns
- Merged02.csv: 748,585 rows and 40 columns

Each dataset contains 39 features and one target column: Label.

## Preprocessing Performed
1. Verified dataset structure and column consistency.
2. Investigated missing values and infinity values.
3. Replaced positive and negative infinity with missing values.
4. Imputed missing values in Rate, Std, and Variance using
   the median calculated separately for each file.
5. Preserved all rows, labels, columns, and duplicate records.
6. Validated the cleaned datasets.
7. Verified that all 34 label counts match before and after cleaning.


## Median-Imputation Verification

| Dataset      | Column   | Median      |
|--------------|----------|-------------|
| Merged01.csv | Std      | 0.0         |
| Merged01.csv | Variance | 0.0         |
| Merged01.csv | Rate     | 24613.015668|
| Merged02.csv | Std      | 0.0         |
| Merged02.csv | Variance | 0.0         |
| Merged02.csv | Rate     | 24588.486341|

The zero medians for Std and Variance are supported by the
original data, where most valid finite values in these
columns are zero. Medians were calculated separately
for each file.
## Results

| Dataset | Original Rows | Cleaned Rows | Original Missing Cells | Original Infinity Values |
|---|---:|---:|---:|---:|
| Merged01.csv | 712311 | 712311 | 22 | 14 |
| Merged02.csv | 748585 | 748585 | 12 | 13 |

Both cleaned datasets contain zero missing cells and zero
infinity values according to the validation report.

## Duplicate Audit

| Dataset | Total Rows | Distinct Records | Duplicate Excess Rows |
|---|---:|---:|---:|
| Merged01.csv | 712311 | 544943 | 167368 |
| Merged02.csv | 748585 | 568678 | 179907 |

Duplicate records were retained.


## Cross-File Overlap and Label Conflicts

- Distinct complete records in Merged01.csv: 544,943
- Distinct complete records in Merged02.csv: 568,678
- Distinct complete records shared by both files: 99,430
- Feature combinations associated with multiple labels across the combined datasets: 44,540

The shared-record count includes the Label column. The multiple-label count is calculated across both datasets combined and requires further investigation before deciding how to handle conflicting labels.

No duplicate records were deleted, and no labels were changed.


## Output Files
- output/Merged01_clean.csv
- output/Merged02_clean.csv

## Reports
- reports/before_after_comparison.csv
- reports/cleaning_validation.csv
- reports/cross_file_overlap.csv
- reports/duplicate_report.csv
- reports/final_preprocessing_report.csv
- reports/label_distribution_comparison.csv


## Scope and Evaluation Leakage Prevention

Only Merged01.csv and Merged02.csv were processed.

- Duplicate records were retained.
- Original labels were not changed.
- No model training or train/test split has been performed.

### Planned leakage prevention

Before the centralized baseline IDS experiment:

1. Identify groups of records with identical feature values.
2. Assign each duplicate group entirely to either the training set or the testing set.
3. Ensure that duplicate groups cannot appear in both sets.
4. Investigate feature-identical records with conflicting labels before deciding how to handle them.
5. Preserve all original records until any data-removal decision is approved.
6. Fit imputation values using the training partition only.
   Apply those same fitted values to validation and test data,
   without recalculating medians on those partitions.
This is a planned evaluation safeguard, not a completed train/test split.

## Conclusion
Both datasets passed the current preprocessing validation checks.
All original rows and columns were preserved.