# GreenFed-IDS — Dataset Audit

## Dataset
CICIoT2023 — MERGED_CSV

## Files Audited

- Merged01.csv
- Merged02.csv

## Audit Results

| Metric         | Merged01.csv | Merged02.csv |
|----------------|--------------|--------------|
| File Size      | 140.30 MB    | 147.48 MB    |
| Rows           | 712,311      | 748,585      |
| Columns        | 40           | 40           |
| Label Present  | Yes          | Yes          |
| Distinct Labels| 34           | 34           |
| Missing Values | 22           | 12           |
| Infinite Values| 14           | 13           |
| Duplicate Rows*| 48,779       | 50,722       |

\* Duplicate counts are within chunks only.

## Schema Compatibility

- Column names: Identical
- Column order: Identical
- Number of columns: 40
- Label column: Present in both
- Data types: Identical
- Data type differences: None

## Compatibility Decision

The two files are compatible for a later merge, subject to reviewing
data-quality findings and documenting the merge plan.

## Data Quality Findings

### Merged01.csv
- Missing values: 22
- Infinite values: 14
- Duplicate rows within chunks: 48,779

### Merged02.csv
- Missing values: 12
- Infinite values: 13
- Duplicate rows within chunks: 50,722

## Limitations

- Duplicate checking was performed within chunks only.
- Duplicates crossing chunk boundaries were not counted.
- Duplicates shared between the two files were not counted.
- Data types were recorded from the first chunk.
- No source data was modified.
- No cleaning, relabeling, merging, splitting, or model training was performed.

## Output Reports

- `dataset_audit.csv`
- `schema_and_quality.csv`
- `label_distribution.csv`

## Status

**Step 1 — Dataset Audit: COMPLETED**