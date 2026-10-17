# GreenFed-IDS

**A Context-Aware Federated Learning Framework for Sustainable IoT Intrusion Detection**

## Project overview

GreenFed-IDS investigates federated learning for IoT intrusion detection using the CICIoT2023 dataset.

The project begins with a centralized intrusion-detection baseline and will extend to simulated federated clients, FedAvg, and context-aware client selection.

## Current progress

- Pilot dataset cleaning and basic validation completed.
- Label distribution and feature-conflict audits completed.
- Train/test split generated with exact matching feature groups kept in the same partition.
- Centralized Random Forest baseline trained and evaluated.
- Federated-learning client simulation is the next milestone.

## Initial Random Forest baseline

| Metric | Result |
|---|---:|
| Accuracy | 76.05% |
| Balanced accuracy | 59.38% |
| Macro-F1 | 57.11% |
| Weighted-F1 | 75.37% |

These are initial results for the current experimental setup, not final project results. The model performs unevenly across classes, so accuracy should not be considered alone.

## Dataset

The project uses CICIoT2023 data.

Large dataset files and train/test splits are intentionally excluded from GitHub. Obtain them from the project owner through the team's approved file-sharing method.

Expected local paths:

- `output/Merged01_clean.csv`
- `output/Merged02_clean.csv`
- `output/splits/train.csv`
- `output/splits/test.csv`

Verify dataset dimensions and labels before running experiments.

## Setup

Create a Python virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproducibility

- Keep the test split separate from training and model tuning.
- Fit learned preprocessing on training data only.
- Use a consistent feature order and class-index mapping.
- Record random seeds, model settings, and evaluation metrics.
- Report macro-F1 and balanced accuracy alongside accuracy.
- Distinguish simulated resource or energy values from actual measurements.

## Project structure

- `GreenFed-TDS-main/`: dataset audit and Task 2 preprocessing work.
- `reports/`: compact audit and model evaluation reports.
- `task3_*.py`: dataset audits, split validation, and centralized baseline scripts.
- `federated/`: federated-learning implementation and task instructions.

## Next milestone

Build IID and Non-IID client partitions, test a shared PyTorch MLP, and run a short Flower FedAvg experiment.

See [`federated/README.md`](federated/README.md).
