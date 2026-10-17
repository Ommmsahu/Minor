# FedAvg baseline: how to run

Tested with Flower 1.40.0 and PyTorch 2.14.1 (CPU). Run everything from the repo root with the venv active.

1. Put `train.csv` at `output/splits/train.csv`. `test.csv` is never read by this code.
2. `python tests/test_federated.py` runs the unit tests.
3. `python -m federated.partition_data` validates the schema (39 features, 34 classes), holds out a stratified
   10% validation set from train.csv, fits the scaler on the remaining training rows, builds the IID and
   Dirichlet non-IID partitions (seed 42, alpha 0.5, minimum 1000 rows per client) and writes
   `reports/client_partition_summary.csv`, `reports/client_class_counts_*.csv`, `reports/schema_config.json`.
4. `python -m federated.run_simulation --partition iid --rounds 2 2>&1 | Tee-Object -FilePath logs\fedavg_smoke_test_iid.log`
   `python -m federated.run_simulation --partition noniid --rounds 2 2>&1 | Tee-Object -FilePath logs\fedavg_smoke_test_noniid.log`

Outputs: `reports/fedavg_smoke_test_<type>.json/.csv`, `reports/fedavg_per_class_val_<type>.json`.

Settings: MLP 39-128-64-34, Adam lr 1e-3, 1 local epoch, batch 256, cross-entropy without class weights,
FedAvg weighted by num-examples, all 5 clients every round, CPU only.

Known limitations: the scaler uses statistics pooled over all client training rows (a simulation shortcut);
standardized features are clipped to [-10, 10]; scores come from a validation hold-out of train.csv, not test.csv;
`flwr.simulation.run_simulation` is flagged as deprecated in Flower 1.40 but still works; no resource or energy
values are measured. Generated files in `output/fl_cache/` and `logs/` must not be committed.
