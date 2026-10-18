# GreenFed-IDS: Federated Learning Baseline (FedAvg)

Federated-learning pipeline for GreenFed-IDS. Five simulated IoT clients train a shared PyTorch MLP on the CICIoT2023 training split. A Flower server combines their updates with Federated Averaging (FedAvg). Context-aware client selection is not part of this milestone.

Status: baseline implemented and smoke-tested; reliability and reproducibility checks added after review. Results in section 5 come from the first real runs on the project's `train.csv`. Results of the re-run with the new checks are to be added once run on the real data.

## 1. Files

| Purpose | File |
|---|---|
| Data validation, trusted schema, cache identity and verification | `federated/data_utils.py` |
| IID and non-IID partitions, partition metadata and integrity checks | `federated/partition_data.py` |
| Shared MLP (39 inputs, 34 classes), local training, evaluation | `federated/fl_model.py` |
| Flower client | `federated/fl_client_simulation.py` |
| Flower server (FedAvg, global validation) | `federated/fl_server.py` |
| Run settings, validation, run IDs and per-run log paths | `federated/fl_common.py` |
| Post-run integrity checks | `federated/run_checks.py` |
| Atomic file writes and hashing | `federated/io_utils.py` |
| Runner, reports and runtime check | `federated/run_simulation.py` |
| Tests (22) | `tests/test_federated.py` |

`federated/README.md` is the original task description and is unchanged.

## 2. Data rules

- Input: `output/splits/train.csv` only. It is not in the repository. `test.csv` is never opened. Any path named like `test*` is refused by `read_train_csv` and `prepare_cache`.
- Target column: `Label` (text). It is never an input feature.
- Schema: the exact ordered list of 39 feature names is stored in `TRUSTED_FEATURES`. A file with different names, or the same names in a different order, is rejected. Reordering is possible only by passing `allow_reorder=True` explicitly, and the cache records when it happened.
- Values: every feature must be numeric and finite, `Label` must be non-null, and there must be exactly 34 classes.
- Label map: the 34 labels are sorted alphabetically and numbered 0 to 33, saved in `reports/schema_config.json` and used by every client.
- Validation set: a stratified 10% hold-out of `train.csv` (seed 42). On the real file this leaves 1,053,019 client rows and 117,003 validation rows.
- Scaling: standardization fit on the client rows only, then clipped to [-10, 10]. The statistics are pooled over all clients, which is a simulation shortcut.
- Cache identity: `output/fl_cache/cache_meta.json` stores the CSV path, size, modification time and SHA-256, the seed, validation fraction, preprocessing settings, feature names and order, label map, row counts and a `cache_id`. The cache is reused only if size, SHA-256, seed, validation fraction, preprocessing and schema all match and the arrays pass verification (shapes, dtypes, finite values, label range, all 34 classes, train and validation source rows disjoint and covering the source exactly once). Otherwise it is rebuilt.

## 3. Client partitions (seed 42)

**IID.** Each class is shuffled and dealt out in near-equal chunks, so every client sees all 34 classes in nearly the same proportions.

**Non-IID.** Label-wise Dirichlet split with alpha = 0.5, repeated until every client holds at least 1,000 rows.

Checks on every partition: every client-training row is assigned exactly once, indices are in range, the client count matches, no client is below the minimum size, and mapped back to `train.csv` row numbers no client row is a validation row. Each partition is saved with metadata (type, seed, alpha, client count, client sizes, `cache_id`). A run refuses to start if the metadata does not match the active cache or the run settings.

Real partition statistics from the first runs:

| Partition | Client | Rows | Classes present | Most common class | Share | Distance to global mix* |
|---|---|---|---|---|---|---|
| IID | 0 | 210,618 | 34 | DDOS-ICMP_FLOOD | 0.153 | 0.0 |
| IID | 1 | 210,611 | 34 | DDOS-ICMP_FLOOD | 0.153 | 0.0 |
| IID | 2 | 210,603 | 34 | DDOS-ICMP_FLOOD | 0.153 | 0.0 |
| IID | 3 | 210,596 | 34 | DDOS-ICMP_FLOOD | 0.153 | 0.0 |
| IID | 4 | 210,591 | 34 | DDOS-ICMP_FLOOD | 0.153 | 0.0 |
| Non-IID | 0 | 197,176 | 34 | DDOS-PSHACK_FLOOD | 0.327 | 0.498 |
| Non-IID | 1 | 344,570 | 33 | DDOS-ICMP_FLOOD | 0.371 | 0.368 |
| Non-IID | 2 | 185,859 | 32 | DDOS-RSTFINFLOOD | 0.378 | 0.446 |
| Non-IID | 3 | 145,342 | 33 | DDOS-UDP_FLOOD | 0.393 | 0.510 |
| Non-IID | 4 | 180,072 | 34 | DDOS-SYNONYMOUSIP_FLOOD | 0.192 | 0.324 |

\*Total variation distance between the client's class distribution and the global one (0 = identical, 1 = completely different).

Reports: `reports/client_class_counts_<type>.csv`, `reports/client_partition_summary.csv`.

## 4. Model, training and run checks

- MLP 39 -> 128 -> 64 -> 34, ReLU, raw logits. Adam, lr 1e-3, 1 local epoch, batch 256, cross-entropy without class weights. CPU only.
- FedAvg: Flower's built-in strategy, `fraction_train=1.0`, client-side evaluation disabled, updates weighted by `num-examples`. The global model is scored on the validation hold-out before training (round 0) and after every round: loss, accuracy, balanced accuracy, macro-F1, weighted-F1 and per-class precision, recall and F1.
- Client round number: taken from `server-round`, which Flower's FedAvg adds to the train config. Clients also check that their ID is in range, their partition size matches the metadata, incoming and outgoing parameter names and shapes match the shared MLP, loss and time are finite, and `num-examples` equals the rows trained on.
- Server checks: model output size, label map and validation labels agree with 34 classes; validation features have 39 columns; every metric is finite.
- Run isolation: every run gets a unique `run_id`, a log folder `logs/runs/<run_id>/` (run config, client log, server log, `console.log`) and a report folder `reports/runs/<run_id>/`. Runs cannot overwrite each other, and the earlier reports in `reports/` are never touched. Reports are written atomically.
- Completion checks after Flower returns: exactly one update per client per round, no duplicates, one finite global validation record for rounds 0 to N, client row counts equal partition sizes, all records belong to this run. If any check fails the run is marked FAILED, a `..._FAILED.json` file is written, no success summary is printed and the exit code is 1. The final record used for the per-class report is the one with the highest round.
- Timing: `server_callback_interval_s` is the gap between global-validation callbacks. It is not client training time and not energy use.
- Runtime check: after the run the console log is scanned for Ray fatal messages (for example `access violation`) and the machine is checked for Ray processes left running. Problems are recorded in the report and do not get suppressed.

Exit codes of `run_simulation`: 0 verified and clean, 1 training integrity failed, 2 invalid configuration or inputs, 3 FedAvg rounds verified but the runtime check found Ray errors or leftover processes.

## 5. First smoke-test results (2 rounds, 5 clients, real data)

From the first real runs (reports `reports/fedavg_smoke_test_iid.json` and `..._noniid.json`, kept unchanged). Scores are on the validation hold-out from `train.csv`, not `test.csv`.

| Partition | Round | Mean train loss | Val accuracy | Val macro-F1 | Val loss |
|---|---|---|---|---|---|
| (untrained) | 0 | n/a | 0.0011 | 0.0014 | 3.529 |
| IID | 1 | 0.660 | 0.7449 | 0.4370 | 0.503 |
| IID | 2 | 0.498 | 0.7504 | 0.4722 | 0.486 |
| Non-IID | 1 | 0.493 | 0.7121 | 0.3032 | 0.599 |
| Non-IID | 2 | 0.353 | 0.7305 | 0.3956 | 0.518 |

Both runs: 5 results and 0 failures in every round, finite loss. The non-IID model scores lower on macro-F1 (0.396 against 0.472). One seed and two rounds, so this is a first indication only. These numbers are not comparable to the Random Forest baseline (76.05% accuracy, 57.11% macro-F1), which is a different model scored on the held-out test split.

The first runs showed Ray worker shutdown errors on Windows (`Windows fatal exception: access violation`). Training results were complete, but this is not treated as healthy runtime. See section 8.

## 6. Environment of the first real runs

Python 3.14.3, PyTorch 2.14.1 (CPU), Flower 1.40.0, Ray 2.59.0, Windows. Each new run records its own versions in its report.

## 7. How to reproduce

From the repository root with the virtual environment active:

```
pip install -r requirements.txt
python tests/test_federated.py
python -m federated.partition_data
python -m federated.run_simulation --partition iid --rounds 2 --seed 42
python -m federated.run_simulation --partition noniid --rounds 2 --seed 42 --alpha 0.5
```

`partition_data` rebuilds the cache the first time and whenever the CSV or settings change. Re-running it with the same CSV and seed should leave `reports/schema_config.json`, `client_partition_summary.csv` and `client_class_counts_*.csv` unchanged (check with `git diff --stat reports`).

After a run, check the exit code (`$LASTEXITCODE` in PowerShell) and open `reports/runs/<run_id>/fedavg_smoke_test_<type>.json`. The console log is saved automatically as `logs/runs/<run_id>/console.log`, so `Tee-Object` is no longer needed. Set `RUN_E2E=1` to include the slow end-to-end test in `python tests/test_federated.py`.

Options: `run_simulation` takes `--rounds` (1 to 3), `--clients`, `--local-epochs`, `--batch-size`, `--lr`, `--seed`, `--alpha`, `--run-id`. `partition_data` takes `--seed`, `--alpha`, `--clients`, `--min-size`, `--val-fraction`, `--force`.

## 8. Windows and Ray

Ray on Windows is experimental according to Flower, and the first runs printed access-violation and worker-shutdown errors after training finished. This has not been fixed or suppressed. To check a run: read `runtime_check` in the report and the exit code (3 means problems were detected). To test WSL2, install Ubuntu under WSL2, clone the repository there, create a Linux virtual environment, install `requirements.txt`, copy `train.csv` to `output/splits/`, and repeat section 7. Compare the exit code and runtime check with the Windows runs and note both in the report.

`flwr.simulation.run_simulation` is marked deprecated in Flower 1.40. It is kept on purpose so that experiment behaviour does not change. Moving to `flwr run` needs a Flower app project layout and should be a separate change, followed by a comparison of results.

## 9. Files not in Git

`output/splits/train.csv` and `test.csv` (datasets), `output/fl_cache/` (processed arrays, over 1 GB) and `logs/` (console and run logs). Obtain the datasets from the project owner.

## 10. Known limitations

- Scaling statistics are pooled over all clients, and features are clipped to [-10, 10].
- Cross-entropy has no class weights. The smallest class has 33 rows, so rare classes score poorly after two rounds, and some non-IID clients have no examples of one or two classes.
- Two rounds is a pipeline check, not a converged result.
- The Windows and Ray shutdown errors are open (section 8).
- No resource, energy or network values are measured.
- No held-out test evaluation has been run. The owner coordinates that.
- No centralized MLP with the same architecture has been trained yet.

## 11. Next steps

1. Re-run both smoke tests on the real data with the new checks and compare with the preserved reports.
2. Run the WSL2 comparison.
3. Train a centralized MLP with the same architecture, scaling and validation split.
4. Run more rounds and several seeds, and try other alpha values.
5. Add context-aware client selection after the baseline is accepted.
