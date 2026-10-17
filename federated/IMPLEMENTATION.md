# GreenFed-IDS: Federated Learning Baseline (FedAvg)

First working federated-learning pipeline for GreenFed-IDS. Five simulated IoT clients train a shared PyTorch MLP on the CICIoT2023 training split. A Flower server combines their updates with Federated Averaging (FedAvg). Context-aware client selection is not part of this milestone.

Status: initial implementation and 2-round smoke test complete. All numbers below come from real runs on the project's `train.csv`.

## 1. What this adds

| Requirement | Where |
|---|---|
| Data validation and one global label map | `federated/data_utils.py` |
| Five clients, IID and non-IID partitions | `federated/partition_data.py` |
| Shared MLP (39 inputs, 34 classes) | `federated/fl_model.py` |
| Flower client (local training) | `federated/fl_client_simulation.py` |
| Flower server (FedAvg) | `federated/fl_server.py` |
| Smoke-test runner and result reports | `federated/run_simulation.py` |
| Shared run settings | `federated/fl_common.py` |
| Unit tests | `tests/test_federated.py` |

`federated/README.md` is the original task description and is unchanged.

## 2. Data rules

- Input: `output/splits/train.csv` only. It is not in the repository.
- `test.csv` is never opened, used for scaling, tuning or model selection.
- Target column: `Label` (text). It is never used as an input feature.
- Verified on the real file: 1,170,022 rows, 39 numeric features, 34 classes, no missing or non-finite values.
- Label mapping: the 34 labels are sorted alphabetically and numbered 0 to 33. The map and the feature order are saved in `reports/schema_config.json`, and every client uses them.
- Validation set: a stratified 10% hold-out taken from `train.csv` (seed 42). This leaves 1,053,019 rows for clients and 117,003 for validation.
- Scaling: standardization fit on the 1,053,019 client rows only, then clipped to [-10, 10]. The statistics are pooled over all clients, which is a simulation shortcut. A real deployment could not compute them centrally.

## 3. Client partitions (seed 42)

**IID.** Each class is shuffled and dealt out in near-equal chunks, so every client sees all 34 classes in nearly the same proportions.

**Non-IID.** Label-wise Dirichlet split with concentration alpha = 0.5. For every class, the share each client receives is drawn from a Dirichlet distribution. The draw is repeated until every client holds at least 1,000 rows.

Both methods assign every row to exactly one client. This is checked in code and in the tests.

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

Full class-by-client counts: `reports/client_class_counts_iid.csv` and `reports/client_class_counts_noniid.csv`. Summary: `reports/client_partition_summary.csv`.

## 4. Model and training

- MLP: 39 -> 128 -> 64 -> 34, ReLU, raw logits out. Shape test: 5 input rows give an output of shape `(5, 34)`.
- Local training: Adam, learning rate 1e-3, 1 local epoch, batch size 256, cross-entropy loss without class weights.
- FedAvg: Flower's built-in strategy. All 5 clients take part in every round, and each update is weighted by the client's number of examples.
- Device: CPU only. One thread per client.
- Seeds: partitioning, validation split, model initialization and local shuffling are all seeded from `--seed` (default 42).
- Evaluation: after each round the server scores the global model on the validation hold-out. Reported: loss, accuracy, balanced accuracy, macro-F1, weighted-F1 and per-class precision, recall and F1.

## 5. Smoke-test results (2 rounds, 5 clients)

Both runs completed with `Received 5 results and 0 failures` in every round and finite loss throughout. Scores are on the validation hold-out from `train.csv`, not on `test.csv`.

| Partition | Round | Mean train loss | Val accuracy | Val macro-F1 | Val loss |
|---|---|---|---|---|---|
| (untrained) | 0 | n/a | 0.0011 | 0.0014 | 3.529 |
| IID | 1 | 0.660 | 0.7449 | 0.4370 | 0.503 |
| IID | 2 | 0.498 | 0.7504 | 0.4722 | 0.486 |
| Non-IID | 1 | 0.493 | 0.7121 | 0.3032 | 0.599 |
| Non-IID | 2 | 0.353 | 0.7305 | 0.3956 | 0.518 |

Run time: the IID run took 25.9 s and the non-IID run 28.4 s (wall-clock, including Ray start-up).

How to read this:
- The non-IID global model scores lower than the IID one, mainly on macro-F1 (0.396 against 0.472 after round 2).
- Train loss is lower for non-IID because each client trains on a narrow class mix, which is easier locally. It does not mean the global model is better.
- This is one seed, two rounds and one hold-out, so the gap between IID and non-IID is a first indication only.
- These numbers are not comparable to the Random Forest baseline (76.05% accuracy, 57.11% macro-F1). That baseline is a different model scored on the held-out test split.

Reports: `reports/fedavg_smoke_test_<iid|noniid>.json` (settings, versions, per-client updates, per-round validation), the matching `.csv` (one row per client per round) and `reports/fedavg_per_class_val_<iid|noniid>.json` (per-class precision, recall, F1).

## 6. Environment used

| Item | Version |
|---|---|
| Python | 3.14.3 |
| PyTorch | 2.14.1 (CPU) |
| Flower | 1.40.0 |
| Ray | 2.59.0 |
| OS | Windows |

`requirements.txt` now lists `flwr[simulation]` and `ray`. Without Ray, the simulation stops with Flower exit code 701.

## 7. How to reproduce

Run from the repository root with the virtual environment active.

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# place train.csv at output\splits\train.csv, then:
python tests/test_federated.py
python -m federated.partition_data
mkdir logs
python -m federated.run_simulation --partition iid --rounds 2 2>&1 | Tee-Object -FilePath logs\fedavg_smoke_test_iid.log
python -m federated.run_simulation --partition noniid --rounds 2 2>&1 | Tee-Object -FilePath logs\fedavg_smoke_test_noniid.log
```

Options for `run_simulation`: `--rounds` (1 to 3), `--local-epochs`, `--batch-size`, `--lr`, `--seed`, `--alpha`. Options for `partition_data`: `--seed`, `--alpha`, `--min-size`, `--val-fraction`, `--force` (rebuild the cache).

The tests cover: schema checks, a stable label map, every row assigned to exactly one client, seeded reproducibility, non-IID being more skewed than IID, the model output shape, and finite loss with changing weights after local training.

## 8. Files not in Git

`output/splits/train.csv` and `test.csv` (datasets), `output/fl_cache/` (processed arrays, over 1 GB) and `logs/` (console logs). They are listed in `.gitignore` or kept local. Obtain the datasets from the project owner.

## 9. Known limitations

- Scaling statistics are pooled over all clients, and features are clipped to [-10, 10].
- Cross-entropy has no class weights. The smallest class has 33 rows, so rare classes score poorly after two rounds, and some non-IID clients have no examples of one or two classes.
- Two rounds is a pipeline check, not a converged result.
- Flower 1.40 marks `flwr.simulation.run_simulation` as deprecated. It works, but the newer `flwr run` project layout would be needed long term.
- Flower describes Ray on Windows as experimental. Runs completed, but the console shows Ray worker shutdown errors after training finishes. Flower recommends WSL2.
- No resource, energy or network values are measured or simulated in this milestone.
- No held-out test evaluation has been run. The owner coordinates that.
- No centralized MLP with the same architecture has been trained yet, so there is no like-for-like centralized comparison.

## 10. Next steps

1. Train a centralized MLP with the same architecture, scaling and validation split.
2. Run more rounds and several seeds, and try other alpha values for the non-IID split.
3. Consider class weighting or other imbalance handling.
4. Evaluate on `test.csv` once the owner coordinates it.
5. Add context-aware client selection after the FedAvg baseline is accepted.
