# Federated Learning Task — GreenFed-IDS

## Objective

Build the first federated-learning pipeline for GreenFed-IDS using simulated IoT clients and our existing CICIoT2023 training split.

The first milestone is standard Federated Averaging (FedAvg). Context-aware client selection will be implemented later, after the basic federated-learning pipeline works.

## 1. Required input files

Obtain these files from the project owner:

- `output/splits/train.csv` — use for client partitioning and training.
- `output/splits/test.csv` — keep separate for evaluation only.

The existing experiment contains approximately 1,170,022 training rows, 290,874 test rows, 39 input features, and 34 classes. Verify these details against the files you receive.

The CSV files are not stored in this GitHub repository because of their size.

## 2. Validate the data

- Confirm that the target column is `Label`.
- Verify the 39 input features and their order.
- Check missing and non-finite values.
- Create one consistent mapping between all 34 class labels and numeric indices.
- Fit learned preprocessing using training data only.

## 3. Simulate five IoT clients

Implement two data partitioning methods:

**IID:** Clients receive approximately similar class distributions.

**Non-IID:** Clients receive different class distributions, simulating devices that observe different network traffic. A Dirichlet-based partition is one possible method.

Use a fixed random seed and report the actual sample count and class distribution for every client.

## 4. Build the shared PyTorch MLP

Start with a feed-forward neural network with:

- Input dimension: 39
- Hidden layers: 128 and 64 units
- ReLU activation
- Output dimension: 34 classes

Test that five input samples produce an output tensor of shape `(5, 34)`.

## 5. Implement Flower and FedAvg

- Simulate five clients.
- Train the same model architecture locally on each client.
- Aggregate model parameters using FedAvg.
- Start with one to three rounds to verify the pipeline.
- Record training loss, runtime, and participating clients.

Do not implement context-aware client selection in this first milestone.

## 6. Evaluation rules

- Never use the held-out test data for client training.
- Use a validation subset derived from training data for model selection.
- Report accuracy, balanced accuracy, macro-F1, weighted-F1, and per-class metrics when evaluation is implemented.
- Record random seeds and experiment settings.
- Clearly distinguish simulated resource measurements from actual measurements.

## 7. Suggested project files

- `partition_data.py`
- `fl_model.py`
- `fl_client_simulation.py`
- `fl_server.py`
- `reports/`

You may adjust filenames to fit the chosen Flower API, but document the run commands.

## 8. Deliverables

1. Working IID and Non-IID partitioning code.
2. Client sample-count and class-distribution reports.
3. Shared PyTorch MLP and shape-test result.
4. Working short Flower FedAvg run.
5. README with setup and execution instructions.

Do not modify the original cleaned CSVs, the existing train/test split, or the centralized Random Forest baseline.