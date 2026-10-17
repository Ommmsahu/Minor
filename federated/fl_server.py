"""Flower ServerApp: standard FedAvg, weighted by each client's number of examples."""
from __future__ import annotations

import time

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from federated import data_utils as du
from federated.fl_common import SERVER_LOG, append_jsonl, load_run_config
from federated.fl_model import MLP, evaluate_model

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = load_run_config()
    schema = du.load_schema_config()
    X_val, y_val = du.load_val_arrays(du.CACHE_DIR)  # validation rows come from train.csv, never from test.csv
    label_names = [n for n, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]

    torch.manual_seed(cfg.seed)
    initial = ArrayRecord(MLP().state_dict())
    clock = {"last": time.perf_counter()}

    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        now = time.perf_counter()
        model = MLP()
        model.load_state_dict(arrays.to_torch_state_dict())
        m = evaluate_model(model, X_val, y_val)
        append_jsonl(SERVER_LOG, {
            "round": server_round, "seconds_since_previous_call": round(now - clock["last"], 3),
            "val_loss": m["loss"], "val_accuracy": m["accuracy"], "val_balanced_accuracy": m["balanced_accuracy"],
            "val_macro_f1": m["macro_f1"], "val_weighted_f1": m["weighted_f1"],
            "per_class": {n: {k: m["per_class"][k][i] for k in m["per_class"]} for i, n in enumerate(label_names)},
        })
        clock["last"] = time.perf_counter()
        return MetricRecord({"val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"], "val_loss": m["loss"]})

    strategy = FedAvg(
        fraction_train=1.0, fraction_evaluate=0.0,
        min_train_nodes=cfg.num_clients, min_available_nodes=cfg.num_clients,
    )
    strategy.start(
        grid=grid, initial_arrays=initial, num_rounds=cfg.rounds,
        train_config=ConfigRecord({}), evaluate_fn=evaluate_fn,
    )
