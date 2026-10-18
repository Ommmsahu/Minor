"""Flower ServerApp: standard FedAvg, weighted by each client's number of examples."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from federated import data_utils as du
from federated.fl_common import append_jsonl, current_run_dir, load_run_config, server_log_path
from federated.fl_model import MLP, N_CLASSES, N_FEATURES, evaluate_model
from federated.run_checks import VAL_KEYS

app = ServerApp()
TIMING_NOTE = "server_callback_interval_s is the wall-clock gap between global-validation callbacks (it includes client training waiting time and validation). It is not client training time and not energy use."


def check_server_inputs(model: torch.nn.Module, X_val: np.ndarray, y_val: np.ndarray, label_map: dict) -> None:
    """Output size, label map and validation labels must all agree with the 34-class schema."""
    out_dim = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][-1].out_features
    if not (out_dim == len(label_map) == N_CLASSES):
        raise ValueError(f"Class count mismatch: model outputs {out_dim}, label map has {len(label_map)}, expected {N_CLASSES}.")
    if sorted(label_map.values()) != list(range(N_CLASSES)):
        raise ValueError("Label map values are not exactly 0..33.")
    if X_val.ndim != 2 or X_val.shape[1] != N_FEATURES:
        raise ValueError(f"Validation features have shape {X_val.shape}; expected (n, {N_FEATURES}).")
    if y_val.min() < 0 or y_val.max() >= N_CLASSES:
        raise ValueError("Validation labels fall outside the 34-class range.")
    if len(X_val) != len(y_val):
        raise ValueError("Validation features and labels differ in length.")


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = load_run_config()
    log_path = server_log_path(current_run_dir())
    schema = du.load_schema_config()
    cache_id = du.load_cache_meta(du.CACHE_DIR)["cache_id"]
    X_val, y_val = du.load_val_arrays(du.CACHE_DIR)  # validation rows come from train.csv, never from test.csv
    label_names = [n for n, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]

    torch.manual_seed(cfg.seed)
    model0 = MLP()
    check_server_inputs(model0, X_val, y_val, schema["label_map"])
    initial = ArrayRecord(model0.state_dict())
    clock = {"last": time.perf_counter()}

    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        now = time.perf_counter()
        model = MLP()
        model.load_state_dict(arrays.to_torch_state_dict())
        m = evaluate_model(model, X_val, y_val)
        values = {"val_loss": m["loss"], "val_accuracy": m["accuracy"], "val_balanced_accuracy": m["balanced_accuracy"],
                  "val_macro_f1": m["macro_f1"], "val_weighted_f1": m["weighted_f1"]}
        if not all(math.isfinite(values[k]) for k in VAL_KEYS):
            raise FloatingPointError(f"Non-finite validation metrics in round {server_round}: {values}")
        append_jsonl(log_path, {
            "run_id": cfg.run_id, "round": server_round, "cache_id": cache_id,
            "server_callback_interval_s": round(now - clock["last"], 3), "timing_note": TIMING_NOTE,
            **values,
            "per_class": {n: {k: m["per_class"][k][i] for k in m["per_class"]} for i, n in enumerate(label_names)},
        })
        clock["last"] = time.perf_counter()
        return MetricRecord({"val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"], "val_loss": m["loss"]})

    strategy = FedAvg(
        fraction_train=1.0, fraction_evaluate=0.0,  # client-side evaluation disabled for this smoke test
        min_train_nodes=cfg.num_clients, min_available_nodes=cfg.num_clients,
    )  # weighted by 'num-examples' (Flower default)
    strategy.start(
        grid=grid, initial_arrays=initial, num_rounds=cfg.rounds,
        train_config=ConfigRecord({}), evaluate_fn=evaluate_fn,
    )
