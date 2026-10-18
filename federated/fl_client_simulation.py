"""Flower ClientApp. Each simulated client trains the shared MLP on its own partition only."""
from __future__ import annotations

import json
import math
import time

import numpy as np
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from federated import data_utils as du
from federated.fl_common import append_jsonl, client_log_path, current_run_dir, load_run_config
from federated.fl_model import MLP, check_state_shapes, train_local
from federated.partition_data import load_partition, partition_meta_path

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    cfg = load_run_config()
    client_id = int(context.node_config["partition-id"])
    n_parts = context.node_config.get("num-partitions")
    if not 0 <= client_id < cfg.num_clients or (n_parts is not None and int(n_parts) != cfg.num_clients):
        raise ValueError(f"Client {client_id} / {n_parts} partitions does not match the configured {cfg.num_clients} clients.")

    # FedAvg injects the current round into the train config as 'server-round'.
    config = msg.content["config"]
    if "server-round" not in config:
        raise KeyError("Server did not supply 'server-round' in the train config.")
    server_round = int(config["server-round"])
    if not 1 <= server_round <= cfg.rounds:
        raise ValueError(f"Round {server_round} is outside 1..{cfg.rounds}.")
    torch.set_num_threads(1)

    pmeta = json.loads(partition_meta_path(du.CACHE_DIR, cfg.partition, cfg.seed).read_text())
    rows = load_partition(du.CACHE_DIR, cfg.partition, cfg.seed, client_id)
    if len(rows) != pmeta["client_sizes"][client_id]:
        raise ValueError(f"Client {client_id}: partition has {len(rows)} rows but its metadata says {pmeta['client_sizes'][client_id]}.")
    X_all, y_all = du.load_cache_arrays(du.CACHE_DIR, mmap=True)
    X, y = np.asarray(X_all[rows]), y_all[rows]  # this client's rows only

    model = MLP()
    incoming = msg.content["arrays"].to_torch_state_dict()
    check_state_shapes(incoming, model.state_dict(), f"client {client_id} incoming")
    model.load_state_dict(incoming)

    t0 = time.perf_counter()
    loss = train_local(model, X, y, cfg.local_epochs, cfg.batch_size, cfg.lr, seed=cfg.seed + 1000 * server_round + client_id)
    elapsed = time.perf_counter() - t0
    if not (math.isfinite(loss) and math.isfinite(elapsed)):
        raise FloatingPointError(f"Client {client_id} round {server_round}: non-finite loss or time.")
    outgoing = model.state_dict()
    check_state_shapes(outgoing, MLP().state_dict(), f"client {client_id} outgoing")
    n = int(len(y))
    if n != len(rows) or n <= 0:
        raise ValueError(f"Client {client_id}: trained on {n} rows but owns {len(rows)}.")

    append_jsonl(client_log_path(current_run_dir()), {
        "run_id": cfg.run_id, "round": server_round, "client_id": client_id, "partition": cfg.partition,
        "num_examples": n, "train_loss": float(loss), "train_seconds": round(elapsed, 3),
    })
    reply = RecordDict({
        "arrays": ArrayRecord(outgoing),
        "metrics": MetricRecord({"train_loss": float(loss), "num-examples": n}),  # FedAvg weights by 'num-examples'
    })
    return Message(content=reply, reply_to=msg)
