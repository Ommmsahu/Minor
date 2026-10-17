"""Flower ClientApp. Each simulated client trains the shared MLP on its own partition only."""
from __future__ import annotations

import time

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from federated import data_utils as du
from federated.fl_common import CLIENT_LOG, append_jsonl, load_run_config
from federated.fl_model import MLP, train_local
from federated.partition_data import load_partition

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    cfg = load_run_config()
    client_id = int(context.node_config["partition-id"])
    # Flower 1.40 gives no round number in message metadata. Every client trains exactly once
    # per round (fraction_train=1.0), so a per-client call counter in context.state equals the round.
    prev = int(context.state["calls"]["n"]) if "calls" in context.state else 0
    server_round = prev + 1
    context.state["calls"] = ConfigRecord({"n": server_round})
    torch.set_num_threads(1)

    X_all, y_all = du.load_cache_arrays(du.CACHE_DIR, mmap=True)
    rows = load_partition(du.CACHE_DIR, cfg.partition, cfg.seed, client_id)
    X, y = np.asarray(X_all[rows]), y_all[rows]  # this client's rows only

    model = MLP()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    t0 = time.perf_counter()
    loss = train_local(model, X, y, cfg.local_epochs, cfg.batch_size, cfg.lr, seed=cfg.seed + 1000 * server_round + client_id)
    elapsed = time.perf_counter() - t0

    append_jsonl(CLIENT_LOG, {
        "round": server_round, "client_id": client_id, "partition": cfg.partition,
        "num_examples": int(len(y)), "train_loss": float(loss), "train_seconds": round(elapsed, 3),
    })
    reply = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({"train_loss": float(loss), "num-examples": int(len(y))}),
    })
    return Message(content=reply, reply_to=msg)
