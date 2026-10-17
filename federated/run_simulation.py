"""Run a short Flower FedAvg smoke test (1-3 rounds, 5 simulated clients, CPU only).

Step 1:  python -m federated.partition_data
Step 2:  python -m federated.run_simulation --partition iid --rounds 2
         python -m federated.run_simulation --partition noniid --rounds 2
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import pandas as pd

from federated import data_utils as du
from federated.fl_common import CLIENT_LOG, SERVER_LOG, RunConfig, save_run_config


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", choices=["iid", "noniid"], default="iid")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.5)
    a = ap.parse_args()
    if not 1 <= a.rounds <= 3:
        sys.exit("Smoke test is limited to 1-3 rounds.")

    from federated.partition_data import partition_path
    if not partition_path(du.CACHE_DIR, a.partition, a.seed).exists():
        sys.exit("Partition files missing. Run first:  python -m federated.partition_data")

    cfg = RunConfig(a.partition, a.rounds, a.clients, a.local_epochs, a.batch_size, a.lr, a.seed, a.alpha)
    save_run_config(cfg)
    for p in (CLIENT_LOG, SERVER_LOG):
        p.unlink(missing_ok=True)

    # Ray worker processes must be able to import the `federated` package.
    os.environ["PYTHONPATH"] = str(du.REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

    import flwr
    import torch
    from flwr.simulation import run_simulation
    from federated.fl_client_simulation import app as client_app
    from federated.fl_server import app as server_app

    print(f"Python {platform.python_version()} | torch {torch.__version__} | flwr {flwr.__version__}")
    print(f"Config: {cfg}")
    t0 = time.perf_counter()
    run_simulation(
        server_app=server_app, client_app=client_app, num_supernodes=cfg.num_clients,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
    )
    total = time.perf_counter() - t0

    clients, server = read_jsonl(CLIENT_LOG), read_jsonl(SERVER_LOG)
    rounds_done = sorted({c["round"] for c in clients})
    rows = []
    for c in clients:
        s = next((x for x in server if x["round"] == c["round"]), {})
        rows.append({**{k: c[k] for k in ("round", "client_id", "num_examples", "train_loss", "train_seconds")},
                     "partition": cfg.partition, "global_val_accuracy": s.get("val_accuracy"),
                     "global_val_macro_f1": s.get("val_macro_f1")})
    du.REPORTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).sort_values(["round", "client_id"]).to_csv(du.REPORTS_DIR / f"fedavg_smoke_test_{cfg.partition}.csv", index=False)
    summary = {
        "command": "python -m federated.run_simulation " + " ".join(sys.argv[1:]),
        "config": cfg.__dict__,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "flwr": flwr.__version__},
        "device": "cpu", "total_wall_clock_seconds": round(total, 2),
        "rounds_completed": rounds_done,
        "participating_clients_per_round": {r: sorted(c["client_id"] for c in clients if c["round"] == r) for r in rounds_done},
        "client_updates": clients,
        "server_validation_by_round": [{k: v for k, v in s.items() if k != "per_class"} for s in server],
        "validation_note": "Validation rows are a stratified 10% hold-out from train.csv. test.csv was not used.",
    }
    (du.REPORTS_DIR / f"fedavg_smoke_test_{cfg.partition}.json").write_text(json.dumps(summary, indent=2))
    last = server[-1] if server else {}
    (du.REPORTS_DIR / f"fedavg_per_class_val_{cfg.partition}.json").write_text(json.dumps(last.get("per_class", {}), indent=2))
    print(f"\nRounds completed: {rounds_done} | wall-clock {total:.1f}s")
    for s in server:
        print(f"  round {s['round']}: val_acc={s['val_accuracy']:.4f} val_macro_f1={s['val_macro_f1']:.4f} val_loss={s['val_loss']:.4f}")
    print(f"Saved reports/fedavg_smoke_test_{cfg.partition}.json/.csv")


if __name__ == "__main__":
    main()
