"""Run settings shared between the runner, ServerApp and ClientApps (via a small JSON file)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from federated import data_utils as du

RUN_CONFIG_PATH = du.CACHE_DIR / "run_config.json"
CLIENT_LOG = du.LOGS_DIR / "client_updates.jsonl"
SERVER_LOG = du.LOGS_DIR / "server_rounds.jsonl"


@dataclass
class RunConfig:
    partition: str = "iid"          # "iid" or "noniid"
    rounds: int = 2
    num_clients: int = 5
    local_epochs: int = 1
    batch_size: int = 256
    lr: float = 1e-3
    seed: int = 42
    alpha: float = 0.5              # Dirichlet concentration, used by the non-IID partition


def save_run_config(cfg: RunConfig) -> None:
    RUN_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_CONFIG_PATH.write_text(json.dumps(asdict(cfg), indent=2))


def load_run_config() -> RunConfig:
    return RunConfig(**json.loads(RUN_CONFIG_PATH.read_text()))


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
