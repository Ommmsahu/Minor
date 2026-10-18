"""Run settings, run isolation and logging helpers shared by the runner, ServerApp and ClientApps.

Every run gets a unique run ID and its own folder under logs/runs/<run_id>/, so runs cannot
overwrite or mix client logs, server logs or the run configuration. The Flower apps find
their run folder through the GREENFED_RUN_DIR environment variable set by run_simulation.py.
"""
from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from federated import data_utils as du
from federated.io_utils import atomic_write_json

ENV_RUN_DIR = "GREENFED_RUN_DIR"
RUNS_LOG_ROOT = du.LOGS_DIR / "runs"
RUNS_REPORT_ROOT = du.REPORTS_DIR / "runs"


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
    run_id: str = ""

    def validate(self) -> None:
        """Reject invalid settings before any simulation starts."""
        problems = []

        def is_int(v):
            return isinstance(v, int) and not isinstance(v, bool)

        if self.partition not in ("iid", "noniid"):
            problems.append(f"partition must be 'iid' or 'noniid', got {self.partition!r}")
        if not is_int(self.rounds) or not 1 <= self.rounds <= 3:
            problems.append(f"rounds must be an integer from 1 to 3 for the smoke test, got {self.rounds!r}")
        if not is_int(self.num_clients) or self.num_clients < 1:
            problems.append(f"num_clients must be a positive integer, got {self.num_clients!r}")
        if not is_int(self.local_epochs) or self.local_epochs < 1:
            problems.append(f"local_epochs must be a positive integer, got {self.local_epochs!r}")
        if not is_int(self.batch_size) or self.batch_size < 1:
            problems.append(f"batch_size must be a positive integer, got {self.batch_size!r}")
        if not isinstance(self.lr, (int, float)) or not math.isfinite(self.lr) or self.lr <= 0:
            problems.append(f"lr must be finite and > 0, got {self.lr!r}")
        if not isinstance(self.alpha, (int, float)) or not math.isfinite(self.alpha) or self.alpha <= 0:
            problems.append(f"alpha must be finite and > 0, got {self.alpha!r}")
        if not is_int(self.seed) or self.seed < 0:
            problems.append(f"seed must be a non-negative integer, got {self.seed!r}")
        if problems:
            raise ValueError("Invalid run configuration:\n  - " + "\n  - ".join(problems))


def new_run_id(partition: str, seed: int) -> str:
    return f"{partition}_seed{seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"


def run_log_dir(run_id: str) -> Path:
    return RUNS_LOG_ROOT / run_id


def run_report_dir(run_id: str) -> Path:
    return RUNS_REPORT_ROOT / run_id


def current_run_dir() -> Path:
    value = os.environ.get(ENV_RUN_DIR)
    if not value:
        raise RuntimeError(f"{ENV_RUN_DIR} is not set. Start runs with: python -m federated.run_simulation")
    return Path(value)


def save_run_config(cfg: RunConfig, run_dir: Path) -> None:
    atomic_write_json(Path(run_dir) / "run_config.json", asdict(cfg))


def load_run_config() -> RunConfig:
    return RunConfig(**json.loads((current_run_dir() / "run_config.json").read_text()))


def client_log_path(run_dir: Path) -> Path:
    return Path(run_dir) / "client_updates.jsonl"


def server_log_path(run_dir: Path) -> Path:
    return Path(run_dir) / "server_rounds.jsonl"


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
