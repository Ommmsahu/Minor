"""Run a short Flower FedAvg smoke test (1-3 rounds, simulated clients, CPU only).

Step 1:  python -m federated.partition_data
Step 2:  python -m federated.run_simulation --partition iid --rounds 2 --seed 42
         python -m federated.run_simulation --partition noniid --rounds 2 --seed 42 --alpha 0.5

Exit codes: 0 verified and clean | 1 training integrity check failed | 2 invalid configuration or inputs
            3 FedAvg rounds verified, but the runtime check found Ray errors or leftover processes
Each run writes logs to logs/runs/<run_id>/ and reports to reports/runs/<run_id>/.
Existing reports in reports/ are never overwritten.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import platform
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from federated import data_utils as du
from federated.fl_common import (ENV_RUN_DIR, RunConfig, client_log_path, new_run_id, read_jsonl, run_log_dir,
                                 run_report_dir, save_run_config, server_log_path)
from federated.io_utils import atomic_write_json, atomic_write_text
from federated.partition_data import PartitionError, verify_partition_files
from federated.run_checks import check_run_integrity, select_final_record

RUNTIME_ERROR_PATTERNS = [r"access violation", r"Windows fatal exception", r"Segmentation fault", r"Fatal Python error"]
RAY_PROCESS_NAMES = {"raylet", "raylet.exe", "gcs_server", "gcs_server.exe"}


class _Tee:
    """Copy everything written to stdout/stderr into the run's console.log."""

    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, s):
        self._stream.write(s)
        try:
            self._fh.write(s)
        except ValueError:
            pass
        return len(s)

    def flush(self):
        self._stream.flush()
        try:
            self._fh.flush()
        except ValueError:
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


def ray_pids() -> set[int]:
    try:
        import psutil
    except ImportError:
        return set()
    found = set()
    for p in psutil.process_iter(["name"]):
        try:
            if (p.info["name"] or "").lower() in RAY_PROCESS_NAMES:
                found.add(p.pid)
        except Exception:
            continue
    return found


def runtime_check(console_log: Path, pids_before: set[int], settle_seconds: float = 3.0) -> dict:
    """Do not hide Ray errors: report fatal messages in the console log and Ray processes left running."""
    text = console_log.read_text(encoding="utf-8", errors="replace") if console_log.exists() else ""
    hits = {pat: len(re.findall(pat, text, flags=re.IGNORECASE)) for pat in RUNTIME_ERROR_PATTERNS}
    hits = {k: v for k, v in hits.items() if v}
    time.sleep(settle_seconds)
    try:
        import psutil
        leftover = sorted(pid for pid in ray_pids() - pids_before if psutil.pid_exists(pid))
        checked = True
    except ImportError:
        leftover, checked = [], False
    clean = not hits and not leftover and checked
    return {"status": "clean" if clean else "problems_detected", "fatal_messages_in_console_log": hits,
            "leftover_ray_processes": leftover, "process_check_available": checked}


def versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("torch", "flwr", "ray", "numpy", "pandas", "scikit-learn"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", choices=["iid", "noniid"], default="iid")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--run-id", default="", help="optional label; a unique suffix is always added")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    cfg = RunConfig(a.partition, a.rounds, a.clients, a.local_epochs, a.batch_size, a.lr, a.seed, a.alpha)
    try:
        cfg.validate()
        cache_meta = du.verify_cache(du.CACHE_DIR)
        pmeta = verify_partition_files(du.CACHE_DIR, cfg.partition, cfg.seed, cfg.num_clients,
                                       cfg.alpha if cfg.partition == "noniid" else None, cache_meta)
    except (ValueError, PartitionError, FileNotFoundError) as exc:
        print(f"CANNOT START: {exc}", file=sys.stderr)
        return 2

    cfg.run_id = (a.run_id + "_" if a.run_id else "") + new_run_id(cfg.partition, cfg.seed)
    log_dir, rep_dir = run_log_dir(cfg.run_id), run_report_dir(cfg.run_id)
    log_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(cfg, log_dir)

    import os
    os.environ[ENV_RUN_DIR] = str(log_dir)
    os.environ["PYTHONPATH"] = str(du.REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")  # Ray workers must import `federated`

    console_log = log_dir / "console.log"
    fh = open(console_log, "w", encoding="utf-8", errors="replace")
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(real_out, fh), _Tee(real_err, fh)  # installed before Flower/Ray are imported
    try:
        return _run(cfg, pmeta, cache_meta, a, log_dir, rep_dir, console_log)
    finally:
        sys.stdout, sys.stderr = real_out, real_err
        fh.close()


def _run(cfg: RunConfig, pmeta: dict, cache_meta: dict, a, log_dir: Path, rep_dir: Path, console_log: Path) -> int:
    ver = versions()
    print(f"Run ID: {cfg.run_id}")
    print(f"Versions: {ver}")
    print(f"Config: {cfg}")
    pids_before = ray_pids()
    started = datetime.now().isoformat(timespec="seconds")
    t0 = time.perf_counter()
    flower_error = None
    try:
        from flwr.simulation import run_simulation
        from federated.fl_client_simulation import app as client_app
        from federated.fl_server import app as server_app
        run_simulation(server_app=server_app, client_app=client_app, num_supernodes=cfg.num_clients,
                       backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}})
    except BaseException as exc:  # report any failure, including Ctrl+C
        flower_error = f"{type(exc).__name__}: {exc}"
    total = time.perf_counter() - t0
    finished = datetime.now().isoformat(timespec="seconds")

    clients = read_jsonl(client_log_path(log_dir))
    server = read_jsonl(server_log_path(log_dir))
    errors = check_run_integrity(cfg, clients, server, expected_sizes=pmeta["client_sizes"])
    if flower_error:
        errors.insert(0, f"Flower raised: {flower_error}")
    runtime = runtime_check(console_log, pids_before)

    base = {
        "run_id": cfg.run_id, "command": "python -m federated.run_simulation " + " ".join(sys.argv[1:]),
        "config": cfg.__dict__, "versions": ver, "device": "cpu", "started_at": started, "finished_at": finished,
        "total_wall_clock_seconds": round(total, 2), "cache_id": cache_meta["cache_id"], "partition_meta": pmeta,
        "isolation_note": "Validation rows are a stratified 10% hold-out of train.csv. test.csv was not opened.",
        "console_log": str(console_log), "runtime_check": runtime,
    }
    if errors:
        base.update({"status": "FAILED", "errors": errors, "client_updates": clients,
                     "server_validation_by_round": [{k: v for k, v in s.items() if k != "per_class"} for s in server]})
        atomic_write_json(rep_dir / f"fedavg_smoke_test_{cfg.partition}_FAILED.json", base)
        print("\nRUN FAILED: the simulation is incomplete or inconsistent. No success report was written.", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        print(f"Failure details: {rep_dir / ('fedavg_smoke_test_' + cfg.partition + '_FAILED.json')}", file=sys.stderr)
        return 1

    final = select_final_record(server)
    rounds_done = sorted({c["round"] for c in clients})
    rows = []
    for c in sorted(clients, key=lambda r: (r["round"], r["client_id"])):
        s = next(x for x in server if x["round"] == c["round"])
        rows.append({"run_id": cfg.run_id, "partition": cfg.partition, "round": c["round"], "client_id": c["client_id"],
                     "num_examples": c["num_examples"], "train_loss": c["train_loss"], "train_seconds": c["train_seconds"],
                     "global_val_accuracy": s["val_accuracy"], "global_val_macro_f1": s["val_macro_f1"]})
    base.update({
        "status": "COMPLETE", "errors": [], "rounds_completed": rounds_done, "final_round": final["round"],
        "participating_clients_per_round": {r: sorted(c["client_id"] for c in clients if c["round"] == r) for r in rounds_done},
        "client_updates": clients,
        "server_validation_by_round": [{k: v for k, v in s.items() if k != "per_class"} for s in sorted(server, key=lambda r: r["round"])],
    })
    atomic_write_json(rep_dir / f"fedavg_smoke_test_{cfg.partition}.json", base)
    atomic_write_text(rep_dir / f"fedavg_smoke_test_{cfg.partition}.csv", pd.DataFrame(rows).to_csv(index=False))
    atomic_write_json(rep_dir / f"fedavg_per_class_val_{cfg.partition}.json", final["per_class"])

    print(f"\nTRAINING VERIFIED: rounds {rounds_done}, {cfg.num_clients} clients per round, wall-clock {total:.1f}s")
    for s in sorted(server, key=lambda r: r["round"]):
        print(f"  round {s['round']}: val_acc={s['val_accuracy']:.4f} val_macro_f1={s['val_macro_f1']:.4f} val_loss={s['val_loss']:.4f}")
    print(f"Reports: {rep_dir}")
    if runtime["status"] != "clean":
        print(f"RUNTIME CHECK FAILED (training results are intact): {runtime}", file=sys.stderr)
        return 3
    print("Runtime check: clean (no Ray fatal messages, no leftover Ray processes).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
