"""Tests for the GreenFed-IDS federated baseline.

Run from the repo root:  python tests/test_federated.py     (or: python -m pytest tests)
Set RUN_E2E=1 to also run the slow end-to-end smoke test (needs the real cache and partitions).
"""
import builtins
import json
import math
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from federated import data_utils as du
from federated import fl_common
from federated.fl_common import RunConfig
from federated.fl_model import MLP, check_state_shapes, evaluate_model, train_local
from federated.partition_data import (PartitionError, check_partition, class_count_matrix, dirichlet_partition,
                                      iid_partition, save_partition, verify_partition_files)
from federated.run_checks import check_run_integrity, select_final_record

# ----------------------------------------------------------------------------- helpers


def synthetic_labels(n_classes=34, seed=0):
    rng = np.random.default_rng(seed)
    sizes = np.maximum((30000 * np.geomspace(1, 0.001, n_classes)).astype(int), 33)
    return rng.permutation(np.concatenate([[i] * s for i, s in enumerate(sizes)]))


def synthetic_frame(n=200):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n, 39)), columns=du.TRUSTED_FEATURES)
    df["Label"] = [f"C{i % 34}" for i in range(n)]
    return df


def make_csv(path, per_class=30, seed=0):
    rng = np.random.default_rng(seed)
    n = 34 * per_class
    df = pd.DataFrame(rng.normal(size=(n, 39)).astype("float32"), columns=du.TRUSTED_FEATURES)
    df["Label"] = [f"C{i % 34:02d}" for i in range(n)]
    df.sample(frac=1, random_state=seed).reset_index(drop=True).to_csv(path, index=False)
    return path


@contextmanager
def workspace(per_class=30, build=True, seed=1):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        csv = make_csv(d / "train.csv", per_class)
        cache, rep = d / "cache", d / "reports"
        if build:
            du.prepare_cache(csv, cache, seed=seed, val_fraction=0.1, reports_dir=rep)
        yield d, csv, cache, rep


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return str(e)
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc.__name__}")


# ----------------------------------------------------------------------------- original six tests


def test_schema_ok_and_errors():
    assert len(du.validate_frame(synthetic_frame())) == 39
    bad = synthetic_frame(); bad.iloc[3, 5] = np.nan
    for broken in (bad, synthetic_frame().drop(columns=[du.TRUSTED_FEATURES[0]]), synthetic_frame().drop(columns=["Label"])):
        try:
            du.validate_frame(broken)
        except ValueError:
            continue
        raise AssertionError("validate_frame accepted a broken frame")


def test_label_map_is_stable():
    a = du.build_label_map(synthetic_frame()["Label"])
    b = du.build_label_map(synthetic_frame().sample(frac=1, random_state=3)["Label"])
    assert a == b and len(a) == 34 and sorted(a.values()) == list(range(34))


def test_partitions_cover_every_row_once_and_reproduce():
    y = synthetic_labels()
    for fn in (iid_partition, dirichlet_partition):
        p1, p2 = fn(y, 5, seed=7), fn(y, 5, seed=7)
        check_partition(p1, len(y))
        assert all(np.array_equal(a, b) for a, b in zip(p1, p2))
        assert min(len(p) for p in p1) > 0


def test_noniid_is_more_skewed_than_iid():
    y = synthetic_labels()

    def tv(parts):
        c = class_count_matrix(y, parts, 34).astype(float)
        g = c.sum(1) / c.sum()
        return np.mean([0.5 * np.abs(c[:, k] / c[:, k].sum() - g).sum() for k in range(5)])

    assert tv(dirichlet_partition(y, 5, 0.5, 1)) > 10 * tv(iid_partition(y, 5, 1))


def test_model_shape():
    out = MLP()(torch.randn(5, 39))
    assert out.shape == (5, 34)


def test_local_training_changes_weights_and_loss_is_finite():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(512, 39)).astype(np.float32), rng.integers(0, 34, 512)
    m = MLP(); before = [p.clone() for p in m.parameters()]
    loss = train_local(m, X, y, epochs=1, batch_size=64, lr=1e-3, seed=0)
    assert np.isfinite(loss)
    assert any(not torch.equal(a, b) for a, b in zip(before, m.parameters()))
    assert np.isfinite(evaluate_model(m, X, y)["loss"])


# ----------------------------------------------------------------------------- cache and schema


def test_cache_reuse_and_invalidation():
    with workspace() as (d, csv, cache, rep):
        assert du.cache_is_current(csv, cache, 1, 0.1)[0]
        stamp = (cache / "X_train.npy").stat().st_mtime_ns
        du.prepare_cache(csv, cache, seed=1, val_fraction=0.1, reports_dir=rep)
        assert (cache / "X_train.npy").stat().st_mtime_ns == stamp, "matching cache must be reused"
        assert not du.cache_is_current(csv, cache, 2, 0.1)[0], "seed change must invalidate"
        assert not du.cache_is_current(csv, cache, 1, 0.2)[0], "validation fraction change must invalidate"
        old = du.CLIP_VALUE
        try:
            du.CLIP_VALUE = 5.0
            assert not du.cache_is_current(csv, cache, 1, 0.1)[0], "preprocessing change must invalidate"
        finally:
            du.CLIP_VALUE = old
        df = pd.read_csv(csv)
        df.loc[0, "Rate"] += 1.0
        df.to_csv(csv, index=False)
        assert not du.cache_is_current(csv, cache, 1, 0.1)[0], "changed CSV must invalidate"
        du.prepare_cache(csv, cache, seed=1, val_fraction=0.1, reports_dir=rep)
        assert du.cache_is_current(csv, cache, 1, 0.1)[0]
        assert (cache / "X_train.npy").stat().st_mtime_ns != stamp, "stale cache must be rebuilt"


def test_incomplete_or_corrupt_cache_is_rejected():
    with workspace() as (d, csv, cache, rep):
        meta = du.verify_cache(cache)
        assert meta["cache_id"] and meta["feature_names"] == du.TRUSTED_FEATURES
        np.save(cache / "y_train.npy", np.load(cache / "y_train.npy")[:-1])
        raises(du.CacheError, du.verify_cache, cache)
        assert not du.cache_is_current(csv, cache, 1, 0.1)[0]
    with workspace() as (d, csv, cache, rep):
        (cache / "X_val.npy").unlink()
        raises(du.CacheError, du.verify_cache, cache)
    with workspace() as (d, csv, cache, rep):
        X = np.load(cache / "X_train.npy"); X[0, 0] = np.inf; np.save(cache / "X_train.npy", X)
        raises(du.CacheError, du.verify_cache, cache)


def test_reordered_feature_columns_rejected_or_explicitly_reordered():
    df = synthetic_frame()
    reordered = df[list(reversed(du.TRUSTED_FEATURES)) + ["Label"]]
    assert "order" in raises(ValueError, du.validate_frame, reordered)
    assert du.validate_frame(reordered, allow_reorder=True) == du.TRUSTED_FEATURES
    renamed = df.rename(columns={du.TRUSTED_FEATURES[0]: "Something_else"})
    assert "Missing" in raises(ValueError, du.validate_frame, renamed)
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "a").mkdir()
        (t / "b").mkdir()
        make_csv(t / "a" / "train.csv")
        src = pd.read_csv(t / "a" / "train.csv")
        src[list(reversed(du.TRUSTED_FEATURES)) + ["Label"]].to_csv(t / "b" / "train.csv", index=False)
        raises(ValueError, du.prepare_cache, t / "b" / "train.csv", t / "cb", 1, 0.1, False, t / "rb")
        du.prepare_cache(t / "a" / "train.csv", t / "ca", 1, 0.1, False, t / "ra")
        du.prepare_cache(t / "b" / "train.csv", t / "cb", 1, 0.1, False, t / "rb", allow_reorder=True)
        assert du.load_cache_meta(t / "cb")["columns_reordered_from_source"] is True
        assert np.array_equal(np.load(t / "ca" / "X_train.npy"), np.load(t / "cb" / "X_train.npy")), "explicit reorder must restore trusted order"


def test_bad_feature_values_rejected():
    f0 = du.TRUSTED_FEATURES[0]
    nan = synthetic_frame(); nan.loc[2, f0] = np.nan
    inf = synthetic_frame(); inf.loc[2, f0] = np.inf
    neg = synthetic_frame(); neg.loc[2, f0] = -np.inf
    text = synthetic_frame(); text[f0] = text[f0].astype(str)
    nolabel = synthetic_frame(); nolabel.loc[1, "Label"] = None
    fewclasses = synthetic_frame(); fewclasses["Label"] = "C0"
    for name, frame in (("nan", nan), ("inf", inf), ("-inf", neg), ("text", text), ("null label", nolabel), ("class count", fewclasses)):
        raises(ValueError, du.validate_frame, frame)


def test_validation_comes_only_from_train_csv_and_test_csv_is_never_read():
    seen = []
    real_open, real_read_csv, real_load = builtins.open, pd.read_csv, np.load

    def spy(real):
        def inner(path, *a, **k):
            seen.append(Path(str(path)).name if not hasattr(path, "read") else "<stream>")
            return real(path, *a, **k)
        return inner

    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        csv = make_csv(t / "train.csv")
        (t / "test.csv").write_text("this must never be read\n")
        with mock.patch.object(builtins, "open", spy(real_open)), mock.patch.object(pd, "read_csv", spy(real_read_csv)), mock.patch.object(np, "load", spy(real_load)):
            du.prepare_cache(csv, t / "cache", 1, 0.1, False, t / "rep")
            du.verify_cache(t / "cache")
        assert "test.csv" not in seen, f"test.csv was opened: {seen}"
        assert "train.csv" in seen
        raises(PermissionError, du.prepare_cache, t / "test.csv", t / "c2", 1, 0.1, False, t / "r2")
        raises(PermissionError, du.read_train_csv, t / "test.csv")
        # every validation row is a train.csv row, with the right label
        meta, df = du.load_cache_meta(t / "cache"), pd.read_csv(csv)
        val_idx, yv = np.load(t / "cache" / "val_source_idx.npy"), np.load(t / "cache" / "y_val.npy")
        assert [meta["label_map"][l] for l in df["Label"].iloc[val_idx]] == yv.tolist()
        assert meta["validation_source"].endswith("train.csv only")


# ----------------------------------------------------------------------------- partitions


def test_partition_invariants_detect_corruption():
    y = synthetic_labels(); n = len(y)
    good = iid_partition(y, 5, 3)
    check_partition(good, n, 5)
    dup = [p.copy() for p in good]; dup[0][0] = dup[1][0]
    missing = [p.copy() for p in good]; missing[2] = missing[2][1:]
    out_of_range = [p.copy() for p in good]; out_of_range[3][-1] = n + 5
    for bad in (dup, missing, out_of_range):
        raises(PartitionError, check_partition, bad, n, 5)
    raises(PartitionError, check_partition, good[:4], n, 5)            # wrong client count
    raises(PartitionError, check_partition, good + [np.array([], dtype=int)], n, 6)  # empty client
    raises(PartitionError, check_partition, good, n, 5, min_size=n)    # client below minimum size


def test_partition_files_must_match_cache_and_run_settings():
    with workspace() as (d, csv, cache, rep):
        meta = du.verify_cache(cache)
        _, y = du.load_cache_arrays(cache, mmap=False)
        save_partition(cache, "iid", 1, iid_partition(y, 5, 1), None, 1)
        save_partition(cache, "noniid", 1, dirichlet_partition(y, 5, 0.5, 1, min_size=20), 0.5, 20)
        verify_partition_files(cache, "iid", 1, 5, None, meta)
        verify_partition_files(cache, "noniid", 1, 5, 0.5, meta)
        raises(PartitionError, verify_partition_files, cache, "iid", 1, 4, None, meta)       # client count mismatch
        raises(PartitionError, verify_partition_files, cache, "noniid", 1, 5, 0.3, meta)     # alpha mismatch
        raises(PartitionError, verify_partition_files, cache, "iid", 9, 5, None, meta)       # no partition for that seed
        raises(PartitionError, verify_partition_files, cache, "iid", 1, 5, None, {**meta, "cache_id": "stale"})
        mp = cache / "partition_iid_seed1_meta.json"
        m = json.loads(mp.read_text()); m["client_sizes"][0] += 1; mp.write_text(json.dumps(m))
        raises(PartitionError, verify_partition_files, cache, "iid", 1, 5, None, meta)      # sizes disagree with file


def test_partitions_exclude_validation_rows():
    with workspace() as (d, csv, cache, rep):
        _, y = du.load_cache_arrays(cache, mmap=False)
        parts = iid_partition(y, 5, 1)
        check_partition(parts, len(y), 5)
        train_src, val_src = np.load(cache / "train_source_idx.npy"), np.load(cache / "val_source_idx.npy")
        assert not np.intersect1d(train_src[np.concatenate(parts)], val_src).size
        np.save(cache / "train_source_idx.npy", np.concatenate([train_src[:-1], val_src[:1]]))  # corrupt: overlap
        raises(du.CacheError, du.verify_cache, cache)


# ----------------------------------------------------------------------------- run integrity and isolation


def fake_run(cfg):
    clients = [{"run_id": cfg.run_id, "round": r, "client_id": c, "num_examples": 100 + c, "train_loss": 0.5, "train_seconds": 1.0}
               for r in range(1, cfg.rounds + 1) for c in range(cfg.num_clients)]
    per_class = {f"C{i}": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 3} for i in range(34)}
    server = [{"run_id": cfg.run_id, "round": r, "val_loss": 1.0 / (r + 1), "val_accuracy": 0.5, "val_balanced_accuracy": 0.4,
               "val_macro_f1": 0.3, "val_weighted_f1": 0.5, "per_class": per_class} for r in range(0, cfg.rounds + 1)]
    return clients, server, [100 + c for c in range(cfg.num_clients)]


def test_run_integrity_checks():
    cfg = RunConfig(rounds=2, num_clients=5, run_id="run-a")
    clients, server, sizes = fake_run(cfg)
    assert check_run_integrity(cfg, clients, server, sizes) == []
    missing = [c for c in clients if not (c["round"] == 2 and c["client_id"] == 3)]
    assert any("missing client update: round 2 client 3" in e for e in check_run_integrity(cfg, missing, server, sizes))
    assert any("duplicate client update" in e for e in check_run_integrity(cfg, clients + [clients[0]], server, sizes))
    assert any("missing global validation record for round 1" in e for e in check_run_integrity(cfg, clients, [s for s in server if s["round"] != 1], sizes))
    assert any("duplicate global validation" in e for e in check_run_integrity(cfg, clients, server + [server[0]], sizes))
    nan_server = [dict(s) for s in server]; nan_server[2]["val_macro_f1"] = float("nan")
    assert any("non-finite" in e for e in check_run_integrity(cfg, clients, nan_server, sizes))
    wrong = [dict(c) for c in clients]; wrong[0]["num_examples"] = 7
    assert any("partition has" in e for e in check_run_integrity(cfg, wrong, server, sizes))
    foreign = [dict(c) for c in clients]; foreign[0]["run_id"] = "run-b"
    assert any("another run" in e for e in check_run_integrity(cfg, foreign, server, sizes))
    assert check_run_integrity(cfg, [], [], sizes), "an empty run must never pass"
    assert select_final_record(list(reversed(server)))["round"] == 2, "final record is the highest round, not the last list item"


def test_runs_are_isolated():
    a, b = fl_common.new_run_id("iid", 42), fl_common.new_run_id("iid", 42)
    assert a != b
    assert fl_common.run_log_dir(a) != fl_common.run_log_dir(b) and fl_common.run_report_dir(a) != fl_common.run_report_dir(b)
    with tempfile.TemporaryDirectory() as t:
        da, db = Path(t) / a, Path(t) / b
        fl_common.save_run_config(RunConfig(partition="iid", run_id=a), da)
        fl_common.save_run_config(RunConfig(partition="noniid", alpha=0.3, run_id=b), db)
        fl_common.append_jsonl(fl_common.client_log_path(da), {"x": 1})
        fl_common.append_jsonl(fl_common.client_log_path(db), {"x": 2})
        assert json.loads((da / "run_config.json").read_text())["partition"] == "iid"
        assert json.loads((db / "run_config.json").read_text())["alpha"] == 0.3
        assert fl_common.read_jsonl(fl_common.client_log_path(da)) == [{"x": 1}]
        assert fl_common.read_jsonl(fl_common.client_log_path(db)) == [{"x": 2}]
        with mock.patch.dict(os.environ, {fl_common.ENV_RUN_DIR: str(da)}):
            assert fl_common.load_run_config().run_id == a
        env = {k: v for k, v in os.environ.items() if k != fl_common.ENV_RUN_DIR}
        with mock.patch.dict(os.environ, env, clear=True):
            raises(RuntimeError, fl_common.current_run_dir)


def test_run_config_validation():
    RunConfig().validate()
    RunConfig(partition="noniid", rounds=3, alpha=0.1).validate()
    for bad in ({"rounds": 0}, {"rounds": 4}, {"rounds": 1.5}, {"num_clients": 0}, {"local_epochs": 0}, {"batch_size": 0},
                {"lr": 0.0}, {"lr": float("nan")}, {"lr": float("inf")}, {"alpha": 0.0}, {"alpha": -1.0}, {"alpha": float("inf")},
                {"partition": "other"}, {"seed": -1}, {"rounds": True}):
        raises(ValueError, RunConfig(**bad).validate)


# ----------------------------------------------------------------------------- model, client, server, FedAvg


def test_model_and_client_update_validation():
    ref = MLP().state_dict()
    check_state_shapes(MLP().state_dict(), ref)
    wrong_shape = {k: v.clone() for k, v in ref.items()}; first = next(iter(wrong_shape)); wrong_shape[first] = wrong_shape[first][:-1]
    raises(ValueError, check_state_shapes, wrong_shape, ref)
    renamed = {("x" + k): v for k, v in ref.items()}
    raises(ValueError, check_state_shapes, renamed, ref)
    nan_state = {k: v.clone() for k, v in ref.items()}; nan_state[first][0, 0] = float("nan")
    raises(ValueError, check_state_shapes, nan_state, ref)


def test_server_input_checks():
    from federated.fl_server import check_server_inputs
    label_map = {f"C{i}": i for i in range(34)}
    X, y = np.zeros((10, 39), np.float32), np.arange(10) % 34
    check_server_inputs(MLP(), X, y, label_map)
    raises(ValueError, check_server_inputs, MLP(), X, y, {k: v for k, v in list(label_map.items())[:33]})
    raises(ValueError, check_server_inputs, MLP(), np.zeros((10, 38), np.float32), y, label_map)
    raises(ValueError, check_server_inputs, MLP(), X, y + 40, label_map)
    raises(ValueError, check_server_inputs, MLP(n_classes=33), X, y % 33, label_map)


def test_fedavg_weights_updates_by_num_examples():
    from flwr.app import ArrayRecord, MetricRecord, RecordDict
    from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords

    def reply(value, n):
        return RecordDict({"arrays": ArrayRecord({"w": torch.full((3,), float(value))}), "metrics": MetricRecord({"num-examples": n})})

    out = aggregate_arrayrecords([reply(1.0, 100), reply(5.0, 300)], "num-examples")
    got = out["w"].numpy()
    assert np.allclose(got, (1.0 * 100 + 5.0 * 300) / 400), got      # 4.0, not the unweighted mean 3.0
    assert not np.allclose(got, 3.0)


def test_runtime_check_reports_ray_errors_instead_of_hiding_them():
    from federated.run_simulation import runtime_check
    with tempfile.TemporaryDirectory() as t:
        log = Path(t) / "console.log"
        log.write_text("INFO round done\n(raylet) Windows fatal exception: access violation\n")
        res = runtime_check(log, set(), settle_seconds=0)
        assert res["status"] == "problems_detected" and res["fatal_messages_in_console_log"]
        log.write_text("INFO round done\n")
        assert runtime_check(log, set(), settle_seconds=0)["status"] == "clean"


def test_end_to_end_smoke_test():
    """Slow. Needs a built cache and partitions. Enable with RUN_E2E=1."""
    if os.environ.get("RUN_E2E") != "1":
        print("      (skipped: set RUN_E2E=1 to run)")
        return
    root = Path(__file__).resolve().parents[1]
    for partition in ("iid", "noniid"):
        r = subprocess.run([sys.executable, "-m", "federated.run_simulation", "--partition", partition, "--rounds", "1"],
                           cwd=root, capture_output=True, text=True)
        assert r.returncode in (0, 3), f"{partition}: exit {r.returncode}\n{r.stderr[-1500:]}"
        assert "TRAINING VERIFIED" in r.stdout


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("PASS", name)
            except Exception as exc:  # report every failure, then exit non-zero
                failed += 1; print("FAIL", name, "->", repr(exc))
    sys.exit(1 if failed else 0)
