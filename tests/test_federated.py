"""Small tests. Run from the repo root:  python tests/test_federated.py   (or: python -m pytest tests)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from federated import data_utils as du
from federated.fl_model import MLP, evaluate_model, train_local
from federated.partition_data import check_partition, class_count_matrix, dirichlet_partition, iid_partition


def synthetic_labels(n_classes=34, seed=0):
    rng = np.random.default_rng(seed)
    sizes = np.maximum((30000 * np.geomspace(1, 0.001, n_classes)).astype(int), 33)
    return rng.permutation(np.concatenate([[i] * s for i, s in enumerate(sizes)]))


def synthetic_frame(n=200):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n, 39)), columns=[f"f{i}" for i in range(39)])
    df["Label"] = [f"C{i % 34}" for i in range(n)]
    return df


def test_schema_ok_and_errors():
    assert len(du.validate_frame(synthetic_frame())) == 39
    bad = synthetic_frame(); bad.iloc[3, 5] = np.nan
    for broken in (bad, synthetic_frame().drop(columns=["f0"]), synthetic_frame().drop(columns=["Label"])):
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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
