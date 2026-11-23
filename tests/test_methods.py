"""Unit tests for the core methods (run with: pytest -q)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sehs.methods import (ccd, ccd_modified, entropy_weights, critic_weights, pca_weights, minmax_pooled,  # noqa
                          obstacle_degree, bridge_strength, normal_scores, glasso_net, dumitrescu_hurlin, classify)
from sehs.config import CLASSES  # noqa: E402


def test_ccd_equal_subsystems_have_full_coupling():
    U = np.full((3, 4), 0.6)
    C, T, D = ccd(U)
    assert np.allclose(C, 1) and np.allclose(T, 0.6) and np.allclose(D, np.sqrt(0.6))


def test_ccd_imbalance_lowers_coupling():
    C1, _, _ = ccd(np.array([[0.5, 0.5, 0.5, 0.5]]))
    C2, _, _ = ccd(np.array([[0.2, 0.8, 0.5, 0.5]]))
    Cm1, _, _ = ccd_modified(np.array([[0.5, 0.5, 0.5, 0.5]]))
    Cm2, _, _ = ccd_modified(np.array([[0.2, 0.8, 0.5, 0.5]]))
    assert C2[0] < C1[0] and Cm2[0] < Cm1[0]
    assert (C1[0] - C2[0]) < (Cm1[0] - Cm2[0])   # the modified degree has a wider range


def test_weights_sum_to_one():
    rng = np.random.default_rng(1)
    X = rng.uniform(0.01, 1, (50, 5))
    for f in (entropy_weights, critic_weights, pca_weights):
        w = f(X)
        assert np.isclose(w.sum(), 1) and (w >= 0).all()


def test_minmax_direction_and_range():
    df = pd.DataFrame({"good": np.arange(100.0), "bad": np.arange(100.0)})
    out = minmax_pooled(df, {"good": 1, "bad": -1})
    assert out.good.iloc[-1] > out.good.iloc[0] and out.bad.iloc[-1] < out.bad.iloc[0]
    assert out.values.min() >= 0.01 and out.values.max() <= 1.0


def test_obstacle_degree_sums_to_one():
    o = obstacle_degree(np.array([0.2, 0.9, 0.5]), np.array([0.5, 0.3, 0.2]))
    assert np.isclose(o.sum(), 1) and o.argmax() == 0


def test_bridge_strength_counts_only_other_groups():
    W = np.array([[0, .5, .2], [.5, 0, 0], [.2, 0, 0]])
    b = bridge_strength(W, np.array(["a", "a", "b"]))
    assert np.allclose(b, [0.2, 0.0, 0.2])


def test_glasso_recovers_chain():
    rng = np.random.default_rng(2)
    x1 = rng.normal(size=400); x2 = x1 + rng.normal(size=400); x3 = x2 + rng.normal(size=400)
    pc, _, r2 = glasso_net(normal_scores(np.column_stack([x1, x2, x3])), alpha=0.05)
    assert abs(pc[0, 2]) < abs(pc[0, 1]) and abs(pc[0, 2]) < abs(pc[1, 2])


def test_dumitrescu_hurlin_detects_causality():
    rng = np.random.default_rng(3)
    rows = []
    for i in range(30):
        x = rng.normal(size=40); y = np.zeros(40)
        for t in range(1, 40):
            y[t] = 0.3 * y[t - 1] + 0.8 * x[t - 1] + rng.normal()
        rows += [dict(iso3=i, year=t, x=x[t], y=y[t]) for t in range(40)]
    df = pd.DataFrame(rows)
    assert dumitrescu_hurlin(df, "y", "x")["p_Ztilde"] < 0.001
    assert dumitrescu_hurlin(df, "x", "y")["p_Ztilde"] > 0.01


def test_classes():
    lab = classify(np.array([0.35, 0.45, 0.55, 0.65, 0.8]), CLASSES)
    assert list(lab) == [c[2] for c in CLASSES]
