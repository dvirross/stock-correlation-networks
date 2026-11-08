import numpy as np
import pandas as pd
import pytest

from stocknet import metrics as mt
from stocknet import networks as nw
from stocknet import viz


def test_summarize_known_series():
    r = pd.Series([0.01, -0.01, 0.02, 0.0] * 63)
    s = mt.summarize(r)
    assert s["max_drawdown"] <= 0 and s["n_days"] == 252
    assert s["ann_return"] == pytest.approx((1 + r).prod() ** (252 / len(r)) - 1)


def test_block_bootstrap_reproducible_and_contains_estimate_for_identical_series(factor_returns):
    a = factor_returns.iloc[:, 0]
    out1 = mt.block_bootstrap_diff(a, a, n_boot=200, seed=1)
    out2 = mt.block_bootstrap_diff(a, a, n_boot=200, seed=1)
    assert out1 == out2 and out1["estimate"] == pytest.approx(0.0) and out1["ci_low"] == pytest.approx(0.0) == pytest.approx(out1["ci_high"])


def test_figure_builders_and_export_write_spec_and_png(tmp_path, factor_returns):
    c = nw.correlation(factor_returns)
    g = nw.mst_graph(nw.distance(c))
    groups = {n: f"Block {n[1]}" for n in g.nodes}                     # fixture labels
    fig = viz.fig_mst(g, groups)
    assert len(fig.data) == 1 + 3
    viz.export(fig, tmp_path, "mst", png=False)
    assert (tmp_path / "mst.json").exists()
    eq = viz.fig_equity_curves(factor_returns.iloc[:, :3])
    assert len(eq.data) == 3
    ml = viz.fig_metric_lines(pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]}, index=pd.bdate_range("2020-01-01", periods=3)), {"a": "A", "b": "B"})
    assert len(ml.data) == 2
    sim = pd.DataFrame(np.eye(3), index=pd.bdate_range("2020-01-01", periods=3), columns=pd.bdate_range("2020-01-01", periods=3))
    assert viz.fig_stability_heatmap(sim).data[0].zmax == 1
