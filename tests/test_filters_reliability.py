import networkx as nx
import numpy as np
import pandas as pd
import pytest

from stocknet import dependence as dp
from stocknet import filters as fl
from stocknet import networks as nw
from stocknet import reliability as rl


def test_pmfg_is_planar_has_3n_minus_6_edges_and_contains_mst(factor_returns):
    c = nw.correlation(factor_returns)
    n = len(c)
    g = fl.pmfg(c)
    assert nx.check_planarity(g)[0] and g.number_of_edges() == 3 * n - 6
    mst = nw.mst_edges(nw.distance(c))
    for u, v, _ in mst.itertuples(index=False):
        assert g.has_edge(u, v)                                     # PMFG contains the MST for a consistent edge order


def test_mst_imposed_properties_and_fixed_density_has_cycles(factor_returns):
    c = nw.correlation(factor_returns)
    t = fl.mst_from_corr(c)
    s = fl.graph_stats(t)
    assert s["edges"] == len(c) - 1 and s["triangles"] == 0 and s["transitivity"] == 0 and s["components"] == 1
    d = fl.fixed_density_graph(c, 3 * (len(c) - 2))
    assert fl.graph_stats(d)["triangles"] > 0 and d.number_of_edges() == 3 * (len(c) - 2)


def test_sector_statistics_and_permutation_null(factor_returns):
    c = nw.correlation(factor_returns)
    groups = {n: f"B{n[1]}" for n in c.index}                         # fixture sector labels = planted blocks
    g = fl.fixed_density_graph(c, 18)                              # 18 = number of within-block pairs
    s = fl.graph_stats(g, groups)
    assert s["same_sector_edge_share"] > 0.9 and s["sector_assortativity"] > 0.5
    p = fl.permutation_pvalue_same_sector(g, groups, n_perm=300)
    assert p["observed"] > p["null_p95"] and p["p_value"] < 0.01
    assert fl.expected_same_group_share(groups, c.index) == pytest.approx((3 * 4 * 3) / (12 * 11))


def test_dependence_estimators_agree_on_gaussian_blocks_but_are_distinct(factor_returns):
    w = factor_returns.iloc[:300]
    p, s, k = dp.pearson(w), dp.spearman(w), dp.kendall_as_correlation(w)
    assert np.allclose(np.diag(p), 1) and np.allclose(np.diag(s), 1) and np.allclose(np.diag(k), 1)
    iu = np.triu_indices(len(p), 1)
    assert np.corrcoef(p.values[iu], s.values[iu])[0, 1] > 0.98 and np.corrcoef(p.values[iu], k.values[iu])[0, 1] > 0.98
    assert not np.allclose(p.values, s.values)
    ref = w.iloc[:, :2].corr("spearman").iloc[0, 1]
    assert s.iloc[0, 1] == pytest.approx(ref)                           # rank-Pearson equals Spearman


def test_market_residuals_remove_common_mode():
    rng = np.random.default_rng(2)                                      # fixture
    m = rng.normal(0, 0.01, 500)
    x = pd.DataFrame({f"s{i}": 1.0 * m + rng.normal(0, 0.005, 500) for i in range(6)})
    raw = dp.pearson(x).values[np.triu_indices(6, 1)].mean()
    res = dp.pearson(dp.market_residuals(x)).values[np.triu_indices(6, 1)].mean()
    assert raw > 0.7 and abs(res) < 0.3


def test_bootstrap_edge_inclusion_and_community_stability(factor_returns):
    w = factor_returns.iloc[:300]
    corr_fn, graph_fn = nw.correlation, lambda c: nw.mst_graph(nw.distance(c))
    base = graph_fn(corr_fn(w))
    inc = rl.edge_inclusion(base, rl.bootstrap_graphs(w, graph_fn, corr_fn, n_boot=40, block=10, seed=3))
    assert len(inc) == len(base.edges) and inc.inclusion.between(0, 1).all() and inc.inclusion.max() > 0.5
    labels = nw.communities(base)
    ari = rl.community_stability(labels, rl.bootstrap_graphs(w, graph_fn, corr_fn, n_boot=20, block=10, seed=4), nw.communities)
    assert np.nanmean(ari) > 0.5


def test_block_indices_are_contiguous_blocks():
    idx = rl.block_indices(100, 10, np.random.default_rng(0))
    assert len(idx) == 100 and np.all(np.diff(idx[:10]) == 1)


def test_planted_recovery_improves_with_sample_length():
    out = rl.planted_block_recovery(24, 4, within=0.5, between=0.1, t_values=[30, 600], n_rep=10, seed=1)   # SYNTHETIC mechanism test
    m = out.groupby("T").same_block_edge_share.mean()
    assert m[600] > m[30]


def test_window_length_overlap_symmetric(factor_returns):
    ov = rl.window_length_overlap(factor_returns, factor_returns.index[-1], list(factor_returns.columns), lengths=(100, 200, 400))
    assert np.allclose(ov.values, ov.values.T) and np.allclose(np.diag(ov.values), 1)
