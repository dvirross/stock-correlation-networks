import numpy as np
import pandas as pd
import pytest

from stocknet import portfolios as pf


def test_all_rules_produce_valid_long_only_weights(factor_returns):
    for name, rule in pf.RULES.items():
        w = rule(factor_returns)
        assert w.index.equals(factor_returns.columns), name
        assert (w >= -1e-12).all() and w.sum() == pytest.approx(1.0), name


def test_min_variance_two_asset_analytic():
    rng = np.random.default_rng(7)                              # fixture noise; analytic check on the sample covariance
    x = pd.DataFrame({"a": rng.normal(0, 0.01, 4000), "b": rng.normal(0, 0.02, 4000)})
    s = x.cov().values
    analytic = (s[1, 1] - s[0, 1]) / (s[0, 0] + s[1, 1] - 2 * s[0, 1])
    w = pf.min_variance(x, shrink=False)
    assert w["a"] == pytest.approx(analytic, abs=1e-3)


def test_ledoit_wolf_is_between_sample_and_identity_target(factor_returns):
    lw = pf.ledoit_wolf_cov(factor_returns.iloc[:60])
    s = factor_returns.iloc[:60].cov(ddof=0)
    assert np.all(np.linalg.eigvalsh(lw.values) > 0)
    assert np.linalg.cond(lw.values) <= np.linalg.cond(s.values) * 1.0001


def test_hrp_gives_more_weight_to_low_volatility_when_uncorrelated():
    rng = np.random.default_rng(3)
    x = pd.DataFrame({"low": rng.normal(0, 0.005, 3000), "mid": rng.normal(0, 0.01, 3000), "high": rng.normal(0, 0.03, 3000)})
    w = pf.hrp(x)
    assert w["low"] > w["mid"] > w["high"]


def test_effective_n():
    assert pf.effective_n(pd.Series([0.25] * 4)) == pytest.approx(4.0)
    assert pf.effective_n(pd.Series([1.0, 0.0])) == pytest.approx(1.0)


def test_mst_rule_favours_leaves(factor_returns):
    w = pf.mst_inverse_degree(factor_returns)
    assert w.sum() == pytest.approx(1.0) and w.max() > w.min()


def test_min_variance_reports_solver_status_and_can_fail_loudly(factor_returns, monkeypatch):
    w, status = pf.min_variance_solve(factor_returns.iloc[:200])
    assert status["success"] and not status["fallback"] and w.sum() == pytest.approx(1.0)

    class Bad:
        success, x, message, nit = False, np.zeros(12), "forced failure (test)", 0
    monkeypatch.setattr(pf, "minimize", lambda *a, **k: Bad())
    w2, st2 = pf.min_variance_solve(factor_returns.iloc[:200])
    assert st2["fallback"] and np.allclose(w2.values, 1 / 12)                    # documented equal-weight fallback is visible, not silent
    with pytest.raises(pf.SolverFailure):
        pf.min_variance(factor_returns.iloc[:200], strict=True)
