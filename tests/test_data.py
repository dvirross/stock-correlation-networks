import numpy as np
import pandas as pd
import pytest

from stocknet import data as dt


def _prices():
    idx = pd.bdate_range("2021-01-04", periods=6)
    return pd.DataFrame({"AAA": [10, 10.5, 10.4, 10.6, 10.7, 10.8], "BBB": [20, 20, 20, 20, np.nan, 21]}, index=idx)


def test_validate_rejects_nonpositive_and_unsorted():
    p = _prices()
    p.iloc[2, 0] = 0
    with pytest.raises(dt.DataError):
        dt.validate_prices(p)
    with pytest.raises(dt.DataError):
        dt.validate_prices(_prices().iloc[::-1])


def test_returns_do_not_fill_missing():
    r = dt.simple_returns(_prices())
    assert r["BBB"].isna().sum() == 2                                  # the NaN price kills the return before and after it
    assert r["AAA"].iloc[0] == pytest.approx(0.05)


def test_quality_report_flags_reversed_jump_and_stale_run_and_missing():
    idx = pd.bdate_range("2021-01-04", periods=8)
    p = pd.DataFrame({"X": [10, 10, 10, 10, 10, 10, 10, 10.0], "Y": [10, 11, 5.5, 11, 11.2, 11.3, 11.1, 11.4]}, index=idx)
    p.loc[idx[3], "X"] = np.nan
    q = dt.quality_report(p, jump_threshold=0.3, reversal_tol=0.25, min_obs=5)
    assert q.stale_runs["X"] >= 4 and q.missing_share["X"] == pytest.approx(1 / 8)
    assert (q.suspect_jumps.ticker == "Y").any() and q.suspect_jumps.query("ticker=='Y'").reversed.iloc[0]
    assert "X" not in q.short_history and q.n_tickers == 2


def test_point_in_time_membership_and_coverage(tmp_path):
    f = tmp_path / "m.csv"
    f.write_text('date,tickers\n2020-01-02,"AAA,BBB (Previously OLD),CCC"\n2020-06-01,"AAA,DDD"\n')
    m = dt.load_membership(f)
    assert dt.members_at(m, "2020-03-01") == frozenset({"AAA", "BBB", "CCC"})   # annotation stripped, no look-ahead to June
    assert dt.members_at(m, "2020-06-01") == frozenset({"AAA", "DDD"})
    with pytest.raises(dt.DataError):
        dt.members_at(m, "2019-12-31")
    idx = pd.bdate_range("2020-01-02", periods=5)
    prices = pd.DataFrame({"AAA": 1.0, "CCC": 2.0}, index=idx)
    cov = dt.coverage_by_date(m, prices, [idx[1]])
    assert cov.loc[idx[1], "members"] == 3 and cov.loc[idx[1], "coverage"] == pytest.approx(2 / 3)   # BBB is a missing (delisted) price series


def test_eligible_universe_requires_membership_and_history(factor_returns, membership):
    t = factor_returns.index[200]
    thin = factor_returns.copy()
    thin.iloc[100:200, 0] = np.nan
    names = dt.eligible_universe(membership, thin, t, lookback=100, min_valid_share=0.95)
    assert factor_returns.columns[0] not in names and len(names) == 11
    assert dt.eligible_universe(membership, thin, factor_returns.index[50], lookback=100) == []


def test_coverage_does_not_count_a_stale_carried_forward_price(tmp_path):
    f = tmp_path / "m.csv"
    f.write_text('date,tickers\n2020-01-02,"AAA,BBB"\n')
    m = dt.load_membership(f)
    idx = pd.bdate_range("2020-01-02", periods=6)
    prices = pd.DataFrame({"AAA": 1.0, "BBB": [2.0, 2.0, 2.0, np.nan, np.nan, np.nan]}, index=idx)   # BBB stops trading after day 3 (delisting)
    cov = dt.coverage_by_date(m, prices, [idx[5]])
    assert cov.loc[idx[5], "with_prices"] == 1                        # BBB's last price is stale and must not count as covered
    cov_lag = dt.coverage_by_date(m, prices, [idx[4]], max_lag_days=2)
    assert cov_lag.loc[idx[4], "with_prices"] == 2                    # within a bounded two-observation lag it still counts
