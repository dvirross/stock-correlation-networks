import numpy as np
import pandas as pd
import pytest

from stocknet import backtest as bt
from stocknet import data as dt
from stocknet import portfolios as pf


def _setup(factor_returns, membership):
    dates = bt.month_end_dates(factor_returns.index, start=factor_returns.index[300])
    return dates


def test_no_lookahead_weights_depend_only_on_past(factor_returns, membership):
    dates = _setup(factor_returns, membership)
    base = bt.run_backtest(factor_returns, membership, pf.min_variance, dates, lookback=120, cost_bps=0)
    t = dates[2]
    changed = factor_returns.copy()
    changed.loc[changed.index > t] = np.random.default_rng(1).normal(0, 0.05, changed.loc[changed.index > t].shape)   # fixture scrambling of the future
    alt = bt.run_backtest(changed, membership, pf.min_variance, dates[:3], lookback=120, cost_bps=0)
    for d in dates[:3]:
        assert np.allclose(base.weights[d].sort_index().values, alt.weights[d].sort_index().values)


def test_turnover_and_cost_accounting(factor_returns, membership):
    dates = _setup(factor_returns, membership)
    free = bt.run_backtest(factor_returns, membership, pf.equal_weight, dates, lookback=120, cost_bps=0)
    paid = bt.run_backtest(factor_returns, membership, pf.equal_weight, dates, lookback=120, cost_bps=25)
    assert free.turnover.iloc[0] == pytest.approx(0.5)               # first investment: 100% traded, one-way 0.5
    assert (paid.net <= free.net + 1e-15).all()
    first_day = free.gross.index[0]                                  # execution day of the first trade: nothing held yet, entry cost only
    assert free.gross[first_day] == 0.0
    assert paid.net[first_day] == pytest.approx(-25e-4 * 1.0)        # sum|dw| = 1 on the first execution


def test_equal_weight_turnover_comes_from_drift_only(factor_returns, membership):
    dates = _setup(factor_returns, membership)
    r = bt.run_backtest(factor_returns, membership, pf.equal_weight, dates, lookback=120, cost_bps=0)
    later = r.turnover.iloc[1:]
    assert (later > 0).all() and (later < 0.1).all()                  # small drift rebalancing, not full turnover


def test_refuses_to_continue_when_a_held_name_loses_data(factor_returns, membership):
    broken = factor_returns.copy()
    broken.iloc[400:, 0] = np.nan                                      # a held ticker disappears (delisting without a delisting return)
    dates = _setup(broken, membership)
    with pytest.raises(dt.DataError):
        bt.run_backtest(broken, membership, pf.equal_weight, dates, lookback=120, cost_bps=0)
    ok = bt.run_backtest(broken, membership, pf.equal_weight, dates, lookback=120, cost_bps=0, on_missing="zero")
    assert ok.missing_events > 0                                       # diagnostic mode counts the events


def test_gross_return_matches_manual_computation(factor_returns, membership):
    dates = _setup(factor_returns, membership)
    r = bt.run_backtest(factor_returns, membership, pf.equal_weight, dates, lookback=120, cost_bps=0)
    d1 = r.gross.index[1]                                               # first full holding day, after the execution day
    assert r.gross[d1] == pytest.approx(factor_returns.loc[d1].mean())  # equal weights


def _alternating_rule(cols):
    state = {"k": 0}

    def rule(window):
        col = cols[state["k"] % len(cols)]
        state["k"] += 1
        w = pd.Series(0.0, index=window.columns)
        w[col] = 1.0
        return w
    return rule


def test_execution_lag_applies_old_weights_until_the_trade_date(factor_returns, membership):
    dates = _setup(factor_returns, membership)[:4]
    idx = factor_returns.index
    cols = ["B0_0", "B1_0"]
    for lag in (0, 1, 2):
        r = bt.run_backtest(factor_returns, membership, _alternating_rule(cols), dates, lookback=120, cost_bps=0, execution_lag=lag)
        for k, t in enumerate(dates):
            e = idx[idx.get_loc(t) + lag]
            if k > 0:                                                    # return on the execution day is earned by the PREVIOUS weights
                assert r.gross[e] == pytest.approx(factor_returns.loc[e, cols[(k - 1) % 2]])
            nxt = idx[idx.get_loc(e) + 1]
            if nxt in r.gross.index and (k + 1 >= len(dates) or nxt != idx[idx.get_loc(dates[k + 1]) + lag]):
                assert r.gross[nxt] == pytest.approx(factor_returns.loc[nxt, cols[k % 2]])   # the day after: the new weights


def test_signal_day_return_is_not_earned_with_positive_lag(factor_returns, membership):
    dates = _setup(factor_returns, membership)[:2]
    r0 = bt.run_backtest(factor_returns, membership, pf.equal_weight, dates, lookback=120, cost_bps=0, execution_lag=0)
    r1 = bt.run_backtest(factor_returns, membership, pf.equal_weight, dates, lookback=120, cost_bps=0, execution_lag=1)
    assert r0.gross.index[0] == dates[0]
    assert r1.gross.index[0] == factor_returns.index[factor_returns.index.get_loc(dates[0]) + 1]


def test_skipped_rebalance_does_not_drop_holding_period_returns(factor_returns):
    idx = factor_returns.index
    dates = bt.month_end_dates(idx, start=idx[300])
    skip = dates[2]                                                     # at this date only 3 names are members: the rebalance is skipped
    full = frozenset(factor_returns.columns)
    mem = pd.Series([full, frozenset(["B0_0", "B0_1", "B0_2"]), full], index=pd.DatetimeIndex([idx[0], skip, dates[3]]))
    r = bt.run_backtest(factor_returns, mem, pf.equal_weight, dates, lookback=120, cost_bps=0, min_names=10)
    assert any("too few eligible names" in x[1] for x in r.log)
    expected = idx[idx.get_loc(dates[0]) + 1:]
    assert list(r.gross.index) == list(expected)                        # every trading day after the first execution has a return
