"""Leakage-free walk-forward evaluation with turnover and proportional transaction costs.

Timeline for a rebalance (signal) date t, with ``execution_lag`` L trading days (default 1):
  1. universe  = index members on t with enough valid returns in the lookback window ending at t  (information up to and including t)
  2. weights   = rule(window of the last ``lookback`` returns up to and including t)
  3. execution = at the close of trading day e = t + L (L = 0 reproduces the idealised "trade at the close you just observed" convention,
                 which is not directly implementable; L >= 1 is). Cost is ``cost_bps`` per unit of traded notional on |target - drifted weights|.
  4. holding   = every trading day from the first execution to the end of the sample is accounted for. A day's return is earned by the
                 weights held *before* that day's execution; weights drift with returns until the next execution. If a rebalance is skipped
                 (too few eligible names) the previous weights simply continue to be held, so no holding-period return is dropped.
Nothing dated after t enters steps 1 and 2, and no weight is applied to a return observed before it could be traded.

Delistings: if a held ticker has a missing return while held, the result is not honest unless a delisting return is supplied. The default
``on_missing='raise'`` therefore refuses to continue; ``'zero'`` and ``'drop'`` exist for diagnostics and are counted in the output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import data as dt


@dataclass
class BacktestResult:
    gross: pd.Series
    net: pd.Series
    turnover: pd.Series                  # one-way turnover per rebalance (half of sum |dw|)
    weights: dict                        # date -> target weights
    n_universe: pd.Series
    missing_events: int = 0
    log: list = field(default_factory=list)


def month_end_dates(index: pd.DatetimeIndex, start=None) -> list:
    s = pd.Series(index, index=index)
    last = s.groupby(index.to_period("M")).last()
    out = list(last.values)
    return [d for d in out if start is None or d >= pd.Timestamp(start)]


def run_backtest(returns: pd.DataFrame, membership: pd.Series, rule: Callable[[pd.DataFrame], pd.Series], rebalance_dates,
                 lookback: int = 252, cost_bps: float = 10.0, min_valid_share: float = 0.95, min_names: int = 10,
                 on_missing: str = "raise", execution_lag: int = 1) -> BacktestResult:
    if on_missing not in {"raise", "zero", "drop"}:
        raise ValueError("on_missing must be raise, zero or drop")
    if execution_lag < 0:
        raise ValueError("execution_lag must be >= 0")
    idx = returns.index
    pos = {d: i for i, d in enumerate(idx)}
    targets, weights, nuni, log = {}, {}, {}, []
    for t in rebalance_dates:
        if t not in pos:
            continue
        e = pos[t] + execution_lag
        if e >= len(idx):
            log.append((t, "skipped: execution date beyond the sample", 0))
            continue
        names = dt.eligible_universe(membership, returns, t, lookback, min_valid_share)
        if len(names) < min_names:
            log.append((t, "skipped: too few eligible names", len(names)))
            continue
        win = returns.loc[:t, names].tail(lookback)
        win = win.loc[:, win.notna().all()]                  # complete columns only (documented simplification)
        if win.shape[1] < min_names:
            log.append((t, "skipped: too few complete columns", win.shape[1]))
            continue
        w = rule(win)
        w = w[w > 0]
        w = w / w.sum()
        targets[e] = (t, w)
        weights[t] = w
        nuni[t] = len(names)
    gross, net, turn = {}, {}, {}
    missing_events = 0
    if targets:
        cur = None
        for i in range(min(targets), len(idx)):
            d = idx[i]
            pr = 0.0
            if cur is not None:
                r = returns.iloc[i][cur.index]
                miss = r.isna()
                if miss.any():
                    missing_events += int(miss.sum())
                    if on_missing == "raise":
                        raise dt.DataError(f"held ticker(s) {list(r.index[miss])} have no return on {d.date()} (delisting or data gap): supply delisting returns or restrict the universe")
                    if on_missing == "drop":
                        cur = cur[~miss]
                        cur = cur / cur.sum()
                        r = r[cur.index]
                    else:
                        r = r.fillna(0.0)
                pr = float((cur * r).sum())
                cur = cur * (1 + r) / (1 + pr)
            cost = 0.0
            if i in targets:
                t, w = targets[i]
                drifted = (cur if cur is not None else pd.Series(dtype=float))
                union = w.index.union(drifted.index)
                dw = (w.reindex(union).fillna(0.0) - drifted.reindex(union).fillna(0.0)).abs().sum()
                turn[t] = 0.5 * dw
                cost = cost_bps / 1e4 * dw
                cur = w.copy()
            gross[d] = pr
            net[d] = (1 + pr) * (1 - cost) - 1
    return BacktestResult(pd.Series(gross, dtype=float), pd.Series(net, dtype=float), pd.Series(turn, dtype=float), weights,
                          pd.Series(nuni, dtype=float), missing_events, log)
