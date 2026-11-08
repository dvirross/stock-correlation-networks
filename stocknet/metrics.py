"""Performance summaries and a moving-block bootstrap for strategy comparisons (resamples *observed* strategy returns only)."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def summarize(daily: pd.Series, turnover: pd.Series | None = None, periods_per_year: int = TRADING_DAYS) -> dict:
    r = daily.dropna()
    wealth = (1 + r).cumprod()
    ann_ret = wealth.iloc[-1] ** (periods_per_year / len(r)) - 1
    vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    out = dict(ann_return=ann_ret, ann_vol=vol, sharpe_no_rf=(r.mean() * periods_per_year) / vol if vol > 0 else np.nan,
               max_drawdown=float((wealth / wealth.cummax() - 1).min()), n_days=len(r))
    if turnover is not None and len(turnover):
        out["avg_monthly_one_way_turnover"] = float(turnover.mean())
    return out


def block_bootstrap_diff(a: pd.Series, b: pd.Series, stat=lambda x: x.mean() / x.std(ddof=1) * np.sqrt(TRADING_DAYS),
                         block: int = 21, n_boot: int = 2000, seed: int = 0) -> dict:
    """Moving-block bootstrap of ``stat(a) - stat(b)`` on paired daily returns (default stat: annualized Sharpe, no risk-free rate).
    Dependence within blocks is preserved; the series themselves are observed strategy returns, never simulated market paths."""
    df = pd.concat([a, b], axis=1, join="inner").dropna()
    x = df.values
    n = len(x)
    rng = np.random.default_rng(seed)
    starts_max = n - block
    k = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, starts_max + 1, size=k)
        samp = np.concatenate([x[j:j + block] for j in s])[:n]
        diffs[i] = stat(pd.Series(samp[:, 0])) - stat(pd.Series(samp[:, 1]))
    est = stat(df.iloc[:, 0]) - stat(df.iloc[:, 1])
    return dict(estimate=float(est), ci_low=float(np.quantile(diffs, 0.025)), ci_high=float(np.quantile(diffs, 0.975)), block=block, n_boot=n_boot)
