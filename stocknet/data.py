"""Loading and validating daily price data (canonical wide format) and point-in-time index membership.

Canonical price file: CSV with a ``date`` column (ISO) and one column per ticker holding a *split- and dividend-adjusted
close* (total-return convention). Missing values are NaN, never zero or forward-filled here.
Canonical membership file: CSV ``date,tickers`` where ``tickers`` is a comma-separated list of the index members on that date
(the format of the public hanshof/sp500_constituents history).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


class DataError(ValueError):
    """Raised when input data violates a hard requirement."""


def load_prices(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return validate_prices(df)


def validate_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Hard checks (raise ``DataError``): unique increasing dates, positive prices where present, numeric columns."""
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise DataError("dates must be unique and increasing")
    if not all(pd.api.types.is_numeric_dtype(t) for t in df.dtypes):
        raise DataError("all price columns must be numeric")
    if (df <= 0).any().any():
        raise DataError("prices must be strictly positive where present (zeros usually encode missing data)")
    if df.columns.has_duplicates:
        raise DataError("duplicate ticker columns")
    return df


@dataclass
class QualityReport:
    """Soft diagnostics. Flags are *reported*, never silently repaired."""
    n_dates: int
    n_tickers: int
    missing_share: pd.Series                      # per ticker
    stale_runs: pd.Series                         # longest run of identical consecutive prices per ticker
    suspect_jumps: pd.DataFrame                   # |return| above threshold, with next-day reversal (split/adjustment error signature)
    short_history: list[str] = field(default_factory=list)


def quality_report(prices: pd.DataFrame, jump_threshold: float = 0.4, reversal_tol: float = 0.25, min_obs: int = 252) -> QualityReport:
    """Diagnose missing data, stale quotes and likely corporate-action errors.

    A suspect jump is a one-day simple return beyond +-``jump_threshold`` whose next-day return reverses most of it
    (within ``reversal_tol`` of the inverse); a persistent level shift without reversal is a genuine move or an unadjusted split
    and is listed with ``reversed=False`` so that it can be inspected by hand.
    """
    r = prices.pct_change(fill_method=None)
    rows = []
    for t in prices.columns:
        s = r[t]
        big = s[s.abs() > jump_threshold]
        for d, v in big.items():
            nxt = s.shift(-1).get(d, np.nan)
            rev = bool(pd.notna(nxt) and abs((1 + v) * (1 + nxt) - 1) < reversal_tol * abs(v))
            rows.append(dict(ticker=t, date=d, ret=float(v), next_ret=float(nxt) if pd.notna(nxt) else np.nan, reversed=rev))
    stale = prices.apply(lambda c: _longest_run(c.dropna().values))
    short = [t for t in prices.columns if prices[t].notna().sum() < min_obs]
    return QualityReport(len(prices), prices.shape[1], prices.isna().mean(), stale, pd.DataFrame(rows, columns=["ticker", "date", "ret", "next_ret", "reversed"]), short)


def _longest_run(a: np.ndarray) -> int:
    if len(a) == 0:
        return 0
    best = cur = 1
    for x, y in zip(a[:-1], a[1:]):
        cur = cur + 1 if x == y else 1
        best = max(best, cur)
    return best


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Close-to-close simple returns from adjusted closes. A return is NaN if either price is missing (no forward fill, no zero fill)."""
    return prices.pct_change(fill_method=None).iloc[1:]


def load_membership(path: str | Path, start: str | None = None, end: str | None = None) -> pd.Series:
    """Point-in-time membership: Series indexed by date with a ``frozenset`` of tickers.

    Tickers carrying annotations such as ``ABC (Previously XYZ)`` are reduced to the bare symbol.
    """
    m = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    if start:
        m = m[m.date >= start]
    if end:
        m = m[m.date <= end]
    sets = m.tickers.apply(lambda s: frozenset(x.split(" (")[0].strip() for x in str(s).split(",") if x.strip()))
    return pd.Series(sets.values, index=pd.DatetimeIndex(m.date), name="members")


def members_at(membership: pd.Series, date) -> frozenset:
    """Index members as of ``date`` using only the last membership snapshot on or before it (no look-ahead)."""
    date = pd.Timestamp(date)
    prior = membership.index[membership.index <= date]
    if len(prior) == 0:
        raise DataError(f"no membership snapshot on or before {date.date()}")
    return membership.loc[prior[-1]]


def coverage_by_date(membership: pd.Series, prices: pd.DataFrame, dates, max_lag_days: int = 0) -> pd.DataFrame:
    """Share of the point-in-time members that have a *current* usable price on each date: the quantitative measure of survivorship bias.

    A member counts as covered only if it has a non-null price on ``date`` itself, or (when ``max_lag_days`` > 0) within that many
    trading observations before it. An old last price carried forward does not count, so a delisted stock is not reported as covered.
    If the share falls well below 1 in early years, the price source lacks delisted/removed names and any performance comparison is
    biased toward survivors.
    """
    rows = []
    for d in pd.DatetimeIndex(dates):
        mem = members_at(membership, d)
        recent = prices.loc[:d].tail(max_lag_days + 1)
        have = [t for t in mem if t in prices.columns and len(recent) > 0 and bool(recent[t].notna().any())]
        rows.append(dict(date=d, members=len(mem), with_prices=len(have), coverage=len(have) / len(mem) if mem else np.nan))
    return pd.DataFrame(rows).set_index("date")


def eligible_universe(membership: pd.Series, returns: pd.DataFrame, date, lookback: int, min_valid_share: float = 0.95) -> list[str]:
    """Tickers that (a) were index members on ``date`` and (b) have at least ``min_valid_share`` valid returns in the ``lookback``
    trading days ending on ``date``. Uses data up to and including ``date`` only."""
    window = returns.loc[:date].tail(lookback)
    if len(window) < lookback:
        return []
    mem = members_at(membership, date)
    ok = [t for t in window.columns if t in mem and window[t].notna().mean() >= min_valid_share]
    return sorted(ok)
