"""Dependence estimators on the same window: Pearson, Spearman, Kendall (and factor-residual Pearson).

They are different estimands. Spearman is the Pearson correlation of ranks (monotonic association); Kendall's tau compares
concordant and discordant pairs. To put Kendall on the correlation scale used by the distance d = sqrt(2 (1 - rho)), the
elliptical-family transform rho = sin(pi tau / 2) is applied; this is an assumption and is labelled wherever used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def pearson(window: pd.DataFrame) -> pd.DataFrame:
    return window.corr(method="pearson")


def spearman(window: pd.DataFrame) -> pd.DataFrame:
    return window.rank().corr(method="pearson")        # identical to method="spearman" with average ranks, but vectorised


def kendall_tau(window: pd.DataFrame) -> pd.DataFrame:
    """Kendall tau-b matrix (pairwise-complete data). O(N^2 T log T): use on selected windows."""
    cols = list(window.columns)
    x = window.to_numpy()
    n = len(cols)
    out = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            ok = ~(np.isnan(x[:, i]) | np.isnan(x[:, j]))
            t = stats.kendalltau(x[ok, i], x[ok, j]).statistic if ok.sum() > 2 else np.nan
            out[i, j] = out[j, i] = t
    return pd.DataFrame(out, index=cols, columns=cols)


def kendall_as_correlation(window: pd.DataFrame) -> pd.DataFrame:
    """sin(pi tau / 2): maps Kendall's tau to the Pearson scale under an elliptical-distribution assumption."""
    tau = kendall_tau(window)
    return np.sin(np.pi * tau / 2.0)


def market_residuals(window: pd.DataFrame) -> pd.DataFrame:
    """Residual of each stock after regressing on the equal-weighted market return of the same window (one-factor, in-window).
    Removes the common mode; the resulting correlations answer a different question from raw Pearson.

    CAUTION: the factor is the equal-weighted mean of the same panel, so the residuals sum to zero across stocks on every date. Their
    covariance matrix is therefore singular (rank at most N-1) and the result is an in-panel residual correlation, not a full partial
    correlation or an out-of-panel market-factor model."""
    m = window.mean(axis=1).to_numpy()
    x = window.to_numpy()
    mc = m - m.mean()
    beta = ((x - x.mean(0)) * mc[:, None]).sum(0) / (mc ** 2).sum()
    resid = x - x.mean(0) - np.outer(mc, beta)
    return pd.DataFrame(resid, index=window.index, columns=window.columns)


ESTIMATORS = {"pearson": pearson, "spearman": spearman, "kendall": kendall_as_correlation,
              "pearson_market_residual": lambda w: pearson(market_residuals(w))}
