"""Weighting rules: equal weight, long-only minimum variance (Ledoit-Wolf shrinkage), Hierarchical Risk Parity and an MST rule.

Every rule takes only a window of *past* simple returns (rows = days, columns = tickers, no missing values) and returns
non-negative weights summing to one. None of them looks at the future by construction; tests/test_leakage.py checks this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from . import networks as nw


def equal_weight(window: pd.DataFrame) -> pd.Series:
    n = window.shape[1]
    return pd.Series(1.0 / n, index=window.columns)


def ledoit_wolf_cov(window: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf (2004) linear shrinkage of the sample covariance toward a scaled identity, with the analytic shrinkage intensity."""
    x = window.values - window.values.mean(0)
    n, p = x.shape
    s = x.T @ x / n
    mu = np.trace(s) / p
    d2 = np.sum((s - mu * np.eye(p)) ** 2) / p
    b2 = sum(np.sum((np.outer(r, r) - s) ** 2) for r in x) / (n * n * p)
    b2 = min(b2, d2)
    shrink = b2 / d2 if d2 > 0 else 0.0
    return pd.DataFrame((1 - shrink) * s + shrink * mu * np.eye(p), index=window.columns, columns=window.columns)


class SolverFailure(RuntimeError):
    """Raised by ``min_variance(strict=True)`` when the optimizer does not converge."""


def min_variance_solve(window: pd.DataFrame, shrink: bool = True) -> tuple[pd.Series, dict]:
    """Long-only fully invested minimum-variance portfolio (SLSQP). Returns ``(weights, status)``; ``status['success']`` is the
    optimizer's flag. If it did not converge the returned weights are the equal-weight fallback and ``status['fallback']`` is True."""
    cov = ledoit_wolf_cov(window) if shrink else window.cov()
    p = cov.shape[0]
    c = cov.values
    res = minimize(lambda w: w @ c @ w, np.full(p, 1 / p), jac=lambda w: 2 * c @ w, bounds=[(0, 1)] * p,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1}, method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    ok = bool(res.success)
    w = res.x if ok else np.full(p, 1 / p)
    w = np.clip(w, 0, None)
    return pd.Series(w / w.sum(), index=cov.index), dict(success=ok, fallback=not ok, message=str(res.message), n_iter=int(res.nit))


def min_variance(window: pd.DataFrame, shrink: bool = True, strict: bool = False) -> pd.Series:
    """Minimum-variance weights. By default a non-converged solve silently falls back to equal weight (use ``min_variance_solve``
    to see the status, or ``strict=True`` to raise ``SolverFailure`` instead)."""
    w, status = min_variance_solve(window, shrink)
    if strict and not status["success"]:
        raise SolverFailure(status["message"])
    return w


def hrp(window: pd.DataFrame) -> pd.Series:
    """Hierarchical Risk Parity (Lopez de Prado, 2016): single-linkage clustering on correlation distance, quasi-diagonalization,
    recursive bisection allocating inversely to cluster variance (inverse-variance weights inside clusters)."""
    cov = window.cov()
    corr = window.corr()
    dist = np.array(np.sqrt(0.5 * (1 - corr.clip(-1, 1).to_numpy(dtype=float))), copy=True)
    np.fill_diagonal(dist, 0.0)
    order = list(leaves_list(linkage(squareform(dist, checks=False), method="single")))
    w = pd.Series(1.0, index=[cov.index[i] for i in order])
    clusters = [list(w.index)]
    while clusters:
        clusters = [c[i:j] for c in clusters for i, j in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for k in range(0, len(clusters), 2):
            c0, c1 = clusters[k], clusters[k + 1]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            alpha = 1 - v0 / (v0 + v1)
            w[c0] *= alpha
            w[c1] *= 1 - alpha
    return (w / w.sum()).reindex(window.columns)


def _cluster_var(cov: pd.DataFrame, items: list) -> float:
    sub = cov.loc[items, items].values
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def mst_inverse_degree(window: pd.DataFrame) -> pd.Series:
    """Exploratory network rule: weight proportional to 1 / degree in the correlation MST, so peripheral (leaf) stocks, which
    are structurally far from the rest of the market network, carry more weight than hubs. A hypothesis to test, not a recommendation."""
    g = nw.mst_graph(nw.distance(nw.correlation(window)))
    w = pd.Series({n: 1.0 / d for n, d in g.degree() if d > 0}, dtype=float).reindex(window.columns).fillna(0.0)
    return w / w.sum()


RULES = {"equal_weight": equal_weight, "min_variance": min_variance, "hrp": hrp, "mst_inverse_degree": mst_inverse_degree}


def effective_n(w: pd.Series) -> float:
    return float(1.0 / np.sum(np.square(w.values)))
