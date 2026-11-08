"""TEST FIXTURES ONLY. The arrays below are hand-made or fixed-seed arithmetic inputs used to test calculations.
They are NOT market data, are never used for any figure or result, and must never be presented as such."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def factor_returns():
    """Fixture: 3 blocks of 4 assets driven by 3 common factors plus noise (fixed seed), 600 business days."""
    rng = np.random.default_rng(12345)
    n, per = 600, 4
    f = rng.normal(0, 0.01, (n, 3))
    cols, data = [], []
    for b in range(3):
        for k in range(per):
            data.append(0.8 * f[:, b] + rng.normal(0, 0.006, n))
            cols.append(f"B{b}_{k}")
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(np.column_stack(data), index=idx, columns=cols)


@pytest.fixture
def membership(factor_returns):
    idx = factor_returns.index
    return pd.Series([frozenset(factor_returns.columns)], index=pd.DatetimeIndex([idx[0]]))
