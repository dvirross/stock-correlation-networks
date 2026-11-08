"""Hand-made fixtures for the dated-label rule in scripts/run_pointintime_sectors.py (no market data)."""
import importlib.util
from pathlib import Path

import networkx as nx
import pandas as pd

spec = importlib.util.spec_from_file_location("pit", Path(__file__).resolve().parents[1] / "scripts/run_pointintime_sectors.py")
pit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pit)


def test_labels_at_uses_last_snapshot_on_or_before_end_and_first_before_any():
    hist = {"X": [(pd.Timestamp("2012-12-27"), "Information Technology"), (pd.Timestamp("2023-04-13"), "Industrials")],
            "Y": [(pd.Timestamp("2012-12-27"), "Financials")]}
    assert pit.labels_at(hist, pd.Timestamp("2008-12-31")) == {"X": "Information Technology", "Y": "Financials"}
    assert pit.labels_at(hist, pd.Timestamp("2023-04-12"))["X"] == "Information Technology"
    assert pit.labels_at(hist, pd.Timestamp("2023-04-13"))["X"] == "Industrials"


def test_same_sector_share_counts_edges_with_equal_labels():
    g = nx.Graph([("a", "b"), ("b", "c"), ("c", "d")])
    assert pit.same_sector_share(g, {"a": 1, "b": 1, "c": 2, "d": 2}) == 2 / 3
