"""Publication-number regression checks for the Stock Networks website article.

Run on a checkout of the dvirross.com site repository (it reads the article text and research/stock-networks/results):
  python research/stock-networks/editorial-validation/check_publication_claims.py --root .

Uses ONLY the Python standard library and committed CSVs. It does not read or
redistribute private prices. Complements, never replaces, notebooks, Python
unit tests, chart inspection, npm site checks, or the scientific source audit.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty/missing result rows: {path}")
    return rows


def near(a: float, b: float, tol: float = 0.00015) -> bool:
    return abs(a - b) <= tol


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    args = p.parse_args()
    root = args.root.resolve()
    results = root / "research/stock-networks/results"
    article_path = root / "src/content/quant-notes/stock-networks-correlation-sna.md"
    a = article_path.read_text(encoding="utf-8")
    errors, warnings, passed = [], [], []

    def check(condition: bool, message: str) -> None:
        (passed if condition else errors).append(message)

    # Syntax and editorial claims that were previously shown to be misleading.
    check("PORTFOLIO_SECTION" not in a, "No unfilled portfolio placeholder in website article")
    check("share only 42% of their MST edges" not in a,
          "MST Jaccard is clearly distinguished from proportion of each tree's edges")
    check("11 equally sized blocks" not in a and "11 equal-size" not in a,
          "Synthetic planted-block groups are described with their actual unequal sizes")
    check("some of the difference between years is real change" not in a,
          "Uncalibrated same-window Bootstrap overlap is not a formal regime-change test")
    check("barely distinguishable from equal weight" not in a,
          "MST volatility difference is described as economically small, not statistically absent")
    check("10.1142/S0218127407018415" in a, "Tumminello et al. (2007) DOI corrected")
    check("10.1002/net.3230010302" in a, "Dreyfus and Wagner (1971) publisher DOI supplied")
    check("unverified" in a.lower() and "survivor" in a.lower(),
          "Unverified sectors and survival-conditioned panel disclosed")
    if "noise floor" in a.lower():
        warnings.append("Bootstrap-pair similarity should be labelled conditional SAME-WINDOW reference, not validated no-change null.")

    s = read_csv(results / "structure_headline.csv")
    stats = {r["graph"]: r for r in s}
    m = stats.get("MST")
    pm = stats.get("PMFG")
    fx = stats.get("Fixed density (3N-6 edges)")
    check(bool(m and pm and fx), "All three graph constructions have headline rows")
    if m and pm and fx:
        check(int(m["edges"]) == 202 and int(pm["edges"]) == int(fx["edges"]) == 603,
              "203-node headline graphs have exactly 202/603/603 retained edges")
        check(int(m["components"]) == int(pm["components"]) == 1 and int(fx["components"]) == 81,
              "Graph connected-components claims match headline result")
        check(near(float(fx["giant_share"]), 0.4631), "Fixed-density giant component is about 46.31%")
        check(int(m["triangles"]) == 0, "No MST triangles counted as discovered information")

    j = read_csv(results / "estimator_edge_jaccard_mst.csv")
    pearson = next((r for r in j if r.get("") == "pearson"), None)
    check(pearson is not None, "Pearson estimator row exists")
    if pearson:
        overlap_j = float(pearson["spearman"])
        # For two trees of 202 edges, J = shared/(404-shared).
        common = overlap_j * 404 / (1 + overlap_j)
        check(near(overlap_j, 0.4225), "Pearson-Spearman MST Jaccard matches 0.4225")
        check(abs(common - 120) < .1,
              "MST Jaccard implies 120 common edges, ~59.4% of each 202-edge tree")

    bootstrap = read_csv(results / "bootstrap_vs_temporal.csv")
    check(len(bootstrap) == 3, "Bootstrap/temporal table has 2008, 2017, 2024")
    for r in bootstrap:
        d = r["window_end"]
        for k in (1, 21, 63, 126, 252):
            label = f"shift_{k}d_overlap_{(252-k)/252:.3f}"
            check(label in r, f"{d}: {k}-day temporal shift uses correct overlap fraction")
            if label in r:
                check(0 <= float(r[label]) <= 1, f"{d}: {k}-day Jaccard within [0,1]")
        check(near(float(r["previous_nonoverlapping_window"]),
                   float(r["shift_252d_overlap_0.000"])),
              f"{d}: previous 252-day nonoverlapping window matches shift-252")
        check(0 <= float(r["bootstrap_pair_noise_floor_median"]) <= 1,
              f"{d}: same-window bootstrap-pair Jaccard within [0,1]")

    st = read_csv(results / "steiner_primary_terminals.csv")
    by_end: dict[str, dict[str, dict]] = {}
    for r in st:
        by_end.setdefault(r["window_end"], {})[r["tree"]] = r
        check(int(r["edges"]) == int(r["nodes"]) - 1,
              f'{r["window_end"]}/{r["tree"]}: tree has nodes-1 edges')
        check(int(r["intermediates"]) == int(r["nodes"]) - 8,
              f'{r["window_end"]}/{r["tree"]}: eight terminals + intermediates')
    full_key = "Steiner on full distance graph (exact)"
    pmfg_key = "Steiner on PMFG (exact)"
    mst_key = "MST subtree (exact, inside the MST)"
    terminals_key = "Terminals only (MST of the 8 stocks, no intermediates)"
    for d, variants in by_end.items():
        check({mst_key, pmfg_key, full_key, terminals_key} <= variants.keys(),
              f"{d}: all four Steiner constructions present")
        if {mst_key, pmfg_key, full_key, terminals_key} <= variants.keys():
            full = float(variants[full_key]["cost"])
            pmfg = float(variants[pmfg_key]["cost"])
            mst = float(variants[mst_key]["cost"])
            terminal_only = float(variants[terminals_key]["cost"])
            check(full <= pmfg + .0002 and pmfg <= mst + .0002,
                  f"{d}: full <= PMFG <= MST costs (nested feasible edge sets)")
            check(full <= terminal_only + .0002,
                  f"{d}: Steiner optimum does not exceed terminal-only MST")
    if "2024-12-30" in by_end:
        v = by_end["2024-12-30"]
        if {mst_key, pmfg_key, full_key} <= v.keys():
            check(near(float(v[mst_key]["cost"]), 26.5798)
                  and near(float(v[pmfg_key]["cost"]), 14.1040)
                  and near(float(v[full_key]["cost"]), 7.9424),
                  "2024 Steiner totals match publication table (26.58/14.10/7.94)")
            check(int(v[full_key]["intermediates"]) == 0,
                  "2024 exact full-graph Steiner reports zero intermediate stocks")

    for png in ("05-persistence.png", "08b-steiner-panels.png"):
        check((root / "public/images/writing/stock-networks" / png).is_file(),
              f"Static fallback image exists: {png}")

    # Generated Plotly JSON is deployed as an artifact and can go stale even
    # after correcting the plot-producing Python script and article prose.
    chart5_path = root / "public/charts/stock-networks/05-persistence.json"
    if chart5_path.is_file():
        chart5 = json.loads(chart5_path.read_text(encoding="utf-8"))
        captions = [str(t.get("name", "")) for t in chart5.get("data", [])]
        check(not any("no-change noise floor" in name.lower() for name in captions),
              "Shipped Figure 5 Plotly legend does not claim a calibrated no-change noise floor")
        check(any("same-window" in name.lower() and "reference" in name.lower()
                  for name in captions),
              "Shipped Figure 5 trace identifies conditional same-window Bootstrap reference")
    else:
        errors.append("Missing generated Figure 5 Plotly JSON")

    # Recompute portfolio risks directly from DAILY net returns, independently
    # from the research code that produced the summary CSV.
    wf = results / "walkforward_summary.csv"
    net = results / "walkforward_daily_net_10bps.csv"
    jfile = results / "walkforward_vol_difference_vs_equal_weight.json"
    portfolio_names = ("equal_weight", "min_variance", "hrp", "mst_inverse_degree")
    if wf.is_file() and net.is_file() and jfile.is_file():
        rows = read_csv(wf)
        check("execution_lag" in rows[0] and "n_rebalances" in rows[0],
              "Portfolio CSV regenerated after next-close execution (lag/rebalance columns present)")
        # Expected 24 result rows: four strategies × two execution lags × three cost levels.
        check(len(rows) == 24, "Portfolio summary has all 24 strategy/lag/cost configurations")
        lagged = {r["strategy"]: r for r in rows
                  if r["execution_lag"] == "1" and r["cost_bps"] == "10"}
        check(set(lagged) == set(portfolio_names),
              "All four next-close 10-bps strategy summaries are available")

        data = read_csv(net)
        check(len(data) == 5012, "Daily 10-bps next-close file has 5,012 trading dates")
        check(data[0].get("") == "2005-02-01" and data[-1].get("") == "2024-12-30",
              "Daily strategy-return file has expected dated endpoints")
        computed = {}
        for name in portfolio_names:
            try:
                series = [float(row[name]) for row in data]
                check(len(series) > 1 and all(math.isfinite(x) and x > -1 for x in series),
                      f"{name}: all daily net returns are finite and > -100%")
                avg = sum(series) / len(series)
                var = sum((x - avg)**2 for x in series) / (len(series) - 1)
                vol = math.sqrt(var * 252)
                wealth, peak, drawdown = 1.0, 1.0, 0.0
                for x in series:
                    wealth *= 1 + x
                    peak = max(peak, wealth)
                    drawdown = min(drawdown, wealth / peak - 1)
                computed[name] = vol
                if name in lagged:
                    check(abs(vol - float(lagged[name]["ann_vol"])) < 0.00006,
                          f"{name}: annual vol recomputed from 5,012 daily net returns")
                    check(abs(drawdown - float(lagged[name]["max_drawdown"])) < 0.00006,
                          f"{name}: maximum drawdown independently recomputed")
                    check(int(lagged[name]["n_days"]) == len(series),
                          f"{name}: correct daily return count")
            except (ValueError, KeyError, ZeroDivisionError) as exc:
                errors.append(f"{name}: unable to recalculate daily strategy risk ({exc})")

        intervals = json.loads(jfile.read_text(encoding="utf-8"))
        if "equal_weight" in computed:
            for name in portfolio_names[1:]:
                if name not in computed or name not in intervals:
                    errors.append(f"{name}: paired Bootstrap result unavailable")
                    continue
                b = intervals[name]
                diff = computed[name] - computed["equal_weight"]
                check(abs(diff - float(b["estimate"])) < 0.00007,
                      f"{name}: paired annual-vol difference matches independent daily calculation")
                check(float(b["ci_low"]) < diff < float(b["ci_high"]),
                      f"{name}: 95% Bootstrap interval brackets reported difference")
                check(int(b["n_boot"]) == 2000 and int(b["block"]) == 21,
                      f"{name}: paired Bootstrap has recorded B=2000, block=21")
        check(all("execution_lag" in r for r in rows),
              "Every strategy output records its execution timing")
    else:
        errors.append("Missing a final portfolio CSV/return-series/bootstrap JSON")

    method_file = root / "research/stock-networks/METHODOLOGY.md"
    if method_file.is_file():
        methods = method_file.read_text(encoding="utf-8")
        check("empirical results not yet produced" not in methods.lower(),
              "Methodology status reflects Phase B empirical completion")
        check("Trade at the close of t paying" not in methods,
              "Methodology portfolio timeline reflects next-close execution")

    print(f"Checked: {len(passed)} PASSED, {len(errors)} BLOCKERS, {len(warnings)} WARNINGS")
    for m in errors:
        print("BLOCK:", m)
    for m in warnings:
        print("WARN:", m)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
