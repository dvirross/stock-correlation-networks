"""Build article figures (Plotly JSON + static PNG) from the real-data panel and results/*.csv. Usage: make_figures.py [names...]"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, leaves_list
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stocknet import pipeline as pl, networks as nw, filters as fl, viz
from stocknet.viz import PALETTE, GRID, AXIS, MUTED, INK, BASE

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent.parent
R = ROOT / "results"
CH = ROOT / "figures/charts"
IM = ROOT / "figures/images"
HEAD = "2024-12-30"
want = set(sys.argv[1:])
px, returns, sec = pl.load_panel(ROOT / "data/panel_adjusted_close.csv", ROOT / "data/sectors.csv")
sec = {k: v for k, v in sec.items() if k in returns.columns}
SECTORS = sorted(set(sec.values()))
COL = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(SECTORS)}
w = pl.window_ending(returns, HEAD, 252)
corr = nw.correlation(w)
AX = dict(gridcolor=GRID, linecolor=AXIS, zeroline=False)

def out(fig, name, size, png=True):
    viz.export(fig, CH, name, size=size, png=png); (IM).mkdir(parents=True, exist_ok=True)   # png=False for button-driven figures: their fallback is the separate static panel image
    p = CH / f"{name}.png"
    if p.exists(): p.replace(IM / f"{name}.png")
    print("wrote", name)

def go_(name): return not want or name in want

if go_("01-sector-heatmap"):
    order = []
    for s in SECTORS:
        names = [n for n in corr.index if sec[n] == s]
        if len(names) > 2:
            names = [names[i] for i in leaves_list(linkage(1 - corr.loc[names, names].values[np.triu_indices(len(names), 1)], "average"))]
        order += names
    c = corr.loc[order, order]
    lab = [f"{n} ({sec[n]})" for n in order]
    fig = go.Figure(go.Heatmap(z=c.values.round(3), x=order, y=order, customdata=np.array([[sec[a] + " / " + sec[b] for b in order] for a in order]),
                               zmin=-0.2, zmax=1, colorscale=[[0, "#c96f33"], [0.17, "#f7f8fb"], [1, "#2f4cc2"]], colorbar=dict(title=dict(text="Pearson", side="right"), orientation="h", x=0.5, xanchor="center", y=-0.01, yanchor="top", len=0.7, thickness=10),
                               hovertemplate="%{y} / %{x}<br>%{customdata}<br>correlation %{z:.2f}<extra></extra>"))
    bounds = np.cumsum([sum(sec[n] == s for n in order) for s in SECTORS])[:-1] - 0.5
    shapes = [dict(type="line", x0=b, x1=b, y0=-0.5, y1=len(order) - 0.5, line=dict(color="rgba(20,32,51,.35)", width=1)) for b in bounds]
    shapes += [dict(type="line", y0=b, y1=b, x0=-0.5, x1=len(order) - 0.5, line=dict(color="rgba(20,32,51,.35)", width=1)) for b in bounds]
    sizes = [sum(sec[n] == s for n in order) for s in SECTORS]; mids = np.cumsum(sizes) - np.array(sizes) / 2 - 0.5
    ann = [dict(x=m, y=1.0, xref="x", yref="paper", text=s.replace("Information Technology", "Info. Tech.").replace("Consumer Discretionary", "Cons. Disc.").replace("Consumer Staples", "Cons. Staples").replace("Communication Services", "Comm."), textangle=-60, showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=11, color=MUTED))
           for m, s, z in zip(mids, SECTORS, sizes) if z >= 7]
    fig.update_layout(**BASE, shapes=shapes, annotations=ann, height=760, margin=dict(l=10, r=10, t=90, b=70), xaxis=dict(visible=False), yaxis=dict(visible=False, autorange="reversed", scaleanchor="x"))
    out(fig, "01-sector-heatmap", (1100, 1150))

def layout(k, g):
    """MST/PMFG: Kamada-Kawai on correlation distance. Fixed density (many components): giant component laid out on its own, small components on a grid below."""
    if k != "Fixed density":
        return nx.kamada_kawai_layout(g, weight="distance")
    comps = sorted(nx.connected_components(g), key=len, reverse=True)
    pos = nx.spring_layout(g.subgraph(comps[0]), k=0.45, iterations=400, seed=5)
    cols, step = 16, 0.19
    for i, comp in enumerate(comps[1:]):
        cx, cy = -1.45 + (i % cols) * step * 1.0 + 0.1, -1.5 - (i // cols) * 0.28
        sub = g.subgraph(comp)
        loc = nx.circular_layout(sub, scale=0.07) if len(comp) > 1 else {n: (0, 0) for n in comp}
        for n, (x, y) in loc.items(): pos[n] = (cx + x, cy + y)
    return pos

GRAPHS = {}
def graphs():
    if not GRAPHS:
        GRAPHS["MST"] = fl.mst_from_corr(corr); GRAPHS["PMFG"] = fl.pmfg(corr); GRAPHS["Fixed density"] = fl.fixed_density_graph(corr, 3 * (len(corr) - 2))
        rows = [(k, u, v, round(d["rho"] if "rho" in d else 1 - d["distance"] ** 2 / 2, 6)) for k, g in GRAPHS.items() for u, v, d in g.edges(data=True)]
        pd.DataFrame(rows, columns=["graph", "u", "v", "rho"]).to_csv(R / "headline_graph_edges.csv", index=False)
    return GRAPHS

if go_("02-three-graphs"):
    G = graphs()
    fig = go.Figure(); groups = []
    for gi, (k, g) in enumerate(G.items()):
        pos = layout(k, g)
        ex, ey = [], []
        for u, v in g.edges: ex += [pos[u][0], pos[v][0], None]; ey += [pos[u][1], pos[v][1], None]
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(93,102,120,.45)", width=.8), hoverinfo="skip", showlegend=False, visible=gi == 0))
        n_tr = 1
        for s in SECTORS:
            nodes = [n for n in sorted(g.nodes) if sec[n] == s]
            fig.add_trace(go.Scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers", name=s, legendgroup=s, showlegend=gi == 0, visible=gi == 0,
                                     text=nodes, customdata=[[g.degree(n)] for n in nodes], marker=dict(size=[min(5 + 1.2 * g.degree(n), 24) for n in nodes], color=COL[s], line=dict(width=.8, color="white")),
                                     hovertemplate="<b>%{text}</b><br>" + s + "<br>degree %{customdata[0]}<extra></extra>"))
            n_tr += 1
        groups.append(n_tr)
    buttons, start = [], 0
    for gi, (k, g) in enumerate(G.items()):
        vis = [False] * sum(groups); vis[start:start + groups[gi]] = [True] * groups[gi]; start += groups[gi]
        comps = nx.number_connected_components(g)
        buttons.append(dict(label=k, method="update", args=[{"visible": vis}]))
    fig.update_layout(**BASE, height=640, margin=dict(l=8, r=8, t=44, b=8), xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                      legend=dict(orientation="h", x=0, y=-0.01, font=dict(size=11)), hovermode="closest",
                      updatemenus=[dict(type="buttons", direction="right", x=0, y=1.08, xanchor="left", buttons=buttons, bgcolor="#f1f3f9", font=dict(size=12))])
    out(fig, "02-three-graphs", (1100, 1000), png=False)
    # static panel version for the thumbnail/print
    fig2 = make_subplots(rows=1, cols=3, subplot_titles=[f"{k}: {g.number_of_edges()} edges" for k, g in G.items()], horizontal_spacing=.02)
    for ci, (k, g) in enumerate(G.items(), 1):
        pos = layout(k, g)
        ex, ey = [], []
        for u, v in g.edges: ex += [pos[u][0], pos[v][0], None]; ey += [pos[u][1], pos[v][1], None]
        fig2.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(93,102,120,.45)", width=.7), hoverinfo="skip", showlegend=False), row=1, col=ci)
        for s in SECTORS:
            nodes = [n for n in sorted(g.nodes) if sec[n] == s]
            fig2.add_trace(go.Scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers", name=s, legendgroup=s, showlegend=ci == 1, text=nodes,
                                      marker=dict(size=[min(4 + 1.0 * g.degree(n), 20) for n in nodes], color=COL[s], line=dict(width=.6, color="white")), hovertemplate="%{text}<extra>" + s + "</extra>"), row=1, col=ci)
    fig2.update_xaxes(visible=False); fig2.update_yaxes(visible=False)
    fig2.update_layout(**BASE, height=520, margin=dict(l=6, r=6, t=40, b=70), legend=dict(orientation="h", x=0, y=-0.02, font=dict(size=12)))
    for a in fig2.layout.annotations: a.update(font=dict(size=14, color=INK))
    viz.export(fig2, CH, "02b-three-graphs-panels", size=(1500, 700)); (CH / "02b-three-graphs-panels.png").replace(IM / "02b-three-graphs-panels.png"); (CH / "02b-three-graphs-panels.json").unlink(missing_ok=True)

if go_("03-rolling-structure"):
    r = pd.read_csv(R / "rolling_structure_monthly.csv", parse_dates=["date"]).set_index("date")
    cols = {"mean_corr": "Average pairwise correlation", "same_sector_edge_share": "Same-sector share of MST edges", "jaccard_prev": "MST overlap with previous month"}
    fig = viz.fig_metric_lines(r, cols)
    out(fig, "03-rolling-structure", (1100, 800))

if go_("04-estimator-overlap"):
    a = pd.read_csv(R / "estimator_pairwise_correlation_agreement.csv", index_col=0); b = pd.read_csv(R / "estimator_edge_jaccard_mst.csv", index_col=0)
    nm = {"pearson": "Pearson", "spearman": "Spearman", "kendall": "Kendall", "pearson_market_residual": "No-market"}
    fig = make_subplots(rows=2, cols=1, subplot_titles=["Pairwise values: correlation", "MST edges in common (Jaccard)"], vertical_spacing=.14)
    for ri, (m, zmin) in enumerate([(a, .6), (b, 0)], 1):
        fig.add_trace(go.Heatmap(z=m.values.round(3), x=[nm[c] for c in m.columns], y=[nm[c] for c in m.index], zmin=zmin, zmax=1, colorscale=[[0, "#f1f3f9"], [1, "#3758d7"]], showscale=False,
                                 text=m.values.round(2), texttemplate="%{text}", hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>"), row=ri, col=1)
    fig.update_yaxes(autorange="reversed"); fig.update_xaxes(tickangle=0, tickfont=dict(size=11))
    fig.update_layout(**BASE, height=720, margin=dict(l=76, r=10, t=36, b=30))
    for an in fig.layout.annotations: an.update(font=dict(size=13, color=MUTED), x=0, xanchor="left")
    out(fig, "04-estimator-overlap", (1100, 1000))

if go_("05-persistence") :
    p = pd.read_csv(R / "mst_persistence_by_gap.csv")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p.gap_months, y=p.p90, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=p.gap_months, y=p.p10, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(55,88,215,.15)", name="10th to 90th percentile", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=p.gap_months, y=p["median"], mode="lines+markers", line=dict(color=PALETTE[0], width=2.4), marker=dict(size=8, color=[PALETTE[0] if o else PALETTE[1] for o in p.overlapping]),
                             name="Median", customdata=np.c_[p.n, p.overlapping], hovertemplate="%{x} months apart<br>median Jaccard %{y:.2f}<br>%{customdata[0]} window pairs<br>windows overlap: %{customdata[1]}<extra></extra>"))
    bt_ = pd.read_csv(R / "bootstrap_vs_temporal.csv").set_index("window_end").loc["2024-12-30"]
    fig.add_trace(go.Scatter(x=[1, 60], y=[bt_.bootstrap_pair_noise_floor_median] * 2, mode="lines", line=dict(color=PALETTE[1], width=2, dash="dash"), name="Same-window bootstrap overlap (reference)",
                             hovertemplate="Two bootstrap resamples of one 252-day window (reference, not a null): median Jaccard %{y:.2f}<extra></extra>"))
    fig.update_layout(**BASE, height=460, margin=dict(l=56, r=16, t=16, b=50), xaxis=dict(title="Months between the two windows", type="log", tickvals=[1, 3, 6, 12, 24, 60], ticktext=["1", "3", "6", "12", "24", "60"], **AX),
                      yaxis=dict(title="MST edge overlap (Jaccard index)", range=[0, 0.8], **AX), legend=dict(orientation="h", x=0, y=1.22, font=dict(size=11)))
    out(fig, "05-persistence", (1100, 560))

if go_("06-edge-reliability"):
    fig = make_subplots(rows=2, cols=1, subplot_titles=["Minimum spanning tree (202 edges)", "Fixed-density graph (603 edges)"], vertical_spacing=.2)
    for ci, f in enumerate(["mst", "fixed"], 1):
        d = pd.read_csv(R / f"bootstrap_edge_inclusion_{f}.csv")
        for lab, mask, col in [("Same-sector edges", d.same_sector, PALETTE[0]), ("Cross-sector edges", ~d.same_sector, PALETTE[1])]:
            fig.add_trace(go.Histogram(x=d.inclusion[mask], xbins=dict(start=-0.05, end=1.05, size=.1), name=lab, marker=dict(color=col, line=dict(color="white", width=1)), opacity=.85, legendgroup=lab,
                                       showlegend=ci == 1, hovertemplate="inclusion near %{x:.1f}<br>%{y} edges<extra>" + lab + "</extra>"), row=ci, col=1)
    fig.update_layout(**BASE, barmode="stack", height=720, margin=dict(l=56, r=16, t=36, b=110), legend=dict(orientation="h", x=0, y=-0.19))
    fig.update_xaxes(title="Share of resamples with the edge", range=[-0.05, 1.05], **AX); fig.update_yaxes(title="Edges", **AX)
    for a in fig.layout.annotations: a.update(x=0, xanchor="left")
    for a in fig.layout.annotations: a.update(font=dict(size=13, color=MUTED))
    out(fig, "06-edge-reliability", (1100, 800))

if go_("07-window-and-recoverability"):
    syn = pd.read_csv(R / "synthetic_recoverability.csv")
    fig = make_subplots(rows=2, cols=1, subplot_titles=["Real data: MST edge overlap (Jaccard index)", "SYNTHETIC: planted blocks recovered (ARI)"], vertical_spacing=.2)
    for i, y in enumerate(["2008", "2017", "2024"]):
        m = pd.read_csv(R / f"window_length_jaccard_{y}.csv", index_col=0)
        pairs = [("126 vs 252", m.loc[126, "252"]), ("252 vs 504", m.loc[252, "504"]), ("126 vs 504", m.loc[126, "504"])]
        fig.add_trace(go.Bar(x=[p[0] for p in pairs], y=[p[1] for p in pairs], name=f"Window ending {y}", marker=dict(color=PALETTE[[1, 2, 0][i]]), hovertemplate="%{x} days<br>Jaccard %{y:.2f}<extra>" + y + "</extra>"), row=1, col=1)
    for i, (s, g) in enumerate(syn.groupby("setting")):
        fig.add_trace(go.Scatter(x=g["T"], y=g.ari, mode="lines+markers", name=s.split(" (")[0], line=dict(color=PALETTE[3 + i], width=2.2, dash="dot"),
                                 hovertemplate="T = %{x} days<br>ARI %{y:.2f}<extra>synthetic " + s.split(" (")[0] + "</extra>"), row=2, col=1)
    fig.update_xaxes(title="Window lengths compared (days)", row=1, col=1, **AX); fig.update_yaxes(title="Edge overlap", range=[0, .7], row=1, col=1, **AX)
    fig.update_xaxes(title="Observations T per stock (days)", type="log", tickvals=[63, 126, 252, 504, 1008, 2016], row=2, col=1, **AX); fig.update_yaxes(title="Adjusted Rand index", range=[0, 1], row=2, col=1, **AX)
    fig.update_layout(**BASE, barmode="group", height=820, margin=dict(l=56, r=16, t=36, b=120), legend=dict(orientation="h", x=0, y=-0.14, font=dict(size=11)))
    for a in fig.layout.annotations: a.update(font=dict(size=13, color=MUTED), x=0, xanchor="left")
    out(fig, "07-window-and-recoverability", (1100, 900))

if go_("08-steiner"):
    e = pd.read_csv(R / "steiner_primary_tree_edges.csv"); e = e[e.window_end == "2024-12-30"]
    TERMS = ["AAPL", "JPM", "XOM", "JNJ", "PG", "CAT", "HD", "SO"]
    order = ["MST subtree (exact, inside the MST)", "Steiner on PMFG (exact)", "Steiner on full distance graph (exact)", "Terminals only (MST of the 8 stocks, no intermediates)"]
    short = {order[0]: "MST subtree", order[1]: "Steiner on PMFG", order[2]: "Steiner on full graph", order[3]: "Terminals only"}
    btn = {order[0]: "MST", order[1]: "PMFG", order[2]: "Full graph", order[3]: "Terminals"}
    ug = nx.Graph()
    for _, r in e.iterrows(): ug.add_edge(r.u, r.v, weight=1 / max(r.distance, 1e-6))
    pos = nx.spring_layout(ug, weight="weight", seed=11, k=0.9, iterations=300)
    costs = {t: e[e.tree == t].distance.sum() for t in order}
    def traces(t, visible, inter_labels=True):
        g = e[e.tree == t]; ex, ey = [], []
        for _, r in g.iterrows(): ex += [pos[r.u][0], pos[r.v][0], None]; ey += [pos[r.u][1], pos[r.v][1], None]
        nodes = sorted(set(g.u) | set(g.v)); inter = [n for n in nodes if n not in TERMS]
        out_ = [go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(93,102,120,.7)", width=2), hoverinfo="skip", showlegend=False, visible=visible)]
        out_.append(go.Scatter(x=[pos[n][0] for n in TERMS], y=[pos[n][1] for n in TERMS], mode="markers+text", text=TERMS, textposition="top center", textfont=dict(size=12, color=INK), name="Terminal stocks",
                               marker=dict(size=16, symbol="diamond", color=[COL[sec[n]] for n in TERMS], line=dict(width=1.5, color=INK)), showlegend=False, visible=visible,
                               customdata=[[sec[n]] for n in TERMS], hovertemplate="<b>%{text}</b> (terminal)<br>%{customdata[0]}<extra></extra>"))
        out_.append(go.Scatter(x=[pos[n][0] for n in inter], y=[pos[n][1] for n in inter], mode="markers+text" if inter_labels else "markers", text=inter, textposition="bottom center", textfont=dict(size=11, color=MUTED), name="Intermediate stocks",
                               marker=dict(size=11, color=[COL[sec[n]] for n in inter], line=dict(width=1, color="white")), showlegend=False, visible=visible,
                               customdata=[[sec[n]] for n in inter], hovertemplate="<b>%{text}</b> (intermediate)<br>%{customdata[0]}<extra></extra>"))
        return out_
    fig = go.Figure(); buttons = []
    for i, t in enumerate(order):
        for tr in traces(t, i == 0): fig.add_trace(tr)
    for i, t in enumerate(order):
        vis = [False] * (3 * len(order)); vis[3 * i:3 * i + 3] = [True] * 3
        n_int = len(set(e[e.tree == t].u) | set(e[e.tree == t].v)) - 8
        buttons.append(dict(label=btn[t], method="update", args=[{"visible": vis}, {"title": dict(text=f"{short[t]}: cost {costs[t]:.2f}, {n_int} intermediates", x=0, font=dict(size=13, color=MUTED))}]))
    t0_ = order[0]; n0 = len(set(e[e.tree == t0_].u) | set(e[e.tree == t0_].v)) - 8
    fig.update_layout(**BASE, height=560, margin=dict(l=8, r=8, t=90, b=8), xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"), hovermode="closest",
                      title=dict(text=f"{short[t0_]}: cost {costs[t0_]:.2f}, {n0} intermediates", x=0, font=dict(size=13, color=MUTED)),
                      updatemenus=[dict(type="buttons", direction="right", x=0, y=1.0, xanchor="left", yanchor="bottom", buttons=buttons, bgcolor="#f1f3f9", font=dict(size=11))])
    out(fig, "08-steiner", (1100, 900), png=False)
    fig2 = make_subplots(rows=2, cols=2, subplot_titles=[f"{short[t]}: cost {costs[t]:.2f}" for t in order], horizontal_spacing=.03, vertical_spacing=.1)
    for i, t in enumerate(order):
        for tr in traces(t, True, inter_labels=False): fig2.add_trace(tr, row=i // 2 + 1, col=i % 2 + 1)   # static fallback: no intermediate labels (they overlap); names are in the interactive version
    fig2.update_xaxes(visible=False); fig2.update_yaxes(visible=False)
    fig2.update_layout(**BASE, height=900, margin=dict(l=6, r=6, t=40, b=6), showlegend=False)
    for a in fig2.layout.annotations: a.update(font=dict(size=14, color=INK))
    viz.export(fig2, CH, "08b-steiner-panels", size=(1500, 900)); (CH / "08b-steiner-panels.png").replace(IM / "08b-steiner-panels.png"); (CH / "08b-steiner-panels.json").unlink(missing_ok=True)
