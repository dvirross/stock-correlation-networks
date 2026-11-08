"""Plotly figure builders (candidate article figures) with JSON-spec and static-PNG export.

Builders only draw what they are given. They never create data. Percentages shown to the reader use two decimals.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

INK, MUTED, GRID, AXIS = "#142033", "#5d6678", "#e3e6ee", "#b9bfcc"
PALETTE = ["#3758d7", "#c96f33", "#1f8a8a", "#7a4fb3", "#b8860b", "#c0392b", "#5d6678", "#2e8b57", "#d35d9f", "#6b7a2a", "#0f7ba8"]
FONT = dict(family="Inter, system-ui, -apple-system, 'Segoe UI', sans-serif", size=13, color=INK)
BASE = dict(template="none", paper_bgcolor="white", plot_bgcolor="white", font=FONT)


def fig_mst(g: nx.Graph, groups: dict[str, str] | None = None, title_note: str = "") -> go.Figure:
    """Minimum spanning tree drawn with a deterministic Kamada-Kawai layout; nodes coloured by group (e.g. sector)."""
    pos = nx.kamada_kawai_layout(g, weight="distance") if g.number_of_nodes() > 2 else nx.circular_layout(g)
    ex, ey = [], []
    for u, v in g.edges:
        ex += [pos[u][0], pos[v][0], None]
        ey += [pos[u][1], pos[v][1], None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=AXIS, width=1), hoverinfo="skip", showlegend=False))
    labels = sorted(set((groups or {}).get(n, "all") for n in g.nodes))
    for i, lab in enumerate(labels):
        nodes = [n for n in sorted(g.nodes) if (groups or {}).get(n, "all") == lab]
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text", name=lab, text=nodes, textposition="top center",
            textfont=dict(size=9, color=MUTED), marker=dict(size=[8 + 3 * g.degree(n) for n in nodes], color=PALETTE[i % len(PALETTE)], line=dict(width=1, color="white")),
            customdata=[[g.degree(n)] for n in nodes], hovertemplate="<b>%{text}</b><br>" + lab + "<br>MST degree: %{customdata[0]}<extra></extra>"))
    fig.update_layout(**BASE, height=620, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                      legend=dict(orientation="h", x=0, y=-0.02, font=dict(size=11)), hovermode="closest")
    return fig


def fig_metric_lines(metrics: pd.DataFrame, columns: dict[str, str]) -> go.Figure:
    """One panel per requested metric (column -> axis title), sharing the date axis."""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=len(columns), cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=list(columns.values()))
    for i, (col, title) in enumerate(columns.items(), start=1):
        fig.add_trace(go.Scatter(x=metrics.index, y=metrics[col], mode="lines", line=dict(color=PALETTE[(i - 1) % len(PALETTE)], width=2), showlegend=False,
                                 hovertemplate="%{x|%b %d, %Y}<br>" + title + ": %{y:.3f}<extra></extra>"), row=i, col=1)
    fig.update_layout(**BASE, height=220 * len(columns) + 60, margin=dict(l=56, r=16, t=40, b=40), hovermode="x")
    fig.update_xaxes(gridcolor=GRID, showgrid=False, linecolor=AXIS)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    for a in fig.layout.annotations:
        a.update(x=0, xanchor="left", font=dict(size=13, color=MUTED))
    return fig


def fig_stability_heatmap(sim: pd.DataFrame, title: str = "Edge persistence between windows (Jaccard)") -> go.Figure:
    fig = go.Figure(go.Heatmap(z=sim.values, x=sim.columns, y=sim.index, zmin=0, zmax=1, colorscale=[[0, "#f1f3f9"], [1, "#3758d7"]],
                               colorbar=dict(title="Jaccard", thickness=10), hovertemplate="%{y|%b %Y} vs %{x|%b %Y}<br>Jaccard: %{z:.2f}<extra></extra>"))
    fig.update_layout(**BASE, height=520, margin=dict(l=64, r=16, t=16, b=56), yaxis=dict(autorange="reversed"), title=dict(text=title, font=dict(size=13, color=MUTED), x=0))
    return fig


def fig_equity_curves(net: pd.DataFrame, log_scale: bool = True) -> go.Figure:
    """Growth of 1 for each strategy (net of costs) from the daily return columns."""
    fig = go.Figure()
    for i, col in enumerate(net.columns):
        w = (1 + net[col].dropna()).cumprod()
        fig.add_trace(go.Scatter(x=w.index, y=w.values, name=col, mode="lines", line=dict(color=PALETTE[i % len(PALETTE)], width=2.2),
                                 hovertemplate="<b>" + col + "</b><br>%{x|%b %d, %Y}<br>Growth of 1: %{y:.2f}<extra></extra>"))
    fig.update_layout(**BASE, height=480, margin=dict(l=56, r=16, t=16, b=50), hovermode="x unified", legend=dict(orientation="h", x=0, y=1.1),
                      xaxis=dict(showgrid=False, linecolor=AXIS), yaxis=dict(type="log" if log_scale else "linear", gridcolor=GRID, title="Growth of 1 (net of costs)"))
    return fig


def export(fig: go.Figure, out_dir: Path, name: str, size=(1100, 520), png: bool = True) -> None:
    """Write ``<name>.json`` (Plotly spec) and, when kaleido and a Chrome/Chromium are available, a static ``<name>.png`` fallback."""
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(pio.to_json(fig))
    spec.get("layout", {}).pop("template", None)
    (out_dir / f"{name}.json").write_text(json.dumps(spec, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
    if png:
        try:
            pio.write_image(fig, str(out_dir / f"{name}.png"), width=size[0], height=size[1], scale=2)
        except Exception as exc:  # environment dependent
            print(f"Static PNG not written for {name} ({type(exc).__name__}); JSON spec written.")
