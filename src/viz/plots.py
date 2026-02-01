# src/viz/plots.py
from __future__ import annotations

import pandas as pd
import plotly.express as px


def plot_equity(equity: pd.Series, benchmark: pd.Series | None = None):
    frames = []
    df = equity.reset_index()
    df.columns = ["date", "equity"]
    df["series"] = "Strategy"
    frames.append(df)
    if benchmark is not None and not benchmark.empty:
        bdf = benchmark.reset_index()
        bdf.columns = ["date", "equity"]
        bdf["series"] = "Benchmark"
        frames.append(bdf)
    plot_df = pd.concat(frames, ignore_index=True)
    fig = px.line(plot_df, x="date", y="equity", color="series", title="Equity Curve")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_drawdown(drawdown: pd.Series, benchmark: pd.Series | None = None):
    frames = []
    df = drawdown.reset_index()
    df.columns = ["date", "drawdown"]
    df["series"] = "Strategy"
    frames.append(df)
    if benchmark is not None and not benchmark.empty:
        bdf = benchmark.reset_index()
        bdf.columns = ["date", "drawdown"]
        bdf["series"] = "Benchmark"
        frames.append(bdf)
    plot_df = pd.concat(frames, ignore_index=True)
    fig = px.line(
        plot_df, x="date", y="drawdown", color="series", title="Drawdown"
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig
