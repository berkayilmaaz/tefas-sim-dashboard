# src/viz/plots.py
from __future__ import annotations

import pandas as pd
import plotly.express as px


def plot_equity(equity: pd.Series, benchmark: pd.Series | None = None):
    frames = []
    df = equity.reset_index()
    df.columns = ["date", "equity"]
    df["series"] = "Strateji"
    frames.append(df)
    if benchmark is not None and not benchmark.empty:
        bdf = benchmark.reset_index()
        bdf.columns = ["date", "equity"]
        bdf["series"] = "Kıyas"
        frames.append(bdf)
    plot_df = pd.concat(frames, ignore_index=True)
    fig = px.line(plot_df, x="date", y="equity", color="series", title="Equity Eğrisi")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_drawdown(drawdown: pd.Series, benchmark: pd.Series | None = None):
    frames = []
    df = drawdown.reset_index()
    df.columns = ["date", "drawdown"]
    df["series"] = "Strateji"
    frames.append(df)
    if benchmark is not None and not benchmark.empty:
        bdf = benchmark.reset_index()
        bdf.columns = ["date", "drawdown"]
        bdf["series"] = "Kıyas"
        frames.append(bdf)
    plot_df = pd.concat(frames, ignore_index=True)
    fig = px.line(
        plot_df, x="date", y="drawdown", color="series", title="Düşüş (Drawdown)"
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_rolling_metrics(rolling_df: pd.DataFrame, benchmark_df: pd.DataFrame | None):
    frames = []
    df = rolling_df.reset_index()
    df.columns = ["date", "Yuvarlanan Volatilite", "Yuvarlanan Sharpe", "Yuvarlanan MDD"]
    df["series"] = "Strateji"
    frames.append(df)
    if benchmark_df is not None and not benchmark_df.empty:
        bdf = benchmark_df.reset_index()
        bdf.columns = [
            "date",
            "Yuvarlanan Volatilite",
            "Yuvarlanan Sharpe",
            "Yuvarlanan MDD",
        ]
        bdf["series"] = "Kıyas"
        frames.append(bdf)
    plot_df = pd.concat(frames, ignore_index=True)
    fig = px.line(
        plot_df.melt(id_vars=["date", "series"], var_name="metric", value_name="value"),
        x="date",
        y="value",
        color="series",
        facet_row="metric",
        title="Yuvarlanan Metrikler",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame):
    fig = px.imshow(
        corr_matrix,
        text_auto=False,
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Korelasyon Isı Haritası",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_return_histogram(returns_long: pd.DataFrame):
    fig = px.histogram(
        returns_long,
        x="ret",
        color="fund_code",
        barmode="overlay",
        nbins=40,
        title="Günlük Getiri Dağılımı",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_equity_comparison(equity_long: pd.DataFrame):
    fig = px.line(
        equity_long,
        x="date",
        y="equity",
        color="fund_code",
        title="Fon Karşılaştırma Equity Eğrisi",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig
