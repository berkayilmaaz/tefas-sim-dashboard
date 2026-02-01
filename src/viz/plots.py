# src/viz/plots.py
from __future__ import annotations

import pandas as pd
import plotly.express as px


def plot_equity(equity: pd.Series):
    df = equity.reset_index()
    df.columns = ["date", "equity"]
    fig = px.line(df, x="date", y="equity", title="Equity Curve")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig
