import streamlit as st
import pandas as pd

from src.types import FetchParams, StrategyParams
from src.data.provider_tefaslib import TefasLibProvider
from src.domain.returns import compute_fund_daily_returns
from src.domain.strategies import equal_weight_portfolio
from src.domain.stats import (
    equity_curve,
    total_return,
    annualized_vol,
    max_drawdown,
    sharpe,
)
from src.viz.plots import plot_equity

st.set_page_config(page_title="TEFAS Strategy Simulator", layout="wide")
st.title("TEFAS Strategy Simulator (Sprint 2)")

# Sidebar inputs
st.sidebar.header("Inputs")

start = st.sidebar.text_input("Start (dd.mm.yyyy)", value="01.01.2025")
end = st.sidebar.text_input("End (dd.mm.yyyy)", value="15.01.2025")

k = st.sidebar.number_input(
    "K (number of funds)", min_value=10, max_value=2000, value=200, step=10
)
seed = st.sidebar.number_input(
    "Random seed", min_value=0, max_value=10_000, value=42, step=1
)

params = FetchParams(
    start=start, end=end, universe="free", fund_type_code=4, tab_code=0
)
sp = StrategyParams(k=int(k), seed=int(seed))


provider = TefasLibProvider()
st.caption(f"Provider: {provider.name}")

try:
    prices = provider.fetch_prices(params)
except Exception as e:
    st.error("Data fetch failed.")
    st.code(str(e))
    st.stop()

st.subheader("Prices (normalized schema)")
st.write("prices shape:", prices.shape)
st.write("unique fund_code:", prices["fund_code"].nunique())
st.dataframe(prices.head(30), use_container_width=True)

# returns
rets = compute_fund_daily_returns(prices)

st.subheader("Daily Returns (fund-level)")
st.write("returns shape:", rets.shape)
st.dataframe(rets.head(30), use_container_width=True)

# portfolio
port_ret = equal_weight_portfolio(rets, sp)
eq = equity_curve(port_ret, start_value=1.0)

# metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Return", f"{total_return(eq)*100:.2f}%")
col2.metric("Vol (ann.)", f"{annualized_vol(port_ret)*100:.2f}%")
col3.metric("Max Drawdown", f"{max_drawdown(eq)*100:.2f}%")
col4.metric("Sharpe", f"{sharpe(port_ret):.2f}")

st.subheader("Equity Curve")
st.plotly_chart(plot_equity(eq), use_container_width=True)

st.subheader("Portfolio Daily Returns")
st.dataframe(
    port_ret.reset_index()
    .rename(columns={"date": "date", "portfolio_ret": "ret"})
    .head(50),
    use_container_width=True,
)
