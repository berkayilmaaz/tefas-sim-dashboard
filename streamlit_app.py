import pandas as pd
import streamlit as st

from src.data.provider_tefaslib import TefasLibProvider
from src.domain.returns import compute_fund_daily_returns
from src.domain.stats import (
    annualized_vol,
    drawdown_series,
    equity_curve,
    max_drawdown,
    sharpe,
    total_return,
)
from src.domain.strategies import (
    build_weights,
    compute_portfolio_returns,
    select_universe_k,
)
from src.types import FetchParams, StrategyParams
from src.viz.plots import plot_drawdown, plot_equity

st.set_page_config(page_title="TEFAS Strategy Simulator", layout="wide")


@st.cache_data(show_spinner=False)
def load_prices(provider_name: str, start: str, end: str, universe: str) -> pd.DataFrame:
    provider = TefasLibProvider()
    params = FetchParams(start=start, end=end, universe=universe)
    return provider.fetch_prices(params)


def parse_date(value: str) -> pd.Timestamp | None:
    try:
        parsed = pd.to_datetime(value, dayfirst=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    return parsed.normalize()


def summarize_funds(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_values(["fund_code", "date"])
    returns = compute_fund_daily_returns(prices)
    vol = returns.groupby("fund_code")["ret"].std(ddof=1)

    expected_days = prices["date"].nunique()

    grouped = prices.groupby("fund_code")
    summary = grouped.agg(
        fund_name=("fund_name", "last"),
        price_last=("price", "last"),
        start_price=("price", "first"),
        end_price=("price", "last"),
        aum_last=("aum", "last"),
        obs_days=("date", "nunique"),
    )
    summary["period_return"] = summary["end_price"] / summary["start_price"] - 1.0
    summary["vol_daily"] = vol
    summary["missing_days"] = expected_days - summary["obs_days"]
    summary = summary.reset_index()
    summary["aum_last"] = pd.to_numeric(summary["aum_last"], errors="coerce").fillna(0.0)
    summary["price_last"] = pd.to_numeric(summary["price_last"], errors="coerce")
    summary["start_price"] = pd.to_numeric(summary["start_price"], errors="coerce")
    summary["end_price"] = pd.to_numeric(summary["end_price"], errors="coerce")
    return summary


def select_funds(
    summary: pd.DataFrame,
    mode: str,
    k: int,
    seed: int,
    manual_codes: list[str],
) -> tuple[list[str], list[str]]:
    fund_codes = summary["fund_code"].tolist()
    missing_manual = []

    if mode == "Top-K by AUM":
        selected = (
            summary.sort_values("aum_last", ascending=False)
            .head(k)["fund_code"]
            .tolist()
        )
        return selected, missing_manual
    if mode == "Random K":
        selected = select_universe_k(fund_codes, k, seed)
        return selected, missing_manual
    if mode == "Manual":
        code_set = set(fund_codes)
        selected = []
        for code in manual_codes:
            if code in code_set:
                selected.append(code)
            else:
                missing_manual.append(code)
        return selected, missing_manual
    return fund_codes, missing_manual


def build_benchmark(
    prices: pd.DataFrame,
    benchmark_mode: str,
    strategy_params: StrategyParams,
) -> tuple[pd.Series, pd.DataFrame]:
    summary = summarize_funds(prices)
    if benchmark_mode == "Equal-weight (Top AUM 200)":
        chosen = (
            summary.sort_values("aum_last", ascending=False)
            .head(200)["fund_code"]
            .tolist()
        )
    else:
        chosen = summary["fund_code"].tolist()

    bench_prices = prices[prices["fund_code"].isin(chosen)]
    bench_returns = compute_fund_daily_returns(bench_prices)
    bench_wide = bench_returns.pivot_table(
        index="date", columns="fund_code", values="ret", aggfunc="mean"
    ).sort_index()
    weights_daily, weights_table = build_weights(bench_wide, strategy_params)
    bench_ret = compute_portfolio_returns(bench_wide, weights_daily)
    return bench_ret, weights_table


st.title("TEFAS Strategy Simulator (Sprint 3)")

# Sidebar inputs
st.sidebar.header("Inputs")

start_input = st.sidebar.text_input("Start (dd.mm.yyyy)", value="01.01.2025")
end_input = st.sidebar.text_input("End (dd.mm.yyyy)", value="15.01.2025")

provider_name = "tefasfon"
universe = st.sidebar.selectbox("Universe", options=["free"], index=0)

k = st.sidebar.slider("K (number of funds)", min_value=10, max_value=500, value=200)
selection_mode = st.sidebar.radio(
    "Selection mode",
    options=["Top-K by AUM", "Random K", "Manual"],
    index=0,
)

seed = 42
if selection_mode == "Random K":
    seed = st.sidebar.number_input(
        "Random seed", min_value=0, max_value=10_000, value=42, step=1
    )

manual_codes: list[str] = []
if selection_mode == "Manual":
    raw_codes = st.sidebar.text_area("Fund codes (comma separated)")
    manual_codes = [
        code.strip().upper() for code in raw_codes.replace("\n", ",").split(",")
        if code.strip()
    ]
    if manual_codes:
        k = len(manual_codes)

strategy_label = st.sidebar.selectbox(
    "Strategy",
    options=[
        "Equal Weight",
        "Momentum Top-K",
        "Low Vol Top-K",
        "Sharpe Top-K",
    ],
    index=0,
)

strategy_map = {
    "Equal Weight": "equal_weight",
    "Momentum Top-K": "momentum_top_k",
    "Low Vol Top-K": "low_vol_top_k",
    "Sharpe Top-K": "sharpe_top_k",
}

strategy_key = strategy_map[strategy_label]
lookback = 20
if strategy_key != "equal_weight":
    lookback = st.sidebar.slider(
        "Lookback (days)", min_value=5, max_value=252, value=20
    )

rebalance = st.sidebar.selectbox(
    "Rebalance frequency", options=["daily", "weekly", "monthly"], index=2
)

show_benchmark = st.sidebar.checkbox("Show benchmark", value=True)
benchmark_mode = st.sidebar.selectbox(
    "Benchmark",
    options=["Equal-weight (Top AUM 200)", "Equal-weight (All selected universe)"],
    index=0,
)

debug_mode = st.sidebar.checkbox("Debug mode: show raw tables", value=False)
if st.sidebar.button("Clear cache"):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared")

start_date = parse_date(start_input)
end_date = parse_date(end_input)
if start_date is None or end_date is None:
    st.error("Invalid date input. Use dd.mm.yyyy format.")
    st.stop()
if start_date > end_date:
    st.error("Start date must be earlier than end date.")
    st.stop()

st.caption(
    f"Provider: {provider_name} | Universe: {universe} | Date range: {start_input} – {end_input}"
)

try:
    prices = load_prices(provider_name, start_input, end_input, universe)
except Exception as exc:
    st.error("Data fetch failed.")
    st.code(str(exc))
    st.stop()

if prices.empty:
    st.error("No data in selected date range.")
    st.stop()

summary_all = summarize_funds(prices)
selected_funds, missing_manual = select_funds(
    summary_all, selection_mode, int(k), int(seed), manual_codes
)

if missing_manual:
    st.warning(f"Some fund codes not found: {', '.join(missing_manual)}")

if not selected_funds:
    st.error("No funds selected. Adjust selection mode or inputs.")
    st.stop()

selected_prices = prices[prices["fund_code"].isin(selected_funds)]

summary_selected = summary_all[summary_all["fund_code"].isin(selected_funds)].copy()
if selection_mode == "Top-K by AUM":
    summary_selected = summary_selected.sort_values("aum_last", ascending=False)
elif selection_mode == "Random K":
    summary_selected = summary_selected.sort_values("fund_code")
elif selection_mode == "Manual":
    order = {code: idx for idx, code in enumerate(selected_funds)}
    summary_selected["_order"] = summary_selected["fund_code"].map(order)
    summary_selected = summary_selected.sort_values("_order").drop(columns=["_order"])

returns_long = compute_fund_daily_returns(selected_prices)
returns_wide = returns_long.pivot_table(
    index="date", columns="fund_code", values="ret", aggfunc="mean"
).sort_index()

if strategy_key != "equal_weight" and lookback >= len(returns_wide.index):
    st.warning("Not enough lookback days. Increase date range or reduce lookback.")
    st.stop()

strategy_params = StrategyParams(
    k=int(k),
    seed=int(seed),
    strategy=strategy_key,
    lookback=int(lookback),
    rebalance=rebalance,
)

weights_daily, weights_table = build_weights(returns_wide, strategy_params)
if weights_daily.empty:
    st.warning("Not enough lookback days. Increase date range or reduce lookback.")
    st.stop()

strategy_returns = compute_portfolio_returns(returns_wide, weights_daily)
strategy_equity = equity_curve(strategy_returns, start_value=1.0)
strategy_drawdown = drawdown_series(strategy_equity)

benchmark_returns = pd.Series(dtype=float)
benchmark_equity = pd.Series(dtype=float)
benchmark_drawdown = pd.Series(dtype=float)
if show_benchmark:
    benchmark_returns, _ = build_benchmark(prices, benchmark_mode, strategy_params)
    if not benchmark_returns.empty:
        benchmark_equity = equity_curve(benchmark_returns, start_value=1.0)
        benchmark_drawdown = drawdown_series(benchmark_equity)

st.subheader("Funds Summary (1 row per fund)")
show_all = st.checkbox("Show all funds", value=False)
summary_display = summary_selected.copy()
summary_display = summary_display[
    [
        "fund_code",
        "fund_name",
        "aum_last",
        "price_last",
        "start_price",
        "end_price",
        "period_return",
        "vol_daily",
        "missing_days",
        "obs_days",
    ]
]
if not show_all:
    summary_display = summary_display.head(50)

st.dataframe(summary_display, use_container_width=True)

st.subheader("Performance (Strategy vs Benchmark)")

str_cols = st.columns(4)
str_cols[0].metric("Total Return", f"{total_return(strategy_equity) * 100:.2f}%")
str_cols[1].metric("Vol (ann.)", f"{annualized_vol(strategy_returns) * 100:.2f}%")
str_cols[2].metric("Max Drawdown", f"{max_drawdown(strategy_equity) * 100:.2f}%")
str_cols[3].metric("Sharpe", f"{sharpe(strategy_returns):.2f}")

if show_benchmark and not benchmark_returns.empty:
    bench_cols = st.columns(4)
    bench_cols[0].metric(
        "Benchmark Total Return", f"{total_return(benchmark_equity) * 100:.2f}%"
    )
    bench_cols[1].metric(
        "Benchmark Vol (ann.)", f"{annualized_vol(benchmark_returns) * 100:.2f}%"
    )
    bench_cols[2].metric(
        "Benchmark Max Drawdown", f"{max_drawdown(benchmark_equity) * 100:.2f}%"
    )
    bench_cols[3].metric("Benchmark Sharpe", f"{sharpe(benchmark_returns):.2f}")

st.subheader("Equity Curve")
st.caption(f"Rebalance frequency: {rebalance} | Lookback: {lookback} days")
st.plotly_chart(
    plot_equity(strategy_equity, benchmark_equity if show_benchmark else None),
    use_container_width=True,
)

st.subheader("Drawdown")
st.plotly_chart(
    plot_drawdown(strategy_drawdown, benchmark_drawdown if show_benchmark else None),
    use_container_width=True,
)

st.subheader("Portfolio Daily Returns")
show_full_returns = st.checkbox("Show full table", value=False)
returns_table = strategy_returns.reset_index().rename(
    columns={"date": "date", "portfolio_ret": "ret"}
)
if not show_full_returns:
    returns_table = returns_table.tail(30)
st.dataframe(returns_table, use_container_width=True)

st.subheader("Weights (rebalance dates)")
if weights_table.empty:
    st.write("No rebalance weights available.")
else:
    st.dataframe(weights_table, use_container_width=True)

st.subheader("Download")

summary_csv = summary_selected.to_csv(index=False).encode("utf-8")
returns_csv = (
    strategy_returns.reset_index()
    .rename(columns={"portfolio_ret": "ret"})
    .assign(
        equity=strategy_equity.values,
        drawdown=strategy_drawdown.values,
    )
    .to_csv(index=False)
    .encode("utf-8")
)

st.download_button(
    "Download funds_summary.csv",
    data=summary_csv,
    file_name="funds_summary.csv",
    mime="text/csv",
)

st.download_button(
    "Download strategy_returns.csv",
    data=returns_csv,
    file_name="strategy_returns.csv",
    mime="text/csv",
)

if show_benchmark and not benchmark_returns.empty:
    bench_csv = (
        benchmark_returns.reset_index()
        .rename(columns={"portfolio_ret": "ret"})
        .assign(
            equity=benchmark_equity.values,
            drawdown=benchmark_drawdown.values,
        )
        .to_csv(index=False)
        .encode("utf-8")
    )
    st.download_button(
        "Download benchmark_returns.csv",
        data=bench_csv,
        file_name="benchmark_returns.csv",
        mime="text/csv",
    )

if not weights_table.empty:
    weights_csv = weights_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download weights.csv",
        data=weights_csv,
        file_name="weights.csv",
        mime="text/csv",
    )

if debug_mode:
    st.subheader("Debug (raw data)")
    st.write("Raw prices (head)")
    st.dataframe(prices.head(200), use_container_width=True)
    st.write("Selected prices (head)")
    st.dataframe(selected_prices.head(200), use_container_width=True)
    st.write("Fund-level returns (head)")
    st.dataframe(returns_long.head(200), use_container_width=True)
    st.write(
        f"Returns pivot shape: {returns_wide.shape}, date min: {returns_wide.index.min()}, date max: {returns_wide.index.max()}"
    )
