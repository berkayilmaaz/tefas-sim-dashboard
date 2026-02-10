# streamlit_app.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import loader
from src.domain.correlation import correlation_matrix, correlation_pairs
from src.domain.returns import compute_fund_daily_returns
from src.domain.stats import (
    annualized_vol,
    drawdown_series,
    equity_curve,
    max_drawdown,
    rolling_metrics,
    sharpe,
    total_return,
    check_normality,
    hurst_exponent,
    shannon_entropy,
    mean_reversion_halflife,
    variance_ratio_test,
    compute_fund_analytics,
    categorize_funds,
    compute_period_returns,
    infer_fund_category,
)
from src.domain.strategies import (
    build_weights,
    backtest_portfolio_assets,
    run_monte_carlo_simulation,
    select_universe_k,
)
from src.types import StrategyParams
from src.viz.plots import (
    plot_correlation_heatmap,
    plot_drawdown,
    plot_equity,
    plot_equity_comparison,
    plot_return_histogram,
    plot_rolling_metrics,
)

st.set_page_config(
    page_title="TEFAS Simülasyon Laboratuvarı",
    page_icon="🔬",
    layout="wide",
)

metadata = loader.get_metadata()


# =============================================================================
# HELPERS
# =============================================================================
@st.cache_data(show_spinner=False)
def summarize_funds(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()

    prices = prices.sort_values(["fund_code", "date"])
    returns = compute_fund_daily_returns(prices)

    if returns.empty:
        return pd.DataFrame()

    ret_group = returns.groupby("fund_code")["ret"]
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
    summary["vol_daily"] = ret_group.std(ddof=1)
    summary = summary.reset_index()

    for col in ["aum_last", "price_last", "start_price", "end_price"]:
        summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(0.0)
    summary["missing_days"] = expected_days - summary["obs_days"]

    # Infer category from fund name
    summary["inferred_category"] = summary["fund_name"].apply(infer_fund_category)

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

    if mode == "AUM'a göre ilk K":
        selected = (
            summary.sort_values("aum_last", ascending=False)
            .head(k)["fund_code"]
            .tolist()
        )
        return selected, missing_manual
    if mode == "Rastgele K":
        selected = select_universe_k(fund_codes, k, seed)
        return selected, missing_manual
    if mode == "Manuel (fon kodlarını seç)":
        code_set = set(fund_codes)
        selected = []
        for code in manual_codes:
            if code in code_set:
                selected.append(code)
            else:
                missing_manual.append(code)
        return selected, missing_manual
    return fund_codes, missing_manual


def format_pct(val: float) -> str:
    if pd.isna(val):
        return "—"
    return f"%{val * 100:.2f}"


# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.header("⚙️ Girdiler")

last_updated = metadata.get("last_updated", "Bilinmiyor")
st.sidebar.caption(f"📅 Son güncelleme: {last_updated}")

if not loader.MASTER_PATH.exists():
    st.error(
        "Veri dosyası bulunamadı. "
        "`python -m scripts.update_data` komutunu çalıştırın."
    )
    st.stop()

master_data = loader.load_master_data()

if master_data.empty:
    st.error(
        "Ana veri dosyası boş veya okunamadı. "
        "`python -m scripts.update_data` ile yeniden indirin."
    )
    st.stop()

macro_data = loader.load_macro_data()
master_data["date"] = pd.to_datetime(master_data["date"], errors="coerce")
master_data = master_data.dropna(subset=["date"])

if master_data.empty:
    st.error("Tarih verisi çözümlenemedi.")
    st.stop()

# Infer categories for the entire master data (cached via summarize_funds)
if "inferred_category" not in master_data.columns:
    if "fund_name" in master_data.columns:
        # Build a code→category map from unique fund names
        _name_map = (
            master_data[["fund_code", "fund_name"]]
            .drop_duplicates("fund_code")
            .set_index("fund_code")["fund_name"]
        )
        _cat_map = _name_map.apply(infer_fund_category)
        master_data["inferred_category"] = master_data["fund_code"].map(_cat_map)
    else:
        master_data["inferred_category"] = "Diğer"

min_date = master_data["date"].min().date()
max_date = master_data["date"].max().date()

date_range = st.sidebar.date_input(
    "Tarih Aralığı",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date_input, end_date_input = date_range
elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
    start_date_input = date_range[0]
    end_date_input = date_range[0]
else:
    start_date_input = min_date
    end_date_input = max_date

# --- CATEGORY FILTER (NEW — like TEFAS website) ---
st.sidebar.subheader("🏷️ Fon Kategorisi")
all_categories = sorted(master_data["inferred_category"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Şemsiye Fon Türü",
    options=all_categories,
    help="Boş bırakırsanız tüm kategoriler dahil edilir.",
)

# Original category filter (from data source)
fund_type_options = (
    master_data["category"].dropna().astype(str).sort_values().unique().tolist()
    if "category" in master_data.columns
    else []
)
if fund_type_options:
    selected_fund_types = st.sidebar.multiselect(
        "Veri Kaynağı Kategorisi", options=fund_type_options
    )
else:
    selected_fund_types = []

fund_code_options = (
    master_data["fund_code"].dropna().astype(str).sort_values().unique().tolist()
)
selected_fund_codes = st.sidebar.multiselect(
    "Fon Kodu (Manuel)", options=fund_code_options
)

st.sidebar.subheader("💰 Sermaye Yönetimi")
per_fund_capital = st.sidebar.number_input(
    "Fon Başına Başlangıç (TL)", value=1000.0, step=100.0
)

st.sidebar.subheader("🔍 Fon Filtresi")
min_aum = st.sidebar.number_input(
    "Minimum AUM (son gün)", min_value=0.0, value=0.0, step=1_000_000.0
)
min_obs_days = st.sidebar.number_input(
    "Minimum gözlem günü", min_value=1, value=60, step=1
)
missing_tolerance = st.sidebar.number_input(
    "Eksik gün toleransı", min_value=0, value=5, step=1
)

st.sidebar.subheader("🎯 Strateji Seçimi")
k = st.sidebar.slider("K (Fon Sayısı)", min_value=2, max_value=500, value=50)

selection_mode = st.sidebar.radio(
    "Havuz Seçimi",
    options=["AUM'a göre ilk K", "Rastgele K", "Manuel (fon kodlarını seç)"],
    index=0,
)

seed = 42
if selection_mode == "Rastgele K":
    seed = st.sidebar.number_input(
        "Random Seed", min_value=0, max_value=10000, value=42
    )

strategy_label = st.sidebar.selectbox(
    "Ağırlıklandırma Stratejisi",
    options=[
        "Eşit Ağırlık",
        "Momentum Top-K",
        "Düşük Vol Top-K",
        "Sharpe Top-K",
        "Risk Parity (basit)",
        "Min Variance (baseline)",
    ],
    index=0,
)

strategy_map = {
    "Eşit Ağırlık": "equal_weight",
    "Momentum Top-K": "momentum_top_k",
    "Düşük Vol Top-K": "low_vol_top_k",
    "Sharpe Top-K": "sharpe_top_k",
    "Risk Parity (basit)": "risk_parity",
    "Min Variance (baseline)": "min_variance",
}
strategy_key = strategy_map[strategy_label]

lookback = 20
if strategy_key != "equal_weight":
    lookback = st.sidebar.slider("Geriye bakış (gün)", 5, 252, 20)

rebalance = st.sidebar.selectbox(
    "Rebalance Sıklığı", ["daily", "weekly", "monthly"], index=2
)

if st.sidebar.button("🗑️ Önbelleği temizle"):
    st.cache_data.clear()
    st.sidebar.success("Temizlendi.")

# =============================================================================
# MAIN LOGIC
# =============================================================================
st.title("🔬 TEFAS Simülasyon & İstatistik Laboratuvarı")

start_date = pd.Timestamp(start_date_input)
end_date = pd.Timestamp(end_date_input)

if start_date > end_date:
    st.error("Lütfen tarih aralığını kontrol edin.")
    st.stop()

# 1. DATA — apply both category filters
with st.spinner("Veriler çekiliyor..."):
    try:
        prices = loader.filter_data(
            master_data,
            start_date,
            end_date,
            selected_fund_codes,
            selected_fund_types,
        )
    except Exception as exc:
        st.error(f"Veri hatası: {exc}")
        st.stop()

# Apply inferred category filter
if selected_categories and not prices.empty:
    if "inferred_category" not in prices.columns and "fund_name" in prices.columns:
        _nm = (
            prices[["fund_code", "fund_name"]]
            .drop_duplicates("fund_code")
            .set_index("fund_code")["fund_name"]
        )
        _cm = _nm.apply(infer_fund_category)
        prices["inferred_category"] = prices["fund_code"].map(_cm)
    if "inferred_category" in prices.columns:
        prices = prices[prices["inferred_category"].isin(selected_categories)]

if prices.empty:
    st.warning("Seçilen filtreler ve tarih aralığı için veri bulunamadı.")
    st.stop()

# 2. FILTERING & SELECTION
summary_all = summarize_funds(prices)

if summary_all.empty:
    st.warning("Getiri hesaplanamadı. Tarih aralığında en az 2 günlük veri gerekli.")
    st.stop()

filtered_summary = summary_all.copy()
filtered_summary = filtered_summary[filtered_summary["aum_last"] >= min_aum]
filtered_summary = filtered_summary[filtered_summary["obs_days"] >= min_obs_days]
filtered_summary = filtered_summary[
    filtered_summary["missing_days"] <= missing_tolerance
]

if filtered_summary.empty:
    st.error(
        "Filtrelere uygun fon bulunamadı. "
        "Minimum AUM, gözlem günü veya eksik gün toleransını gevşetmeyi deneyin."
    )
    st.stop()

# Show pool stats
pool_size = len(filtered_summary)
cat_counts = filtered_summary["inferred_category"].value_counts()

manual_codes = []
if selection_mode == "Manuel (fon kodlarını seç)":
    manual_codes = selected_fund_codes

selected_funds, missing = select_funds(
    filtered_summary, selection_mode, int(k), int(seed), manual_codes
)

if missing:
    st.warning(f"Bu fon kodları veride bulunamadı: {', '.join(missing)}")

if not selected_funds:
    st.error("Fon seçilemedi.")
    st.stop()

actual_k = len(selected_funds)
initial_capital = actual_k * per_fund_capital

# 3. BACKTEST
selected_prices = prices[prices["fund_code"].isin(selected_funds)]
returns_long = compute_fund_daily_returns(selected_prices)

if returns_long.empty:
    st.error("Seçilen fonlar için getiri hesaplanamadı (yetersiz veri).")
    st.stop()

returns_wide = returns_long.pivot_table(
    index="date", columns="fund_code", values="ret", aggfunc="mean"
).sort_index()

strategy_params = StrategyParams(
    k=actual_k,
    seed=int(seed),
    strategy=strategy_key,
    lookback=lookback,
    rebalance=rebalance,
)

weights_daily, weights_table = build_weights(returns_wide, strategy_params)

prices_wide = selected_prices.pivot_table(
    index="date", columns="fund_code", values="price", aggfunc="last"
)

# Also build FULL pool prices for Monte Carlo (this is the key fix)
all_pool_prices_wide = prices.pivot_table(
    index="date", columns="fund_code", values="price", aggfunc="last"
)

df_results = backtest_portfolio_assets(prices_wide, weights_table, initial_capital)

if df_results.empty:
    st.error(
        "Backtest hesaplanamadı. "
        "Tarih aralığını genişletmeyi veya lookback süresini kısaltmayı deneyin."
    )
    st.stop()

strategy_equity = df_results["equity"]
strategy_returns = df_results["ret"]

# 4. TOP KPIs
final_value = strategy_equity.iloc[-1]
net_profit = final_value - initial_capital
total_ret_pct = net_profit / initial_capital if initial_capital > 0 else 0.0

# Category breakdown of selection
sel_cats = filtered_summary[filtered_summary["fund_code"].isin(selected_funds)][
    "inferred_category"
].value_counts()
cat_text = " · ".join([f"{cat}: {cnt}" for cat, cnt in sel_cats.items()])

st.info(
    f"**{actual_k}** / {pool_size} fondan seçildi · "
    f"Fon başı **{per_fund_capital:,.0f} TL** · "
    f"Toplam: **{initial_capital:,.0f} TL** · "
    f"Strateji: **{strategy_label}** · Rebalance: **{rebalance}**\n\n"
    f"📊 Kategori dağılımı: {cat_text}"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Başlangıç", f"{initial_capital:,.0f} TL")
col2.metric("Bitiş", f"{final_value:,.0f} TL", f"%{total_ret_pct * 100:.2f}")
col3.metric("Net Kâr/Zarar", f"{net_profit:,.0f} TL")
col4.metric("Max Drawdown", format_pct(max_drawdown(strategy_equity)))

# EQUITY CHART
st.subheader("Portföy Değeri (TL)")
eq_fig = go.Figure()
eq_fig.add_trace(
    go.Scatter(
        x=strategy_equity.index,
        y=strategy_equity.values,
        mode="lines",
        name="Portföy",
        line=dict(width=2),
    )
)
eq_fig.update_layout(
    yaxis_title="TL",
    xaxis_title="Tarih",
    margin=dict(l=10, r=10, t=30, b=10),
    height=350,
)
st.plotly_chart(eq_fig, use_container_width=True)


# =============================================================================
# TABS
# =============================================================================
(
    tab_funds,
    tab_compare,
    tab_stats,
    tab_monte,
    tab_physics,
    tab_metrics,
) = st.tabs(
    [
        "📋 Seçilen Fonlar",
        "⚔️ Fon Karşılaştırma",
        "📊 İstatistiksel Analiz",
        "🎲 Monte Carlo",
        "⚛️ Fizik Metrikleri",
        "📈 Metrikler & Tablo",
    ]
)


# ── TAB 1: FUND EXPLORER (TEFAS-style table) ─────────────────────────────────
with tab_funds:
    st.subheader(f"Portföydeki {actual_k} Fon")

    # Compute period returns like TEFAS website
    period_rets = compute_period_returns(selected_prices)
    fund_analytics = compute_fund_analytics(returns_long, prices_wide)

    if fund_analytics.empty:
        st.warning("Fon analitikleri hesaplanamadı.")
    else:
        name_map = (
            selected_prices[["fund_code", "fund_name"]]
            .drop_duplicates("fund_code")
            .set_index("fund_code")["fund_name"]
        )
        fund_analytics["fund_name"] = fund_analytics["fund_code"].map(name_map)
        fund_analytics["inferred_category"] = fund_analytics["fund_name"].apply(
            infer_fund_category
        )

        # Merge period returns
        if not period_rets.empty:
            fund_analytics = fund_analytics.merge(
                period_rets, on="fund_code", how="left"
            )

        # Category filter within tab
        tab_categories = sorted(fund_analytics["inferred_category"].unique().tolist())
        if len(tab_categories) > 1:
            cat_filter = st.multiselect(
                "Kategoriye göre filtrele:",
                options=tab_categories,
                default=tab_categories,
                key="fund_tab_cat_filter",
            )
            fund_analytics_display = fund_analytics[
                fund_analytics["inferred_category"].isin(cat_filter)
            ]
        else:
            fund_analytics_display = fund_analytics

        sort_col = st.selectbox(
            "Sıralama Kriteri",
            options=[
                "total_return",
                "ret_1m",
                "ret_3m",
                "ret_6m",
                "ret_1y",
                "annualized_vol",
                "sharpe",
                "max_drawdown",
                "hurst",
            ],
            format_func=lambda x: {
                "total_return": "Toplam Getiri",
                "ret_1m": "1 Aylık Getiri",
                "ret_3m": "3 Aylık Getiri",
                "ret_6m": "6 Aylık Getiri",
                "ret_1y": "1 Yıllık Getiri",
                "annualized_vol": "Yıllık Volatilite",
                "sharpe": "Sharpe Oranı",
                "max_drawdown": "Max Drawdown",
                "hurst": "Hurst Üsteli",
            }.get(x, x),
        )

        sort_asc = sort_col in ["max_drawdown", "annualized_vol"]
        display_df = fund_analytics_display.sort_values(
            sort_col, ascending=sort_asc, na_position="last"
        ).copy()

        # Build display columns
        show_cols = ["fund_code", "fund_name", "inferred_category"]
        col_names = ["Fon Kodu", "Fon Adı", "Kategori"]
        fmt = {}

        for c, label, f in [
            ("ret_1m", "1 Ay", "{:.2%}"),
            ("ret_3m", "3 Ay", "{:.2%}"),
            ("ret_6m", "6 Ay", "{:.2%}"),
            ("ret_1y", "1 Yıl", "{:.2%}"),
            ("total_return", "Toplam", "{:.2%}"),
            ("annualized_vol", "Yıllık Vol", "{:.2%}"),
            ("sharpe", "Sharpe", "{:.2f}"),
            ("max_drawdown", "MDD", "{:.2%}"),
            ("hurst", "Hurst", "{:.3f}"),
            ("obs_days", "Gözlem", "{:.0f}"),
        ]:
            if c in display_df.columns:
                show_cols.append(c)
                col_names.append(label)
                if c != "obs_days":
                    fmt[label] = f

        display_table = display_df[show_cols].copy()
        display_table.columns = col_names

        st.dataframe(
            display_table.style.format(fmt, na_rep="—"),
            use_container_width=True,
            height=min(35 * len(display_table) + 38, 600),
        )

        # Category performance summary
        st.subheader("Kategori Bazında Performans")
        cat_perf = (
            fund_analytics.groupby("inferred_category")
            .agg(
                fon_sayisi=("fund_code", "count"),
                ort_getiri=("total_return", "mean"),
                ort_vol=("annualized_vol", "mean"),
                ort_sharpe=("sharpe", "mean"),
                ort_mdd=("max_drawdown", "mean"),
            )
            .reset_index()
        )
        cat_perf.columns = [
            "Kategori",
            "Fon Sayısı",
            "Ort. Getiri",
            "Ort. Vol",
            "Ort. Sharpe",
            "Ort. MDD",
        ]
        cat_perf = cat_perf.sort_values("Ort. Getiri", ascending=False)

        fig_cat = px.bar(
            cat_perf,
            x="Kategori",
            y="Ort. Getiri",
            color="Ort. Sharpe",
            color_continuous_scale="RdYlGn",
            text=cat_perf["Ort. Getiri"].apply(lambda x: f"%{x*100:.1f}"),
            title="Kategori Bazında Ortalama Getiri",
        )
        fig_cat.update_layout(
            margin=dict(l=10, r=10, t=50, b=10),
            height=350,
        )
        st.plotly_chart(fig_cat, use_container_width=True)

        st.dataframe(
            cat_perf.style.format(
                {
                    "Ort. Getiri": "{:.2%}",
                    "Ort. Vol": "{:.2%}",
                    "Ort. Sharpe": "{:.2f}",
                    "Ort. MDD": "{:.2%}",
                }
            ),
            use_container_width=True,
        )

        # Risk-Return scatter
        st.subheader("Risk-Getiri Haritası")
        scatter_df = fund_analytics.dropna(subset=["annualized_vol", "total_return"])
        if not scatter_df.empty:
            fig_scatter = px.scatter(
                scatter_df,
                x="annualized_vol",
                y="total_return",
                hover_name="fund_code",
                color="inferred_category",
                size=np.abs(scatter_df["total_return"]).clip(0.001) * 100,
                labels={
                    "annualized_vol": "Yıllık Volatilite",
                    "total_return": "Toplam Getiri",
                    "inferred_category": "Kategori",
                },
                title="Her nokta bir fon — renk: Kategori, boyut: |getiri|",
            )
            fig_scatter.update_layout(
                margin=dict(l=10, r=10, t=50, b=10),
                height=450,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)


# ── TAB 2: FUND COMPARISON ──────────────────────────────────────────────────
with tab_compare:
    st.subheader("Fon Karşılaştırma")

    available_funds = sorted(selected_funds)
    compare_funds = st.multiselect(
        "Karşılaştırmak istediğiniz fonları seçin (en fazla 10):",
        options=available_funds,
        default=available_funds[: min(5, len(available_funds))],
        max_selections=10,
    )

    if compare_funds:
        # Equity curves
        st.markdown("#### Normalize Edilmiş Fiyat (₺1'den başlayarak)")
        compare_prices = prices_wide[compare_funds].dropna(how="all")
        if not compare_prices.empty:
            normalized = compare_prices / compare_prices.iloc[0]
            fig_eq = go.Figure()
            for fund in compare_funds:
                if fund in normalized.columns:
                    series = normalized[fund].dropna()
                    fig_eq.add_trace(
                        go.Scatter(
                            x=series.index,
                            y=series.values,
                            mode="lines",
                            name=fund,
                        )
                    )
            fig_eq.update_layout(
                yaxis_title="Normalize Fiyat",
                margin=dict(l=10, r=10, t=30, b=10),
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_eq, use_container_width=True)

        # Return distribution
        st.markdown("#### Günlük Getiri Dağılımı")
        compare_rets = returns_long[returns_long["fund_code"].isin(compare_funds)]
        if not compare_rets.empty:
            fig_hist = px.histogram(
                compare_rets,
                x="ret",
                color="fund_code",
                barmode="overlay",
                nbins=50,
                opacity=0.6,
                labels={"ret": "Günlük Getiri", "fund_code": "Fon"},
            )
            fig_hist.update_layout(
                margin=dict(l=10, r=10, t=30, b=10),
                height=350,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Correlation
        if len(compare_funds) >= 2:
            st.markdown("#### Korelasyon Matrisi")
            corr_data = returns_wide[
                [f for f in compare_funds if f in returns_wide.columns]
            ].corr()
            fig_corr = px.imshow(
                corr_data,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
            )
            fig_corr.update_layout(
                margin=dict(l=10, r=10, t=30, b=10),
                height=400,
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        # KPI comparison
        st.markdown("#### KPI Karşılaştırma")
        if not fund_analytics.empty:
            kpi_cols = [
                "fund_code",
                "total_return",
                "annualized_vol",
                "sharpe",
                "max_drawdown",
                "best_day",
                "worst_day",
            ]
            extra = [
                c for c in ["ret_1m", "ret_3m", "ret_6m"] if c in fund_analytics.columns
            ]
            kpi_df = fund_analytics[fund_analytics["fund_code"].isin(compare_funds)][
                kpi_cols + extra
            ].copy()

            rename_map = {
                "fund_code": "Fon",
                "total_return": "Toplam Getiri",
                "annualized_vol": "Yıllık Vol",
                "sharpe": "Sharpe",
                "max_drawdown": "MDD",
                "best_day": "En İyi Gün",
                "worst_day": "En Kötü Gün",
                "ret_1m": "1 Ay",
                "ret_3m": "3 Ay",
                "ret_6m": "6 Ay",
            }
            kpi_df = kpi_df.rename(columns=rename_map)

            pct_cols = [c for c in kpi_df.columns if c not in ["Fon", "Sharpe"]]
            fmt = {c: "{:.2%}" for c in pct_cols}
            fmt["Sharpe"] = "{:.2f}"

            st.dataframe(kpi_df.style.format(fmt, na_rep="—"), use_container_width=True)
    else:
        st.info("Karşılaştırmak için en az 1 fon seçin.")


# ── TAB 3: STATISTICAL ANALYSIS ─────────────────────────────────────────────
with tab_stats:
    st.subheader("Normallik ve Kuyruk Riski Analizi")

    norm_res = check_normality(strategy_returns)

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric(
        "Normal Dağılım mı?",
        "EVET ✅" if norm_res["is_normal"] else "HAYIR ❌",
        help="Shapiro-Wilk: p > 0.05 → Normal kabul edilir.",
    )
    sc2.metric("p-value", f"{norm_res['p_value']:.4f}")
    sc3.metric(
        "Çarpıklık (Skew)",
        f"{norm_res['skew']:.3f}",
        help="< 0: Sol kuyruk riski. > 0: Sağ kuyruk.",
    )
    sc4.metric(
        "Basıklık (Kurtosis)",
        f"{norm_res['kurtosis']:.3f}",
        help="Normal = 0. Yüksek → 'Fat tails' (Siyah Kuğu riski).",
    )

    st.plotly_chart(
        plot_return_histogram(
            strategy_returns.to_frame(name="ret").assign(fund_code="Strateji")
        ),
        use_container_width=True,
    )

    with st.expander("📖 Nasıl Okunur?"):
        st.markdown("""
        **Shapiro-Wilk Testi:** p < 0.05 → getiriler normal dağılmıyor.

        **Çarpıklık:** Negatif → ani çöküş riski. Pozitif → yukarı sürpriz.

        **Basıklık:** Yüksek = "fat tails" = uç olaylar daha sık (Siyah Kuğu).
        """)


# ── TAB 4: MONTE CARLO ──────────────────────────────────────────────────────
with tab_monte:
    st.subheader(f"Şans Testi: {actual_k} Fonu Rastgele Seçseydik?")

    # Show pool vs selection info
    st.markdown(
        f"Havuzda **{pool_size}** fon var, siz **{actual_k}** tane seçtiniz. "
        f"Monte Carlo, havuzdaki {pool_size} fondan rastgele {actual_k} tane "
        f"seçerek sizin stratejinizle karşılaştırır."
    )

    if actual_k >= pool_size:
        st.warning(
            f"⚠️ Seçtiğiniz fon sayısı ({actual_k}) havuzdaki toplam fon sayısına "
            f"({pool_size}) eşit veya daha fazla. Monte Carlo anlamlı sonuç veremez "
            f"çünkü her simülasyon aynı fonları seçer. "
            f"**K değerini düşürün** veya **filtreleri gevşetin** (daha fazla fon dahil edin)."
        )

    mc_n_sims = st.slider("Simülasyon sayısı", 100, 10_000, 1_000, step=100)

    if st.button("🚀 Simülasyonu Başlat", type="primary"):
        with st.spinner("Monte Carlo çalışıyor..."):
            # KEY FIX: Use FULL pool prices, not just selected funds
            mc_result = run_monte_carlo_simulation(
                prices_wide=all_pool_prices_wide,
                actual_k=actual_k,
                strategy_total_return=total_ret_pct,
                n_sims=mc_n_sims,
                full_pool_size=pool_size,
            )

            if mc_result.simulated_returns.size == 0:
                st.warning("Monte Carlo çalıştırılamadı — geçerli fon getirisi yok.")
            elif mc_result.is_degenerate:
                st.error(
                    f"⛔ K ({actual_k}) ≥ havuz ({mc_result.n_valid_funds}). "
                    f"Tüm simülasyonlar aynı sonucu veriyor. K'yı düşürün."
                )
                st.metric("Havuz Ortalaması", f"%{mc_result.median_random * 100:.2f}")
                st.metric("Sizin Getiriniz", f"%{total_ret_pct * 100:.2f}")
            else:
                mc_series = pd.Series(mc_result.simulated_returns)

                fig = go.Figure()
                fig.add_trace(
                    go.Histogram(
                        x=mc_series,
                        nbinsx=60,
                        name="Rastgele Portföyler",
                        marker_color="rgba(100, 149, 237, 0.6)",
                    )
                )
                fig.add_vline(
                    x=total_ret_pct,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Sizin Stratejiniz ({total_ret_pct*100:.1f}%)",
                    annotation_position="top left",
                )
                fig.add_vline(
                    x=mc_result.median_random,
                    line_dash="dot",
                    line_color="gray",
                    annotation_text=f"Medyan ({mc_result.median_random*100:.1f}%)",
                    annotation_position="top right",
                )
                fig.update_layout(
                    title=f"{mc_n_sims:,} Rastgele Portföy · "
                    f"{mc_result.n_valid_funds} fondan {mc_result.draw_k} seçim",
                    xaxis_title="Toplam Getiri",
                    yaxis_title="Frekans",
                    margin=dict(l=10, r=10, t=50, b=10),
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Sizin Getiriniz", f"%{total_ret_pct * 100:.2f}")
                mc2.metric("Medyan Rastgele", f"%{mc_result.median_random * 100:.2f}")
                mc3.metric(
                    "%5 – %95 Aralığı",
                    f"%{mc_result.p5_random * 100:.1f} – "
                    f"%{mc_result.p95_random * 100:.1f}",
                )
                mc4.metric("Yüzdelik Sıra", f"%{mc_result.percentile_rank:.1f}")

                if mc_result.percentile_rank >= 75:
                    st.success(
                        f"🎯 Stratejiniz rastgele portföylerin "
                        f"**%{mc_result.percentile_rank:.1f}**'inden iyi — alfa sinyali."
                    )
                elif mc_result.percentile_rank >= 50:
                    st.info(
                        f"📊 Ortalamanın üstünde (%{mc_result.percentile_rank:.1f}), "
                        f"ama güçlü alfa yok."
                    )
                else:
                    st.warning(
                        f"⚠️ Rastgele seçimin altında (%{mc_result.percentile_rank:.1f}). "
                        f"Stratejiyi gözden geçirin."
                    )


# ── TAB 5: PHYSICS METRICS ──────────────────────────────────────────────────
with tab_physics:
    st.subheader("⚛️ Fizik-İlhamlı Piyasa Analizi")

    # --- VISUAL GAUGES ---
    ph1, ph2, ph3, ph4 = st.columns(4)

    # Hurst
    h_val = hurst_exponent(strategy_equity)
    with ph1:
        st.metric(
            "Hurst Üsteli (H)",
            f"{h_val:.3f}" if not np.isnan(h_val) else "—",
        )
        if not np.isnan(h_val):
            # Visual gauge
            gauge_color = (
                "#27ae60"
                if h_val > 0.55  # trending → green
                else (
                    "#3498db" if h_val < 0.45 else "#95a5a6"  # mean-rev → blue
                )  # random → gray
            )
            label = (
                "📈 Trending (momentum)"
                if h_val > 0.55
                else "📉 Mean-reverting" if h_val < 0.45 else "🎲 Random walk"
            )
            st.progress(min(h_val, 1.0))
            st.caption(label)

    # Entropy
    ent_val = shannon_entropy(strategy_returns)
    with ph2:
        st.metric(
            "Shannon Entropisi",
            f"{ent_val:.3f}" if not np.isnan(ent_val) else "—",
        )
        if not np.isnan(ent_val):
            # Normalize: typical range 1.5-4.0
            ent_norm = min(max((ent_val - 1.0) / 3.0, 0.0), 1.0)
            st.progress(ent_norm)
            label = (
                "🔮 Düşük → daha öngörülebilir"
                if ent_norm < 0.4
                else "🎲 Yüksek → kaotik" if ent_norm > 0.7 else "⚖️ Orta düzey"
            )
            st.caption(label)

    # Half-life
    hl_val = mean_reversion_halflife(strategy_equity)
    with ph3:
        if np.isinf(hl_val):
            st.metric("Yarı-Ömür (OU)", "∞")
            st.caption("📈 Trending — ortalamaya dönüş yok")
        elif np.isnan(hl_val):
            st.metric("Yarı-Ömür (OU)", "—")
        else:
            st.metric("Yarı-Ömür (OU)", f"{hl_val:.1f} gün")
            if hl_val < 10:
                st.caption("⚡ Çok hızlı ortalamaya dönüş")
            elif hl_val < 30:
                st.caption("🔄 Orta hızda mean-reversion")
            else:
                st.caption("🐢 Yavaş ortalamaya dönüş")

    # Variance ratio
    vr_res = variance_ratio_test(strategy_returns, holding_period=5)
    with ph4:
        vr_val = vr_res["variance_ratio"]
        st.metric(
            "Varyans Oranı (q=5)",
            f"{vr_val:.3f}" if not np.isnan(vr_val) else "—",
        )
        st.caption(vr_res["regime"])

    st.markdown("---")

    # --- INTERPRETATION PANEL ---
    st.markdown("### 🧭 Ne Anlama Geliyor?")

    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown("#### Piyasa Rejimi Tespiti")
        if not np.isnan(h_val) and not np.isnan(vr_val):
            if h_val > 0.55 and vr_val > 1.05:
                st.success(
                    "🟢 **Momentum rejimi tespit edildi.** "
                    "Hem Hurst hem VR trending sinyali veriyor. "
                    "Trend-takip stratejileri avantajlı olabilir."
                )
            elif h_val < 0.45 and vr_val < 0.95:
                st.info(
                    "🔵 **Mean-reversion rejimi tespit edildi.** "
                    "Fiyatlar ortalamaya dönme eğiliminde. "
                    "Contrarian / pairs trading stratejileri düşünün."
                )
            else:
                st.warning(
                    "⚪ **Karışık sinyaller.** "
                    "Piyasa belirgin bir rejim göstermiyor. "
                    "Diversifikasyonla risk yönetimi önemli."
                )
        else:
            st.info("Yeterli veri yok — rejim tespiti yapılamadı.")

    with ic2:
        st.markdown("#### Strateji Önerisi")
        suggestions = []
        if not np.isnan(h_val):
            if h_val > 0.55:
                suggestions.append("✅ Momentum Top-K stratejisini deneyin")
            elif h_val < 0.45:
                suggestions.append("✅ Low-Vol veya Mean-Reversion stratejisi")
        if not np.isinf(hl_val) and not np.isnan(hl_val) and hl_val < 20:
            suggestions.append(f"✅ Yarı-ömür {hl_val:.0f} gün → kısa vadeli rebalance")
        if not suggestions:
            suggestions.append("📊 Eşit ağırlık iyi bir başlangıç noktası")
        for s in suggestions:
            st.markdown(s)

    # Hurst scatter per fund
    if not fund_analytics.empty and "hurst" in fund_analytics.columns:
        st.markdown("---")
        st.markdown("### Fon Bazında Hurst Üsteli")
        hurst_df = fund_analytics[["fund_code", "hurst", "total_return"]].dropna(
            subset=["hurst"]
        )
        if "inferred_category" in fund_analytics.columns:
            hurst_df = hurst_df.merge(
                fund_analytics[["fund_code", "inferred_category"]],
                on="fund_code",
                how="left",
            )
            color_col = "inferred_category"
        else:
            color_col = None

        if not hurst_df.empty:
            fig_hurst = px.scatter(
                hurst_df,
                x="hurst",
                y="total_return",
                hover_name="fund_code",
                color=color_col,
                labels={
                    "hurst": "Hurst Üsteli",
                    "total_return": "Toplam Getiri",
                    "inferred_category": "Kategori",
                },
                title="Hurst vs Getiri — Hangi fonlar trending, hangileri mean-reverting?",
            )
            fig_hurst.add_vline(
                x=0.5,
                line_dash="dash",
                line_color="gray",
                annotation_text="H=0.5 (Random Walk)",
            )
            fig_hurst.update_layout(
                margin=dict(l=10, r=10, t=50, b=10),
                height=400,
            )
            st.plotly_chart(fig_hurst, use_container_width=True)

    with st.expander("📖 Metrik Rehberi"):
        st.markdown("""
        | Metrik | Fizik Karşılığı | Değer | Anlam |
        |--------|----------------|-------|-------|
        | **Hurst** | Fractional Brownian Motion | H > 0.5 | Trending (momentum) |
        | | | H < 0.5 | Mean-reverting |
        | | | H ≈ 0.5 | Random walk |
        | **Entropi** | Termodinamik | Yüksek | Kaotik / öngörülemez |
        | | | Düşük | Düzenli / öngörülebilir |
        | **Yarı-Ömür** | Sönümlü osilatör | Kısa (< 10 gün) | Hızlı mean-reversion |
        | | | Uzun / ∞ | Trending |
        | **VR Testi** | Anomalous diffusion | VR > 1 | Momentum |
        | | | VR < 1 | Mean-reversion |
        | | | VR ≈ 1 | Verimli piyasa |
        """)


# ── TAB 6: DETAILED METRICS & TABLE ─────────────────────────────────────────
with tab_metrics:
    st.subheader("Detaylı Metrikler")

    macro_ready = not macro_data.empty and {
        "date",
        "usdtry",
        "cpi_us",
        "cpi_tr",
    }.issubset(set(macro_data.columns))

    if not macro_ready:
        st.info(
            "Makro veri bulunamadı. USD/TRY veya TÜFE ayarlamaları için "
            "`python -m scripts.update_data` çalıştırın."
        )

    use_usd = st.toggle("USD Bazlı Getiri", value=False)
    use_real = st.toggle("Enflasyondan Arındır (Reel Getiri)", value=False)

    adjusted_equity = strategy_equity.copy()
    macro_aligned = pd.DataFrame()

    if macro_ready:
        macro_aligned = macro_data.copy()
        macro_aligned["date"] = pd.to_datetime(macro_aligned["date"], errors="coerce")
        macro_aligned = macro_aligned.set_index("date").sort_index()
        macro_aligned = macro_aligned.reindex(adjusted_equity.index).ffill()

        if use_usd and "usdtry" in macro_aligned.columns:
            usdtry = macro_aligned["usdtry"].replace(0, np.nan)
            adjusted_equity = adjusted_equity / usdtry

        if use_real:
            cpi_col = "cpi_us" if use_usd else "cpi_tr"
            if cpi_col in macro_aligned.columns:
                cpi_series = macro_aligned[cpi_col]
                first_valid = cpi_series.first_valid_index()
                if first_valid is not None:
                    inflation_index = cpi_series / cpi_series.loc[first_valid]
                    inflation_index = inflation_index.replace(0, np.nan)
                    adjusted_equity = adjusted_equity / inflation_index

    adjusted_returns = adjusted_equity.pct_change().fillna(0.0)

    # Rolling Metrics
    rolling_window = min(30, len(adjusted_returns) - 1)
    if rolling_window >= 5:
        rolling_df = rolling_metrics(adjusted_returns, window=rolling_window)
        st.plotly_chart(
            plot_rolling_metrics(rolling_df, None), use_container_width=True
        )
    else:
        st.info("Yuvarlanan metrikler için yetersiz veri (en az 5 gün gerekli).")

    # Drawdown
    dd_series = drawdown_series(adjusted_equity)
    st.plotly_chart(plot_drawdown(dd_series), use_container_width=True)

    if macro_ready and st.checkbox("Karşılaştırma çizgilerini göster"):
        comparison_frames = []
        eq_first = adjusted_equity.iloc[0]
        if eq_first != 0 and not np.isnan(eq_first):
            equity_norm = adjusted_equity / eq_first
            comparison_frames.append(
                equity_norm.rename("equity").to_frame().assign(fund_code="Strateji")
            )

        if "usdtry" in macro_aligned.columns:
            usd_first = macro_aligned["usdtry"].iloc[0]
            if usd_first != 0 and not np.isnan(usd_first):
                usdtry_norm = macro_aligned["usdtry"] / usd_first
                comparison_frames.append(
                    usdtry_norm.rename("equity").to_frame().assign(fund_code="USD/TRY")
                )

        cpi_col = "cpi_us" if use_usd else "cpi_tr"
        if cpi_col in macro_aligned.columns:
            cpi_base = macro_aligned[cpi_col]
            cpi_first = cpi_base.iloc[0]
            if cpi_first != 0 and not np.isnan(cpi_first):
                cpi_norm = cpi_base / cpi_first
                comparison_frames.append(
                    cpi_norm.rename("equity").to_frame().assign(fund_code="TÜFE")
                )

        if comparison_frames:
            comparison_df = (
                pd.concat(comparison_frames)
                .reset_index()
                .rename(columns={"index": "date"})
            )
            st.plotly_chart(
                plot_equity_comparison(comparison_df), use_container_width=True
            )

    st.subheader("Filtrelenmiş Fon Özeti")
    st.dataframe(filtered_summary, use_container_width=True)
