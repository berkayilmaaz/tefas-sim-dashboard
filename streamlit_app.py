# streamlit_app.py
import pandas as pd
import numpy as np
import plotly.express as px
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

st.set_page_config(page_title="TEFAS Simülasyonu", layout="wide")

metadata = loader.get_metadata()


# -----------------------------------------------------------------------------
# DATA LOADERS
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.header("Girdiler")

last_updated = metadata.get("last_updated", "Bilinmiyor")
st.sidebar.caption(f"Veriler en son {last_updated} tarihinde güncellendi.")

if not loader.MASTER_PATH.exists():
    st.error(
        "Veri dosyası bulunamadı. Lütfen `python -m scripts.update_data` "
        "komutunu çalıştırın."
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

min_date = master_data["date"].min().date()
max_date = master_data["date"].max().date()

# FIX: Removed stray `d` character that was here (syntax error)

date_range = st.sidebar.date_input(
    "Tarih Aralığı",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Safe date range unpacking
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date_input, end_date_input = date_range
elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
    start_date_input = date_range[0]
    end_date_input = date_range[0]
else:
    start_date_input = min_date
    end_date_input = max_date


fund_type_options = (
    master_data["category"].dropna().astype(str).sort_values().unique().tolist()
)
selected_fund_types = st.sidebar.multiselect("Fon Türü", options=fund_type_options)

fund_code_options = (
    master_data["fund_code"].dropna().astype(str).sort_values().unique().tolist()
)
selected_fund_codes = st.sidebar.multiselect("Fon Kodu", options=fund_code_options)

st.sidebar.subheader("Sermaye Yönetimi")
per_fund_capital = st.sidebar.number_input(
    "Fon Başına Başlangıç (TL)", value=1000.0, step=100.0
)

st.sidebar.subheader("Fon Filtresi")
min_aum = st.sidebar.number_input(
    "Minimum AUM (son gün)", min_value=0.0, value=0.0, step=1_000_000.0
)
min_obs_days = st.sidebar.number_input(
    "Minimum gözlem günü", min_value=1, value=60, step=1
)
missing_tolerance = st.sidebar.number_input(
    "Eksik gün toleransı", min_value=0, value=5, step=1
)

st.sidebar.subheader("Strateji Seçimi")
k = st.sidebar.slider("K (Fon Sayısı)", min_value=10, max_value=500, value=50)

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

if st.sidebar.button("Önbelleği temizle"):
    st.cache_data.clear()
    st.sidebar.success("Temizlendi.")

# -----------------------------------------------------------------------------
# MAIN LOGIC
# -----------------------------------------------------------------------------
st.title("TEFAS Simülasyon & İstatistik Laboratuvarı")

start_date = pd.Timestamp(start_date_input)
end_date = pd.Timestamp(end_date_input)

if start_date > end_date:
    st.error("Lütfen tarih aralığını kontrol edin.")
    st.stop()

# 1. DATA
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

st.info(
    f"**Simülasyon Ayarları:** {actual_k} fon seçildi. "
    f"Fon başı {per_fund_capital:,.0f} TL. "
    f"Toplam Sermaye: **{initial_capital:,.0f} TL**"
)

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
df_results = backtest_portfolio_assets(prices_wide, weights_table, initial_capital)

if df_results.empty:
    st.error(
        "Backtest hesaplanamadı. "
        "Tarih aralığını genişletmeyi veya lookback süresini kısaltmayı deneyin."
    )
    st.stop()

strategy_equity = df_results["equity"]
strategy_returns = df_results["ret"]

# 4. RESULTS
final_value = strategy_equity.iloc[-1]
net_profit = final_value - initial_capital
total_ret_pct = net_profit / initial_capital if initial_capital > 0 else 0.0

col1, col2, col3 = st.columns(3)
col1.metric("Başlangıç Sermayesi", f"{initial_capital:,.0f} TL")
col2.metric(
    "Bitiş Sermayesi",
    f"{final_value:,.0f} TL",
    f"%{total_ret_pct * 100:.2f}",
)
col3.metric("Net Kâr/Zarar", f"{net_profit:,.0f} TL")

# EQUITY CHART
st.subheader("Portföy Değeri (TL)")
st.plotly_chart(
    plot_equity(strategy_equity / initial_capital), use_container_width=True
)

# TABS
tab_stats, tab_monte, tab_metrics = st.tabs(
    ["📊 İstatistiksel Analiz", "🎲 Monte Carlo Simülasyonu", "📈 Metrikler & Tablo"]
)

# ---- TAB 1: Statistical Analysis ----------------------------------------
with tab_stats:
    st.subheader("Normallik ve Kuyruk Riski Analizi")
    st.markdown("Getirilerin istatistiksel dağılımı:")

    norm_res = check_normality(strategy_returns)

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric(
        "Normal Dağılım mı?",
        "EVET" if norm_res["is_normal"] else "HAYIR",
        help="p-value > 0.05 ise Normal kabul edilir.",
    )
    sc2.metric("p-value", f"{norm_res['p_value']:.4f}")
    sc3.metric(
        "Çarpıklık (Skew)",
        f"{norm_res['skew']:.2f}",
        help="Negatif skew = Ani çöküş riski (Sol kuyruk).",
    )
    sc4.metric(
        "Basıklık (Kurtosis)",
        f"{norm_res['kurtosis']:.2f}",
        help="Yüksek değer = 'Fat Tail' (Siyah Kuğu) olasılığı.",
    )

    st.plotly_chart(
        plot_return_histogram(
            strategy_returns.to_frame(name="ret").assign(fund_code="Strateji")
        ),
        use_container_width=True,
    )

# ---- TAB 2: Monte Carlo -------------------------------------------------
with tab_monte:
    st.subheader(f"Şans Testi: {actual_k} Fonu Rastgele Seçseydik?")
    st.markdown(
        "Bu simülasyon, başarınızın 'strateji' mi yoksa "
        "sadece 'piyasanın yükselmesi' mi olduğunu test eder."
    )

    if st.button("Simülasyonu Başlat (1000 Tekrar)"):
        with st.spinner("Monte Carlo çalışıyor..."):
            mc_result = run_monte_carlo_simulation(
                prices_wide=prices_wide,
                actual_k=actual_k,
                strategy_total_return=total_ret_pct,
                n_sims=1_000,
            )

            if mc_result.simulated_returns.size == 0:
                st.warning("Monte Carlo çalıştırılamadı – geçerli fon getirisi yok.")
            else:
                mc_series = pd.Series(mc_result.simulated_returns)

                fig = px.histogram(
                    mc_series,
                    nbins=50,
                    title="Rastgele Portföylerin Getiri Dağılımı",
                    labels={"value": "Getiri"},
                )
                fig.add_vline(
                    x=total_ret_pct,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Sizin Stratejiniz",
                )
                st.plotly_chart(fig, use_container_width=True)

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric(
                    "Medyan Rastgele Getiri",
                    f"%{mc_result.median_random * 100:.2f}",
                )
                mc2.metric(
                    "%5 - %95 Aralığı",
                    f"%{mc_result.p5_random * 100:.1f} – "
                    f"%{mc_result.p95_random * 100:.1f}",
                )
                mc3.metric(
                    "Yüzdelik Sıra",
                    f"%{mc_result.percentile_rank:.1f}",
                )

                st.success(
                    f"Sizin stratejiniz, rastgele oluşturulan portföylerin "
                    f"**%{mc_result.percentile_rank:.1f}**'inden daha iyi "
                    f"performans gösterdi."
                )

# ---- TAB 3: Detailed Metrics --------------------------------------------
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

    # Fund summary table
    st.subheader("Fon Özeti")
    st.dataframe(filtered_summary, use_container_width=True)
