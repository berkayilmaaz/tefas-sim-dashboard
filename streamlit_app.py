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
    check_normality,  # YENİ EKLENDİ
)
from src.domain.strategies import (
    build_weights,
    backtest_portfolio_assets,  # YENİ EKLENDİ
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
    prices = prices.sort_values(["fund_code", "date"])
    returns = compute_fund_daily_returns(prices)
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
    # Numeric conversion
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

    if mode == "AUM’a göre ilk K":
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
        "Veri dosyası bulunamadı. Lütfen `python -m scripts.update_data` komutunu çalıştırın."
    )
    st.stop()

master_data = loader.load_master_data()
macro_data = loader.load_macro_data()
master_data["date"] = pd.to_datetime(master_data["date"], errors="coerce")
min_date = master_data["date"].min().date()
max_date = master_data["date"].max().date()

d  # ... önceki kodlar ...

date_range = st.sidebar.date_input(
    "Tarih Aralığı",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# GÜVENLİ TARİH ÇÖZÜMLEME (YENİ KOD)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date_input, end_date_input = date_range
elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
    # Kullanıcı sadece başlangıç tarihini seçtiyse veya tek tarih döndüyse
    start_date_input = date_range[0]
    end_date_input = date_range[0]
else:
    # Hata durumunda varsayılan değerleri kullan
    start_date_input = min_date
    end_date_input = max_date


fund_type_options = (
    master_data["category"].dropna().astype(str).sort_values().unique().tolist()
)
selected_fund_types = st.sidebar.multiselect(
    "Fon Türü",
    options=fund_type_options,
)

fund_code_options = (
    master_data["fund_code"].dropna().astype(str).sort_values().unique().tolist()
)
selected_fund_codes = st.sidebar.multiselect(
    "Fon Kodu",
    options=fund_code_options,
)

st.sidebar.subheader("Sermaye Yönetimi (YENİ)")
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
    options=["AUM’a göre ilk K", "Rastgele K", "Manuel (fon kodlarını seç)"],
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

# 1. VERİ ÇEKME
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
    st.warning("Veri yok.")
    st.stop()

# 2. FİLTRELEME & SEÇİM
summary_all = summarize_funds(prices)
filtered_summary = summary_all.copy()
filtered_summary = filtered_summary[filtered_summary["aum_last"] >= min_aum]
filtered_summary = filtered_summary[filtered_summary["obs_days"] >= min_obs_days]
filtered_summary = filtered_summary[
    filtered_summary["missing_days"] <= missing_tolerance
]

if filtered_summary.empty:
    st.error("Filtrelere uygun fon bulunamadı.")
    st.stop()

manual_codes = []  # (Basitlik için manuel kısmı kısalttım, gerekirse eklenebilir)
if selection_mode == "Manuel (fon kodlarını seç)":
    manual_codes = selected_fund_codes
selected_funds, missing = select_funds(
    filtered_summary, selection_mode, int(k), int(seed), manual_codes
)

if not selected_funds:
    st.error("Fon seçilemedi.")
    st.stop()

# Seçilen fon sayısı gerçekte mevcut olandan az olabilir
actual_k = len(selected_funds)
initial_capital = actual_k * per_fund_capital

st.info(
    f"**Simülasyon Ayarları:** {actual_k} fon seçildi. Fon başı {per_fund_capital:,.0f} TL. Toplam Sermaye: **{initial_capital:,.0f} TL**"
)

# 3. BACKTEST
selected_prices = prices[prices["fund_code"].isin(selected_funds)]
returns_long = compute_fund_daily_returns(selected_prices)
returns_wide = returns_long.pivot_table(
    index="date", columns="fund_code", values="ret", aggfunc="mean"
).sort_index()

# Strateji Parametreleri
strategy_params = StrategyParams(
    k=actual_k,  # Seçilen havuzun tamamını kullanıyoruz
    seed=int(seed),
    strategy=strategy_key,
    lookback=lookback,
    rebalance=rebalance,
)

# Ağırlık Hesapla
weights_daily, weights_table = build_weights(returns_wide, strategy_params)

# YENİ: Gerçekçi Backtest Motorunu Çağır
prices_wide = selected_prices.pivot(index="date", columns="fund_code", values="price")
df_results = backtest_portfolio_assets(prices_wide, weights_table, initial_capital)

if df_results.empty:
    st.error("Backtest hesaplanamadı. Tarih aralığını veya verileri kontrol edin.")
    st.stop()

strategy_equity = df_results["equity"]  # TL değeri
strategy_returns = df_results["ret"]  # Günlük % getiri

# 4. SONUÇLAR
final_value = strategy_equity.iloc[-1]
net_profit = final_value - initial_capital
total_ret_pct = net_profit / initial_capital

col1, col2, col3 = st.columns(3)
col1.metric("Başlangıç Sermayesi", f"{initial_capital:,.0f} TL")
col2.metric("Bitiş Sermayesi", f"{final_value:,.0f} TL", f"%{total_ret_pct*100:.2f}")
col3.metric("Net Kâr/Zarar", f"{net_profit:,.0f} TL")

# GRAFİKLER
st.subheader("Portföy Değeri (TL)")
# Mevcut plot fonksiyonu normalize edilmiş (1.0'dan başlayan) equity bekliyor,
# O yüzden grafiği çizerken normalize edip gönderiyoruz
st.plotly_chart(
    plot_equity(strategy_equity / initial_capital), use_container_width=True
)

# İSTATİSTİK & MONTE CARLO SEKMELERİ
tab_stats, tab_monte, tab_metrics = st.tabs(
    ["📊 İstatistiksel Analiz", "🎲 Monte Carlo Simülasyonu", "📈 Metrikler & Tablo"]
)

with tab_stats:
    st.subheader("Normallik ve Kuyruk Riski Analizi")
    st.markdown("Getirilerin istatistiksel dağılımı (Fizikçi Gözüyle):")

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

with tab_monte:
    st.subheader(f"Şans Testi: {actual_k} Fonu Rastgele Seçseydik?")
    st.markdown(
        "Bu simülasyon, başarınızın 'strateji' mi yoksa sadece 'piyasanın yükselmesi' mi olduğunu test eder."
    )

    if st.button("Simülasyonu Başlat (1000 Tekrar)"):
        with st.spinner("Monte Carlo çalışıyor..."):
            # Hızlı simülasyon için Buy & Hold getirileri
            first_prices = prices_wide.iloc[0]
            last_prices = prices_wide.iloc[-1]
            fund_total_returns = (last_prices / first_prices) - 1.0
            fund_total_returns = fund_total_returns.fillna(0.0)

            valid_rets = fund_total_returns.values
            n_sims = 1000
            mc_results = []
            rng = np.random.default_rng()

            # Progress bar
            bar = st.progress(0)
            for i in range(n_sims):
                chosen = rng.choice(valid_rets, size=actual_k, replace=False)
                # Eşit ağırlıklı portföy getirisi = ortalama getiri
                mc_results.append(np.mean(chosen))
                if i % 100 == 0:
                    bar.progress((i + 1) / n_sims)
            bar.progress(1.0)

            mc_series = pd.Series(mc_results)

            # Görselleştirme
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

            # Yorum
            better_than_random = (total_ret_pct > mc_series).mean() * 100
            st.success(
                f"Sizin stratejiniz, rastgele oluşturulan portföylerin **%{better_than_random:.1f}** tanesinden daha iyi performans gösterdi."
            )

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
            "Makro veri bulunamadı. USD/TRY veya TÜFE ayarlamaları için `python -m scripts.update_data` çalıştırın."
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

        if use_usd:
            adjusted_equity = adjusted_equity / macro_aligned["usdtry"]

        if use_real:
            cpi_series = macro_aligned["cpi_us"] if use_usd else macro_aligned["cpi_tr"]
            inflation_index = cpi_series / cpi_series.iloc[0]
            adjusted_equity = adjusted_equity / inflation_index

    adjusted_returns = adjusted_equity.pct_change().fillna(0.0)

    # Rolling Metrics
    rolling_df = rolling_metrics(adjusted_returns, window=30)
    st.plotly_chart(plot_rolling_metrics(rolling_df, None), use_container_width=True)

    # Drawdown
    dd_series = drawdown_series(adjusted_equity)
    st.plotly_chart(plot_drawdown(dd_series), use_container_width=True)

    if macro_ready and st.checkbox("Karşılaştırma çizgilerini göster"):
        comparison_frames = []
        equity_norm = adjusted_equity / adjusted_equity.iloc[0]
        comparison_frames.append(
            equity_norm.rename("equity").to_frame().assign(fund_code="Strateji")
        )
        usdtry_norm = macro_aligned["usdtry"] / macro_aligned["usdtry"].iloc[0]
        comparison_frames.append(
            usdtry_norm.rename("equity").to_frame().assign(fund_code="USD/TRY")
        )
        cpi_base = macro_aligned["cpi_us"] if use_usd else macro_aligned["cpi_tr"]
        cpi_norm = cpi_base / cpi_base.iloc[0]
        comparison_frames.append(
            cpi_norm.rename("equity").to_frame().assign(fund_code="TÜFE")
        )
        comparison_df = (
            pd.concat(comparison_frames).reset_index().rename(columns={"index": "date"})
        )
        st.plotly_chart(plot_equity_comparison(comparison_df), use_container_width=True)

    # Fon Özeti Tablosu
    st.dataframe(filtered_summary)
