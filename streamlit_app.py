import pandas as pd
import streamlit as st

from src.data.provider_tefaslib import TefasLibProvider
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
)
from src.domain.strategies import (
    build_weights,
    compute_portfolio_returns,
    select_universe_k,
)
from src.types import FetchParams, StrategyParams
from src.viz.plots import (
    plot_correlation_heatmap,
    plot_drawdown,
    plot_equity,
    plot_equity_comparison,
    plot_return_histogram,
    plot_rolling_metrics,
)

st.set_page_config(page_title="TEFAS Strateji Simülatörü", layout="wide")


@st.cache_data(show_spinner=False)
def load_prices(provider_name: str, start: str, end: str, universe: str) -> pd.DataFrame:
    provider = TefasLibProvider()
    params = FetchParams(start=start, end=end, universe=universe)
    return provider.fetch_prices(params)


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
    summary["sharpe_est"] = ret_group.apply(sharpe)
    summary["max_drawdown_fund"] = ret_group.apply(
        lambda s: max_drawdown(equity_curve(s))
    )
    summary["worst_day"] = ret_group.min()
    summary["best_day"] = ret_group.max()
    summary["missing_days"] = expected_days - summary["obs_days"]
    summary = summary.reset_index()
    summary["aum_last"] = pd.to_numeric(summary["aum_last"], errors="coerce").fillna(0.0)
    summary["price_last"] = pd.to_numeric(summary["price_last"], errors="coerce")
    summary["start_price"] = pd.to_numeric(summary["start_price"], errors="coerce")
    summary["end_price"] = pd.to_numeric(summary["end_price"], errors="coerce")
    return summary


def parse_date(value: str) -> pd.Timestamp | None:
    try:
        parsed = pd.to_datetime(value, dayfirst=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    return parsed.normalize()


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


def build_benchmark(
    prices: pd.DataFrame,
    benchmark_mode: str,
    strategy_params: StrategyParams,
) -> tuple[pd.Series, pd.DataFrame]:
    summary = summarize_funds(prices)
    if benchmark_mode == "Eşit Ağırlık (Toplam AUM ilk 200)":
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


def format_percent(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value * 100:.2f}%"


st.title("TEFAS Strateji Simülatörü (Sprint 4)")

st.sidebar.header("Girdiler")

start_input = st.sidebar.text_input("Başlangıç (gg.aa.yyyy)", value="01.01.2025")
end_input = st.sidebar.text_input("Bitiş (gg.aa.yyyy)", value="15.01.2025")

provider_name = "tefasfon"
universe = st.sidebar.selectbox("Evren", options=["free"], index=0)

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
name_query = st.sidebar.text_input("Fon adı arama")
code_query = st.sidebar.text_input("Fon kodu arama")

st.sidebar.subheader("Fon Seçimi")

k = st.sidebar.slider("K (fon sayısı)", min_value=10, max_value=500, value=200)
selection_mode = st.sidebar.radio(
    "Seçim modu",
    options=["AUM’a göre ilk K", "Rastgele K", "Manuel (fon kodlarını seç)"],
    index=0,
)

seed = 42
if selection_mode == "Rastgele K":
    seed = st.sidebar.number_input(
        "Rastgelelik tohumu (seed)", min_value=0, max_value=10_000, value=42, step=1
    )

strategy_label = st.sidebar.selectbox(
    "Strateji",
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
    lookback = st.sidebar.slider(
        "Geriye bakış (gün)", min_value=5, max_value=252, value=20
    )

rebalance = st.sidebar.selectbox(
    "Yeniden dengeleme sıklığı", options=["daily", "weekly", "monthly"], index=2
)

show_benchmark = st.sidebar.checkbox("Kıyas (Benchmark) göster", value=True)
benchmark_mode = st.sidebar.selectbox(
    "Kıyas (Benchmark)",
    options=["Eşit Ağırlık (Toplam AUM ilk 200)", "Eşit Ağırlık (Tüm evren)"],
    index=0,
)

debug_mode = st.sidebar.checkbox("Hata ayıklama modu", value=False)
if st.sidebar.button("Önbelleği temizle"):
    st.cache_data.clear()
    st.sidebar.success("Önbellek temizlendi")

start_date = parse_date(start_input)
end_date = parse_date(end_input)
if start_date is None or end_date is None:
    st.error("Geçersiz tarih girişi. gg.aa.yyyy formatını kullanın.")
    st.stop()
if start_date > end_date:
    st.error("Başlangıç tarihi bitiş tarihinden önce olmalıdır.")
    st.stop()

st.caption(
    f"Sağlayıcı: {provider_name} | Evren: {universe} | Tarih aralığı: {start_input} – {end_input}"
)

try:
    prices = load_prices(provider_name, start_input, end_input, universe)
except Exception as exc:
    st.error("Veri çekme başarısız oldu.")
    st.code(str(exc))
    st.stop()

if prices.empty:
    st.error("Seçilen tarih aralığında veri bulunamadı.")
    st.stop()

summary_all = summarize_funds(prices)

filtered_summary = summary_all.copy()
filtered_summary = filtered_summary[filtered_summary["aum_last"] >= min_aum]
filtered_summary = filtered_summary[filtered_summary["obs_days"] >= min_obs_days]
filtered_summary = filtered_summary[
    filtered_summary["missing_days"] <= missing_tolerance
]
if name_query:
    filtered_summary = filtered_summary[
        filtered_summary["fund_name"].str.contains(name_query, case=False, na=False)
    ]
if code_query:
    filtered_summary = filtered_summary[
        filtered_summary["fund_code"].str.contains(code_query, case=False, na=False)
    ]

available_count = len(filtered_summary)
if available_count == 0:
    st.error("Filtre sonrası uygun fon bulunamadı.")
    st.stop()
if selection_mode != "Manuel (fon kodlarını seç)" and available_count < k:
    st.warning(
        "Filtre sonrası fon sayısı K değerinden az. K otomatik güncellendi."
    )
    k = available_count

manual_codes: list[str] = []
manual_multiselect: list[str] = []
manual_text_codes: list[str] = []
if selection_mode == "Manuel (fon kodlarını seç)":
    manual_multiselect = st.sidebar.multiselect(
        "Fon kodlarını seç",
        options=sorted(filtered_summary["fund_code"].tolist()),
    )
    raw_codes = st.sidebar.text_area("Fon kodlarını virgülle gir")
    manual_text_codes = [
        code.strip().upper() for code in raw_codes.replace("\n", ",").split(",")
        if code.strip()
    ]
    manual_codes = sorted(set(manual_multiselect + manual_text_codes))
    if manual_codes:
        k = len(manual_codes)

selected_funds, missing_manual = select_funds(
    filtered_summary, selection_mode, int(k), int(seed), manual_codes
)

if missing_manual:
    st.warning(f"Bazı fon kodları bulunamadı: {', '.join(missing_manual)}")

if not selected_funds:
    st.error("Seçilen fon bulunamadı. Filtreleri veya seçim modunu kontrol edin.")
    st.stop()

selected_prices = prices[prices["fund_code"].isin(selected_funds)]
summary_selected = summary_all[summary_all["fund_code"].isin(selected_funds)].copy()

returns_long = compute_fund_daily_returns(selected_prices)
returns_wide = returns_long.pivot_table(
    index="date", columns="fund_code", values="ret", aggfunc="mean"
).sort_index()

if strategy_key != "equal_weight" and lookback >= len(returns_wide.index):
    st.warning(
        "Geriye bakış için yeterli gün yok. Tarih aralığını genişletin veya geriye bakışı azaltın."
    )
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
    st.warning(
        "Geriye bakış için yeterli gün yok. Tarih aralığını genişletin veya geriye bakışı azaltın."
    )
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

st.subheader("Fon Özeti (fon başına 1 satır)")
summary_display = summary_selected.copy()
summary_display = summary_display[
    [
        "fund_code",
        "fund_name",
        "aum_last",
        "price_last",
        "period_return",
        "vol_daily",
        "sharpe_est",
        "max_drawdown_fund",
        "worst_day",
        "best_day",
        "missing_days",
        "obs_days",
    ]
]

sort_option = st.selectbox(
    "Sıralama",
    options=[
        "AUM desc",
        "Toplam getiri desc",
        "Sharpe desc",
        "Vol asc",
        "MDD asc (daha az negatif iyi)",
    ],
    index=0,
)

if sort_option == "AUM desc":
    summary_display = summary_display.sort_values("aum_last", ascending=False)
elif sort_option == "Toplam getiri desc":
    summary_display = summary_display.sort_values("period_return", ascending=False)
elif sort_option == "Sharpe desc":
    summary_display = summary_display.sort_values("sharpe_est", ascending=False)
elif sort_option == "Vol asc":
    summary_display = summary_display.sort_values("vol_daily", ascending=True)
elif sort_option == "MDD asc (daha az negatif iyi)":
    summary_display = summary_display.sort_values("max_drawdown_fund", ascending=False)

summary_display = summary_display.rename(
    columns={
        "fund_code": "Fon Kodu",
        "fund_name": "Fon Adı",
        "aum_last": "AUM (son)",
        "price_last": "Son Fiyat",
        "period_return": "getiri_toplam",
        "vol_daily": "vol_gunluk",
        "sharpe_est": "sharpe_tahmini",
        "max_drawdown_fund": "max_drawdown_fon",
        "worst_day": "en_kotu_gun",
        "best_day": "en_iyi_gun",
        "missing_days": "eksik_gun",
        "obs_days": "gozlem_gunu",
    }
)

st.dataframe(summary_display, use_container_width=True)

st.subheader("Performans (Strateji vs Kıyas)")

str_cols = st.columns(4)
str_cols[0].metric("Toplam Getiri", format_percent(total_return(strategy_equity)))
str_cols[1].metric("Yıllık Volatilite", format_percent(annualized_vol(strategy_returns)))
str_cols[2].metric("Maksimum Düşüş (MDD)", format_percent(max_drawdown(strategy_equity)))
str_cols[3].metric("Sharpe Oranı", f"{sharpe(strategy_returns):.2f}")

if show_benchmark and not benchmark_returns.empty:
    bench_cols = st.columns(4)
    bench_cols[0].metric(
        "Kıyas Toplam Getiri", format_percent(total_return(benchmark_equity))
    )
    bench_cols[1].metric(
        "Kıyas Yıllık Volatilite", format_percent(annualized_vol(benchmark_returns))
    )
    bench_cols[2].metric(
        "Kıyas Maksimum Düşüş (MDD)", format_percent(max_drawdown(benchmark_equity))
    )
    bench_cols[3].metric("Kıyas Sharpe Oranı", f"{sharpe(benchmark_returns):.2f}")

st.subheader("Equity Eğrisi")
st.caption(f"Yeniden dengeleme: {rebalance} | Geriye bakış: {lookback} gün")
st.plotly_chart(
    plot_equity(strategy_equity, benchmark_equity if show_benchmark else None),
    use_container_width=True,
)

st.subheader("Drawdown")
st.plotly_chart(
    plot_drawdown(strategy_drawdown, benchmark_drawdown if show_benchmark else None),
    use_container_width=True,
)

st.subheader("Yuvarlanan (Rolling) Metrikler")
rolling_window = st.slider(
    "Rolling pencere (gün)", min_value=30, max_value=252, value=63
)
strategy_rolling = rolling_metrics(strategy_returns, window=rolling_window)
if show_benchmark and not benchmark_returns.empty:
    benchmark_rolling = rolling_metrics(benchmark_returns, window=rolling_window)
else:
    benchmark_rolling = pd.DataFrame()

st.plotly_chart(
    plot_rolling_metrics(strategy_rolling, benchmark_rolling),
    use_container_width=True,
)

st.subheader("Korelasyon (Seçilen Fonlar)")
max_corr_funds = st.slider(
    "Maks fon sayısı (korelasyon)", min_value=5, max_value=60, value=30
)

summary_for_corr = summary_selected.sort_values("aum_last", ascending=False).head(
    max_corr_funds
)
returns_corr = returns_wide[summary_for_corr["fund_code"].tolist()]

corr_matrix = correlation_matrix(returns_corr)
pair_table = pd.DataFrame()
if not corr_matrix.empty:
    st.plotly_chart(plot_correlation_heatmap(corr_matrix), use_container_width=True)

    pair_table = correlation_pairs(corr_matrix).dropna()
    pair_display = pair_table.rename(
        columns={"fund_a": "fon_a", "fund_b": "fon_b", "corr": "korelasyon"}
    )

    st.markdown("**En yüksek korelasyonlu 10 çift**")
    st.dataframe(pair_display.head(10), use_container_width=True)

    st.markdown("**En düşük korelasyonlu 10 çift**")
    st.dataframe(
        pair_display.tail(10).sort_values("korelasyon"), use_container_width=True
    )
else:
    st.info("Korelasyon için yeterli veri yok.")

st.subheader("Fon Karşılaştır")
compare_funds = st.multiselect(
    "Karşılaştırılacak fonlar (2-5 adet)",
    options=summary_selected["fund_code"].tolist(),
)

if 2 <= len(compare_funds) <= 5:
    compare_returns = returns_long[returns_long["fund_code"].isin(compare_funds)]
    equity_frames = []
    for fund_code, series in compare_returns.groupby("fund_code"):
        series = series.sort_values("date")
        equity = equity_curve(series.set_index("date")["ret"], start_value=1.0)
        frame = equity.reset_index()
        frame.columns = ["date", "equity"]
        frame["fund_code"] = fund_code
        equity_frames.append(frame)
    equity_df = pd.concat(equity_frames, ignore_index=True)
    st.plotly_chart(
        plot_equity_comparison(equity_df),
        use_container_width=True,
    )

    st.plotly_chart(
        plot_return_histogram(
            compare_returns.rename(columns={"fund_code": "fund_code"})
        ),
        use_container_width=True,
    )

    compare_metrics = (
        summary_selected[summary_selected["fund_code"].isin(compare_funds)]
        .set_index("fund_code")
        .loc[compare_funds]
        .reset_index()
    )
    compare_metrics = compare_metrics[
        [
            "fund_code",
            "period_return",
            "vol_daily",
            "max_drawdown_fund",
            "sharpe_est",
        ]
    ]
    compare_metrics = compare_metrics.rename(
        columns={
            "fund_code": "Fon Kodu",
            "period_return": "Toplam Getiri",
            "vol_daily": "Günlük Vol",
            "max_drawdown_fund": "MDD",
            "sharpe_est": "Sharpe",
        }
    )
    st.dataframe(compare_metrics, use_container_width=True)
else:
    st.info("Karşılaştırma için 2-5 fon seçin.")

if debug_mode:
    st.subheader("Debug: Strateji Günlük Getirileri")
    returns_table = strategy_returns.reset_index().rename(
        columns={"date": "Tarih", "portfolio_ret": "Getiri"}
    )
    st.dataframe(returns_table, use_container_width=True)

    st.subheader("Debug: Ağırlıklar (rebalance günleri)")
    if weights_table.empty:
        st.write("Ağırlık bulunamadı.")
    else:
        st.dataframe(weights_table, use_container_width=True)

st.subheader("İndir")

summary_csv = summary_display.to_csv(index=False).encode("utf-8")
rolling_df = strategy_rolling.copy()
rolling_df.index.name = "date"
returns_csv = (
    strategy_returns.reset_index()
    .rename(columns={"portfolio_ret": "ret"})
    .assign(
        equity=strategy_equity.values,
        drawdown=strategy_drawdown.values,
        rolling_vol=rolling_df["rolling_vol"].values,
        rolling_sharpe=rolling_df["rolling_sharpe"].values,
    )
    .to_csv(index=False)
    .encode("utf-8")
)

st.download_button(
    "fon_ozeti.csv indir",
    data=summary_csv,
    file_name="fon_ozeti.csv",
    mime="text/csv",
)

st.download_button(
    "strateji_serisi.csv indir",
    data=returns_csv,
    file_name="strateji_serisi.csv",
    mime="text/csv",
)

weights_csv = weights_table.to_csv(index=False).encode("utf-8")
st.download_button(
    "agirliklar.csv indir",
    data=weights_csv,
    file_name="agirliklar.csv",
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
        "benchmark_serisi.csv indir",
        data=bench_csv,
        file_name="benchmark_serisi.csv",
        mime="text/csv",
    )

if not pair_table.empty:
    pairs_export = pd.concat(
        [
            pair_table.head(10).assign(tip="en_yuksek"),
            pair_table.tail(10).sort_values("corr").assign(tip="en_dusuk"),
        ],
        ignore_index=True,
    )
    pairs_csv = pairs_export.to_csv(index=False).encode("utf-8")
else:
    pairs_csv = b""
st.download_button(
    "korelasyon_en_iyi_kotu_ciftler.csv indir",
    data=pairs_csv,
    file_name="korelasyon_en_iyi_kotu_ciftler.csv",
    mime="text/csv",
)
