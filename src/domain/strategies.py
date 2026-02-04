from __future__ import annotations

import numpy as np
import pandas as pd

from src.types import StrategyParams


def select_universe_k(fund_codes: list[str], k: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    if k >= len(fund_codes):
        return fund_codes
    idx = rng.choice(len(fund_codes), size=k, replace=False)
    return [fund_codes[i] for i in idx]


def compute_rebalance_dates(
    dates: pd.DatetimeIndex, frequency: str
) -> pd.DatetimeIndex:
    if dates.empty:
        return dates
    if frequency == "daily":
        return dates
    series = pd.Series(dates, index=dates)
    if frequency == "weekly":
        return series.groupby(dates.to_period("W")).last().values
    if frequency == "monthly":
        return series.groupby(dates.to_period("M")).last().values
    raise ValueError(f"Unknown rebalance frequency: {frequency}")


def _momentum_scores(window: pd.DataFrame) -> pd.Series:
    return (1.0 + window).prod(skipna=True) - 1.0


def _low_vol_scores(window: pd.DataFrame) -> pd.Series:
    return window.std(skipna=True)


def _sharpe_scores(window: pd.DataFrame) -> pd.Series:
    mean = window.mean(skipna=True)
    std = window.std(skipna=True)
    scores = mean / std.replace(0.0, np.nan)
    return scores


def _risk_parity_weights(window: pd.DataFrame) -> pd.Series:
    vol = window.std(skipna=True)
    inv_vol = 1.0 / vol.replace(0.0, np.nan)
    inv_vol = inv_vol.dropna()
    if inv_vol.empty:
        return pd.Series(dtype=float)
    weights = inv_vol / inv_vol.sum()
    return weights


def _min_variance_weights(window: pd.DataFrame) -> pd.Series:
    cov = window.cov()
    if cov.empty:
        return pd.Series(dtype=float)
    inv_cov = np.linalg.pinv(cov.values)
    ones = np.ones(inv_cov.shape[0])
    raw = inv_cov @ ones
    if np.all(np.isnan(raw)):
        return pd.Series(dtype=float)
    raw = np.clip(raw, 0.0, None)
    if raw.sum() == 0.0:
        return pd.Series(dtype=float)
    weights = raw / raw.sum()
    return pd.Series(weights, index=cov.columns)


def _select_top_k(scores: pd.Series, k: int, low_is_best: bool) -> list[str]:
    scores = scores.dropna()
    if scores.empty:
        return []
    if low_is_best:
        return scores.nsmallest(k).index.tolist()
    return scores.nlargest(k).index.tolist()


def build_weights(
    returns_wide: pd.DataFrame,
    params: StrategyParams,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.DatetimeIndex(returns_wide.index)
    rebalance_dates = compute_rebalance_dates(dates, params.rebalance)
    weights_by_date: dict[pd.Timestamp, pd.Series] = {}

    for date in rebalance_dates:
        idx = dates.get_loc(date)
        if params.strategy == "equal_weight":
            selected = returns_wide.columns.tolist()
            weights = None
        else:
            if idx < params.lookback:
                continue
            window = returns_wide.iloc[idx - params.lookback : idx]
            if params.strategy == "momentum_top_k":
                scores = _momentum_scores(window)
                selected = _select_top_k(scores, params.k, low_is_best=False)
                weights = None
            elif params.strategy == "low_vol_top_k":
                scores = _low_vol_scores(window)
                selected = _select_top_k(scores, params.k, low_is_best=True)
                weights = None
            elif params.strategy == "sharpe_top_k":
                scores = _sharpe_scores(window)
                selected = _select_top_k(scores, params.k, low_is_best=False)
                weights = None
            elif params.strategy == "risk_parity":
                weights = _risk_parity_weights(window)
                selected = weights.index.tolist()
            elif params.strategy == "min_variance":
                weights = _min_variance_weights(window)
                selected = weights.index.tolist()
            else:
                raise ValueError(f"Unknown strategy: {params.strategy}")

        if not selected:
            continue
        if weights is None:
            weight = 1.0 / len(selected)
            weights_by_date[pd.Timestamp(date)] = pd.Series(
                weight, index=selected, name="weight"
            )
        else:
            weights_by_date[pd.Timestamp(date)] = weights.rename("weight")

    if not weights_by_date:
        return pd.DataFrame(), pd.DataFrame()

    weights_daily = pd.DataFrame(index=dates, columns=returns_wide.columns, dtype=float)
    rows = []
    for date, weights in weights_by_date.items():
        weights_daily.loc[date, weights.index] = weights.values
        df = weights.reset_index()
        df.columns = ["fund_code", "weight"]
        df["rebalance_date"] = date
        rows.append(df)

    weights_daily = weights_daily.sort_index().ffill()
    first_date = min(weights_by_date.keys())
    weights_daily = weights_daily.loc[first_date:]

    weights_table = pd.concat(rows, ignore_index=True)[
        ["rebalance_date", "fund_code", "weight"]
    ]
    return weights_daily, weights_table


def compute_portfolio_returns(
    returns_wide: pd.DataFrame, weights_daily: pd.DataFrame
) -> pd.Series:
    if weights_daily.empty:
        return pd.Series(dtype=float, name="portfolio_ret")
    aligned_returns = returns_wide.loc[weights_daily.index]
    port_ret = (aligned_returns.fillna(0.0) * weights_daily.fillna(0.0)).sum(axis=1)
    port_ret = port_ret.dropna()
    port_ret.name = "portfolio_ret"
    return port_ret


def equal_weight_portfolio(
    returns_long: pd.DataFrame, params: StrategyParams
) -> pd.Series:
    """
    returns_long: [date, fund_code, ret]
    Output: portfolio daily return series indexed by date
    """
    returns_wide = returns_long.pivot_table(
        index="date", columns="fund_code", values="ret", aggfunc="mean"
    ).sort_index()
    weights_daily, _ = build_weights(returns_wide, params)
    return compute_portfolio_returns(returns_wide, weights_daily)


def backtest_portfolio_assets(
    prices: pd.DataFrame, weights_table: pd.DataFrame, initial_capital: float
) -> pd.DataFrame:
    """
    Gerçekçi Backtest Motoru (Asset/Share Based):
    - Belirli tarihlerde (rebalance) portföyü hedef ağırlıklara göre yeniden kurar.
    - Ara günlerde 'shares * price' mantığıyla değer taşır (Weight Drift'e izin verir).

    Args:
        prices: [index=date, columns=fund_code] (Wide format fiyatlar)
        weights_table: [rebalance_date, fund_code, weight]
        initial_capital: Başlangıç sermayesi (örn: 100.000 TL)

    Returns:
        pd.DataFrame: [equity, ret] index=date
    """
    if weights_table.empty or prices.empty:
        return pd.DataFrame()

    # Rebalance tarihlerini sırala
    rebalance_dates = sorted(weights_table["rebalance_date"].unique())

    # Analizi ilk rebalance gününden başlat (Örn: İlk işlem günü)
    start_date = rebalance_dates[0]
    prices = prices.loc[start_date:].copy()
    dates = prices.index

    # Hazırlık: Hedef ağırlıkları pivot tabloya çevir
    target_weights_df = weights_table.pivot(
        index="rebalance_date", columns="fund_code", values="weight"
    ).fillna(0.0)

    # Simülasyon değişkenleri
    current_cash = initial_capital
    current_shares = pd.Series(0.0, index=prices.columns)

    # Günlük portföy değerlerini saklayacak sözlük
    p_values = {}

    # Rebalance günlerini hızlı kontrol için set'e çevir
    reb_set = set(rebalance_dates)

    for d in dates:
        # 1. O günkü fiyatlar
        # (Fiyatı olmayan fonlar NaN gelebilir, fillna(0) ile değerini 0 sayıyoruz)
        p = prices.loc[d].fillna(0.0)

        # 2. Rebalance Öncesi Portföy Değeri Hesapla
        # Mevcut hisseler * Bugünkü fiyat + Nakit
        val_assets = (current_shares * p).sum()
        total_value = current_cash + val_assets

        # 3. Eğer bugün rebalance günü ise portföyü yeniden dağıt
        if d in reb_set:
            # O gün için hedeflenen ağırlıklar
            w = target_weights_df.loc[d]

            # Elimizdeki toplam parayı (Hisse + Nakit) ağırlıklara göre bölüştür
            # (İşlem maliyeti 0 varsayıyoruz)
            target_amounts = total_value * w

            # Yeni hisse adetleri = Hedef Tutar / Fiyat
            # Fiyatı 0 olan fona bölersek sonsuz çıkar, onu 0 yapalım
            new_shares = target_amounts / p
            new_shares = new_shares.replace([np.inf, -np.inf], 0.0).fillna(0.0)

            current_shares = new_shares
            current_cash = 0.0  # Full invest varsayımı (küsüratlar ihmal)

            # Rebalance sonrası değeri teyit et (Kontrol amaçlı)
            val_assets = (current_shares * p).sum()
            total_value = current_cash + val_assets

        p_values[d] = total_value

    # Sonuç serisi: Equity Curve (TL cinsinden)
    result = pd.Series(p_values, name="equity")

    # Günlük Getiri hesapla (Analizler için gerekli: % değişim)
    df_res = pd.DataFrame(result)
    df_res["ret"] = df_res["equity"].pct_change().fillna(0.0)

    return df_res
