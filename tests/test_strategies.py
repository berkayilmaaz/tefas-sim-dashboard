import pandas as pd
import pytest

from src.domain.strategies import build_weights, equal_weight_portfolio
from src.types import StrategyParams


def test_equal_weight():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"]
            ),
            "fund_code": ["AAA", "BBB", "AAA", "BBB"],
            "ret": [0.01, 0.03, 0.02, 0.00],
        }
    )
    s = equal_weight_portfolio(df, StrategyParams(k=2, seed=1, rebalance="daily"))
    assert len(s) == 2
    assert abs(s.iloc[0] - 0.02) < 1e-9


def test_risk_parity_weights_sum_to_one():
    dates = pd.date_range("2025-01-01", periods=5, freq="D")
    returns_wide = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, -0.01, 0.03, 0.00],
            "BBB": [0.00, -0.01, 0.02, 0.01, 0.02],
        },
        index=dates,
    )
    params = StrategyParams(
        k=2, seed=1, strategy="risk_parity", lookback=2, rebalance="daily"
    )
    weights_daily, _ = build_weights(returns_wide, params)
    first_weights = weights_daily.iloc[0].fillna(0.0)
    assert first_weights.sum() == pytest.approx(1.0)
    assert (first_weights >= 0.0).all()


def test_min_variance_weights_sum_to_one():
    dates = pd.date_range("2025-01-01", periods=5, freq="D")
    returns_wide = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, -0.01, 0.03, 0.00],
            "BBB": [0.00, -0.01, 0.02, 0.01, 0.02],
        },
        index=dates,
    )
    params = StrategyParams(
        k=2, seed=1, strategy="min_variance", lookback=2, rebalance="daily"
    )
    weights_daily, _ = build_weights(returns_wide, params)
    first_weights = weights_daily.iloc[0].fillna(0.0)
    assert first_weights.sum() == pytest.approx(1.0)
    assert (first_weights >= 0.0).all()
