import pandas as pd

from src.domain.stats import rolling_metrics


def test_rolling_metrics_length_and_nans():
    daily_ret = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01, 0.02, 0.0, 0.01, -0.01, 0.02])
    metrics = rolling_metrics(daily_ret, window=5)
    assert len(metrics) == len(daily_ret)
    assert metrics["rolling_vol"].iloc[:4].isna().all()
    assert metrics["rolling_sharpe"].iloc[:4].isna().all()
    assert metrics["rolling_max_drawdown"].iloc[:4].isna().all()
