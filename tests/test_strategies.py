import pandas as pd
from src.domain.strategies import equal_weight_portfolio
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
    s = equal_weight_portfolio(df, StrategyParams(k=2, seed=1))
    assert len(s) == 2
    assert abs(s.iloc[0] - 0.02) < 1e-9
