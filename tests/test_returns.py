import pandas as pd
from src.domain.returns import compute_fund_daily_returns


def test_returns_shape():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"]
            ),
            "fund_code": ["AAA", "AAA", "BBB", "BBB"],
            "fund_name": ["A", "A", "B", "B"],
            "price": [100, 110, 200, 220],
        }
    )
    rets = compute_fund_daily_returns(prices)
    assert "ret" in rets.columns
    assert len(rets) == 2
