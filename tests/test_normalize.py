import pandas as pd

from src.data.normalize import normalize_prices


def test_normalize_prices_required_columns_and_types():
    df = pd.DataFrame(
        {
            "Tarih": ["01.01.2025", "02.01.2025"],
            "Fon Kodu": ["AAA", "AAA"],
            "Fon Adı": ["Fon A", "Fon A"],
            "Fiyat": [100.0, 101.0],
        }
    )
    out = normalize_prices(df)
    assert set(["date", "fund_code", "fund_name", "price"]).issubset(out.columns)
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert pd.api.types.is_numeric_dtype(out["price"])
    assert out["fund_code"].dtype == object
