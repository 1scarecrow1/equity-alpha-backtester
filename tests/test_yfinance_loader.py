import numpy as np
import pandas as pd
import pytest
from equitybt.data import yfinance_loader as loader
from equitybt.trade_configs import Universe

BARS = pd.DataFrame(
    {
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"] * 2),
        "ticker": ["AAA"] * 3 + ["BBB"] * 3,
        "open": [10.0, 10.0, 5.0, 20.0, 20.0, 20.0],
        "high": [11.0, 11.0, 6.0, 21.0, 21.0, 21.0],
        "low": [9.0, 9.0, 4.0, 19.0, 19.0, 19.0],
        "close": [10.0, 10.0, 5.0, 20.0, 20.0, 19.5],
        "volume": [1e6, 1e6, 2e6, 3e6, 3e6, 3e6],
        "dividends": [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
        "splits": [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
        "shares_outstanding": [1e9] * 3 + [5e8] * 3,
    }
)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "CACHE_DIR", tmp_path / "bars")
    monkeypatch.setattr(loader, "RANGES_PATH", tmp_path / "bars" / "ranges.json")
    monkeypatch.setattr(loader, "STATIC_PATH", tmp_path / "static_fields.parquet")

    return tmp_path


def test_a_split_is_not_a_return():
    derived = loader._add_python_derived_fields(BARS.copy(), {"returns"})
    split_day = derived[(derived["ticker"] == "AAA") & (derived["splits"] > 0)]

    assert split_day["close"].iloc[0] == 5.0
    assert split_day["returns"].iloc[0] == pytest.approx(0.0)


def test_a_dividend_is_credited_not_lost():
    derived = loader._add_python_derived_fields(BARS.copy(), {"returns"})
    ex_date = derived[(derived["ticker"] == "BBB") & (derived["dividends"] > 0)]

    assert ex_date["close"].iloc[0] == 19.5
    assert ex_date["returns"].iloc[0] == pytest.approx(0.0)


def test_derived_fields_are_what_they_say():
    fields = {"typical_price", "dollar_volume", "adv20", "market_cap", "returns", "volatility20"}
    derived = loader._add_python_derived_fields(BARS.copy(), fields)
    row = derived.iloc[0]

    assert row["typical_price"] == pytest.approx((10.0 + 11.0 + 9.0) / 3)
    assert row["dollar_volume"] == pytest.approx(10.0 * 1e6)
    assert row["market_cap"] == pytest.approx(10.0 * 1e9)
    assert derived["adv20"].notna().all()


def test_only_the_requested_derived_fields_are_built():
    derived = loader._add_python_derived_fields(BARS.copy(), {"typical_price"})

    assert "typical_price" in derived.columns
    assert "adv20" not in derived.columns


def test_a_derived_field_pulls_in_what_it_needs():
    requested = loader._as_variable_list(["adv20", "market_cap"])
    names = {variable.name for variable in loader._expand_python_derived_variables(requested)}

    assert {"adv20", "dollar_volume", "close", "volume"} <= names
    assert {"market_cap", "shares_outstanding"} <= names


def test_get_tickers_passes_a_list_through_and_refuses_a_universe(tickers):
    assert loader.get_tickers(tickers) == tickers

    with pytest.raises(ValueError, match="does not name its members"):
        loader.get_tickers(Universe.TOP500_PIT)


def test_the_cache_is_reused_and_offline_never_downloads(cache, monkeypatch):
    calls = []

    def fake_download(names, start, end):
        calls.append(list(names))
        return BARS[BARS["ticker"].isin(names)].copy()

    monkeypatch.setattr(loader, "_download", fake_download)

    first = loader.load_bars(["AAA", "BBB"], "2024-01-02", "2024-01-04")
    second = loader.load_bars(["AAA", "BBB"], "2024-01-02", "2024-01-04")

    assert calls == [["AAA", "BBB"]]
    pd.testing.assert_frame_equal(first, second)

    offline = loader.load_bars(["AAA"], "2024-01-02", "2024-01-04", offline=True)
    assert list(offline["ticker"].unique()) == ["AAA"]
    assert len(calls) == 1


def test_a_range_wider_than_the_cache_is_downloaded_again(cache, monkeypatch):
    calls = []

    def fake_download(names, start, end):
        calls.append((start, end))
        return BARS[BARS["ticker"].isin(names)].copy()

    monkeypatch.setattr(loader, "_download", fake_download)

    loader.load_bars(["AAA"], "2024-01-02", "2024-01-04")
    loader.load_bars(["AAA"], "2024-01-01", "2024-01-04")

    assert calls == [("2024-01-02", "2024-01-04"), ("2024-01-01", "2024-01-04")]


def test_offline_says_which_tickers_are_missing(cache):
    with pytest.raises(RuntimeError, match="ZZZ"):
        loader.load_bars(["ZZZ"], "2024-01-02", "2024-01-04", offline=True)


def test_the_universe_mask_keeps_the_most_liquid_names():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 4 + ["2024-01-03"] * 4),
            "ticker": ["A", "B", "C", "D"] * 2,
            "adv20": [40.0, 30.0, 20.0, 10.0, 10.0, 20.0, 30.0, 40.0],
        }
    )
    masked = loader.add_universe_mask(df, 2)

    assert masked["in_universe"].tolist() == [True, True, False, False, False, False, True, True]


def test_an_unranked_name_is_out_of_the_universe():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 3),
            "ticker": ["A", "B", "C"],
            "adv20": [40.0, np.nan, 20.0],
        }
    )
    masked = loader.add_universe_mask(df, 3)

    assert masked["in_universe"].tolist() == [True, False, True]
