"""
Data-field mappings for daily equity price data.
"""

from typing import Any

EQUITIES_DATA_FIELD_CONFIG = {
    "date": {
        "description": "Observation date",
    },
    "ticker": {
        "description": "Ticker",
    },
    "open": {
        "description": "Opening price, unadjusted",
    },
    "high": {
        "description": "Highest price of the day, unadjusted",
    },
    "low": {
        "description": "Lowest price of the day, unadjusted",
    },
    "close": {
        "description": "Closing price, unadjusted",
    },
    "volume": {
        "description": "Shares traded",
    },
    "dividends": {
        "description": "Dividend per share, on the ex-date",
    },
    "splits": {
        "description": "Split ratio on the effective date, 0 on every other date",
    },
    "sector": {
        "description": "Sector classification",
    },
    "industry": {
        "description": "Industry classification",
    },
    "shares_outstanding": {
        "description": "Shares outstanding, as reported now",
    },
    "returns": {
        "description": "Close-to-close total return, adjusted for splits and dividends",
        "derived": {
            "method": "python",
            "function": "total_return",
            "dependencies": ["close", "dividends", "splits"],
            "group_by": ["ticker"],
            "order_by": "date",
        },
    },
    "typical_price": {
        "description": "Mean of high, low and close. A proxy for VWAP, which daily bars do not carry",
        "derived": {
            "method": "python",
            "function": "typical_price",
            "dependencies": ["high", "low", "close"],
        },
    },
    "dollar_volume": {
        "description": "Close times volume",
        "derived": {
            "method": "python",
            "function": "dollar_volume",
            "dependencies": ["close", "volume"],
        },
    },
    "market_cap": {
        "description": "Close times current shares outstanding",
        "derived": {
            "method": "python",
            "function": "market_cap",
            "dependencies": ["close", "shares_outstanding"],
        },
    },
    "adv20": {
        "description": "Average daily dollar volume in past 20 days",
        "derived": {
            "method": "python",
            "function": "rolling_mean",
            "dependencies": ["dollar_volume"],
            "source_column": "dollar_volume",
            "window": 20,
            "min_periods": 1,
            "group_by": ["ticker"],
            "order_by": "date",
            "warmup_days": 60,
        },
    },
    "volatility20": {
        "description": "Standard deviation of returns in past 20 days",
        "derived": {
            "method": "python",
            "function": "rolling_std",
            "dependencies": ["returns"],
            "source_column": "returns",
            "window": 20,
            "min_periods": 20,
            "group_by": ["ticker"],
            "order_by": "date",
            "warmup_days": 60,
        },
    },
}

PRICE_FIELDS = ("open", "high", "low", "close")
STATIC_FIELDS = ("sector", "industry", "shares_outstanding")


def get_variable_source(variable_name: str) -> dict[str, Any]:
    field_config = EQUITIES_DATA_FIELD_CONFIG.get(variable_name)
    if not field_config:
        raise KeyError(f"Unknown variable: {variable_name}")

    derived = field_config.get("derived")

    return {
        "variable_name": variable_name,
        "column_name": variable_name,
        "description": field_config["description"],
        "derived": derived,
        "is_custom": bool(derived),
    }


def get_variable_sources(
    variable_names: list[str] | tuple[str, ...] | set[str],
) -> dict[str, dict[str, Any]]:
    return {name: get_variable_source(name) for name in variable_names}


def get_custom_variable_definitions(
    config: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    config = config or EQUITIES_DATA_FIELD_CONFIG
    return {
        variable_name: field_config["derived"]
        for variable_name, field_config in config.items()
        if field_config.get("derived")
    }


def get_python_derived_fields(
    config: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        variable_name: definition
        for variable_name, definition in get_custom_variable_definitions(config).items()
        if definition.get("method") == "python"
    }


PYTHON_DERIVED_FIELDS = get_python_derived_fields()
