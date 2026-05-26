import re
from equitybt.data_fields import (
    EQUITIES_DATA_FIELD_CONFIG,
    get_variable_source,
    get_variable_sources,
)
from equitybt.dataexpression.variable import Variable

adv20: Variable
close: Variable
date: Variable
dividends: Variable
dollar_volume: Variable
high: Variable
industry: Variable
low: Variable
market_cap: Variable
open: Variable
returns: Variable
sector: Variable
shares_outstanding: Variable
splits: Variable
ticker: Variable
typical_price: Variable
volatility20: Variable
volume: Variable

def extract_known_variables(expression: str) -> list[str]:
    tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expression))
    return sorted(token for token in tokens if token in EQUITIES_DATA_FIELD_CONFIG)


def _install_variable_globals() -> list[str]:
    installed = []
    for variable_name in EQUITIES_DATA_FIELD_CONFIG:
        if not variable_name.isidentifier():
            continue
        if variable_name in globals():
            continue

        globals()[variable_name] = Variable(variable_name)
        installed.append(variable_name)

    return installed


VARIABLE_NAMES = _install_variable_globals()

__all__ = [
    "Variable",
    "get_variable_source",
    "get_variable_sources",
    "extract_known_variables",
    "VARIABLE_NAMES",
    *VARIABLE_NAMES,
]
