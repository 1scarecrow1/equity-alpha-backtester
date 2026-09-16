from equitybt.strategy.alpha import Alpha
from equitybt.strategy.operators import if_else, inverse
from equitybt.variables.equities import close, market_cap, volatility20


class EquallyWeighted(Alpha):
    def __init__(self):
        super().__init__(
            signals={'eq_weighted': if_else(close > 0, 1, 0)}
        )

class InverseVolatilityWeighted(Alpha):
    def __init__(self):
        super().__init__(
            signals={'inv_vol_weighted': inverse(volatility20)}
        )

class MarketCapWeighted(Alpha):
    def __init__(self):
        super().__init__(
            signals={'mkt_cap_weighted': market_cap}
        )
