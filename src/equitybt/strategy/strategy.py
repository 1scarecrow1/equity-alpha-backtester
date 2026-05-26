import numbers
import numpy as np
from equitybt.strategy.alpha import Alpha
from equitybt.trade_configs import (
    Decay,
    Delay,
    InstrumentType,
    Neutralization,
    Region,
    Truncation,
    Universe,
    UniverseSize,
)


class Strategy:

    _immutable_fields = {"instrument_type", "region"}

    def __init__(self, alpha=None, **configs):
        self.instrument_type = configs.get("instrument_type", InstrumentType.EQUITY)
        self.region = configs.get("region", Region.USA)

        self.universe = configs.get("universe", Universe.TOP500_PIT)
        self.universe_size = UniverseSize[self.universe.name] \
            if isinstance(self.universe, Universe) else len(self.universe)

        if alpha is None and any(key in configs for key in ("signal", "signals", "constraint", "constraints")):
            alpha = Alpha(
                signal=configs.get("signal"),
                signals=configs.get("signals"),
                constraint=configs.get("constraint"),
                constraints=configs.get("constraints"),
            )

        self.alpha = alpha

        self.delay = configs.get("delay", Delay.ONE)
        self.decay = configs.get("decay", Decay.NONE)
        self.truncation = configs.get("truncation", 0.0)
        self.neutralization = configs.get("neutralization", Neutralization.NONE)

        self.weights = np.zeros((10, self.universe_size)) 
        self.aggregate_metrics = {}

        self.df = configs.get("df")
        self.daily = configs.get("daily")
        self.summary = configs.get("summary", {})
        self.figures = configs.get("figures", {})
        self.results = configs.get("results", {})
        
    @property
    def instrument_type(self):
        return self._instrument_type

    @property
    def region(self):
        return self._region

    @property
    def alpha(self):
        return self._alpha
    
    @property
    def signal(self):
        return self._alpha.signal
    
    @property
    def constraints(self):
        return self._alpha.constraints

    @property
    def neutralization(self):
        return self._neutralization

    @property
    def delay(self):
        return self._delay

    @property
    def decay(self):
        return self._decay

    @property
    def truncation(self):
        return self._truncation

    def __getattr__(self, name):
        alpha = self.__dict__.get("_alpha")
        if alpha is not None and hasattr(alpha, name):
            return getattr(alpha, name)
        
        metrics = self.__dict__.get("aggregate_metrics")
        if metrics is not None and name in metrics:
            return metrics[name]

        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
    
    def __setattr__(self, name, value):
        if name in self._immutable_fields and f"_{name}" in self.__dict__:
            raise AttributeError(f"{name} is immutable.")
        super().__setattr__(name, value)

    @instrument_type.setter
    def instrument_type(self, instrument_type: InstrumentType):
        if not isinstance(instrument_type, InstrumentType):
            raise ValueError(f"Instrument type must be from following options: {InstrumentType.__members__}")
        self._instrument_type = instrument_type

    @region.setter
    def region(self, region: Region):
        if not isinstance(region, Region):
            raise ValueError(f"Region must be from following options: {Region.__members__}")
        self._region = region

    @alpha.setter
    def alpha(self, alpha: Alpha | None):
        if alpha is not None and not isinstance(alpha, Alpha):
            raise ValueError("Pass an alpha object")
        self._alpha = alpha

    @delay.setter
    def delay(self, delay: Delay):
        if not isinstance(delay, Delay):
            raise ValueError(f"Delay must be from following options: {Delay.__members__}")
        self._delay = delay

    @neutralization.setter
    def neutralization(self, neutralization: Neutralization):
        if not isinstance(neutralization, Neutralization):
            raise ValueError(f"Neutralization must be from following options: {Neutralization.__members__}")
        self._neutralization = neutralization

    @decay.setter
    def decay(self, decay: int):
        if not (isinstance(decay, numbers.Integral) and decay >= 0):
            raise ValueError("Decay must be a non-negative integer") 
        self._decay = decay

    @truncation.setter
    def truncation(self, truncation: float | Truncation):
        value = truncation.value if isinstance(truncation, Truncation) else truncation
        if isinstance(value, bool) or not (0 <= value <= 1):
            raise ValueError("Truncation must be between 0 and 1")
        self._truncation = truncation

    def record_result(self, name, value):
        self.results[name] = value
        setattr(self, name, value)
        return value

    def record_results(self, **results):
        for name, value in results.items():
            self.record_result(name, value)
        return self
