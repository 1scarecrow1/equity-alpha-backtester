from dataclasses import dataclass, replace
from datetime import date
from dateutil import parser as dt_parser
from enum import Enum, IntEnum, StrEnum


class InstrumentType(StrEnum):
    EQUITY = "EQUITY"

class Region(StrEnum):
    USA = "USA"

class Universe(Enum):
    TOP500_PIT = "TOP500_PIT"

class UniverseSize(IntEnum):
    TOP500_PIT = 500

class Delay(Enum):
    """
    Delay=0 alphas trade in the evening using data from today
    Delay=1 alphas trade in the morning using data from yesterday. 
    """
    ZERO = 0
    ONE = 1

class Truncation(Enum):
    """
    Max daily weight of each instrument
    Bigger universe -> more truncation
    """
    DEFAULT_TRUNCATION = 0.05
    TOP500_PIT = 0.05

class Neutralization(StrEnum):
    NONE = "NONE"
    MARKET = "MARKET"
    SECTOR = "SECTOR"
    INDUSTRY = "INDUSTRY"
    SUBINDUSTRY = "SUBINDUSTRY"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            normalized = value.upper()
            for member in cls:
                if member.value == normalized:
                    return member
        return None

class Decay(IntEnum):
    """
    Sets input data equal to a linearly decreasing weighted average of 
    that data over the past selected number of days
    """
    NONE = 0
    WEEK = 5
    TWO_WEEKS = 10
    MONTH = 21

@dataclass(frozen=True)
class TradingPeriod:
    start: date
    end: date
    name: str = ""

    def __post_init__(self):
        start = self._to_date(self.start)
        end = self._to_date(self.end)

        if start > end:
            raise ValueError(f"Start must be before end: {start} > {end}")

        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @staticmethod
    def _to_date(value: str | date) -> date:
        if isinstance(value, date):
            return value
        return dt_parser.parse(value).date()

    @property
    def days(self) -> int:
        return (self.end - self.start).days

    def contains(self, other: "TradingPeriod") -> bool:
        return self.start <= other.start and other.end <= self.end

    def overlaps(self, other: "TradingPeriod") -> bool:
        return self.start <= other.end and other.start <= self.end

    def update(self, **changes):
        return replace(self, **changes)

    def __str__(self) -> str:
        label = f"{self.name}: " if self.name else ""
        return f"{label}{self.start} to {self.end} ({self.days} days)"




