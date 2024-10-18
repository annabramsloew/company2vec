import pickle
from datetime import date, datetime
from itertools import chain, repeat
from typing import Dict, Iterable, Iterator, Tuple, TypeVar, Any
from src.data.source.base import TokenSource

import pandas as pd
from dateutil import relativedelta
from pathlib import Path

T = TypeVar("T")


def interleave(iterable: Iterable[T], element: T, in_end: bool = False) -> Iterator[T]:
    """Return iterator with element interleaved between each element"""
    iter_ = iter(iterable)
    yield next(iter_)
    yield from chain.from_iterable(zip(repeat(element), iter_))
    if in_end:
        yield element


def add_years(d: datetime, years_to_add: int) -> datetime:
    """Add X years to the current date"""
    try:
        return d.replace(year=d.year + years_to_add)
    except ValueError:
        return d + (date(d.year + years_to_add, 1, 1) - date(d.year, 1, 1))


def num_months(current_date: datetime, final_date: datetime) -> float:
    """Calculate months difference"""
    r = relativedelta.relativedelta(current_date, final_date)
    return float((r.years * 12) + r.months)


def load_lookup(path: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Load the dictionary with the vocabulary"""
    with open(path, "rb") as f:
        indx2token, token2indx = pickle.load(f)
    return indx2token, token2indx


def load_positive_targets(path: str):  # type: ignore
    """Load CLS Targets"""
    df = pd.read_json(path, compression="gzip", orient="index")
    assert isinstance(df, pd.DataFrame)
    df["EVENT_FINAL_DATE"] = df["EVENT_FINAL_DATE"].apply(
        lambda x: pd.to_datetime(x, unit="ms")
    )
    return df.to_dict(orient="index")


def stringify(value: Any) -> str:
    """Convert a value to its string representation."""
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(stringify(v) for v in value) + "]"
    elif isinstance(value, dict):
        return "{" + ", ".join(f"{stringify(k)}: {stringify(v)}" for k, v in value.items()) + "}"
    elif isinstance(value, (int, float, str, bool)):
        return str(value)
    elif isinstance(value, datetime):
        return value.isoformat()
    elif isinstance(value, date):
        return value.isoformat()
    elif isinstance(value, Path):
        return str(value)
    elif isinstance(value, TokenSource):
        return value.name
    else:
        return value.name # most likely a self created class
        #raise TypeError(f"Unsupported type: {type(value)}")