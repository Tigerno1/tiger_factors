from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class NewsEntropyColumns:
    date: str = "date_"
    code: str = "code"
    topic: str = "event_topic"
    sentiment: str = "event_sentiment"
    novelty: str = "event_novelty"
    weight: str = "event_weight"


@dataclass(frozen=True, slots=True)
class NewsEntropyResult:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    metadata: dict[str, object]


def news_entropy_factor_names() -> list[str]:
    return ["news_entropy"]


def _as_frame(data: object) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    to_pandas = getattr(data, "to_pandas", None)
    if callable(to_pandas):
        return to_pandas()
    return pd.DataFrame(data)


def _shannon_entropy(values: pd.Series, weights: pd.Series | None = None) -> float:
    cleaned = values.astype("string").fillna("unknown")
    if cleaned.empty:
        return 0.0
    if weights is None:
        counts = cleaned.value_counts(dropna=False).astype(float)
    else:
        weights_series = pd.to_numeric(weights, errors="coerce").fillna(0.0)
        counts = (
            pd.DataFrame({"value": cleaned, "weight": weights_series})
            .groupby("value", dropna=False)["weight"]
            .sum()
            .astype(float)
        )
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    if probs.empty:
        return 0.0
    entropy = float(-(probs * np.log(probs)).sum())
    if len(probs) <= 1:
        return 0.0
    return entropy / float(np.log(len(probs)))


def _rolling_zscore(series: pd.Series, groups: pd.Series, window: int) -> pd.Series:
    min_periods = max(3, min(window, 5))
    mean = series.groupby(groups, sort=False).transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
    std = series.groupby(groups, sort=False).transform(lambda s: s.rolling(window, min_periods=min_periods).std(ddof=0))
    return (series - mean) / std.replace(0.0, np.nan)


class NewsEntropyEngine:
    """Experimental news entropy factor.

    The engine expects self-named event-level columns:

    - ``date_``
    - ``code``
    - ``event_topic``
    - optional ``event_sentiment``
    - optional ``event_novelty``
    - optional ``event_weight``

    The output is a daily cross-section score measuring topic entropy,
    sentiment entropy, event density, and novelty.
    """

    FIELD_SPEC: ClassVar[dict[str, tuple[str, ...]]] = {
        "required": ("date_", "code", "event_topic"),
        "optional": ("event_sentiment", "event_novelty", "event_weight"),
    }

    FIELD_ALIASES: ClassVar[dict[str, str]] = {
        "date": "date_",
        "tradetime": "date_",
        "symbol": "code",
        "securityid": "code",
        "topic": "event_topic",
        "category": "event_topic",
        "sentiment": "event_sentiment",
        "novelty_score": "event_novelty",
        "event_count": "event_weight",
        "weight": "event_weight",
    }

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        columns: NewsEntropyColumns | None = None,
        window: int = 20,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        self.columns = columns or NewsEntropyColumns()
        self.window = max(int(window), 3)
        self.weights = dict(
            weights
            or {
                "topic_entropy": 0.40,
                "sentiment_entropy": 0.20,
                "event_density": 0.25,
                "novelty": 0.15,
            }
        )
        self.data = self._prepare_input(data)

    def _prepare_input(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = _as_frame(data)
        rename_map: dict[str, str] = {}
        for source, target in self.FIELD_ALIASES.items():
            if source in frame.columns and target not in frame.columns:
                rename_map[source] = target
        frame = frame.rename(columns=rename_map)

        missing = [column for column in self.FIELD_SPEC["required"] if column not in frame.columns]
        if missing:
            raise ValueError(f"news_entropy input is missing columns: {missing!r}")

        frame[self.columns.date] = pd.to_datetime(frame[self.columns.date], errors="coerce").dt.tz_localize(None)
        frame[self.columns.code] = frame[self.columns.code].astype(str)
        frame[self.columns.topic] = frame[self.columns.topic].astype("string").fillna("unknown")

        if self.columns.sentiment not in frame.columns:
            frame[self.columns.sentiment] = "neutral"
        else:
            frame[self.columns.sentiment] = frame[self.columns.sentiment].astype("string").fillna("neutral")

        if self.columns.novelty not in frame.columns:
            frame[self.columns.novelty] = 0.0
        else:
            frame[self.columns.novelty] = pd.to_numeric(frame[self.columns.novelty], errors="coerce")

        if self.columns.weight not in frame.columns:
            frame[self.columns.weight] = 1.0
        else:
            frame[self.columns.weight] = pd.to_numeric(frame[self.columns.weight], errors="coerce").fillna(1.0)

        frame = frame.sort_values([self.columns.code, self.columns.date], kind="stable").reset_index(drop=True)
        return frame

    def _aggregate_daily(self) -> pd.DataFrame:
        frame = self.data.copy()
        columns = self.columns
        group_columns = [columns.code, columns.date]

        rows: list[dict[str, object]] = []
        for (code, date_), group in frame.groupby(group_columns, sort=False):
            weight = pd.to_numeric(group[columns.weight], errors="coerce").fillna(1.0)
            event_weight = float(weight.sum())
            rows.append(
                {
                    "code": code,
                    "date_": date_,
                    "event_weight": event_weight,
                    "event_count": int(len(group)),
                    "unique_topics": int(group[columns.topic].nunique(dropna=False)),
                    "topic_entropy": _shannon_entropy(group[columns.topic], weight),
                    "sentiment_entropy": _shannon_entropy(group[columns.sentiment], weight),
                    "novelty": float(pd.to_numeric(group[columns.novelty], errors="coerce").fillna(0.0).mul(weight).sum() / event_weight) if event_weight > 0 else 0.0,
                }
            )

        daily = pd.DataFrame.from_records(rows)
        if daily.empty:
            return daily

        daily = daily.sort_values(["code", "date_"], kind="stable").reset_index(drop=True)
        return daily

    def compute(self) -> NewsEntropyResult:
        daily = self._aggregate_daily()
        if daily.empty:
            output = pd.DataFrame(columns=["date_", "code", "value"])
            return NewsEntropyResult(
                frame=output,
                feature_columns=("topic_entropy", "sentiment_entropy", "event_density", "novelty"),
                metadata={"window": self.window, "weights": dict(self.weights), "factor_name": "news_entropy"},
            )

        daily["event_density"] = np.log1p(daily["event_weight"].astype(float))
        daily["topic_entropy_z"] = _rolling_zscore(daily["topic_entropy"], daily["code"], self.window)
        daily["sentiment_entropy_z"] = _rolling_zscore(daily["sentiment_entropy"], daily["code"], self.window)
        daily["event_density_z"] = _rolling_zscore(daily["event_density"], daily["code"], self.window)
        daily["novelty_z"] = _rolling_zscore(daily["novelty"], daily["code"], self.window)

        daily["value"] = 0.0
        for component, weight in self.weights.items():
            column_name = {
                "topic_entropy": "topic_entropy_z",
                "sentiment_entropy": "sentiment_entropy_z",
                "event_density": "event_density_z",
                "novelty": "novelty_z",
            }.get(component)
            if column_name is None:
                continue
            daily["value"] = daily["value"] + pd.to_numeric(daily[column_name], errors="coerce").fillna(0.0) * float(weight)

        output = daily.loc[:, ["date_", "code", "value", "topic_entropy", "sentiment_entropy", "event_density", "novelty"]].copy()
        output["factor_name"] = "news_entropy"
        output["value"] = pd.to_numeric(output["value"], errors="coerce")
        return NewsEntropyResult(
            frame=output.reset_index(drop=True),
            feature_columns=("topic_entropy", "sentiment_entropy", "event_density", "novelty"),
            metadata={
                "window": self.window,
                "weights": dict(self.weights),
                "factor_name": "news_entropy",
            },
        )

    def compute_matrix(self) -> pd.DataFrame:
        result = self.compute().frame
        return result.pivot(index="date_", columns="code", values="value").sort_index()

