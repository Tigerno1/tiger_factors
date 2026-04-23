from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from dataclasses import replace
from typing import ClassVar

import numpy as np
import pandas as pd

from tiger_factors.factor_algorithm.alpha101.descriptions import alpha101_description, alpha101_descriptions
from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore
from tiger_factors.utils.merge import merge_by_keys


def _round_window(value: float) -> int:
    return max(int(round(float(value))), 1)


@dataclass(frozen=True)
class NeutralizationColumns:
    sector: str = "sector"
    industry: str = "industry"
    subindustry: str = "subindustry"


class Alpha101ResearchCalculator:
    FIELD_SPEC: ClassVar[dict[str, tuple[str, ...]]] = {
        "required": ("date", "code", "open", "high", "low", "close", "volume"),
        "optional": ("vwap", "return", "market_value", "shares_outstanding"),
        "classification": ("sector", "industry", "subindustry"),
        "dropped": ("adj_close", "dividend", "price_adjustment_factor"),
    }
    FIELD_ALIASES: ClassVar[dict[str, str]] = {
        "date_": "date",
        "returns": "return",
        "cap": "market_value",
    }
    FACTOR_NAMES: ClassVar[tuple[str, ...]] = (
        "alpha_001",
        "alpha_002",
        "alpha_003",
        "alpha_004",
        "alpha_005",
        "alpha_006",
        "alpha_007",
        "alpha_008",
        "alpha_009",
        "alpha_010",
        "alpha_011",
        "alpha_012",
        "alpha_013",
        "alpha_014",
        "alpha_015",
        "alpha_016",
        "alpha_017",
        "alpha_018",
        "alpha_019",
        "alpha_020",
        "alpha_021",
        "alpha_022",
        "alpha_023",
        "alpha_024",
        "alpha_025",
        "alpha_026",
        "alpha_027",
        "alpha_028",
        "alpha_029",
        "alpha_030",
        "alpha_031",
        "alpha_032",
        "alpha_033",
        "alpha_034",
        "alpha_035",
        "alpha_036",
        "alpha_037",
        "alpha_038",
        "alpha_039",
        "alpha_040",
        "alpha_041",
        "alpha_042",
        "alpha_043",
        "alpha_044",
        "alpha_045",
        "alpha_046",
        "alpha_047",
        "alpha_048",
        "alpha_049",
        "alpha_050",
        "alpha_051",
        "alpha_052",
        "alpha_053",
        "alpha_054",
        "alpha_055",
        "alpha_056",
        "alpha_057",
        "alpha_058",
        "alpha_059",
        "alpha_060",
        "alpha_061",
        "alpha_062",
        "alpha_063",
        "alpha_064",
        "alpha_065",
        "alpha_066",
        "alpha_067",
        "alpha_068",
        "alpha_069",
        "alpha_070",
        "alpha_071",
        "alpha_072",
        "alpha_073",
        "alpha_074",
        "alpha_075",
        "alpha_076",
        "alpha_077",
        "alpha_078",
        "alpha_079",
        "alpha_080",
        "alpha_081",
        "alpha_082",
        "alpha_083",
        "alpha_084",
        "alpha_085",
        "alpha_086",
        "alpha_087",
        "alpha_088",
        "alpha_089",
        "alpha_090",
        "alpha_091",
        "alpha_092",
        "alpha_093",
        "alpha_094",
        "alpha_095",
        "alpha_096",
        "alpha_097",
        "alpha_098",
        "alpha_099",
        "alpha_100",
        "alpha_101",
    )
    def __init__(
        self,
        ctx: object | None = None,
        *,
        alpha_id: int | None = None,
        alpha_ids: Iterable[int] | None = None,
        spec: FactorSpec | None = None,
        factor_store: FactorStore | None = None,
        neutralization_columns: NeutralizationColumns | None = None,
        verbose: bool = False,
        save: bool = True,
    ) -> None:
        self.neutralization_columns = neutralization_columns or NeutralizationColumns()
        self.alpha_id = alpha_id
        self.alpha_ids = tuple(int(alpha) for alpha in alpha_ids) if alpha_ids is not None else None
        self.factor_names = self.FACTOR_NAMES
        self.spec = spec
        self.factor_store = factor_store
        self.verbose = verbose
        self.save = save
        self.eps = 1e-8
        self.data = pd.DataFrame()
        if ctx is not None:
            self._prepare_data(ctx)

    def _prepare_data(self, ctx: object) -> pd.DataFrame:
        price_input = ctx.feed_frame("price")
        classification_input = ctx.feed_frame("classification")
        frame = merge_by_keys([price_input, classification_input], join_keys=["code"])
        rename_map = {}
        seen_targets: set[str] = set(frame.columns)
        for source, target in self.FIELD_ALIASES.items():
            if source in frame.columns and target not in seen_targets:
                rename_map[source] = target
                seen_targets.add(target)
        for source, target in (
            (self.neutralization_columns.sector, "sector"),
            (self.neutralization_columns.industry, "industry"),
            (self.neutralization_columns.subindustry, "subindustry"),
        ):
            if source in frame.columns and source != target and target not in seen_targets:
                rename_map[source] = target
                seen_targets.add(target)
        frame = frame.rename(columns=rename_map)

        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
        frame["code"] = frame["code"].astype(str)
        frame = frame.sort_values(["code", "date"]).reset_index(drop=True)

        numeric_columns = [
            column
            for column in [
                *self.FIELD_SPEC["required"][2:],
                *self.FIELD_SPEC["optional"],
            ]
            if column in frame.columns
        ]
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame["vwap"] = (frame["open"] + frame["high"] + frame["low"] + frame["close"]) / 4.0
        frame["return"] = frame.groupby("code", sort=False)["close"].pct_change(fill_method=None)

        if "shares_outstanding" in frame.columns:
            frame["market_value"] = frame["close"] * frame["shares_outstanding"]
        elif "market_value" not in frame.columns:
            frame["market_value"] = frame["close"] * frame["volume"]

        frame = frame.drop(columns=list(self.FIELD_SPEC["dropped"]), errors="ignore")

        def _fill_group_column(target: str, fallbacks: tuple[str, ...]) -> None:
            if target in frame.columns:
                series = frame[target]
            else:
                series = None
                for fallback in fallbacks:
                    if fallback in frame.columns:
                        series = frame[fallback]
                        break
                if series is None:
                    frame[target] = "unknown"
                    return
            frame[target] = pd.Series(series, index=frame.index).astype("string").fillna("unknown")

        _fill_group_column("sector", ("industry", "subindustry"))
        _fill_group_column("industry", ("subindustry", "sector"))
        _fill_group_column("subindustry", ("industry", "sector"))

        self.data = frame
        return frame

    def _require_prepared(self) -> None:
        if self.data.empty:
            raise RuntimeError("Alpha101ResearchCalculator has not been prepared with a context yet.")

    def _rank(self, x: pd.Series) -> pd.Series:
        return x.groupby(self.data["date"]).rank(pct=True)

    def _delay(self, x: pd.Series, d: int) -> pd.Series:
        return x.groupby(self.data["code"]).shift(_round_window(d))

    def _delta(self, x: pd.Series, d: int) -> pd.Series:
        return x.groupby(self.data["code"]).diff(_round_window(d))

    # The rolling rank of the last value in a window, used for ts_rank.
    def _rolling_last_rank(self, values: np.ndarray) -> float:
        series = pd.Series(values)
        return float(series.rank(pct=True).iloc[-1])

    def _correlation(self, x: pd.Series, y: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        pieces: list[pd.Series] = []
        for _, indexer in self.data.groupby("code", sort=False).groups.items():
            series_x = pd.Series(x.loc[indexer].to_numpy(), index=indexer)
            series_y = pd.Series(y.loc[indexer].to_numpy(), index=indexer)
            pieces.append(series_x.rolling(window).corr(series_y))
        result = pd.concat(pieces).sort_index()
        return result.replace([np.inf, -np.inf], np.nan)

    def _covariance(self, x: pd.Series, y: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        pieces: list[pd.Series] = []
        for _, indexer in self.data.groupby("code", sort=False).groups.items():
            series_x = pd.Series(x.loc[indexer].to_numpy(), index=indexer)
            series_y = pd.Series(y.loc[indexer].to_numpy(), index=indexer)
            pieces.append(series_x.rolling(window).cov(series_y))
        return pd.concat(pieces).sort_index()

    # Scale factor values cross-sectionally by date.
    def _scale(self, x: pd.Series, a: float = 1.0) -> pd.Series:
        return x.groupby(self.data["date"]).transform(lambda s: s * a / (s.abs().sum() + self.eps))

    def _decay_linear(self, x: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        weights = np.arange(1, window + 1, dtype=float)
        result = (
            x.groupby(self.data["code"])
            .rolling(window)
            .apply(lambda s: float(np.dot(s, weights) / weights.sum()), raw=True)
            .reset_index(level=0, drop=True)
        )
        result.index = x.index
        return result

    #industry neutralization 
    def _neutralize(self, x: pd.Series, level: str = "industry") -> pd.Series:
        if level not in {"sector", "industry", "subindustry"}:
            raise ValueError(f"Unsupported neutralization level: {level}")
        groups = self.data[level].fillna("unknown")
        return x - x.groupby([self.data["date"], groups]).transform("mean")

    def _ts_min(self, x: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        result = x.groupby(self.data["code"]).rolling(window).min().reset_index(level=0, drop=True)
        result.index = x.index
        return result

    def _ts_max(self, x: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        result = x.groupby(self.data["code"]).rolling(window).max().reset_index(level=0, drop=True)
        result.index = x.index
        return result

    def _ts_argmin(self, x: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        result = x.groupby(self.data["code"]).rolling(window).apply(lambda s: float(np.argmin(s)), raw=True).reset_index(level=0, drop=True)
        result.index = x.index
        return result

    # get the position of the max value in a rolling window, used for alpha_1.
    def _ts_argmax(self, x: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        result = x.groupby(self.data["code"]).rolling(window).apply(lambda s: float(np.argmax(s)), raw=True).reset_index(level=0, drop=True)
        result.index = x.index
        return result

    def _ts_rank(self, x: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        result = x.groupby(self.data["code"]).rolling(window).apply(self._rolling_last_rank, raw=True).reset_index(level=0, drop=True)
        result.index = x.index
        return result

    def _sum(self, x: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        result = x.groupby(self.data["code"]).rolling(window).sum().reset_index(level=0, drop=True)
        result.index = x.index
        return result
    
    #rolling production
    def _product(self, x: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        result = x.groupby(self.data["code"]).rolling(window).apply(np.prod, raw=True).reset_index(level=0, drop=True)
        result.index = x.index
        return result

    def _stddev(self, x: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        result = x.groupby(self.data["code"]).rolling(window).std().reset_index(level=0, drop=True)
        result.index = x.index
        return result

    #avarage daily volume over the past d days, used for alpha_7 and alpha_28.
    def _adv(self, d: int) -> pd.Series:
        window = _round_window(d)
        result = self.data["volume"].groupby(self.data["code"]).rolling(window).mean().reset_index(level=0, drop=True)
        result.index = self.data.index
        return result

    # Convert boolean conditions to 0/1 floats for use in factor calculations. This is used in some alphas that involve counting the number of times a condition is true over a window.
    def _bool(self, condition: pd.Series) -> pd.Series:
        return condition.astype(float)

    # to finalize the factor values by combining with date and code columns, and dropping rows with NaN factor values.
    def _finalize(self, values: pd.Series, factor_name: str) -> pd.DataFrame:
        result = pd.DataFrame(
            {
                "date_": self.data["date"],
                "code": self.data["code"],
                factor_name: pd.to_numeric(values, errors="coerce"),
            }
        )
        return result.dropna(subset=[factor_name]).reset_index(drop=True)

    def alpha_description(self, alpha_id: int) -> str:
        return alpha101_description(alpha_id)

    def alpha_descriptions(self) -> dict[int, str]:
        return alpha101_descriptions()

    def alpha_1(self) -> pd.Series:
        base = self._stddev(self.data["return"], 20).where(self.data["return"] < 0, self.data["close"])
        return self._rank(self._ts_argmax(base.pow(2), 5)) - 0.5

    def alpha_2(self) -> pd.Series:
        x = self._rank(self._delta(np.log(self.data["volume"].replace(0, np.nan)), 2))
        y = self._rank((self.data["close"] - self.data["open"]) / (self.data["open"] + self.eps))
        return -self._correlation(x, y, 6)

    def alpha_3(self) -> pd.Series:
        return -self._correlation(self._rank(self.data["open"]), self._rank(self.data["volume"]), 10)

    def alpha_4(self) -> pd.Series:
        return -self._ts_rank(self._rank(self.data["low"]), 9)

    def alpha_5(self) -> pd.Series:
        left = self._rank(self.data["open"] - self._sum(self.data["vwap"], 10) / 10)
        right = -self._rank(self.data["close"] - self.data["vwap"]).abs()
        return left * right

    def alpha_6(self) -> pd.Series:
        return -self._correlation(self.data["open"], self.data["volume"], 10)

    def alpha_7(self) -> pd.Series:
        signal = self._ts_rank(self._delta(self.data["close"], 7).abs(), 60) * np.sign(self._delta(self.data["close"], 7))
        alpha = signal.where(self._adv(20) < self.data["volume"], -1.0)
        alpha.loc[self._adv(20).isna()] = np.nan
        return alpha

    def alpha_8(self) -> pd.Series:
        left = self._sum(self.data["open"], 5) * self._sum(self.data["return"], 5)
        return -self._rank(left - self._delay(left, 10))

    def alpha_9(self) -> pd.Series:
        inner = self._delta(self.data["close"], 1)
        alpha = inner.where(self._ts_min(inner, 5) > 0, inner.where(self._ts_max(inner, 5) < 0, -inner))
        alpha.loc[self._ts_min(inner, 5).isna()] = np.nan
        return alpha

    def alpha_10(self) -> pd.Series:
        inner = self._delta(self.data["close"], 1)
        signal = inner.where(self._ts_min(inner, 4) > 0, inner.where(self._ts_max(inner, 4) < 0, -inner))
        signal.loc[self._ts_min(inner, 4).isna()] = np.nan
        return self._rank(signal)

    def alpha_11(self) -> pd.Series:
        inner = self.data["vwap"] - self.data["close"]
        return (self._rank(self._ts_max(inner, 3)) + self._rank(self._ts_min(inner, 3))) * self._rank(self._delta(self.data["volume"], 3))

    def alpha_12(self) -> pd.Series:
        return np.sign(self._delta(self.data["volume"], 1)) * (-self._delta(self.data["close"], 1))

    def alpha_13(self) -> pd.Series:
        return -self._rank(self._covariance(self._rank(self.data["close"]), self._rank(self.data["volume"]), 5))

    def alpha_14(self) -> pd.Series:
        return -self._rank(self._delta(self.data["return"], 3)) * self._correlation(self.data["open"], self.data["volume"], 10)

    def alpha_15(self) -> pd.Series:
        return -self._sum(self._rank(self._correlation(self._rank(self.data["high"]), self._rank(self.data["volume"]), 3)), 3)

    def alpha_16(self) -> pd.Series:
        return -self._rank(self._covariance(self._rank(self.data["high"]), self._rank(self.data["volume"]), 5))

    def alpha_17(self) -> pd.Series:
        part_0 = -self._rank(self._ts_rank(self.data["close"], 10))
        part_1 = self._rank(self._delta(self._delta(self.data["close"], 1), 1))
        part_2 = self._rank(self._ts_rank(self.data["volume"] / (self._adv(20) + self.eps), 5))
        return part_0 * part_1 * part_2

    def alpha_18(self) -> pd.Series:
        return -self._rank(self._stddev((self.data["close"] - self.data["open"]).abs(), 5) + (self.data["close"] - self.data["open"]) + self._correlation(self.data["close"], self.data["open"], 10))

    def alpha_19(self) -> pd.Series:
        left = -np.sign((self.data["close"] - self._delay(self.data["close"], 7)) + self._delta(self.data["close"], 7))
        right = 1 + self._rank(1 + self._sum(self.data["return"], 250))
        return left * right

    def alpha_20(self) -> pd.Series:
        return (
            -self._rank(self.data["open"] - self._delay(self.data["high"], 1))
            * self._rank(self.data["open"] - self._delay(self.data["close"], 1))
            * self._rank(self.data["open"] - self._delay(self.data["low"], 1))
        )

    def alpha_21(self) -> pd.Series:
        mean8 = self._sum(self.data["close"], 8) / 8
        std8 = self._stddev(self.data["close"], 8)
        mean2 = self._sum(self.data["close"], 2) / 2
        adv20 = self._adv(20)
        alpha = pd.Series(
            np.where(
                (mean8 + std8) < mean2,
                -1.0,
                np.where(mean2 < (mean8 - std8), 1.0, np.where((self.data["volume"] / (adv20 + self.eps)) >= 1.0, 1.0, -1.0)),
            ),
            index=self.data.index,
        )
        alpha.loc[(mean8.isna()) | (std8.isna()) | (mean2.isna()) | (adv20.isna())] = np.nan
        return alpha

    def alpha_22(self) -> pd.Series:
        return -self._delta(self._correlation(self.data["high"], self.data["volume"], 5), 5) * self._rank(self._stddev(self.data["close"], 20))

    def alpha_23(self) -> pd.Series:
        alpha = (-self._delta(self.data["high"], 2)).where((self._sum(self.data["high"], 20) / 20) < self.data["high"], 0.0)
        alpha.loc[self._sum(self.data["high"], 20).isna()] = np.nan
        return alpha

    def alpha_24(self) -> pd.Series:
        left = self._delta(self._sum(self.data["close"], 100) / 100, 100) / (self._delay(self.data["close"], 100) + self.eps)
        alpha = (-(self.data["close"] - self._ts_min(self.data["close"], 100))).where(left <= 0.05, -self._delta(self.data["close"], 3))
        alpha.loc[left.isna()] = np.nan
        return alpha

    def alpha_25(self) -> pd.Series:
        return self._rank(-self.data["return"] * self._adv(20) * self.data["vwap"] * (self.data["high"] - self.data["close"]))

    def alpha_26(self) -> pd.Series:
        return -self._ts_max(self._correlation(self._ts_rank(self.data["volume"], 5), self._ts_rank(self.data["high"], 5), 5), 3)

    def alpha_27(self) -> pd.Series:
        signal = self._rank(self._sum(self._correlation(self._rank(self.data["volume"]), self._rank(self.data["vwap"]), 6), 2) / 2.0)
        alpha = pd.Series(np.where(signal > 0.5, -1.0, 1.0), index=self.data.index)
        alpha.loc[signal.isna()] = np.nan
        return alpha

    def alpha_28(self) -> pd.Series:
        return self._scale(self._correlation(self._adv(20), self.data["low"], 5) + (self.data["high"] + self.data["low"]) / 2 - self.data["close"])

    def alpha_29(self) -> pd.Series:
        inner = self._rank(self._rank(-self._rank(self._delta(self.data["close"] - 1, 5))))
        rolling_min = self._ts_min(inner, 2)
        logged = np.log(self._sum(rolling_min, 1) + self.eps)
        left = self._ts_min(self._rank(self._rank(self._scale(logged))), 5)
        right = self._ts_rank(self._delay(-self.data["return"], 6), 5)
        return left + right

    def alpha_30(self) -> pd.Series:
        left = 1 - self._rank(
            np.sign(self.data["close"] - self._delay(self.data["close"], 1))
            + np.sign(self._delay(self.data["close"], 1) - self._delay(self.data["close"], 2))
            + np.sign(self._delay(self.data["close"], 2) - self._delay(self.data["close"], 3))
        )
        return left * self._sum(self.data["volume"], 5) / (self._sum(self.data["volume"], 20) + self.eps)

    def alpha_31(self) -> pd.Series:
        part_0 = self._rank(self._rank(self._rank(self._decay_linear(-self._rank(self._rank(self._delta(self.data["close"], 10))), 10))))
        part_1 = self._rank(-self._delta(self.data["close"], 3))
        part_2 = np.sign(self._scale(self._correlation(self._adv(20), self.data["low"], 12)))
        return part_0 + part_1 + part_2

    def alpha_32(self) -> pd.Series:
        left = self._scale((self._sum(self.data["close"], 7) / 7) - self.data["close"])
        right = 20 * self._scale(self._correlation(self.data["vwap"], self._delay(self.data["close"], 5), 230))
        return left + right

    def alpha_33(self) -> pd.Series:
        return self._rank(-(1 - (self.data["open"] / (self.data["close"] + self.eps))))

    def alpha_34(self) -> pd.Series:
        left = 1 - self._rank(self._stddev(self.data["return"], 2) / (self._stddev(self.data["return"], 5) + self.eps))
        right = 1 - self._rank(self._delta(self.data["close"], 1))
        return self._rank(left + right)

    def alpha_35(self) -> pd.Series:
        return self._ts_rank(self.data["volume"], 32) * (1 - self._ts_rank(self.data["close"] + self.data["high"] - self.data["low"], 16)) * (1 - self._ts_rank(self.data["return"], 32))

    def alpha_36(self) -> pd.Series:
        part_0 = 2.21 * self._rank(self._correlation(self.data["close"] - self.data["open"], self._delay(self.data["volume"], 1), 15))
        part_1 = 0.7 * self._rank(self.data["open"] - self.data["close"])
        part_2 = 0.73 * self._rank(self._ts_rank(self._delay(-self.data["return"], 6), 5))
        part_3 = self._rank(self._correlation(self.data["vwap"], self._adv(20), 6).abs())
        part_4 = 0.6 * self._rank((self._sum(self.data["close"], 200) / 200 - self.data["open"]) * (self.data["close"] - self.data["open"]))
        return part_0 + part_1 + part_2 + part_3 + part_4

    def alpha_37(self) -> pd.Series:
        return self._rank(self._correlation(self._delay(self.data["open"] - self.data["close"], 1), self.data["close"], 200)) + self._rank(self.data["open"] - self.data["close"])

    def alpha_38(self) -> pd.Series:
        return -self._rank(self._ts_rank(self.data["close"], 10)) * self._rank(self.data["close"] / (self.data["open"] + self.eps))

    def alpha_39(self) -> pd.Series:
        left = -self._rank(self._delta(self.data["close"], 7) * (1 - self._rank(self._decay_linear(self.data["volume"] / (self._adv(20) + self.eps), 9))))
        right = 1 + self._rank(self._sum(self.data["return"], 250))
        return left * right

    def alpha_40(self) -> pd.Series:
        return -self._rank(self._stddev(self.data["high"], 10)) * self._correlation(self.data["high"], self.data["volume"], 10)

    def alpha_41(self) -> pd.Series:
        return np.sqrt(self.data["high"] * self.data["low"]) - self.data["vwap"]

    def alpha_42(self) -> pd.Series:
        return self._rank(self.data["vwap"] - self.data["close"]) / (self._rank(self.data["vwap"] + self.data["close"]) + self.eps)

    def alpha_43(self) -> pd.Series:
        return self._ts_rank(self.data["volume"] / (self._adv(20) + self.eps), 20) * self._ts_rank(-self._delta(self.data["close"], 7), 8)

    def alpha_44(self) -> pd.Series:
        return -self._correlation(self.data["high"], self._rank(self.data["volume"]), 5)

    def alpha_45(self) -> pd.Series:
        part_0 = self._rank(self._sum(self._delay(self.data["close"], 5), 20) / 20)
        part_1 = self._correlation(self.data["close"], self.data["volume"], 2)
        part_2 = self._rank(self._correlation(self._sum(self.data["close"], 5), self._sum(self.data["close"], 20), 2))
        return -part_0 * part_1 * part_2

    def alpha_46(self) -> pd.Series:
        left = (self._delay(self.data["close"], 20) - self._delay(self.data["close"], 10)) / 10 - (self._delay(self.data["close"], 10) - self.data["close"]) / 10
        y = -(self.data["close"] - self._delay(self.data["close"], 1))
        alpha = pd.Series(np.where(left > 0.25, -1.0, np.where(left < 0.0, 1.0, y)), index=self.data.index)
        alpha.loc[left.isna()] = np.nan
        return alpha

    def alpha_47(self) -> pd.Series:
        part_0 = self._rank(1 / (self.data["close"] + self.eps)) * self.data["volume"] / (self._adv(20) + self.eps)
        part_1 = self.data["high"] * self._rank(self.data["high"] - self.data["close"]) / (self._sum(self.data["high"], 5) / 5 + self.eps)
        part_2 = self._rank(self.data["vwap"] - self._delay(self.data["vwap"], 5))
        return part_0 * part_1 - part_2

    def alpha_48(self) -> pd.Series:
        left = self._neutralize(
            self._correlation(self._delta(self.data["close"], 1), self._delta(self._delay(self.data["close"], 1), 1), 250)
            * self._delta(self.data["close"], 1)
            / (self.data["close"] + self.eps),
            "subindustry",
        )
        right = self._sum((self._delta(self.data["close"], 1) / (self._delay(self.data["close"], 1) + self.eps)).pow(2), 250)
        return left / (right + self.eps)

    def alpha_49(self) -> pd.Series:
        left = (self._delay(self.data["close"], 20) - self._delay(self.data["close"], 10)) / 10 - (self._delay(self.data["close"], 10) - self.data["close"]) / 10
        y = -(self.data["close"] - self._delay(self.data["close"], 1))
        alpha = pd.Series(np.where(left < -0.1, 1.0, y), index=self.data.index)
        alpha.loc[left.isna()] = np.nan
        return alpha

    def alpha_50(self) -> pd.Series:
        return -self._ts_max(self._rank(self._correlation(self._rank(self.data["volume"]), self._rank(self.data["vwap"]), 5)), 5)

    def alpha_51(self) -> pd.Series:
        left = (self._delay(self.data["close"], 20) - self._delay(self.data["close"], 10)) / 10 - (self._delay(self.data["close"], 10) - self.data["close"]) / 10
        y = -(self.data["close"] - self._delay(self.data["close"], 1))
        alpha = pd.Series(np.where(left < -0.05, 1.0, y), index=self.data.index)
        alpha.loc[left.isna()] = np.nan
        return alpha

    def alpha_52(self) -> pd.Series:
        left = -self._ts_min(self.data["low"], 5)
        right = self._delay(self._ts_min(self.data["low"], 5), 5) * self._rank((self._sum(self.data["return"], 240) - self._sum(self.data["return"], 20)) / 220) * self._ts_rank(self.data["volume"], 5)
        return left + right

    def alpha_53(self) -> pd.Series:
        return -self._delta(((self.data["close"] - self.data["low"]) - (self.data["high"] - self.data["close"])) / (self.data["close"] - self.data["low"] + self.eps), 9)

    def alpha_54(self) -> pd.Series:
        return -((self.data["low"] - self.data["close"]) * self.data["open"].pow(5)) / (((self.data["low"] - self.data["high"]) * self.data["close"].pow(5)) + self.eps)

    def alpha_55(self) -> pd.Series:
        left = self._rank((self.data["close"] - self._ts_min(self.data["low"], 12)) / (self._ts_max(self.data["high"], 12) - self._ts_min(self.data["low"], 12) + self.eps))
        return -self._correlation(left, self._rank(self.data["volume"]), 6)

    def alpha_56(self) -> pd.Series:
        left = self._rank(self._sum(self.data["return"], 10) / (self._sum(self._sum(self.data["return"], 2), 3) + self.eps))
        right = self._rank(self.data["return"] * self.data["market_value"])
        return -left * right

    def alpha_57(self) -> pd.Series:
        return -(self.data["close"] - self.data["vwap"]) / (self._decay_linear(self._rank(self._ts_argmax(self.data["close"], 30)), 2) + self.eps)

    def alpha_58(self) -> pd.Series:
        return -self._ts_rank(self._decay_linear(self._correlation(self._neutralize(self.data["vwap"], "sector"), self.data["volume"], 4), 8), 6)

    def alpha_59(self) -> pd.Series:
        return -self._ts_rank(self._decay_linear(self._correlation(self._neutralize(self.data["vwap"], "industry"), self.data["volume"], 4), 16), 8)

    def alpha_60(self) -> pd.Series:
        left = 2 * self._scale(self._rank((((self.data["close"] - self.data["low"]) - (self.data["high"] - self.data["close"])) / (self.data["high"] - self.data["low"] + self.eps)) * self.data["volume"]))
        right = self._scale(self._rank(self._ts_argmax(self.data["close"], 10)))
        return -(left - right)

    def alpha_61(self) -> pd.Series:
        left = self._rank(self.data["vwap"] - self._ts_min(self.data["vwap"], 16))
        right = self._rank(self._correlation(self.data["vwap"], self._adv(180), 18))
        return self._bool(left < right)

    def alpha_62(self) -> pd.Series:
        left = self._rank(self._correlation(self.data["vwap"], self._sum(self._adv(20), 22), 10))
        inner = self._bool((self._rank(self.data["open"]) + self._rank(self.data["open"])) < (self._rank((self.data["high"] + self.data["low"]) / 2) + self._rank(self.data["high"])))
        right = self._rank(inner)
        return -self._bool(left < right)

    def alpha_63(self) -> pd.Series:
        left = self._rank(self._decay_linear(self._delta(self._neutralize(self.data["close"], "industry"), 2), 8))
        right = self._rank(self._decay_linear(self._correlation(self.data["vwap"] * 0.318108 + self.data["open"] * (1 - 0.318108), self._sum(self._adv(180), 37), 14), 12))
        return -(left - right)

    def alpha_64(self) -> pd.Series:
        left = self._rank(self._correlation(self._sum(self.data["open"] * 0.178404 + self.data["low"] * (1 - 0.178404), 13), self._sum(self._adv(120), 13), 17))
        right = self._rank(self._delta(((self.data["high"] + self.data["low"]) / 2) * 0.178404 + self.data["vwap"] * (1 - 0.178404), 4))
        return -self._bool(left < right)

    def alpha_65(self) -> pd.Series:
        left = self._rank(self._correlation(self.data["open"] * 0.00817205 + self.data["vwap"] * (1 - 0.00817205), self._sum(self._adv(60), 9), 6))
        right = self._rank(self.data["open"] - self._ts_min(self.data["open"], 14))
        return -self._bool(left < right)

    def alpha_66(self) -> pd.Series:
        left = self._rank(self._decay_linear(self._delta(self.data["vwap"], 4), 7))
        right = self._ts_rank(self._decay_linear((self.data["low"] - self.data["vwap"]) / (self.data["open"] - (self.data["high"] + self.data["low"]) / 2 + self.eps), 11), 7)
        return -(left + right)

    def alpha_67(self) -> pd.Series:
        left = self._rank(self.data["high"] - self._ts_min(self.data["high"], 2))
        right = self._rank(self._correlation(self._neutralize(self.data["vwap"], "sector"), self._neutralize(self._adv(20), "subindustry"), 6))
        return -(left ** right)

    def alpha_68(self) -> pd.Series:
        left = self._ts_rank(self._correlation(self._rank(self.data["high"]), self._rank(self._adv(15)), 9), 14)
        right = self._rank(self._delta(self.data["close"] * 0.518371 + self.data["low"] * (1 - 0.518371), 1))
        return -self._bool(left < right)

    def alpha_69(self) -> pd.Series:
        left = self._rank(self._ts_max(self._delta(self._neutralize(self.data["vwap"], "industry"), 3), 5))
        right = self._ts_rank(self._correlation(self.data["close"] * 0.490655 + self.data["vwap"] * (1 - 0.490655), self._adv(20), 5), 9)
        return -(left ** right)

    def alpha_70(self) -> pd.Series:
        left = self._rank(self._delta(self.data["vwap"], 1))
        right = self._ts_rank(self._correlation(self._neutralize(self.data["close"], "industry"), self._adv(50), 18), 18)
        return -(left ** right)

    def alpha_71(self) -> pd.Series:
        left = self._ts_rank(self._decay_linear(self._correlation(self._ts_rank(self.data["close"], 3), self._ts_rank(self._adv(180), 12), 18), 4), 16)
        right = self._ts_rank(self._decay_linear(self._rank((self.data["low"] + self.data["open"]) - 2 * self.data["vwap"]).pow(2), 16), 4)
        return pd.concat([left, right], axis=1).max(axis=1)

    def alpha_72(self) -> pd.Series:
        left = self._rank(self._decay_linear(self._correlation((self.data["high"] + self.data["low"]) / 2, self._adv(40), 9), 10))
        right = self._rank(self._decay_linear(self._correlation(self._ts_rank(self.data["vwap"], 4), self._ts_rank(self.data["volume"], 19), 7), 3))
        return left / (right + self.eps)

    def alpha_73(self) -> pd.Series:
        left = self._rank(self._decay_linear(self._delta(self.data["vwap"], 5), 3))
        base = self.data["open"] * 0.147155 + self.data["low"] * (1 - 0.147155)
        right = self._ts_rank(self._decay_linear(-self._delta(base, 2) / (base + self.eps), 3), 17)
        return -pd.concat([left, right], axis=1).max(axis=1)

    def alpha_74(self) -> pd.Series:
        left = self._rank(self._correlation(self.data["close"], self._sum(self._adv(30), 37), 15))
        right = self._rank(self._correlation(self._rank(self.data["high"] * 0.0261661 + self.data["vwap"] * (1 - 0.0261661)), self._rank(self.data["volume"]), 11))
        return -self._bool(left < right)

    def alpha_75(self) -> pd.Series:
        left = self._rank(self._correlation(self.data["vwap"], self.data["volume"], 4))
        right = self._rank(self._correlation(self._rank(self.data["low"]), self._rank(self._adv(50)), 12))
        return self._bool(left < right)

    def alpha_76(self) -> pd.Series:
        left = self._rank(self._decay_linear(self._delta(self.data["vwap"], 1), 12))
        right = self._ts_rank(self._decay_linear(self._ts_rank(self._correlation(self._neutralize(self.data["low"], "sector"), self._adv(81), 8), 20), 17), 19)
        return -pd.concat([left, right], axis=1).max(axis=1)

    def alpha_77(self) -> pd.Series:
        left = self._rank(self._decay_linear(((self.data["high"] + self.data["low"]) / 2) - self.data["vwap"], 20))
        right = self._rank(self._decay_linear(self._correlation((self.data["high"] + self.data["low"]) / 2, self._adv(40), 3), 6))
        return pd.concat([left, right], axis=1).min(axis=1)

    def alpha_78(self) -> pd.Series:
        left = self._rank(self._correlation(self._sum(self.data["low"] * 0.352233 + self.data["vwap"] * (1 - 0.352233), 20), self._sum(self._adv(40), 20), 7))
        right = self._rank(self._correlation(self._rank(self.data["vwap"]), self._rank(self.data["volume"]), 6))
        return left ** right

    def alpha_79(self) -> pd.Series:
        left = self._rank(self._delta(self._neutralize(self.data["close"] * 0.60733 + self.data["open"] * (1 - 0.60733), "sector"), 1))
        right = self._rank(self._correlation(self._ts_rank(self.data["vwap"], 4), self._ts_rank(self._adv(150), 9), 15))
        return self._bool(left < right)

    def alpha_80(self) -> pd.Series:
        left = self._rank(np.sign(self._delta(self._neutralize(self.data["open"] * 0.868128 + self.data["high"] * (1 - 0.868128), "industry"), 4)))
        right = self._ts_rank(self._correlation(self.data["high"], self._adv(10), 5), 6)
        return -(left ** right)

    def alpha_81(self) -> pd.Series:
        inner = self._rank(self._correlation(self.data["vwap"], self._sum(self._adv(10), 50), 8)).pow(4)
        left = self._rank(np.log(self._product(self._rank(inner), 15) + self.eps))
        right = self._rank(self._correlation(self._rank(self.data["vwap"]), self._rank(self.data["volume"]), 5))
        return -self._bool(left < right)

    def alpha_82(self) -> pd.Series:
        left = self._rank(self._decay_linear(self._delta(self.data["open"], 1), 15))
        right = self._ts_rank(self._decay_linear(self._correlation(self._neutralize(self.data["volume"], "sector"), self.data["open"], 17), 7), 13)
        return -pd.concat([left, right], axis=1).min(axis=1)

    def alpha_83(self) -> pd.Series:
        part_0 = self._rank(self._delay((self.data["high"] - self.data["low"]) / (self._sum(self.data["close"], 5) / 5 + self.eps), 2))
        part_1 = self._rank(self._rank(self.data["volume"]))
        part_2 = ((self.data["high"] - self.data["low"]) / (self._sum(self.data["close"], 5) / 5 + self.eps)) / (self.data["vwap"] - self.data["close"] + self.eps)
        return part_0 * part_1 / (part_2 + self.eps)

    def alpha_84(self) -> pd.Series:
        left = self._ts_rank(self.data["vwap"] - self._ts_max(self.data["vwap"], 15), 21)
        return left ** self._delta(self.data["close"], 5)

    def alpha_85(self) -> pd.Series:
        left = self._rank(self._correlation(self.data["high"] * 0.876703 + self.data["close"] * (1 - 0.876703), self._adv(30), 10))
        right = self._rank(self._correlation(self._ts_rank((self.data["high"] + self.data["low"]) / 2, 4), self._ts_rank(self.data["volume"], 10), 7))
        return left ** right

    def alpha_86(self) -> pd.Series:
        left = self._ts_rank(self._correlation(self.data["close"], self._sum(self._adv(20), 15), 6), 20)
        right = self._rank(self.data["close"] - self.data["vwap"])
        return -self._bool(left < right)

    def alpha_87(self) -> pd.Series:
        left = self._rank(self._decay_linear(self._delta(self.data["close"] * 0.369701 + self.data["vwap"] * (1 - 0.369701), 2), 3))
        right = self._ts_rank(self._decay_linear(self._correlation(self._neutralize(self._adv(81), "industry"), self.data["close"], 13).abs(), 5), 14)
        return -pd.concat([left, right], axis=1).max(axis=1)

    def alpha_88(self) -> pd.Series:
        left = self._rank(self._decay_linear((self._rank(self.data["open"]) + self._rank(self.data["low"])) - (self._rank(self.data["high"]) + self._rank(self.data["close"])), 8))
        right = self._ts_rank(self._decay_linear(self._correlation(self._ts_rank(self.data["close"], 8), self._ts_rank(self._adv(60), 21), 8), 7), 3)
        return pd.concat([left, right], axis=1).min(axis=1)

    def alpha_89(self) -> pd.Series:
        left = self._ts_rank(self._decay_linear(self._correlation(self.data["low"], self._adv(10), 7), 6), 4)
        right = self._ts_rank(self._decay_linear(self._delta(self._neutralize(self.data["vwap"], "industry"), 3), 10), 15)
        return left - right

    def alpha_90(self) -> pd.Series:
        left = self._rank(self.data["close"] - self._ts_max(self.data["close"], 5))
        right = self._ts_rank(self._correlation(self._neutralize(self._adv(40), "subindustry"), self.data["low"], 5), 3)
        return -(left ** right)

    def alpha_91(self) -> pd.Series:
        left = self._ts_rank(self._decay_linear(self._decay_linear(self._correlation(self._neutralize(self.data["close"], "industry"), self.data["volume"], 10), 16), 4), 5)
        right = self._rank(self._decay_linear(self._correlation(self.data["vwap"], self._adv(30), 4), 3))
        return -(left - right)

    def alpha_92(self) -> pd.Series:
        left = self._ts_rank(self._decay_linear((((self.data["high"] + self.data["low"]) / 2 + self.data["close"]) < (self.data["low"] + self.data["open"])).astype(float), 15), 19)
        right = self._ts_rank(self._decay_linear(self._correlation(self._rank(self.data["low"]), self._rank(self._adv(30)), 8), 7), 7)
        return pd.concat([left, right], axis=1).min(axis=1)

    def alpha_93(self) -> pd.Series:
        left = self._ts_rank(self._decay_linear(self._correlation(self._neutralize(self.data["vwap"], "industry"), self._adv(81), 17), 20), 8)
        right = self._rank(self._decay_linear(self._delta(self.data["close"] * 0.524434 + self.data["vwap"] * (1 - 0.524434), 3), 16))
        return left / (right + self.eps)

    def alpha_94(self) -> pd.Series:
        left = self._rank(self.data["vwap"] - self._ts_min(self.data["vwap"], 12))
        right = self._ts_rank(self._correlation(self._ts_rank(self.data["vwap"], 20), self._ts_rank(self._adv(60), 4), 18), 3)
        return -(left ** right)

    def alpha_95(self) -> pd.Series:
        left = self._rank(self.data["open"] - self._ts_min(self.data["open"], 12))
        right = self._ts_rank(self._rank(self._correlation(self._sum((self.data["high"] + self.data["low"]) / 2, 19), self._sum(self._adv(40), 19), 13)).pow(5), 12)
        return self._bool(left < right)

    def alpha_96(self) -> pd.Series:
        left = self._ts_rank(self._decay_linear(self._correlation(self._rank(self.data["vwap"]), self._rank(self.data["volume"]), 4), 4), 8)
        right = self._ts_rank(self._decay_linear(self._ts_argmax(self._correlation(self._ts_rank(self.data["close"], 7), self._ts_rank(self._adv(60), 4), 4), 13), 14), 13)
        return -pd.concat([left, right], axis=1).max(axis=1)

    def alpha_97(self) -> pd.Series:
        left = self._rank(self._decay_linear(self._delta(self._neutralize(self.data["low"] * 0.721001 + self.data["vwap"] * (1 - 0.721001), "industry"), 3), 20))
        right = self._ts_rank(self._decay_linear(self._ts_rank(self._correlation(self._ts_rank(self.data["low"], 8), self._ts_rank(self._adv(60), 17), 5), 19), 16), 7)
        return -(left - right)

    def alpha_98(self) -> pd.Series:
        left = self._rank(self._decay_linear(self._correlation(self.data["vwap"], self._sum(self._adv(5), 26), 5), 7))
        right = self._rank(self._decay_linear(self._ts_rank(self._ts_argmin(self._correlation(self._rank(self.data["open"]), self._rank(self._adv(15)), 21), 9), 7), 8))
        return left - right

    def alpha_99(self) -> pd.Series:
        left = self._rank(self._correlation(self._sum((self.data["high"] + self.data["low"]) / 2, 20), self._sum(self._adv(60), 20), 9))
        right = self._rank(self._correlation(self.data["low"], self.data["volume"], 6))
        return -self._bool(left < right)

    def alpha_100(self) -> pd.Series:
        ranked_flow = self._rank(
            (
                ((self.data["close"] - self.data["low"]) - (self.data["high"] - self.data["close"]))
                / (self.data["high"] - self.data["low"] + self.eps)
            )
            * self.data["volume"]
        )
        part_0 = 1.5 * self._scale(
            self._neutralize(self._neutralize(ranked_flow, "subindustry"), "subindustry")
        )
        left = self._correlation(self.data["close"], self._rank(self._adv(20)), 5)
        right = self._rank(self._ts_argmin(self.data["close"], 30))
        part_1 = self._scale(self._neutralize(left - right, "subindustry"))
        part_2 = self.data["volume"] / (self._adv(20) + self.eps)
        return -(part_0 - part_1) * part_2

    def alpha_101(self) -> pd.Series:
        return (self.data["close"] - self.data["open"]) / (self.data["high"] - self.data["low"] + 0.001)

    def _alpha_methods(self) -> dict[int, Callable[[], pd.Series]]:
        return {i: getattr(self, f"alpha_{i}") for i in range(1, 102)}

    def compute_series(self, alpha_id: int) -> pd.Series:
        self._require_prepared()
        if alpha_id < 1 or alpha_id > 101:
            raise ValueError("alpha_id must be in [1, 101]")
        series = self._alpha_methods()[alpha_id]().rename(f"alpha_{alpha_id:03d}")
        series.attrs["description"] = self.alpha_description(alpha_id)
        return series

    def compute(self, alpha_id: int) -> pd.DataFrame:
        self._require_prepared()
        factor_name = f"alpha_{alpha_id:03d}"
        result = self._finalize(self.compute_series(alpha_id), factor_name)
        result.attrs["description"] = self.alpha_description(alpha_id)
        return result

    def compute_all(self, alpha_ids: Iterable[int] | None = None) -> pd.DataFrame:
        self._require_prepared()
        selected = list(alpha_ids) if alpha_ids is not None else list(range(1, 102))
        frame = self.data[["date", "code"]].rename(columns={"date": "date_"}).copy()
        factor_columns: dict[str, pd.Series] = {}
        for alpha_id in selected:
            factor_name = f"alpha_{alpha_id:03d}"
            factor_columns[factor_name] = pd.to_numeric(self.compute_series(alpha_id), errors="coerce")
        if not factor_columns:
            return frame
        result = pd.concat([frame, pd.DataFrame(factor_columns, index=self.data.index)], axis=1)
        result.attrs["alpha_descriptions"] = {f"alpha_{alpha_id:03d}": self.alpha_description(alpha_id) for alpha_id in selected}
        return result

    def compute_matrix(self, alpha_id: int) -> pd.DataFrame:
        self._require_prepared()
        factor_name = f"alpha_{alpha_id:03d}"
        result = self.compute(alpha_id)
        return result.pivot(index="date_", columns="code", values=factor_name).sort_index()

    def __call__(self, ctx: object) -> None:
        if self.verbose:
            print("    [alpha101] computing", flush=True)
        self._prepare_data(ctx)
        alpha_ids = self.alpha_ids or ([self.alpha_id] if self.alpha_id is not None else list(range(1, len(self.FACTOR_NAMES) + 1)))
        for alpha_id in alpha_ids:
            factor_name = f"alpha_{alpha_id:03d}"
            if self.verbose:
                print(f"        [{factor_name}] computing", flush=True)
            result = self.compute(alpha_id)
            if self.save and self.factor_store is not None:
                if self.spec is None:
                    raise ValueError("spec is required when factor_store is provided")
                factor_spec = replace(self.spec, table_name=factor_name)
                value_column = getattr(self.spec, "value_column", "value")
                factor_frame = result.loc[:, ["date_", "code", factor_name]].rename(columns={factor_name: value_column})
                self.factor_store.save_factor(factor_spec, factor_frame, force_updated=True)
            if self.verbose:
                print(f"        [{factor_name}] done ({len(result):,} rows)", flush=True)
        if self.verbose:
            print("    [alpha101] done", flush=True)
        return None
