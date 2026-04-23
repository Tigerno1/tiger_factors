from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from .factor_functions import available_factors
from .factor_functions import run_original_factor


@dataclass(frozen=True)
class CommonFactorSpec:
    """Tiger-side common factor registry entry.

    The registry is intentionally pragmatic:

    - direct factors delegate to the vendored OpenAssetPricing implementation
    - proxy factors reuse a related vendored factor with a sign flip or a
      simple cross-sectional composite
    - placeholder factors are tracked explicitly when the factor needs a
      separate data source not yet wired into Tiger
    """

    name: str
    family: str
    kind: str
    description: str
    source_signal: str | None = None
    sign: float = 1.0
    components: tuple[tuple[str, float], ...] = ()
    aliases: tuple[str, ...] = ()
    macro_candidates: tuple[str, ...] = ()
    macro_transform: str | None = None
    notes: str | None = None

    @property
    def is_placeholder(self) -> bool:
        return self.kind == "placeholder"

    @property
    def is_composite(self) -> bool:
        return self.kind == "composite"

    @property
    def is_macro(self) -> bool:
        return self.kind == "macro"


def _spec(
    name: str,
    family: str,
    description: str,
    *,
    kind: str | None = None,
    source_signal: str | None = None,
    sign: float = 1.0,
    components: tuple[tuple[str, float], ...] = (),
    aliases: tuple[str, ...] = (),
    macro_candidates: tuple[str, ...] = (),
    macro_transform: str | None = None,
    notes: str | None = None,
) -> CommonFactorSpec:
    resolved_kind = "placeholder" if kind is None else kind
    if kind is None and source_signal is not None:
        resolved_kind = "proxy" if sign != 1.0 or aliases else "direct"
    if components:
        resolved_kind = "composite"
    return CommonFactorSpec(
        name=name,
        family=family,
        kind=resolved_kind,
        description=description,
        source_signal=source_signal,
        sign=float(sign),
        components=components,
        aliases=aliases,
        macro_candidates=macro_candidates,
        macro_transform=macro_transform,
        notes=notes,
    )


_COMMON_FACTOR_SPECS: tuple[CommonFactorSpec, ...] = (
    _spec(
        "PEADB",
        "events",
        "Benchmark-adjusted post-earnings announcement drift proxy.",
        source_signal="AnnouncementReturn",
        aliases=("PEADB", "PEAD_b", "PEAD-adjusted", "benchmark PEAD"),
        notes="Mapped to the public AnnouncementReturn signal.",
    ),
    _spec(
        "PEAD",
        "events",
        "Post-earnings announcement drift proxy.",
        source_signal="AnnouncementReturn",
        aliases=("PEADB", "PEAD_b", "PEAD-adjusted", "post earnings announcement drift"),
        notes="Mapped to the public AnnouncementReturn signal.",
    ),
    _spec(
        "IVOL",
        "risk",
        "Idiosyncratic volatility proxy.",
        source_signal="RIVolSpread",
        notes="Mapped to the public RIVolSpread signal.",
    ),
    _spec(
        "CREDIT",
        "risk",
        "Credit risk proxy.",
        source_signal="FailureProbability",
        aliases=("CRF", "credit risk", "failure probability"),
        notes="Mapped to the public FailureProbability signal.",
    ),
    _spec(
        "YSP",
        "yield",
        "Yield spread / payout yield proxy.",
        source_signal="NetPayoutYield",
        aliases=("payout yield", "net payout yield"),
        notes="Mapped to the public NetPayoutYield signal.",
    ),
    _spec(
        "MOMBS",
        "momentum",
        "Momentum-business spread proxy.",
        source_signal="Mom6mJunk",
        aliases=("MOMB", "momentum business spread", "6m momentum junk"),
        notes="Mapped to the public Mom6mJunk signal.",
    ),
    _spec(
        "INFLC",
        "macro",
        "Inflation proxy from a macro price index such as CPI or PCE.",
        kind="macro",
        aliases=("inflation", "cpi", "pce", "inflc"),
        macro_candidates=("cpi", "cpiaucsl", "pce", "core_pce", "inflation"),
        macro_transform="yoy",
        notes="Uses a macro inflation series (for example CPI/PCE) and aligns it by date.",
    ),
    _spec(
        "INFLV",
        "macro",
        "Inflation volatility proxy.",
        kind="macro",
        aliases=("inflation volatility", "inflv"),
        macro_candidates=("cpi", "cpiaucsl", "pce", "core_pce", "inflation"),
        macro_transform="vol",
        notes="Uses the rolling volatility of the same macro inflation series used by INFLC.",
    ),
    _spec(
        "EPU",
        "macro",
        "Economic policy uncertainty proxy.",
        kind="macro",
        aliases=("epu", "policy uncertainty", "economic policy uncertainty"),
        macro_candidates=("epu", "policy_uncertainty", "economic_policy_uncertainty"),
        macro_transform="level",
        notes="Uses a supplied EPU / policy uncertainty series if provided.",
    ),
    _spec(
        "EPUT",
        "macro",
        "Economic policy uncertainty trend proxy.",
        kind="macro",
        aliases=("eput", "epu trend"),
        macro_candidates=("epu", "policy_uncertainty", "economic_policy_uncertainty"),
        macro_transform="trend",
        notes="Uses the trend of the supplied EPU / policy uncertainty series.",
    ),
    _spec(
        "UNC",
        "macro",
        "General uncertainty proxy.",
        kind="macro",
        aliases=("uncertainty", "unc"),
        macro_candidates=("unc", "uncertainty", "general_uncertainty"),
        macro_transform="level",
        notes="Uses a supplied uncertainty series if provided.",
    ),
    _spec(
        "UNCF",
        "macro",
        "Uncertainty volatility proxy.",
        kind="macro",
        aliases=("uncf", "uncertainty volatility"),
        macro_candidates=("unc", "uncertainty", "general_uncertainty"),
        macro_transform="vol",
        notes="Uses the rolling volatility of a supplied uncertainty series.",
    ),
    _spec(
        "UNCr",
        "macro",
        "Uncertainty change proxy.",
        kind="macro",
        aliases=("uncr", "uncertainty change"),
        macro_candidates=("unc", "uncertainty", "general_uncertainty"),
        macro_transform="change",
        notes="Uses the change in a supplied uncertainty series.",
    ),
    _spec(
        "TERM",
        "macro",
        "Term spread proxy.",
        kind="macro",
        aliases=("term spread", "yield curve", "term"),
        macro_candidates=("term_spread", "term", "yield_curve", "dgs10_minus_tb3ms"),
        macro_transform="level",
        notes="Uses a supplied term-spread series if provided.",
    ),
    _spec(
        "MKTS",
        "macro",
        "Market stress proxy.",
        kind="macro",
        aliases=("market stress", "mkts"),
        macro_candidates=("mkts", "market_stress", "stress"),
        macro_transform="level",
        notes="Uses a supplied market-stress series if provided.",
    ),
    _spec(
        "LIQNT",
        "liquidity",
        "Liquidity proxy for non-traded / market liquidity.",
        source_signal="AssetLiquidityMarket",
        aliases=("asset liquidity market", "market liquidity", "liquidity nt"),
        notes="Mapped to the public AssetLiquidityMarket signal.",
    ),
    _spec(
        "SZE",
        "size",
        "Firm size proxy.",
        source_signal="Size",
        aliases=("size", "market equity size"),
        notes="Mapped to the public Size signal.",
    ),
    _spec(
        "LIQ",
        "liquidity",
        "Liquidity factor proxy.",
        source_signal="Illiquidity",
        sign=-1.0,
        aliases=("illiquidity", "-illiquidity", "amihud liquidity"),
        notes="Uses the negative of Illiquidity so larger values mean more liquidity.",
    ),
    _spec(
        "MGMT",
        "governance",
        "Management / governance proxy.",
        source_signal="Governance",
        aliases=("governance", "management"),
        notes="Mapped to the public Governance signal.",
    ),
    _spec(
        "MOMS",
        "momentum",
        "Seasonal momentum proxy.",
        source_signal="MomSeason",
        aliases=("mom season", "seasonal momentum"),
        notes="Mapped to the public MomSeason signal.",
    ),
    _spec(
        "DUR",
        "risk",
        "Equity duration / rate sensitivity proxy.",
        source_signal="EquityDuration",
        aliases=("equity duration", "duration"),
        notes="Mapped to the public EquityDuration signal.",
    ),
    _spec(
        "CMA",
        "investment",
        "Conservative-minus-aggressive investment proxy.",
        source_signal="AssetGrowth",
        sign=-1.0,
        aliases=("CMAs", "asset growth", "conservative minus aggressive"),
        notes="Uses negative AssetGrowth so larger values mean more conservative investment.",
    ),
    _spec(
        "CRY",
        "credit",
        "Credit rating / credit-risk proxy.",
        source_signal="CredRatDG",
        aliases=("credit rating", "credit downgrade"),
        notes="Mapped to the public CredRatDG signal.",
    ),
    _spec(
        "CPTL",
        "investment",
        "Capital turnover proxy.",
        source_signal="AssetTurnover",
        aliases=("capital turnover", "CPTLT", "asset turnover"),
        notes="Mapped to the public AssetTurnover signal.",
    ),
    _spec(
        "BAB",
        "risk",
        "Betting-against-beta proxy.",
        source_signal="BetaFP",
        sign=-1.0,
        aliases=("betting against beta", "BetaFP", "low beta"),
        notes="Uses the negative of Frazzini-Pedersen beta so larger values mean lower beta exposure.",
    ),
    _spec(
        "CRF",
        "credit",
        "Credit risk factor proxy.",
        source_signal="FailureProbability",
        aliases=("CREDIT", "failure risk", "failure probability"),
        notes="Mapped to the public FailureProbability signal.",
    ),
    _spec(
        "STREVB",
        "momentum",
        "Short-term reversal proxy.",
        source_signal="MomRev",
        aliases=("STREV", "short-term reversal", "reversal"),
        notes="Mapped to the public MomRev signal.",
    ),
    _spec(
        "DRF",
        "income",
        "Dividend yield / payout proxy.",
        source_signal="DivYield",
        aliases=("dividend yield", "payout yield", "div yield"),
        notes="Mapped to the public DivYield signal.",
    ),
    _spec(
        "LVL",
        "leverage",
        "Leverage proxy.",
        source_signal="BookLeverage",
        aliases=("book leverage", "leverage"),
        notes="Mapped to the public BookLeverage signal.",
    ),
    _spec(
        "BM",
        "value",
        "Book-to-market proxy.",
        kind="direct",
        source_signal="BM",
        aliases=("B/M", "Book-to-Market", "book-to-market", "book to market"),
        notes="Mapped to the public BM signal.",
    ),
    _spec(
        "QMJ",
        "quality",
        "Quality-minus-junk composite proxy.",
        components=(
            ("OperProf", 1.0),
            ("AccrualQuality", 1.0),
            ("FailureProbability", -1.0),
            ("BetaFP", -1.0),
            ("NetPayoutYield", 1.0),
            ("AssetGrowth", -1.0),
        ),
        notes="Built as a rank-mean composite of profitability, accrual quality, low-risk, payout, and low-investment proxies.",
    ),
    _spec(
        "MKTB",
        "risk",
        "Market beta proxy.",
        source_signal="Beta",
        aliases=("MKTBs", "market beta", "beta"),
        notes="Mapped to the public Beta signal.",
    ),
    _spec(
        "VIX",
        "risk",
        "Systematic volatility / VIX beta proxy.",
        source_signal="betaVIX",
        aliases=("betaVIX", "volatility beta", "VIX beta"),
        notes="Mapped to the public betaVIX signal.",
    ),
    _spec(
        "VAL",
        "value",
        "Value composite proxy.",
        components=(
            ("BMdec", 1.0),
            ("IntrinsicValue", 1.0),
            ("AM", 1.0),
        ),
        notes="Built as a rank-mean composite of book-to-market and intrinsic-value proxies.",
    ),
    _spec(
        "HMLs",
        "value",
        "Size-adjusted high-minus-low value proxy.",
        source_signal="BMdec",
        notes="Uses the public BMdec signal as a value proxy.",
    ),
    _spec(
        "R_IA",
        "investment",
        "Investment / capital-allocation proxy.",
        source_signal="ChInvIA",
        notes="Mapped to the public ChInvIA signal.",
    ),
    _spec(
        "FIN",
        "financing",
        "Net financing / financing pressure proxy.",
        source_signal="XFIN",
        notes="Mapped to the public XFIN signal.",
    ),
    _spec(
        "RMW",
        "profitability",
        "Robust-minus-weak profitability proxy.",
        source_signal="OperProf",
        aliases=("RMWs", "operating profitability", "profitability"),
        notes="Mapped to the public OperProf signal.",
    ),
    _spec(
        "SMB",
        "size",
        "Small-minus-big size proxy.",
        source_signal="Size",
        sign=-1.0,
        aliases=("SMBs", "small-minus-big", "small minus big"),
        notes="Uses the negative of Size so larger values mean smaller firms.",
    ),
    _spec(
        "HML",
        "value",
        "High-minus-low value proxy.",
        source_signal="BMdec",
        aliases=("HMLs", "high-minus-low", "value spread"),
        notes="Mapped to the public BMdec signal.",
    ),
    _spec(
        "FSCORE",
        "quality",
        "Piotroski F-score proxy.",
        kind="direct",
        source_signal="PS",
        aliases=("F-Score", "FScore", "Piotroski F-score", "Piotroski", "PS"),
        notes="Mapped to the public PS signal.",
    ),
    _spec(
        "BMFSCORE",
        "value_quality",
        "Book-to-market and F-score composite proxy.",
        components=(
            ("BM", 1.0),
            ("FSCORE", 1.0),
        ),
        aliases=("B/M + F-Score", "BM + F-Score", "value-quality", "value quality"),
        notes="Built as a rank-mean composite of value and fundamental strength.",
    ),
)


@lru_cache(maxsize=1)
def _catalog() -> dict[str, CommonFactorSpec]:
    specs: dict[str, CommonFactorSpec] = {}
    for spec in _COMMON_FACTOR_SPECS:
        specs[spec.name] = spec
        for alias in spec.aliases:
            specs.setdefault(alias, spec)
    return specs


def available_common_factors() -> list[str]:
    """Return the canonical common-factor labels used by the Tiger catalog."""

    return [spec.name for spec in _COMMON_FACTOR_SPECS]


def common_factor_aliases() -> dict[str, str]:
    """Return alias -> canonical-name mappings."""

    aliases: dict[str, str] = {}
    for spec in _COMMON_FACTOR_SPECS:
        for alias in spec.aliases:
            aliases[alias] = spec.name
    return aliases


def common_factor_spec(name: str) -> CommonFactorSpec:
    """Return the catalog entry for a common factor label."""

    try:
        return _catalog()[name]
    except KeyError as exc:
        raise KeyError(f"Unknown common factor: {name}") from exc


def common_factor_group_index() -> dict[str, list[str]]:
    """Group canonical common factors by family."""

    families: dict[str, list[str]] = {}
    for spec in _COMMON_FACTOR_SPECS:
        families.setdefault(spec.family, []).append(spec.name)
    return {family: sorted(names) for family, names in sorted(families.items())}


def common_factor_group_frame() -> pd.DataFrame:
    """Return the common-factor catalog as a tidy DataFrame."""

    rows = []
    for spec in _COMMON_FACTOR_SPECS:
        rows.append(
            {
                "name": spec.name,
                "display_name": spec.aliases[0] if spec.aliases else spec.name,
                "family": spec.family,
                "kind": spec.kind,
                "source_signal": spec.source_signal,
                "sign": spec.sign,
                "aliases": ", ".join(spec.aliases),
                "description": spec.description,
                "notes": spec.notes,
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "name"]).reset_index(drop=True)


def common_factor_group_markdown() -> str:
    """Return the full common-factor catalog as a markdown table."""

    return common_factor_group_frame().to_markdown(index=False)


def common_factor_family_summary() -> pd.DataFrame:
    """Return a one-row-per-family summary of the common-factor catalog."""

    rows = []
    for family, names in common_factor_group_index().items():
        subset = [spec for spec in _COMMON_FACTOR_SPECS if spec.family == family]
        rows.append(
            {
                "family": family,
                "count": len(names),
                "names": ", ".join(names),
                "display_names": ", ".join(
                    spec.aliases[0] if spec.aliases else spec.name for spec in subset
                ),
                "direct": sum(1 for spec in subset if spec.kind == "direct"),
                "proxy": sum(1 for spec in subset if spec.kind == "proxy"),
                "macro": sum(1 for spec in subset if spec.kind == "macro"),
                "composite": sum(1 for spec in subset if spec.kind == "composite"),
                "placeholder": sum(1 for spec in subset if spec.kind == "placeholder"),
            }
        )
    return pd.DataFrame(rows).sort_values(["family"]).reset_index(drop=True)


def common_factor_family_markdown() -> str:
    """Return the family summary as a markdown table."""

    return common_factor_family_summary().to_markdown(index=False)


def common_factor_group_names() -> list[str]:
    """Return the canonical family names used by the common-factor catalog."""

    return sorted(common_factor_group_index())


def common_factor_display_names() -> dict[str, str]:
    """Return a canonical-name -> display-name mapping for chart-style labels.

    The first alias is treated as the preferred display label when available.
    """

    display_names: dict[str, str] = {}
    for spec in _COMMON_FACTOR_SPECS:
        display_names[spec.name] = spec.aliases[0] if spec.aliases else spec.name
    return display_names


def find_common_factor_group(name: str) -> str:
    """Return the family name for a common-factor label."""

    return common_factor_spec(name).family


def _ensure_input_frame(data: pd.DataFrame | None, factor_name: str) -> pd.DataFrame:
    if data is None:
        raise ValueError(
            f"{factor_name} needs a long-form input frame with at least date_/code columns "
            "so the common-factor proxy can be aligned or combined."
        )
    if "date_" not in data.columns and "date" not in data.columns:
        raise ValueError(f"{factor_name} expects a date_ or date column.")
    if "code" not in data.columns and "permno" not in data.columns:
        raise ValueError(f"{factor_name} expects a code or permno column.")
    return data


def _ensure_macro_input_frame(data: pd.DataFrame | None, factor_name: str) -> pd.DataFrame:
    if data is None:
        raise ValueError(
            f"{factor_name} needs a macro date series or a long-form stock frame to align the macro proxy."
        )
    if "date_" not in data.columns and "date" not in data.columns:
        raise ValueError(f"{factor_name} expects a date_ or date column.")
    return data


def _canonical_dates(data: pd.DataFrame) -> pd.Series:
    if "date_" in data.columns:
        return pd.to_datetime(data["date_"])
    return pd.to_datetime(data["date"])


def _canonical_codes(data: pd.DataFrame) -> pd.Series:
    if "code" in data.columns:
        return data["code"].astype(str)
    return data["permno"].astype(str)


def _normalize_dataset_aliases(
    datasets: Mapping[str, object] | None,
) -> dict[str, object]:
    aliases: dict[str, object] = {}
    for key, value in (datasets or {}).items():
        path_key = Path(str(key))
        aliases[path_key.name] = value
        aliases[path_key.stem] = value
        aliases[path_key.name.lower()] = value
        aliases[path_key.stem.lower()] = value
    return aliases


def _macro_date_column(frame: pd.DataFrame) -> str:
    if "date_" in frame.columns:
        return "date_"
    if "date" in frame.columns:
        return "date"
    raise ValueError("macro input expects a date_ or date column.")


def _macro_value_column(frame: pd.DataFrame, *, factor_name: str) -> str:
    if "value" in frame.columns:
        return "value"
    candidates = [
        column
        for column in frame.columns
        if column not in {"date_", "date", "code", "permno"}
    ]
    if len(candidates) == 1:
        return candidates[0]
    numeric_candidates: list[str] = []
    for column in candidates:
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.notna().any():
            numeric_candidates.append(column)
    if len(numeric_candidates) == 1:
        return numeric_candidates[0]
    raise ValueError(
        f"{factor_name} expects a macro frame with a single numeric value column or an explicit 'value' column; "
        f"got {candidates!r}"
    )


def _normalize_macro_frame(frame: pd.DataFrame, *, factor_name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date_", "value"])
    normalized = frame.copy()
    date_column = _macro_date_column(normalized)
    value_column = _macro_value_column(normalized, factor_name=factor_name)
    normalized[date_column] = pd.to_datetime(normalized[date_column], errors="coerce").dt.tz_localize(None)
    normalized[value_column] = pd.to_numeric(normalized[value_column], errors="coerce")
    keep_columns = [date_column, value_column]
    if "code" in normalized.columns or "permno" in normalized.columns:
        code_column = "code" if "code" in normalized.columns else "permno"
        normalized[code_column] = normalized[code_column].astype(str)
        grouped = normalized.loc[:, [date_column, value_column]].groupby(date_column, sort=False, as_index=False).mean()
        normalized = grouped.rename(columns={date_column: "date_", value_column: "value"})
    else:
        normalized = normalized.loc[:, keep_columns].rename(columns={date_column: "date_", value_column: "value"})
    normalized = normalized.dropna(subset=["date_", "value"])
    normalized = normalized.sort_values(["date_"], kind="stable").drop_duplicates(subset=["date_"], keep="last")
    if normalized.empty:
        raise ValueError(f"{factor_name} macro frame contains no valid rows after normalization")
    return normalized.reset_index(drop=True)


def _find_macro_dataset(
    datasets: Mapping[str, object] | None,
    *,
    names: tuple[str, ...],
) -> pd.DataFrame | None:
    if not datasets:
        return None
    aliases = _normalize_dataset_aliases(datasets)
    lowered = {str(key).lower(): value for key, value in aliases.items()}
    for name in names:
        candidate_keys = {
            name,
            name.lower(),
            name.upper(),
            Path(name).name,
            Path(name).stem,
            Path(name).name.lower(),
            Path(name).stem.lower(),
        }
        for key in candidate_keys:
            if key in aliases:
                value = aliases[key]
                if isinstance(value, pd.DataFrame):
                    return value
                if hasattr(value, "to_pandas"):
                    return value.to_pandas()
            value = lowered.get(key.lower())
            if value is None:
                continue
            if isinstance(value, pd.DataFrame):
                return value
            if hasattr(value, "to_pandas"):
                return value.to_pandas()
    return None


def _infer_macro_periods(dates: pd.Series) -> int:
    ordered = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if len(ordered) < 2:
        return 12
    deltas = ordered.diff().dt.days.dropna()
    if deltas.empty:
        return 12
    median_delta = float(deltas.median())
    if median_delta <= 10:
        return 252
    if median_delta <= 45:
        return 12
    if median_delta <= 120:
        return 4
    return 1


def _macro_transform_series(
    frame: pd.DataFrame,
    *,
    factor_name: str,
    transform: str,
) -> pd.Series:
    macro = _normalize_macro_frame(frame, factor_name=factor_name)
    periods = _infer_macro_periods(macro["date_"])
    value = macro["value"].astype(float)
    transform_key = transform.lower()
    if transform_key in {"level", "raw"}:
        series = value
    elif transform_key in {"yoy", "inflation"}:
        series = value.pct_change(periods=periods) * 100.0
    elif transform_key in {"change", "diff"}:
        series = value.diff(periods=periods)
    elif transform_key in {"vol", "volatility"}:
        base = value.pct_change(periods=periods) * 100.0
        vol_window = max(3, periods)
        series = base.rolling(window=vol_window, min_periods=max(2, vol_window // 2)).std(ddof=0)
    elif transform_key in {"trend"}:
        short_window = max(2, periods)
        long_window = max(short_window + 1, periods * 3)
        short_ma = value.rolling(window=short_window, min_periods=max(2, short_window // 2)).mean()
        long_ma = value.rolling(window=long_window, min_periods=max(2, long_window // 2)).mean()
        series = short_ma - long_ma
    else:
        raise ValueError(f"Unsupported macro transform {transform!r} for {factor_name}")
    series.name = factor_name
    return pd.Series(series.to_numpy(), index=macro["date_"], name=factor_name)


def _align_macro_values(
    frame: pd.DataFrame,
    macro_values: pd.Series,
    *,
    factor_name: str,
) -> pd.Series:
    macro_frame = macro_values.rename("value").reset_index().rename(columns={macro_values.index.name or "index": "date_"})
    macro_frame["date_"] = pd.to_datetime(macro_frame["date_"], errors="coerce").dt.tz_localize(None)
    macro_frame = macro_frame.dropna(subset=["date_"]).sort_values("date_", kind="stable")
    macro_frame = macro_frame.drop_duplicates(subset=["date_"], keep="last")
    working = frame.copy()
    if "code" not in working.columns and "permno" not in working.columns:
        working["code"] = "__macro__"
    working_dates = _canonical_dates(working)
    lookup = pd.DataFrame({"__row__": working.index.to_numpy(), "date_": working_dates.to_numpy()})
    lookup["date_"] = pd.to_datetime(lookup["date_"], errors="coerce")
    aligned = pd.merge_asof(
        lookup.sort_values("date_"),
        macro_frame.sort_values("date_"),
        on="date_",
        direction="backward",
    ).sort_values("__row__", kind="stable")
    values = pd.to_numeric(aligned["value"], errors="coerce")
    values.index = working.index
    values.name = factor_name
    return values


def _run_macro_factor(
    factor_name: str,
    data: pd.DataFrame | None,
    *,
    input_name: str | None = None,
    datasets: Mapping[str, object] | None = None,
    return_frame: bool = False,
) -> pd.Series | pd.DataFrame:
    spec = common_factor_spec(factor_name)
    frame = data.copy() if data is not None else None
    if frame is not None and "date_" not in frame.columns and "date" not in frame.columns:
        raise ValueError(f"{factor_name} expects a date_ or date column.")

    macro_frame = None
    if frame is not None and ("code" not in frame.columns and "permno" not in frame.columns):
        macro_frame = frame
    if macro_frame is None:
        candidate_names = (spec.macro_candidates or ()) + (factor_name.lower(), factor_name.upper(), "macro")
        macro_frame = _find_macro_dataset(datasets, names=candidate_names)
    if macro_frame is None:
        raise ValueError(
            f"{factor_name} needs a macro input frame or datasets={{'cpi': ...}} / datasets={{'macro': ...}}."
        )

    macro_transform = spec.macro_transform
    if macro_transform is None:
        macro_transform = "yoy" if factor_name == "INFLC" else "level"
    macro_series = _macro_transform_series(macro_frame, factor_name=factor_name, transform=macro_transform)
    if frame is None:
        macro_output = macro_series.rename(factor_name).reset_index().rename(columns={"index": "date_"})
        macro_output["date_"] = pd.to_datetime(macro_output["date_"], errors="coerce").dt.tz_localize(None)
        macro_output = macro_output.dropna(subset=["date_"]).sort_values("date_", kind="stable")
        if return_frame:
            macro_output = macro_output.assign(code="__macro__")
            return macro_output.loc[:, ["date_", "code", factor_name]].reset_index(drop=True)
        values = pd.to_numeric(macro_output[factor_name], errors="coerce")
        values.index = macro_output.index
        values.name = factor_name
        return values

    values = _align_macro_values(frame, macro_series, factor_name=factor_name)
    return _align_result(frame, values, factor_name, return_frame=return_frame)


def _align_result(
    data: pd.DataFrame,
    values: pd.Series,
    factor_name: str,
    *,
    return_frame: bool,
) -> pd.Series | pd.DataFrame:
    if return_frame:
        return pd.DataFrame(
            {
                "date_": _canonical_dates(data),
                "code": _canonical_codes(data),
                factor_name: values.to_numpy(),
            }
        )
    values = values.copy()
    values.index = data.index
    values.name = factor_name
    return values


def _run_direct_factor(
    factor_name: str,
    data: pd.DataFrame | None,
    *,
    source_signal: str,
    sign: float = 1.0,
    input_name: str | None = None,
    datasets: Mapping[str, object] | None = None,
    return_frame: bool = False,
) -> pd.Series | pd.DataFrame:
    frame = _ensure_input_frame(data, factor_name)
    signal = run_original_factor(
        source_signal,
        frame,
        input_name=input_name,
        datasets=datasets,
        return_frame=False,
    )
    if not isinstance(signal, pd.Series):
        signal = pd.Series(signal, index=frame.index, name=source_signal)
    signal = pd.to_numeric(signal, errors="coerce") * float(sign)
    return _align_result(frame, signal, factor_name, return_frame=return_frame)


def _run_composite_factor(
    spec: CommonFactorSpec,
    data: pd.DataFrame | None,
    *,
    input_name: str | None = None,
    datasets: Mapping[str, object] | None = None,
    return_frame: bool = False,
) -> pd.Series | pd.DataFrame:
    frame = _ensure_input_frame(data, spec.name)
    date_values = _canonical_dates(frame)

    component_scores: dict[str, pd.Series] = {}
    for component_name, component_sign in spec.components:
        if component_name in _catalog():
            component_spec = common_factor_spec(component_name)
            if component_spec.is_composite:
                component_values = run_common_factor(
                    component_spec.name,
                    frame,
                    input_name=input_name,
                    datasets=datasets,
                    return_frame=False,
                )
            else:
                component_values = _run_direct_factor(
                    component_spec.name,
                    frame,
                    source_signal=component_spec.source_signal or component_spec.name,
                    sign=component_spec.sign,
                    input_name=input_name,
                    datasets=datasets,
                    return_frame=False,
                )
        elif component_name in available_factors():
            component_values = run_original_factor(
                component_name,
                frame,
                input_name=input_name,
                datasets=datasets,
                return_frame=False,
            )
        else:
            raise KeyError(
                f"Unknown component {component_name!r} in composite common factor {spec.name!r}."
            )
        if not isinstance(component_values, pd.Series):
            component_values = pd.Series(component_values, index=frame.index, name=component_name)
        component_scores[component_name] = pd.to_numeric(component_values, errors="coerce") * float(component_sign)

    stacked = pd.DataFrame(component_scores)
    stacked["__date__"] = date_values.to_numpy()
    ranked = stacked.groupby("__date__", sort=False)[list(component_scores)].rank(pct=True)
    score = ranked.mean(axis=1)
    return _align_result(frame, score.rename(spec.name), spec.name, return_frame=return_frame)


def run_common_factor(
    name: str,
    data: pd.DataFrame | None,
    *,
    input_name: str | None = None,
    datasets: Mapping[str, object] | None = None,
    return_frame: bool = False,
) -> pd.Series | pd.DataFrame:
    """Run a Tiger-side common factor.

    The catalog prefers the public OpenAssetPricing implementation whenever we
    have a direct source signal. Family-style factors such as ``QMJ`` and
    ``VAL`` are computed as small cross-sectional composites of public signals.
    """

    spec = common_factor_spec(name)
    if spec.is_placeholder:
        raise NotImplementedError(
            f"{spec.name} is currently a placeholder. {spec.notes or 'No implementation note available.'}"
        )
    if spec.is_macro:
        return _run_macro_factor(
            spec.name,
            data,
            input_name=input_name,
            datasets=datasets,
            return_frame=return_frame,
        )
    if spec.is_composite:
        return _run_composite_factor(
            spec,
            data,
            input_name=input_name,
            datasets=datasets,
            return_frame=return_frame,
        )
    if spec.source_signal is None:
        raise NotImplementedError(f"{spec.name} does not have a source signal yet.")
    return _run_direct_factor(
        spec.name,
        data,
        source_signal=spec.source_signal,
        sign=spec.sign,
        input_name=input_name,
        datasets=datasets,
        return_frame=return_frame,
    )


def run_common_factors(
    names: Iterable[str] | None,
    data: pd.DataFrame | None,
    *,
    input_name: str | None = None,
    datasets: Mapping[str, object] | None = None,
    return_frame: bool = False,
) -> pd.DataFrame:
    selected = list(names or available_common_factors())
    frames: list[pd.DataFrame] = []
    for name in selected:
        result = run_common_factor(
            name,
            data,
            input_name=input_name,
            datasets=datasets,
            return_frame=True,
        )
        if not isinstance(result, pd.DataFrame):
            raise TypeError(f"{name} did not return a DataFrame in return_frame mode.")
        frames.append(result)
    if not frames:
        return pd.DataFrame(columns=["date_", "code"])
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["date_", "code"], how="outer")
    if not return_frame:
        return merged.sort_values(["date_", "code"]).reset_index(drop=True)
    return merged.sort_values(["date_", "code"]).reset_index(drop=True)


def run_value_quality_combo(
    data: pd.DataFrame,
    *,
    input_name: str | None = None,
    datasets: Mapping[str, object] | None = None,
    high_quantile: float = 0.5,
) -> pd.DataFrame:
    """Return BM, F-Score, a composite score, and a 2x2 value-quality bucket.

    The returned frame is date-cross-sectional and includes:

    - ``BM``: raw book-to-market proxy
    - ``FSCORE``: Piotroski F-score proxy
    - ``BMFSCORE``: rank-mean composite of the two
    - ``BM_rank`` / ``FSCORE_rank``: within-date percentile ranks
    - ``BM_high`` / ``FSCORE_high``: high-group flags using ``high_quantile``
    - ``value_quality_bucket``: one of ``HH``, ``HL``, ``LH``, ``LL``
    """

    if not 0.0 < float(high_quantile) < 1.0:
        raise ValueError("high_quantile must be in (0, 1).")
    frame = _ensure_input_frame(data, "value_quality_combo")
    combo = run_common_factors(
        ["BM", "FSCORE", "BMFSCORE"],
        frame,
        input_name=input_name,
        datasets=datasets,
        return_frame=True,
    )
    if combo.empty:
        return combo
    combo = combo.sort_values(["date_", "code"], kind="stable").reset_index(drop=True)
    combo["BM_rank"] = combo.groupby("date_", sort=False)["BM"].rank(pct=True, method="average")
    combo["FSCORE_rank"] = combo.groupby("date_", sort=False)["FSCORE"].rank(pct=True, method="average")
    combo["BM_high"] = combo["BM_rank"] > float(high_quantile)
    combo["FSCORE_high"] = combo["FSCORE_rank"] > float(high_quantile)
    combo["value_quality_bucket"] = "LL"
    combo.loc[combo["BM_high"] & combo["FSCORE_high"], "value_quality_bucket"] = "HH"
    combo.loc[combo["BM_high"] & ~combo["FSCORE_high"], "value_quality_bucket"] = "HL"
    combo.loc[~combo["BM_high"] & combo["FSCORE_high"], "value_quality_bucket"] = "LH"
    combo["value_quality_bucket_score"] = combo["value_quality_bucket"].map(
        {"LL": 0.0, "LH": 1.0, "HL": 2.0, "HH": 3.0}
    ).astype(float)
    return combo


def run_value_quality_combo_from_columns(
    data: pd.DataFrame,
    *,
    bm_column: str,
    fscore_column: str,
    high_quantile: float = 0.5,
) -> pd.DataFrame:
    """Build the BM/F-score 2x2 screen directly from existing columns."""

    if not 0.0 < float(high_quantile) < 1.0:
        raise ValueError("high_quantile must be in (0, 1).")
    frame = _ensure_input_frame(data, "value_quality_combo_from_columns")
    required = {"date_", "code", bm_column, fscore_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"missing required columns for value-quality combo: {missing}")

    combo = frame.loc[:, ["date_", "code", bm_column, fscore_column]].copy()
    combo = combo.rename(columns={bm_column: "BM", fscore_column: "FSCORE"})
    combo = combo.sort_values(["date_", "code"], kind="stable").reset_index(drop=True)
    combo["BM_rank"] = combo.groupby("date_", sort=False)["BM"].rank(pct=True, method="average")
    combo["FSCORE_rank"] = combo.groupby("date_", sort=False)["FSCORE"].rank(pct=True, method="average")
    combo["BMFSCORE"] = (combo["BM_rank"] + combo["FSCORE_rank"]) / 2.0
    combo["BM_high"] = combo["BM_rank"] > float(high_quantile)
    combo["FSCORE_high"] = combo["FSCORE_rank"] > float(high_quantile)
    combo["value_quality_bucket"] = "LL"
    combo.loc[combo["BM_high"] & combo["FSCORE_high"], "value_quality_bucket"] = "HH"
    combo.loc[combo["BM_high"] & ~combo["FSCORE_high"], "value_quality_bucket"] = "HL"
    combo.loc[~combo["BM_high"] & combo["FSCORE_high"], "value_quality_bucket"] = "LH"
    combo["value_quality_bucket_score"] = combo["value_quality_bucket"].map(
        {"LL": 0.0, "LH": 1.0, "HL": 2.0, "HH": 3.0}
    ).astype(float)
    return combo


def run_value_quality_long_short_backtest(
    data: pd.DataFrame,
    close_panel: pd.DataFrame,
    *,
    input_name: str | None = None,
    datasets: Mapping[str, object] | None = None,
    bm_column: str | None = None,
    fscore_column: str | None = None,
    high_quantile: float = 0.5,
    long_pct: float = 0.2,
    rebalance_freq: str = "ME",
    long_short: bool = True,
    annual_trading_days: int = 252,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> dict[str, object]:
    """Run a bucket-score long/short backtest for the BM/F-score screen."""

    from tiger_factors.multifactor_evaluation.pipeline import run_factor_backtest

    if bm_column is not None or fscore_column is not None:
        if bm_column is None or fscore_column is None:
            raise ValueError("bm_column and fscore_column must either both be set or both be None.")
        combo = run_value_quality_combo_from_columns(
            data,
            bm_column=bm_column,
            fscore_column=fscore_column,
            high_quantile=high_quantile,
        )
    else:
        combo = run_value_quality_combo(
            data,
            input_name=input_name,
            datasets=datasets,
            high_quantile=high_quantile,
        )
    if combo.empty:
        raise ValueError("value-quality combo is empty.")
    signal_panel = (
        combo.loc[:, ["date_", "code", "value_quality_bucket_score"]]
        .pivot(index="date_", columns="code", values="value_quality_bucket_score")
        .sort_index()
    )
    signal_panel.index = pd.DatetimeIndex(signal_panel.index, name="date_")
    backtest, stats = run_factor_backtest(
        signal_panel,
        close_panel,
        long_pct=long_pct,
        rebalance_freq=rebalance_freq,
        long_short=long_short,
        annual_trading_days=annual_trading_days,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )
    return {
        "combo_frame": combo,
        "signal_panel": signal_panel,
        "backtest": backtest,
        "stats": stats,
        "portfolio_returns": backtest["portfolio"] if "portfolio" in backtest else pd.Series(dtype=float),
        "benchmark_returns": backtest["benchmark"] if "benchmark" in backtest else pd.Series(dtype=float),
    }


__all__ = [
    "CommonFactorSpec",
    "available_common_factors",
    "common_factor_aliases",
    "common_factor_family_summary",
    "common_factor_family_markdown",
    "common_factor_display_names",
    "common_factor_group_frame",
    "common_factor_group_markdown",
    "common_factor_group_index",
    "common_factor_group_names",
    "common_factor_spec",
    "find_common_factor_group",
    "run_common_factor",
    "run_common_factors",
    "run_value_quality_combo",
    "run_value_quality_combo_from_columns",
    "run_value_quality_long_short_backtest",
]
