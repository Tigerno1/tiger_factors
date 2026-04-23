"""Local Python portfolio engine for traditional OpenAssetPricing signals.

This module ports the core monthly portfolio construction logic from the
upstream R pipeline into a local Python implementation that can live alongside
the vendored factor scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .factor_functions import OPEN_ASSET_PRICING_ROOT

TRADITIONAL_FACTORS_DIR = Path(__file__).resolve().parent
DEFAULT_SIGNAL_DOC_PATH = OPEN_ASSET_PRICING_ROOT / "SignalDoc.csv"


def load_signal_doc(path: str | Path = DEFAULT_SIGNAL_DOC_PATH) -> pd.DataFrame:
    """Load the local SignalDoc catalog with Python-friendly column names."""

    docs = pd.read_csv(path)
    docs = docs.rename(
        columns={
            "Acronym": "signalname",
            "Stock Weight": "sweight",
            "LS Quantile": "q_cut",
            "Quantile Filter": "q_filt",
            "Portfolio Period": "portperiod",
            "Start Month": "startmonth",
            "Filter": "filterstr",
        }
    )
    docs["filterstr"] = docs["filterstr"].replace({"NA": np.nan, "None": np.nan, "none": np.nan})
    return docs


def check_signal_csvs(
    signal_doc: pd.DataFrame,
    *,
    paths: TraditionalPortfolioPaths,
) -> dict[str, list[str]]:
    """Python equivalent of the upstream signal completeness check."""

    expected = sorted(
        signal_doc.loc[signal_doc["Cat.Signal"] != "Drop", "signalname"].dropna().unique().tolist()
    )
    predictors = sorted(path.stem for path in paths.predictors_dir.glob("*.csv")) if paths.predictors_dir.exists() else []
    placebos = sorted(path.stem for path in paths.placebos_dir.glob("*.csv")) if paths.placebos_dir.exists() else []
    crsp_predictors = (
        sorted(path.stem for path in paths.crsp_predictors_dir.glob("*.csv"))
        if paths.crsp_predictors_dir is not None and paths.crsp_predictors_dir.exists()
        else []
    )
    found = sorted(set(predictors) | set(placebos) | set(crsp_predictors))
    return {
        "missing": sorted(set(expected) - set(found)),
        "extra": sorted(set(found) - set(expected)),
    }


@dataclass(frozen=True)
class TraditionalPortfolioPaths:
    """Filesystem layout for locally stored signal csvs."""

    predictors_dir: Path
    placebos_dir: Path
    crsp_predictors_dir: Path | None = None
    temp_dir: Path | None = None


def _coerce_paths(
    paths: TraditionalPortfolioPaths | None,
    *,
    project_root: str | Path | None = None,
) -> TraditionalPortfolioPaths | None:
    if paths is not None:
        return paths
    if project_root is None:
        return None
    root = Path(project_root)
    return TraditionalPortfolioPaths(
        predictors_dir=root / "Signals" / "pyData" / "Predictors",
        placebos_dir=root / "Signals" / "pyData" / "Placebos",
        crsp_predictors_dir=root / "Signals" / "pyData" / "CRSPPredictors",
        temp_dir=root / "Signals" / "pyData" / "temp",
    )


def create_crsp_predictors(
    crspret: pd.DataFrame,
    crspinfo: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Create the simple CRSP-only predictors used by the upstream portfolios."""

    info = crspinfo.copy()
    if "yyyymm" not in info.columns:
        raise ValueError("crspinfo needs a yyyymm column")

    predictors = {
        "Price": info.loc[:, ["permno", "yyyymm"]].assign(Price=np.log(info["prc"].abs())),
        "Size": info.loc[:, ["permno", "yyyymm"]].assign(Size=np.log(info["me"])),
    }

    ret = crspret.copy()
    if "date" not in ret.columns:
        raise ValueError("crspret needs a date column for STreversal")
    if "yyyymm" not in ret.columns:
        ret["yyyymm"] = pd.to_datetime(ret["date"]).dt.year * 100 + pd.to_datetime(ret["date"]).dt.month
    predictors["STreversal"] = ret.loc[:, ["permno", "yyyymm"]].assign(
        STreversal=ret["ret"].fillna(0.0)
    )

    for frame in predictors.values():
        signal_col = [c for c in frame.columns if c not in {"permno", "yyyymm"}][0]
        frame.dropna(subset=[signal_col], inplace=True)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, frame in predictors.items():
            frame.to_csv(out / f"{name}.csv", index=False)

    return predictors


class TraditionalPortfolioEngine:
    """Python port of the monthly OpenAssetPricing portfolio formation logic."""

    def __init__(
        self,
        *,
        crspret: pd.DataFrame,
        crspinfo: pd.DataFrame,
        signal_doc: pd.DataFrame | None = None,
        paths: TraditionalPortfolioPaths | None = None,
        project_root: str | Path | None = None,
        verbose: bool = False,
    ) -> None:
        self.crspret = self._prepare_crspret(crspret)
        self.crspinfo = self._prepare_crspinfo(crspinfo)
        self.signal_doc = load_signal_doc() if signal_doc is None else signal_doc.copy()
        self.paths = _coerce_paths(paths, project_root=project_root)
        self.verbose = verbose

    @staticmethod
    def _prepare_crspinfo(frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        if "yyyymm" not in data.columns:
            if "date" in data.columns:
                date = pd.to_datetime(data["date"])
                data["yyyymm"] = date.dt.year * 100 + date.dt.month
            else:
                raise ValueError("crspinfo needs yyyymm or date")
        return data

    @staticmethod
    def _prepare_crspret(frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        if "date" not in data.columns:
            raise ValueError("crspret needs a date column")
        data["date"] = pd.to_datetime(data["date"])
        if "yyyymm" not in data.columns:
            data["yyyymm"] = data["date"].dt.year * 100 + data["date"].dt.month
        return data

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _strategy_row(self, signalname: str) -> pd.Series:
        docs = self.signal_doc.loc[self.signal_doc["signalname"] == signalname]
        if docs.empty:
            raise KeyError(f"{signalname} is not documented in SignalDoc")
        return docs.iloc[0]

    def strategy_subset(
        self,
        *,
        cat_signal: str | None = None,
        continuous_only: bool = False,
        quickrun: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Select a documented subset of strategies."""

        strategies = self.signal_doc.copy()
        if cat_signal is not None:
            strategies = strategies.loc[strategies["Cat.Signal"] == cat_signal].copy()
        if continuous_only:
            strategies = strategies.loc[strategies["Cat.Form"] == "continuous"].copy()
        if quickrun is not None:
            quickset = set(quickrun)
            strategies = strategies.loc[strategies["signalname"].isin(quickset)].copy()
        return strategies.reset_index(drop=True)

    def _read_signal_csv(self, signalname: str) -> pd.DataFrame:
        if self.paths is None:
            raise FileNotFoundError(
                f"No local signal paths configured for {signalname}. "
                "Pass signal_frame=... or initialize the engine with project_root=... / paths=..."
            )
        candidates = [
            self.paths.predictors_dir / f"{signalname}.csv",
            *( [self.paths.crsp_predictors_dir / f"{signalname}.csv"] if self.paths.crsp_predictors_dir else [] ),
            self.paths.placebos_dir / f"{signalname}.csv",
            *( [self.paths.temp_dir / f"{signalname}.csv"] if self.paths.temp_dir else [] ),
        ]
        for path in candidates:
            if path.exists():
                return pd.read_csv(path)
        raise FileNotFoundError(f"Signal csv not found locally for {signalname}")

    def import_signal(
        self,
        signalname: str,
        *,
        sign: float = 1.0,
        filterstr: str | None = None,
        signal_frame: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Load a signal frame and join implementation metadata from crspinfo."""

        signal = self._read_signal_csv(signalname) if signal_frame is None else signal_frame.copy()
        if signalname not in signal.columns:
            candidate_cols = [c for c in signal.columns if c not in {"permno", "yyyymm"}]
            if len(candidate_cols) != 1:
                raise ValueError(
                    f"Signal frame for {signalname} must contain a {signalname} column "
                    f"or exactly one signal column. Found: {candidate_cols}"
                )
            signal = signal.rename(columns={candidate_cols[0]: signalname})

        signal = signal.rename(columns={signalname: "signal"}).copy()
        signal = signal.dropna(subset=["signal"]).copy()
        signal["yyyymm"] = signal["yyyymm"].astype(int)

        signal = signal.merge(self.crspinfo, on=["permno", "yyyymm"], how="left")

        if filterstr:
            signal = signal.query(filterstr, engine="python").copy()

        signal["signal"] = signal["signal"] * float(sign)
        return signal

    @staticmethod
    def _quantile_breaks(signal: pd.DataFrame, q_cut: float, q_filt: str | None) -> pd.DataFrame:
        tempbreak = signal.copy()
        if q_filt == "NYSE":
            tempbreak = tempbreak.loc[tempbreak["exchcd"] == 1].copy()

        if q_cut <= (1 / 3):
            plist = np.arange(q_cut, 1 - (2 * q_cut) + 1e-12, q_cut).tolist() + [1 - q_cut]
        else:
            plist = sorted(set([q_cut, 1 - q_cut]))

        break_frames: list[pd.DataFrame] = []
        for idx, prob in enumerate(plist, start=1):
            current = (
                tempbreak.groupby("yyyymm")["signal"]
                .quantile(prob)
                .rename(f"break{idx}")
                .reset_index()
            )
            break_frames.append(current)

        breaks = break_frames[0]
        for frame in break_frames[1:]:
            breaks = breaks.merge(frame, on="yyyymm", how="outer")

        if len(plist) > 1:
            last_col = f"break{len(plist)}"
            breaks = breaks.loc[(breaks[last_col] - breaks["break1"]) > 0].copy()

        merged = signal.merge(breaks, on="yyyymm", how="left")
        merged["port"] = pd.Series(pd.NA, index=merged.index, dtype="Int64")

        merged.loc[merged["signal"] <= merged["break1"], "port"] = 1
        for porti in range(2, len(plist) + 1):
            break_col = f"break{porti}"
            mask = merged["port"].isna() & (merged["signal"] < merged[break_col])
            merged.loc[mask, "port"] = porti

        final_col = f"break{len(plist)}"
        merged.loc[merged["port"].isna() & (merged["signal"] >= merged[final_col]), "port"] = len(plist) + 1

        drop_cols = [f"break{i}" for i in range(1, len(plist) + 1)]
        return merged.drop(columns=drop_cols)

    @staticmethod
    def _assign_portfolios(
        signal: pd.DataFrame,
        *,
        cat_form: str,
        q_cut: float,
        q_filt: str | None,
    ) -> pd.DataFrame:
        if cat_form == "continuous":
            return TraditionalPortfolioEngine._quantile_breaks(signal, q_cut=q_cut, q_filt=q_filt)
        if cat_form == "discrete":
            support = sorted(pd.Series(signal["signal"]).dropna().unique().tolist())
            mapping = {value: idx for idx, value in enumerate(support, start=1)}
            assigned = signal.copy()
            assigned["port"] = assigned["signal"].map(mapping).astype("Int64")
            return assigned
        if cat_form == "custom":
            assigned = signal.copy()
            assigned["port"] = pd.Series(assigned["signal"], dtype="Int64")
            return assigned
        raise ValueError(f"Unsupported Cat.Form: {cat_form}")

    @staticmethod
    def _rebalance_months(startmonth: int, portperiod: int) -> list[int]:
        rebmonths = sorted({((startmonth + i * portperiod) % 12) or 12 for i in range(13)})
        return rebmonths

    @staticmethod
    def _lag_yyyymm(yyyymm: pd.Series) -> pd.Series:
        month = yyyymm % 100
        year = yyyymm // 100
        month = month + 1
        year = year + (month == 13).astype(int)
        month = month.where(month != 13, 1)
        return year * 100 + month

    def build_signal_portfolio(
        self,
        signalname: str,
        *,
        signal_frame: pd.DataFrame | None = None,
        cat_form: str | None = None,
        q_cut: float | None = None,
        sweight: str | None = None,
        sign: float | None = None,
        longportname: Iterable[str | int] | None = None,
        shortportname: Iterable[str | int] | None = None,
        startmonth: int | None = None,
        portperiod: int | None = None,
        q_filt: str | None = None,
        filterstr: str | None = None,
        passive_gain: bool = False,
    ) -> pd.DataFrame:
        """Build monthly portfolios for one signal."""

        strategy = self._strategy_row(signalname)
        cat_form = strategy["Cat.Form"] if cat_form is None or pd.isna(cat_form) else cat_form
        q_cut = 0.2 if q_cut is None or pd.isna(q_cut) else float(q_cut)
        sweight = "EW" if sweight is None or pd.isna(sweight) else str(sweight)
        sign = 1.0 if sign is None or pd.isna(sign) else float(sign)
        startmonth = 6 if startmonth is None or pd.isna(startmonth) else int(startmonth)
        portperiod = 1 if portperiod is None or pd.isna(portperiod) else int(portperiod)
        q_filt = strategy.get("q_filt") if q_filt is None else q_filt
        q_filt = None if pd.isna(q_filt) else q_filt
        filterstr = strategy.get("filterstr") if filterstr is None else filterstr
        filterstr = None if pd.isna(filterstr) else filterstr

        signal = self.import_signal(
            signalname,
            sign=sign,
            filterstr=filterstr,
            signal_frame=signal_frame,
        )
        signal = self._assign_portfolios(signal, cat_form=cat_form, q_cut=q_cut, q_filt=q_filt)

        rebmonths = self._rebalance_months(startmonth, portperiod)
        signal = signal.sort_values(["permno", "yyyymm"]).copy()
        signal.loc[~signal["yyyymm"].mod(100).isin(rebmonths), "port"] = pd.NA
        signal["port"] = signal.groupby("permno")["port"].ffill().astype("Int64")
        signal = signal.loc[signal["port"].notna()].copy()

        signallag = signal.loc[:, ["permno", "yyyymm", "signal", "port"]].copy()
        signallag["yyyymm"] = self._lag_yyyymm(signallag["yyyymm"])
        signallag = signallag.rename(columns={"signal": "signallag"})

        crspret = self.crspret.merge(signallag, on=["permno", "yyyymm"], how="left")

        if sweight == "VW":
            if "melag" not in crspret.columns:
                raise ValueError("VW portfolios require melag on crspret")
            crspret["weight"] = crspret["melag"]
        else:
            crspret["weight"] = 1.0

        if passive_gain:
            if "passgain" not in crspret.columns:
                raise ValueError("passive_gain=True requires passgain on crspret")
            crspret["weight"] = crspret["weight"] * crspret["passgain"]

        valid = crspret.loc[
            crspret["port"].notna() & crspret["ret"].notna() & crspret["weight"].notna()
        ].copy()

        def _weighted_mean(group: pd.DataFrame, column: str) -> float:
            return float(np.average(group[column], weights=group["weight"]))

        grouped_rows: list[dict[str, object]] = []
        for (port, date), group in valid.groupby(["port", "date"], sort=True):
            grouped_rows.append(
                {
                    "port": int(port),
                    "date": pd.Timestamp(date),
                    "ret": _weighted_mean(group, "ret"),
                    "signallag": _weighted_mean(group, "signallag"),
                    "Nlong": int(len(group)),
                    "Nshort": 0,
                }
            )
        port = pd.DataFrame(grouped_rows)
        if port.empty:
            return pd.DataFrame(columns=["signalname", "port", "date", "ret", "signallag", "Nlong", "Nshort"])

        longportname = list(longportname) if longportname is not None else ["max"]
        shortportname = list(shortportname) if shortportname is not None else ["min"]
        if longportname[0] == "max":
            longportname = [int(port["port"].max())]
        if shortportname[0] == "min":
            shortportname = [int(port["port"].min())]

        long = (
            port.loc[port["port"].isin(longportname)]
            .groupby("date", as_index=False)
            .agg(retL=("ret", "mean"), Nlong=("Nlong", "sum"))
        )
        short = (
            port.loc[port["port"].isin(shortportname)]
            .groupby("date", as_index=False)
            .agg(retS=("ret", lambda s: -float(np.mean(s))), Nshort=("Nlong", "sum"))
        )
        longshort = long.merge(short, on="date", how="inner")
        longshort = longshort.assign(
            ret=longshort["retL"] + longshort["retS"],
            port="LS",
            signallag=np.nan,
        )[["port", "date", "ret", "signallag", "Nlong", "Nshort"]]

        port["port"] = port["port"].map(lambda value: f"{int(value):02d}")
        port = pd.concat([port, longshort], ignore_index=True, sort=False)
        port.insert(0, "signalname", signalname)
        port = port.sort_values(["signalname", "port", "date"]).reset_index(drop=True)
        return port

    def summarize_portfolios(
        self,
        portret: pd.DataFrame,
        *,
        groupby: Iterable[str] = ("signalname", "samptype", "port"),
        n_stocks_min: int = 20,
    ) -> pd.DataFrame:
        """Python port of the upstream monthly summary table."""

        temp = portret.merge(
            self.signal_doc.loc[:, ["signalname", "SampleStartYear", "SampleEndYear", "Year"]],
            on="signalname",
            how="left",
        )
        year = pd.to_datetime(temp["date"]).dt.year
        temp["samptype"] = np.select(
            [
                (year >= temp["SampleStartYear"]) & (year <= temp["SampleEndYear"]),
                (year > temp["SampleEndYear"]) & (year <= temp["Year"]),
                year > temp["Year"],
            ],
            ["insamp", "between", "postpub"],
            default="",
        )
        temp["Ncheck"] = np.where(temp["port"] != "LS", temp["Nlong"], np.minimum(temp["Nlong"], temp["Nshort"]))
        temp = temp.loc[temp["Ncheck"] >= n_stocks_min].copy()
        if temp.empty:
            return pd.DataFrame(
                columns=[*groupby, "tstat", "rbar", "vol", "T", "Nlong", "Nshort", "signallag"]
            )

        groupby = list(groupby)
        grouped = temp.groupby(groupby, dropna=False)

        def _summary(group: pd.DataFrame) -> pd.Series:
            ret = group["ret"].astype(float)
            tstat = np.nan
            if len(ret) > 1 and ret.std(ddof=1) not in (0, np.nan):
                std = ret.std(ddof=1)
                if pd.notna(std) and std != 0:
                    tstat = round(ret.mean() / std * np.sqrt(len(ret)), 2)
            return pd.Series(
                {
                    "tstat": tstat,
                    "rbar": round(float(ret.mean()), 2),
                    "vol": round(float(ret.std(ddof=1)), 2) if len(ret) > 1 else np.nan,
                    "T": int(len(ret)),
                    "Nlong": round(float(group["Nlong"].mean()), 1),
                    "Nshort": round(float(group["Nshort"].mean()), 1),
                    "signallag": round(float(group["signallag"].mean()), 3)
                    if group["signallag"].notna().any()
                    else np.nan,
                }
            )

        result = grouped.apply(_summary, include_groups=False).reset_index()
        return result.sort_values(groupby).reset_index(drop=True)

    @staticmethod
    def long_short_wide(portret: pd.DataFrame) -> pd.DataFrame:
        """Create the LS wide return matrix used by the upstream predictor exhibits."""

        ls = portret.loc[portret["port"] == "LS", ["date", "signalname", "ret"]].copy()
        if ls.empty:
            return pd.DataFrame(columns=["date"])
        return (
            ls.pivot(index="date", columns="signalname", values="ret")
            .reset_index()
            .sort_values("date")
            .reset_index(drop=True)
        )

    def build_predictor_report(
        self,
        *,
        signal_frames: Mapping[str, pd.DataFrame] | None = None,
        quickrun: Iterable[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Reproduce the main outputs from 20_PredictorPorts.R."""

        ports = self.build_predictor_portfolios(signal_frames=signal_frames, quickrun=quickrun)
        return {
            "PredictorPortsFull": ports,
            "PredictorLSretWide": self.long_short_wide(ports),
            "PredictorSummaryFull": self.summarize_portfolios(ports, n_stocks_min=1),
            "PredictorSummaryLSInSample": self.summarize_portfolios(ports, n_stocks_min=1).query(
                "samptype == 'insamp' and port == 'LS'"
            ),
        }

    def build_placebo_report(
        self,
        *,
        signal_frames: Mapping[str, pd.DataFrame] | None = None,
        quickrun: Iterable[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Reproduce the main outputs from 40_PlaceboPorts.R."""

        ports = self.build_placebo_portfolios(signal_frames=signal_frames, quickrun=quickrun)
        summary = self.summarize_portfolios(ports, n_stocks_min=20)
        return {
            "PlaceboPortsFull": ports,
            "PlaceboSummaryFull": summary,
            "PlaceboSummaryLSInSample": summary.query("samptype == 'insamp' and port == 'LS'"),
        }

    def build_strategy_portfolios(
        self,
        strategylist: pd.DataFrame,
        *,
        signal_frames: Mapping[str, pd.DataFrame] | None = None,
        save_port_csv: bool = False,
        save_port_path: str | Path | None = None,
        save_port_n_min: int = 1,
        passive_gain: bool = False,
    ) -> pd.DataFrame:
        """Loop over a strategy table and build portfolios for each documented signal."""

        all_ports: list[pd.DataFrame] = []
        output_dir = Path(save_port_path) if save_port_path is not None else None
        if output_dir is not None and save_port_csv:
            output_dir.mkdir(parents=True, exist_ok=True)

        for _, row in strategylist.iterrows():
            signalname = row["signalname"]
            self._log(f"building {signalname}")
            tempport = self.build_signal_portfolio(
                signalname,
                signal_frame=None if signal_frames is None else signal_frames.get(signalname),
                cat_form=row.get("Cat.Form"),
                q_cut=row.get("q_cut"),
                sweight=row.get("sweight"),
                sign=row.get("Sign"),
                startmonth=row.get("startmonth"),
                portperiod=row.get("portperiod"),
                q_filt=row.get("q_filt"),
                filterstr=row.get("filterstr"),
                passive_gain=passive_gain,
            )
            all_ports.append(tempport)

            if save_port_csv and output_dir is not None and not tempport.empty:
                tempwide = (
                    tempport.loc[tempport["Nlong"] >= save_port_n_min, ["port", "date", "ret"]]
                    .pivot(index="date", columns="port", values="ret")
                    .reset_index()
                )
                tempwide.columns = ["date", *[f"port{col}" for col in tempwide.columns[1:]]]
                tempwide.to_csv(output_dir / f"{signalname}_ret.csv", index=False)

        if not all_ports:
            return pd.DataFrame(columns=["signalname", "port", "date", "ret", "signallag", "Nlong", "Nshort"])
        return pd.concat(all_ports, ignore_index=True)

    def build_predictor_portfolios(
        self,
        *,
        signal_frames: Mapping[str, pd.DataFrame] | None = None,
        quickrun: Iterable[str] | None = None,
        save_port_csv: bool = False,
        save_port_path: str | Path | None = None,
        save_port_n_min: int = 1,
    ) -> pd.DataFrame:
        """Build the baseline monthly portfolios for documented predictors."""

        strategylist = self.strategy_subset(cat_signal="Predictor", quickrun=quickrun)
        return self.build_strategy_portfolios(
            strategylist,
            signal_frames=signal_frames,
            save_port_csv=save_port_csv,
            save_port_path=save_port_path,
            save_port_n_min=save_port_n_min,
        )

    def build_placebo_portfolios(
        self,
        *,
        signal_frames: Mapping[str, pd.DataFrame] | None = None,
        quickrun: Iterable[str] | None = None,
        save_port_csv: bool = False,
        save_port_path: str | Path | None = None,
        save_port_n_min: int = 1,
    ) -> pd.DataFrame:
        """Build the baseline monthly portfolios for documented placebos."""

        strategylist = self.strategy_subset(cat_signal="Placebo", quickrun=quickrun)
        return self.build_strategy_portfolios(
            strategylist,
            signal_frames=signal_frames,
            save_port_csv=save_port_csv,
            save_port_path=save_port_path,
            save_port_n_min=save_port_n_min,
        )

    def build_alternative_portfolios(
        self,
        *,
        signal_frames: Mapping[str, pd.DataFrame] | None = None,
        quickrun: Iterable[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Python port of 30_PredictorAltPorts.R."""

        strategylist0 = self.strategy_subset(cat_signal="Predictor", quickrun=quickrun)
        strategylistcts = strategylist0.loc[strategylist0["Cat.Form"] == "continuous"].copy()
        outputs: dict[str, pd.DataFrame] = {}

        for hold_period in (1, 3, 6, 12):
            outputs[f"PredictorAltPorts_HoldPer_{hold_period}.csv"] = self.build_strategy_portfolios(
                strategylist0.assign(portperiod=hold_period),
                signal_frames=signal_frames,
            )

        outputs["PredictorAltPorts_LiqScreen_ME_gt_NYSE20pct.csv"] = self.build_strategy_portfolios(
            strategylist0.assign(filterstr="me > me_nyse20"),
            signal_frames=signal_frames,
        )
        outputs["PredictorAltPorts_LiqScreen_Price_gt_5.csv"] = self.build_strategy_portfolios(
            strategylist0.assign(filterstr="abs(prc) > 5"),
            signal_frames=signal_frames,
        )
        outputs["PredictorAltPorts_LiqScreen_NYSEonly.csv"] = self.build_strategy_portfolios(
            strategylist0.assign(filterstr="exchcd == 1"),
            signal_frames=signal_frames,
        )
        outputs["PredictorAltPorts_LiqScreen_VWforce.csv"] = self.build_strategy_portfolios(
            strategylist0.assign(sweight="VW"),
            signal_frames=signal_frames,
        )

        outputs["PredictorAltPorts_Deciles.csv"] = self.build_strategy_portfolios(
            strategylistcts.assign(q_cut=0.1),
            signal_frames=signal_frames,
        )
        outputs["PredictorAltPorts_DecilesVW.csv"] = self.build_strategy_portfolios(
            strategylistcts.assign(q_cut=0.1, sweight="VW"),
            signal_frames=signal_frames,
        )
        outputs["PredictorAltPorts_DecilesEW.csv"] = self.build_strategy_portfolios(
            strategylistcts.assign(q_cut=0.1, sweight="EW"),
            signal_frames=signal_frames,
        )
        outputs["PredictorAltPorts_Quintiles.csv"] = self.build_strategy_portfolios(
            strategylistcts.assign(q_cut=0.2),
            signal_frames=signal_frames,
        )
        outputs["PredictorAltPorts_QuintilesVW.csv"] = self.build_strategy_portfolios(
            strategylistcts.assign(q_cut=0.2, sweight="VW"),
            signal_frames=signal_frames,
        )
        outputs["PredictorAltPorts_QuintilesEW.csv"] = self.build_strategy_portfolios(
            strategylistcts.assign(q_cut=0.2, sweight="EW"),
            signal_frames=signal_frames,
        )
        return outputs

    def build_ff93_signal_portfolio(
        self,
        signalname: str,
        *,
        signal_frame: pd.DataFrame | None = None,
        sign: float | None = None,
    ) -> pd.DataFrame:
        """Python port of signalname_to_2x3 from 32_Predictor2x3Ports.R."""

        strategy = self._strategy_row(signalname)
        sign = strategy["Sign"] if sign is None or pd.isna(sign) else sign
        signal = self.import_signal(signalname, sign=float(sign), signal_frame=signal_frame)
        signaljune = signal.loc[signal["yyyymm"] % 100 == 6].copy()

        if not {"exchcd", "shrcd", "me"} <= set(signaljune.columns):
            raise ValueError("FF93-style portfolios require exchcd, shrcd, and me in crspinfo")
        if "melag" not in self.crspret.columns:
            raise ValueError("FF93-style portfolios require melag in crspret")

        nyse = signaljune.loc[(signaljune["exchcd"] == 1) & (signaljune["shrcd"].isin([10, 11]))]
        nysebreaks = (
            nyse.groupby("yyyymm")
            .agg(
                qsignal_l=("signal", lambda s: s.quantile(0.3)),
                qsignal_h=("signal", lambda s: s.quantile(0.7)),
                qme_mid=("me", lambda s: s.quantile(0.5)),
            )
            .reset_index()
        )

        port6 = (
            signaljune.loc[
                signaljune["exchcd"].isin([1, 2, 3]) & signaljune["shrcd"].isin([10, 11]),
                ["permno", "yyyymm", "signal", "exchcd", "shrcd", "me"],
            ]
            .merge(nysebreaks, on="yyyymm", how="left")
        )
        port6["q_signal"] = np.select(
            [
                port6["signal"] <= port6["qsignal_l"],
                port6["signal"] <= port6["qsignal_h"],
                port6["signal"] > port6["qsignal_h"],
            ],
            ["L", "M", "H"],
            default="",
        )
        port6["q_me"] = np.where(port6["me"] <= port6["qme_mid"], "S", "B")
        port6["port6"] = port6["q_me"] + port6["q_signal"]
        port6 = port6.loc[:, ["permno", "yyyymm", "port6", "signal"]]

        merged = self.crspret.loc[:, ["permno", "date", "yyyymm", "ret", "melag"]].merge(
            port6, on=["permno", "yyyymm"], how="left"
        )
        merged = merged.sort_values(["permno", "date"]).copy()
        merged["port6"] = merged.groupby("permno")["port6"].ffill()
        merged["signal"] = merged.groupby("permno")["signal"].ffill()
        merged["port6_lag"] = merged.groupby("permno")["port6"].shift(1)
        merged["signal_lag"] = merged.groupby("permno")["signal"].shift(1)
        merged = merged.loc[merged["melag"].notna()].copy()

        rows: list[dict[str, object]] = []
        for (port, date), group in merged.groupby(["port6_lag", "date"], dropna=True, sort=True):
            if port not in {"SL", "SM", "SH", "BL", "BM", "BH"}:
                continue
            rows.append(
                {
                    "signalname": signalname,
                    "port": port,
                    "date": pd.Timestamp(date),
                    "ret": float(np.average(group["ret"], weights=group["melag"])),
                    "signallag": float(np.average(group["signal_lag"], weights=group["melag"])),
                    "Nlong": int(len(group)),
                    "Nshort": 0,
                }
            )
        port6ret = pd.DataFrame(rows)
        if port6ret.empty:
            return pd.DataFrame(columns=["signalname", "port", "date", "ret", "signallag", "Nlong", "Nshort"])

        ret_wide = port6ret.pivot(index="date", columns="port", values="ret").reset_index()
        n_wide = port6ret.pivot(index="date", columns="port", values="Nlong").reset_index()
        longshort = ret_wide.merge(n_wide, on="date", suffixes=("_ret", "_n"))
        longshort = pd.DataFrame(
            {
                "signalname": signalname,
                "port": "LS",
                "date": longshort["date"],
                "ret": 0.5 * (longshort["SH_ret"] + longshort["BH_ret"]) - 0.5 * (longshort["SL_ret"] + longshort["BL_ret"]),
                "signallag": np.nan,
                "Nlong": longshort["SH_n"] + longshort["BH_n"],
                "Nshort": longshort["SL_n"] + longshort["BL_n"],
            }
        )

        port = pd.concat([port6ret, longshort], ignore_index=True, sort=False)
        port["port"] = pd.Categorical(port["port"], categories=["SL", "SM", "SH", "BL", "BM", "BH", "LS"], ordered=True)
        return port.sort_values(["port", "date"]).reset_index(drop=True)

    def build_ff93_portfolios(
        self,
        *,
        signal_frames: Mapping[str, pd.DataFrame] | None = None,
        quickrun: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Python port of 32_Predictor2x3Ports.R."""

        strategylist = self.strategy_subset(cat_signal="Predictor", continuous_only=True, quickrun=quickrun)
        ports = []
        for _, row in strategylist.iterrows():
            signalname = row["signalname"]
            ports.append(
                self.build_ff93_signal_portfolio(
                    signalname,
                    signal_frame=None if signal_frames is None else signal_frames.get(signalname),
                    sign=row.get("Sign"),
                )
            )
        if not ports:
            return pd.DataFrame(columns=["signalname", "port", "date", "ret", "signallag", "Nlong", "Nshort"])
        return pd.concat(ports, ignore_index=True)

    def build_daily_portfolio_sets(
        self,
        *,
        signal_frames: Mapping[str, pd.DataFrame] | None = None,
        quickrun: Iterable[str] | None = None,
        save_root: str | Path | None = None,
        n_stocks_min: int = 1,
    ) -> dict[str, pd.DataFrame]:
        """Python port of 50_DailyPredictorPorts.R for baseline save sets."""

        strategylist0 = self.strategy_subset(cat_signal="Predictor", quickrun=quickrun)
        strategylistcts = strategylist0.loc[strategylist0["Cat.Form"] == "continuous"].copy()
        root = Path(save_root) if save_root is not None else None

        def _path(name: str) -> Path | None:
            return None if root is None else root / name

        results = {
            "Predictor": self.build_strategy_portfolios(
                strategylist0,
                signal_frames=signal_frames,
                save_port_csv=root is not None,
                save_port_path=_path("Predictor"),
                save_port_n_min=n_stocks_min,
                passive_gain=True,
            ),
            "PredictorVW": self.build_strategy_portfolios(
                strategylist0.assign(sweight="VW"),
                signal_frames=signal_frames,
                save_port_csv=root is not None,
                save_port_path=_path("PredictorVW"),
                save_port_n_min=n_stocks_min,
                passive_gain=True,
            ),
            "CtsPredictorDecile": self.build_strategy_portfolios(
                strategylistcts.assign(q_cut=0.1),
                signal_frames=signal_frames,
                save_port_csv=root is not None,
                save_port_path=_path("CtsPredictorDecile"),
                save_port_n_min=n_stocks_min,
                passive_gain=True,
            ),
            "CtsPredictorDecileVW": self.build_strategy_portfolios(
                strategylistcts.assign(q_cut=0.1, sweight="VW"),
                signal_frames=signal_frames,
                save_port_csv=root is not None,
                save_port_path=_path("CtsPredictorDecileVW"),
                save_port_n_min=n_stocks_min,
                passive_gain=True,
            ),
            "CtsPredictorQuintile": self.build_strategy_portfolios(
                strategylistcts.assign(q_cut=0.2),
                signal_frames=signal_frames,
                save_port_csv=root is not None,
                save_port_path=_path("CtsPredictorQuintile"),
                save_port_n_min=n_stocks_min,
                passive_gain=True,
            ),
            "CtsPredictorQuintileVW": self.build_strategy_portfolios(
                strategylistcts.assign(q_cut=0.2, sweight="VW"),
                signal_frames=signal_frames,
                save_port_csv=root is not None,
                save_port_path=_path("CtsPredictorQuintileVW"),
                save_port_n_min=n_stocks_min,
                passive_gain=True,
            ),
        }
        return results

    @staticmethod
    def daily_wide_to_long(frame: pd.DataFrame) -> pd.DataFrame:
        """Convert saved daily wide returns back to long form."""

        temp = frame.copy()
        if "date" not in temp.columns:
            raise ValueError("daily portfolio csv needs a date column")
        port_cols = [col for col in temp.columns if col != "date"]
        long = temp.melt(id_vars="date", value_vars=port_cols, var_name="port", value_name="ret")
        long["port"] = long["port"].astype(str).str.replace("^port", "", regex=True)
        long["date"] = pd.to_datetime(long["date"])
        return long.loc[long["ret"].notna()].reset_index(drop=True)

    @staticmethod
    def summarize_daily_portfolios(portfolio_sets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        """Summarize daily long-return outputs like checkdir() in the upstream R code."""

        rows: list[dict[str, object]] = []
        for implementation, long in portfolio_sets.items():
            if long.empty:
                continue
            stats = (
                long.groupby(["signalname", "port"])["ret"]
                .agg(["count", "mean"])
                .reset_index()
            )
            for _, stat in stats.iterrows():
                rows.append(
                    {
                        "implementation": implementation,
                        "signalname": stat["signalname"],
                        "port": stat["port"],
                        "nobs_years": float(stat["count"]) / 250.0,
                        "rbar_monthly": float(stat["mean"]) * 20.0,
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def compare_daily_monthly_timing(
        daily_long: pd.DataFrame,
        monthly_long: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate daily portfolios to month-end and compare with monthly portfolios."""

        daily = daily_long.copy()
        daily["date"] = pd.to_datetime(daily["date"])
        daily["datem"] = daily["date"].dt.to_period("M").dt.to_timestamp("M")
        daily_monthly = (
            daily.groupby(["signalname", "port", "datem"])["ret"]
            .apply(lambda s: float(np.prod(1 + s) - 1))
            .reset_index(name="retm_agg")
        )

        monthly = monthly_long.copy()
        monthly["date"] = pd.to_datetime(monthly["date"])
        monthly = monthly.rename(columns={"date": "datem", "ret": "retm"})

        rows: list[dict[str, object]] = []
        merged = monthly.merge(daily_monthly, on=["signalname", "port", "datem"], how="left")
        for (signalname, port), group in merged.groupby(["signalname", "port"]):
            group = group.dropna(subset=["retm", "retm_agg"])
            if len(group) <= 10:
                continue
            x = group["retm_agg"].to_numpy(dtype=float)
            y = group["retm"].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            yhat = intercept + slope * x
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            rsq = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
            rows.append(
                {
                    "signalname": signalname,
                    "port": port,
                    "intercept": float(intercept),
                    "slope": float(slope),
                    "rsq": float(rsq),
                }
            )
        return pd.DataFrame(rows)


__all__ = [
    "DEFAULT_SIGNAL_DOC_PATH",
    "TRADITIONAL_FACTORS_DIR",
    "TraditionalPortfolioEngine",
    "TraditionalPortfolioPaths",
    "check_signal_csvs",
    "create_crsp_predictors",
    "load_signal_doc",
]
