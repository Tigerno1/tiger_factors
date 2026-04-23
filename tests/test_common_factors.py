from __future__ import annotations

import pandas as pd

from tiger_factors.factor_algorithm import common_factor_group_frame
from tiger_factors.factor_algorithm import common_factor_group_markdown
from tiger_factors.factor_algorithm import common_factor_display_names
from tiger_factors.factor_algorithm import common_factor_family_markdown
from tiger_factors.factor_algorithm import common_factor_family_summary
from tiger_factors.factor_algorithm import common_factor_group_index
from tiger_factors.factor_algorithm import common_factor_group_names
from tiger_factors.factor_algorithm import common_factor_spec
from tiger_factors.factor_algorithm import find_common_factor_group
from tiger_factors.factor_algorithm import available_common_factors
from tiger_factors.factor_algorithm import run_common_factor
from tiger_factors.factor_algorithm import run_value_quality_long_short_backtest
from tiger_factors.factor_algorithm import run_value_quality_combo
from tiger_factors.factor_algorithm import run_value_quality_combo_from_columns


def test_common_factor_catalog_exposes_common_labels() -> None:
    names = available_common_factors()
    assert "BM" in names
    assert "FSCORE" in names
    assert "BMFSCORE" in names
    assert "PEAD" in names
    assert "IVOL" in names
    assert "SMB" in names
    assert "HML" in names
    assert "QMJ" in names
    assert "INFLC" in names


def test_common_factor_specs_resolve_to_expected_sources() -> None:
    assert common_factor_spec("BM").source_signal == "BM"
    assert common_factor_spec("BM").kind == "direct"
    assert common_factor_spec("FSCORE").source_signal == "PS"
    assert common_factor_spec("FSCORE").kind == "direct"
    assert common_factor_spec("PS").name == "FSCORE"
    assert common_factor_spec("BMFSCORE").kind == "composite"
    assert common_factor_spec("PEAD").source_signal == "AnnouncementReturn"
    assert common_factor_spec("IVOL").source_signal == "RIVolSpread"
    assert common_factor_spec("LIQ").source_signal == "Illiquidity"
    assert common_factor_spec("SMB").source_signal == "Size"
    assert common_factor_spec("SMB").sign == -1.0
    assert common_factor_spec("CMA").source_signal == "AssetGrowth"
    assert common_factor_spec("CMA").sign == -1.0
    assert common_factor_spec("RMW").source_signal == "OperProf"
    assert common_factor_spec("BAB").source_signal == "BetaFP"
    assert common_factor_spec("BAB").sign == -1.0
    assert common_factor_spec("VIX").source_signal == "betaVIX"
    assert common_factor_spec("SMBs").name == "SMB"
    assert common_factor_spec("HMLs").source_signal == "BMdec"
    assert common_factor_spec("CMAs").name == "CMA"
    assert common_factor_spec("RMWs").name == "RMW"
    assert common_factor_spec("MKTBs").name == "MKTB"
    assert common_factor_spec("STREV").name == "STREVB"
    assert common_factor_spec("B/M").name == "BM"
    assert common_factor_spec("F-Score").name == "FSCORE"
    assert common_factor_spec("B/M + F-Score").name == "BMFSCORE"
    assert common_factor_spec("QMJ").kind == "composite"
    assert common_factor_spec("INFLC").kind == "macro"
    assert common_factor_spec("INFLV").kind == "macro"
    assert common_factor_spec("EPU").kind == "macro"
    assert common_factor_spec("TERM").kind == "macro"
    display_names = common_factor_display_names()
    assert display_names["SMB"] == "SMBs"
    assert display_names["HML"] == "HMLs"
    assert display_names["CMA"] == "CMAs"
    assert display_names["RMW"] == "RMWs"
    assert display_names["MKTB"] == "MKTBs"
    assert display_names["STREVB"] == "STREV"
    assert display_names["BM"] == "B/M"
    assert display_names["FSCORE"] == "F-Score"
    assert display_names["BMFSCORE"] == "B/M + F-Score"


def test_common_factor_group_frame_and_index_are_consistent() -> None:
    frame = common_factor_group_frame()
    index = common_factor_group_index()
    summary = common_factor_family_summary()

    assert not frame.empty
    assert not summary.empty
    assert {"name", "family", "kind", "source_signal", "description"}.issubset(frame.columns)
    assert {"family", "count", "names", "display_names"}.issubset(summary.columns)
    markdown = common_factor_family_markdown()
    assert isinstance(markdown, str)
    assert "family" in markdown.lower()
    catalog_markdown = common_factor_group_markdown()
    assert isinstance(catalog_markdown, str)
    assert "source_signal" in catalog_markdown.lower()
    assert "quality" in index
    assert "value" in index
    assert "risk" in index
    assert "SMB" in index["size"]
    assert "QMJ" in index["quality"]
    assert "size" in common_factor_group_names()
    assert find_common_factor_group("SMB") == "size"
    assert find_common_factor_group("INFLC") == "macro"
    assert find_common_factor_group("VIX") == "risk"
    assert find_common_factor_group("EPU") == "macro"


def test_common_factor_inflation_aligns_macro_series() -> None:
    stock = pd.DataFrame(
        {
            "date_": pd.date_range("2020-01-31", periods=13, freq="M"),
            "code": ["A"] * 13,
        }
    )
    cpi = pd.DataFrame(
        {
            "date_": pd.date_range("2019-01-31", periods=13, freq="M"),
            "value": [100.0] * 12 + [110.0],
        }
    )

    result = run_common_factor("INFLC", stock, datasets={"cpi": cpi}, return_frame=True)

    assert isinstance(result, pd.DataFrame)
    assert {"date_", "code", "INFLC"}.issubset(result.columns)
    last_value = result.loc[result["date_"] == pd.Timestamp("2021-01-31"), "INFLC"].iloc[0]
    assert abs(float(last_value) - 10.0) < 1e-9


def test_common_factor_inflation_volatility_uses_macro_series() -> None:
    cpi = pd.DataFrame(
        {
            "date_": pd.date_range("2018-01-31", periods=24, freq="M"),
            "value": [100.0 + float(i) for i in range(24)],
        }
    )

    result = run_common_factor("INFLV", None, datasets={"cpi": cpi}, return_frame=True)

    assert isinstance(result, pd.DataFrame)
    assert {"date_", "code", "INFLV"}.issubset(result.columns)
    assert result["INFLV"].notna().any()


def test_common_factor_epu_and_term_use_macro_series() -> None:
    epu = pd.DataFrame(
        {
            "date_": pd.date_range("2019-01-31", periods=6, freq="M"),
            "value": [100.0, 102.0, 101.0, 105.0, 110.0, 115.0],
        }
    )
    term = pd.DataFrame(
        {
            "date_": pd.date_range("2019-01-31", periods=6, freq="M"),
            "value": [1.0, 1.1, 1.2, 1.15, 1.25, 1.3],
        }
    )

    epu_result = run_common_factor("EPU", None, datasets={"epu": epu}, return_frame=True)
    term_result = run_common_factor("TERM", None, datasets={"term_spread": term}, return_frame=True)

    assert isinstance(epu_result, pd.DataFrame)
    assert isinstance(term_result, pd.DataFrame)
    assert {"date_", "code", "EPU"}.issubset(epu_result.columns)
    assert {"date_", "code", "TERM"}.issubset(term_result.columns)
    assert abs(float(epu_result.iloc[-1]["EPU"]) - 115.0) < 1e-9
    assert abs(float(term_result.iloc[-1]["TERM"]) - 1.3) < 1e-9


def test_common_factor_value_quality_combo_builds_two_by_two_screen(monkeypatch) -> None:
    from tiger_factors.factor_algorithm.traditional_factors import common_factors as module

    def fake_run_original_factor(signal_name, data, **kwargs):
        frame = data.copy()
        values = {
            "BM": [0.9, 0.1, 0.2, 0.8],
            "PS": [9.0, 1.0, 2.0, 8.0],
        }[signal_name]
        result = pd.Series(values[: len(frame)], index=frame.index, name=signal_name)
        if kwargs.get("return_frame"):
            return pd.DataFrame(
                {
                    "date_": frame["date_"].to_numpy(),
                    "code": frame["code"].to_numpy(),
                    signal_name: result.to_numpy(),
                }
            )
        return result

    monkeypatch.setattr(module, "run_original_factor", fake_run_original_factor)

    frame = pd.DataFrame(
        {
            "date_": pd.to_datetime(
                ["2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29"]
            ),
            "code": ["A", "B", "C", "D"],
        }
    )

    combo = run_value_quality_combo(frame)

    assert {"date_", "code", "BM", "FSCORE", "BMFSCORE", "BM_rank", "FSCORE_rank", "value_quality_bucket"}.issubset(
        combo.columns
    )
    assert combo.loc[combo["code"] == "A", "value_quality_bucket"].iloc[0] == "HH"
    assert combo.loc[combo["code"] == "B", "value_quality_bucket"].iloc[0] == "LL"
    assert combo.loc[combo["code"] == "C", "value_quality_bucket"].iloc[0] == "LL"
    assert combo.loc[combo["code"] == "D", "value_quality_bucket"].iloc[0] == "HH"
    assert combo.loc[combo["code"] == "A", "BMFSCORE"].iloc[0] > combo.loc[combo["code"] == "B", "BMFSCORE"].iloc[0]


def test_common_factor_value_quality_combo_from_columns() -> None:
    frame = pd.DataFrame(
        {
            "date_": pd.to_datetime(
                ["2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29"]
            ),
            "code": ["A", "B", "C", "D"],
            "BM": [0.9, 0.1, 0.2, 0.8],
            "FSCORE": [9.0, 1.0, 2.0, 8.0],
        }
    )

    combo = run_value_quality_combo_from_columns(frame, bm_column="BM", fscore_column="FSCORE")

    assert {"BM", "FSCORE", "BMFSCORE", "value_quality_bucket_score"}.issubset(combo.columns)
    assert combo.loc[combo["code"] == "A", "value_quality_bucket"].iloc[0] == "HH"
    assert combo.loc[combo["code"] == "B", "value_quality_bucket"].iloc[0] == "LL"
    assert combo.loc[combo["code"] == "A", "BMFSCORE"].iloc[0] > combo.loc[combo["code"] == "B", "BMFSCORE"].iloc[0]


def test_common_factor_value_quality_long_short_backtest_runs(monkeypatch) -> None:
    from tiger_factors.factor_algorithm.traditional_factors import common_factors as module

    def fake_run_original_factor(signal_name, data, **kwargs):
        frame = data.copy()
        if signal_name == "BM":
            values = pd.Series([4.0, 1.0, 3.5, 0.8], index=frame.index, name=signal_name)
        elif signal_name == "PS":
            values = pd.Series([8.0, 2.0, 7.5, 1.0], index=frame.index, name=signal_name)
        else:
            raise KeyError(signal_name)
        if kwargs.get("return_frame"):
            return pd.DataFrame(
                {
                    "date_": frame["date_"].to_numpy(),
                    "code": frame["code"].to_numpy(),
                    signal_name: values.to_numpy(),
                }
            )
        return values

    monkeypatch.setattr(module, "run_original_factor", fake_run_original_factor)

    frame = pd.DataFrame(
        {
            "date_": pd.to_datetime(
                ["2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29"]
            ),
            "code": ["A", "B", "C", "D"],
        }
    )
    close_panel = pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 103.0],
            "B": [100.0, 99.5, 99.0, 98.5],
            "C": [100.0, 100.5, 101.0, 101.5],
            "D": [100.0, 99.0, 98.0, 97.0],
        },
        index=pd.bdate_range("2020-01-31", periods=4),
    )
    close_panel.index.name = "date_"

    result = run_value_quality_long_short_backtest(frame, close_panel, long_pct=0.5)

    assert "backtest" in result
    assert "stats" in result
    assert isinstance(result["backtest"], pd.DataFrame)
    assert not result["backtest"].empty
    assert "portfolio" in result["backtest"].columns
    assert "benchmark" in result["backtest"].columns
    assert result["combo_frame"]["value_quality_bucket"].isin({"HH", "HL", "LH", "LL"}).all()
