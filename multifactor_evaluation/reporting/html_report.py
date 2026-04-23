from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Mapping
import webbrowser

import pandas as pd


# Shared report composition primitives used by all multifactor HTML renderers.
@dataclass(frozen=True)
class ReportSectionSpec:
    title: str
    names: list[str] | None = None
    intro: str | None = None
    kind: str = "tables"


@dataclass(frozen=True)
class SummaryFigureSpec:
    key: str
    stem_suffix: str
    container_id: str | None = None


SUMMARY_FIGURE_SPECS = [
    SummaryFigureSpec(key="snapshot", stem_suffix="snapshot"),
    SummaryFigureSpec(key="earnings", stem_suffix="earnings"),
    SummaryFigureSpec(key="returns", stem_suffix="equity_curve"),
    SummaryFigureSpec(key="log_returns", stem_suffix="log_returns", container_id="log_returns"),
    SummaryFigureSpec(key="vol_returns", stem_suffix="vol_returns", container_id="vol_returns"),
    SummaryFigureSpec(key="eoy_returns", stem_suffix="annual_returns", container_id="eoy_returns"),
    SummaryFigureSpec(key="monthly_dist", stem_suffix="return_histogram", container_id="monthly_dist"),
    SummaryFigureSpec(key="daily_returns", stem_suffix="daily_returns", container_id="daily_returns"),
    SummaryFigureSpec(key="rolling_beta", stem_suffix="rolling_beta", container_id="rolling_beta"),
    SummaryFigureSpec(key="rolling_vol", stem_suffix="rolling_vol", container_id="rolling_vol"),
    SummaryFigureSpec(key="rolling_sharpe", stem_suffix="rolling_sharpe", container_id="rolling_sharpe"),
    SummaryFigureSpec(key="rolling_sortino", stem_suffix="rolling_sortino", container_id="rolling_sortino"),
    SummaryFigureSpec(key="dd_periods", stem_suffix="drawdown_periods", container_id="dd_periods"),
    SummaryFigureSpec(key="dd_plot", stem_suffix="drawdown", container_id="dd_plot"),
    SummaryFigureSpec(key="monthly_heatmap", stem_suffix="monthly_heatmap", container_id="monthly_heatmap"),
    SummaryFigureSpec(key="returns_dist", stem_suffix="returns_dist", container_id="returns_dist"),
]


def _table_html(table: pd.DataFrame) -> str:
    return _table_html_with_class(table, classes="table")


def _table_html_with_class(table: pd.DataFrame, *, classes: str) -> str:
    if table.empty:
        return "<p><em>No table data available.</em></p>"
    return table.to_html(border=0, classes=classes, na_rep="", escape=False)


def _summary_metric_cards(table: pd.DataFrame | None, *, limit: int = 4) -> str:
    if table is None or table.empty:
        return ""
    frame = pd.DataFrame(table).reset_index()
    cards: list[str] = []
    for _, row in frame.head(limit).iterrows():
        label = escape(str(row.iloc[0]))
        raw_value = row.iloc[1] if len(row) > 1 else ""
        if pd.isna(raw_value):
            value = ""
        elif isinstance(raw_value, float):
            value = f"{raw_value:,.4f}"
        else:
            value = str(raw_value)
        cards.append(
            "<div class='index-card metric-card'>"
            f"<div class='label'>{label}</div>"
            f"<div class='value metric-value'>{escape(value)}</div>"
            "</div>"
        )
    if not cards:
        return ""
    return "<div class='index-grid metric-grid'>" + "".join(cards) + "</div>"


def _metric_cards(table: pd.DataFrame | None, *, limit: int = 4) -> str:
    return _summary_metric_cards(table, limit=limit)


def _slugify(value: str) -> str:
    slug = []
    for char in value.lower():
        if char.isalnum():
            slug.append(char)
        else:
            slug.append("-")
    cleaned = "".join(slug).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned or "section"


def _report_shell(title: str, subtitle: str | None, body: str, *, nav: str | None = None, width: int = 1100) -> str:
    subtitle_html = f"<p class='subtitle'>{escape(subtitle)}</p>" if subtitle else ""
    nav_html = f"<nav class='nav'>{nav}</nav>" if nav else ""
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{escape(title)}</title>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:24px;background:linear-gradient(180deg,#f7f8fb 0,#ffffff 180px);color:#111}"
        f".wrap{{max-width:{width}px;margin:0 auto}}"
        ".subtitle{color:#5f6b7a;margin:0 0 16px;font-size:14px;line-height:1.5}"
        ".nav{display:flex;flex-wrap:wrap;gap:8px 10px;padding:12px 14px;border:1px solid #e5e7eb;border-radius:14px;background:#ffffff;box-shadow:0 1px 2px rgba(15,23,42,0.04);margin:0 0 24px}"
        ".nav a{color:#0b5cab;text-decoration:none;font-size:13px;padding:4px 8px;border-radius:999px;background:#f8fbff;border:1px solid #dbe7f4}"
        ".nav a:hover{text-decoration:underline}"
        "table{border-collapse:separate;border-spacing:0;width:100%;margin:0 0 20px;border:1px solid #e5e7eb;border-radius:14px;overflow:hidden;background:#fff;box-shadow:0 1px 2px rgba(15,23,42,0.04)}"
        "th,td{border-bottom:1px solid #edf2f7;padding:8px 10px;font-size:13px;vertical-align:top}"
        "th{background:#f8fafc;text-align:left;color:#334155;font-weight:600}"
        "tr:last-child td{border-bottom:none}"
        "img{max-width:100%;height:auto;border:1px solid #e5e7eb;border-radius:12px;box-shadow:0 1px 2px rgba(15,23,42,0.04)}"
        ".block{margin:0 0 28px;padding:18px;border:1px solid #e5e7eb;border-radius:16px;background:#fff;box-shadow:0 1px 3px rgba(15,23,42,0.04)}"
        ".section{margin:0 0 24px}"
        "h1{font-weight:700;margin:0 0 8px;letter-spacing:-0.02em}h2{font-weight:600;margin:0 0 10px;letter-spacing:-0.01em}h3{font-weight:600;margin:16px 0 10px}"
        ".toplink{display:inline-block;margin-top:10px;font-size:12px;color:#0b5cab;text-decoration:none}"
        ".muted{color:#64748b}"
        ".module-head{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:12px;margin:0 0 12px}"
        ".module-meta{display:flex;flex-wrap:wrap;gap:8px}"
        ".chip{display:inline-flex;align-items:center;padding:4px 9px;border-radius:999px;border:1px solid #dbe7f4;background:#f8fbff;color:#1e3a5f;font-size:12px;font-weight:600}"
        ".index-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:0 0 20px}"
        ".index-card{padding:14px 16px;border:1px solid #e5e7eb;border-radius:14px;background:#fff;box-shadow:0 1px 2px rgba(15,23,42,0.04)}"
        ".index-card .label{font-size:12px;color:#64748b;margin-bottom:6px}"
        ".index-card .value{font-size:24px;font-weight:700;letter-spacing:-0.03em}"
        ".section-intro{margin:0 0 14px;color:#64748b;font-size:13px;line-height:1.5}"
        ".metric-card .value.metric-value{font-size:18px;letter-spacing:-0.01em}"
        ".gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}"
        ".figure-card{margin:0;padding:14px;border:1px solid #e5e7eb;border-radius:14px;background:#fff}"
        ".figure-card h3{margin:0 0 10px;font-size:14px}"
        "</style></head><body><div class='wrap'>"
        f"<h1>{escape(title)}</h1>"
        f"{subtitle_html}"
        f"{nav_html}"
        f"{body}"
        "</div></body></html>"
    )


def render_artifact_report(
    *,
    title: str,
    output_dir: Path,
    report_name: str,
    tables: Mapping[str, pd.DataFrame],
    figure_paths: list[Path],
    open_browser: bool = False,
    subtitle: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{report_name}.html"
    sections: list[str] = []
    nav_links: list[str] = []
    anchors: list[str] = []
    for name, table in tables.items():
        anchor = _slugify(name)
        nav_links.append(f"<a href='#{anchor}'>{escape(name)}</a>")
        anchors.append(
            f"<section class='block' id='{anchor}'><h2>{escape(name)}</h2>{_table_html(table)}<a class='toplink' href='#top'>Back to top</a></section>"
        )
    for path in figure_paths:
        if path.exists():
            try:
                rel = path.relative_to(output_dir).as_posix()
            except ValueError:
                rel = path.name
            anchor = _slugify(path.stem)
            nav_links.append(f"<a href='#{anchor}'>{escape(path.stem)}</a>")
            anchors.append(
                f"<section class='block' id='{anchor}'><h2>{escape(path.stem)}</h2><img src='{escape(rel)}' alt='{escape(path.stem)}'><a class='toplink' href='#top'>Back to top</a></section>"
            )
    body = "".join(anchors)
    nav = "".join(nav_links)
    html = _report_shell(title, subtitle, f"<div id='top'></div>{body}", nav=nav)
    html_path.write_text(html, encoding="utf-8")
    if open_browser:
        webbrowser.open(html_path.as_uri())
    return html_path


def _resolve_figure_ref(path: Path, output_dir: Path) -> str:
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        return path.name


def _find_figure_html(
    figure_paths: list[Path],
    *,
    stem_or_name: str,
    src_transform: Callable[[Path], str] | None = None,
) -> str:
    for path in figure_paths:
        if path.stem == stem_or_name or path.name == stem_or_name:
            src = path.name if src_transform is None else str(src_transform(path))
            return f"<img src='{escape(src)}' alt='{escape(path.stem)}' />"
    return ""


def _optional_titled_table_html(
    title: str,
    table: pd.DataFrame | None,
    *,
    classes: str,
) -> str:
    if table is None or table.empty:
        return ""
    return f"<h3>{escape(title)}</h3>{_table_html_with_class(table, classes=classes)}"


def _optional_panel_html(
    *,
    panel_id: str,
    title: str,
    body_html: str,
) -> str:
    if not body_html:
        return ""
    return f"<div id='{escape(panel_id)}'><h3>{escape(title)}</h3>{body_html}</div>"


def _summary_figure_strip(figure_paths: list[Path], report_name: str, specs: list[SummaryFigureSpec]) -> str:
    blocks: list[str] = []
    for spec in specs:
        figure_html = _find_figure_html(figure_paths, stem_or_name=f"{report_name}_{spec.stem_suffix}")
        if not figure_html:
            continue
        if spec.container_id:
            blocks.append(f"<div id='{escape(spec.container_id)}'>{figure_html}</div>")
        else:
            blocks.append(f"<div>{figure_html}</div>")
    return "".join(blocks)


def _render_table_section(
    *,
    title: str,
    tables: Mapping[str, pd.DataFrame],
    names: list[str],
    intro: str | None = None,
) -> str:
    blocks: list[str] = []
    for name in names:
        table = tables.get(name)
        if table is None:
            continue
        anchor = _slugify(name)
        blocks.append(f"<section id='{anchor}'><h3>{escape(name.replace('_', ' ').title())}</h3>{_table_html(table)}</section>")
    if not blocks:
        return ""
    intro_html = f"<p class='section-intro'>{escape(intro)}</p>" if intro else ""
    return f"<section class='block'><h2>{escape(title)}</h2>{intro_html}{''.join(blocks)}</section>"


def _render_figure_section(
    *,
    title: str,
    figure_paths: list[Path],
    output_dir: Path,
    intro: str | None = None,
) -> str:
    cards: list[str] = []
    for path in figure_paths:
        if not path.exists():
            continue
        rel = _resolve_figure_ref(path, output_dir)
        cards.append(
            "<figure class='figure-card'>"
            f"<h3>{escape(path.stem.replace('_', ' ').title())}</h3>"
            f"<img src='{escape(rel)}' alt='{escape(path.stem)}'>"
            "</figure>"
        )
    if not cards:
        return ""
    intro_html = f"<p class='section-intro'>{escape(intro)}</p>" if intro else ""
    return f"<section class='block'><h2>{escape(title)}</h2>{intro_html}<div class='gallery'>{''.join(cards)}</div></section>"


# Specialized artifact page builder shared by positions, trades, and portfolio.
def _render_specialized_report(
    *,
    title: str,
    output_dir: Path,
    report_name: str,
    tables: Mapping[str, pd.DataFrame],
    figure_paths: list[Path],
    subtitle: str | None,
    open_browser: bool,
    width: int,
    specs: list[ReportSectionSpec],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{report_name}.html"
    body_parts = [_metric_cards(tables.get("summary"))]
    for spec in specs:
        if spec.kind == "figures":
            body_parts.append(
                _render_figure_section(
                    title=spec.title,
                    figure_paths=figure_paths,
                    output_dir=output_dir,
                    intro=spec.intro,
                )
            )
            continue
        body_parts.append(
            _render_table_section(
                title=spec.title,
                tables=tables,
                names=spec.names or [],
                intro=spec.intro,
            )
        )
    html = _report_shell(title, subtitle, f"<div id='top'></div>{''.join(body_parts)}", width=width)
    html_path.write_text(html, encoding="utf-8")
    if open_browser:
        webbrowser.open(html_path.as_uri())
    return html_path


# Module-specific artifact renderers.
def render_position_report_html(
    *,
    output_dir: Path,
    report_name: str,
    tables: Mapping[str, pd.DataFrame],
    figure_paths: list[Path],
    open_browser: bool = False,
    subtitle: str | None = None,
) -> Path:
    return _render_specialized_report(
        title="Position Report",
        output_dir=output_dir,
        report_name=report_name,
        tables=tables,
        figure_paths=figure_paths,
        subtitle=subtitle,
        open_browser=open_browser,
        width=1180,
        specs=[
            ReportSectionSpec(
                title="Holdings Overview",
                names=["summary", "position_summary", "positions", "latest_holdings"],
                intro="Position-level composition, latest holdings, and headline concentration snapshots.",
            ),
            ReportSectionSpec(
                title="Exposure Detail",
                names=["concentration", "sector_allocations"],
                intro="Concentration and sector mix useful for validating portfolio shape.",
            ),
            ReportSectionSpec(
                title="Position Charts",
                intro="Visual views of holdings, concentration, and allocation mix.",
                kind="figures",
            ),
        ],
    )


def render_trade_report_html(
    *,
    output_dir: Path,
    report_name: str,
    tables: Mapping[str, pd.DataFrame],
    figure_paths: list[Path],
    open_browser: bool = False,
    subtitle: str | None = None,
) -> Path:
    return _render_specialized_report(
        title="Trade Analysis",
        output_dir=output_dir,
        report_name=report_name,
        tables=tables,
        figure_paths=figure_paths,
        subtitle=subtitle,
        open_browser=open_browser,
        width=1240,
        specs=[
            ReportSectionSpec(
                title="Trade Summary",
                names=["summary", "transaction_summary", "round_trip_summary"],
                intro="Headline trade diagnostics, turnover, and round-trip performance summaries.",
            ),
            ReportSectionSpec(
                title="Trade Ledger",
                names=["transactions", "round_trips"],
                intro="Underlying transactions and round-trip records used in the analysis.",
            ),
            ReportSectionSpec(
                title="Capacity And Attribution",
                names=["capacity_summary", "factor_attribution"],
                intro="Capacity constraints and factor-driven explanations of trading outcomes.",
            ),
            ReportSectionSpec(
                title="Trade Charts",
                intro="Turnover, PnL, holding period, and other trade behavior charts.",
                kind="figures",
            ),
        ],
    )


def render_portfolio_report_html(
    *,
    output_dir: Path,
    report_name: str,
    tables: Mapping[str, pd.DataFrame],
    figure_paths: list[Path],
    open_browser: bool = False,
    subtitle: str | None = None,
) -> Path:
    return _render_specialized_report(
        title="Portfolio Tear Sheet",
        output_dir=output_dir,
        report_name=report_name,
        tables=tables,
        figure_paths=figure_paths,
        subtitle=subtitle,
        open_browser=open_browser,
        width=1280,
        specs=[
            ReportSectionSpec(
                title="Performance Overview",
                names=["summary", "portfolio_returns", "benchmark_returns"],
                intro="Core portfolio and benchmark return streams together with top-level tear sheet metrics.",
            ),
            ReportSectionSpec(
                title="Holdings And Exposure",
                names=["positions", "position_summary", "latest_holdings", "concentration", "sector_allocations"],
                intro="Current and historical holdings plus concentration and sector exposure context.",
            ),
            ReportSectionSpec(
                title="Trading Detail",
                names=["transactions", "transaction_summary", "round_trips", "round_trip_summary"],
                intro="Execution and round-trip detail embedded into the full portfolio tear sheet.",
            ),
            ReportSectionSpec(
                title="Capacity And Attribution",
                names=["capacity_summary", "factor_attribution"],
                intro="Capacity diagnostics and factor attribution alongside portfolio outcomes.",
            ),
            ReportSectionSpec(
                title="Portfolio Charts",
                intro="Performance, drawdown, holdings, turnover, and attribution visuals for the combined portfolio view.",
                kind="figures",
            ),
        ],
    )


# Summary has its own layout, but still lives beside the other module renderers.
def render_summary_report_html(
    summary_table: pd.DataFrame,
    figure_paths: list[Path],
    report_name: str,
    *,
    comparison_table: pd.DataFrame | None = None,
    compare_tables: dict[str, pd.DataFrame] | None = None,
    drawdown_table: pd.DataFrame | None = None,
    monthly_returns_table: pd.DataFrame | None = None,
    montecarlo_summary: pd.DataFrame | None = None,
    title: str = "Tearsheet",
    date_range: str = "",
    params_text: str = "",
    matched_dates_text: str = "",
) -> str:
    metrics_html = _table_html_with_class(summary_table, classes="summary-table")
    eoy_html = ""
    if comparison_table is not None and not comparison_table.empty:
        eoy_html = _table_html_with_class(comparison_table, classes="summary-table")
    elif comparison_table is None:
        eoy_html = ""

    compare_html = ""
    if compare_tables:
        compare_chunks: list[str] = []
        compare_order = ["daily", "weekly", "monthly", "quarterly", "yearly"]
        compare_titles = {
            "daily": "Daily Compare",
            "weekly": "Weekly Compare",
            "monthly": "Monthly Compare",
            "quarterly": "Quarterly Compare",
            "yearly": "Yearly Compare",
        }
        for key in compare_order:
            table = compare_tables.get(key)
            if table is None or table.empty:
                continue
            compare_chunks.append(f"<h4>{escape(compare_titles[key])}</h4>")
            compare_chunks.append(_table_html_with_class(table, classes="summary-table"))
        compare_html = "".join(compare_chunks)

    dd_html = _table_html_with_class(drawdown_table, classes="summary-table") if drawdown_table is not None else ""
    monthly_html = _optional_titled_table_html("Monthly Returns", monthly_returns_table, classes="summary-table")
    mc_html = _table_html_with_class(montecarlo_summary, classes="summary-table") if montecarlo_summary is not None else ""
    left_column_html = _summary_figure_strip(
        figure_paths,
        report_name,
        specs=SUMMARY_FIGURE_SPECS,
    )
    montecarlo_plot_html = _find_figure_html(figure_paths, stem_or_name=f"{report_name}_montecarlo")
    return (
        "<!-- generated by TigerQuant for multifactor evaluation -->"
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta http-equiv='X-UA-Compatible' content='IE=edge,chrome=1'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1, shrink-to-fit=no'>"
        f"<title>{escape(title)}</title>"
        "<meta name='robots' content='noindex, nofollow'>"
        "<style>"
        "body{-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;margin:30px;background:#fff;color:#000}"
        "body,p,table,td,th{font:13px/1.4 Arial,sans-serif}.container{max-width:960px;margin:auto}"
        "img,svg{width:100%}h1,h2,h3,h4{font-weight:400;margin:0}h1 dt{display:inline;margin-left:10px;font-size:14px}h3{margin-bottom:10px;font-weight:700}h4{color:grey}h4 a{color:#09c;text-decoration:none}h4 a:hover{color:#069;text-decoration:underline}hr{margin:25px 0 40px;height:0;border:0;border-top:1px solid #ccc}#left{width:620px;margin-right:18px;margin-top:-1.2rem;float:left}#right{width:320px;float:right}#left svg{margin:-1.5rem 0}#monthly_heatmap{overflow:hidden}#monthly_heatmap svg{margin:-1.5rem 0}table{margin:0 0 40px;border:0;border-spacing:0;width:100%}table td,table th{text-align:right;padding:4px 5px 3px 5px}table th{text-align:right;padding:6px 5px 5px 5px}table td:first-of-type,table th:first-of-type{text-align:left;padding-left:2px}table td:last-of-type,table th:last-of-type{text-align:right;padding-right:2px}td hr{margin:5px 0}table th{font-weight:400}table thead th{font-weight:700;background:#eee}#eoy table td:after{content:\"%\"}#eoy table td:first-of-type:after,#eoy table td:last-of-type:after,#eoy table td:nth-of-type(4):after{content:\"\"}#eoy table th{text-align:right}#eoy table th:first-of-type{text-align:left}#eoy table td:after{content:\"%\"}#eoy table td:first-of-type:after,#eoy table td:last-of-type:after{content:\"\"}#ddinfo table td:nth-of-type(3):after{content:\"%\"}#ddinfo table th{text-align:right}#ddinfo table td:first-of-type,#ddinfo table td:nth-of-type(2),#ddinfo table th:first-of-type,#ddinfo table th:nth-of-type(2){text-align:left}#ddinfo table td:nth-of-type(3):after{content:\"%\"}"
        "@media print{hr{margin:25px 0}body{margin:0}.container{max-width:100%;margin:0}#left{width:55%;margin:0}#left svg{margin:0 0 -10%}#left svg:first-of-type{margin-top:-30%}#right{margin:0;width:45%}}"
        "</style></head><body>"
        "<div class='container'>"
        f"<h1>{escape(title)} <dt>{escape(date_range)}{escape(matched_dates_text)}</dt></h1>"
        f"<h4>{escape(params_text)} Generated by <a href='http://quantstats.io' target='quantstats'>QuantStats</a> style report (TigerQuant)</h4>"
        "<hr>"
        "<div id='left'>"
        f"{left_column_html}"
        "</div>"
        "<div id='right'>"
        "<h3>Summary Metrics / Key Performance Metrics</h3>"
        f"{metrics_html}"
        f"{_optional_panel_html(panel_id='eoy', title='EOY Returns vs Benchmark' if comparison_table is not None and not comparison_table.empty else 'EOY Returns', body_html=eoy_html)}"
        f"{_optional_panel_html(panel_id='compare', title='Compare Tables', body_html=compare_html)}"
        f"<div id='monthly_returns'>{monthly_html}</div>"
        f"{_optional_panel_html(panel_id='ddinfo', title='Worst 10 Drawdowns', body_html=dd_html)}"
        f"{_optional_panel_html(panel_id='montecarlo', title='Monte Carlo', body_html=mc_html + montecarlo_plot_html)}"
        "</div></div><style>*{white-space:auto !important;}</style></body></html>"
    )


# Combined multifactor landing page that links the four module-specific reports.
def render_multifactor_index_report(
    *,
    title: str,
    output_dir: Path,
    report_name: str,
    modules: list[dict[str, Any]],
    open_browser: bool = False,
    subtitle: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{report_name}.html"

    index_rows: list[str] = []
    module_sections: list[str] = []
    nav_links: list[str] = []
    total_tables = 0
    total_figures = 0
    for module in modules:
        module_name = str(module.get("name", "module"))
        module_title = str(module.get("title", module_name))
        report_path = module.get("report_path")
        tables = module.get("tables", {})
        figures = module.get("figure_paths", [])
        summary_text = escape(str(module.get("summary", "")))
        table_count = len(tables) if isinstance(tables, Mapping) else 0
        figure_count = len(figures) if figures else 0
        total_tables += table_count
        total_figures += figure_count

        if isinstance(report_path, Path) and report_path.exists():
            try:
                report_href = report_path.relative_to(output_dir).as_posix()
            except ValueError:
                report_href = report_path.name
            report_link = f"<a href='{escape(report_href)}'>open report</a>"
        else:
            report_link = "<em>report unavailable</em>"

        nav_links.append(f"<a href='#{_slugify(module_name)}'>{escape(module_title)}</a>")

        index_rows.append(
            "<tr>"
            f"<td>{escape(module_title)}</td>"
            f"<td>{report_link}</td>"
            f"<td>{escape(', '.join(sorted(tables.keys())) if isinstance(tables, Mapping) else '')}</td>"
            f"<td>{escape(', '.join(Path(path).stem for path in figures) if figures else '')}</td>"
            f"<td>{summary_text}</td>"
            "</tr>"
        )

        section_parts: list[str] = [
            "<div class='module-head'>"
            f"<div><h3>{escape(module_title)}</h3><div class='module-meta'><span class='chip'>{table_count} tables</span><span class='chip'>{figure_count} figures</span></div></div>"
            f"<div class='module-meta'><span class='chip'>{report_link}</span></div>"
            "</div>"
        ]
        if summary_text:
            section_parts.append(f"<p class='muted'>{summary_text}</p>")
        if isinstance(tables, Mapping):
            for table_name, table in tables.items():
                section_parts.append(f"<h3>{escape(str(table_name))}</h3>{_table_html(table)}")
        for figure in figures or []:
            path = Path(figure)
            if path.exists():
                try:
                    rel = path.relative_to(output_dir).as_posix()
                except ValueError:
                    rel = path.name
                section_parts.append(
                    f"<section><h3>{escape(path.stem)}</h3><img src='{escape(rel)}' alt='{escape(path.stem)}'></section>"
                )
        module_sections.append(
            f"<section class='block module' id='{_slugify(module_name)}'>"
            + "".join(section_parts)
            + "<a class='toplink' href='#top'>Back to top</a></section>"
        )

    body = (
        "<div id='top'></div>"
        "<div class='index-grid'>"
        f"<div class='index-card'><div class='label'>Modules</div><div class='value'>{len(modules)}</div></div>"
        f"<div class='index-card'><div class='label'>Tables</div><div class='value'>{total_tables}</div></div>"
        f"<div class='index-card'><div class='label'>Figures</div><div class='value'>{total_figures}</div></div>"
        "</div>"
        "<h2>Module Index</h2>"
        "<table><thead><tr><th>Module</th><th>Report</th><th>Tables</th><th>Figures</th><th>Notes</th></tr></thead><tbody>"
        + "".join(index_rows)
        + "</tbody></table>"
        + "".join(module_sections)
    )
    html = _report_shell(title, subtitle, body, nav="".join(nav_links), width=1240)
    html_path.write_text(html, encoding="utf-8")
    if open_browser:
        webbrowser.open(html_path.as_uri())
    return html_path
