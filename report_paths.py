from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def tiger_tmp_root() -> Path:
    return project_root() / "tiger_analysis_outputs" / ".tmp"


def figure_output_dir_for(name: str) -> Path:
    return tiger_tmp_root() / name


def report_output_root_for(*parts: str) -> Path:
    """Return the default on-disk root for rendered reports."""
    return project_root() / "tiger_analysis_outputs" / "reports" / Path(*parts)
