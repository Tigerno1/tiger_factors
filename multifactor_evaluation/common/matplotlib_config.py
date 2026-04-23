from __future__ import annotations

import os
from pathlib import Path

from tiger_factors.multifactor_evaluation.common.plotting import configure_quantstats_style


def configure_matplotlib(*, cache_dir: str | Path | None = None, font_family: str = "DejaVu Sans") -> Path:
    project_root = Path(__file__).resolve().parents[2]
    default_cache_dir = project_root / "tiger_analysis_outputs" / ".cache" / "matplotlib"
    # Use a project-local cache directory so matplotlib never falls back to an
    # unwritable home-directory cache in sandboxed or locked-down environments.
    resolved_cache_dir = Path(cache_dir or default_cache_dir).expanduser()
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(resolved_cache_dir)
    os.environ["MPLBACKEND"] = "Agg"

    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import rcParams

    rcParams["font.family"] = [font_family]
    rcParams["axes.unicode_minus"] = False
    configure_quantstats_style(font_family=font_family)
    return resolved_cache_dir
