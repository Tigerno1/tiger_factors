from __future__ import annotations

from pathlib import Path

import pandas as pd


def to_parquet_clean(frame: pd.DataFrame | pd.Series, path: str | Path, **kwargs) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    options = dict(kwargs)
    options.setdefault("engine", "pyarrow")
    if isinstance(frame, pd.Series):
        clean = frame.copy()
        clean.attrs = {}
        clean.to_frame().to_parquet(output, **options)
    else:
        clean = frame.copy()
        clean.attrs = {}
        clean.to_parquet(output, **options)
    return output
