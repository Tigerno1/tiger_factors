from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class TearSheetResultMixin:
    output_dir: Path
    figure_paths: list[Path]
    report_name: str

    def _table_paths(self) -> dict[str, Path]:
        raise NotImplementedError

    def tables(self) -> list[str]:
        return sorted(name for name, path in self._table_paths().items() if path.exists())

    def preferred_table_order(self) -> list[str]:
        return []

    def ordered_tables(self) -> list[str]:
        available = self.tables()
        ordered = [name for name in self.preferred_table_order() if name in available]
        ordered.extend(name for name in available if name not in ordered)
        return ordered

    def imgs(self) -> list[str]:
        return sorted(path.stem for path in self.figure_paths if path.exists())

    def report(self) -> str | None:
        return None

    @staticmethod
    def _read_table(path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported table format: {path}")

    def get_table(self, table_name: str | None = None) -> pd.DataFrame:
        name = "summary" if table_name is None else str(table_name)
        path = self._table_paths().get(name)
        if path is None or not path.exists():
            raise FileNotFoundError(f"table '{name}' not found in tear sheet output")
        return self._read_table(path)

    def get_img(self, img_name: str):
        path = next((path for path in self.figure_paths if path.stem == img_name or path.stem.endswith(f"_{img_name}")), None)
        if path is None or not path.exists():
            raise FileNotFoundError(f"image '{img_name}' not found in tear sheet output")
        from PIL import Image

        return Image.open(path)

    def to_summary(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "figure_paths": [str(path) for path in self.figure_paths],
            "report_name": self.report_name,
        }
