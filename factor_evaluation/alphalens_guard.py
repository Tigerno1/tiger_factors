from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass
class AlphalensFigureGuard:
    output_dir: str | Path
    dpi: int = 200
    disable_show: bool = True
    disable_close: bool = True
    close_all_on_exit: bool = True

    saved_figures: list[Path] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._original_show = None
        self._original_close = None
        self._existing_figs: set[int] = set()

    def __enter__(self) -> "AlphalensFigureGuard":
        self._existing_figs = set(plt.get_fignums())
        self._original_show = plt.show
        self._original_close = plt.close

        if self.disable_show:
            plt.show = lambda *args, **kwargs: None  # type: ignore[assignment]
        if self.disable_close:
            plt.close = lambda *args, **kwargs: None  # type: ignore[assignment]

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.saved_figures = self.save_new_figures()

        if self._original_show is not None:
            plt.show = self._original_show  # type: ignore[assignment]
        if self._original_close is not None:
            plt.close = self._original_close  # type: ignore[assignment]

        if self.close_all_on_exit and self._original_close is not None:
            self._original_close("all")

    def save_new_figures(self) -> list[Path]:
        saved_files: list[Path] = []

        for fig_num in plt.get_fignums():
            if fig_num in self._existing_figs:
                continue

            figure = plt.figure(fig_num)
            if not figure.axes:
                continue

            figure.canvas.draw()

            title = self._extract_figure_title(figure)
            safe_title = self._sanitize_filename(title)

            filename = f"figure_{fig_num:02d}"
            if safe_title:
                filename = f"{filename}_{safe_title[:60]}"

            filepath = self.output_dir / f"{filename}.png"
            figure.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
            saved_files.append(filepath)

        return saved_files

    @staticmethod
    def _extract_figure_title(figure) -> str:
        for axis in figure.axes:
            title = axis.get_title()
            if title:
                return title
        return ""

    @staticmethod
    def _sanitize_filename(text: str) -> str:
        return "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "_"
            for ch in text
        ).strip("_")