from __future__ import annotations

import os


def pytest_configure() -> None:
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"
