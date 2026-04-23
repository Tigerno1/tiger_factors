from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiger_factors.factor_store import FactorResult, TigerFactorLibrary, main

__all__ = ["FactorResult", "TigerFactorLibrary", "main"]


if __name__ == "__main__":
    main()
