from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiger_factors.factor_algorithm.valuation_factors import record_valuation_factors


OUTPUT_ROOT = Path("/Volumes/Quant_Disk/evaluation/valuation_factors")
START = "2018-01-01"
END = "2024-12-31"


def main() -> None:
    runs = record_valuation_factors(
        start=START,
        end=END,
        output_root=OUTPUT_ROOT,
        price_provider="simfin",
        monthly_output=True,
        verbose=False,
    )
    manifest = OUTPUT_ROOT / "valuation_factors_manifest.json"
    print(json.dumps(runs[:2], indent=2, default=str))
    print(f"\nManifest saved to {manifest}")


if __name__ == "__main__":
    main()
