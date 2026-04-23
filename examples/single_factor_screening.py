from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiger_factors.factor_evaluation.factor_screening import evaluate_and_screen_factor_root


FINANCIAL_FACTOR_ROOT = Path("/Volumes/Quant_Disk/evaluation/financial_factors")
VALUATION_FACTOR_ROOT = Path("/Volumes/Quant_Disk/evaluation/valuation_factors")
SCREENING_DIRNAME = "screening"


def run_screening(input_root: Path) -> dict[str, object]:
    output_root = input_root / SCREENING_DIRNAME
    registry = evaluate_and_screen_factor_root(
        input_root=input_root,
        output_root=output_root,
        price_provider="simfin",
    )
    manifest = output_root / "single_factor_screening_manifest.json"
    return {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "factor_count": int(len(registry)),
        "manifest_path": str(manifest),
    }


def main() -> None:
    results: list[dict[str, object]] = []
    for root in (FINANCIAL_FACTOR_ROOT, VALUATION_FACTOR_ROOT):
        if not root.exists():
            continue
        results.append(run_screening(root))

    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
