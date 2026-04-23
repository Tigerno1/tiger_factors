from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiger_factors.factor_maker.vectorization.indicators import Alpha101IndicatorTransformer


def main() -> None:
    """
    Minimal Alpha101 indicator example.

    This shows the recommended instantiation pattern:

    - create the transformer with a calendar and time range
    - run a single alpha or the full parallel pipeline
    - inspect the returned frames in memory
    """
    transformer = Alpha101IndicatorTransformer(
        calendar="XNYS",
        start="2020-06-01",
        end="2024-06-01",
    )

    # Small, safe smoke test: compute one factor.
    result = transformer.compute_alpha101(
        1,
        save_factors=False,
        save_adj_price=True,
    )

    print("codes:", len(result.codes))
    print("raw_price_frame:", result.raw_price_frame.shape)
    print("calendar_frame:", result.calendar_frame.shape)
    print("adjusted_frame:", result.adjusted_frame.shape)
    print("classification_frame:", result.classification_frame.shape)
    print("alpha_input_frame:", result.alpha_input_frame.shape)
    print("factor_frame:", result.factor_frame.shape)
    print("saved_factor_paths:", result.saved_factor_paths)
    print("saved_adjusted_price_path:", result.saved_adjusted_price_path)
    print(result.factor_frame.head())


if __name__ == "__main__":
    main()
