from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from tiger_api.core.domain_facade import ensure_domain_registered

from tiger_factors.factor_algorithm.financial_factors.financial_factors import AnnualFinancialFactorEngine
from tiger_factors.factor_algorithm.financial_factors.financial_factors import QuarterlyFinancialFactorEngine
from tiger_factors.factor_algorithm.financial_factors.financial_factors import TTMFinancialFactorEngine
from tiger_factors.factor_store import TigerFactorLibrary


DEFAULT_OUTPUT_ROOT = Path("/Volumes/Quant_Disk/evaluation/financial_factors")
DEFAULT_START = "2018-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_UNIVERSE_PROVIDER = "github"
DEFAULT_UNIVERSE_NAME = "sp500_constituents"
DEFAULT_UNIVERSE_CODE_COLUMN = "code"
DEFAULT_VARIANTS: dict[str, str | None] = {
    "base": None,
    "bank": "bank",
    "insurance": "insurance",
}


def resolve_codes(
    library: TigerFactorLibrary,
    *,
    universe_provider: str = DEFAULT_UNIVERSE_PROVIDER,
    universe_name: str = DEFAULT_UNIVERSE_NAME,
    universe_code_column: str = DEFAULT_UNIVERSE_CODE_COLUMN,
    universe_limit: int | None = None,
    universe_sort_by: str | None = None,
    universe_ascending: bool = True,
    code_limit: int | None = None,
    as_ex: bool | None = None,
) -> list[str]:
    limit = universe_limit if universe_limit is not None else code_limit
    codes = library.resolve_universe_codes(
        provider=universe_provider,
        dataset=universe_name,
        code_column=universe_code_column,
        limit=limit,
        sort_by=universe_sort_by,
        ascending=universe_ascending,
        as_ex=as_ex,
    )
    if not codes:
        raise RuntimeError(f"No universe codes were resolved for {universe_provider}:{universe_name}.")
    return codes


def record_financial_factors(
    *,
    library: TigerFactorLibrary | None = None,
    codes: list[str] | None = None,
    start: str,
    end: str,
    output_root: str | Path,
    region: str = "us",
    sec_type: str = "stock",
    price_provider: str = "simfin",
    universe_provider: str = DEFAULT_UNIVERSE_PROVIDER,
    universe_name: str = DEFAULT_UNIVERSE_NAME,
    universe_code_column: str = DEFAULT_UNIVERSE_CODE_COLUMN,
    universe_limit: int | None = None,
    universe_sort_by: str | None = None,
    universe_ascending: bool = True,
    code_limit: int | None = None,
    as_ex: bool | None = None,
    monthly_output: bool = True,
    max_factors: int | None = None,
    variants: Mapping[str, str | None] | None = None,
    verbose: bool = False,
    ensure_domain: bool = True,
) -> list[dict[str, object]]:
    if ensure_domain:
        ensure_domain_registered(provider="simfin", region=region, sec_type=sec_type)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    library = library or TigerFactorLibrary(output_dir=output_root, price_provider=price_provider, verbose=verbose)
    resolved_codes = list(
        codes
        or resolve_codes(
            library,
            universe_provider=universe_provider,
            universe_name=universe_name,
            universe_code_column=universe_code_column,
            universe_limit=universe_limit,
            universe_sort_by=universe_sort_by,
            universe_ascending=universe_ascending,
            code_limit=code_limit,
            as_ex=as_ex,
        )
    )
    variant_map = dict(variants or DEFAULT_VARIANTS)

    runs: list[dict[str, object]] = []
    for engine_cls in (QuarterlyFinancialFactorEngine, AnnualFinancialFactorEngine, TTMFinancialFactorEngine):
        for variant_name, variant in variant_map.items():
            engine = engine_cls(
                library=library,
                codes=resolved_codes,
                start=start,
                end=end,
                variant=variant,
                price_provider=price_provider,
                output_dir=output_root,
                monthly_output=monthly_output,
                max_factors=max_factors,
            )
            factor_frame = engine.compute_factor_frame(max_factors=max_factors)
            bundle = engine.save_factor_files(
                name=f"{engine.output_tag}_{variant_name}_financial_factors",
                factor_frame=factor_frame,
            )
            runs.append(
                {
                    "family": "financial",
                    "engine": engine_cls.__name__,
                    "variant_name": variant_name,
                    "variant": variant,
                    "rows": int(len(factor_frame)),
                    "factor_count": int(max(0, len(factor_frame.columns) - 2)),
                    "parquet_path": str(bundle["files"][0]["parquet_path"]) if bundle.get("files") else None,
                    "manifest_path": str(bundle["manifest_path"]),
                    "date_min": str(factor_frame["date_"].min()) if not factor_frame.empty else None,
                    "date_max": str(factor_frame["date_"].max()) if not factor_frame.empty else None,
                }
            )

    manifest = output_root / "financial_factors_manifest.json"
    manifest.write_text(json.dumps(runs, indent=2, default=str), encoding="utf-8")
    return runs


def main() -> list[dict[str, object]]:
    parser = argparse.ArgumentParser(description="Record SimFin financial factors.")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--price-provider", default="simfin")
    parser.add_argument("--universe-provider", default=DEFAULT_UNIVERSE_PROVIDER)
    parser.add_argument("--universe-name", default=DEFAULT_UNIVERSE_NAME)
    parser.add_argument("--universe-code-column", default=DEFAULT_UNIVERSE_CODE_COLUMN)
    parser.add_argument("--universe-limit", type=int, default=None)
    parser.add_argument("--universe-sort-by", default=None)
    parser.add_argument(
        "--universe-ascending",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sort the resolved universe codes ascending by the universe sort column.",
    )
    parser.add_argument("--max-factors", type=int, default=None)
    parser.add_argument("--region", default="us")
    parser.add_argument("--sec-type", default="stock")
    parser.add_argument("--as-ex", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ensure-domain", action="store_true")
    args = parser.parse_args()
    return record_financial_factors(
        start=args.start,
        end=args.end,
        output_root=args.output_root,
        region=args.region,
        sec_type=args.sec_type,
        price_provider=args.price_provider,
        universe_provider=args.universe_provider,
        universe_name=args.universe_name,
        universe_code_column=args.universe_code_column,
        universe_limit=args.universe_limit,
        universe_sort_by=args.universe_sort_by,
        universe_ascending=args.universe_ascending,
        as_ex=args.as_ex,
        monthly_output=True,
        max_factors=args.max_factors,
        verbose=False,
        ensure_domain=args.ensure_domain,
    )


__all__ = [
    "DEFAULT_END",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_START",
    "DEFAULT_UNIVERSE_CODE_COLUMN",
    "DEFAULT_UNIVERSE_NAME",
    "DEFAULT_UNIVERSE_PROVIDER",
    "DEFAULT_VARIANTS",
    "main",
    "record_financial_factors",
    "resolve_codes",
]


if __name__ == "__main__":
    main()
