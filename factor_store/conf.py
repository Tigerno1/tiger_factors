from __future__ import annotations

from pathlib import Path


DEFAULT_FACTOR_STORE_ROOT = Path("/Volumes/Quant_Disk/factor_store")
DEFAULT_EVALUATION_ROOT = Path("/Volumes/Quant_Disk/factor_store")
DEFAULT_MACRO_DATA_SOURCE = "fred"

# Common macro/economic readers supported by pandas-datareader.
SUPPORTED_MACRO_DATA_SOURCES = (
    "fred",
    "bankofcanada",
    "econdb",
    "eurostat",
    "famafrench",
    "oecd",
    "worldbank",
    "quandl",
)


__all__ = [
    "DEFAULT_EVALUATION_ROOT",
    "DEFAULT_FACTOR_STORE_ROOT",
    "DEFAULT_MACRO_DATA_SOURCE",
    "SUPPORTED_MACRO_DATA_SOURCES",
]
