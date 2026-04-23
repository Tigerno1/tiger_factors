from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import websockets

from tiger_api.conf import EOD_API_KEY

EOD_WS_BASE = "wss://ws.eodhistoricaldata.com/ws"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FACTORS_TICK_OUTPUT_DIR = PROJECT_ROOT / "src" / "output" / "factors" / "market_tick" / "eod_us_trade"


@dataclass
class TickBatchWriter:
    """Buffer websocket ticks and flush them to partitioned parquet files."""

    output_dir: Path
    flush_rows: int = 5000
    write_jsonl: bool = False
    _buffer: list[dict[str, Any]] = field(default_factory=list)

    def add(self, row: dict[str, Any]) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return

        df = pd.DataFrame(self._buffer)
        self._buffer.clear()

        min_ts = int(df["event_ts_ms"].min())
        max_ts = int(df["event_ts_ms"].max())

        for symbol, sdf in df.groupby("symbol", sort=False):
            if sdf.empty:
                continue
            trade_date = str(sdf["trade_date"].iloc[0])
            target_dir = self.output_dir / f"symbol={symbol}" / f"trade_date={trade_date}"
            target_dir.mkdir(parents=True, exist_ok=True)

            part_name = f"part-{min_ts}-{max_ts}.parquet"
            parquet_path = target_dir / part_name
            sdf.to_parquet(parquet_path, index=False)

            if self.write_jsonl:
                jsonl_path = target_dir / part_name.replace(".parquet", ".jsonl")
                sdf.to_json(jsonl_path, orient="records", lines=True, date_format="iso")


@dataclass
class RawMessageWriter:
    """Persist raw websocket messages as append-only JSONL for replay/debug."""

    output_dir: Path

    def write(self, raw_message: str, *, received_at: datetime) -> None:
        date_part = received_at.strftime("%Y-%m-%d")
        hour_part = received_at.strftime("%H")
        target_dir = self.output_dir / "raw" / f"date={date_part}" / f"hour={hour_part}"
        target_dir.mkdir(parents=True, exist_ok=True)

        path = target_dir / "events.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(raw_message)
            f.write("\n")


def _utc_iso_from_ms(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).isoformat()


def _to_tick_row(payload: dict[str, Any]) -> dict[str, Any] | None:
    symbol = payload.get("s")
    ts_ms = payload.get("t")
    if symbol is None or ts_ms is None:
        return None

    ts_ms = int(ts_ms)
    event_time = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)

    return {
        "provider": "eodhd",
        "stream": "us_trade",
        "symbol": symbol,
        "event_ts_ms": ts_ms,
        "event_time_utc": event_time,
        "trade_date": event_time.date(),
        "price": payload.get("p"),
        "size": payload.get("v"),
        "condition_code": payload.get("c"),
        "dark_pool": payload.get("dp"),
        "market_status": payload.get("ms"),
        "received_at_utc": datetime.now(UTC),
        "raw_json": json.dumps(payload, separators=(",", ":"), ensure_ascii=True),
    }


async def run_ws_tick_capture(
    *,
    symbols: list[str],
    api_token: str,
    output_dir: Path,
    flush_rows: int,
    duration_sec: int | None,
    write_jsonl: bool,
    write_raw_jsonl: bool,
) -> None:
    if not api_token:
        raise ValueError("EOD API token is required. Set EOD_API_KEY or pass --api-token.")
    if not symbols:
        raise ValueError("At least one symbol is required.")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "dataset": "market_tick_eod_us_trade",
                "provider": "eodhd",
                "transport": "websocket",
                "endpoint": "/ws/us",
                "storage": {
                    "structured": "parquet",
                    "raw_stream": "jsonl",
                },
                "symbols": symbols,
                "created_at_utc": datetime.now(UTC).isoformat(),
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    writer = TickBatchWriter(output_dir=output_dir, flush_rows=flush_rows, write_jsonl=write_jsonl)
    raw_writer = RawMessageWriter(output_dir=output_dir)

    uri = f"{EOD_WS_BASE}/us?api_token={api_token}"
    subscribe_msg = {"action": "subscribe", "symbols": ",".join(symbols)}

    start = datetime.now(UTC)
    print(f"connecting: {uri}")
    print(f"subscribing: {subscribe_msg['symbols']}")

    async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(json.dumps(subscribe_msg))

        while True:
            if duration_sec is not None:
                elapsed = (datetime.now(UTC) - start).total_seconds()
                if elapsed >= duration_sec:
                    break

            raw = await ws.recv()
            if write_raw_jsonl and isinstance(raw, str):
                raw_writer.write(raw, received_at=datetime.now(UTC))

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        row = _to_tick_row(item)
                        if row is not None:
                            writer.add(row)
                continue

            if isinstance(payload, dict):
                row = _to_tick_row(payload)
                if row is not None:
                    writer.add(row)
                    if len(writer._buffer) % 200 == 0:
                        print(
                            f"rows buffered={len(writer._buffer)} latest={row['symbol']} {row['price']} @ {_utc_iso_from_ms(row['event_ts_ms'])}"
                        )

    writer.flush()
    print(f"done. files written under: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture EODHD US websocket trade ticks and write parquet files.")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g. AAPL,MSFT,TSLA")
    parser.add_argument("--api-token", default=EOD_API_KEY, help="EOD API token. Defaults to EOD_API_KEY env var.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_FACTORS_TICK_OUTPUT_DIR),
        help="Output directory for parquet partitions (defaults to tiger_factors/output/factors/market_tick/eod_us_trade).",
    )
    parser.add_argument("--flush-rows", type=int, default=5000, help="Flush parquet after N rows.")
    parser.add_argument("--duration-sec", type=int, default=None, help="Stop after N seconds. Omit for continuous run.")
    parser.add_argument(
        "--write-jsonl",
        action="store_true",
        help="Also write JSONL sidecar files for quick inspection/debugging.",
    )
    parser.add_argument(
        "--no-raw-jsonl",
        action="store_true",
        help="Disable raw websocket JSONL capture. Raw capture is enabled by default.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    symbol_list = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    try:
        asyncio.run(
            run_ws_tick_capture(
                symbols=symbol_list,
                api_token=args.api_token,
                output_dir=Path(args.output_dir),
                flush_rows=args.flush_rows,
                duration_sec=args.duration_sec,
                write_jsonl=args.write_jsonl,
                write_raw_jsonl=not args.no_raw_jsonl,
            )
        )
    except KeyboardInterrupt:
        print("stopped by user")


__all__ = [
    "DEFAULT_FACTORS_TICK_OUTPUT_DIR",
    "EOD_WS_BASE",
    "RawMessageWriter",
    "TickBatchWriter",
    "parse_args",
    "run_ws_tick_capture",
]
