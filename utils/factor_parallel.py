from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from tiger_factors.factor_store import FactorStore
from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store.store import DatasetSaveResult


@dataclass(frozen=True)
class FactorStorageResult:
    parquet_path: Path
    metadata_path: Path
    payload: dict[str, Any]


@dataclass(frozen=True)
class FactorParallelResult:
    computed_frames: dict[int, pd.DataFrame]
    saved_factor_paths: dict[str, str] | None = None
    saved_metadata_paths: dict[str, str] | None = None
    execution_mode: str = "thread"


def resolve_parallel_workers(task_count: int, requested_workers: int | None, *, default_cap: int = 4) -> int:
    if requested_workers is not None and requested_workers > 0:
        return int(requested_workers)
    return max(1, min(int(task_count), int(default_cap)))


def run_factor_tasks_parallel(
    task_ids: Sequence[int],
    *,
    compute_fn: Callable[[int], tuple[int, pd.DataFrame]],
    compute_workers: int | None = None,
    save_workers: int | None = None,
    save: bool = True,
    output_dir: str | Path | None = None,
    factor_name_for_task: Callable[[int], str] | None = None,
    metadata_for_task: Callable[[int, str], dict[str, Any] | None] | None = None,
    initializer: Callable[..., None] | None = None,
    initargs: Sequence[Any] = (),
    prefer_process: bool = True,
) -> FactorParallelResult:
    resolved_task_ids = [int(task_id) for task_id in task_ids]
    if not resolved_task_ids:
        return FactorParallelResult(computed_frames={})

    compute_workers = resolve_parallel_workers(len(resolved_task_ids), compute_workers)
    save_workers = resolve_parallel_workers(len(resolved_task_ids), save_workers)

    saved_factor_paths: dict[str, str] | None = {} if save else None
    saved_metadata_paths: dict[str, str] | None = {} if save else None

    executor_classes = [ProcessPoolExecutor, ThreadPoolExecutor] if prefer_process else [ThreadPoolExecutor, ProcessPoolExecutor]
    last_error: Exception | None = None

    for executor_cls in executor_classes:
        computed_frames: dict[int, pd.DataFrame] = {}
        execution_mode = "process" if executor_cls is ProcessPoolExecutor else "thread"
        try:
            with executor_cls(
                max_workers=compute_workers,
                initializer=initializer,
                initargs=tuple(initargs) if initializer is not None else (),
            ) as compute_pool:
                compute_futures = {
                    compute_pool.submit(compute_fn, task_id): int(task_id)
                    for task_id in resolved_task_ids
                }

                pending_save_futures: dict[Any, tuple[int, str]] = {}
                with ThreadPoolExecutor(max_workers=save_workers) as save_pool:
                    for future in as_completed(compute_futures):
                        task_id, factor_frame = future.result()
                        computed_frames[int(task_id)] = factor_frame
                        if save:
                            if factor_name_for_task is None:
                                raise ValueError("factor_name_for_task is required when save=True.")
                            factor_name = factor_name_for_task(int(task_id))
                            metadata = metadata_for_task(int(task_id), execution_mode) if metadata_for_task is not None else None
                            pending_save_futures[
                                save_pool.submit(
                                    _save_factor_frame,
                                    factor_name=factor_name,
                                    factor_frame=factor_frame,
                                    output_dir=output_dir,
                                    metadata=metadata,
                                )
                            ] = (int(task_id), factor_name)

                    if save:
                        assert saved_factor_paths is not None and saved_metadata_paths is not None
                        for save_future in as_completed(list(pending_save_futures)):
                            _, factor_name = pending_save_futures[save_future]
                            storage_result: FactorStorageResult = save_future.result()
                            saved_factor_paths[factor_name] = str(storage_result.parquet_path)
                            saved_metadata_paths[factor_name] = str(storage_result.metadata_path)

            return FactorParallelResult(
                computed_frames=computed_frames,
                saved_factor_paths=saved_factor_paths if save else None,
                saved_metadata_paths=saved_metadata_paths if save else None,
                execution_mode=execution_mode,
            )
        except (NotImplementedError, PermissionError, OSError) as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError("Parallel factor runner could not start.")


def _save_factor_frame(
    *,
    factor_name: str,
    factor_frame: pd.DataFrame,
    output_dir: str | Path | None,
    metadata: dict[str, Any] | None,
) -> FactorStorageResult:
    factor_store = FactorStore(output_dir)
    frame = factor_frame.copy()
    if factor_name in frame.columns and factor_name != "value":
        frame = frame.rename(columns={factor_name: "value"})
    elif "value" not in frame.columns:
        candidate_columns = [column for column in frame.columns if column not in {"date_", "code"}]
        if len(candidate_columns) == 1:
            frame = frame.rename(columns={candidate_columns[0]: "value"})
        else:
            raise ValueError(
                "factor frame must contain either the factor column or a single value column; "
                f"got columns={list(frame.columns)!r}"
            )
    spec = FactorSpec(
        region=str((metadata or {}).get("region", "us")),
        sec_type=str((metadata or {}).get("sec_type", "stock")),
        freq=str((metadata or {}).get("freq", "1d")),
        table_name=factor_name,
        variant=(metadata or {}).get("variant", None),
        provider=str((metadata or {}).get("provider", "tiger")),
    )
    save_result = factor_store.save_factor(spec, frame.loc[:, ["date_", "code", "value"]], metadata=metadata)
    return FactorStorageResult(
        parquet_path=save_result.files[0],
        metadata_path=save_result.manifest_path,
        payload={
            "factor_name": factor_name,
            "rows": int(save_result.rows),
            "date_min": save_result.date_min,
            "date_max": save_result.date_max,
            "codes": int(frame["code"].nunique()) if "code" in frame.columns else 0,
            "dataset_dir": str(save_result.dataset_dir),
            "manifest_path": str(save_result.manifest_path),
            "files": [str(path) for path in save_result.files],
            "spec": spec.to_dict(),
        },
    )


__all__ = ["FactorParallelResult", "resolve_parallel_workers", "run_factor_tasks_parallel"]
