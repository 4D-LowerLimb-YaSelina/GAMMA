import contextvars
import json
import math
import os
import time
import uuid
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import torch

RUNTIME_SCHEMA_VERSION = 1
_ACTIVE_RECORDER = contextvars.ContextVar("articulation_runtime_recorder", default=None)


def _synchronize_cuda(device: Optional[str]):
    if device is None or not str(device).startswith("cuda"):
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize(device=device)


class ShapeRuntimeRecorder:
    """Collect ordered, wall-clock stage timings for one shape attempt."""

    def __init__(self, *, device: Optional[str], synchronize_cuda: bool = True):
        self.device = device
        self.synchronize_cuda = bool(synchronize_cuda)
        self.stages = []

    def _synchronize(self):
        if self.synchronize_cuda:
            _synchronize_cuda(self.device)

    @contextmanager
    def activate(self):
        token = _ACTIVE_RECORDER.set(self)
        try:
            yield self
        finally:
            _ACTIVE_RECORDER.reset(token)

    @contextmanager
    def stage(self, name: str):
        if not isinstance(name, str) or not name:
            raise ValueError("Runtime stage names must be nonempty strings")

        self._synchronize()
        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            self._synchronize()
            elapsed_ns = time.perf_counter_ns() - start_ns
            self.stages.append(
                {
                    "name": name,
                    "seconds": elapsed_ns / 1_000_000_000,
                }
            )

    @property
    def articulation_seconds(self) -> float:
        return math.fsum(stage["seconds"] for stage in self.stages)


@contextmanager
def runtime_stage(name: str):
    """Record a named stage when a shape recorder is active."""
    recorder = _ACTIVE_RECORDER.get()
    if recorder is None:
        yield
        return
    with recorder.stage(name):
        yield


def new_run_id() -> str:
    return str(uuid.uuid4())


def make_runtime_record(
    *,
    run_id: str,
    pipeline: str,
    shape_path,
    lifecycle_seconds: float,
    recorder: ShapeRuntimeRecorder,
):
    resolved_shape_path = Path(shape_path).expanduser().resolve()
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "run_id": str(run_id),
        "pipeline": str(pipeline),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "shape": {
            "name": resolved_shape_path.stem,
            "path": str(resolved_shape_path),
        },
        "lifecycle_seconds": float(lifecycle_seconds),
        "articulation_seconds": recorder.articulation_seconds,
        "stages": [dict(stage) for stage in recorder.stages],
    }


def append_runtime_record(path, record):
    """Append one durable JSONL record without interleaving concurrent writers."""
    import fcntl

    _validate_runtime_record(record, line_number=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")

    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written == 0:
                raise OSError("Could not append the runtime record")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _validate_runtime_record(record, line_number: int):
    location = f"line {line_number}" if line_number > 0 else "runtime record"
    if not isinstance(record, dict):
        raise ValueError(f"{location}: expected a JSON object")
    version = record.get("schema_version")
    if version != RUNTIME_SCHEMA_VERSION:
        raise ValueError(f"{location}: unsupported runtime schema version {version!r}")
    for field in ("run_id", "pipeline", "finished_at_utc"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise ValueError(f"{location}: missing {field}")

    shape = record.get("shape")
    if not isinstance(shape, dict) or not isinstance(shape.get("path"), str):
        raise ValueError(f"{location}: missing shape.path")

    for field in ("lifecycle_seconds", "articulation_seconds"):
        value = record.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{location}: invalid {field}")

    stages = record.get("stages")
    if not isinstance(stages, list):
        raise ValueError(f"{location}: stages must be a list")
    for stage_index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ValueError(f"{location}: stage {stage_index} must be an object")
        name = stage.get("name")
        seconds = stage.get("seconds")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{location}: stage {stage_index} has an invalid name")
        if (
            not isinstance(seconds, (int, float))
            or isinstance(seconds, bool)
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ValueError(f"{location}: stage {stage_index} has invalid seconds")


def read_runtime_records(path) -> list[dict]:
    """Read valid JSONL records, warning about malformed or truncated lines."""
    records = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                warnings.warn(
                    f"Skipping malformed runtime log line {line_number}: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            try:
                _validate_runtime_record(record, line_number)
            except ValueError as error:
                if "unsupported runtime schema version" in str(error):
                    raise
                warnings.warn(
                    f"Skipping invalid runtime log record: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            records.append(record)
    return records


def select_runtime_records(
    records: Iterable[dict],
    *,
    latest_per_shape: bool = True,
    run_id: Optional[str] = None,
) -> list[dict]:
    """Filter records while preserving append order."""
    selected = [
        record for record in records if run_id is None or record.get("run_id") == run_id
    ]
    if not latest_per_shape:
        return selected

    last_index_by_path = {
        record["shape"]["path"]: index for index, record in enumerate(selected)
    }
    return [
        record
        for index, record in enumerate(selected)
        if last_index_by_path[record["shape"]["path"]] == index
    ]


def runtime_records_to_frames(records: Iterable[dict]):
    """Convert generic runtime records to record, long-stage, and wide tables."""
    import pandas as pd

    records = list(records)
    record_rows = []
    stage_rows = []
    stage_order = []

    for record_index, record in enumerate(records):
        base = {
            "record_index": record_index,
            "run_id": record.get("run_id"),
            "pipeline": record.get("pipeline"),
            "finished_at_utc": record.get("finished_at_utc"),
            "shape_name": record["shape"].get("name"),
            "shape_path": record["shape"]["path"],
            "lifecycle_seconds": record.get("lifecycle_seconds"),
            "articulation_seconds": record.get("articulation_seconds"),
        }
        record_rows.append(base)
        for stage_index, stage in enumerate(record["stages"]):
            name = stage["name"]
            if name not in stage_order:
                stage_order.append(name)
            stage_rows.append(
                {
                    **base,
                    "stage_index": stage_index,
                    "stage": name,
                    "seconds": stage["seconds"],
                }
            )

    record_frame = pd.DataFrame(record_rows)
    stage_frame = pd.DataFrame(stage_rows)
    if stage_frame.empty:
        wide_frame = record_frame.copy()
        return record_frame, stage_frame, wide_frame

    stage_values = stage_frame.pivot_table(
        index="record_index",
        columns="stage",
        values="seconds",
        aggfunc="sum",
        sort=False,
    )
    stage_values = stage_values.reindex(columns=stage_order)
    wide_frame = record_frame.merge(
        stage_values,
        left_on="record_index",
        right_index=True,
        how="left",
    )
    return record_frame, stage_frame, wide_frame
