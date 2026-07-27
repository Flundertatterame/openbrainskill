#!/usr/bin/env python3
"""Convert NeuroSkill ``epochs[]`` JSON to ``start_sec,stage`` CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


STAGE_MAP = {
    "W": 0,
    "WAKE": 0,
    "N1": 1,
    "1": 1,
    "N2": 2,
    "2": 2,
    "N3": 3,
    "N4": 3,
    "3": 3,
    "4": 3,
    "R": 4,
    "REM": 4,
}


def find_epochs(value: Any) -> list[Any] | None:
    if isinstance(value, dict):
        epochs = value.get("epochs")
        if isinstance(epochs, list):
            return epochs
        for child in value.values():
            found = find_epochs(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_epochs(child)
            if found is not None:
                return found
    return None


def normalize_stage(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not a sleep stage")
    if isinstance(value, (int, float)):
        numeric = int(value)
        if numeric == 5:
            return 4
        if numeric in range(5):
            return numeric
    key = str(value).strip().upper().replace("SLEEP STAGE ", "").replace("STAGE ", "")
    if key in STAGE_MAP:
        return STAGE_MAP[key]
    raise ValueError(f"unsupported sleep stage: {value!r}")


def pick_stage(values: list[int]) -> int:
    counts = Counter(values)
    best_count = max(counts.values())
    return min(stage for stage, count in counts.items() if count == best_count)


def to_float(value: Any, *, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric: {value!r}") from exc


def convert(payload: Any, epoch_sec: float) -> list[dict[str, Any]]:
    epochs = find_epochs(payload)
    if epochs is None:
        raise ValueError("input JSON does not contain an epochs[] array")

    inferred_epoch = None
    if isinstance(payload, dict):
        inferred_epoch = payload.get("epoch_secs", payload.get("epoch_sec"))
    if inferred_epoch is not None:
        epoch_sec = to_float(inferred_epoch, name="epoch_secs")

    utc_rows: dict[float, list[int]] = {}
    sequential_rows = []
    for index, epoch in enumerate(epochs):
        if isinstance(epoch, dict):
            stage_value = next((epoch[key] for key in ("stage", "label", "predicted_stage", "prediction") if key in epoch), None)
            if stage_value is None:
                raise ValueError(f"epoch {index} has no stage field")
            stage = normalize_stage(stage_value)
            if "utc" in epoch:
                utc = to_float(epoch["utc"], name="utc")
                utc_rows.setdefault(utc, []).append(stage)
                continue
            start_sec = next((epoch[key] for key in ("start_sec", "start", "time_sec", "timestamp_sec") if key in epoch), index * epoch_sec)
        else:
            stage = normalize_stage(epoch)
            start_sec = index * epoch_sec
        sequential_rows.append({"start_sec": float(start_sec), "stage": stage})

    if utc_rows:
        base_utc = min(utc_rows)
        buckets: dict[int, list[int]] = {}
        for utc, stages in sorted(utc_rows.items()):
            second_stage = pick_stage(stages)
            bucket = int(math.floor((utc - base_utc) / 30.0))
            buckets.setdefault(bucket, []).append(second_stage)
        return [
            {"start_sec": float(bucket * 30), "stage": pick_stage(stages)}
            for bucket, stages in sorted(buckets.items())
        ]

    return sequential_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert NeuroSkill epochs[] JSON to label CSV")
    parser.add_argument("--input", required=True, help="NeuroSkill sleep JSON")
    parser.add_argument("--out", required=True, help="output CSV path")
    parser.add_argument("--epoch-sec", type=float, default=30.0)
    parser.add_argument("--error-out", help="optional structured error JSON path")
    args = parser.parse_args()

    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        rows = convert(payload, args.epoch_sec)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=["start_sec", "stage"])
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps({"ok": True, "input": args.input, "out": args.out, "rows": len(rows)}, ensure_ascii=False))
        return 0
    except Exception as exc:
        error = {"ok": False, "input": args.input, "out": args.out, "error": {"type": type(exc).__name__, "message": str(exc)}}
        if args.error_out:
            path = Path(args.error_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(error, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(error, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
