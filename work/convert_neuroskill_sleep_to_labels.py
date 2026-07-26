#!/usr/bin/env python3
"""Convert NeuroSkill ``epochs[]`` JSON to ``start_sec,stage`` CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
    if isinstance(value, (int, float)) and int(value) in range(5):
        return int(value)
    key = str(value).strip().upper().replace("SLEEP STAGE ", "").replace("STAGE ", "")
    if key in STAGE_MAP:
        return STAGE_MAP[key]
    raise ValueError(f"unsupported sleep stage: {value!r}")


def convert(payload: Any, epoch_sec: float) -> list[dict[str, Any]]:
    epochs = find_epochs(payload)
    if epochs is None:
        raise ValueError("input JSON does not contain an epochs[] array")
    rows = []
    for index, epoch in enumerate(epochs):
        if isinstance(epoch, dict):
            stage_value = next((epoch[key] for key in ("stage", "label", "predicted_stage", "prediction") if key in epoch), None)
            if stage_value is None:
                raise ValueError(f"epoch {index} has no stage field")
            start_sec = next((epoch[key] for key in ("start_sec", "start", "time_sec", "timestamp_sec") if key in epoch), index * epoch_sec)
        else:
            stage_value = epoch
            start_sec = index * epoch_sec
        rows.append({"start_sec": float(start_sec), "stage": normalize_stage(stage_value)})
    return rows


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
