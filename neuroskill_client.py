#!/usr/bin/env python3
"""Day 1: NeuroSkill command-line client skeleton."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def save_json(payload: dict, out: str) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def call_status(port: int, timeout: float) -> dict:
    url = f"http://127.0.0.1:{port}/v1/status"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = raw
            return {"ok": True, "url": url, "response": body, "error": None}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "ok": False,
            "url": url,
            "response": None,
            "error": {"type": "connection_error", "message": str(getattr(exc, "reason", exc))},
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NeuroSkill client")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("status", "lsl-discover", "sleep"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--port", type=int, default=18444)
        subparser.add_argument("--out", required=True)
        subparser.add_argument("--timeout", type=float, default=3.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "status":
        result = call_status(args.port, args.timeout)
    else:
        result = {
            "ok": False,
            "url": None,
            "response": None,
            "error": {"type": "not_implemented", "message": f"{args.command} is a Day 1 command skeleton"},
        }
    result.update(
        {
            "command": args.command,
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        }
    )
    save_json(result, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
