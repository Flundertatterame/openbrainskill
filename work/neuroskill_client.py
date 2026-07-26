#!/usr/bin/env python3
"""Small, dependency-free NeuroSkill HTTP client.

Every command prints JSON and can persist the same payload with ``--out``.
Connection failures are normal results (exit code 1), not Python tracebacks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def json_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc.msg}") from exc


def request_json(
    method: str,
    url: str,
    *,
    timeout: float,
    token: str | None,
    body: Any = None,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    data = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")

    started = time.perf_counter()
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                parsed = raw
            return {
                "ok": 200 <= response.status < 300,
                "method": method,
                "url": url,
                "http_status": response.status,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                "response": parsed,
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            response: Any = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            response = raw
        return {
            "ok": False,
            "method": method,
            "url": url,
            "http_status": exc.code,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "response": response,
            "error": {"type": "http_error", "message": str(exc)},
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        error_type = "timeout" if isinstance(reason, TimeoutError) else "connection_error"
        return {
            "ok": False,
            "method": method,
            "url": url,
            "http_status": None,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "response": None,
            "error": {"type": error_type, "message": str(reason)},
        }


def try_endpoints(args: argparse.Namespace, specs: list[tuple[str, str, Any]]) -> dict[str, Any]:
    attempts = []
    for method, path, body in specs:
        result = request_json(
            method,
            f"{args.base_url}{path}",
            timeout=args.timeout,
            token=args.token,
            body=body,
        )
        attempts.append(result)
        if result["ok"]:
            break
        # A reachable server returning an error may still support another known route.
    selected = next((item for item in attempts if item["ok"]), attempts[-1])
    return {
        "schema_version": "1.0",
        "command": args.command,
        "timestamp": now_iso(),
        "ok": selected["ok"],
        "url": selected["url"],
        "response": selected["response"],
        "error": selected["error"],
        "attempts": attempts,
    }


def command_specs(args: argparse.Namespace) -> list[tuple[str, str, Any]]:
    if args.command == "status":
        return [("GET", "/v1/status", None), ("POST", "/", {"command": "status"})]
    if args.command == "lsl-discover":
        return [("GET", "/v1/lsl/discover", None), ("GET", "/lsl/discover", None)]
    if args.command == "sleep":
        body = args.payload if args.payload is not None else {}
        return [("POST", "/v1/sleep", body), ("POST", "/sleep", body), ("POST", "/", {"command": "sleep", **body})]
    if args.command == "connect":
        body = args.payload if args.payload is not None else {}
        return [("POST", "/v1/lsl/connect", body), ("POST", "/lsl/connect", body), ("POST", "/", {"command": "connect", **body})]
    if args.command == "session":
        body = args.payload if args.payload is not None else {}
        return [("POST", "/v1/session", body), ("POST", "/session", body), ("POST", "/", {"command": "session", **body})]
    raise ValueError(f"unsupported command: {args.command}")


def write_result(result: dict[str, Any], out: str | None) -> None:
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")


def find_field(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = find_field(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_field(child, key)
            if found is not None:
                return found
    return None


def write_status_slim(result: dict[str, Any], out: str | None) -> None:
    if not out:
        return
    response = result.get("response")
    slim = {
        "schema_version": "1.0",
        "command": "status",
        "timestamp": result.get("timestamp"),
        "ok": result.get("ok", False),
        "state": find_field(response, "state"),
        "eeg_samples": find_field(response, "eeg_samples"),
        "session": find_field(response, "session"),
        "discovered_devices": find_field(response, "discovered_devices"),
        "source": result.get("url"),
        "error": result.get("error"),
    }
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(slim, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def add_common(parser: argparse.ArgumentParser, *, payload: bool = False) -> None:
    parser.add_argument("--port", type=int, default=18444, help="NeuroSkill daemon port (default: 18444)")
    parser.add_argument("--host", default="127.0.0.1", help="NeuroSkill daemon host")
    parser.add_argument("--timeout", type=float, default=3.0, help="request timeout in seconds")
    parser.add_argument("--token", default=os.getenv("NEUROSKILL_TOKEN"), help="Bearer token; defaults to NEUROSKILL_TOKEN")
    parser.add_argument("--out", help="path to output JSON")
    if payload:
        parser.add_argument("--payload", type=json_value, help="JSON request object")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call NeuroSkill daemon and always return structured JSON")
    subparsers = parser.add_subparsers(dest="command", required=True)
    status = subparsers.add_parser("status")
    add_common(status)
    status.add_argument("--slim-out", help="optional normalized status JSON path")
    add_common(subparsers.add_parser("lsl-discover"))
    for name in ("sleep", "connect", "session"):
        add_common(subparsers.add_parser(name), payload=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.base_url = f"http://{args.host}:{args.port}"
    try:
        result = try_endpoints(args, command_specs(args))
    except Exception as exc:  # Last-resort JSON boundary for automation callers.
        result = {
            "schema_version": "1.0",
            "command": args.command,
            "timestamp": now_iso(),
            "ok": False,
            "url": args.base_url,
            "response": None,
            "error": {"type": "client_error", "message": str(exc)},
            "attempts": [],
        }
    try:
        write_result(result, args.out)
        if args.command == "status":
            write_status_slim(result, args.slim_out)
    except Exception as exc:
        # Keep the CLI JSON-only even when an output path is invalid or unwritable.
        output_error = {
            "schema_version": "1.0",
            "command": args.command,
            "timestamp": now_iso(),
            "ok": False,
            "url": getattr(args, "base_url", None),
            "response": None,
            "error": {"type": "output_error", "message": str(exc)},
            "attempts": result.get("attempts", []),
        }
        print(json.dumps(output_error, ensure_ascii=False, indent=2))
        return 1
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
