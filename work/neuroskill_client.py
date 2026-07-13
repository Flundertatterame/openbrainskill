"""Small NeuroSkill HTTP helper for the MVP.

The exact local API may differ by NeuroSkill version. This client records raw
responses and errors as JSON so the Agent can diagnose failures instead of
silently pretending the backend worked.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def request_json(method: str, url: str, body: dict[str, Any] | None = None, token: str | None = None) -> dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = {"raw_text": text}
            return {"ok": True, "status": resp.status, "url": url, "response": parsed}
    except urllib.error.HTTPError as e:
        return {"ok": False, "url": url, "status": e.code, "error": e.read().decode("utf-8", errors="replace")}
    except Exception as e:
        return {"ok": False, "url": url, "error": type(e).__name__, "message": str(e)}


def write_result(path: str | None, payload: dict[str, Any]) -> None:
    if path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroSkill HTTP helper.")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ["status", "lsl-discover", "sleep"]:
        p = sub.add_parser(name)
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=18444)
        p.add_argument("--token", default="")
        p.add_argument("--out", default="")

    args = parser.parse_args()
    base = f"http://{args.host}:{args.port}"

    if args.command == "status":
        result = request_json("GET", f"{base}/v1/status", token=args.token or None)
        if not result.get("ok"):
            result = request_json("POST", f"{base}/", {"command": "status"}, token=args.token or None)
    elif args.command == "lsl-discover":
        result = request_json("GET", f"{base}/v1/lsl/discover", token=args.token or None)
        if not result.get("ok"):
            result = request_json("GET", f"{base}/lsl/discover", token=args.token or None)
    else:
        result = request_json("POST", f"{base}/", {"command": "sleep"}, token=args.token or None)

    write_result(args.out, result)


if __name__ == "__main__":
    main()
