"""Check whether local LSL EEG streams are discoverable.

Use this before blaming NeuroSkill. If this script cannot discover the stream,
the problem is likely pylsl/liblsl/network/firewall rather than NeuroSkill.
"""

from __future__ import annotations

import argparse
import json
import time


def discover(seconds: float, stream_type: str | None) -> list[dict]:
    from pylsl import resolve_streams

    deadline = time.time() + seconds
    found: dict[str, dict] = {}
    while time.time() < deadline:
        for stream in resolve_streams(wait_time=1.0):
            if stream_type and stream.type() != stream_type:
                continue
            key = stream.source_id() or f"{stream.name()}::{stream.hostname()}"
            found[key] = {
                "name": stream.name(),
                "type": stream.type(),
                "channel_count": stream.channel_count(),
                "sample_rate": stream.nominal_srate(),
                "source_id": stream.source_id(),
                "hostname": stream.hostname(),
            }
    return list(found.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover local LSL streams.")
    parser.add_argument("--seconds", type=float, default=5.0, help="Discovery duration.")
    parser.add_argument("--type", default="EEG", help="Filter by LSL stream type. Use empty string for all.")
    args = parser.parse_args()

    stream_type = args.type or None
    streams = discover(args.seconds, stream_type)
    print(json.dumps({"count": len(streams), "streams": streams}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
