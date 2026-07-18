"""Automated LSL parameter sweep.

Launches edf_to_lsl_stream.py with different configurations in background,
discovers each stream via pylsl, records success/failure, and writes a
structured sweep summary JSON.

Usage:
  python work/lsl_param_sweep.py --edf path/to/SC4001E0-PSG.edf --out outputs/runs/day10/lsl_param_sweep_results.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SWEEP_CONFIGS: list[dict] = [
    {
        "label": "1ch_100Hz_SleepEDF_raw_sample",
        "edf_channels": "EEG Fpz-Cz",
        "lsl_labels": "EEG_Fpz_Cz",
        "sample_rate": 100,
        "stream_name": "Sweep_1ch_100Hz",
        "push_mode": "sample",
        "expected_channels": 1,
    },
    {
        "label": "2ch_100Hz_SleepEDF_orig_sample",
        "edf_channels": "EEG Fpz-Cz,EEG Pz-Oz",
        "lsl_labels": "Fpz_Cz,Pz_Oz",
        "sample_rate": 100,
        "stream_name": "Sweep_2ch_100Hz",
        "push_mode": "sample",
        "expected_channels": 2,
    },
    {
        "label": "4ch_256Hz_Muse_labels_sample",
        "edf_channels": "EEG Fpz-Cz,EEG Pz-Oz",
        "lsl_labels": "TP9,AF7,AF8,TP10",
        "sample_rate": 256,
        "stream_name": "Sweep_4ch_256Hz",
        "push_mode": "sample",
        "expected_channels": 4,
    },
    {
        "label": "4ch_256Hz_Muse_labels_chunk",
        "edf_channels": "EEG Fpz-Cz,EEG Pz-Oz",
        "lsl_labels": "TP9,AF7,AF8,TP10",
        "sample_rate": 256,
        "stream_name": "Sweep_4ch_256Hz_chunk",
        "push_mode": "chunk",
        "expected_channels": 4,
    },
    {
        "label": "4ch_128Hz_Muse_labels_sample",
        "edf_channels": "EEG Fpz-Cz,EEG Pz-Oz",
        "lsl_labels": "TP9,AF7,AF8,TP10",
        "sample_rate": 128,
        "stream_name": "Sweep_4ch_128Hz",
        "push_mode": "sample",
        "expected_channels": 4,
    },
]


def discover_streams(wait_sec: float, stream_type: str | None = None) -> list[dict]:
    """Discover local LSL streams via pylsl, returning structured info."""
    from pylsl import resolve_streams

    deadline = time.time() + wait_sec
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


def run_one_sweep(config: dict, edf_path: str, stream_seconds: float, discover_seconds: float, python_exe: str) -> dict:
    """Start a LSL stream in background, discover it, then kill the stream."""
    label = config["label"]
    stream_name = config["stream_name"]
    t0 = time.time()

    script_dir = Path(__file__).resolve().parent
    edf_script = script_dir / "edf_to_lsl_stream.py"

    cmd = [
        python_exe,
        str(edf_script),
        "--edf", edf_path,
        "--channels", config["edf_channels"],
        "--lsl-labels", config["lsl_labels"],
        "--sample-rate", str(config["sample_rate"]),
        "--stream-name", stream_name,
        "--minutes", str(stream_seconds / 60.0),
        "--push-mode", config["push_mode"],
    ]

    result: dict = {
        "label": label,
        "config": config,
        "attempted_at": datetime.now(timezone.utc).isoformat(),
        "stream_started": False,
        "discover_success": False,
        "stream_count_found": 0,
        "streams_found": [],
        "matched_stream": None,
        "error": "",
        "duration_sec": 0.0,
    }

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Allow the stream outlet to start and register with the network.
        time.sleep(2.0)

        # Check if the process is still alive.
        if proc.poll() is not None:
            stderr_text = proc.stderr.read() if proc.stderr else ""
            result["error"] = f"Stream process exited early (rc={proc.returncode}): {stderr_text[-500:]}"
            return result

        result["stream_started"] = True

        # Discover.
        streams = discover_streams(discover_seconds, stream_type="EEG")
        matching = [s for s in streams if s["name"] == stream_name]
        result["streams_found"] = streams
        result["stream_count_found"] = len(matching)  # count exact name matches
        result["discover_success"] = len(matching) > 0
        if matching:
            result["matched_stream"] = matching[0]

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            except Exception:
                pass

    result["duration_sec"] = round(time.time() - t0, 2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep LSL stream parameters and record discover results.")
    parser.add_argument("--edf", required=True, help="Path to PSG EDF to use for streaming.")
    parser.add_argument("--out", default="outputs/runs/day10/lsl_param_sweep_results.json", help="Output JSON path.")
    parser.add_argument("--stream-seconds", type=float, default=15.0, help="Seconds to stream per config.")
    parser.add_argument("--discover-seconds", type=float, default=5.0, help="Seconds to discover after stream starts.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use for subprocess.")
    args = parser.parse_args()

    edf_path = Path(args.edf)
    if not edf_path.exists():
        print(f"ERROR: EDF file not found: {edf_path}")
        sys.exit(1)

    timestamp = datetime.now(timezone.utc).isoformat()
    results: list[dict] = []

    print(f"Running {len(SWEEP_CONFIGS)} LSL parameter sweep(s) on {edf_path.name}")
    print(f"Stream duration per config: {args.stream_seconds:.0f}s, discover window: {args.discover_seconds:.0f}s")
    print("-" * 60)

    for i, config in enumerate(SWEEP_CONFIGS, 1):
        print(f"[{i}/{len(SWEEP_CONFIGS)}] {config['label']} ... ", end="", flush=True)
        result = run_one_sweep(config, str(edf_path), args.stream_seconds, args.discover_seconds, args.python)
        results.append(result)

        if result["discover_success"]:
            s = result["matched_stream"]
            print(f"OK ({s['channel_count']}ch, {s['sample_rate']:.0f}Hz, {result['duration_sec']:.1f}s)")
        elif result["error"]:
            print(f"ERROR: {result['error'][:80]}")
        else:
            print(f"NOT FOUND ({result['stream_count_found']} stream(s) total, none matched)")

    # Build summary.
    summary: dict = {
        "sweep_timestamp": timestamp,
        "edf_path": str(edf_path),
        "stream_seconds_per_config": args.stream_seconds,
        "discover_seconds_per_config": args.discover_seconds,
        "total_configs": len(SWEEP_CONFIGS),
        "successful_configs": sum(1 for r in results if r["discover_success"]),
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSweep complete. {summary['successful_configs']}/{summary['total_configs']} configs discovered.")
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    main()
