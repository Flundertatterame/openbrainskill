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
import uuid
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
        "channel_policy": "strict",
    },
    {
        "label": "2ch_100Hz_SleepEDF_orig_sample",
        "edf_channels": "EEG Fpz-Cz,EEG Pz-Oz",
        "lsl_labels": "Fpz_Cz,Pz_Oz",
        "sample_rate": 100,
        "stream_name": "Sweep_2ch_100Hz",
        "push_mode": "sample",
        "expected_channels": 2,
        "channel_policy": "strict",
    },
    {
        "label": "4ch_256Hz_Muse_labels_sample",
        "edf_channels": "EEG Fpz-Cz,EEG Pz-Oz",
        "lsl_labels": "TP9,AF7,AF8,TP10",
        "sample_rate": 256,
        "stream_name": "Sweep_4ch_256Hz",
        "push_mode": "sample",
        "expected_channels": 4,
        "channel_policy": "duplicate",
    },
    {
        "label": "4ch_256Hz_Muse_labels_chunk",
        "edf_channels": "EEG Fpz-Cz,EEG Pz-Oz",
        "lsl_labels": "TP9,AF7,AF8,TP10",
        "sample_rate": 256,
        "stream_name": "Sweep_4ch_256Hz_chunk",
        "push_mode": "chunk",
        "expected_channels": 4,
        "channel_policy": "duplicate",
    },
    {
        "label": "4ch_128Hz_Muse_labels_sample",
        "edf_channels": "EEG Fpz-Cz,EEG Pz-Oz",
        "lsl_labels": "TP9,AF7,AF8,TP10",
        "sample_rate": 128,
        "stream_name": "Sweep_4ch_128Hz",
        "push_mode": "sample",
        "expected_channels": 4,
        "channel_policy": "duplicate",
    },
]


def stream_to_dict(stream) -> dict:
    return {
        "name": stream.name(),
        "type": stream.type(),
        "channel_count": stream.channel_count(),
        "sample_rate": stream.nominal_srate(),
        "source_id": stream.source_id(),
        "hostname": stream.hostname(),
    }


def stream_matches(stream: dict, stream_name: str, expected_channels: int, expected_sample_rate: float) -> bool:
    return (
        stream["name"] == stream_name
        and stream["channel_count"] == expected_channels
        and abs(float(stream["sample_rate"]) - float(expected_sample_rate)) < 1e-3
    )


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
            found[key] = stream_to_dict(stream)
    return list(found.values())


def wait_for_stream_start(
    proc: subprocess.Popen,
    stream_name: str,
    expected_channels: int,
    expected_sample_rate: float,
    startup_timeout: float,
    stream_type: str | None = "EEG",
) -> tuple[list[dict], dict | None, float, str]:
    """Wait until the target LSL outlet is actually discoverable."""
    from pylsl import resolve_streams

    deadline = time.time() + startup_timeout
    found: dict[str, dict] = {}
    start_time = time.time()

    while time.time() < deadline:
        if proc.poll() is not None:
            elapsed = round(time.time() - start_time, 2)
            return list(found.values()), None, elapsed, f"Stream process exited early (rc={proc.returncode})"

        wait_time = min(1.0, max(0.1, deadline - time.time()))
        for stream in resolve_streams(wait_time=wait_time):
            if stream_type and stream.type() != stream_type:
                continue
            stream_info = stream_to_dict(stream)
            key = stream_info["source_id"] or f"{stream_info['name']}::{stream_info['hostname']}"
            found[key] = stream_info
            if stream_matches(stream_info, stream_name, expected_channels, expected_sample_rate):
                elapsed = round(time.time() - start_time, 2)
                return list(found.values()), stream_info, elapsed, ""

    elapsed = round(time.time() - start_time, 2)
    return list(found.values()), None, elapsed, f"Timed out after {startup_timeout:.1f}s waiting for stream registration"


def run_one_sweep(
    config: dict,
    edf_path: str,
    stream_seconds: float,
    discover_seconds: float,
    startup_timeout: float,
    python_exe: str,
    neuroskill_port: int | None,
    neuroskill_timeout: float,
    neuroskill_dir: Path,
) -> dict:
    """Start a LSL stream in background, discover it, then kill the stream."""
    label = config["label"]
    stream_name = config["stream_name"]
    t0 = time.time()

    script_dir = Path(__file__).resolve().parent
    edf_script = script_dir / "edf_to_lsl_stream.py"
    source_id = f"sweep-{label.lower()}-{uuid.uuid4().hex[:8]}"

    cmd = [
        python_exe,
        str(edf_script),
        "--edf", edf_path,
        "--channels", config["edf_channels"],
        "--lsl-labels", config["lsl_labels"],
        "--sample-rate", str(config["sample_rate"]),
        "--stream-name", stream_name,
        "--source-id", source_id,
        "--channel-policy", config["channel_policy"],
        "--minutes", str(stream_seconds / 60.0),
        "--push-mode", config["push_mode"],
    ]

    result: dict = {
        "label": label,
        "config": config,
        "attempted_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "source_id": source_id,
        "stream_started": False,
        "discover_success": False,
        "stream_count_found": 0,
        "matching_stream_count": 0,
        "streams_found": [],
        "matched_stream": None,
        "startup_timeout_sec": startup_timeout,
        "startup_elapsed_sec": 0.0,
        "error": "",
        "duration_sec": 0.0,
        "neuroskill_checked": neuroskill_port is not None,
        "neuroskill_discover_success": None,
        "neuroskill_result_path": "",
    }

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        startup_streams, matched, startup_elapsed, startup_error = wait_for_stream_start(
            proc=proc,
            stream_name=stream_name,
            expected_channels=int(config["expected_channels"]),
            expected_sample_rate=float(config["sample_rate"]),
            startup_timeout=startup_timeout,
            stream_type="EEG",
        )
        result["streams_found"] = startup_streams
        result["stream_count_found"] = len(startup_streams)
        result["startup_elapsed_sec"] = startup_elapsed

        if proc.poll() is not None:
            stderr_text = proc.stderr.read() if proc.stderr else ""
            result["error"] = f"{startup_error}: {stderr_text[-500:]}"
            return result

        if matched is None:
            result["error"] = startup_error
            return result
        else:
            result["stream_started"] = True
            result["discover_success"] = True
            result["status"] = "success"
            result["matched_stream"] = matched

        # Discover again after startup so the result also captures nearby streams
        # during a stable post-registration window.
        streams = discover_streams(discover_seconds, stream_type="EEG")
        combined_streams = {s["source_id"] or f"{s['name']}::{s['hostname']}": s for s in startup_streams}
        combined_streams.update({s["source_id"] or f"{s['name']}::{s['hostname']}": s for s in streams})
        streams = list(combined_streams.values())
        matching = [
            s
            for s in streams
            if stream_matches(s, stream_name, int(config["expected_channels"]), float(config["sample_rate"]))
        ]
        result["streams_found"] = streams
        result["stream_count_found"] = len(streams)
        result["matching_stream_count"] = len(matching)
        result["discover_success"] = len(matching) > 0
        if matching:
            result["matched_stream"] = matching[0]
            result["status"] = "success"

        if neuroskill_port is not None:
            client_script = script_dir / "neuroskill_client.py"
            result_path = neuroskill_dir / f"{label}.json"
            client_cmd = [
                python_exe, str(client_script), "lsl-discover",
                "--port", str(neuroskill_port),
                "--timeout", str(neuroskill_timeout),
                "--out", str(result_path),
            ]
            completed = subprocess.run(client_cmd, text=True, capture_output=True, check=False)
            result["neuroskill_result_path"] = str(result_path)
            if result_path.exists():
                try:
                    neuroskill_result = json.loads(result_path.read_text(encoding="utf-8"))
                    result["neuroskill_discover_success"] = bool(neuroskill_result.get("ok", False))
                except json.JSONDecodeError:
                    result["neuroskill_discover_success"] = False
                    result["error"] = f"NeuroSkill produced invalid JSON: {completed.stderr[-300:]}"
            else:
                result["neuroskill_discover_success"] = False
                result["error"] = f"NeuroSkill client did not produce output (rc={completed.returncode}): {completed.stderr[-300:]}"

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
    parser.add_argument(
        "--out",
        default="outputs/runs/day15/lsl_sweep_summary.json",
        help="Output JSON summary path.",
    )
    parser.add_argument("--stream-seconds", type=float, default=15.0, help="Seconds to stream per config.")
    parser.add_argument("--discover-seconds", type=float, default=5.0, help="Seconds to discover after stream starts.")
    parser.add_argument("--startup-timeout", type=float, default=30.0, help="Seconds to wait for EDF loading and LSL registration.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use for subprocess.")
    parser.add_argument("--neuroskill-port", type=int, default=None, help="Optionally test NeuroSkill LSL discovery for every running sweep stream.")
    parser.add_argument("--neuroskill-timeout", type=float, default=3.0, help="Per-config NeuroSkill discovery timeout in seconds.")
    args = parser.parse_args()

    edf_path = Path(args.edf)
    if not edf_path.exists():
        summary = {
            "status": "failed",
            "sweep_timestamp": datetime.now(timezone.utc).isoformat(),
            "edf_path": str(edf_path),
            "total_configs": len(SWEEP_CONFIGS),
            "successful_configs": 0,
            "failed_configs": len(SWEEP_CONFIGS),
            "total_duration_sec": 0.0,
            "error": {
                "error_type": "FileNotFoundError",
                "message": f"EDF file not found: {edf_path}",
                "command_hint": "Pass an existing PSG EDF path to --edf.",
            },
            "results": [],
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"ERROR: EDF file not found: {edf_path}")
        print(f"Failure summary written to: {out_path}")
        sys.exit(1)

    timestamp = datetime.now(timezone.utc).isoformat()
    results: list[dict] = []
    neuroskill_dir = Path(args.out).parent / "lsl_sweep_neuroskill"
    if args.neuroskill_port is not None:
        neuroskill_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(SWEEP_CONFIGS)} LSL parameter sweep(s) on {edf_path.name}")
    print(
        f"Stream duration per config: {args.stream_seconds:.0f}s, "
        f"startup timeout: {args.startup_timeout:.0f}s, "
        f"discover window: {args.discover_seconds:.0f}s"
    )
    print("-" * 60)

    for i, config in enumerate(SWEEP_CONFIGS, 1):
        print(f"[{i}/{len(SWEEP_CONFIGS)}] {config['label']} ... ", end="", flush=True)
        result = run_one_sweep(
            config,
            str(edf_path),
            args.stream_seconds,
            args.discover_seconds,
            args.startup_timeout,
            args.python,
            args.neuroskill_port,
            args.neuroskill_timeout,
            neuroskill_dir,
        )
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
        "status": "completed",
        "sweep_timestamp": timestamp,
        "edf_path": str(edf_path),
        "stream_seconds_per_config": args.stream_seconds,
        "discover_seconds_per_config": args.discover_seconds,
        "startup_timeout_per_config": args.startup_timeout,
        "total_configs": len(SWEEP_CONFIGS),
        "successful_configs": sum(1 for r in results if r["discover_success"]),
        "failed_configs": sum(1 for r in results if not r["discover_success"]),
        "total_duration_sec": round(sum(float(r["duration_sec"]) for r in results), 2),
        "neuroskill_checked": args.neuroskill_port is not None,
        "neuroskill_port": args.neuroskill_port,
        "neuroskill_successful_configs": sum(1 for r in results if r["neuroskill_discover_success"] is True),
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSweep complete. {summary['successful_configs']}/{summary['total_configs']} configs discovered.")
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    main()
