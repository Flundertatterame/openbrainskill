"""End-to-end LSL test: EDF → LSL outlet → LSL inlet discovery.

One command verifies the full pipeline:
  1. Start edf_to_lsl_stream.py in background.
  2. Poll lsl_resolve_check.py until the stream is discovered.
  3. Validate stream metadata (name, type, channel count, sample rate).
  4. Kill the background stream and report PASS / FAIL.

Usage:
  "C:/Users/shen/anaconda3/envs/brainfusion/python.exe" work/e2e_lsl_test.py

  # Or with a custom EDF and shorter stream:
  "C:/Users/shen/anaconda3/envs/brainfusion/python.exe" work/e2e_lsl_test.py --edf path/to/file.edf --seconds 15
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EDF_STREAM_SCRIPT = ROOT / "edf_to_lsl_stream.py"
DISCOVER_SCRIPT = ROOT / "lsl_resolve_check.py"
DEFAULT_EDF = ROOT.parent.parent / "BrainFusion" / "SleepEDF" / "SC4001E0-PSG.edf"
PYTHON_EXE = r"C:\Users\shen\anaconda3\envs\brainfusion\python.exe"


def _resolve_python() -> str:
    """Pick the Python interpreter that has mne + pylsl available."""
    if Path(PYTHON_EXE).exists():
        return PYTHON_EXE
    # Fallback: try the current interpreter.
    print("WARNING: brainfusion Python not found at expected path, using sys.executable as fallback.",
          file=sys.stderr)
    return sys.executable


def _run_json(args: list[str], timeout: float = 30) -> tuple[int, str, str]:
    completed = subprocess.run(
        args, text=True, capture_output=True, timeout=timeout, check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def discover_streams(python: str, seconds: float = 5) -> list[dict]:
    """Return all LSL EEG streams found by lsl_resolve_check."""
    code, stdout, stderr = _run_json(
        [python, str(DISCOVER_SCRIPT), "--seconds", str(seconds), "--type", ""],
    )
    if code != 0:
        print(f"  discover returned exit code {code}: {stderr.strip()[-200:]}")
        return []
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return []
    return payload.get("streams", [])


def find_by_name(streams: list[dict], name: str) -> dict | None:
    for s in streams:
        if s.get("name") == name:
            return s
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot E2E LSL pipeline test.")
    parser.add_argument("--edf", default=str(DEFAULT_EDF), help="Path to PSG EDF file.")
    parser.add_argument("--seconds", type=float, default=30, help="Seconds to stream.")
    parser.add_argument("--stream-name", default="E2E_Test", help="LSL stream name for this test.")
    parser.add_argument("--expected-channels", type=int, default=2, help="Expected physical channel count for strict mode.")
    parser.add_argument("--startup-timeout", type=float, default=60,
                        help="Max seconds to wait for the LSL outlet to appear.")
    parser.add_argument("--discover-seconds", type=float, default=5,
                        help="Seconds to scan for LSL streams.")
    args = parser.parse_args()

    python = _resolve_python()
    if not Path(python).exists():
        print(f"FATAL: Python interpreter not found: {python}")
        sys.exit(1)

    if not Path(args.edf).exists():
        print(f"FATAL: EDF file not found: {args.edf}")
        sys.exit(1)

    print("=" * 60)
    print("E2E LSL Pipeline Test")
    print(f"  Python  : {python}")
    print(f"  EDF     : {args.edf}")
    print(f"  Duration: {args.seconds:.0f}s")
    print(f"  Stream  : {args.stream_name}")
    print("=" * 60)

    # 1. Start the stream process.
    minutes = max(0.05, args.seconds / 60.0)
    stream_cmd = [
        python, str(EDF_STREAM_SCRIPT),
        "--edf", args.edf,
        "--stream-name", args.stream_name,
        "--minutes", str(minutes),
        "--push-mode", "chunk",
    ]
    print(f"\n[1/4] Starting LSL stream ...")
    proc = subprocess.Popen(
        stream_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 2. Poll until the stream appears.
    print(f"[2/4] Waiting up to {args.startup_timeout:.0f}s for stream '{args.stream_name}' ...")
    deadline = time.time() + args.startup_timeout
    found_stream: dict | None = None
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        if proc.poll() is not None:
            print(f"  FAIL: stream process exited early (rc={proc.returncode})")
            proc = None
            break

        streams = discover_streams(python, seconds=min(2, max(0.5, deadline - time.time())))
        found_stream = find_by_name(streams, args.stream_name)
        if found_stream is not None:
            print(f"  found after {attempt} attempt(s) ({len(streams)} total stream(s))")
            break
        print(f"  attempt {attempt}: not found yet, retrying ...")

    if found_stream is None:
        print("\nFAIL: LSL stream was not discovered.")
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        sys.exit(1)

    # 3. Validate metadata.
    print(f"\n[3/4] Validating stream metadata ...")
    ok = True
    checks = [
        ("type", "EEG"),
        ("channel_count", args.expected_channels),
        ("sample_rate", 256.0),
    ]
    for field, expected in checks:
        actual = found_stream.get(field)
        match = actual == expected
        if isinstance(expected, float):
            match = abs(float(actual or 0) - expected) < 1.0
        status = "OK" if match else f"MISMATCH (expected {expected}, got {actual})"
        if not match:
            ok = False
        print(f"  {field:>15}: {actual:<10} {status}")

    print(f"  {'hostname':>15}: {found_stream.get('hostname', '?')}")
    print(f"  {'source_id':>15}: {found_stream.get('source_id', '?')}")

    # 4. Clean up.
    print(f"\n[4/4] Cleaning up ...")
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
            print("  stream process terminated.")
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            print("  stream process killed.")

    if ok:
        print("\n" + "=" * 60)
        print("RESULT: PASS")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("RESULT: FAIL (metadata mismatch)")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
