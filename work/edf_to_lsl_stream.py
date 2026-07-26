"""Stream an EDF file as a local LSL EEG outlet.

Default configuration:
  stream_type=EEG, source channel labels preserved, 256Hz, microvolts,
  push_mode=chunk, chunk_seconds=0.125, strict physical-channel policy.

Stability evidence:
  - Day 4: 3-way parameter sweep (1ch/100Hz, 4ch/256Hz Muse, 2ch/100Hz orig)
           all found by Python discover at 100% across 8s streams.
  - Day 6: sample vs chunk push_mode comparison — 2x2min streams, each
           discovered 7 times at 15s intervals, 100% discovery rate for both.
           chunk is now the default because it avoids bursty sample timestamps.
  - Day 7: 10-minute continuous push (4ch/256Hz/sample) — discovered once per
           minute, 100% discovery rate, zero disconnections, no srate drift.

Known alternative configurations (functional but not validated with NeuroSkill):
  - 2ch/100Hz with original Sleep-EDF channel names ("Fpz_Cz,Pz_Oz") — Python
    discover works, but NeuroSkill may expect Muse-style labels.
  - 1ch/100Hz with raw EDF channel name ("EEG_Fpz_Cz") — same caveat.
  - push_mode=sample — kept as an option and paced sample-by-sample.

Channel duplication is available only through --channel-policy duplicate and
is declared explicitly in the output metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FEATURE_COLUMNS = {"delta", "theta", "alpha", "beta", "gamma", "epoch", "sleep_stage", "stage"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path_value: str, payload: dict) -> None:
    """Write a JSON artifact when the caller requested one."""
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_source_manifest(path_value: str) -> dict:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Source manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("source_kind") != "waveform":
        raise ValueError("Source manifest must declare source_kind='waveform'; feature tables cannot be streamed as EEG.")
    return manifest


def command_hint(exc: Exception) -> str:
    """Give the caller one concrete next action for common adapter failures."""
    if isinstance(exc, FileNotFoundError):
        return "Verify the input path and pass exactly one of --edf, --from-npy, or --from-csv."
    if isinstance(exc, ImportError) or "not found in the current Python environment" in str(exc):
        return "Use the Python environment with mne and pylsl installed, then rerun this command."
    if isinstance(exc, ValueError):
        return "Check the input format, channel labels, and numeric stream parameters, then rerun with --dry-run."
    return "Rerun with --dry-run and inspect the input path, stream parameters, and saved error JSON."


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_edf_data(edf_path: Path, requested_channels: list[str], target_sfreq: float) -> tuple[np.ndarray, float, list[str]]:
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    import mne

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    available = raw.ch_names

    if requested_channels:
        missing = [ch for ch in requested_channels if ch not in available]
        if missing:
            raise ValueError(
                "Requested EDF channel(s) not found: "
                f"{', '.join(missing)}. Available channels: {', '.join(available)}"
            )
        selected = requested_channels
    else:
        eeg_like = [ch for ch in available if "EEG" in ch.upper()]
        selected = eeg_like[: min(2, len(eeg_like))] or available[:1]

    raw.pick(selected)
    if abs(float(raw.info["sfreq"]) - target_sfreq) > 1e-6:
        raw.resample(target_sfreq, npad="auto", verbose=False)

    data = raw.get_data().astype(np.float32)

    # MNE returns SI units for EDF physical dimensions when possible. Sleep-EDF
    # EEG is commonly represented in volts after loading, so convert small
    # amplitudes to microvolts for NeuroSkill's expected stream convention.
    if np.nanmedian(np.abs(data)) < 1e-3:
        data *= 1_000_000.0

    return data, float(raw.info["sfreq"]), selected


def load_from_npy(npy_path: Path, layout: str) -> tuple[np.ndarray, list[str], str]:
    """Load a waveform NPY and normalize it to (channels, samples)."""
    if not npy_path.exists():
        raise FileNotFoundError(f"NPY file not found: {npy_path}")
    data = np.load(npy_path).astype(np.float32)
    if layout == "auto":
        if data.ndim == 3:
            layout = "epochs-channels-samples"
        elif data.ndim == 2:
            layout = "channels-samples" if data.shape[0] <= 16 else "epochs-samples"
        else:
            raise ValueError(f"Expected a 2-D or 3-D waveform array, got shape {data.shape}")
    if layout == "channels-samples" and data.ndim == 2:
        return data, [f"EEG {index + 1}" for index in range(data.shape[0])], layout
    if layout == "epochs-samples" and data.ndim == 2:
        return data.reshape(1, -1), ["EEG 1"], layout
    if layout == "epochs-channels-samples" and data.ndim == 3:
        epochs, channels, samples = data.shape
        return data.transpose(1, 0, 2).reshape(channels, epochs * samples), [f"EEG {index + 1}" for index in range(channels)], layout
    raise ValueError(f"Input layout {layout!r} does not match NPY shape {data.shape}")


def load_from_csv(csv_path: Path) -> tuple[np.ndarray, list[str]]:
    """Load EEG data from a CSV file.

    Each column is a channel, each row is a time sample.
    The first row is skipped if it contains non-numeric headers.
    Returns data in (channels, samples) shape with float32 dtype.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    import csv as _csv

    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = _csv.reader(fh)
        rows_raw = [[v.strip() for v in row if v.strip() != ""] for row in reader]

    if not rows_raw:
        raise ValueError(f"CSV file is empty: {csv_path}")

    # If the first row contains non-numeric values, treat it as a header.
    rows_raw = [row for row in rows_raw if row]  # drop fully empty rows
    labels: list[str] = []
    try:
        _ = [float(v) for v in rows_raw[0]]
    except ValueError:
        labels = rows_raw[0]
        normalized = {label.strip().lower() for label in labels}
        rejected = sorted(normalized.intersection(FEATURE_COLUMNS))
        if rejected:
            raise ValueError(
                "CSV appears to contain epoch features rather than raw EEG waveform "
                f"(found columns: {', '.join(rejected)}). Use mne_baseline for features.csv."
            )
        rows_raw = rows_raw[1:]

    if not rows_raw:
        raise ValueError(f"CSV file has no data rows after header: {csv_path}")

    data = np.array([[float(v) for v in row] for row in rows_raw], dtype=np.float32).T
    if not labels:
        labels = [f"EEG {index + 1}" for index in range(data.shape[0])]
    if len(labels) != data.shape[0]:
        raise ValueError("CSV header column count does not match waveform data columns")
    return data, labels


def apply_channel_policy(
    data: np.ndarray,
    source_labels: list[str],
    lsl_labels: list[str],
    policy: str,
) -> tuple[np.ndarray, list[dict[str, str]], bool]:
    if not lsl_labels:
        lsl_labels = source_labels
    if len(lsl_labels) == data.shape[0]:
        return data, [{"lsl_label": label, "source_channel": source_labels[index]} for index, label in enumerate(lsl_labels)], False
    if policy == "strict":
        raise ValueError(
            f"--lsl-labels defines {len(lsl_labels)} channels but the source contains {data.shape[0]}. "
            "Use matching labels or explicitly set --channel-policy duplicate for a compatibility experiment."
        )
    if len(lsl_labels) < data.shape[0]:
        raise ValueError("duplicate channel policy cannot discard physical source channels")
    indices = [index % data.shape[0] for index in range(len(lsl_labels))]
    mapping = [{"lsl_label": label, "source_channel": source_labels[index]} for label, index in zip(lsl_labels, indices)]
    return data[indices, :], mapping, True


def build_stream_info(name: str, stream_type: str, labels: list[str], sfreq: float, source_id: str):
    from pylsl import StreamInfo, cf_float32

    info = StreamInfo(
        name=name,
        type=stream_type,
        channel_count=len(labels),
        nominal_srate=sfreq,
        channel_format=cf_float32,
        source_id=source_id,
    )

    channels = info.desc().append_child("channels")
    for label in labels:
        channel = channels.append_child("channel")
        channel.append_child_value("label", label)
        channel.append_child_value("unit", "microvolts")
        channel.append_child_value("type", "EEG")
    return info


def sleep_until(target_time: float) -> None:
    remaining = target_time - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def stream_data(
    data: np.ndarray,
    sfreq: float,
    stream_name: str,
    stream_type: str,
    labels: list[str],
    source_id: str,
    minutes: float,
    chunk_seconds: float,
    push_mode: str,
) -> None:
    from pylsl import StreamOutlet, local_clock

    info = build_stream_info(stream_name, stream_type, labels, sfreq, source_id)
    outlet = StreamOutlet(info)

    max_samples = min(data.shape[1], int(minutes * 60 * sfreq))
    chunk_size = max(1, int(chunk_seconds * sfreq))

    print(f"LSL stream ready: name={stream_name}, type={stream_type}, source_id={source_id}")
    print(f"Channels: {labels}")
    print(f"Sample rate: {sfreq:g} Hz; duration: {max_samples / sfreq:.1f}s; push_mode={push_mode}")
    print("Open NeuroSkill Settings -> LSL and scan while this script is running.")

    stream_start_wall = time.perf_counter()
    stream_start_lsl = local_clock()
    sample_interval = 1.0 / sfreq

    for start in range(0, max_samples, chunk_size):
        stop = min(max_samples, start + chunk_size)
        samples = data[:, start:stop].T.astype(np.float32)

        if push_mode == "sample":
            for offset, sample in enumerate(samples):
                sample_index = start + offset
                sleep_until(stream_start_wall + sample_index * sample_interval)
                outlet.push_sample(sample.tolist(), timestamp=stream_start_lsl + sample_index * sample_interval)
        else:
            last_sample_index = stop - 1
            sleep_until(stream_start_wall + last_sample_index * sample_interval)
            outlet.push_chunk(samples.tolist(), timestamp=stream_start_lsl + last_sample_index * sample_interval)

        elapsed = stop / sfreq
        print(f"pushed {elapsed:8.2f}s / {max_samples / sfreq:.2f}s", end="\r")

    print("\nPush complete.")


def check_dependencies(require_mne: bool, require_pylsl: bool) -> None:
    """Verify only the dependencies needed by the selected source."""
    missing: list[str] = []
    if require_mne:
        try:
            import mne  # noqa: F811
        except ImportError:
            missing.append("mne")
    if require_pylsl:
        try:
            import pylsl  # noqa: F811
        except ImportError:
            missing.append("pylsl")
    if not missing:
        return
    packages = " and ".join(missing)
    raise ImportError(
        f"{packages} not found in the current Python environment. "
        f"Current Python: {sys.executable}. "
        "Use the brainfusion environment or install the missing dependencies."
    )
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream an EDF file or pre-saved data as LSL EEG.")
    # --- data source: exactly one of --edf, --from-npy, --from-csv required ---
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--edf", default="", help="Path to PSG EDF file.")
    source_group.add_argument("--from-npy", default="", help="Path to .npy file (channels x samples, float32).")
    source_group.add_argument("--from-csv", default="", help="Path to CSV file (columns=channels, rows=time samples).")
    # --- EDF-specific ---
    parser.add_argument("--channels", default="EEG Fpz-Cz,EEG Pz-Oz", help="Comma-separated EDF channel names. Only used with --edf.")
    parser.add_argument("--stream-name", default="SleepEDF_LSL", help="LSL stream name.")
    parser.add_argument("--stream-type", default="EEG", help="LSL stream type.")
    parser.add_argument("--source-id", default="", help="LSL source_id. Auto-generated uniquely when omitted.")
    parser.add_argument("--lsl-labels", default="", help="Comma-separated LSL channel labels. Defaults to source labels.")
    parser.add_argument(
        "--channel-policy",
        choices=["strict", "duplicate"],
        default="strict",
        help="Require matching physical channels, or explicitly duplicate channels for a compatibility experiment.",
    )
    parser.add_argument(
        "--input-layout",
        choices=["auto", "channels-samples", "epochs-samples", "epochs-channels-samples"],
        default="auto",
        help="Layout for --from-npy. Auto treats a 2-D first dimension over 16 as epochs.",
    )
    parser.add_argument("--sample-rate", type=float, default=None, help="Target LSL sample rate; defaults to manifest rate or 256 Hz.")
    parser.add_argument("--source-manifest", default="", help="Optional waveform source manifest JSON from the data loader.")
    parser.add_argument("--minutes", type=float, default=10.0, help="Minutes to stream.")
    parser.add_argument("--chunk-seconds", type=float, default=0.125, help="Chunk size in seconds.")
    parser.add_argument("--push-mode", choices=["sample", "chunk"], default="chunk", help="Push samples one-by-one or in chunks.")
    parser.add_argument("--dry-run", action="store_true", help="Load and summarize EDF without starting LSL.")
    parser.add_argument("--metadata-json-out", default="", help="Write stream metadata JSON before pushing.")
    parser.add_argument(
        "--state-json-out",
        default="",
        help="Write adapter status JSON for loading, dry-run, completion, or failure.",
    )
    parser.add_argument(
        "--error-json-out",
        default="",
        help="Write structured error JSON when the adapter exits unsuccessfully.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    manifest = load_source_manifest(args.source_manifest)
    input_path = Path(args.edf or args.from_npy or args.from_csv)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    check_dependencies(require_mne=bool(args.edf), require_pylsl=not args.dry_run)
    target_sfreq = args.sample_rate or float(manifest.get("sample_rate_hz", 256.0))
    input_layout = args.input_layout if args.input_layout != "auto" else manifest.get("layout", "auto")

    state: dict = {
        "component": "lsl_adapter",
        "status": "running",
        "started_at": utc_now(),
        "command": [sys.executable, *sys.argv],
        "config": {
            "stream_name": args.stream_name,
            "stream_type": args.stream_type,
            "source_id": args.source_id or "auto",
            "labels": parse_csv(args.lsl_labels),
            "sample_rate": target_sfreq,
            "minutes": args.minutes,
            "chunk_seconds": args.chunk_seconds,
            "push_mode": args.push_mode,
            "channel_policy": args.channel_policy,
            "input_layout": input_layout,
            "dry_run": args.dry_run,
        },
    }
    write_json(args.state_json_out, state)

    # --- load data from the selected source ---
    if args.edf:
        requested_channels = parse_csv(args.channels)
        data, sfreq, selected = load_edf_data(Path(args.edf), requested_channels, target_sfreq)
        source_tag = "edf"
        source_labels = selected
        source_detail = {"edf_channels_selected": selected, "input_layout": "channels-samples"}
    elif args.from_npy:
        data, source_labels, resolved_layout = load_from_npy(Path(args.from_npy), input_layout)
        sfreq = target_sfreq
        source_tag = "npy"
        source_detail = {"npy_path": args.from_npy, "input_layout": resolved_layout}
    else:  # --from-csv
        data, source_labels = load_from_csv(Path(args.from_csv))
        sfreq = target_sfreq
        source_tag = "csv"
        source_detail = {"csv_path": args.from_csv, "input_layout": "channels-samples"}

    labels = parse_csv(args.lsl_labels) or source_labels
    manifest_labels = manifest.get("channel_names")
    if isinstance(manifest_labels, list) and len(manifest_labels) == len(source_labels):
        source_labels = [str(label) for label in manifest_labels]
        labels = parse_csv(args.lsl_labels) or source_labels
    data, channel_mapping, synthetic_duplication = apply_channel_policy(
        data, source_labels, labels, args.channel_policy
    )
    source_id = args.source_id or f"sleep-edf-{input_path.stem.lower()}-{uuid.uuid4().hex[:8]}"
    state["config"].update({"source_id": source_id, "labels": labels})

    metadata: dict = {
        "name": args.stream_name,
        "type": args.stream_type,
        "channel_count": len(labels),
        "nominal_srate": sfreq,
        "channel_format": "cf_float32",
        "source_id": source_id,
        "labels": labels,
        "data_source": source_tag,
        "unit": "microvolts",
        "data_shape": [data.shape[0], data.shape[1]],
        "source_channel_count": len(source_labels),
        "lsl_channel_count": len(labels),
        "channel_policy": args.channel_policy,
        "channel_mapping": channel_mapping,
        "synthetic_channel_duplication": synthetic_duplication,
        "source_manifest": args.source_manifest or None,
        "push_mode": args.push_mode,
        "minutes": args.minutes,
        "chunk_seconds": args.chunk_seconds,
    }
    metadata.update(source_detail)
    state["metadata"] = metadata
    write_json(args.state_json_out, state)

    if source_tag == "edf":
        print(f"EDF channels selected: {selected}")
    else:
        print(f"Loaded from {source_tag}: {args.from_npy or args.from_csv}")
    print(f"Data shape for LSL: channels={data.shape[0]}, samples={data.shape[1]}")
    print(f"Amplitude median abs: {float(np.nanmedian(np.abs(data))):.3f} microvolts")

    if args.metadata_json_out:
        write_json(args.metadata_json_out, metadata)
        print(f"Metadata written to: {args.metadata_json_out}")

    if args.dry_run:
        state.update({"status": "dry_run_completed", "completed_at": utc_now()})
        write_json(args.state_json_out, state)
        return

    stream_data(
        data=data,
        sfreq=sfreq,
        stream_name=args.stream_name,
        stream_type=args.stream_type,
        labels=labels,
        source_id=source_id,
        minutes=args.minutes,
        chunk_seconds=args.chunk_seconds,
        push_mode=args.push_mode,
    )
    state.update({"status": "completed", "completed_at": utc_now()})
    write_json(args.state_json_out, state)


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        error = {
            "component": "lsl_adapter",
            "status": "failed",
            "failed_at": utc_now(),
            "command": [sys.executable, *sys.argv],
            "error_type": type(exc).__name__,
            "message": str(exc),
            "command_hint": command_hint(exc),
        }
        write_json(args.error_json_out, error)
        write_json(args.state_json_out, error)
        print(f"ERROR: {error['error_type']}: {error['message']}", file=sys.stderr)
        if args.error_json_out:
            print(f"Error JSON written to: {args.error_json_out}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
