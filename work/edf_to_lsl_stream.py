"""Stream an EDF file as a local LSL EEG outlet.

The default configuration follows NeuroSkill's LSL documentation more closely
than the first single-channel 100 Hz attempt:

- stream type: EEG
- 4 channels
- 256 Hz
- channel labels: TP9, AF7, AF8, TP10
- unit: microvolts

If the EDF has fewer channels than requested, the script repeats available
channels to reach the requested LSL channel count.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


DEFAULT_LABELS = ["TP9", "AF7", "AF8", "TP10"]


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_edf_data(edf_path: Path, requested_channels: list[str], target_sfreq: float) -> tuple[np.ndarray, float, list[str]]:
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    import mne

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    available = raw.ch_names

    selected = [ch for ch in requested_channels if ch in available]
    if not selected:
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


def expand_channels(data: np.ndarray, target_count: int) -> np.ndarray:
    if data.shape[0] == target_count:
        return data
    indices = [i % data.shape[0] for i in range(target_count)]
    return data[indices, :]


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
    from pylsl import StreamOutlet

    info = build_stream_info(stream_name, stream_type, labels, sfreq, source_id)
    outlet = StreamOutlet(info)

    max_samples = min(data.shape[1], int(minutes * 60 * sfreq))
    chunk_size = max(1, int(chunk_seconds * sfreq))

    print(f"LSL stream ready: name={stream_name}, type={stream_type}, source_id={source_id}")
    print(f"Channels: {labels}")
    print(f"Sample rate: {sfreq:g} Hz; duration: {max_samples / sfreq:.1f}s; push_mode={push_mode}")
    print("Open NeuroSkill Settings -> LSL and scan while this script is running.")

    for start in range(0, max_samples, chunk_size):
        stop = min(max_samples, start + chunk_size)
        samples = data[:, start:stop].T.astype(np.float32)

        if push_mode == "sample":
            for sample in samples:
                outlet.push_sample(sample.tolist())
        else:
            outlet.push_chunk(samples.tolist())

        elapsed = stop / sfreq
        print(f"pushed {elapsed:8.2f}s / {max_samples / sfreq:.2f}s", end="\r")
        time.sleep((stop - start) / sfreq)

    print("\nPush complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream an EDF file as LSL EEG.")
    parser.add_argument("--edf", required=True, help="Path to PSG EDF file.")
    parser.add_argument("--channels", default="EEG Fpz-Cz,EEG Pz-Oz", help="Comma-separated EDF channel names.")
    parser.add_argument("--stream-name", default="SleepEDF_LSL", help="LSL stream name.")
    parser.add_argument("--stream-type", default="EEG", help="LSL stream type.")
    parser.add_argument("--source-id", default="sleep-edf-lsl-001", help="Stable LSL source_id.")
    parser.add_argument("--lsl-labels", default="TP9,AF7,AF8,TP10", help="Comma-separated LSL channel labels.")
    parser.add_argument("--sample-rate", type=float, default=256.0, help="Target LSL sample rate.")
    parser.add_argument("--minutes", type=float, default=10.0, help="Minutes to stream.")
    parser.add_argument("--chunk-seconds", type=float, default=0.125, help="Chunk size in seconds.")
    parser.add_argument("--push-mode", choices=["sample", "chunk"], default="sample", help="Push samples one-by-one or in chunks.")
    parser.add_argument("--dry-run", action="store_true", help="Load and summarize EDF without starting LSL.")
    parser.add_argument("--metadata-json-out", default="", help="Write stream metadata JSON before pushing.")
    args = parser.parse_args()

    labels = parse_csv(args.lsl_labels) or DEFAULT_LABELS
    requested_channels = parse_csv(args.channels)
    data, sfreq, selected = load_edf_data(Path(args.edf), requested_channels, args.sample_rate)
    data = expand_channels(data, len(labels))

    print(f"EDF channels selected: {selected}")
    print(f"Data shape for LSL: channels={data.shape[0]}, samples={data.shape[1]}")
    print(f"Amplitude median abs: {float(np.nanmedian(np.abs(data))):.3f} microvolts")

    if args.metadata_json_out:
        metadata = {
            "name": args.stream_name,
            "type": args.stream_type,
            "channel_count": len(labels),
            "nominal_srate": sfreq,
            "channel_format": "cf_float32",
            "source_id": args.source_id,
            "labels": labels,
            "edf_channels_selected": selected,
            "unit": "microvolts",
            "data_shape": [data.shape[0], data.shape[1]],
            "push_mode": args.push_mode,
            "minutes": args.minutes,
            "chunk_seconds": args.chunk_seconds,
        }
        out_path = Path(args.metadata_json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Metadata written to: {out_path}")

    if args.dry_run:
        return

    stream_data(
        data=data,
        sfreq=sfreq,
        stream_name=args.stream_name,
        stream_type=args.stream_type,
        labels=labels,
        source_id=args.source_id,
        minutes=args.minutes,
        chunk_seconds=args.chunk_seconds,
        push_mode=args.push_mode,
    )


if __name__ == "__main__":
    main()
