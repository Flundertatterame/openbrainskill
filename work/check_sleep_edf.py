import argparse
import json
import mne


parser = argparse.ArgumentParser()

parser.add_argument("--edf", required=True)
parser.add_argument("--json-out", required=True)

args = parser.parse_args()


result = {
    "exists": False,
    "channels": 0,
    "sfreq": 0,
    "duration_sec": 0,
    "eeg_channels": [],
    "error": None
}

try:
    raw = mne.io.read_raw_edf(args.edf, preload=False)

    result["exists"] = True
    result["channels"] = len(raw.ch_names)
    result["sfreq"] = raw.info["sfreq"]
    result["duration_sec"] = raw.times[-1]

    eeg_channels = []

    for ch in raw.info["chs"]:
        if ch["kind"] == mne.io.constants.FIFF.FIFFV_EEG_CH:
            eeg_channels.append(ch["ch_name"])

    result["eeg_channels"] = eeg_channels

except Exception as e:
    result["error"] = str(e)

with open(args.json_out, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4)

print("JSON已保存：", args.json_out)