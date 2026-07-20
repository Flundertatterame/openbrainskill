"""
Day9
MNE baseline feature extraction

Input:
    PSG EDF

Output:
    features.csv

Features:
    delta/theta/alpha/beta relative power

Epoch:
    30 seconds
"""

import argparse
import os
import numpy as np
import pandas as pd
import mne

from scipy.signal import welch



BANDS = {
    "delta_power": (0.5, 4),
    "theta_power": (4, 8),
    "alpha_power": (8, 13),
    "beta_power": (13, 30)
}



def read_psg(path):

    print("Reading PSG:")
    print(path)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PSG not found: {path}"
        )

    raw = mne.io.read_raw_edf(
        path,
        preload=True
    )

    print(
        "Channels:",
        len(raw.ch_names)
    )

    print(
        "Sampling frequency:",
        raw.info["sfreq"]
    )

    return raw



def select_eeg(raw):

    picks = mne.pick_types(
        raw.info,
        eeg=True
    )

    if len(picks) == 0:
        raise RuntimeError(
            "No EEG channel found"
        )

    print(
        "Use EEG channel:",
        raw.ch_names[picks[0]]
    )

    return picks[0]



def band_power(data, sfreq):

    freqs, psd = welch(
        data,
        fs=sfreq,
        nperseg=int(sfreq * 2)
    )


    total = np.sum(psd)


    result = {}


    for name, (low, high) in BANDS.items():

        idx = (
            (freqs >= low)
            &
            (freqs <= high)
        )

        power = np.sum(
            psd[idx]
        )

        result[name] = (
            power / total
            if total > 0
            else 0
        )


    return result



def extract_features(raw, channel):

    sfreq = raw.info["sfreq"]


    epoch_length = int(
        sfreq * 30
    )


    data = raw.get_data(
        picks=[channel]
    )[0]


    n_epochs = len(data) // epoch_length


    rows = []


    for epoch in range(n_epochs):

        start_sec = epoch * 30


        segment = data[
            epoch * epoch_length:
            (epoch + 1) * epoch_length
        ]


        features = band_power(
            segment,
            sfreq
        )


        rows.append(
            {
                "epoch": epoch,
                "start_sec": start_sec,
                **features
            }
        )


    return pd.DataFrame(rows)



def main():

    parser = argparse.ArgumentParser(
        description="Extract EEG frequency features"
    )


    parser.add_argument(
        "--psg",
        required=True,
        help="PSG EDF path"
    )


    parser.add_argument(
        "--out",
        required=True,
        help="features csv output"
    )


    args = parser.parse_args()


    raw = read_psg(
        args.psg
    )


    channel = select_eeg(
        raw
    )


    df = extract_features(
        raw,
        channel
    )


    output_dir = os.path.dirname(
        args.out
    )


    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )


    df.to_csv(
        args.out,
        index=False
    )


    print("\nPreview:")
    print(df.head(10))


    print(
        "\nSaved:",
        args.out
    )



if __name__ == "__main__":

    main()