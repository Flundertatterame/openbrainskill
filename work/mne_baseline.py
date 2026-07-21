"""
MNE baseline sleep staging

Day10:

Input:
    PSG EDF
    true_labels.csv

Output:
    pred_labels.csv

Interface:

python work\mne_baseline.py
--psg PSG.edf
--truth true_labels.csv
--out pred_labels.csv


Method:

30s epoch
+
frequency band power

Rule:

delta high:
    N3

alpha/beta high:
    Wake

otherwise:
    N2
"""


import argparse
import os
import numpy as np
import pandas as pd
import mne

from scipy.signal import welch



# ==========================
# Frequency bands
# ==========================

BANDS = {

    "delta": (0.5,4),

    "theta": (4,8),

    "alpha": (8,13),

    "beta": (13,30)

}



# ==========================
# Read PSG
# ==========================

def read_psg(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"PSG not found: {path}"
        )


    print("Reading PSG:")
    print(path)


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



# ==========================
# Select EEG channel
# ==========================

def select_eeg(raw):


    picks = mne.pick_types(
        raw.info,
        eeg=True
    )


    if len(picks)==0:

        raise RuntimeError(
            "No EEG channel found"
        )


    print(
        "Using EEG:",
        raw.ch_names[picks[0]]
    )


    return picks[0]



# ==========================
# Calculate relative power
# ==========================

def band_power(
        data,
        sfreq
):


    freqs, psd = welch(
        data,
        fs=sfreq,
        nperseg=int(sfreq*2)
    )


    total_power=np.sum(psd)


    result={}


    for name,(low,high) in BANDS.items():


        idx=(

            (freqs>=low)

            &

            (freqs<high)

        )


        power=np.sum(
            psd[idx]
        )


        result[name+"_power"]=(
            power/total_power
            if total_power>0
            else 0
        )


    return result



# ==========================
# Feature extraction
# ==========================

def extract_features(
        raw,
        channel
):


    sfreq=raw.info["sfreq"]


    epoch_samples=int(
        sfreq*30
    )


    data=raw.get_data(
        picks=[channel]
    )[0]


    n_epochs=len(data)//epoch_samples


    rows=[]


    for epoch in range(n_epochs):


        start_sec=epoch*30


        segment=data[
            epoch*epoch_samples:
            (epoch+1)*epoch_samples
        ]


        feature=band_power(
            segment,
            sfreq
        )


        rows.append(
            {
                "start_sec":start_sec,
                **feature
            }
        )


    return pd.DataFrame(rows)



# ==========================
# Rule classifier
# ==========================

def predict_stage(row):


    delta=row["delta_power"]

    alpha=row["alpha_power"]

    beta=row["beta_power"]


    # N3

    if delta>0.5:

        return 3



    # Wake

    if alpha+beta>0.4:

        return 0



    # N2

    return 2



# ==========================
# Main
# ==========================

def main():


    parser=argparse.ArgumentParser(
        description=
        "MNE baseline sleep staging"
    )


    parser.add_argument(
        "--psg",
        required=True
    )


    parser.add_argument(
        "--truth",
        required=True
    )


    parser.add_argument(
        "--out",
        required=True
    )


    args=parser.parse_args()



    raw=read_psg(
        args.psg
    )


    channel=select_eeg(
        raw
    )


    print(
        "\nExtracting features..."
    )


    features=extract_features(
        raw,
        channel
    )


    print(
        features.head()
    )


    print(
        "Epoch number:",
        len(features)
    )



    print(
        "\nPredicting..."
    )


    features["stage"]=features.apply(
        predict_stage,
        axis=1
    )



    pred=features[
        [
            "start_sec",
            "stage"
        ]
    ]



    output_dir=os.path.dirname(
        args.out
    )


    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )


    pred.to_csv(
        args.out,
        index=False
    )


    print(
        "\nPrediction preview:"
    )

    print(
        pred.head(10)
    )


    print(
        "\nSaved:",
        args.out
    )



if __name__=="__main__":

    main()