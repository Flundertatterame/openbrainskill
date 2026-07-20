"""
MNE baseline sleep staging

Input:
    PSG EDF
    true_labels.csv

Output:
    pred_labels.csv

Interface:

python work/mne_baseline.py \
    --psg SC4002E0-PSG.edf \
    --truth true_labels.csv \
    --out pred_labels.csv


Output format:

start_sec,stage

Example:

0,0
30,0
60,0

Current:
    baseline placeholder
    all predictions = stage 0
"""

import argparse
import os
import pandas as pd
import mne



# =========================
# Read PSG EDF
# =========================

def read_psg(psg_path):

    print("Reading PSG:")
    print(psg_path)

    raw = mne.io.read_raw_edf(
        psg_path,
        preload=False
    )

    print("Channels:", len(raw.ch_names))
    print(
        "Sampling frequency:",
        raw.info["sfreq"]
    )

    return raw



# =========================
# Read true_labels.csv
# =========================

def read_truth(truth_path):

    print("\nReading truth:")
    print(truth_path)

    truth = pd.read_csv(
        truth_path
    )


    required_columns = [
        "start_sec",
        "stage"
    ]


    for col in required_columns:

        if col not in truth.columns:
            raise ValueError(
                f"Missing column: {col}"
            )


    print("\nPreview:")
    print(truth.head())


    print(
        "\nEpoch count:",
        len(truth)
    )


    return truth



# =========================
# Create baseline prediction
# =========================

def create_prediction(
        truth,
        output_path
):

    print(
        "\nCreating prediction..."
    )


    pred = pd.DataFrame()


    # 保留30秒epoch时间点

    pred["start_sec"] = (
        truth["start_sec"]
    )


    # baseline占位预测
    # 当前全部预测Wake

    pred["stage"] = 0



    # 自动创建输出目录

    output_dir = os.path.dirname(
        output_path
    )


    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )



    pred.to_csv(
        output_path,
        index=False
    )


    print(
        "\nSaved:",
        output_path
    )



# =========================
# Main
# =========================

def main():


    parser = argparse.ArgumentParser(
        description=
        "MNE baseline sleep staging"
    )


    parser.add_argument(
        "--psg",
        required=True,
        help="PSG EDF path"
    )


    parser.add_argument(
        "--truth",
        required=True,
        help="true_labels.csv path"
    )


    parser.add_argument(
        "--out",
        required=True,
        help="prediction CSV output path"
    )



    args = parser.parse_args()



    # 1.
    # Load PSG

    read_psg(
        args.psg
    )



    # 2.
    # Load true labels

    truth = read_truth(
        args.truth
    )



    # 3.
    # Generate prediction

    create_prediction(
        truth,
        args.out
    )



if __name__ == "__main__":

    main()