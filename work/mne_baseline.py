"""
MNE baseline sleep staging

Day10:

Input:
    PSG EDF
    true_labels.csv

Output:
    pred_labels.csv

Interface:

python work/mne_baseline.py
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
import subprocess

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


    valid_idx = (
        (freqs >= 0.5)
        &
        (freqs < 30)
    )

    total_power = np.sum(
     psd[valid_idx]
    )


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
        channel,
        truth
):

    sfreq = raw.info["sfreq"]

    epoch_samples = int(
        sfreq * 30
    )

    data = raw.get_data(
        picks=[channel]
    )[0]


    # 使用真实标签中的时间点
    epoch_times = truth["start_sec"].tolist()


    rows = []


    for start_sec in epoch_times:

        start_sample = int(
            start_sec * sfreq
        )

        end_sample = (
            start_sample
            +
            epoch_samples
        )


        segment = data[
            start_sample:end_sample
        ]


        # 防止最后一个epoch长度不足30秒
        if len(segment) < epoch_samples:
            continue


        feature = band_power(
            segment,
            sfreq
        )


        rows.append(
            {
                "start_sec": start_sec,
                **feature
            }
        )


    return pd.DataFrame(rows)



# ==========================
# Rule classifier
# ==========================

def predict_stage(row):

    delta = row["delta_power"]
    theta = row["theta_power"]
    alpha = row["alpha_power"]
    beta = row["beta_power"]

    # N3：delta占明显优势
    if delta > 0.75 and delta > theta:
        return 3

    # Wake：alpha+beta较高
    if alpha + beta > 0.18:
        return 0

    # 默认N2
    return 2



# ==========================
# Main
# ==========================

def run_baseline(
    psg_path,
    truth_path,
    out_path
):
    """
    Run baseline prediction for one sample
    """

    print("Reading truth:")
    print(truth_path)

    if not os.path.exists(truth_path):
        raise FileNotFoundError(
            f"Truth file not found: {truth_path}"
        )

    truth = pd.read_csv(truth_path)

    print(truth.head())

    print("Epoch count:", len(truth))

    raw = read_psg(
        psg_path
    )

    channel = select_eeg(
        raw
    )

    print(
        "\nExtracting features..."
    )

    features = extract_features(
        raw,
        channel,
        truth
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

    features["stage"] = features.apply(
        predict_stage,
        axis=1
    )

    pred = features[
        [
            "start_sec",
            "stage"
        ]
    ]

    output_dir = os.path.dirname(
        out_path
    )

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    pred.to_csv(
        out_path,
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
        out_path
    )




def main():

    parser = argparse.ArgumentParser(
        description="MNE baseline sleep staging"
    )

    # ---------- 单样本模式 ----------
    parser.add_argument(
        "--psg",
        help="PSG EDF path"
    )

    parser.add_argument(
        "--truth",
        help="Ground truth csv"
    )

    parser.add_argument(
        "--out",
        help="Prediction csv output"
    )

    # ---------- 批量模式 ----------
    parser.add_argument(
        "--samples",
        help="Example: SC4001E0,SC4002E0"
    )

    parser.add_argument(
        "--samples-json",
        help="JSON file containing sample list"
    )

    parser.add_argument(
        "--config",
        default="sample_path_config.example.json",
        help="Sample path config"
    )

    args = parser.parse_args()

    # ==========================
    # 批量模式
    # ==========================
    if args.samples or args.samples_json:

        from path_config import resolve_sample_paths, load_config
        import json

        # 读取样本列表
        if args.samples:

            sample_list = [
                s.strip()
                for s in args.samples.split(",")
                if s.strip()
            ]

        else:

            with open(
                args.samples_json,
                "r",
                encoding="utf-8"
            ) as f:

                sample_list = json.load(f)

                if isinstance(sample_list, dict):
                    sample_list = sample_list["samples"]

        print("Batch samples:")
        print(sample_list)

        for sample in sample_list:

            print("=" * 60)
            print("Processing:", sample)

            paths = resolve_sample_paths(
                sample,
                args.config
            )

            config = load_config(args.config)

            output_root = config["output_root"]

            output_dir = os.path.join(
                output_root,
                sample
            )

            os.makedirs(
                output_dir,
                exist_ok=True
            )

            truth_path = paths["truth"]


            # ==========================
            # 检查 true_labels.csv
            # ==========================

            if not os.path.exists(truth_path):

                print(
                    "Missing truth file:",
                    truth_path
                )

                print(
                    "Generating true_labels.csv..."
                )


                hypnogram_path = paths["hypnogram"]


                cmd = [
                    "python",
                    "work/extract_sleep_edf_labels.py",
                    "--hypnogram",
                    hypnogram_path,
                    "--out",
                    truth_path
                ]


                result = subprocess.run(cmd)


                if result.returncode != 0:

                    raise RuntimeError(
                        "Failed to generate true_labels.csv"
                    )


            else:

                print(
                    "Found truth:",
                    truth_path
                )

            pred_path = os.path.join(
                output_dir,
                "pred_labels.csv"
            )

            run_baseline(
                paths["psg"],
                truth_path,
                pred_path
            )

        print("=" * 60)
        print("Batch finished.")

        return

    # ==========================
    # 单样本模式（保持以前接口）
    # ==========================

    if args.psg is None:
        raise ValueError(
            "--psg is required in single mode."
        )

    if args.truth is None:
        raise ValueError(
            "--truth is required in single mode."
        )

    if args.out is None:
        raise ValueError(
            "--out is required in single mode."
        )

    run_baseline(
        args.psg,
        args.truth,
        args.out
    )



if __name__=="__main__":

    main()