import argparse
import os
import json
import subprocess
import sys

from pathlib import Path

from path_config import (
    resolve_sample_paths,
    load_config
)

import numpy as np
import pandas as pd
import mne
from scipy.signal import welch



# =========================
# EDF check
# =========================

def check_edf(
    edf_path,
    output_json
):

    result = {
        "exists": False,
        "channels": 0,
        "sfreq": 0,
        "duration_sec": 0,
        "eeg_channels": [],
        "error": None
    }


    try:

        raw = mne.io.read_raw_edf(
            edf_path,
            preload=False
        )


        result["exists"] = True

        result["channels"] = len(
            raw.ch_names
        )

        result["sfreq"] = raw.info["sfreq"]

        result["duration_sec"] = raw.times[-1]


        eeg_channels = []


        for ch in raw.info["chs"]:

            if ch["kind"] == mne.io.constants.FIFF.FIFFV_EEG_CH:

                eeg_channels.append(
                    ch["ch_name"]
                )


        result["eeg_channels"] = eeg_channels


    except Exception as e:

        result["error"] = str(e)



    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=4
        )



# =========================
# generate true labels
# =========================

def prepare_truth(paths):


    truth_path = paths["truth"]


    if os.path.exists(truth_path):

        print(
            "Truth exists:",
            truth_path
        )

        return



    print(
        "Generating truth:"
        ,
        truth_path
    )


    cmd = [

        sys.executable,

        "work/extract_sleep_edf_labels.py",

        "--hypnogram",

        paths["hypnogram"],

        "--out",

        truth_path

    ]


    result = subprocess.run(cmd)


    if result.returncode != 0:

        raise RuntimeError(
            "Failed generating truth"
        )



# =========================
# feature extraction
# =========================


BANDS = {

    "delta":(0.5,4),

    "theta":(4,8),

    "alpha":(8,13),

    "beta":(13,30)

}



def extract_features(
    psg_path,
    truth_path,
    output_csv
):


    truth = pd.read_csv(
        truth_path
    )


    raw = mne.io.read_raw_edf(
        psg_path,
        preload=True
    )


    channel = mne.pick_types(
        raw.info,
        eeg=True
    )[0]


    data = raw.get_data(
        picks=[channel]
    )[0]


    sfreq = raw.info["sfreq"]


    epoch_samples = int(
        sfreq * 30
    )


    rows=[]


    for start_sec in truth["start_sec"]:


        start_sample=int(
            start_sec * sfreq
        )


        end_sample = (
            start_sample
            +
            epoch_samples
        )


        segment=data[
            start_sample:end_sample
        ]


        freqs,psd = welch(
            segment,
            fs=sfreq,
            nperseg=int(sfreq*2)
        )


        valid = (
            (freqs>=0.5)
            &
            (freqs<30)
        )


        total=np.sum(
            psd[valid]
        )


        row={

            "start_sec":start_sec

        }


        for name,(low,high) in BANDS.items():

            idx=(

                (freqs>=low)
                &
                (freqs<high)

            )


            power=np.sum(
                psd[idx]
            )


            row[name+"_power"] = (
                power/total
                if total>0
                else 0
            )


        rows.append(row)



    df=pd.DataFrame(rows)


    df.to_csv(
        output_csv,
        index=False
    )


    print(
        "Saved features:",
        output_csv
    )



# =========================
# main
# =========================


def main():

    parser = argparse.ArgumentParser(
        description="Data Loader v1"
    )


    # =========================
    # single sample
    # =========================

    parser.add_argument(
        "--sample",
        help="Single sample id"
    )


    # =========================
    # batch samples
    # =========================

    parser.add_argument(
        "--samples",
        help="Comma separated sample ids"
    )


    parser.add_argument(
        "--config",
        default="sample_path_config.example.json"
    )


    args = parser.parse_args()



    # =========================
    # 解析样本列表
    # =========================

    if args.samples:


        sample_list = [

            s.strip()

            for s in args.samples.split(",")

            if s.strip()

        ]


    elif args.sample:


        sample_list = [

            args.sample

        ]


    else:

        raise ValueError(
            "Please provide --sample or --samples"
        )


    print(
        "Samples:"
    )

    print(
        sample_list
    )



    # =========================
    # 批量处理
    # =========================

    for sample in sample_list:


        print("="*60)

        print(
            "Processing:",
            sample
        )


        try:


            paths = resolve_sample_paths(
                sample,
                args.config
            )



            # -------------------------
            # 检查 PSG / Hypnogram
            # -------------------------

            if not os.path.exists(
                paths["psg"]
            ):

                raise FileNotFoundError(
                    f"PSG missing: {paths['psg']}"
                )


            if not os.path.exists(
                paths["hypnogram"]
            ):

                raise FileNotFoundError(
                    f"Hypnogram missing: {paths['hypnogram']}"
                )



            config = load_config(
                args.config
            )



            run_dir = (

                Path(
                    config["output_root"]
                )

                /

                sample

                /

                "run"

            )


            run_dir.mkdir(
                parents=True,
                exist_ok=True
            )



            # =====================
            # 1. true_labels.csv
            # =====================

            prepare_truth(
                paths
            )


            truth_out = (

                run_dir

                /

                "true_labels.csv"

            )


            pd.read_csv(
                paths["truth"]
            ).to_csv(
                truth_out,
                index=False
            )


            print(
                "Saved:",
                truth_out
            )



            # =====================
            # 2. edf_check.json
            # =====================

            check_edf(

                paths["psg"],

                run_dir /

                "edf_check.json"

            )


            print(
                "Saved:",
                run_dir /
                "edf_check.json"
            )



            # =====================
            # 3. features.csv
            # =====================

            extract_features(

                paths["psg"],

                paths["truth"],

                run_dir /

                "features.csv"

            )



            print(
                "Finished:",
                sample
            )


        except Exception as e:


            print(
                f"Skip {sample}: {e}"
            )


            continue



    print("="*60)

    print(
        "Batch Data Loader finished"
    )



if __name__=="__main__":

    main()


if __name__=="__main__":

    main()