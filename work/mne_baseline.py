"""
MNE baseline sleep staging skeleton

Day8:
读取 PSG + Hypnogram
输出占位 pred_labels.csv

暂不训练模型
"""

import argparse
import os
import pandas as pd
import mne


def read_psg(psg_path):
    """
    读取 PSG EDF
    """
    print("Reading PSG:")
    print(psg_path)

    raw = mne.io.read_raw_edf(
        psg_path,
        preload=False
    )

    print("Channels:", len(raw.ch_names))
    print("Sampling frequency:", raw.info["sfreq"])

    return raw



def read_hypnogram(hypnogram_path):
    """
    读取 Hypnogram annotation
    """

    print("Reading Hypnogram:")
    print(hypnogram_path)

    annotations = mne.read_annotations(
        hypnogram_path
    )

    print(
        "Annotation count:",
        len(annotations)
    )

    return annotations



def create_placeholder_prediction(
        annotations,
        output_path
):
    """
    创建占位预测结果

    每30秒一个epoch
    """

    rows = []

    for idx, ann in enumerate(annotations):

        start = ann["onset"]

        rows.append(
            {
                "epoch": idx,
                "start_sec": start,
                "pred_stage": 0
            }
        )


    df = pd.DataFrame(rows)

    df.to_csv(
        output_path,
        index=False
    )

    print(
        "Saved:",
        output_path
    )



def main():

    parser = argparse.ArgumentParser(
        description=
        "MNE baseline sleep staging skeleton"
    )


    parser.add_argument(
        "--psg",
        required=True,
        help="PSG EDF path"
    )


    parser.add_argument(
        "--hypnogram",
        required=True,
        help="Hypnogram EDF path"
    )


    parser.add_argument(
        "--out",
        default="outputs_examples/pred_labels.csv",
        help="prediction csv output"
    )


    args = parser.parse_args()


    raw = read_psg(args.psg)

    annotations = read_hypnogram(
        args.hypnogram
    )


    create_placeholder_prediction(
        annotations,
        args.out
    )



if __name__ == "__main__":
    main()