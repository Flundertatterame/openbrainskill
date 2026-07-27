"""
Day19:

Plot hypnogram comparison

Input:
    true_labels.csv
    pred_labels.csv

Output:
    hypnogram_compare.png


Usage example:

python work/plot_hypnogram.py \
--truth <run_dir>/true_labels.csv \
--pred <run_dir>/pred_labels.csv \
--out <run_dir>/hypnogram_compare.png

"""


import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt



# ==========================
# Sleep stage names
# ==========================

STAGE_NAMES = {

    0: "Wake",

    1: "N1",

    2: "N2",

    3: "N3",

    4: "REM"

}



# ==========================
# Load csv
# ==========================

def load_labels(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"File not found: {path}"
        )


    df = pd.read_csv(path)


    required = [
        "start_sec",
        "stage"
    ]


    for col in required:

        if col not in df.columns:

            raise ValueError(
                f"Missing column {col} in {path}"
            )


    return df



# ==========================
# Plot
# ==========================

def plot_compare(
        truth,
        pred,
        out
):


    # ----------------------
    # align by start_sec
    # ----------------------

    merged = pd.merge(
        truth[
            [
                "start_sec",
                "stage"
            ]
        ],

        pred[
            [
                "start_sec",
                "stage"
            ]
        ],

        on="start_sec",

        how="inner",

        suffixes=(
            "_true",
            "_pred"
        )
    )


    if len(merged)==0:

        raise RuntimeError(
            "No matching epochs"
        )


    print(
        "Matched epochs:",
        len(merged)
    )



    # ----------------------
    # create figure
    # ----------------------

    plt.figure(
        figsize=(16,5)
    )


    x = (
        merged["start_sec"]
        /
        3600
    )


    plt.step(
        x,
        merged["stage_true"],
        where="post",
        label="True"
    )


    plt.step(
        x,
        merged["stage_pred"],
        where="post",
        label="Prediction"
    )


    plt.yticks(
        list(STAGE_NAMES.keys()),
        list(STAGE_NAMES.values())
    )


    plt.xlabel(
        "Time (hour)"
    )


    plt.ylabel(
        "Sleep Stage"
    )


    plt.title(
        "Hypnogram: True vs Prediction"
    )


    plt.grid(
        True
    )


    plt.legend()



    output_dir = os.path.dirname(out)

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )


    plt.savefig(
        out,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()



    print(
        "Saved:",
        out
    )



# ==========================
# Main
# ==========================

def main():


    parser = argparse.ArgumentParser(
        description="Plot hypnogram comparison"
    )


    parser.add_argument(
        "--truth",
        required=True
    )


    parser.add_argument(
        "--pred",
        required=True
    )


    parser.add_argument(
        "--out",
        required=True
    )


    args = parser.parse_args()



    truth = load_labels(
        args.truth
    )


    pred = load_labels(
        args.pred
    )


    plot_compare(
        truth,
        pred,
        args.out
    )



if __name__=="__main__":

    main()