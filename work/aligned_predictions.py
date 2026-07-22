"""
Day12

Align true labels and predictions

Input:
    true_labels.csv
    pred_labels.csv


Output:
    aligned_predictions.csv


Match:
    start_sec

Fields:

start_sec
true_stage
pred_stage

"""


import argparse
import os
import pandas as pd



def main():


    parser = argparse.ArgumentParser(
        description="Align true and predicted labels"
    )


    parser.add_argument(
        "--truth",
        required=True,
        help="true_labels.csv"
    )


    parser.add_argument(
        "--pred",
        required=True,
        help="pred_labels.csv"
    )


    parser.add_argument(
        "--out",
        required=True,
        help="aligned_predictions.csv"
    )


    args = parser.parse_args()



    print("Reading truth:")
    print(args.truth)


    print("Reading prediction:")
    print(args.pred)



    true_df = pd.read_csv(
        args.truth
    )


    pred_df = pd.read_csv(
        args.pred
    )


    print(
        "Truth count:",
        len(true_df)
    )


    print(
        "Prediction count:",
        len(pred_df)
    )



    # =========================
    # 根据时间戳对齐
    # =========================

    aligned = pd.merge(

        true_df[
            [
                "start_sec",
                "stage"
            ]
        ],

        pred_df[
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


    aligned = aligned.rename(

        columns={

            "stage_true":
            "true_stage",

            "stage_pred":
            "pred_stage"

        }

    )



    aligned = aligned[
        [
            "start_sec",
            "true_stage",
            "pred_stage"
        ]
    ]



    print(
        "Aligned epochs:",
        len(aligned)
    )



    output_dir = os.path.dirname(
        args.out
    )


    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )



    aligned.to_csv(
        args.out,
        index=False
    )



    print("\nPreview:")
    print(
        aligned.head(10)
    )


    print(
        "\nSaved:",
        args.out
    )



if __name__ == "__main__":

    main()