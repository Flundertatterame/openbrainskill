"""
Day11
Evaluate sleep staging baseline


Input:

true_labels.csv

pred_labels.csv


Output:

metrics.json


Match:

start_sec

not list index
"""
import argparse
import json
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score  # 新增：卡帕系数计算工具
)


def main():
    parser = argparse.ArgumentParser()

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
        "Truth rows:",
        len(true_df)
    )

    print(
        "Prediction rows:",
        len(pred_df)
    )

    # =========================
    # 核心修改
    # 按时间匹配
    # =========================
    merged = pd.merge(
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

    print(
        "Matched epochs:",
        len(merged)
    )

    y_true = merged[
        "stage_true"
    ]

    y_pred = merged[
        "stage_pred"
    ]

    # 计算基础指标
    acc = accuracy_score(
        y_true,
        y_pred
    )
    cm = confusion_matrix(
        y_true,
        y_pred
    )
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )
    # 新增：计算Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    # 提取宏平均F1分数
    macro_f1 = report["macro avg"]["f1-score"]

    # 统一标准化metrics字段
    metrics = {
        "n_aligned_epochs": len(merged),  # 替换旧字段 matched_epochs
        "accuracy": acc,
        "macro_f1": macro_f1,              # 新增宏平均F1
        "cohen_kappa": kappa,               # 新增卡帕系数
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }

    with open(
        args.out,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4
        )

    print(
        "\nAccuracy:",
        acc
    )
    print("Macro F1:", macro_f1)
    print("Cohen Kappa:", kappa)
    print(
        "Saved:",
        args.out
    )


if __name__ == "__main__":
    main()