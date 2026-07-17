import argparse
import pandas as pd

# -------------------------------
# 接收命令行参数
# -------------------------------

parser = argparse.ArgumentParser(
    description="提取Sleep-EDF标签"
)

parser.add_argument(
    "--hypnogram",
    type=str,
    required=True,
    help="Hypnogram文件路径"
)

parser.add_argument(
    "--out",
    type=str,
    required=True,
    help="CSV输出路径"
)

args = parser.parse_args()

print("========== Day1 ==========")
print("Hypnogram文件：", args.hypnogram)
print("输出CSV：", args.out)

# -------------------------------
# 创建空CSV（Day1任务）
# -------------------------------

df = pd.DataFrame(
    columns=[
        "start_sec",
        "stage",
        "stage_name"
    ]
)

df.to_csv(
    args.out,
    index=False
)

print("CSV文件生成成功！")