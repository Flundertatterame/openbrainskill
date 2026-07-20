import argparse
import pandas as pd
import mne


# -------------------------------
# 参数
# -------------------------------

parser = argparse.ArgumentParser(
    description="Extract Sleep-EDF Hypnogram labels to CSV"
)

parser.add_argument(
    "--hypnogram",
    required=True,
    help="Hypnogram EDF path"
)

parser.add_argument(
    "--out",
    required=True,
    help="Output CSV path"
)

args = parser.parse_args()


print("========== Extract Hypnogram ==========")
print("Hypnogram:", args.hypnogram)
print("Output:", args.out)


# -------------------------------
# 读取 Hypnogram annotation
# -------------------------------

annotations = mne.read_annotations(
    args.hypnogram
)


# -------------------------------
# Sleep stage mapping
# -------------------------------

stage_map = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}


# -------------------------------
# 展开为30秒 epoch
# -------------------------------

rows = []


for onset, duration, desc in zip(
    annotations.onset,
    annotations.duration,
    annotations.description,
):

    if desc not in stage_map:
        continue


    stage = stage_map[desc]


    # 一个annotation可能120秒、300秒
    # 必须拆成30秒epoch

    n_epoch = int(duration // 30)


    for i in range(n_epoch):

        rows.append(
            {
                "start_sec": int(onset + i * 30),
                "stage": stage,
                "stage_name": desc,
            }
        )


# -------------------------------
# 保存
# -------------------------------

df = pd.DataFrame(rows)


df.to_csv(
    args.out,
    index=False
)


# -------------------------------
# 输出检查
# -------------------------------

print("\n前20行:")
print(df.head(20))


print(
    f"\n共生成 {len(df)} 个30秒epoch"
)

print(
    "输出文件:",
    args.out
)