import argparse
import mne
import pandas as pd


parser = argparse.ArgumentParser()

parser.add_argument(
    "--hypnogram",
    required=True
)

parser.add_argument(
    "--out",
    required=True
)


args = parser.parse_args()


print("读取 Hypnogram EDF...")

# 读取睡眠标注
annotations = mne.read_annotations(args.hypnogram)


print("annotation数量:", len(annotations))


# 睡眠阶段映射
stage_map = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4
}


rows = []


for onset, duration, description in zip(
        annotations.onset,
        annotations.duration,
        annotations.description):

    if description in stage_map:

        rows.append({
            "onset": onset,
            "duration": duration,
            "stage": description,
            "label": stage_map[description]
        })


df = pd.DataFrame(rows)


print("\n前20行:")
print(df.head(20))


df.head(20).to_csv(
    args.out,
    index=False
)


print("\n保存完成:")
print(args.out)