import argparse
import pandas as pd
import mne

parser = argparse.ArgumentParser()

parser.add_argument("--hypnogram", required=True)
parser.add_argument("--out", required=True)

args = parser.parse_args()

annotations = mne.read_annotations(args.hypnogram)

stage_map = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}

rows = []

for onset, duration, desc in zip(
    annotations.onset,
    annotations.duration,
    annotations.description,
):
    if desc not in stage_map:
        continue

    stage = stage_map[desc]

    # 每30秒输出一条
    n_epoch = int(duration // 30)

    for i in range(n_epoch):
        rows.append({
            "start_sec": int(onset + i * 30),
            "stage": stage,
            "stage_name": desc,
        })

df = pd.DataFrame(rows)

df.to_csv(args.out, index=False)

print(df.head(20))

print(f"\n共生成 {len(df)} 个epoch")
print(f"输出文件：{args.out}")