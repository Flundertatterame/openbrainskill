import argparse
import pandas as pd
import json


parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    required=True
)

parser.add_argument(
    "--csv-out",
    required=True
)

parser.add_argument(
    "--json-out",
    required=True
)


parser.add_argument(
    "--start-sec",
    type=int,
    default=0
)


args = parser.parse_args()


# 读取完整标签
df = pd.read_csv(args.input)


start = args.start_sec
end = start + 600


# 截取10分钟
segment = df[
    (df["start_sec"] >= start)
    &
    (df["start_sec"] < end)
]


# 保存csv

segment.to_csv(
    args.csv_out,
    index=False
)


info = {
    "source_file": args.input,
    "start_sec": start,
    "end_sec": end,
    "duration_sec": 600,
    "epochs": len(segment)
}


with open(
    args.json_out,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        info,
        f,
        indent=4,
        ensure_ascii=False
    )


print(segment.head(20))

print(
    "\n生成epoch数量:",
    len(segment)
)

print(
    "输出:",
    args.csv_out
)