"""Evaluate sleep staging predictions against ground truth labels.

Expected CSV format for both truth and prediction:

start_sec,stage
0,0
30,2
60,2

If prediction uses 5-second epochs, pass `--pred-epoch-sec 5`. The script will
aggregate predictions to the truth epoch length with majority voting.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


STAGES = ["0", "1", "2", "3", "4"]
STAGE_NAMES = {
    "0": "Wake",
    "1": "N1",
    "2": "N2",
    "3": "N3",
    "4": "REM",
}


def read_stage_csv(path: Path) -> list[tuple[float, str]]:
    rows: list[tuple[float, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "start_sec" not in reader.fieldnames or "stage" not in reader.fieldnames:
            raise ValueError(f"{path} must contain start_sec and stage columns")
        for row in reader:
            rows.append((float(row["start_sec"]), normalize_stage(row["stage"])))
    return rows


def normalize_stage(value: str) -> str:
    v = value.strip().upper()
    aliases = {
        "W": "0",
        "WAKE": "0",
        "N1": "1",
        "S1": "1",
        "N2": "2",
        "S2": "2",
        "N3": "3",
        "S3": "3",
        "S4": "3",
        "N4": "3",
        "REM": "4",
        "R": "4",
    }
    return aliases.get(v, v)


def aggregate_predictions(pred: list[tuple[float, str]], truth_starts: list[float], truth_epoch_sec: float) -> dict[float, str]:
    by_truth_epoch: dict[float, list[str]] = defaultdict(list)
    truth_starts_sorted = sorted(truth_starts)
    if not truth_starts_sorted:
        return {}

    start_to_index = {start: i for i, start in enumerate(truth_starts_sorted)}
    first = truth_starts_sorted[0]

    for pred_start, stage in pred:
        idx = int((pred_start - first) // truth_epoch_sec)
        if idx < 0 or idx >= len(truth_starts_sorted):
            continue
        epoch_start = truth_starts_sorted[idx]
        by_truth_epoch[epoch_start].append(stage)

    out: dict[float, str] = {}
    for start in truth_starts_sorted:
        votes = by_truth_epoch.get(start, [])
        if votes:
            out[start] = Counter(votes).most_common(1)[0][0]
    return out


def confusion_matrix(y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, int]]:
    matrix = {t: {p: 0 for p in STAGES} for t in STAGES}
    for t, p in zip(y_true, y_pred):
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1
    return matrix


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def macro_f1(matrix: dict[str, dict[str, int]]) -> float:
    scores: list[float] = []
    for stage in STAGES:
        tp = matrix[stage][stage]
        fp = sum(matrix[t][stage] for t in STAGES if t != stage)
        fn = sum(matrix[stage][p] for p in STAGES if p != stage)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        scores.append(f1)
    return sum(scores) / len(scores)


def cohen_kappa(y_true: list[str], y_pred: list[str], matrix: dict[str, dict[str, int]]) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    po = accuracy(y_true, y_pred)
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    pe = sum((true_counts[s] / n) * (pred_counts[s] / n) for s in STAGES)
    if abs(1.0 - pe) < 1e-12:
        return 0.0
    return (po - pe) / (1.0 - pe)


def evaluate(truth_rows: list[tuple[float, str]], pred_rows: list[tuple[float, str]], truth_epoch_sec: float) -> dict:
    truth_by_start = {start: stage for start, stage in truth_rows}
    pred_by_start = aggregate_predictions(pred_rows, list(truth_by_start.keys()), truth_epoch_sec)

    common_starts = sorted(set(truth_by_start) & set(pred_by_start))
    y_true = [truth_by_start[s] for s in common_starts]
    y_pred = [pred_by_start[s] for s in common_starts]

    matrix = confusion_matrix(y_true, y_pred)
    return {
        "n_truth_epochs": len(truth_rows),
        "n_pred_epochs_raw": len(pred_rows),
        "n_aligned_epochs": len(common_starts),
        "stage_names": STAGE_NAMES,
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(matrix),
        "cohen_kappa": cohen_kappa(y_true, y_pred, matrix),
        "confusion_matrix": matrix,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sleep staging labels.")
    parser.add_argument("--truth", required=True, help="Ground truth CSV with start_sec,stage.")
    parser.add_argument("--pred", required=True, help="Prediction CSV with start_sec,stage.")
    parser.add_argument("--truth-epoch-sec", type=float, default=30.0, help="Truth epoch length.")
    parser.add_argument("--out", help="Output JSON path.")
    args = parser.parse_args()

    result = evaluate(
        truth_rows=read_stage_csv(Path(args.truth)),
        pred_rows=read_stage_csv(Path(args.pred)),
        truth_epoch_sec=args.truth_epoch_sec,
    )

    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
