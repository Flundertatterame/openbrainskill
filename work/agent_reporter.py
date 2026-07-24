"""Generate a Markdown report from real sleep-staging JSON artifacts.

This module deliberately does not invent metrics. If a field is missing, the
report says it is missing. Later, an LLM can be inserted after this step to
polish wording, but the factual substrate must remain these JSON files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STAGE_NAMES = {
    "0": "Wake",
    "1": "N1",
    "2": "N2",
    "3": "N3",
    "4": "REM",
}


def load_json(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def get_artifacts(state: dict[str, Any]) -> dict[str, Any]:
    return state.get("artifacts",{})


def fmt_float(value: Any) -> str:
    if isinstance(value, (float, int)):
        return f"{value:.4f}"
    return "missing"

def get_metric_value(metrics: dict[str, Any], key: str):

    # 新旧字段兼容

    if key == "n_aligned_epochs":

        return metrics.get(
            "n_aligned_epochs",
            metrics.get(
                "matched_epochs",
                "missing"
            )
        )


    if key == "macro_f1":

        # 旧格式
        if "macro_f1" in metrics:
            return metrics["macro_f1"]

        # 新格式 classification_report
        report = metrics.get(
            "classification_report"
        )

        if isinstance(report,list):

            values=[]

            for item in report:

                if isinstance(item,dict):

                    f1=item.get(
                        "f1-score"
                    )

                    if isinstance(f1,(int,float)):
                        values.append(f1)


            if values:
                return sum(values)/len(values)


        return "missing"



    if key == "cohen_kappa":

        return metrics.get(
            "cohen_kappa",
            "missing"
        )


    return metrics.get(
        key,
        "missing"
    )


def explain_metrics(metrics: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    acc = get_metric_value(metrics,"accuracy")
    macro_f1 = get_metric_value(metrics,"macro_f1")
    kappa = get_metric_value(metrics,"cohen_kappa")

    lines.append(f"- Accuracy: {fmt_float(acc)}")
    lines.append(f"- Macro-F1: {fmt_float(macro_f1)}")
    lines.append(f"- Cohen's Kappa: {fmt_float(kappa)}")

    if isinstance(acc, (float, int)) and isinstance(macro_f1, (float, int)):
        diff = abs(acc - macro_f1)
        lines.append(f"- Accuracy 与 Macro-F1 差值为 {diff:.4f}")
        if diff > 0.15:
            lines.append("- Accuracy 明显高于 Macro-F1，可能存在类别不平衡，不能只参考 Accuracy。")
        else:
            lines.append("- Accuracy 与 Macro-F1 差值较小，本报告不据此推断类别分布情况。")

    if isinstance(kappa, (float, int)):
        if kappa < 0.4:
            lines.append("- Kappa 偏低，模型与真实标注的一致性有限。")
        elif kappa < 0.6:
            lines.append("- Kappa 中等，结果有一定参考价值但仍需改进。")
        else:
            lines.append("- Kappa 较好，预测与真实标注具有较明显一致性。")

    return lines


def render_confusion_matrix(metrics: dict[str, Any]) -> list[str]:

    matrix = metrics.get(
        "confusion_matrix"
    )


    if matrix is None:
        return [
            "未找到混淆矩阵。"
        ]


    lines=[
        "| True \\ Pred | Wake | N1 | N2 | N3 | REM |",
        "|---|---:|---:|---:|---:|---:|"
    ]


    # 旧格式 dict

    if isinstance(matrix,dict):

        for stage in ["0","1","2","3","4"]:

            row=matrix.get(
                stage,
                {}
            )

            values=[
                str(
                    row.get(pred,0)
                )
                for pred in ["0","1","2","3","4"]
            ]

            lines.append(
                f"| {STAGE_NAMES[stage]} | "
                +
                " | ".join(values)
                +
                " |"
            )


    # 新格式 list

    elif isinstance(matrix,list):

        for idx,row in enumerate(matrix):

            if idx >=5:
                break


            values=[
                str(x)
                for x in row
            ]


            while len(values)<5:
                values.append("0")


            lines.append(
                f"| {STAGE_NAMES[str(idx)]} | "
                +
                " | ".join(values[:5])
                +
                " |"
            )


    else:

        return [
            "混淆矩阵格式无法解析。"
        ]


    return lines

def render_classification_report(metrics):

    report = metrics.get(
        "classification_report"
    )


    if not isinstance(report,list):

        return []


    lines=[]

    lines.append(
        "## Classification Report"
    )

    lines.append("")

    lines.append(
        "| Stage | Precision | Recall | F1 |"
    )

    lines.append(
        "|---|---:|---:|---:|"
    )


    for item in report:

        if isinstance(item,dict):

            lines.append(
                f"| {item.get('stage','missing')} "
                f"| {item.get('precision','missing')} "
                f"| {item.get('recall','missing')} "
                f"| {item.get('f1-score','missing')} |"
            )


    return lines

def explain_confusion_pattern(metrics: dict[str,Any])->list[str]:
    matrix = metrics.get("confusion_matrix")
    if not isinstance(matrix,dict):
        return []

    lines=[]

    if "1" in matrix:
        lines.append("- N1阶段混淆情况需要结合混淆矩阵观察。")

    return lines

def explain_diagnosis(diagnosis):

    result=[]

    for item in diagnosis:

        if isinstance(item,dict):

            msg=item.get("message","")

        else:

            msg=str(item)

        if "LSL" in msg:

            result.append(
                "确认 LSL 推流程序已启动。"
            )

        elif "NeuroSkill" in msg:

            result.append(
                "确认 daemon 已启动。"
            )

        elif "EDF" in msg:

            result.append(
                "检查 EDF 文件路径。"
            )

    return result[:5]

def build_llm_prompt(metrics: dict[str, Any],state: dict[str, Any],neuroskill: dict[str, Any] = {},status: dict[str, Any] = {}) -> str:
    prompt = """# 睡眠分期报告生成指令

## 角色定位
你是睡眠分期结果报告生成助手，仅基于提供的结构化数据生成自然语言说明，不做算法判断、不补充医学知识、不编造数据。

## 可用输入
1. metrics.json
2. agent_state.json
3. neuroskill_sleep.json
4. neuroskill_status.json
5. lsl_discover.json

## 硬性禁止规则
1. 禁止编造数据中未出现的指标、数值、结论
2. 禁止使用「优秀」「良好」「理想」「较差」等主观评价词汇
3. 禁止补充数据中没有的医学解释或健康建议
4. 数据缺失的字段必须明确标注「missing」

## 报告固定结构
1. 任务执行概览
2. 分期评估指标表格
3. 指标含义客观说明
4. 失败原因与下一步动作（有诊断信息时出现）
5. 引用文件清单

---
以下是本次运行的真实数据：
"""
    prompt += "\n### metrics.json\n"
    prompt += "```json\n" + json.dumps(metrics, ensure_ascii=False, indent=2) + "\n```\n"
    prompt += "\n### agent_state.json\n"
    prompt += "```json\n" + json.dumps(state, ensure_ascii=False, indent=2) + "\n```\n"
    prompt += "\n### neuroskill_sleep.json\n"
    if neuroskill:
        prompt += "```json\n" + json.dumps(neuroskill, ensure_ascii=False, indent=2) + "\n```\n"
    else:
        prompt += "missing（本次运行未提供）\n"
    
    prompt += "\n### neuroskill_status.json\n"
    if status:
        prompt += "```json\n" + json.dumps(status, ensure_ascii=False, indent=2) + "\n```\n"
    else:
        prompt += "missing（本次运行未提供）\n"
    return prompt

def build_report(metrics: dict[str, Any],state: dict[str, Any],neuroskill_status: dict[str, Any],lsl: dict[str, Any],neuroskill=None) -> str:#要添加参数的话记得把下面要调用的地方也改了，比如report = build_report那里
    lines: list[str] = []
    artifacts = get_artifacts(state)
    backend=state.get("backend_effective",state.get("backend"))
    run_status = state.get("status", "missing")

    if backend=="mne_baseline":
        title="# Fallback Sleep Report"
    elif run_status == "failed":
        title="# Failure Report"
    else:
        title="# Sleep Staging Report"

    lines.append(title)
    lines.append("")
    lines.append("## Run Summary")
    lines.append("## NeuroSkill Status")
    lines.append("")

    if not neuroskill_status:
        lines.append("missing")
    else:
        ok=neuroskill_status.get("ok")
        if ok:
            lines.append("- NeuroSkill daemon: Available")
        else:
            lines.append("- NeuroSkill daemon: Unavailable")

    if isinstance(lsl,list):
        lines.append(f"- LSL streams found: {len(lsl)}")
    elif isinstance(lsl,dict):
        streams=lsl.get("streams",[])
        lines.append(f"- LSL streams found: {len(streams)}")
    else:
        lines.append("- LSL: missing")

    lines.append("")
    lines.append(f"- Request: {state.get('request', 'missing')}")
    lines.append(f"- Backend: {state.get('backend', 'missing')}")
    lines.append(f"- Status: {state.get('status', 'missing')}")
    lines.append(f"- Aligned epochs: {get_metric_value(metrics,'n_aligned_epochs')}")
    lines.append("")

    lines.append("## Metrics")
    lines.append("")
    lines.extend(explain_metrics(metrics))
    lines.append("")

    lines.append("## Confusion Matrix")
    lines.append("")
    lines.extend(render_confusion_matrix(metrics))
    lines.append("")
    lines.extend(render_classification_report(metrics))
    lines.extend(explain_confusion_pattern(metrics))
    lines.append("")

    diagnosis = state.get("diagnosis", [])
    if diagnosis:
        lines.append("## Diagnostics")
        lines.append("")
        for item in diagnosis:
            if isinstance(item, dict):
                level = item.get("level", "info")
                message = item.get("message", "")
                evidence = item.get("evidence", "")
                next_action = item.get("next_action", "")
                lines.append(f"- [{level}] {message}")
                if evidence:
                    lines.append(f"  Evidence: `{evidence}`")
                if next_action:
                    lines.append(f"  Next action: {next_action}")
        lines.append("")

        tips=explain_diagnosis(diagnosis)
        if tips:
            lines.append("")
            lines.append("### Suggestions")
            for t in tips:
                lines.append(f"- {t}")
    
    next_action=state.get("next_action")
    if next_action:
        lines.append("")
        lines.append("## Next Action")
        lines.append("")
        lines.append(str(next_action))
    else:
        lines.append("missing")
        
    lines.append("## Referenced Files")
    lines.append("")
    lines.append("### Core Evaluation")
    lines.append("- metrics.json")
    lines.append("- aligned_predictions.csv")
    lines.append("")

    lines.append("### Agent State")
    lines.append("- agent_state.json")

    if neuroskill_status:
        lines.append("- neuroskill_status.json")
    else:
        lines.append("- neuroskill_status.json : missing")

    if lsl:
        lines.append("- lsl_discover.json")
    else:
        lines.append("- lsl_discover.json : missing")

    if neuroskill:
        lines.append("- neuroskill_sleep.json")
    else:
        lines.append("- neuroskill_sleep.json : missing")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("本报告只基于脚本输出的真实 JSON/CSV 结果生成。LLM 可以用于润色表达，但不能新增未出现在结果文件中的指标或结论。")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Markdown report from sleep-staging artifacts.")
    parser.add_argument("--metrics", required=True, help="Path to metrics.json.")
    parser.add_argument("--state", help="Path to agent_state.json.")
    parser.add_argument("--out", required=True, help="Output Markdown path.")
    parser.add_argument("--neuroskill", help="Path to neuroskill_sleep.json (optional)")
    parser.add_argument("--llm-prompt-out", help="Output LLM prompt text file path.")
    parser.add_argument("--neuroskill-status",help="Path to neuroskill_status.json")
    parser.add_argument("--lsl-discover",help="Path to lsl_discover.json")
    args = parser.parse_args()

    metrics = load_json(Path(args.metrics))
    state = load_json(Path(args.state)) if args.state else {}
    neuroskill = load_json(Path(args.neuroskill)) if args.neuroskill else {}
    status = load_json(Path(args.neuroskill_status)) \
        if args.neuroskill_status else {}
    lsl = load_json(Path(args.lsl_discover)) \
        if args.lsl_discover else {}
    if args.llm_prompt_out:
        prompt = build_llm_prompt(metrics, state, neuroskill, status)
        prompt_path = Path(args.llm_prompt_out)
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt, encoding="utf-8")
        print(f"LLM prompt written to: {prompt_path}")
    report = build_report(metrics,state,status,lsl,neuroskill)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()
