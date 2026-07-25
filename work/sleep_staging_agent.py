"""Minimal SleepStagingAgent for the OpenBrainSkill MVP.

The agent is a workflow controller. It does not classify EEG by itself. Its job
is to translate a request into a structured plan, call deterministic tools,
record artifacts, diagnose failures, and ask the reporter to generate a report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
SAMPLE_ID_PATTERN = re.compile(r"\b(SC\d{4}[A-Z]\d?)\b", re.IGNORECASE)


@dataclass
class ToolSpec:
    """描述一个可被 Agent 调用的工具，以及这个工具失败时该怎么处理。"""

    step: str
    tool: str
    command: list[str]
    expected_output: str
    failure_policy: str


@dataclass
class Step:
    """记录执行计划中的一个具体步骤，以及这一步运行后的状态和输出。"""

    step: str
    tool: str
    command: list[str] = field(default_factory=list)
    status: str = "pending"
    expected_output: str = ""
    failure_policy: str = ""
    exit_code: int | None = None
    output_path: str = ""
    stdout_summary: str = ""
    output: str = ""
    error: str = ""


@dataclass
class AgentState:
    """保存一次 Agent 运行的完整状态，包括输入、计划、产物和诊断信息。"""

    run_id: str
    request: str
    backend: str
    backend_effective: str
    task: dict[str, Any]
    status: str
    inputs: dict[str, str]
    plan: list[Step]
    artifacts: dict[str, str]
    next_action: str = ""
    diagnosis: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """把 AgentState 转成普通 dict，方便写入 agent_state.json。"""

        return {
            "run_id": self.run_id,
            "request": self.request,
            "backend": self.backend,
            "backend_effective": self.backend_effective,
            "task": self.task,
            "status": self.status,
            "inputs": self.inputs,
            "plan": [step.__dict__ for step in self.plan],
            "artifacts": self.artifacts,
            "next_action": self.next_action,
            "diagnosis": self.diagnosis,
        }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """把一个字典写成格式化 JSON 文件，并自动创建父目录。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件；如果文件不存在或 JSON 格式错误，就返回可诊断的空结果。"""

    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"_json_error": str(exc), "_path": str(path)}


def parse_json_text(text: str) -> dict[str, Any]:
    """尝试把一段字符串解析成 JSON；解析失败时返回空字典。"""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def run_command(command: list[str], cwd: Path) -> tuple[int, str, str]:
    """以 UTF-8 运行工具，并返回退出码、标准输出和错误输出。

    Windows 的非 UTF-8 控制台会让含中文日志的 Python 工具在 print 时崩溃；
    因此 Agent 在进程边界统一设置编码，工具本身无需依赖用户终端代码页。
    """

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        env=env,
    )
    return completed.returncode, completed.stdout, completed.stderr


def summarize_text(text: str, max_lines: int = 8) -> str:
    """从一大段终端输出中提取前几行有效内容，便于写入状态摘要。"""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[:max_lines])


def extract_sample_id(request: str) -> str:
    """从自然语言请求中提取 Sleep-EDF 样本编号，例如 SC4002E0。"""

    match = SAMPLE_ID_PATTERN.search(request)
    return match.group(1).upper() if match else ""


def load_task_json(path: str) -> dict[str, Any]:
    """读取结构化任务 JSON；没有提供路径时返回空字典。"""

    if not path:
        return {}
    return read_json(Path(path))


def normalize_input_path(value: str) -> str:
    """把用户输入文件统一解析为绝对路径，空值保持为空。

    Agent 会在 run 目录中启动工具；提前规范化路径可避免相对路径被子进程
    错误地解释成 ``run_dir/relative/path``。相对路径以启动 Agent 的目录为准。
    """

    if not value:
        return ""
    return str(Path(value).expanduser().resolve())


def build_structured_task(args: argparse.Namespace) -> dict[str, Any]:
    """把命令行参数或 task-json 统一整理成 Agent 内部使用的结构化任务。"""

    task = load_task_json(args.task_json)
    if task:
        return {
            "task": task.get("task", "sleep_staging"),
            "sample_id": task.get("sample_id", extract_sample_id(args.request)),
            "backend_requested": task.get("backend", args.backend),
            "need_report": task.get("need_report", True),
            "psg": normalize_input_path(task.get("psg", args.psg or "")),
            "hypnogram": normalize_input_path(task.get("hypnogram", args.hypnogram or "")),
        }

    return {
        "task": "sleep_staging",
        "sample_id": extract_sample_id(args.request),
        "backend_requested": args.backend,
        "need_report": True,
        "psg": normalize_input_path(args.psg or ""),
        "hypnogram": normalize_input_path(args.hypnogram or ""),
    }


def get_stream_count(payload: dict[str, Any]) -> int | None:
    """从 LSL discover 的不同返回格式中尽量提取发现到的流数量。"""

    if isinstance(payload.get("count"), int):
        return payload["count"]
    streams = payload.get("streams")
    if isinstance(streams, list):
        return len(streams)

    response = payload.get("response")
    if isinstance(response, dict):
        if isinstance(response.get("count"), int):
            return response["count"]
        for key in ["streams", "devices", "lsl_streams", "discovered_devices"]:
            value = response.get(key)
            if isinstance(value, list):
                return len(value)
    if isinstance(response, list):
        return len(response)
    return None


def build_tool_registry(args: argparse.Namespace, artifacts: dict[str, str]) -> dict[str, ToolSpec]:
    """建立工具注册表。

    每个 ToolSpec 都明确声明命令输入、预期输出和失败策略。第三周期的
    baseline 路线只使用离线 EDF/CSV 文件，不依赖 LSL 或 NeuroSkill。
    """

    python = args.python

    return {
        "check_edf": ToolSpec(
            step="check_edf",
            tool="check_sleep_edf.py",
            command=[
                python,
                str(ROOT / "check_sleep_edf.py"),
                "--edf",
                args.psg,
                "--json-out",
                artifacts["edf_check"],
            ],
            expected_output=artifacts["edf_check"],
            failure_policy="Stop PSG-dependent steps and report the EDF path/read error.",
        ),
        "extract_labels": ToolSpec(
            step="extract_labels",
            tool="extract_sleep_edf_labels.py",
            command=[
                python,
                str(ROOT / "extract_sleep_edf_labels.py"),
                "--hypnogram",
                args.hypnogram or "",
                "--out",
                artifacts["truth_labels"],
            ],
            expected_output=artifacts["truth_labels"],
            failure_policy="Stop label-dependent steps and check the matching Hypnogram EDF.",
        ),
        "extract_features": ToolSpec(
            step="extract_features",
            tool="extract_features.py",
            command=[
                python,
                str(ROOT / "extract_features.py"),
                "--psg",
                args.psg,
                "--out",
                artifacts["features"],
            ],
            expected_output=artifacts["features"],
            failure_policy="Stop baseline prediction and check EEG channels, sampling rate, and dependencies.",
        ),
        "baseline_predict": ToolSpec(
            step="baseline_predict",
            tool="mne_baseline.py",
            command=[
                python,
                str(ROOT / "mne_baseline.py"),
                "--psg",
                args.psg,
                "--truth",
                artifacts["truth_labels"],
                "--out",
                artifacts["pred_labels"],
            ],
            expected_output=artifacts["pred_labels"],
            failure_policy="Stop evaluation and inspect the PSG/truth-label inputs.",
        ),
        "check_lsl_discovery": ToolSpec(
            step="check_lsl_discovery",
            tool="lsl_resolve_check.py",
            command=[
                python,
                str(ROOT / "lsl_resolve_check.py"),
                "--seconds",
                str(args.lsl_check_seconds),
            ],
            expected_output="stdout JSON with count and streams",
            failure_policy="If count is 0, keep state but recommend starting EDF->LSL stream and LSL parameter sweep.",
        ),
        "check_neuroskill_status": ToolSpec(
            step="check_neuroskill_status",
            tool="neuroskill_client.py",
            command=[
                python,
                str(ROOT / "neuroskill_client.py"),
                "status",
                "--port",
                str(args.neuroskill_port),
                "--out",
                artifacts["neuroskill_status"],
            ],
            expected_output=artifacts["neuroskill_status"],
            failure_policy="If daemon is unreachable, record JSON evidence and switch later to mne_baseline fallback.",
        ),
        "check_neuroskill_lsl_discover": ToolSpec(
            step="check_neuroskill_lsl_discover",
            tool="neuroskill_client.py",
            command=[
                python,
                str(ROOT / "neuroskill_client.py"),
                "lsl-discover",
                "--port",
                str(args.neuroskill_port),
                "--out",
                artifacts["neuroskill_lsl_discover"],
            ],
            expected_output=artifacts["neuroskill_lsl_discover"],
            failure_policy="If NeuroSkill sees no LSL stream, write diagnosis and switch later to mne_baseline fallback.",
        ),
        "align_predictions": ToolSpec(
            step="align_predictions",
            tool="aligned_predictions.py",
            command=[
                python,
                str(ROOT / "aligned_predictions.py"),
                "--truth",
                artifacts["truth_labels"],
                "--pred",
                artifacts["pred_labels"],
                "--out",
                artifacts["aligned_predictions"],
            ],
            expected_output=artifacts["aligned_predictions"],
            failure_policy="Skip until true_labels.csv and pred_labels.csv both exist.",
        ),
        "evaluate": ToolSpec(
            step="evaluate",
            tool="evaluate_baseline.py",
            command=[
                python,
                str(ROOT / "evaluate_baseline.py"),
                "--truth",
                artifacts["truth_labels"],
                "--pred",
                artifacts["pred_labels"],
                "--out",
                artifacts["metrics"],
            ],
            expected_output=artifacts["metrics"],
            failure_policy="Skip until true_labels.csv and pred_labels.csv both exist.",
        ),
        "generate_report": ToolSpec(
            step="generate_report",
            tool="agent_reporter.py",
            command=[
                python,
                str(ROOT / "agent_reporter.py"),
                "--metrics",
                artifacts["metrics"],
                "--state",
                artifacts["agent_state"],
                "--neuroskill-status",
                artifacts["neuroskill_status"],
                "--lsl-discover",
                artifacts["lsl_discover"],
                "--out",
                artifacts["final_report"],
            ],
            expected_output=artifacts["final_report"],
            failure_policy="Skip until metrics.json and agent_state.json exist.",
        ),
    }


def step_from_spec(spec: ToolSpec) -> Step:
    """把工具说明 ToolSpec 转成一次计划中真正要执行的 Step。"""

    return Step(
        step=spec.step,
        tool=spec.tool,
        command=spec.command,
        expected_output=spec.expected_output,
        failure_policy=spec.failure_policy,
        output_path=spec.expected_output if spec.expected_output.endswith((".json", ".csv", ".md")) else "",
    )


def build_plan(args: argparse.Namespace, run_dir: Path) -> AgentState:
    """根据用户请求、输入路径和 backend 生成完整执行计划与输出文件路径。"""

    run_id = run_dir.name
    artifacts = {
        "agent_plan": str(run_dir / "agent_plan.json"),
        "agent_state": str(run_dir / "agent_state.json"),
        "edf_check": str(run_dir / "edf_check.json"),
        "lsl_discover": str(run_dir / "lsl_discover.json"),
        "neuroskill_status": str(run_dir / "neuroskill_status.json"),
        "neuroskill_lsl_discover": str(run_dir / "neuroskill_lsl_discover.json"),
        "neuroskill_sleep": str(run_dir / "neuroskill_sleep.json"),
        "truth_labels": str(run_dir / sample_id / "true_labels.csv"),
        "pred_labels": str(run_dir / sample_id / "pred_labels.csv"),
        "truth_labels": str(run_dir / "true_labels.csv"),
        "features": str(run_dir / "features.csv"),
        "pred_labels": str(run_dir / "pred_labels.csv"),
        "aligned_predictions": str(run_dir / "aligned_predictions.csv"),
        "metrics": str(run_dir / "metrics.json"),
        "final_report": str(run_dir / "final_report.md"),
    }

    task = build_structured_task(args)
    args.psg = task["psg"]
    args.hypnogram = task["hypnogram"]
    args.backend = task["backend_requested"]

    registry = build_tool_registry(args, artifacts)

    # 输入/输出边界：baseline 从两个 EDF 开始，依次生成 JSON、CSV、指标和报告。
    # NeuroSkill 路线仍保留现有发现/状态检查，等待 B 线提供正式 sleep 输出转换步骤。
    if args.backend == "mne_baseline":
        sequence = [
            "check_edf",
            "extract_labels",
            "extract_features",
            "baseline_predict",
            "align_predictions",
            "evaluate",
            "generate_report",
        ]
    else:
        sequence = [
            "check_edf",
            "check_lsl_discovery",
            "check_neuroskill_status",
            "check_neuroskill_lsl_discover",
            "align_predictions",
            "evaluate",
            "generate_report",
        ]
    plan = [step_from_spec(registry[name]) for name in sequence]

    return AgentState(
        run_id=run_id,
        request=args.request,
        backend=args.backend,
        backend_effective=args.backend,
        task=task,
        status="planned",
        inputs={"psg_edf": task["psg"], "hypnogram_edf": task["hypnogram"]},
        plan=plan,
        artifacts=artifacts,
    )


def add_diagnosis(state: AgentState, level: str, message: str, evidence: str, next_action: str) -> None:
    """向 Agent 状态中追加一条诊断信息，并同步更新下一步建议。"""

    item = {
        "level": level,
        "message": message,
        "evidence": evidence,
        "next_action": next_action,
    }
    if item not in state.diagnosis:
        state.diagnosis.append(item)
    state.next_action = next_action


def switch_to_fallback(state: AgentState, reason: str, evidence: str, next_action: str) -> None:
    """当 NeuroSkill/LSL 主路线不可用时，把有效后端切换到 mne_baseline。"""

    state.backend_effective = "mne_baseline"
    state.status = "fallback"
    add_diagnosis(state, "warning", reason, evidence, next_action)


def write_step_artifact(state: AgentState, step: Step) -> None:
    """把某些关键步骤的执行结果保存成独立 JSON，方便之后复查。"""

    payload = {
        "ok": step.status == "completed",
        "step": step.step,
        "tool": step.tool,
        "command": step.command,
        "exit_code": step.exit_code,
        "stdout_summary": step.stdout_summary,
        "stdout": step.output,
        "stderr": step.error,
    }

    if step.step == "check_edf" and not Path(state.artifacts["edf_check"]).exists():
        # check_sleep_edf.py 正常情况下会写业务 JSON；仅在脚本提前异常时写诊断包装。
        write_json(Path(state.artifacts["edf_check"]), payload)
    elif step.step == "check_lsl_discovery":
        parsed = parse_json_text(step.output)
        write_json(Path(state.artifacts["lsl_discover"]), parsed or payload)


def add_preflight_diagnostics(state: AgentState) -> None:
    """在真正执行前先检查输入文件路径，并把明显问题写入 diagnosis。"""

    psg_value = state.inputs.get("psg_edf", "")
    hypnogram_value = state.inputs.get("hypnogram_edf", "")

    if not psg_value or not Path(psg_value).is_file():
        add_diagnosis(
            state,
            "error",
            "PSG EDF path does not exist.",
            psg_value or "<empty path>",
            "Replace --psg with a real Sleep-EDF PSG file before executing the EDF/LSL route.",
        )

    if state.backend_effective == "mne_baseline" and (
        not hypnogram_value or not Path(hypnogram_value).is_file()
    ):
        add_diagnosis(
            state,
            "warning",
            "Hypnogram EDF path does not exist.",
            hypnogram_value or "<empty path>",
            "Replace --hypnogram with the matching Sleep-EDF Hypnogram before evaluation.",
        )


def inspect_step_result(state: AgentState, step: Step) -> None:
    """根据某一步的输出结果判断是否需要追加诊断或触发 fallback。"""

    if step.step == "check_edf":
        payload = read_json(Path(state.artifacts["edf_check"]))
        if not payload.get("exists", False) or payload.get("error"):
            step.status = "failed"
            add_diagnosis(
                state,
                "error",
                "PSG EDF inspection failed.",
                str(payload.get("error") or payload),
                "Check the PSG path and confirm that MNE can read this EDF file.",
            )

    if step.step == "check_lsl_discovery" and step.status == "completed":
        payload = parse_json_text(step.output)
        if payload.get("count") == 0:
            add_diagnosis(
                state,
                "warning",
                "No LSL stream was discovered.",
                step.output.strip() or "count=0",
                "Start edf_to_lsl_stream.py in another terminal, then rerun discovery. If still empty, test stream type EEG and firewall settings.",
            )

    if step.step == "check_neuroskill_status":
        payload = read_json(Path(state.artifacts["neuroskill_status"]))
        if payload and not payload.get("ok", False):
            step.status = "failed"
            evidence = payload.get("message") or payload.get("error") or json.dumps(payload, ensure_ascii=False)
            switch_to_fallback(
                state,
                "NeuroSkill daemon is not reachable.",
                str(evidence),
                "Start NeuroSkill/daemon and confirm the port. Before it is fixed, continue with mne_baseline fallback.",
            )

    if step.step == "check_neuroskill_lsl_discover":
        payload = read_json(Path(state.artifacts["neuroskill_lsl_discover"]))
        if payload and not payload.get("ok", False):
            step.status = "failed"
            evidence = payload.get("message") or payload.get("error") or json.dumps(payload, ensure_ascii=False)
            switch_to_fallback(
                state,
                "NeuroSkill LSL discovery failed.",
                str(evidence),
                "Check NeuroSkill LSL settings and keep the mne_baseline fallback available.",
            )
            return

        stream_count = get_stream_count(payload)
        if stream_count == 0:
            switch_to_fallback(
                state,
                "NeuroSkill did not discover any LSL stream.",
                json.dumps(payload, ensure_ascii=False),
                "Run lsl_param_sweep or adjust stream name/type/channels/sample rate, then retry NeuroSkill discovery.",
            )
        elif stream_count is not None and stream_count > 0:
            add_diagnosis(
                state,
                "info",
                "NeuroSkill discovered LSL stream candidates.",
                f"stream_count={stream_count}",
                "Next B-line step: attempt connect/session and then call NeuroSkill sleep.",
            )


def step_status(state: AgentState, step_name: str) -> str | None:
    """返回指定计划步骤的当前状态；该步骤不在当前 backend 计划中时返回 None。"""

    for item in state.plan:
        if item.step == step_name:
            return item.status
    return None


def missing_step_requirements(state: AgentState, step: Step) -> list[str]:
    """检查某一步运行前必须存在的文件和已完成的上游步骤。

    输入是当前 AgentState 与待执行 Step，输出是缺失条件的可读字符串列表。
    这样上游失败时下游会被明确标记为 skipped，而不会生成看似成功的空结果。
    """

    files: dict[str, list[str]] = {
        "extract_labels": [state.inputs.get("hypnogram_edf", "")],
        "extract_features": [state.inputs.get("psg_edf", "")],
        "baseline_predict": [state.inputs.get("psg_edf", ""), state.artifacts["truth_labels"]],
        "align_predictions": [state.artifacts["truth_labels"], state.artifacts["pred_labels"]],
        "evaluate": [
            state.artifacts["truth_labels"],
            state.artifacts["pred_labels"],
            state.artifacts["aligned_predictions"],
        ],
        "generate_report": [state.artifacts["metrics"]],
    }
    upstream: dict[str, list[str]] = {
        "extract_features": ["check_edf"],
        "baseline_predict": ["check_edf", "extract_labels"],
        "align_predictions": ["extract_labels", "baseline_predict"],
        "evaluate": ["align_predictions"],
        "generate_report": ["evaluate"],
    }

    missing: list[str] = []
    for value in files.get(step.step, []):
        if not value or not Path(value).is_file():
            missing.append(value or "<empty path>")
    for dependency in upstream.get(step.step, []):
        status = step_status(state, dependency)
        if status is not None and status != "completed":
            missing.append(f"step:{dependency}={status}")
    return missing


def provisional_run_status(state: AgentState) -> str:
    """在 Reporter 启动前计算可写入 agent_state.json 的阶段性运行状态。"""

    statuses = [step.status for step in state.plan if step.step != "generate_report"]
    if any(status == "failed" for status in statuses):
        return "failed"
    if any(status == "skipped" for status in statuses):
        return "failed"
    if state.backend_effective != state.backend:
        return "fallback"
    return "completed"


def execute_plan(state: AgentState, run_dir: Path, execute: bool) -> AgentState:
    """按顺序执行计划中的步骤；如果是 dry-run，则只写计划不运行工具。"""

    state.status = "running" if execute else "planned"
    add_preflight_diagnostics(state)

    if not execute:
        add_diagnosis(
            state,
            "info",
            "Dry run only. No tools were executed.",
            state.artifacts["agent_plan"],
            "Run again with --execute after checking paths.",
        )
        return state

    for step in state.plan:
        missing = missing_step_requirements(state, step)
        if missing:
            step.status = "skipped"
            step.error = "Missing requirements: " + ", ".join(missing)
            add_diagnosis(
                state,
                "warning",
                f"Step {step.step} skipped because its inputs are not ready.",
                step.error,
                diagnose_next_action(step.step),
            )
            continue

        if step.step == "generate_report":
            # Reporter 的输入包含 agent_state.json，因此必须在调用 Reporter 前持久化。
            state.status = provisional_run_status(state)
            write_json(Path(state.artifacts["agent_state"]), state.to_dict())

        # 同一个 run 目录允许重跑，但当前步骤不能把旧产物误当成本次成功输出。
        if step.output_path:
            stale_output = Path(step.output_path)
            if stale_output.is_file():
                stale_output.unlink()

        step.status = "running"
        code, stdout, stderr = run_command(step.command, cwd=run_dir)
        step.exit_code = code
        step.output = stdout[-4000:]
        step.error = stderr[-4000:]
        step.stdout_summary = summarize_text(stdout)
        step.status = "completed" if code == 0 else "failed"
        if step.status == "completed" and step.output_path and not Path(step.output_path).is_file():
            step.status = "failed"
            step.error = f"Tool exited with code 0 but did not create {step.output_path}"
        write_step_artifact(state, step)
        inspect_step_result(state, step)

        if step.status == "failed":
            add_diagnosis(
                state,
                "error",
                f"Step {step.step} failed.",
                step.error or step.output,
                diagnose_next_action(step.step),
            )
        # 每一步结束立即保存状态，便于中断后复盘，也保证 Reporter 读取到真实执行记录。
        write_json(Path(state.artifacts["agent_state"]), state.to_dict())

    failed = any(step.status == "failed" for step in state.plan)
    skipped = any(step.status == "skipped" for step in state.plan)
    if failed or skipped:
        state.status = "failed"
    elif state.backend_effective != state.backend:
        state.status = "fallback"
    else:
        state.status = "completed"
    return state


def diagnose_next_action(step: str) -> str:
    """根据失败的步骤名称，返回一条适合初学者继续排查的建议。"""

    mapping = {
        "check_edf": "Check EDF path, channel names, and whether MNE can read the file.",
        "extract_labels": "Check the Hypnogram EDF path and its Sleep-EDF annotations.",
        "extract_features": "Check PSG EEG channels, sampling rate, and MNE/SciPy dependencies.",
        "baseline_predict": "Check PSG and true_labels.csv before running the baseline.",
        "check_lsl_discovery": "Run the LSL stream script first, then repeat discovery. Check firewall if still empty.",
        "check_neuroskill_status": "Start NeuroSkill daemon/app and verify the port/token configuration.",
        "check_neuroskill_lsl_discover": "Start EDF->LSL stream first. If Python can discover it but NeuroSkill cannot, run LSL parameter sweep.",
        "align_predictions": "Check true_labels.csv and pred_labels.csv format: start_sec,stage.",
        "evaluate": "Check aligned_predictions.csv and the evaluator inputs.",
        "generate_report": "Check metrics.json and agent_state.json.",
    }
    return mapping.get(step, "Inspect stdout/stderr and rerun the step manually.")


def main() -> None:
    """命令行入口：解析参数、创建 run 目录、生成计划、执行并输出状态 JSON。"""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="OpenBrainSkill SleepStagingAgent MVP.")
    parser.add_argument("--request", required=True, help="Natural language user request.")
    parser.add_argument("--task-json", default="", help="Optional structured task JSON path.")
    parser.add_argument("--psg", default="", help="PSG EDF path.")
    parser.add_argument("--hypnogram", help="Hypnogram EDF path.")
    parser.add_argument("--backend", choices=["neuroskill_lsl", "mne_baseline"], default="neuroskill_lsl")
    parser.add_argument("--run-dir", default="", help="Output run directory.")
    parser.add_argument("--execute", action="store_true", help="Actually execute available tool steps.")
    parser.add_argument("--lsl-check-seconds", type=float, default=5.0)
    parser.add_argument("--neuroskill-port", type=int, default=18444)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for tool subprocesses; defaults to the current interpreter.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else PROJECT_ROOT / "outputs" / "runs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    state = build_plan(args, run_dir)
    write_json(Path(state.artifacts["agent_plan"]), {"plan": [step.__dict__ for step in state.plan]})
    state = execute_plan(state, run_dir, args.execute)
    write_json(Path(state.artifacts["agent_state"]), state.to_dict())
    print(json.dumps(state.to_dict(), indent=2, ensure_ascii=False))
    if args.execute and state.status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
