"""Minimal SleepStagingAgent for the OpenBrainSkill MVP.

The agent is a workflow controller. It does not classify EEG by itself. Its job
is to translate a request into a structured plan, call deterministic tools,
record artifacts, diagnose failures, and ask the reporter to generate a report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO


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
    runtime: dict[str, Any] = field(default_factory=dict)
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
            "runtime": self.runtime,
            "next_action": self.next_action,
            "diagnosis": self.diagnosis,
        }


@dataclass
class LSLProcessHandle:
    """保存由 Agent 启动的 Adapter 进程和日志句柄，供 finally 可靠清理。"""

    process: subprocess.Popen[Any]
    log_handle: BinaryIO
    command: list[str]


def utc_now() -> str:
    """返回带时区的 UTC ISO-8601 时间，供跨组件状态文件统一使用。"""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def selected_lsl_source(args: argparse.Namespace) -> tuple[str, str]:
    """返回 Adapter 的输入参数名和绝对路径。

    优先级由 Agent CLI 明确控制：NPY、CSV 二者互斥；都未提供时使用 PSG EDF。
    输出示例为 ``("--edf", "C:/.../SC4002E0-PSG.edf")``。
    """

    if args.lsl_from_npy:
        return "--from-npy", args.lsl_from_npy
    if args.lsl_from_csv:
        return "--from-csv", args.lsl_from_csv
    return "--edf", args.psg


def build_lsl_adapter_command(args: argparse.Namespace, artifacts: dict[str, str]) -> list[str]:
    """按沈博昊 Adapter 契约构造参数列表，不拼接 shell 命令字符串。"""

    source_flag, source_path = selected_lsl_source(args)
    command = [
        args.python,
        "-u",
        str(ROOT / "edf_to_lsl_stream.py"),
        source_flag,
        source_path,
        "--stream-name",
        args.lsl_stream_name,
        "--stream-type",
        args.lsl_stream_type,
        "--sample-rate",
        str(args.lsl_sample_rate),
        "--minutes",
        str(args.lsl_minutes),
        "--channel-policy",
        args.lsl_channel_policy,
        "--metadata-json-out",
        artifacts["lsl_stream_metadata"],
        "--state-json-out",
        artifacts["lsl_stream_state"],
        "--error-json-out",
        artifacts["lsl_stream_error"],
    ]
    if source_flag == "--edf":
        command.extend(["--channels", args.lsl_channels])
    if args.lsl_labels:
        command.extend(["--lsl-labels", args.lsl_labels])
    if source_flag == "--from-npy":
        command.extend(["--input-layout", args.lsl_input_layout])
    if args.lsl_source_manifest:
        command.extend(["--source-manifest", args.lsl_source_manifest])
    return command


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
        "start_lsl_stream": ToolSpec(
            step="start_lsl_stream",
            tool="edf_to_lsl_stream.py",
            command=build_lsl_adapter_command(args, artifacts),
            expected_output=artifacts["lsl_stream_metadata"],
            failure_policy="Stop the Adapter and switch to baseline when PSG/Hypnogram inputs are available.",
        ),
        "check_lsl_discovery": ToolSpec(
            step="check_lsl_discovery",
            tool="lsl_resolve_check.py",
            command=[
                python,
                str(ROOT / "lsl_resolve_check.py"),
                "--seconds",
                str(args.lsl_check_seconds),
                "--type",
                args.lsl_stream_type,
                "--out",
                artifacts["lsl_discover"],
            ],
            expected_output=artifacts["lsl_discover"],
            failure_policy="Require this run's source_id and metadata fields to match before continuing.",
        ),
        "check_neuroskill_status": ToolSpec(
            step="check_neuroskill_status",
            tool="neuroskill_client.py",
            command=[
                python,
                str(ROOT / "neuroskill_client.py"),
                "status",
                "--host",
                args.neuroskill_host,
                "--port",
                str(args.neuroskill_port),
                "--timeout",
                str(args.neuroskill_timeout),
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
                "--host",
                args.neuroskill_host,
                "--port",
                str(args.neuroskill_port),
                "--timeout",
                str(args.neuroskill_timeout),
                "--out",
                artifacts["neuroskill_lsl_discover"],
            ],
            expected_output=artifacts["neuroskill_lsl_discover"],
            failure_policy="If NeuroSkill sees no LSL stream, write diagnosis and switch later to mne_baseline fallback.",
        ),
        "ensure_neuroskill_predictions": ToolSpec(
            step="ensure_neuroskill_predictions",
            tool="artifact_check",
            command=[],
            expected_output=artifacts["pred_labels"],
            failure_policy="If NeuroSkill has not produced pred_labels.csv, run the real baseline fallback.",
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
        "parsed_request": str(run_dir / "parsed_request.json"),
        "edf_check": str(run_dir / "edf_check.json"),
        "lsl_discover": str(run_dir / "lsl_discover.json"),
        "lsl_stream_metadata": str(run_dir / "lsl_stream_metadata.json"),
        "lsl_stream_state": str(run_dir / "lsl_stream_state.json"),
        "lsl_stream_error": str(run_dir / "lsl_stream_error.json"),
        "lsl_stream_log": str(run_dir / "lsl_stream.log"),
        "neuroskill_status": str(run_dir / "neuroskill_status.json"),
        "neuroskill_lsl_discover": str(run_dir / "neuroskill_lsl_discover.json"),
        "neuroskill_sleep": str(run_dir / "neuroskill_sleep.json"),
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
    args.lsl_from_npy = normalize_input_path(args.lsl_from_npy)
    args.lsl_from_csv = normalize_input_path(args.lsl_from_csv)
    args.lsl_source_manifest = normalize_input_path(args.lsl_source_manifest)

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
            *(["check_edf"] if not (args.lsl_from_npy or args.lsl_from_csv) else []),
            "start_lsl_stream",
            "check_lsl_discovery",
            "check_neuroskill_status",
            "check_neuroskill_lsl_discover",
            "ensure_neuroskill_predictions",
        ]
    plan = [step_from_spec(registry[name]) for name in sequence]

    return AgentState(
        run_id=run_id,
        request=args.request,
        backend=args.backend,
        backend_effective=args.backend,
        task=task,
        status="planned",
        inputs={
            "psg_edf": task["psg"],
            "hypnogram_edf": task["hypnogram"],
            "lsl_source_kind": selected_lsl_source(args)[0].removeprefix("--"),
            "lsl_source_path": selected_lsl_source(args)[1],
        },
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


def write_parsed_request(state: AgentState) -> None:
    """单独保存自然语言解析结果，供 Day 14 验收和下游工具复用。

    输入是原始 request 与 Agent 已规范化的结构化 task；输出字段保持在 JSON
    顶层，便于其他成员直接读取 sample_id、backend_requested 和数据路径。
    """

    write_json(
        Path(state.artifacts["parsed_request"]),
        {"request": state.request, **state.task},
    )


def switch_to_fallback(state: AgentState, reason: str, evidence: str, next_action: str) -> None:
    """当 NeuroSkill/LSL 主路线不可用时，把有效后端切换到 mne_baseline。"""

    state.backend_effective = "mne_baseline"
    state.status = "fallback"
    add_diagnosis(state, "warning", reason, evidence, next_action)


def tail_text(path: Path, max_chars: int = 4000) -> str:
    """读取 UTF-8 日志末尾；日志不存在时返回空字符串。"""

    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]


def write_lsl_agent_error(state: AgentState, message: str, error_type: str) -> None:
    """当 Adapter 尚未来得及写错误文件时，由 Agent 补齐结构化错误证据。"""

    path = Path(state.artifacts["lsl_stream_error"])
    if not path.is_file():
        write_json(
            path,
            {
                "component": "sleep_staging_agent",
                "status": "failed",
                "failed_at": utc_now(),
                "error_type": error_type,
                "message": message,
            },
        )


def start_lsl_adapter(
    state: AgentState,
    step: Step,
    run_dir: Path,
    startup_timeout: float,
) -> LSLProcessHandle | None:
    """启动长期运行的 EDF/NPY/CSV→LSL Adapter 并等待 metadata。

    输入是参数列表形式的 ``step.command``；输出是仍存活的进程句柄。
    stdout/stderr 合并写入 ``lsl_stream.log``，PID 同步登记到 Agent runtime。
    """

    for key in (
        "lsl_stream_metadata",
        "lsl_stream_state",
        "lsl_stream_error",
        "lsl_stream_log",
        "lsl_discover",
    ):
        path = Path(state.artifacts[key])
        if path.is_file():
            path.unlink()

    log_path = Path(state.artifacts["lsl_stream_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("wb")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        process = subprocess.Popen(
            step.command,
            cwd=str(run_dir),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
    except Exception as exc:
        log_handle.close()
        step.status = "failed"
        step.error = f"Could not start LSL Adapter: {exc}"
        write_lsl_agent_error(state, step.error, type(exc).__name__)
        return None

    handle = LSLProcessHandle(process=process, log_handle=log_handle, command=step.command)
    state.runtime.update(
        {
            "lsl_process_pid": process.pid,
            "lsl_process_started_at": utc_now(),
            "lsl_process_command": step.command,
        }
    )
    metadata_path = Path(state.artifacts["lsl_stream_metadata"])
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if metadata_path.is_file():
            metadata = read_json(metadata_path)
            required = ("source_id", "name", "channel_count", "nominal_srate")
            missing = [key for key in required if metadata.get(key) in (None, "")]
            if not missing and process.poll() is None:
                step.status = "completed"
                step.exit_code = None
                step.output = json.dumps(metadata, ensure_ascii=False, indent=2)
                step.stdout_summary = (
                    f"pid={process.pid}, source_id={metadata['source_id']}, "
                    f"channels={metadata['channel_count']}, rate={metadata['nominal_srate']}"
                )
                return handle
            if missing:
                step.status = "failed"
                step.error = f"LSL metadata missing required fields: {', '.join(missing)}"
                write_lsl_agent_error(state, step.error, "InvalidMetadata")
                return handle

        exit_code = process.poll()
        if exit_code is not None:
            step.status = "failed"
            step.exit_code = exit_code
            step.error = tail_text(log_path) or f"LSL Adapter exited before metadata with code {exit_code}"
            write_lsl_agent_error(state, step.error, "AdapterExitedEarly")
            return handle
        time.sleep(0.2)

    step.status = "failed"
    step.error = f"Timed out after {startup_timeout:g}s waiting for LSL metadata"
    write_lsl_agent_error(state, step.error, "MetadataTimeout")
    return handle


def compare_lsl_stream(metadata: dict[str, Any], stream: dict[str, Any]) -> list[str]:
    """比较 Adapter metadata 与 discover stream，返回所有不匹配字段。

    ``source_id`` 必须精确相等；采样率比较 metadata.nominal_srate 与
    discover.sample_rate，并允许 1e-3 Hz 的浮点误差。
    """

    differences: list[str] = []
    for metadata_key, stream_key in (("source_id", "source_id"), ("name", "name")):
        if metadata.get(metadata_key) != stream.get(stream_key):
            differences.append(
                f"{metadata_key}: expected={metadata.get(metadata_key)!r}, actual={stream.get(stream_key)!r}"
            )
    try:
        if int(metadata.get("channel_count")) != int(stream.get("channel_count")):
            differences.append(
                f"channel_count: expected={metadata.get('channel_count')!r}, actual={stream.get('channel_count')!r}"
            )
    except (TypeError, ValueError):
        differences.append("channel_count is missing or invalid")
    try:
        expected_rate = float(metadata.get("nominal_srate"))
        actual_rate = float(stream.get("sample_rate"))
        if not math.isclose(expected_rate, actual_rate, rel_tol=1e-6, abs_tol=1e-3):
            differences.append(f"sample_rate: expected={expected_rate}, actual={actual_rate}")
    except (TypeError, ValueError):
        differences.append("sample_rate is missing or invalid")
    return differences


def discover_started_lsl_stream(
    state: AgentState,
    step: Step,
    run_dir: Path,
    handle: LSLProcessHandle,
    timeout: float,
) -> None:
    """循环执行 LSL discover，直到精确找到本次 Adapter 创建的流或超时。"""

    metadata = read_json(Path(state.artifacts["lsl_stream_metadata"]))
    expected_source_id = metadata.get("source_id")
    deadline = time.monotonic() + timeout
    attempts = 0
    last_stdout = ""
    last_stderr = ""
    last_payload: dict[str, Any] = {}
    last_differences: list[str] = []

    while time.monotonic() < deadline:
        if handle.process.poll() is not None:
            step.status = "failed"
            step.exit_code = handle.process.returncode
            step.error = "LSL Adapter exited while waiting for discovery.\n" + tail_text(
                Path(state.artifacts["lsl_stream_log"])
            )
            return

        attempts += 1
        code, stdout, stderr = run_command(step.command, cwd=run_dir)
        last_stdout, last_stderr = stdout, stderr
        payload = read_json(Path(state.artifacts["lsl_discover"]))
        last_payload = payload
        streams = payload.get("streams", []) if isinstance(payload, dict) else []
        source_candidate = next(
            (
                item
                for item in streams
                if isinstance(item, dict) and item.get("source_id") == expected_source_id
            ),
            None,
        )
        last_differences = (
            compare_lsl_stream(metadata, source_candidate) if source_candidate is not None else [
                f"source_id {expected_source_id!r} was not discovered"
            ]
        )
        enriched = dict(payload) if isinstance(payload, dict) else {}
        enriched.update(
            {
                "expected_stream": {
                    "source_id": metadata.get("source_id"),
                    "name": metadata.get("name"),
                    "channel_count": metadata.get("channel_count"),
                    "nominal_srate": metadata.get("nominal_srate"),
                },
                "matched_stream": source_candidate,
                "match_ok": source_candidate is not None and not last_differences,
                "match_differences": last_differences,
                "attempts": attempts,
            }
        )
        write_json(Path(state.artifacts["lsl_discover"]), enriched)
        if enriched["match_ok"]:
            step.status = "completed"
            step.exit_code = code
            step.output = json.dumps(enriched, ensure_ascii=False, indent=2)
            step.stdout_summary = (
                f"Matched source_id={expected_source_id} after {attempts} discover attempt(s)"
            )
            step.error = stderr[-4000:]
            return
        time.sleep(0.2)

    step.status = "failed"
    step.exit_code = None
    step.output = json.dumps(last_payload, ensure_ascii=False, indent=2) if last_payload else last_stdout[-4000:]
    step.error = (
        f"No matching LSL stream within {timeout:g}s after {attempts} attempt(s): "
        + "; ".join(last_differences)
        + (f"\n{last_stderr[-2000:]}" if last_stderr else "")
    )


def stop_lsl_adapter(
    state: AgentState,
    handle: LSLProcessHandle | None,
    timeout: float,
) -> None:
    """在所有退出路径停止 Adapter：terminate→等待→kill，并更新 running state。"""

    if handle is None:
        return
    process = handle.process
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        state.runtime["lsl_process_exit_code"] = process.returncode
        state.runtime["lsl_process_stopped_at"] = utc_now()
    finally:
        if not handle.log_handle.closed:
            handle.log_handle.flush()
            handle.log_handle.close()

    state_path = Path(state.artifacts["lsl_stream_state"])
    adapter_state = read_json(state_path)
    # Adapter 已经记录 failed/completed 时保留原始证据，只接管仍为 running 的状态。
    if adapter_state.get("status") == "running":
        adapter_state["status"] = "stopped_by_agent"
        adapter_state["stopped_at"] = utc_now()
        write_json(state_path, adapter_state)


BASELINE_SEQUENCE = [
    "check_edf",
    "extract_labels",
    "extract_features",
    "baseline_predict",
    "align_predictions",
    "evaluate",
    "generate_report",
]


def activate_baseline_fallback(
    state: AgentState,
    args: argparse.Namespace,
) -> None:
    """把完整 baseline 步骤动态追加到失败的 NeuroSkill plan，并持久化计划。"""

    registry = build_tool_registry(args, state.artifacts)
    existing = {item.step for item in state.plan}
    for name in BASELINE_SEQUENCE:
        if name not in existing:
            state.plan.append(step_from_spec(registry[name]))
            existing.add(name)
    write_json(Path(state.artifacts["agent_plan"]), {"plan": [step.__dict__ for step in state.plan]})


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

    psg_is_primary_input = state.backend_effective == "mne_baseline" or state.inputs.get("lsl_source_kind") == "edf"
    if psg_is_primary_input and (not psg_value or not Path(psg_value).is_file()):
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
    elif state.backend == "neuroskill_lsl" and (
        not psg_value
        or not Path(psg_value).is_file()
        or not hypnogram_value
        or not Path(hypnogram_value).is_file()
    ):
        add_diagnosis(
            state,
            "warning",
            "Automatic baseline fallback is unavailable without matching PSG and Hypnogram EDF files.",
            f"psg={psg_value or '<empty>'}, hypnogram={hypnogram_value or '<empty>'}",
            "Provide --psg and --hypnogram if the NeuroSkill route must fall back automatically.",
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

    if step.step == "check_lsl_discovery" and step.status == "failed":
        switch_to_fallback(
            state,
            "The LSL stream started by this Agent could not be verified.",
            step.error or step.output,
            "Stop the Adapter and run the complete mne_baseline fallback.",
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
        "start_lsl_stream": [state.inputs.get("lsl_source_path", "")],
        "check_lsl_discovery": [state.artifacts["lsl_stream_metadata"]],
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
        "start_lsl_stream": ["check_edf"],
        "check_lsl_discovery": ["start_lsl_stream"],
        "check_neuroskill_status": ["check_lsl_discovery"],
        "check_neuroskill_lsl_discover": ["check_lsl_discovery", "check_neuroskill_status"],
        "ensure_neuroskill_predictions": ["check_neuroskill_lsl_discover"],
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

    if state.backend_effective != state.backend:
        required = [name for name in BASELINE_SEQUENCE if name != "generate_report"]
        return "fallback" if all(step_status(state, name) == "completed" for name in required) else "failed"
    statuses = [step.status for step in state.plan if step.step != "generate_report"]
    if any(status == "failed" for status in statuses):
        return "failed"
    if any(status == "skipped" for status in statuses):
        return "failed"
    return "completed"


def final_run_status(state: AgentState) -> str:
    """计算最终状态；主路线失败只有在完整 baseline 成功后才算 fallback。"""

    if state.backend_effective != state.backend:
        return (
            "fallback"
            if all(step_status(state, name) == "completed" for name in BASELINE_SEQUENCE)
            else "failed"
        )
    statuses = [step.status for step in state.plan]
    return "failed" if any(status in {"failed", "skipped"} for status in statuses) else "completed"


def execute_plan(
    state: AgentState,
    run_dir: Path,
    execute: bool,
    args: argparse.Namespace,
) -> AgentState:
    """执行当前计划，并负责 LSL 进程生命周期及动态 baseline fallback。"""

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

    lsl_handle: LSLProcessHandle | None = None
    fallback_activated = False
    primary_only_steps = {
        "start_lsl_stream",
        "check_lsl_discovery",
        "check_neuroskill_status",
        "check_neuroskill_lsl_discover",
        "ensure_neuroskill_predictions",
    }

    try:
        # Python 的 list iterator 会继续处理运行期间追加的 fallback steps。
        for step in state.plan:
            if fallback_activated and step.step in primary_only_steps:
                step.status = "skipped"
                step.error = "Primary NeuroSkill route was abandoned after fallback activation."
                write_json(Path(state.artifacts["agent_state"]), state.to_dict())
                continue

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
                if state.backend == "neuroskill_lsl" and step.step in primary_only_steps:
                    switch_to_fallback(
                        state,
                        f"NeuroSkill route cannot continue at {step.step}.",
                        step.error,
                        "Stop LSL and execute the complete mne_baseline fallback.",
                    )
                if state.backend_effective != state.backend and not fallback_activated:
                    stop_lsl_adapter(state, lsl_handle, args.lsl_stop_timeout)
                    lsl_handle = None
                    activate_baseline_fallback(state, args)
                    fallback_activated = True
                write_json(Path(state.artifacts["agent_state"]), state.to_dict())
                continue

            if step.step == "generate_report":
                # Reporter 读取 agent_state.json，所以先写入包含 fallback 证据的阶段状态。
                state.status = provisional_run_status(state)
                write_json(Path(state.artifacts["agent_state"]), state.to_dict())

            if step.step not in {"start_lsl_stream", "check_lsl_discovery"} and step.output_path:
                stale_output = Path(step.output_path)
                if stale_output.is_file():
                    stale_output.unlink()

            step.status = "running"
            if step.step == "start_lsl_stream":
                lsl_handle = start_lsl_adapter(state, step, run_dir, args.lsl_startup_timeout)
                if step.status == "failed":
                    switch_to_fallback(
                        state,
                        "The LSL Adapter failed to start or publish metadata.",
                        step.error,
                        "Stop the Adapter and execute the complete mne_baseline fallback.",
                    )
            elif step.step == "check_lsl_discovery":
                if lsl_handle is None:
                    step.status = "failed"
                    step.error = "LSL process handle is missing"
                else:
                    discover_started_lsl_stream(
                        state,
                        step,
                        run_dir,
                        lsl_handle,
                        args.lsl_discover_timeout,
                    )
                inspect_step_result(state, step)
            elif step.step == "ensure_neuroskill_predictions":
                if Path(state.artifacts["pred_labels"]).is_file():
                    step.status = "completed"
                    step.exit_code = 0
                    step.output = state.artifacts["pred_labels"]
                else:
                    step.status = "failed"
                    step.error = "NeuroSkill pred_labels.csv has not been produced by the current client integration."
                    switch_to_fallback(
                        state,
                        "NeuroSkill did not produce an evaluation-ready prediction artifact.",
                        step.error,
                        "Use mne_baseline to produce pred_labels.csv, metrics.json, and final_report.md.",
                    )
            else:
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

            # Adapter 意外退出也属于主路线失败，不能继续假设 LSL 仍然可用。
            if (
                lsl_handle is not None
                and step.step in primary_only_steps
                and step.step != "start_lsl_stream"
                and lsl_handle.process.poll() is not None
                and state.backend_effective == state.backend
            ):
                switch_to_fallback(
                    state,
                    "The LSL Adapter exited before the NeuroSkill route completed.",
                    tail_text(Path(state.artifacts["lsl_stream_log"])),
                    "Run the complete mne_baseline fallback.",
                )

            if step.status == "failed":
                add_diagnosis(
                    state,
                    "error",
                    f"Step {step.step} failed.",
                    step.error or step.output,
                    diagnose_next_action(step.step),
                )

            if state.backend_effective != state.backend and not fallback_activated:
                # fallback 启动前立即停流；finally 会作为第二道兜底再次执行清理。
                stop_lsl_adapter(state, lsl_handle, args.lsl_stop_timeout)
                lsl_handle = None
                activate_baseline_fallback(state, args)
                fallback_activated = True

            write_json(Path(state.artifacts["agent_state"]), state.to_dict())

    except KeyboardInterrupt:
        state.status = "failed"
        add_diagnosis(
            state,
            "error",
            "Agent execution was interrupted by the user.",
            "KeyboardInterrupt",
            "Inspect saved artifacts and rerun with a new run directory.",
        )
        raise
    except Exception as exc:
        state.status = "failed"
        add_diagnosis(
            state,
            "error",
            "Unexpected Agent execution error.",
            f"{type(exc).__name__}: {exc}",
            "Inspect agent_state.json and lsl_stream.log before retrying.",
        )
        raise
    finally:
        stop_lsl_adapter(state, lsl_handle, args.lsl_stop_timeout)
        write_json(Path(state.artifacts["agent_state"]), state.to_dict())

    state.status = final_run_status(state)
    if state.status == "fallback":
        add_diagnosis(
            state,
            "info",
            "The complete mne_baseline fallback finished successfully.",
            state.artifacts["final_report"],
            "Review metrics.json and final_report.md; fix the recorded NeuroSkill/LSL cause before retrying the primary route.",
        )
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
    # 显式支持 Day 15 的 dry-run/execute，两者互斥；不传参数时仍按旧行为 dry-run。
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Generate plan/state artifacts without running tools.")
    mode.add_argument("--execute", action="store_true", help="Actually execute available tool steps.")
    parser.add_argument("--lsl-channels", default="EEG Fpz-Cz,EEG Pz-Oz")
    parser.add_argument("--lsl-stream-name", default="SleepEDF_LSL")
    parser.add_argument("--lsl-stream-type", default="EEG")
    parser.add_argument("--lsl-sample-rate", type=float, default=256.0)
    parser.add_argument("--lsl-minutes", type=float, default=10.0)
    parser.add_argument("--lsl-channel-policy", choices=["strict", "duplicate"], default="strict")
    parser.add_argument("--lsl-labels", default="", help="Use Muse labels only for explicit duplicate experiments.")
    parser.add_argument("--lsl-from-npy", default="", help="Optional raw waveform NPY input for the Adapter.")
    parser.add_argument("--lsl-from-csv", default="", help="Optional raw waveform CSV input for the Adapter.")
    parser.add_argument(
        "--lsl-input-layout",
        choices=["auto", "channels-samples", "epochs-samples", "epochs-channels-samples"],
        default="auto",
    )
    parser.add_argument("--lsl-source-manifest", default="")
    parser.add_argument("--lsl-check-seconds", type=float, default=5.0)
    parser.add_argument("--lsl-startup-timeout", type=float, default=30.0)
    parser.add_argument("--lsl-discover-timeout", type=float, default=20.0)
    parser.add_argument("--lsl-stop-timeout", type=float, default=5.0)
    parser.add_argument("--neuroskill-host", default="127.0.0.1")
    parser.add_argument("--neuroskill-port", type=int, default=18444)
    parser.add_argument("--neuroskill-timeout", type=float, default=3.0)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for tool subprocesses; defaults to the current interpreter.",
    )
    args = parser.parse_args()
    if args.lsl_from_npy and args.lsl_from_csv:
        parser.error("--lsl-from-npy and --lsl-from-csv are mutually exclusive")
    for name in (
        "lsl_sample_rate",
        "lsl_minutes",
        "lsl_check_seconds",
        "lsl_startup_timeout",
        "lsl_discover_timeout",
        "lsl_stop_timeout",
        "neuroskill_timeout",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than 0")

    run_dir = Path(args.run_dir) if args.run_dir else PROJECT_ROOT / "outputs" / "runs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    state = build_plan(args, run_dir)
    write_parsed_request(state)
    write_json(Path(state.artifacts["agent_plan"]), {"plan": [step.__dict__ for step in state.plan]})
    state = execute_plan(state, run_dir, args.execute, args)
    write_json(Path(state.artifacts["agent_state"]), state.to_dict())
    print(json.dumps(state.to_dict(), indent=2, ensure_ascii=False))
    if args.execute and state.status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
