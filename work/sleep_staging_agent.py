"""Minimal SleepStagingAgent for the OpenBrainSkill MVP.

The agent is a workflow controller. It does not classify EEG by itself. Its job
is to translate a request into a structured plan, call deterministic tools,
record artifacts, diagnose failures, and ask the reporter to generate a report.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


@dataclass
class ToolSpec:
    step: str
    tool: str
    command: list[str]
    expected_output: str
    failure_policy: str


@dataclass
class Step:
    step: str
    tool: str
    command: list[str] = field(default_factory=list)
    status: str = "pending"
    expected_output: str = ""
    failure_policy: str = ""
    exit_code: int | None = None
    output: str = ""
    error: str = ""


@dataclass
class AgentState:
    run_id: str
    request: str
    backend: str
    status: str
    inputs: dict[str, str]
    plan: list[Step]
    artifacts: dict[str, str]
    diagnosis: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "request": self.request,
            "backend": self.backend,
            "status": self.status,
            "inputs": self.inputs,
            "plan": [step.__dict__ for step in self.plan],
            "artifacts": self.artifacts,
            "diagnosis": self.diagnosis,
        }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"_json_error": str(exc), "_path": str(path)}


def parse_json_text(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def run_command(command: list[str], cwd: Path) -> tuple[int, str, str]:
    completed = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=False)
    return completed.returncode, completed.stdout, completed.stderr


def build_tool_registry(args: argparse.Namespace, artifacts: dict[str, str]) -> dict[str, ToolSpec]:
    python = sys.executable

    return {
        "check_edf": ToolSpec(
            step="check_edf",
            tool="edf_to_lsl_stream.py",
            command=[
                python,
                str(ROOT / "edf_to_lsl_stream.py"),
                "--edf",
                args.psg,
                "--dry-run",
            ],
            expected_output="stdout contains selected EDF channels, LSL shape, and amplitude summary",
            failure_policy="Stop NeuroSkill route and ask B line to fix PSG path/channel loading.",
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
        "evaluate": ToolSpec(
            step="evaluate",
            tool="evaluate_sleep_staging.py",
            command=[
                python,
                str(ROOT / "evaluate_sleep_staging.py"),
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
                "--out",
                artifacts["final_report"],
            ],
            expected_output=artifacts["final_report"],
            failure_policy="Skip until metrics.json and agent_state.json exist.",
        ),
    }


def step_from_spec(spec: ToolSpec) -> Step:
    return Step(
        step=spec.step,
        tool=spec.tool,
        command=spec.command,
        expected_output=spec.expected_output,
        failure_policy=spec.failure_policy,
    )


def build_plan(args: argparse.Namespace, run_dir: Path) -> AgentState:
    run_id = run_dir.name
    artifacts = {
        "agent_plan": str(run_dir / "agent_plan.json"),
        "agent_state": str(run_dir / "agent_state.json"),
        "edf_check": str(run_dir / "edf_check.json"),
        "lsl_discover": str(run_dir / "lsl_discover.json"),
        "neuroskill_status": str(run_dir / "neuroskill_status.json"),
        "neuroskill_sleep": str(run_dir / "neuroskill_sleep.json"),
        "truth_labels": str(run_dir / "true_labels.csv"),
        "pred_labels": str(run_dir / "pred_labels.csv"),
        "metrics": str(run_dir / "metrics.json"),
        "final_report": str(run_dir / "final_report.md"),
    }

    registry = build_tool_registry(args, artifacts)
    sequence = ["check_edf", "check_lsl_discovery", "evaluate", "generate_report"]
    if args.backend == "neuroskill_lsl":
        sequence.insert(2, "check_neuroskill_status")
    plan = [step_from_spec(registry[name]) for name in sequence]

    return AgentState(
        run_id=run_id,
        request=args.request,
        backend=args.backend,
        status="planned",
        inputs={"psg_edf": args.psg, "hypnogram_edf": args.hypnogram or ""},
        plan=plan,
        artifacts=artifacts,
    )


def add_diagnosis(state: AgentState, level: str, message: str, evidence: str, next_action: str) -> None:
    item = {
        "level": level,
        "message": message,
        "evidence": evidence,
        "next_action": next_action,
    }
    if item not in state.diagnosis:
        state.diagnosis.append(item)


def add_preflight_diagnostics(state: AgentState) -> None:
    psg = Path(state.inputs.get("psg_edf", ""))
    hypnogram_value = state.inputs.get("hypnogram_edf", "")

    if not psg.exists():
        add_diagnosis(
            state,
            "error",
            "PSG EDF path does not exist.",
            str(psg),
            "Replace --psg with a real Sleep-EDF PSG file before executing the EDF/LSL route.",
        )

    if hypnogram_value and not Path(hypnogram_value).exists():
        add_diagnosis(
            state,
            "warning",
            "Hypnogram EDF path does not exist.",
            hypnogram_value,
            "Replace --hypnogram with the matching Sleep-EDF Hypnogram before evaluation.",
        )


def inspect_step_result(state: AgentState, step: Step) -> None:
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
            add_diagnosis(
                state,
                "error",
                "NeuroSkill daemon is not reachable.",
                str(evidence),
                "Start NeuroSkill/daemon, confirm the port, then rerun status. Keep B line evidence JSON for debugging.",
            )


def execute_plan(state: AgentState, run_dir: Path, execute: bool) -> AgentState:
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
        if step.step == "evaluate":
            if not Path(state.artifacts["truth_labels"]).exists() or not Path(state.artifacts["pred_labels"]).exists():
                step.status = "skipped"
                step.error = "truth_labels.csv or pred_labels.csv missing"
                add_diagnosis(
                    state,
                    "warning",
                    "Evaluation skipped because labels are missing.",
                    step.error,
                    "Generate true_labels.csv and pred_labels.csv, then rerun evaluation.",
                )
                continue

        if step.step == "generate_report" and not Path(state.artifacts["metrics"]).exists():
            step.status = "skipped"
            step.error = "metrics.json missing"
            add_diagnosis(
                state,
                "warning",
                "Report generation skipped because metrics.json is missing.",
                step.error,
                "Run evaluation after labels are ready, then generate the final report.",
            )
            continue

        step.status = "running"
        code, stdout, stderr = run_command(step.command, cwd=run_dir)
        step.exit_code = code
        step.output = stdout[-4000:]
        step.error = stderr[-4000:]
        step.status = "completed" if code == 0 else "failed"
        inspect_step_result(state, step)

        if step.status == "failed":
            add_diagnosis(
                state,
                "error",
                f"Step {step.step} failed.",
                step.error or step.output,
                diagnose_next_action(step.step),
            )
            if step.step in {"check_edf", "check_neuroskill_status"}:
                state.status = "fallback"
                break

    if state.status not in {"fallback", "failed"}:
        failed = any(step.status == "failed" for step in state.plan)
        state.status = "failed" if failed else "completed"
    return state


def diagnose_next_action(step: str) -> str:
    mapping = {
        "check_edf": "Check EDF path, channel names, and whether MNE can read the file.",
        "check_lsl_discovery": "Run the LSL stream script first, then repeat discovery. Check firewall if still empty.",
        "check_neuroskill_status": "Start NeuroSkill daemon/app and verify the port/token configuration.",
        "evaluate": "Check true_labels.csv and pred_labels.csv format: start_sec,stage.",
        "generate_report": "Check metrics.json and agent_state.json.",
    }
    return mapping.get(step, "Inspect stdout/stderr and rerun the step manually.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="OpenBrainSkill SleepStagingAgent MVP.")
    parser.add_argument("--request", required=True, help="Natural language user request.")
    parser.add_argument("--psg", required=True, help="PSG EDF path.")
    parser.add_argument("--hypnogram", help="Hypnogram EDF path.")
    parser.add_argument("--backend", choices=["neuroskill_lsl", "mne_baseline"], default="neuroskill_lsl")
    parser.add_argument("--run-dir", default="", help="Output run directory.")
    parser.add_argument("--execute", action="store_true", help="Actually execute available tool steps.")
    parser.add_argument("--lsl-check-seconds", type=float, default=5.0)
    parser.add_argument("--neuroskill-port", type=int, default=18444)
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else Path("outputs") / "runs" / f"run_{int(time.time())}"
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    state = build_plan(args, run_dir)
    write_json(Path(state.artifacts["agent_plan"]), {"plan": [step.__dict__ for step in state.plan]})
    state = execute_plan(state, run_dir, args.execute)
    write_json(Path(state.artifacts["agent_state"]), state.to_dict())
    print(json.dumps(state.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
