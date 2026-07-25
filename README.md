# OpenBrainSkill Sleep Staging MVP

这是一个面向小组项目的睡眠分期 MVP 工程。当前目标不是直接做完整“多模态脑信号框架”，而是先完成一个可运行、可验证、可展示的最小闭环：

```text
Sleep-EDF 数据
  -> MNE 读取 EDF 与 Hypnogram
  -> EDF 转 LSL 虚拟实时 EEG 流
  -> NeuroSkill 发现并连接 LSL 流
  -> NeuroSkill sleep 输出分期结果
  -> 与 Sleep-EDF 标注对齐评估
  -> Agent 生成自然语言报告
```

## 分工

```text
A 线：Agent / LLM / OpenBrainSkill 封装


B 线：NeuroSkill 调度 / EDF-LSL / 数据接入

```

## 目录

```text
work/
  sleep_staging_agent.py
  agent_reporter.py
  neuroskill_client.py
  edf_to_lsl_stream.py
  lsl_resolve_check.py
  evaluate_sleep_staging.py
  agent_state_schema.json
  openbrainskill_sleep_staging_contract.json
  requirements.txt

docs/
  24天项目推进计划.md

outputs_examples/
  run_agent_plan_demo/
```

## 快速检查

```powershell
python .\work\sleep_staging_agent.py --help
python .\work\agent_reporter.py --help
python .\work\neuroskill_client.py --help
python .\work\edf_to_lsl_stream.py --help
python .\work\evaluate_sleep_staging.py --help
```

## LSL 推流

推荐使用 PowerShell 演示脚本。它会启动 EDF -> LSL 推流、执行 Python LSL 自检，并将 metadata、运行状态、discover 结果和日志写入同一个 run 目录。

默认 `strict` 通道策略会保留真实 EDF 通道和标签。需要将 2 通道 Sleep-EDF 映射为 4 通道 Muse 标签时，必须显式传入 `--channel-policy duplicate`，输出 metadata 会标记为合成重复通道。`features.csv` 是每 epoch 的频段特征，不能用作 LSL EEG；它应交给 `mne_baseline`。`epochs.npy` 则应通过 `--input-layout epochs-samples`（或 source manifest）声明其 epoch 维度。

```powershell
.\run_lsl_demo.ps1 -Edf "..\BrainFusion\SleepEDF\SC4001E0-PSG.edf"
```

也可以直接调用 Adapter。`--state-json-out` 记录 `running`、`dry_run_completed`、`completed` 或 `failed`；`--error-json-out` 在失败时提供 `error_type`、`message` 和下一步命令建议。

```powershell
python .\work\edf_to_lsl_stream.py `
  --edf "..\BrainFusion\SleepEDF\SC4001E0-PSG.edf" `
  --minutes 10 `
  --metadata-json-out .\outputs\runs\lsl_demo\lsl_stream_metadata.json `
  --state-json-out .\outputs\runs\lsl_demo\lsl_stream_state.json `
  --error-json-out .\outputs\runs\lsl_demo\lsl_stream_error.json

python .\work\lsl_param_sweep.py `
  --edf "..\BrainFusion\SleepEDF\SC4001E0-PSG.edf" `
  --out .\outputs\runs\day15\lsl_sweep_summary.json `
  --neuroskill-port 18444
```

## 生成 Agent 执行计划

```powershell
python .\work\sleep_staging_agent.py `
  --request "请对 SC4002E0 做睡眠分期并生成报告" `
  --psg "C:\path\to\SC4002E0-PSG.edf" `
  --hypnogram "C:\path\to\SC4002EC-Hypnogram.edf" `
  --backend neuroskill_lsl `
  --run-dir ".\outputs\runs\run_001"
```

## 注意

- 不要把 Sleep-EDF 原始 EDF 文件提交到 Git。
- 不要把模型权重、大量运行日志、个人路径配置提交到 Git。
- `outputs/runs/` 是本地运行输出目录，默认不提交。
- 可展示的小样例输出放到 `outputs_examples/`。
