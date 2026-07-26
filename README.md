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

## LSL 独立工具

`run_lsl_demo.ps1` 会独立启动 EDF 推流、等待数据加载、执行 LSL discover，并把 metadata、state、discover 和日志写入同一个运行目录；推流失败时还会生成 error JSON。默认使用当前环境中的 Python，也可以通过 `-Python` 指定包含 `mne` 和 `pylsl` 的解释器。

```powershell
.\run_lsl_demo.ps1 -Edf "..\BrainFusion\SleepEDF\SC4001E0-PSG.edf"
```

Adapter 默认采用 `strict` 通道策略，保留真实 EDF 通道。只有兼容性实验需要把 2 个物理通道映射为 4 个 Muse 标签时，才显式传入 `--channel-policy duplicate`。NPY epoch 数据使用 `--input-layout epochs-samples` 或 source manifest 声明布局；频段特征 CSV 不能作为 EEG 波形推流。

```powershell
python .\work\edf_to_lsl_stream.py `
  --edf "..\BrainFusion\SleepEDF\SC4001E0-PSG.edf" `
  --metadata-json-out .\outputs\runs\lsl_demo\lsl_stream_metadata.json `
  --state-json-out .\outputs\runs\lsl_demo\lsl_stream_state.json `
  --error-json-out .\outputs\runs\lsl_demo\lsl_stream_error.json

python .\work\lsl_resolve_check.py `
  --seconds 5 `
  --out .\outputs\runs\lsl_demo\lsl_discover.json

python .\work\lsl_param_sweep.py `
  --edf "..\BrainFusion\SleepEDF\SC4001E0-PSG.edf" `
  --out .\outputs\runs\day15\lsl_sweep_summary.json
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
