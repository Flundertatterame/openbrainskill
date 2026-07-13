# Sleep Staging Agent Report

## Run Summary

- Request: 请对 SC4002E0 做睡眠分期并生成报告
- Backend: neuroskill_lsl
- Status: planned
- Aligned epochs: missing

## Metrics

- Accuracy: missing
- Macro-F1: missing
- Cohen's Kappa: missing

## Confusion Matrix

未找到混淆矩阵。

## Diagnostics

- [error] PSG EDF path does not exist.
  Evidence: `C:\missing\SC4002E0-PSG.edf`
  Next action: Replace --psg with a real Sleep-EDF PSG file before executing the EDF/LSL route.
- [warning] Hypnogram EDF path does not exist.
  Evidence: `C:\missing\SC4002EC-Hypnogram.edf`
  Next action: Replace --hypnogram with the matching Sleep-EDF Hypnogram before evaluation.
- [info] Dry run only. No tools were executed.
  Evidence: `C:\Users\linrz\Desktop\openbrainskill\outputs_examples\lrz_cycle1_dry_run\agent_plan.json`
  Next action: Run again with --execute after checking paths.

## Interpretation

本报告只基于脚本输出的真实 JSON/CSV 结果生成。LLM 可以用于润色表达，但不能新增未出现在结果文件中的指标或结论。
