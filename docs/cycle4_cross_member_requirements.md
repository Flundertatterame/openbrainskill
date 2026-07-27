# 第四周期跨成员接口需求

本文只登记不属于林容增 Agent 职责范围的后续工作，不在本分支直接修改其他成员的实现。

## Reporter 负责人

建议合并 PR #16，或实现等价行为；不得改变现有 Agent 的状态和产物契约。

验收要求：

1. 正确读取当前 `evaluate_baseline.py` 输出中的：
   - `matched_epochs`
   - 列表形式的 `confusion_matrix`
   - `classification_report["macro avg"]["f1-score"]`
2. 报告必须展示 Macro-F1 和混淆矩阵，不能再显示为 `missing` 或“未找到”。
3. 从 `agent_state.json.artifacts` 展示每个产物的真实路径。
4. 当 `backend == backend_effective == "mne_baseline"` 时，报告标题应为正常的 baseline 报告，不能写成 Fallback Sleep Report。
5. fallback 报告必须同时展示：
   - `backend`：用户请求的路线；
   - `backend_effective`：实际产生结果的路线。
6. 使用现有 baseline 与 NeuroSkill fallback 运行目录各回归测试一次。

## NeuroSkill Client 负责人

Agent 只有在拿到可评估的 `pred_labels.csv` 后，才会把 NeuroSkill 分期判定为成功。

验收要求：

1. 使用当前需要认证的 NeuroSkill daemon API 创建会话并执行睡眠分析。
2. 把 NeuroSkill 返回的逐 epoch 结果转换成 `start_sec,stage` CSV 格式。
3. 将结果写入 Agent 提供的 `pred_labels` 产物路径，不能只返回状态 JSON。
4. HTTP 401、daemon 不可用、超时和响应格式错误都必须保留结构化 JSON 错误证据。
5. 生成合法 `pred_labels.csv` 后，Agent 的 `ensure_neuroskill_predictions` 步骤应通过，并且不得触发 baseline fallback。

## Pull Request 协调

- PR #10 与 `work/sleep_staging_agent.py` 直接重叠：请删除其中的 Agent 修改，或 rebase 后确保不覆盖第四周期实现。
- PR #11 的 Reporter 修改已被 PR #16 替代，不应覆盖新版 Reporter。
- PR #16 不修改 Agent 文件，可以与本分支组合；优先使用该 Reporter 实现。
- PR #15 已进入当前 `main`；Agent 以其中的 LSL Adapter 契约为准。
