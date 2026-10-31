# 04 · hanging_mug 接管规则 MR1–MR5（2026-10-28）

- 普查（`mug_state_census.py`）：25 条失败中 22 条是杯子在挂钩高度（约第 284–318 步）掉落；14 条侧躺、9 条直立。
- `mug_takeover_step.py`：确定接管点（48 步边界）。dropped 指到过挂钩高度后又落下；stalled 指从没抬起来、一直不动。
- `mug_recovery.py`（MRv1–v4：静置 + 回零/停靠 + 在中间位重新摆正）：最高 9/25，但含回零和交接，不能用作训练数据。
- `mug_recovery_direct.py`（MR5：从掉落处直接抓、直接挂，不回零）：4–5/25，轨迹干净。
- `takeover_diag.py`：物理异常标记（掉落、出桌、甩飞）。`bench_runner.sh` 等为基准测试工具。

用户在此阶段定下硬约束：禁止回零，数据质量优先，产率低可以接受。
注：`mug_recovery.py` 与 `takeover_diag.py` 为最终版本（10-29 仍有修改）。
