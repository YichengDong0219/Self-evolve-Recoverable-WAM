# 05 · MR6 整链规划 + simulate-and-select + 门控（2026-10-28 ~ 10-29）

本目录中 `mug_recovery_chain.py`、`sim_select.py`、`takeover_bench.py`、`takeover_fixes.py` 是 **MR6 原版**（MR7 改动前的备份）；MR7 版本见 07。

- `mug_recovery_chain.py`（MR6）：等杯子静止 → 松手/避让 → 整条链路在关节空间预规划（预抓取 → 直线接近 → 抬升 10/7/5 cm → 预挂 → 挂），以架子和杯子为障碍物，夹爪中心不低于桌面 + 3 cm；每次只尝试一次；松手后保持 100 步再复查成功。
- `sim_select.py`：同一失败状态按候选逐个重放尝试，取第一个通过全部门控的。
- `quality_gates.py`：G1 稳定成功、G2 无回零、G3 单次抓取、G4 直接挂、G5 抓前不扰动、G6 挂杯前无物理异常。
- `mug_sft_lists.py`：配对（检出跌落 + 同 seed 专家成功 + G7 单帧关节跳变 ≤ 0.2 rad）。
- `collect_expert_demo.py` / `expert_batch.py`：同 seed 原生专家演示。
- `mug_stream*.sh` 等：多机流式采集。
- `results/sft_lists_final_g7_1029/`：第一轮 100 对 seed 对齐的数据列表。

采集规模：1606 次 FACT rollout → 942 次失败 → 130 条入选 → 100 对。
