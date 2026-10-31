# 07 · MR7 关节盒 + 混合组（2026-10-30）

本目录的 `mug_recovery_chain.py` / `sim_select.py` / `takeover_bench.py` / `takeover_fixes.py` 是 MR7 版本，可与 05 中的 MR6 版本直接对比。

- `mr7_joint_box.json`：规划只在"专家演示 ∪ FACT 失败前缀"的每关节最小/最大值 ±0.4 rad 内进行（cuRobo 原先使用 URDF 的 ±10 rad 限位）；另加抓取翻转 π（flip）和逐次规划断言（boxchk）；筛选时启用 G8（关节范围 ±0.5 rad）。
- `build_rf_assets.py` / `rf_assets.py`：把 845 个失败状态冻结为可复用资产，按 MR6 结果分为 A（救回）/ B（有方案但没成功）/ C（无方案）/ U（未跑完）四档。
- `mug_mix_lists.py` / `audit_mix_lists.py` / `mug_mix_swap.py`：混合组列表（50 条恢复 + 50 条专家，seed 与 E 组对齐），训练前做数值审计，并按事先定好的规则替换 6 条有软标记的轨迹。
- `mug_pipeline_amax_mix.sh`：混合组训练（复用 E 组作对照）。

结果：混合组 step 1000/2000/3000 = 20/45/52（E 组 30/50/69）；step 3000 配对 McNemar p = 0.008。
