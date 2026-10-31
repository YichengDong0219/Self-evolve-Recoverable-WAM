# 01 · 失败普查与想象探针（2026-10-27）

- `p1_census.sh` / `p1_label.py` / `p1_summarize.py`：在 adjust_bottle、hanging_mug、turn_switch、move_stapler_pad 上普查 FACT 的失败。
- `p1_render.py`：按记录的动作逐位重放并渲染（头部 | 左腕 | 右腕），同时核对重放结果与原始结果一致；后续所有评测视频都用它生成。
- `mug_failure_onset.py`：hanging_mug 失败起点规则。a：杯子与两只 TCP 脱离并下落；b：到过挂钩高度后回到桌面；c：静止 ≥ 96 步。
- `imagine_*` / `render_sbs.py`：把 FACT 想象的未来帧与真实帧并排对比（失败 episode 后期想象偏离明显升高）。
- `collect_mug_failures.sh`：在集群上批量采集 FACT 在 hanging_mug 上的失败 rollout。
