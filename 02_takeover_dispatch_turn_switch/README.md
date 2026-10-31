# 02 · 专家接管的第一批尝试（2026-10-27）

- `takeover_smoke.py`：从失败状态直接调用 RoboTwin 原生脚本 `play_once()` 接管。
- `recovery_dispatch.py` / `recovery_dispatch_multi.py`：按任务和状态特征选择接管计划，含同手规则（turn_switch、move_stapler_pad）。
- `collect_recovery.py` / `collect_switch.sh`：turn_switch 恢复轨迹采集（202 条，同手审计 0 违规）。
- `grasp_scan.py` 与各 `make_*_videos.py`：可行性扫描与审查视频。
