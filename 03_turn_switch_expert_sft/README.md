# 03 · turn_switch 专家 SFT（2026-10-28）

第一次把数据接入 FACT 的 SFT：用专家演示做 SFT（E189 组）、训练后评测。
- `make_sft_config.py`：生成 FACT 训练配置（此处为最终版，10-29 后来为 hanging_mug 扩展过）。
- `fact_sft_configs/turn_switch_sft.py`、`train_sft.sh`、`post_train_eval.sh`、`r189_pipeline.sh`。

随后用户判断 turn_switch 的失败信号太弱，暂停该任务，主线转到 hanging_mug。npz→LeRobot 转换脚本的最终版放在 06。
