# 06 · 第一轮 SFT：专家组 E vs 恢复组 R（2026-10-29 ~ 10-30）

- `npz_to_lerobot.py`：npz → LeRobot v2.1。字段 `end` 用于截断失败前缀；`failure_active_from` 把 episode 标为 FACT 失败 episode（整段不算 action loss）；`action_masked_until` 指定某帧之前不算 action loss（cuRobo 动起来之前的静置帧）。
- `fact_sft_configs/`：`mug_sft_novalue.py`（value 权重 0；total = visual + 10×action + 0.4×future_state）、`action_mask_patch.py`（以 monkeypatch 读取逐帧 mask，不修改 FACT 仓库）。`verify_masks.py` 逐帧核对 mask。
- `mug_pipeline_amax*.sh` / `train_mug_sft*.sh`：转换 → T5 → VAE → 训练（3194 步，有效 batch 32，ZeRO-2）。
- `eval_task*.sh` / `ckpt_eval_sched.py` / `snap_ckpts.sh`：checkpoint 一出现就自动排队评测。评测 seed 为 seeds100，与训练 seed 不重叠。`start_server.sh` 为 10-30 加了 GPU UUID 绑定后的版本。

结果（seeds100）：base 42；E 组 step 1000/2000/3000 = 30/50/69；旧 R 组 = 0（100 条恢复里 56 条关节角绕圈越界，最高 6.9 rad；视频和 G7 都看不出来）。
