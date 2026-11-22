# 12 · 挂杯 TTS 最小闭环（分支 `mug-tts-loop`，2026-11-21 起）

协议见 [PROTOCOL.md](PROTOCOL.md)（开跑前冻结，后附实现细节补充）。

**当前状态（2026-11-22 10:00）：第 0 轮完成，critic 门槛未通过（增益 +0.038，95% CI [−0.002, +0.084]），按协议停在第 1 轮之前。**

## 门槛结果（`results/critic_gate.json`）
- 数据：前 100 个专家有效种子（170000–170119，扫描 140 个，21 个专家检查不通过）；80 个场景训练，20 个留出场景中随机抽 70 个决策状态（`results/gate_pairs.json`），每个状态 8 个候选执行后都交给 π0 续跑到结束（560 次，全部起始状态逐位一致）。
- 留出状态上：随机挑 0.205，critic 挑 0.243，事后最优（oracle）0.329。
- 增益 +0.0375，bootstrap 95% CI [−0.0018, +0.0840]，下界未超过 0 → **不通过**。
- 诊断：

| 指标 | 值 | 含义 |
|---|---|---|
| V 的 AUC（训练 / 留出） | 0.81 / 0.81 | PRM 进度 + 已用时间能较好地预测整局成败 |
| Q 在训练目标上的 CV 两两准确率 | 0.42 | 想象特征连训练目标的排序都学不出来 |
| 只用 PRM 对**真实** +48 画面打分挑选的增益 | +0.005 | 即使看到真实执行后的画面，一个 chunk 后的进度也几乎不能区分哪个候选最终成功 |
| 想象 vs 真实 +48 的 PRM 排序相关（Spearman） | 0.01（Φ）/ 0.06（hop） | FACT 想象画面的 PRM 排序与真实执行后的排序几乎无关 |

## 第 0 轮
- 随机挑选：前 100 局成功 37 局（原版 FACT 在另一组种子 seeds100 上为 42%，不可直接比较）；随机挑中的候选与 FACT 自带 BoN 一致 13%（≈1/8）。
- 离线执行：119 局、13776 个“状态 × 候选”全部起始状态逐位一致，1603 次一致性检查全部一致（`results/EXEC_DONE`）。
- PRM 特征：119 局全部完成。检查视频由 `code/render_prm_video.py` 生成（真实画面 + PRM 进度曲线）。

## 实现中发现并处理的问题（如实记录）
1. 11-21 22:55–23:10：离线执行任务与 rollout 挤在同一张卡上，显存不足（Vulkan DeviceLost），6 局归档后重跑，另 2 局卡死后被 `timeout` 结束并重跑。此后离线执行只在 rollout 结束的卡上运行。
2. 离线执行改为只在 +12/24/36/48 关键帧渲染，并采用 FACT 自带的“执行动作时跳过渲染同步”；两者在 16 个候选上与原方式逐位一致（关键帧、结束状态、成败），跨机器（AMAX_HOST/CLUSTER_HOST）也一致。
3. PRM：批量 vLLM 贪心解码**不能逐位复现**（同一批重复发送约 5% 的查询变化，平均 |Δ|≈0.001，开关前缀缓存都一样，`results/prm_determinism_*.json`）。保留官方设置；特征只算一次并保存，下游只用这一份。PRM 用的是开源权重 Robo-Dopamine-GRM-2.0-8B-Preview（HF 版本 980f3d1，sha256 已核对），未训练。
4. FACT 推理服务一次只服务一个 TCP 连接：续跑改为每进程一个客户端、每个续跑进程独占一个服务；每卡 2 组服务+进程导致显存不足，5 个门槛状态的 40 条结果因渲染报错整体作废并重跑。
5. 合成数据检验门槛判据：60 次无信号重复，误报 2 次（3.3%），平均增益 0.007±0.006（`results/synthetic_null_gate_60reps.txt`，`code/tests/`）。

## 代码
| 文件 | 作用 |
|---|---|
| `build_server_tts.py`、`start_server_tts_amax.sh`、`run_worker_amax.sh`、`launch_round_amax.sh`、`second_pass_amax.sh` | 在线 TTS rollout |
| `tts_driver.py` | 一局 TTS：随机或按 critic 挑选，记录所有候选 |
| `exec_candidates.py`、`exec_scheduler.py` | 离线逐位重放并执行全部候选 |
| `round_select.py`、`gate_select.py` | 前 100 个有效种子、留出场景、门槛状态抽取 |
| `continue_candidates.py`、`gate_run.py`、`start_server_probe_cluster.sh`、`gate_sync_cluster.sh` | 门槛续跑（Amax + 集群） |
| `prm_service.py`、`prm_common.py`、`prm_features.py`、`prm_chain_amax.sh`、`prm_smoke.py`、`prm_determinism.py`、`prm_diag_amax.sh` | PRM 服务、特征、检查 |
| `critic_train.py` | V、Q 与门槛评估 |
| `render_prm_video.py` | 真实画面 + PRM 输出检查视频 |
