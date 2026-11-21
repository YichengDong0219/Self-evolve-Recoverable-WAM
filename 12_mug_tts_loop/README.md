# 12 · 挂杯 TTS 最小闭环（分支 `mug-tts-loop`，2026-11-21 起，进行中）

协议见 [PROTOCOL.md](PROTOCOL.md)（开跑前冻结，后附实现细节补充）。

## 进度
- [x] 推理服务补丁 `build_server_tts.py`：一次请求返回 8 个候选的动作和想象关键帧，FACT 仓库本身不改。
- [x] 在线驱动 `tts_driver.py`：测试 1 局，共 19 个决策点；服务端每个决策约 6.3 s，整局约 3 分钟。
- [x] 离线执行器 `exec_candidates.py` 已验证逐位一致：
  - 16 个候选在决策点的状态与在线记录完全相同；
  - 被选中的候选执行后，状态与下一个决策点的在线记录完全相同；
  - 前缀重放时不逐步渲染，状态和渲染出的帧仍然逐位相同。
- [x] PRM 环境：新建独立 venv（vLLM 0.11），权重在本机下载后转到 Amax。
- [ ] 第 0 轮 rollout（随机挑选）：22:48 起在 Amax 5 张 4090 上运行，扫描种子 170000–170139。
- [ ] 离线执行、PRM 特征、V/Q 训练、门槛检查
- [ ] 第 1 轮、蒸馏三组（A/B/C）、评测

## 代码
| 文件 | 作用 |
|---|---|
| `build_server_tts.py` | 构建 TTS 推理服务 |
| `start_server_tts_amax.sh` / `run_worker_amax.sh` / `launch_round_amax.sh` | 启动在线 rollout |
| `tts_driver.py` | 一局 TTS：随机或按 critic 挑选，记录所有候选 |
| `exec_candidates.py` / `exec_scheduler.py` | 离线逐位重放并执行全部候选；调度器只使用全部进程都是我们自己的卡 |
| `continue_candidates.py` | 门槛检查：候选执行后交给 π0 续跑到结束 |
| `prm_service.py` / `prm_common.py` / `prm_features.py` | Robo-Dopamine 打分服务与特征提取 |
| `critic_train.py` | V、Q 与门槛评估 |
