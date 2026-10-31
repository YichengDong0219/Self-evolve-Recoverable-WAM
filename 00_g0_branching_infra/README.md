# 00 · G0：可复现分叉基础设施（2026-10-26 ~ 10-27）

目标：证明 FACT + RoboTwin 的闭环可以被逐位复现，后面所有"从失败状态出发"的工作都依赖这一点。

- `inference_server_probe.py` / `build_server_probe.py` / `start_server.sh`：在 FACT 推理服务上加一层探针，每次决策使用显式的续写随机种子，并记录请求与响应。
- `client_adapter.py` / `replay_adapter.py`：RoboTwin 侧客户端；`seed_for(root_seed, start_id, 'continuation', decision)` 决定每次决策的噪声。
- `g0_driver.py`：三种模式。rollout（闭环）、replay（不用模型，重放已执行动作）、branch（重放前 N 步后交还模型；用原 root_seed 必须逐位复现原后缀，换 root_seed 即一次可复现的新采样）。
- `g0_compare.py` / `g0_model_check.py` / `cross_machine_replay_check.py`：状态与行为一致性检查，含跨机器检查。

关键结论（G0 通过）：RoboTwin 的指令措辞来自未播种的 Python random，必须 `random.seed(env_seed)`；之后闭环、动作重放、从前缀分叉在 3090 与 4090 机器间均逐位一致。
prefix KV cache 只是单次推理 20 步去噪内的加速，开关会带来 bf16 级的数值差异，所以采集和评测必须使用同一配置。

注：本阶段部分基础模块（collect_sources、branch_compare、seed_plan、test_core、probe_request、client/replay_adapter 等）由 Codex 在 Claude×Codex 联合阶段编写，`*_codex.py` 同理。
