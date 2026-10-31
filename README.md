# Self-evolve-Recoverable-WAM

在 RoboTwin `hanging_mug` 上，把 cuRobo 专家接管得到的"失败 → 恢复"轨迹蒸馏进世界动作模型 FACT，并朝自演进的方向迭代。
目录按迭代顺序组织，每个阶段对应一次提交；各阶段的说明见该目录下的 README.md。

| 阶段 | 日期（2026） | 内容 |
|---|---|---|
| `00_g0_branching_infra` | 10-26 ~ 10-27 | FACT 推理探针、显式种子、逐位重放与分叉驱动（G0 通过） |
| `01_failure_census_and_imagination_probes` | 10-27 | 多任务失败普查、跌落起点规则、想象与现实对比 |
| `02_takeover_dispatch_turn_switch` | 10-27 | 原生脚本接管、按规则分派接管计划、turn_switch 恢复采集 |
| `03_turn_switch_expert_sft` | 10-28 | 第一次接入 FACT SFT（之后暂停 turn_switch） |
| `04_hanging_mug_takeover_MR1_to_MR5` | 10-28 | hanging_mug 接管规则迭代；用户定下"禁止回零、质量优先" |
| `05_MR6_chain_planner_simulate_and_select` | 10-28 ~ 10-29 | MR6 整链规划、simulate-and-select、G1–G7、第一轮 100 对数据 |
| `06_sft_round1_expert_vs_recovery` | 10-29 ~ 10-30 | SFT 管线（逐帧 mask、关闭 value）、自动评测；E 组 69，旧 R 组 0 |
| `07_MR7_joint_box_and_mix_arm` | 10-30 | MR7 关节盒与 G8、冻结失败状态资产、混合组（52/100） |
| `08_eval_videos_and_failure_analysis` | 10-31 | 逐位重放渲染评测视频、失败阶段分类、恢复率指标 |
| `09_round2_self_evolution_wip` | 10-31 ~ | 第二轮：线 1 扩充 cuRobo 数据，线 2 策略自采样 + cuRobo 兜底 |

## 第一轮结论（seeds100 上的成功数）
| 组 | 训练数据 | step 1000 | step 2000 | step 3000 | 恢复率 |
|---|---|---|---|---|---|
| base（发布的 fact-wam） | — | 42 | 42 | 42 | 0/46 |
| E | 100 条专家演示 | 30 | 50 | 69 | 0/27 |
| 旧 R | 100 条恢复（56 条关节越界） | 0 | — | 0 | — |
| Mix | 50 条恢复 + 50 条专家 | 20 | 45 | 52 | 1/37 |

## 阅读说明
- 每个阶段目录只放该阶段**新增或修改**的文件。中间版本有备份的文件给出当时的版本：例如 05 是 MR6 原版，07 是 MR7 版。没有中间备份的文件放在首次出现的阶段，内容为最终版本，各阶段 README 中有注明。
- 代码依赖 FACT、RoboTwin（RoboTwin-FACT fork）、cuRobo 0.7.8；所有改动都在这些仓库之外完成（独立脚本、monkeypatch、PYTHONPATH 注入）。
- 公开前已做脱敏：`/workspace` 指本机 / Amax 上的工作根目录，`/workspace_cluster` 指 4090 集群上的工作根目录，`CLUSTER_HOST` / `AMAX_HOST` / `user` 为占位符。代码与当时的多机环境（3090 本机、Amax 4090、4090 集群）强相关，不能直接运行，需要按自己的环境替换路径和主机。
- 各阶段的 `results/` 只放小型结果文件（数据列表、审计报告、评测状态、视频索引），不含数据、checkpoint 和视频。
- 文件头部的 "Claude, 日期" 注释表示由 Claude Code 编写，`*_codex.py` 以及 00 阶段的部分模块由 Codex 编写。
