# 相关工作与 baseline（2026-11-21 整理）

围绕 ATC 方案：
- WAM 用自己对候选动作后果的想象打分；
- 用真实执行出来的备选动作，把想象训准；
- 测试时挑选 + 蒸馏 + 闭环。

arXiv 编号都核对过。“只读了摘要”或“只看了标题”的条目，在备注里标明。

## 一、WAM 测试时挑选（离“自评”最近，优先读）
| 工作 | arXiv | 做了什么 | 和我们的关系 |
|---|---|---|---|
| WISE（Seeing Through the Illusion） | [2611.03585](https://arxiv.org/abs/2611.03585) | WAM 冻结（Cosmos 3 Edge、DreamZero），每步采 4 个“动作 + 想象”。三项打分：动作是否合法平滑、逆动力学一致性、想象最后一帧的 Robometer 进度。RoboLab 6→24%（仅 50 局），LIBERO 75.1→78.3；运行时间约 3.4 倍 | 最接近。但想象冻结、不闭环，没有比较想象与真实结局，也没测想象是否随动作变化；“用分数做后训练”列为未来工作 |
| Gated GeoBoN | [2607.17454](https://arxiv.org/abs/2607.17454) | 免训练。用 VGGT 检查想象画面的深度重投影一致性来挑选；用光流和末端运动是否矛盾作为门控。RoboTwin 2.0 上 Motus 87.8→89.9（N=8）；误选比例在 N=2/8/16 时为 7/12/32% | 同一平台，可直接作为 TTS 对照；“N 越大误选越多”是冻结打分的典型问题 |
| τ0-WM | [2606.01027](https://arxiv.org/abs/2606.01027) | 用“重新去噪的一致性”给候选打分；分数低时用动作条件视频模拟器纠正 | 一致性类打分对照 |
| Is the Future Compatible? | [2605.07514](https://arxiv.org/abs/2605.07514) | 免训练。按预测转移与观察转移是否吻合来挑候选；LingBot-VA，RoboTwin | 一致性类打分对照 |
| FACT | [2608.10232](https://arxiv.org/abs/2608.10232) | 我们的基座。先去噪动作，再以干净动作为条件预测未来视频和 value（剩余步数比例，失败轨迹 +1）；可选用 value 做 best-of-N | 冻结自评的直接对照（G1 测的就是它） |
| FOREWARN | [2502.01828](https://arxiv.org/abs/2502.01828) | 2025 年。用潜空间世界模型想象各候选方案的结果，由 VLM 判断后挑选；作者说主要失败来自想象不准 | 早期的“想象 + 判断”路线 |
| Fast-WAM | [2603.16666](https://arxiv.org/abs/2603.16666) | 论证 WAM 在测试时未必需要想象 | 反方背景，写动机时要回应 |

## 二、用真实执行修想象，或让模型对动作敏感
| 工作 | arXiv | 做了什么 | 和我们的关系 |
|---|---|---|---|
| CureWM（World Models Dream of Success） | [2610.09134](https://arxiv.org/abs/2610.09134) | 世界模型对失败过于乐观，用执行过的反事实回放来修正；只更新世界模型；作者自己说预测变好不等于控制变好 | 修想象这一侧最接近；它没把修好的想象用来挑动作 |
| AD-WM | [2609.30264](https://www.alphaxiv.org/abs/2609.30264) | 指出训练只监督“实际执行动作”的结果，而规划要比较“同一状态下不同动作”，所以模型分不清候选；加入动作可恢复正则。OGBench Cube 难起点 3.7→52% | 支持 H1 的机制。它是潜空间世界模型 + 规划，不是 WAM，也没用真实备选结局 |
| R²-WAM | [2610.04913](https://arxiv.org/abs/2610.04913) | 在 Fast-WAM 上离线后训练。repair：压低不跟随示教动作的想象；reject：压低想象进度低的采样动作。RoboTwin 93.8 vs 91.9 | 训练侧最接近，但离线只做一次，没有闭环和备选执行 |
| BadWAM | [2607.15207](https://arxiv.org/abs/2607.15207) | 攻击实验：动作已经偏了，想象却不变 | 证明“想象对、动作错”的脱钩真实存在 |
| WMPO | [2511.09515](https://arxiv.org/abs/2511.09515) | 在世界模型里做 VLA 的策略优化 | 想象中 RL 的对照 |
| Ctrl-World | [2510.10125](https://arxiv.org/abs/2510.10125) | 可控生成世界模型，在想象里评估和改进策略（+44.7%）；作者说精细交互会失败 | 想象 + 蒸馏的对照 |

## 三、价值或打分器引导的动作挑选（VLA，没有世界模型）
| 工作 | arXiv | 做了什么 | 和我们的关系 |
|---|---|---|---|
| V-GPS | [2410.13816](https://arxiv.org/abs/2410.13816) | 用离线 RL 学到的 Q 给通用策略的候选重排，不改策略；12 个任务、5 个策略 | “(观测, 动作) 直接学 Q”这一路的代表，是 H2 的非想象对照 |
| RoboMonkey | [2506.17811](https://arxiv.org/abs/2506.17811) | 采样加扰动，再用 VLM 验证器挑选；分布外实机 +25%，SIMPLER +9% | 外部验证器对照 |
| Hume | [2505.21432](https://arxiv.org/abs/2505.21432) | VLA 加 value-query 头，多次采样后挑选（System 2） | 架构内 value 头 + TTS 的 VLA 版本 |
| PA-RL | [2412.06685](https://arxiv.org/abs/2412.06685) | 从策略采样，先用 Q 重排，再沿梯度优化动作，最后监督蒸馏回策略；OpenVLA 实机 40→70%；有代码 | 最接近 senior 的“挑选 + 蒸馏”；作者也指出采样成本高 |

## 四、从自己的经验学价值，自我改进循环
| 工作 | arXiv | 做了什么 | 和我们的关系 |
|---|---|---|---|
| π*0.6 / RECAP | [2511.14759](https://arxiv.org/abs/2511.14759) | 多任务 value，用自身经验和人工纠正训练；策略以优势为条件 | 多任务 value 的工业级参照 |
| Self-Improving EFMs | [2509.15155](https://arxiv.org/abs/2509.15155) | 预测剩余步数，用作奖励和成功检测，再在线自我改进；比加示教更省数据 | 与 FACT 的 value 定义（剩余步数）相同，是自改进对照 |
| PLD | [2511.00091](https://arxiv.org/abs/2511.00091) | 残差 RL 接管，混合 rollout，再 SFT 回 VLA；LIBERO 99% | 插件再蒸馏这一路线 |
| RLDG | [2412.09858](https://arxiv.org/abs/2412.09858) | 把 RL 专家数据蒸馏进通用策略，效果优于人类示教 | 同上 |
| EVO-WAM | [2609.38057](https://arxiv.org/abs/2609.38057) | RoboTwin 2.0 上的自演进 WAM：VLM 挑出生成 rollout 中完成的片段，迭代 SFT；不做真实执行 | 同平台的自演进对照；我们的区别是真实执行和自评 |
| Recursive Self-Improvement via Local Recovery Supervision | [2610.05151](https://arxiv.org/abs/2610.05151) | 多轮 RSI，教师 + 审计；LIBERO-Goal 88 vs 78（只读了摘要） | 已经用了“RSI”这个名字 |
| RoboCat | [2306.11706](https://arxiv.org/abs/2306.11706) | 用自己生成的数据做自我改进循环的早期代表 | 背景 |

## 五、打分器 / 过程奖励模型（PRM）
| 工作 | 链接 | 做了什么 | 和我们的关系 |
|---|---|---|---|
| Robo-Dopamine GRM | [2512.23703](https://arxiv.org/abs/2512.23703)，[GRM-2.0-8B-Preview](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview) | 输入多视角的前后画面、任务描述，可选目标图，输出进度变化（约 25 档）；论文报告 VOC 0.95；用一条示教适配新任务；需要 CUDA ≥ 12.8 | senior 提的标签来源；要先测它能否分辨 1–2 cm |
| Robometer | [2603.02115](https://arxiv.org/abs/2603.02115) | 在 RBM-1M 上训练，目标同时包括进度和同任务轨迹之间的偏好，并给出帧级成功判断；权重公开（Robometer-4B） | WISE 用它给想象画面打分，是我们“PRM 看想象”的对照 |
| VLAC | [2509.15937](https://arxiv.org/abs/2509.15937) | 在 InternVL 上把 actor 和 critic 合成一个模型，输出进度奖励；实机 30→90% | PRM 对照 |
| STEAM | [2606.29834](https://arxiv.org/abs/2606.29834) | 从时间偏移自监督地学习进度 / 优势（R²-WAM 用它） | 进度估计对照 |
| CriticHack | [2610.02527](https://arxiv.org/abs/2610.02527) | 评测视觉奖励在策略优化压力下会不会被钻空子（只看到标题，待读） | 用 PRM 当 Q 标签的风险参照 |
| GVL（Vision Language Models are In-Context Value Learners） | 编号待核 | 用 VLM 在上下文中排序帧来估计 value | Robo-Dopamine 的对照之一 |

## 六、我们做过或会用到的平台与对照
- FACT（基座），Motus（有 RoboTwin 权重，我们复现过），Fast-WAM，LingBot-VA。
- FAR：[2607.01111](https://arxiv.org/abs/2607.01111)，我们复现过；其中“盲目重试会重复同样的错误”的观点与我们的鼠标数据一致。

## 建议的阅读顺序
1. WISE、Gated GeoBoN：先看现在“冻结自评”做到哪一步。
2. AD-WM、CureWM、R²-WAM：再看“修想象”这一侧，以及它们为什么还没闭环。
3. PA-RL、π*0.6：理解“挑选 + 蒸馏”和“多任务 value”的成熟做法。
4. Robometer、Robo-Dopamine：选 PRM，以及看它们的已知局限。
5. 其余条目用作对照和写作背景。
