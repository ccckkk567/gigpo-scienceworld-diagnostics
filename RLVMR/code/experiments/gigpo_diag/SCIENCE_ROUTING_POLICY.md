# ScienceWorld reset 检查边界：science-routing-v4

日期：2026-10-14。基于 `81c6bfd98aa2baf59ec1365c76d504e100d98c10`，沿用 `exp/gigpo-train-diagnostics-scienceworld-20261013`。本文件随实现提交，实际提交号以新 job 的 `RUN_MANIFEST.json/code_commit` 为准。

## 本次边界

| 检查对象 | 政策 | 证据与限制 |
|---|---|---|
| 父请求与 load/reset 完成回执 | 不一致即中止 | task 编号/原生名称、variation、simplifications、step limit、seed、worker slot、reset 序号、分配 token。核对实际调用参数及已有 Python wrapper 属性；不声称读取了完整 JVM 状态 |
| collector 实际 UID 与分配 token | 不一致即中止 | 双向一对一、组大小、槽位、唯一 trajectory UID；对父请求再次核对。训练 K=8，评测 K=1；重复采到同一 task/variation 的组仍有不同 token |
| 必需模型输入、原生 load/reset/step、必需动作枚举 | 缺失/失败即中止 | 不重试、不换任务；step score 必须为有限数，因为参考 reward 使用它 |
| 已确认的目标/关键初态语义违例 | 明确阻断匹配的分配 | 通过下面的人工审计发现登记；不同文本或 UUID 不自动构成确认 |
| observation、对象别名、动作字符串、有效 task description 的内容差异 | 记录并继续 | 相同行集合的顺序差异标为 `known_rendering_difference`；其他内容变化标为 `suspicious_semantic_difference`/需审计 |
| 初始 score 不同、未证明稳定的 UUID/动作结构不同 | 记录并继续 | 保存 score min/max、已知/未知数量；结构差异只是审计线索 |
| diagnostic hash/schema/结构字段/辅助字段丢失或解析失败 | 记录 unknown 并继续 | `evidence_unavailable`；不能借此绕过前两项独立契约 |
| 可选 getScore/getCompleted 或转换失败 | strict_success=null | 不重做动作，不改原始 reward/obs/done；网络、连接、Java 执行故障仍抛出 |
| 日志写入失败 | 可见告警并继续 | 契约失败先输出 `SCIENCE_CONTRACT_FAILURE`，不依赖诊断是否启用 |

这些检查确认路由契约，不证明全部 worker 的完整物理世界相同。默认 `initial_state_status=unverified`、`full_hidden_state_verified=false`。没有给未知世界等价设置“必须证明才训练”的门槛。

## 生产路径与实验可比性

恢复 RLVMR 原生字符串调用顺序：`load → reset → get_task_description → get_possible_actions → get_possible_objects → get_valid_action_object_combinations`。诊断只捕获已有返回值，不额外枚举、不 replay、不为诊断 reseed，不改写 prompt/anchor。旧结构解码器仅用于离线历史记录审计；本次不再从训练 JVM 采集结构信息。

父进程的 `choice(..., replace=True)` 随后 `randint(...)` 保持原顺序、数量和 sampler sidecar；token 来自 run/split/reset序号/group槽位，不消耗 RNG。奖励、无效动作惩罚、GiGPO、原生 anchor 分组、16×8、参考评测、模型、学习率、epoch、Triton 缓存和 prompt 长度限制不变。评测的 collector UID 按实际 group_n=1 生成，此前误按训练 K=8 分配，且未参加训练 guard。

恢复字符串接口相对 81c6bfd 的结构接口可能产生不同随机渲染；不承诺旧/新任务逐 token 轨迹相同。停止筛除别名差异会让过去被 guard 中断的任务继续参与训练，不把这一变化描述成“覆盖任务集合完全没变”。

每个 job 不再强制跑额外 reset 预检。确有需要时设 `GIGPO_SCIENCE_ROUTING_CHECK=1`：仅四个独立模拟器、两次 reset 的通信/任务路由 smoke；正常训练直接由第一批真实 reset 检查契约。没有新增全任务状态证明、自动丢组或重置到相同的机制。

## 确认语义违例的升级入口

`GIGPO_SCIENCE_CONFIRMED_VIOLATIONS` 可指定一个人工复核后的 JSON 列表。每项必须包含 `task_name`、`variation_id`、`reason`、`evidence_ref`。文件默认未设置，仓库未预置任何“确认违例”。例如人工已经确认目标变更或关键对象缺失，应记录原始证据路径及判定依据，再登记匹配任务；父进程一旦分配到该任务就报 `confirmed_semantics` 并中止，不跳过、不重采。

这不是自动语义证明器，未有可靠事实支持的名称/UUID 差异不能登记成“确认”。登记文件在 job 启动时加载；正在运行的作业若发现确认违例，需要受控停止，不依赖热修改文件。

## 验证范围与真实例子

- 本地 44 项回归覆盖错载同一旧任务、task/variation/seed/slot/token、跨组 UID、重复 trajectory UID、评测 group1、必需输入、原生失败、dead JVM、可选 getter/日志/解析失败、采样 RNG 与恢复、原生奖励/输入/anchor 保留。
- 2 个真实 ScienceWorld 1.2.3 JVM：task24/variation141，2 轮 reset 共4个回执、4步 `look around`；父请求与 collector 绑定通过，原生 strict getter 成功。无 GPU 模型、无学习更新。这组新小样本没有观察到跨 worker 文本差异。
- 对 81c6bfd 时保存的8份真实本地 JVM 捕获离线重放。它们包含 `pride and prejudice on shelf` 等书名变体；新 observer 输出 `classification=["suspicious_semantic_difference"]`，差异字段含 objects/possible_actions，`alias_equivalence=null`，`full_hidden_state_verified=false`，继续而不退出。旧对象 UUID/类型相同也不升级成完整世界等价。
- `SCIENCE_ROUTING_VALIDATION.json` 保存新真实模拟器的 JAR 哈希、原始捕获哈希、实际回报和上述历史重放的真实告警样例。历史重放来源是本地真 JVM 捕获，不能冒充 HPC160878 中未采集的对象身份。
- 更新2早存与周期10存的触发语法已核对；完整 GPU checkpoint 的实际产物仍需新 HPC run 确认。

## 已确认的实际运行状态与恢复方式

用户于2026-10-14从HPC返回的只读输出：

| 项目 | 实际证据 |
|---|---|
| 旧 run | `scienceworld-qwen25-15b-gigpo-160883-20261013T141249Z` |
| job | 160883，FAILED，ExitCode=1:0，Elapsed=00:18:56 |
| 运行 commit / HPC checkout | `94eeed014d22641ab6836ce42691aafd21cb3623` |
| 更新与退出 | 2/150；下一批 reset 在旧 verify_groups 抛出 mixed/missing evidence |
| checkpoint | 没有 latest 指针，checkpoint 目录查询无列项；没有可用恢复产物 |
| SwanLab | https://swanlab.cn/@KANSEI/gigpo-training-diagnostics/runs/9rhypv94 ，终止时显示上传2052条记录 |
| 活动作业 | 回报时没有 ScienceWorld；ALFWorld160764和gmas160127仍在运行，各自目录不动 |
| 未跟踪内容 | 旧 ScienceWorld checkout 的 `RLVMR/code/outputs/` 保留，不清理 |

这次不能从“两次更新日志”恢复权重。新 job 使用相同 Qwen2.5-1.5B-Instruct 固定版本、相同配置和新 run ID，`GIGPO_PARENT_RUN` 记录旧 run，`resume_mode=disable`。第2次更新新增完整原生 checkpoint，之后仍每10次及末次保存。

将来有完整 actor/optimizer/scheduler 和 sampler sidecar 时，用 `GIGPO_RESUME_MODE=resume_path` 与 `GIGPO_RESUME_FROM=.../global_step_N`；只有 actor 权重属于 warm restart。即使 sidecar 恢复了采样 RNG，也不承诺重建 JVM 后逐 token/全环境随机状态复演。

提交新 job 前使用固定 `GIGPO_EXPECTED_COMMIT`，manifest记录实际代码、parent run、resume path、guard policy、capture mode和早存更新号。若提交前又有 ScienceWorld 使用该 checkout，改用独立目录，不能直接 pull 活动目录。无需重新 prepare 环境。

## 指标与下一步判断

- SwanLab/本地 metrics：`diag/reset_cumulative/{train,eval}/...` 包含检查回执数、UID检查组数、已知渲染差异、可疑语义差异、证据不可用记录数/率、observer失败数。计数为本进程累计；unknown与分母同时保留。硬违例会立即抛出，先见 stdout 的失败分类，不保证能等到下一次 SwanLab 上传。
- 原生 anchor：`diag/anchor/cross_trajectory_supported_steps`、`support_known_steps`、`support_unknown_steps`、`cross_trajectory_support_rate`，以及 `diag/adv/AS_nonzero_{supported,unsupported}_steps` 与相应 denominator。使用实际 step_group_id 与去重的 analysis_step_id，不改变 GiGPO 分组。不完整轨迹标识记 unknown。
- 原有 AE/AS、joint非零率、重构误差、参考成功率、strict_success（含unknown）及真实更新曲线继续输出。每组观察/完整 reset policy text 变体数与 score min/max 在 `science_reset_observations.*.jsonl.gz`；未采结构证据和未分类alias不伪装成零差异。
- 自动分析只用实际日志，新增 `tables/routing_anchor_metrics.csv`、`tables/science_reset_observations.jsonl`，以及有真实记录才生成的 reset classification / native anchor support 图。低成功率、未知等价或缺失可选标签都不阻止交付。

下一步先让真实16×8训练跨过过去的第三批 reset，取得第2步 checkpoint，核对新的 commit 与真实学习曲线。没有再停训只证明这段运行持续了，不证明长期无错或所有世界等价。本次本地验证没有任何训练成功率结论；新的GPU运行结果以返回的job/manifest/日志为准。
