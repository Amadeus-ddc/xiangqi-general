# Xiangqi General 当前状态

本页是运行与接手入口。模型、数据和质量结论以[当前模型卡](docs/MODEL_CARD.md)为准；[旧状态原文](STATUS_HISTORY.md)保留历史用途。

## 已验证

以下是2026-11-08 06:09:37（UTC+8）的固定训练快照，并非实时步数。轻量[执行证据](evidence/auxiliary-two-gpu-actual-resume-v1.json)记录原始收据的相对位置与SHA256；大模型、原始回答和日志不包含在源码发布中。

| 项目 | 状态 |
| --- | --- |
| 新四课 | 已按实际接续边界完成，第四课选42000步；[原始完成记录](docs/PAPER_FOUR_COURSES_COMPLETED.md) |
| 初始讲解及联合回放SFT | 均已完成；质量局限见模型卡，不等同于可靠教练 |
| 七类辅助语料 | 363445条训练、3749条验证，完整来源和消费检查完成 |
| 三卡到两卡迁移 | 原第1000步完整参数、Adam、计数、原数据前缀及验证顺序保留；正式两卡更新从1001开始 |
| 两卡真实预检 | 624个FP32张量／4694274592个参数；连续4步与2+2恢复的参数最大绝对差为0，Adam／随机／游标合同通过 |
| 本项目资源 | 物理GPU0、3；GPU1、2已释放本项目占用并预留，未知来源任务未处理 |
| 正式辅助训练 | 小时快照记录1320步，保留1000步后新增320步；下一次完整七类验证在2000步 |
| 六类搜索回放 | 12288条训练／3365条验证全量独立重选和原生消费完成，尚未用于新的合格搜索SGD |

实际执行源码是冻结的131模块，revision为`auxiliary-resource-transition-131-v1`。这次文档整理不切换运行中的源码、数据、配方或模型。代码资格、资源预检、实际参数更新和能力评测分别验收。

## 正在完成

| 任务 | 当前依赖与后续动作 |
| --- | --- |
| 两卡七类辅助训练 | 保持全局批量30、微批量3、完整FP32参数、FSDP2、2048词元上限；至少12115总全局步满足一遍预算，最多48460步／四遍，以完整七类验证选模 |
| 辅助模型开发评测 | 等该轮真实完成并选定模型，再做2600道旧课、384条讲解的正常／清零／错配对照、完整BF16辅助评审和384局原始对弈 |
| 固定走法／数值诊断 | 同一完成模型的381道数值题和384道单走法题；与原历史推理及评分合同对齐 |
| 后代搜索试批 | 等辅助训练、两类评测和真实回放消费全部完成；128根、完整学生子讲解、完整BF16教师；达到原验收条件后才考虑扩大和训练 |
| 最终交付 | 合格搜索训练、独立测试、中文教学核查和正式推理目录仍未完成 |

学生与完整教师按任务串行使用允许卡，全部共享允许卡租约。辅助训练和搜索蒸馏轮数分别记录；当前均没有新增已完成合格轮次。

## 结果解释

当前两卡模型尚无完整开发质量结果。旧联合SFT第2304步的固定评测、非法动作诊断及输入路径回放属于旧模型；它们没有测量本轮辅助训练收益。原始错误、自然败局、和棋、截断与辅助教师评分分开报告。

初始模型的失败、实际164次更新后仍选中起点的未获选蒸馏尝试，以及历史等待器均保留原始证据。软件测试、可搬动目录、输入准备或原生规则通过不能作为强棋力、优秀讲解或已交付的证明。

## 当前运行与续跑

长任务由tmux托管，启动命令、准确进程身份、配置和依赖写在各运行目录的`launch.json`、`owner.json`、`plan.json`或README中。源码仓库中的命令说明不代替本服务器已有运行记录。

| 用途 | tmux会话 | 本地运行目录 |
| --- | --- | --- |
| 两卡正式辅助训练 | `xqgeneral-two-GPU-native1000-resume-micro3-v1` | `runs/auxiliary-two-gpu-resource-resume-micro3-v1` |
| 只读小时观察 | `xqgeneral-auxiliary-two-gpu-hourly-monitor-v1` | `runs/auxiliary-two-gpu-hourly-monitor-v1` |
| 后续开发评测等待器 | `xqgeneral-auxiliary-two-gpu-development-evaluation-v1` | `runs/auxiliary-policy-two-gpu-development-evaluation-v1` |
| 固定765题等待器 | `xqgeneral-two-gpu-fixed765-development-diagnostics-v1` | `runs/oracle-policy-two-gpu-development-diagnostics-v1` |
| 后代搜索等待器 | `xqgeneral-auxiliary-two-gpu-search-pilot-v1` | `runs/auxiliary-two-gpu-search-pilot-v1` |

日常读回使用小时观察器。`latest.json`给出实际读回时间，`state.json`给出下一次观察时间；不可变`readback-*.json`保留历史读数：

```bash
cat runs/auxiliary-two-gpu-hourly-monitor-v1/latest.json
cat runs/auxiliary-two-gpu-hourly-monitor-v1/state.json
cat runs/auxiliary-two-gpu-resource-resume-micro3-v1/round/state.json
```

两卡外层`state.json`有取得租约后未刷新的旧等待标签；实际阶段应结合`round/state.json`、原生执行及内核租约证明判断。观察超时或旧标签不等于训练停止。只有核对准确进程身份和终态后，才按该运行保存的固定源码与恢复计划续跑。

GPU1、2预留策略见本地`runs/GPU_ALLOCATION_POLICY.json`。原三／四卡队列仍受旧租约守护，不能恢复为现役队列；各历史实验、检查点、失败和工作树继续保留。当前根检出及两份用户自有未跟踪证据保持原样。

没有已验收的最终模型或模型发布。权重许可与分发边界见[第三方说明](THIRD_PARTY_NOTICES.md)，最终模型选择与独立评测见[固定协议](docs/FINAL_INDEPENDENT_EVALUATION.md)。
