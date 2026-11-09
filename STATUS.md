# Xiangqi General 当前状态

本页是运行与接手入口。模型、数据和质量结论以[当前模型卡](docs/MODEL_CARD.md)为准；[旧状态原文](STATUS_HISTORY.md)保留历史用途。

## 已验证

以下是2026-11-09 23:19:17（UTC+8）的固定快照。公开仓库不提供实时进度；[完成与诊断摘要](evidence/auxiliary-round-completed-fixed765-v1.json)绑定实际私有收据，原始权重、回答、数据和日志不随源码发布。

| 项目 | 状态 |
| --- | --- |
| 新四课 | 已按实际接续边界完成，第四课选42000步；[原始完成记录](docs/PAPER_FOUR_COURSES_COMPLETED.md) |
| 初始讲解及联合回放SFT | 均已完成；旧模型局限见模型卡 |
| 首轮七类辅助训练 | 正常退出，累计16000步，选中12000步；完整辅助轮次现为1 |
| 全量数据与预算 | 363445条训练、217271个训练历史；3749条验证、1249个验证历史；实际首遍在12115步完成 |
| 检查点与选模 | 完整续训保存到16000步；所选12000步消耗360000个有效行出现次数，先于首遍边界，不能称其已见全部源行 |
| 实际重载所选模型 | 同一3749题正常／清零／错配NLL分别0.384295／0.820599／1.168369；不是自由生成准确率或棋力 |
| 固定765题诊断 | 已完成；765份原始回答与原输入顺序全部重算，详见[本轮实测](docs/AUXILIARY_ROUND_COMPLETED.md) |
| 本项目资源 | 仅允许物理GPU0、3；GPU1、2已释放本项目占用并预留，未知来源任务未处理 |
| 六类搜索回放 | 12288条训练／3365条验证完成独立重选和原生消费，尚未形成新的合格搜索SGD轮次 |

本轮实际执行冻结131模块，revision为`auxiliary-resource-transition-131-v1`。原三卡1000次更新和后续两卡15000次更新共同计入本轮；参数、Adam、随机状态与数据位置按[完整状态迁移](docs/AUXILIARY_RESOURCE_TRANSITION.md)保留。实际辅助训练与新搜索蒸馏完成轮次分别为1和0。

## 正在完成

| 任务 | 快照时的阶段与后续动作 |
| --- | --- |
| 辅助模型完整开发评测 | 所选12000步权重的CPU消费核验通过，已取得允许卡租约；执行2600道旧课、384条正常／清零／错配讲解、完整BF16辅助评审和384局原始对弈 |
| 后代搜索试批 | 等原开发评测真实完成；固定765题已满足完成依赖。128根试批使用完整学生子讲解和完整BF16教师，按真实产量和质量决定扩大与训练 |
| 最终交付 | 足量合格搜索训练、最终独立测试、中文教学核查及正式推理目录仍未完成 |

学生与完整教师按原队列串行使用允许卡，共享现有租约。验证题不进入后续训练；独立最终测试保持隔离。

## 结果解释

数值题首着合法由340/381增至369/381，六个半回合或提前终局变化有效由119/381增至228/381；单走法题当前369/384合法、171/384匹配引擎参考。仍有12道数值题首着非法、153道主要变化非法及15道单走法非法。错误类别可重叠，不能相加当作独立错题数。

765个回答共用384个验证历史，不是765个独立棋局。旧单走法题全部受格式拒绝，不能将其零接受率解释为零棋力。新模型的整局胜负、讲解事实和教学效果仍待完整评测；旧J2304的384局0胜383负1和属于旧模型。

软件测试、输入准备、NLL下降及可搬动目录不构成强棋力或优秀讲解的验收。详细分母、源码可比性和原始错误见[本轮训练与诊断](docs/AUXILIARY_ROUND_COMPLETED.md)。

## 当前运行与续跑

长任务由tmux托管，准确进程身份、源码、配置和依赖保存在服务器本地运行目录。这些`runs/`文件不属于源码发行包；下表用于接手现有服务器任务。

| 用途 | tmux会话 | 本地运行目录 |
| --- | --- | --- |
| 完整开发评测 | `xqgeneral-auxiliary-two-gpu-development-evaluation-v1` | `runs/auxiliary-policy-two-gpu-development-evaluation-v1` |
| 后代搜索等待器 | `xqgeneral-auxiliary-two-gpu-search-pilot-v1` | `runs/auxiliary-two-gpu-search-pilot-v1` |
| 已完成：辅助训练 | 原`xqgeneral-two-GPU-native1000-resume-micro3-v1` | `runs/auxiliary-two-gpu-resource-resume-micro3-v1` |
| 已完成：训练小时观察 | 原`xqgeneral-auxiliary-two-gpu-hourly-monitor-v1` | `runs/auxiliary-two-gpu-hourly-monitor-v1` |
| 已完成：固定765题 | 原`xqgeneral-two-gpu-fixed765-development-diagnostics-v1` | `runs/oracle-policy-two-gpu-development-diagnostics-v1` |

已完成的训练、观察器和765题进程均有正常退出记录，不重启。后续每小时读回由本地按需入口保存，不存在新的常驻观察daemon：

```bash
cat runs/auxiliary-posttraining-hourly-readbacks-v1/latest.json
cat runs/auxiliary-posttraining-hourly-readbacks-v1/state.json
cat runs/auxiliary-policy-two-gpu-development-evaluation-v1/state.json
cat runs/auxiliary-two-gpu-resource-resume-micro3-v1/exit.json
```

运行目录的旧等待标签不能推翻真实完成凭据。单个子进程退出可能是正常阶段切换，须核对准确父进程、其终态及原始产物；观察超时不作为重新启动依据。

GPU预留策略见本地`runs/GPU_ALLOCATION_POLICY.json`。原三／四卡队列继续受旧租约守护，不恢复为现役队列。各历史实验、失败、检查点和工作树保留。

正式模型发布与最终独立验收仍待完成。分发边界见[第三方说明](THIRD_PARTY_NOTICES.md)，最终测试见[固定协议](docs/FINAL_INDEPENDENT_EVALUATION.md)。
