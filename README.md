# Xiangqi General（象棋将军）

将象棋专家网络与语言模型连接起来，目标是提供棋力强、能用中文解释走法的个人象棋教练。本项目是受Queen研究流程启发的独立中国象棋实现。

## 当前进度

2026-11-06，[批量严格搜索真实试批](docs/BATCHED_STRICT_SEARCH_PILOT.md)完成：128个新起始局面得到99份教师查询，严格收取89条、拒收10条，并通过独立原生合并与再次重建。后续四卡4096根正在原配置下继续；控制器已接管原工作进程，未重启数据任务。该阶段没有学生参数更新，完整四批产量、讲解语义、下一轮训练与最终模型仍待验收。

截至2026-11-06，四门课程和两套初始讲解训练已完成；第四课实际48000步结束、选择42000步。首轮扩充开发蒸馏实际完成164次四卡全参数更新，但讲解验证NLL选中起点0步，未获选新模型。[完整诊断](docs/EXPANDED_DISTILLATION_DIAGNOSIS.md)保留训练、原始回答和质量变化，最终棋力、可靠教学及正式模型交付仍待验收。

| 相同2600道旧课开发题 | 正确数 | 正确率 |
| --- | ---: | ---: |
| 讲解前同设置对照 | 2465 | 94.81% |
| 原初始讲解1792步 | 1896 | 72.92% |
| 联合回放初始讲解2304步 | 2345 | 90.19% |
| 未获选的实际164步候选 | 1860 | 71.54% |

这些是固定问答的保持率。实际164步候选在原384条讲解开发题中，首着合法339条、主变化合法163条、完整结构／规则检查通过36条；原父模型对应346／99／15。变化结构改善，旧课保持及首着指标没有改善，正文语义、完整对弈和最终独立验收尚未在该候选上完成。

[严格重新收集](docs/EXPANDED_DISTILLATION_DIAGNOSIS.md)已从五批1185份完整BF16教师原答重建1036条接受记录，去重保留1025个完整历史。原1046行中发现21条即时将死评分／PV不一致；五个新来源都启用该窄门槛，原384／512保留文件逐字节一致，完整战略语义仍未全部验证。原始训练失败及未筛选输入保留。

[批量学生搜索](docs/BATCHED_STUDENT_SEARCH.md)使用独立请求随机流、原始批次账本和有界工作线程。原冻结批量分支已在真实完整学生权重上完成短解码和四请求完整预算容量检查；本集成分支接入严格收集及显式推理权重缓存，并已完成上述128根真实试批。这些测量不证明棋力或讲解改善。

后续先扩大不同训练历史的实际搜索／严格收取规模，并按实际数据配置旧课回放覆盖与学习率，再执行同题讲解、对弈和最终独立验收。当前每门旧课65条的采样已核对；现有可用回放池为6656条。首轮开发数据仍远小于论文每轮150K–285K的搜索蒸馏规模。

[子讲解原文入口](docs/CHILD_PROSE_CONSOLIDATION.md)支持显式保留学生原文供教师核对，缺省消息保持。45份实际保存查询已复验126段原文完整传递，另有1个终局子项；本次无新教师查询、标签或SGD，讲解质量收益仍待真实对照。

[训练专用的学生对引擎棋局](docs/STUDENT_TRAINING_GAMES.md)已接入原始回答、显式合法回退及断点恢复。脚本学生／真实CPU引擎控制通过；真实神经对局与新来源挖掘合并仍待完成，尚无新增训练标签或棋力结论。

[训练棋局的强引擎重评](docs/STUDENT_TRAINING_GAME_SCORING.md)沿用作者公开代码的错误优先及每局普通局面配额，保留完整历史、实际执行走法和全部原始引擎响应。机械重评与筛选独立验证；真实神经执行认证、保留位置隔离及新来源标签合并仍须完成。

[学生对弈来源的搜索接入](docs/STUDENT_GAME_SEARCH_INPUTS.md)增加执行证明与完整模型依赖核验、共享保留位置隔离、明确类型的搜索／收取，以及显式多来源合并。受控CLI链及重开通过，实际神经对弈和对应标签链仍待完成，不计作新模型收益。

已完成记录：[四课](docs/PAPER_FOUR_COURSES_COMPLETED.md)、[初始讲解](docs/INITIAL_SFT_COMPLETED.md)、[联合讲解与回放](docs/JOINT_SFT_COMPLETED.md)、[原讲解384局](docs/INITIAL_SFT_FULL_MATCHES.md)、[专家与标签检查](docs/CURRENT_EXPERT_AND_SEARCH_LABELS.md)、[首次真实更新快照](docs/EXPANDED_SEARCH_DATA_READY.md)。原模型384局为0胜383负1和；联合候选对弈另行保留进行中的原始记录，尚无该候选完整成绩。

[推理目录导出与搬动](docs/FULL_SFT_INFERENCE_BUNDLE.md)已验证旧开发权重的可用性，正式交付仍需模型能力验收。完整范围见[状态](STATUS.md)和[模型说明](docs/MODEL_CARD.md)。

## 安装与 CPU 验证

需要Python 3.11+。在仓库根目录创建独立环境；GPU训练还需适合本机的PyTorch CUDA构建。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,train,reference]'
python -m pytest -q
python -m build
python scripts/check_release.py
```

CPU测试使用受控样例，不下载模型权重。测试通过验证软件行为，模型能力以实际评测结果为准。

## 原型推理

终端学习入口支持显示棋盘、输入走法、查看模型建议、悔棋和保存完整棋局历史：

```bash
python -m xqgeneral.coach --help
```

以下命令用于实际模型交付目录准备好之后；`MODEL_DIR`须替换为真实目录。当前正式模型交付尚未完成，导出方法及已验证范围见[模型交付目录](docs/MODEL_INFERENCE_BUNDLE.md)。

```bash
python -m xqgeneral.coach --checkpoint MODEL_DIR --interactive --output runs/study.json
python -m xqgeneral.coach --checkpoint MODEL_DIR --interactive --resume runs/study.json
python -m xqgeneral.coach --checkpoint MODEL_DIR --interactive --resume runs/study.json --output runs/study-branch.json
```

交互命令为UCCI走法（如`b0c2`）、`hint`、`undo`、`board`和`quit`。恢复记录时默认续写原文件；指定`--output`可另存分支。记录保留原始模型回答及规则核验结果。规则合法不等于选招或讲解正确。

## 数据与训练

[新监督语料的来源重建与整库配对](docs/PAPER_RATIONALE_CORPUS_PRODUCTION.md)及其CPU准备另有记录；本次四课和初始讲解始终使用各自冻结的原数据及配方，未采用该新监督另训学生。[原始源码历史](docs/SOURCE_HISTORY.md)可按实验引用的提交编号恢复。

四门课程用于训练连接桥，随后进行讲解训练和实际搜索蒸馏。四门课程按棋局划分训练／验证／独立测试，并混合已学课程。验证集用于选模，独立测试保留到最终模型与配置锁定之后。

| 阶段 | 执行说明 |
| --- | --- |
| 连接桥与四门课程 | [论文对齐课程实验](docs/PAPER_FOUNDATION_EXPERIMENT.md) |
| 四课后的讲解训练与开发评测 | [后续训练流程](docs/PAPER_POST_CURRICULUM.md) |
| 最终独立棋力与教学验收 | [独立评测协议](docs/FINAL_INDEPENDENT_EVALUATION.md) |
| 训练模型导出与搬动 | [推理交付目录](docs/MODEL_INFERENCE_BUNDLE.md) |

长训练和批量任务在tmux中执行，保存冻结源码、配置、输入输出身份和恢复信息；每次变更配置使用新的输出目录。

大缓存可按[已核验输入哈希复用](docs/VERIFIED_INPUT_DIGESTS.md)显式使用生产者receipt；四个搜索CLI均建立独立的进程内会话，无receipt时仍完整哈希。

完整参数讲解训练可显式使用[参数和梯度分片](docs/FULL_DECODER_PARAMETER_SHARDING.md)，继续保留FP32主参数与BF16计算；实际完整模型容量和恢复仍须单独核验。

完整BF16讲解教师支持[多卡推理和显存预算](docs/FULL_BF16_TEACHER_MULTIGPU.md)；实际加载与推理容量另行实测。

## 讲解教师

初始讲解标签沿用用户授权的GPT-6-Astra Low生成和交叉复核记录；后续以本地完整BF16 Qwen3.8-27B作为蒸馏汇总教师。教师标签、学生实际搜索产量与蒸馏收益分别核验；实际进度见[后续训练流程](docs/PAPER_POST_CURRICULUM.md)。

## 讲解训练与搜索蒸馏

初始讲解已从实际完成四课的模型完成完整参数训练，[完成读回](docs/INITIAL_SFT_COMPLETED.md)记录原生容量／精确恢复、全部更新、所选模型及现有质量问题。[显存控制历史](docs/FULL_DECODER_MEMORY_CONTROLS.md)保留此前失败；当前运行定位见[接续入口](docs/QUALIFIED_SFT_ROUTE.md#本服务器运行定位与恢复)。后续搜索规模和蒸馏轮数根据实际合格数据产量与开发结果决定。[详细开发指南](docs/DEVELOPMENT_GUIDE.md)保留各版本命令和历史证据。

## 原始输出与对弈评测

问答、讲解和对弈均保留原始输出及失败。独立评测不使用引擎替换模型的非法走法；规则约束解码的结果与原始生成分别报告。固定开发题、教师辅助评分和CPU测试不能替代最终独立棋力与教学验收。

## 公共源码与模型

源码及模型资产身份由[来源配置](configs/sources.json)与对应完成清单固定。专家网络、教师权重、数据、特征缓存和训练产物不提交到源码仓库。下载、构建、数据生成及早期实验命令见[开发指南与历史记录](docs/DEVELOPMENT_GUIDE.md)。

## 开源与贡献

源码采用GPL-3.0-or-later，模型、数据和依赖保留各自的许可边界。参阅[第三方说明](THIRD_PARTY_NOTICES.md)、[模型说明](docs/MODEL_CARD.md)与[贡献指南](CONTRIBUTING.md)。本项目独立于Queen、Px0、Qwen和皮卡鱼官方项目。
