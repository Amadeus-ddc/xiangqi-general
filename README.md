# Xiangqi General（象棋将军）

将象棋专家网络与语言模型连接起来，目标是提供棋力强、能用中文解释走法的个人象棋教练。本项目是受Queen研究流程启发的独立中国象棋实现。

## 当前进度

四门课程、初始讲解和讲解／旧课联合训练已完成；**可靠教练模型尚未交付**。旧联合模型的384局开发对弈为0胜383负1和，其中266局因原始走法非法判负。软件测试通过、课程正确率和训练损失均不能替代棋力或教学验收。

截至2026-11-09 23:19（UTC+8）的固定快照，首轮七类辅助训练已正常完成：累计16000步，选中12000步最佳权重。固定765题的全部原始回答已重算；数值题首着合法369/381、六个半回合变化有效228/381，仍有非法走法与变化线错误。完整讲解、原始对弈、合格新搜索训练和最终独立验收继续推进。2026-11-10起，本项目仅允许物理GPU0，GPU1、2、3供用户预留。

当前事实与证据统一见[模型卡](docs/MODEL_CARD.md)；实际任务、小时观察和恢复入口见[状态页](STATUS.md)。两页均注明快照日期，公开仓库不提供实时训练步数。

| 入口 | 内容 |
| --- | --- |
| [当前模型与质量证据](docs/MODEL_CARD.md) | 已完成阶段、已测失败、数据范围、未完成验收 |
| [首轮辅助训练与765题实测](docs/AUXILIARY_ROUND_COMPLETED.md) | 实际16000步完成、所选12000步、原始回答诊断和剩余缺口 |
| [两卡训练与完整状态迁移](docs/AUXILIARY_RESOURCE_TRANSITION.md) | 全参数／Adam／进度保留、真实容量和恢复证明 |
| [七类辅助训练](docs/AUXILIARY_POLICY_TRAINING.md) | 讲解、四课、走法和数值分析的全量有限遍历 |
| [后续开发评测](docs/AUXILIARY_POLICY_EVALUATION.md) | 原始问答、三组讲解、完整教师辅助评审和原始对弈 |
| [搜索及六类回放](docs/AUXILIARY_SEARCH_REPLAY.md) | 合格新标签、旧能力保留及按指定CPU进程数执行的完整原生复核 |
| [搜索讲解复核剔除](docs/SEARCH_PROSE_EXCLUSIONS.md) | 保留原始回答，绑定逐条剔除原因并在合并／训练预检时重建 |
| [候选杀棋步数的原生证明](docs/NATIVE_TERMINAL_AUDIT.md) | 完整历史下覆盖全部对手应手，区分更短杀棋反例、未找到证明与资源耗尽 |
| [汇总讲解的原生核对信息](docs/GROUNDED_CONSOLIDATION.md) | 显式新提示区分分数来源、应将方式和双方吃子统计；教师质量仍须实测 |
| [单卡 FP32 Adam 状态卸载及搜索蒸馏](docs/SINGLE_GPU_ADAM_OFFLOAD.md) | CPU 保存动量；单卡正式入口核验实际状态位置、七类回放及完整恢复；真实容量仍需预检 |
| [验证参考解析器兼容](docs/VALIDATION_DECODER_COMPATIBILITY.md) | 类型化缺分错误的严格源码核验与新保护清单重建 |
| [最终独立验收](docs/FINAL_INDEPENDENT_EVALUATION.md) | 锁定最终模型后执行的独立测试 |

历史里程碑保留在[状态档案](STATUS_HISTORY.md)、[模型卡档案](docs/MODEL_CARD_HISTORY.md)及各实验文档中；其中的旧卡数、等待器和中间结果按当时快照解释。

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

Python 推理接口可显式使用 `Predictor(..., device='cpu')`。CPU 精度与交付目录的运行时版本要求见[CPU 推理说明](docs/MODEL_INFERENCE_BUNDLE.md#cpu-推理)。

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

完整BF16讲解教师支持[多卡推理和显存预算](docs/FULL_BF16_TEACHER_MULTIGPU.md)。搜索阶段可[单独声明教师部署](docs/SEARCH_TEACHER_DEPLOYMENT.md)；同一固定权重已完成单卡 TP1 的两条实际生成，完整递归搜索吞吐仍待实测。

## 讲解教师

初始讲解标签沿用用户授权的GPT-6-Astra Low生成和交叉复核记录；后续以本地完整BF16 Qwen3.8-27B作为蒸馏汇总教师。教师标签、学生实际搜索产量与蒸馏收益分别核验；实际进度见[状态页](STATUS.md)，方法和原始配方见[后续训练流程](docs/PAPER_POST_CURRICULUM.md)。

## 讲解训练与搜索蒸馏

初始讲解已从实际完成四课的模型完成完整参数训练，[完成读回](docs/INITIAL_SFT_COMPLETED.md)记录原生容量／精确恢复、全部更新、所选模型及现有质量问题。[显存控制历史](docs/FULL_DECODER_MEMORY_CONTROLS.md)保留此前失败；当前运行定位见[状态页](STATUS.md#当前运行与续跑)。后续搜索规模和蒸馏轮数根据实际合格数据产量与开发结果决定。[详细开发指南](docs/DEVELOPMENT_GUIDE.md)保留各版本命令和历史证据。

## 原始输出与对弈评测

问答、讲解和对弈均保留原始输出及失败。独立评测不使用引擎替换模型的非法走法；规则约束解码的结果与原始生成分别报告。固定开发题、教师辅助评分和CPU测试不能替代最终独立棋力与教学验收。

引擎在固定预算下只给出评分上下界时，会保留“精确评分未测量”的覆盖信息；已保存的完整讲解可[单独恢复 CPU 评分](docs/EXPLANATION_SCORE_RECOVERY.md)，无需重新生成模型回答。

对弈可[显式设置推理转换缓存](docs/MATCH_INFERENCE_CACHE.md)。默认沿用检查点配置；显式开关写入续跑合同，实际吞吐和显存仍需按模型与上下文测量。

## 公共源码与模型

源码及模型资产身份由[来源配置](configs/sources.json)与对应完成清单固定。专家网络、教师权重、数据、特征缓存和训练产物不提交到源码仓库。下载、构建、数据生成及早期实验命令见[开发指南与历史记录](docs/DEVELOPMENT_GUIDE.md)。

## 开源与贡献

源码采用GPL-3.0-or-later，模型、数据和依赖保留各自的许可边界。参阅[第三方说明](THIRD_PARTY_NOTICES.md)、[模型说明](docs/MODEL_CARD.md)与[贡献指南](CONTRIBUTING.md)。本项目独立于Queen、Px0、Qwen和皮卡鱼官方项目。
