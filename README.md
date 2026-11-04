# Xiangqi General（象棋将军）

将象棋专家网络与语言模型连接起来，目标是提供棋力强、能用中文解释走法的个人象棋教练。本项目是受Queen研究流程启发的独立中国象棋实现。

## 当前进度

截至2026-11-04，四门课程已完成。第四课在48000步按连续三次无新高规则结束，最终选择42000步权重。

| 验证范围 | 最新48000步 | 最终选中42000步 |
| --- | ---: | ---: |
| 第四课 | 530/600（88.33%） | 535/600（89.17%） |
| 旧三课回放 | 1920/2000（96.00%） | 1924/2000（96.20%） |
| 合计 | 2450/2600（94.23%） | 2459/2600（94.58%） |

[完整初始讲解训练已结束](docs/INITIAL_SFT_COMPLETED.md)：实际四轮2456次更新，验证NLL选1792步（0.63950）。[同设置课程复查](docs/INITIAL_SFT_RETENTION.md)确认训前FP32对照94.81%降至训后72.92%，旧任务遗忘仍需处理；清零／错配特征分别17.81%／17.27%。末尾8条小型讲解验证样例中，5条首着合法、0条完整通过主线与分支等全部检查。完整讲解、对弈及搜索产量开发评测继续，正式蒸馏、独立棋力／教学验收及正式教练模型仍待完成。上表是讲解前课程模型的结果。

详见[最新状态](STATUS.md)、[模型说明](docs/MODEL_CARD.md)和[四课完成读回](docs/PAPER_FOUR_COURSES_COMPLETED.md)。

[384条正常讲解及末步对照](docs/INITIAL_SFT_DEVELOPMENT_BASELINE.md)已完成：首着合法90.10%，结构与连续变化等合同有效3.91%；2456步旧课72.50%，未改善1792步的72.92%。原始验证已按[完整缓存依赖与完成产物复用](docs/PAIRED_RAW_REUSE.md)接续。[完整未量化27B教师四卡实际推理](docs/FULL_BF16_TEACHER_NATIVE.md)也已读回通过，正式三组汇总、正文盲评、对弈及蒸馏仍待完成。

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
