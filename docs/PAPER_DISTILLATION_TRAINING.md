# 搜索蒸馏的正式训练入口

这套入口接在新四课及完整解码器初始讲解 SFT 之后。只有所选初始 SFT 或已完成蒸馏轮次的实际完整检查点可以作为父模型。当前正式四课、初始 SFT 和蒸馏均未完成；受控 CPU 测试不能证明学生棋力或讲解收益。

## 标签与缓存

`collect_search` 保留原始查询、教师回答和拒收原因。接受标签前重新检查完整过去、主线、所有分支、显式讲解走法以及保留集隔离；验证和测试文件按原字节复制。正式数据必须绑定完成的、使用同一父模型和未用实战池的搜索清单，以及完整 BF16 教师身份。

新增评分筛选的默认阈值为 `--max-eval-error-exclusive 0.075`：选中分支的原始学生评分与 oracle 根评分，通过固定皮卡鱼的子力条件模型换成同一根行棋方的期望得分，误差达到阈值即拒收。保留原始评分、两项期望得分、误差和阈值，CPU 预检再次计算，不能直接比较已由 oracle 填好的根字段而得到零误差。

这与作者 [packing.yaml](https://github.com/queen-project/queen/blob/c372da22ca75e95c3c79bee8a83c220fc54d0dcb/configs/self_distill/iteration/packing.yaml) 的阈值一致，测量对象存在差别：作者从汇总正文解析评分，本项目的教师只解释固定结构化事实，因此筛选学生分支评分。正文中的所有数值、战略语义及人类教学质量尚未由此证明；清单显式记为 `consolidated_prose_evaluation_error_measured: false`。这里的期望得分也不是实测人类获胜概率。

新标签的专家特征使用[增量分片缓存](PAPER_FEATURE_STORE.md)。以前的键顺序、张量文件引用、专家身份和层深必须保留，只追加新历史；每次正式预检重新核验全部声明文件的字节。单次预检内复用已哈希绑定并持续检查文件身份，避免对整库反复哈希。旧缓存和完成实验保持不变。

## 配方与预算

[search-distillation-sft-paper-v1.json](../configs/search-distillation-sft-paper-v1.json) 是待编译模板，不能直接交给普通 `sft` 命令。入口按实际合格标签数写出数据路径、训练／验证索引、完整验证数量、有限遍历轮数和最少训练更新；不建立测试索引。

| 设置 | 当前正式模板 |
|---|---|
| 可训练模型 | 完整解码器、连接桥、棋盘词元，FP32 主参数 |
| 四 GPU 全局批量／微批量 | 32／每卡 1，累计至完整全局批量 |
| 桥接／解码器／棋盘词元学习率 | `1e-4`／`1e-5`／`1e-5` |
| 调度／预热／权重衰减 | 恒定／5%／0.1 |
| 标签长度／损失 | 最多 2048 token，无截断，回答 token 平均 |
| 更新上限／验证间隔 | 30000／250，完整保护验证集损失选模 |
| 遍历与早停 | 有限打乱遍历；整轮至少覆盖一次合格训练集且完成预热，之后允许验证早停 |
| 显存设置 | mmap 缓存、DDP 梯度视图、禁用 Adam foreach |

150000 条标签每轮需 `ceil(150000/32)=4688` 次更新；30000 上限分配七轮有限遍历，最后一轮可能不完整。最少更新为一轮完整遍历和预热所需更新的较大值。该下限约束实际整轮训练时长；验证选中的较早检查点可能尚未经历完整遍历，清单分别记录所选步数与实际更新数。不足 150000 条时明确标为未达到论文首轮标签规模，不能把搜索候选数量计为合格标签。

全局批量、学习率和长度参考作者 [self_distill.yaml](https://github.com/queen-project/queen/blob/c372da22ca75e95c3c79bee8a83c220fc54d0dcb/configs/train/self_distill.yaml)。该公开代表配置是 10000 更新，论文实验训练上限与它存在差异；本模板采用[论文表 13](https://arxiv.org/html/2610.03695v1)的 30000 上限。微批量 1、完整既有验证集、一次完整遍历的早停下限及 DDP 是本地选择，不能宣称逐项重放作者的每卡批量 4、累计 2、100 条验证和 FSDP。轮数与后续数据规模仍依据实际棋力和讲解验证调整。

## 执行与恢复

以下路径为示例；必须使用实际完成清单记录的父模型、标签、缓存和基础预检路径。搜索挖掘、真实 BF16 汇总和 `collect_search` 完成后，先提取新标签历史的特征，再运行 CPU 预检：

```bash
python -m xqgeneral.distillation_preflight \
  --init runs/initial-sft/sft/adapter.pt \
  --parent-completion runs/initial-sft/manifest.json \
  --data data/search-round-1 \
  --features data/search-round-1-features/manifest.json \
  --foundation-preflight runs/foundation-preflight/manifest.json \
  --output runs/search-round-1-data-preflight --max-tokens 2048 --workers 8
```

CPU 预检逐条核查全部分割的实际答案和分支隔离，仅编码训练与验证标签。任一标签超长、原始评分筛选缺失、父模型未完成或旧缓存变动，均不产生完成清单。

长任务放入 tmux，并先用 `scripts/freeze_run.py` 固定执行源码：

```bash
python scripts/freeze_run.py --output runs/search-round-1-source -- \
  python -u -m xqgeneral.distillation_training \
  --init runs/initial-sft/sft/adapter.pt \
  --recipe configs/search-distillation-sft-paper-v1.json \
  --token-preflight runs/search-round-1-data-preflight/manifest.json \
  --output runs/search-round-1-training
```

入口先选择实际最长的 32 条训练和 12 条验证标签，以正式全局批量检查四卡连续四步与二加二恢复。完整父模型每个张量都要原值继承；解码器、桥接和棋盘参数都必须实际更新，全部权重、Adam、随机状态、有限遍历位置和日志要求精确一致，各卡记录显存峰值。预检成功才运行正式 SFT。

查看 `state.json`、`preflight/verification.json` 和 `sft-controller.log`。中断后在同一冻结源码、配置、父模型和输出上加 `--resume`；完成的预检和索引复用，实际 SFT 通过现有 `latest.pt` 恢复。变动合同必须使用新输出目录。最终完成清单重新绑定实际训练、有限遍历状态、所选完整参数、索引和缓存；已完成结果的续读也重新核验文件。

## 每轮训练之后的开发验证

初始讲解与已完成的蒸馏轮次共用 `clean_sft_evaluation`、`clean_sft_matches` 入口。蒸馏模式要求实际 `full_decoder_search_distillation_sft_round` 完成清单、该轮编译的 `recipe.json` 及原始数据预检；核对选中模型、全部 FP32 参数与初始化形状、真实更新数、训练输入和源码。验证集及其原缓存继承必须保持原合同，独立测试不用于训练或选模。

[evaluation-paper-distillation-v1.json](../configs/evaluation-paper-distillation-v1.json) 对全部 384 条保护验证题分别生成正常／清零／错配专家特征的原始回答，再执行原生规则、独立引擎判断和完整 BF16 教师的匿名正文评分。预算与初始讲解开发基线一致。[matches-paper-distillation-v1.json](../configs/matches-paper-distillation-v1.json) 随后对同一实际选中模型运行 64 个验证开局、红黑互换及三个节点预算，共 384 局。非法原始答案照实记录，截断对局单列，不算和棋。

这两份配置的路径是待替换示例；必须与实际完成清单中的数据、编译配方、预检和缓存路径一致。每轮使用新的配置副本与输出目录，长任务固定源码后放在 tmux 中：

```bash
python scripts/freeze_run.py --output runs/search-round-1-evaluation-source -- \
  python -u -m xqgeneral.clean_sft_evaluation \
  --pipeline runs/search-round-1-training --producer-session xqgeneral-search-round-1-training \
  --config configs/evaluation-paper-distillation-v1.json --output runs/search-round-1-validation
python scripts/freeze_run.py --output runs/search-round-1-evaluation-source -- \
  python -u -m xqgeneral.clean_sft_matches \
  --pipeline runs/search-round-1-training --validation runs/search-round-1-validation \
  --producer-session xqgeneral-search-round-1-evaluation \
  --config configs/matches-paper-distillation-v1.json --output runs/search-round-1-matches
```

完成报告分别记为 `search_distillation_capability_validation` 和 `search_distillation_full_match_baseline`，记录实际已训练轮次。对弈要求同一模型的整套三组能力验证先完成。恢复复用已有原始记录；完成后恢复不会再次生成回答或对局，也会核对报告正文与清单中的聚合值。教师评分不等于人类评价，这些验证不构成最终独立棋力或教学验收。

[现役等待器](QUALIFIED_SFT_ROUTE.md#本服务器运行定位与恢复)依次覆盖四课后的初始讲解、三组开发验证、384局对弈基线和512根真实搜索产量试批。正式蒸馏仍须按实际合格标签及本轮结果另行安排；上述开发验证入口可用于每个完成的蒸馏轮次。蒸馏轮次、棋力收益与最终模型验收尚未完成。
