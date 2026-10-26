# 新四课之后的讲解与评测

现役路线使用筛选后的9823条训练标签、384条验证和512条独立测试，执行位置、所有权和恢复边界统一见[筛选后讲解接续](QUALIFIED_SFT_ROUTE.md#本服务器运行定位与恢复)。它接在`curriculum-paper-user-direct26000-v1`实际完成的四课之后，并显式核验[用户指定的三课前缀](USER_AUTHORIZED_COURSE_PREFIX.md)。当前第四课正在训练，完整初始讲解、真实搜索蒸馏与最终验收尚待执行。

## 初始讲解训练

初始教师沿用用户授权的GPT-6-Astra Low。原10758条训练标签的引擎筛选、保留集字节和正文接受链见[数据准入](ENGINE_QUALIFIED_EXPLANATIONS.md)与[参考正文修订](REFERENCE_PROSE.md)。训练、验证建立索引；独立测试不建立训练索引，也不参与选模。

讲解继承实际第四课选中的桥接／棋盘权重及同一预训练解码器。完整解码器、桥接和棋盘词元采用FP32主参数、BF16计算及回答token平均损失；四GPU、全局批量16、每卡微批量1。桥接／解码器／棋盘词元学习率为`1e-5`／`1e-6`／`1e-6`，预热5%后恒定，权重衰减0.1。Adam foreach关闭，DDP梯度复用通信缓冲区。

四轮有限遍历每轮完整呈现9823条训练项，编译为每轮614次、共2456次更新；共39292条有效呈现、4条分布式填充。配置中的原4096步上限不是本次实际更新数。验证NLL选择讲解检查点，随后另行检查原始走法、完整变化、正文及对弈。

当前新索引及实际缓存／词元预检已完成。全部10719个特征上下文存在，10207条训练／验证编码最长649／625，在1024上限内无监督截断。这些证明不含最终四课父权重或完整解码器GPU容量；四课结束后仍须真实四卡连续四步、二加二续跑及最长标签容量检查，成功后才训练讲解。

## 评测与蒸馏

初始讲解完成后使用[四课验证集复查](POST_SFT_CURRICULUM_VALIDATION.md)记录课程能力保留、原始格式错误和三组专家特征结果。每个完成的蒸馏轮次也可执行同一入口；与原课程题目一致时提供逐题变化，结果用于开发比较。

同一初始SFT检查点对全部384条验证标签生成正常、清零、错配专家特征三组原始回答，再做原生规则、独立引擎及完整BF16 Qwen3.8-27B的匿名正文辅助评分。学生和教师顺序加载；教师评分不等于人工评价。

开发对弈使用64个验证开局、红黑交换及Pikafish每步100／1000／10000节点，共384局，最多新增256半回合。非法首着或变化保留，截断单列。对弈要求同一检查点的整套开发能力验证先完成。

等待器接着运行512根真实搜索产量试批，沿用固定未用棋谱池和完整BF16教师。试批只用于量测合格标签产量与错误；它不会自动训练一个小数据蒸馏轮。正式搜索量、蒸馏轮数和后续开发比较按实际结果继续安排，见[正式蒸馏入口](PAPER_DISTILLATION_TRAINING.md)。

[最终独立协议](FINAL_INDEPENDENT_EVALUATION.md)另用保护测试集。正式训练和开发选模完成后先锁定模型，再执行独立棋力、讲解及教学核查，最后导出实际训练模型。

## 可复用命令

以下用于自己的新部署；本服务器已有拥有者，按[当前运行定位](QUALIFIED_SFT_ROUTE.md#本服务器运行定位与恢复)检查既有任务。配置副本须统一实际数据、特征、索引、基础预检和输出路径的写法。公共模板保留兼容的示例数据路径，使用前按本次实际数据调整。

先将`XQ_POST_DATA`、`XQ_POST_INDEX`、`XQ_POST_SOURCE`、`XQ_POST_OUTPUT`指向本次数据、索引、新冻结源码目录和新输出目录；`XQ_FOUNDATION_CONFIG`、`XQ_FOUNDATION_PREFLIGHT`来自本次完成的基础输入。长任务放在tmux或调度器中。

```bash
.venv/bin/python scripts/freeze_run.py --output "$XQ_POST_SOURCE" -- \
  .venv/bin/python -m xqgeneral.training_index \
  --source "$XQ_POST_DATA/train.jsonl" --split train --output "$XQ_POST_INDEX/train"
.venv/bin/python scripts/freeze_run.py --output "$XQ_POST_SOURCE" -- \
  .venv/bin/python -m xqgeneral.training_index \
  --source "$XQ_POST_DATA/validation.jsonl" --split validation --output "$XQ_POST_INDEX/validation"
.venv/bin/python scripts/freeze_run.py --output "$XQ_POST_SOURCE" -- \
  .venv/bin/python -m xqgeneral.explanation_preflight \
  --data "$XQ_POST_DATA" --foundation-config "$XQ_FOUNDATION_CONFIG" \
  --foundation-preflight "$XQ_FOUNDATION_PREFLIGHT" --max-tokens 1024 --workers 8 \
  --output "$XQ_POST_OUTPUT/token-preflight"
```

已完成的实际索引和词元预检应读回复用，避免覆盖。设置`XQ_POST_CURRICULUM`和`XQ_CURRICULUM_SESSION`为实际四课输出与对应拥有会话；`XQ_POST_RECIPE`、`XQ_POST_EVALUATION`、`XQ_POST_MATCHES`为本次配置副本，`XQ_POST_OWNER_SESSION`为实际拥有对应生产工作的会话。

```bash
.venv/bin/python scripts/freeze_run.py --output "$XQ_POST_SOURCE" -- \
  .venv/bin/python -u -m xqgeneral.sft_pipeline \
  --curriculum "$XQ_POST_CURRICULUM" --producer-session "$XQ_CURRICULUM_SESSION" \
  --recipe "$XQ_POST_RECIPE" --token-preflight "$XQ_POST_OUTPUT/token-preflight/manifest.json" \
  --output "$XQ_POST_OUTPUT/pipeline"
.venv/bin/python scripts/freeze_run.py --output "$XQ_POST_SOURCE" -- \
  .venv/bin/python -u -m xqgeneral.clean_sft_evaluation \
  --pipeline "$XQ_POST_OUTPUT/pipeline" --producer-session "$XQ_POST_OWNER_SESSION" \
  --config "$XQ_POST_EVALUATION" --output "$XQ_POST_OUTPUT/capability-validation"
.venv/bin/python scripts/freeze_run.py --output "$XQ_POST_SOURCE" -- \
  .venv/bin/python -u -m xqgeneral.clean_sft_matches \
  --pipeline "$XQ_POST_OUTPUT/pipeline" --validation "$XQ_POST_OUTPUT/capability-validation" \
  --producer-session "$XQ_POST_OWNER_SESSION" --config "$XQ_POST_MATCHES" \
  --output "$XQ_POST_OUTPUT/baseline"
```

`--producer-session`必须指向实际拥有对应前序工作的会话。公共SFT、开发验证、对弈与搜索试批入口支持在原拥有者确已终止、输入／源码及已有合同一致时显式`--resume`；不可因观察超时重复启动。不可续跑的部分阶段使用新目录。

## 本服务器当前队列

现役登记集中在[筛选后讲解接续](QUALIFIED_SFT_ROUTE.md#本服务器运行定位与恢复)。旧`paper-followthrough-catalog-v1`讲解等待器已退役；它的源码、数据、索引和日志继续作为历史证据保留。目录生产时的历史交接见[实战局面目录](NATIVE_POSITION_CATALOG.md)。
