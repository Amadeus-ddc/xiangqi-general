# 学生训练棋局的强引擎重评

`xqgeneral.training_game_scoring`读取已完成的[训练棋局](STUDENT_TRAINING_GAMES.md)，逐步重演原始记录，再用独立的CPU引擎比较实际走法与强搜索选招。它生成待挖掘的局面，不生成讲解标签，也不更新学生权重。

## 评分与抽样

默认设置对应作者公开代码`datagen/self_distill/score_play_positions.py`和`play_engines.py`中的选择方式，保留其与论文正文不同的参数：

- 每个完整历史先搜索100000节点。实际走法不等于引擎选招时，用200000节点、MultiPV 2限定比较两招；缺少某招的精确评分时，单独限定该招搜索100000节点。
- 节点停止时，新选招可能只有上下界。原始输出保留，不能把上下界当精确评分。该招仍可作为待比较走法，由后续限定搜索取得精确评分。
- 去重使用棋子布局及行棋方。重复局面保留期望得分损失最大的实际出现；并列按棋局索引、步数、行棋方排序，保留被选出现的完整历史。
- 保留全部期望得分损失至少0.05的错误局面。普通局面每局目标数为去重前实际步数除以10向上取整；绝对评分至少250 cp的普通局面先按4倍下采样。实际不足的普通局面配额如实报告，不补造记录。
- 中国象棋使用项目现有的皮卡鱼WDL及材料相关标定。这里的期望得分损失不是人类实战胜率；将死评分保持原类型，不转成任意大的cp值。

红黑双方实际走过的局面均可进入机械重评。学生原始回答、原始种子和生成设置保留在`cached_root`，实际执行的合法回退走法仍标明来源；不会把原答中的非法走法改写成合法原答。

## 运行与恢复

```bash
python -m xqgeneral.training_game_scoring \
  --games runs/student-training-games/native \
  --policy configs/student-training-score-v1.json \
  --output runs/student-training-scores \
  --executable vendor/pikafish/src/pikafish \
  --weights vendor/pikafish/src/pikafish.nnue
```

长批次放入tmux或调度器。记录原命令，恢复时使用完全相同的输入、源码和参数并加`--resume`。同一输出目录有排他锁。变更配置应使用新目录。

`oracle-queries.partial.jsonl`在返回结果前保存完整查询、原始输出及成功／失败状态；失败输出不丢弃。`positions.partial.jsonl`保存已完成的逐步评分。续跑重用原查询并重建评分，最终输出`scored-positions.jsonl`、`selected-positions.jsonl`、`summary.json`和绑定输入／输出的清单。已完成产物重开时先校验完整覆盖及文件摘要，再零引擎加载地重演，不改写原产物。

## 验证与当前边界

35项新增／75项相关CPU检查覆盖原始历史、精确候选缺失、上下界选招、将死类型、两种颜色、实际回退来源、失败原答、续跑前缀、去重／配额及完成后重开。真实CPU控制对4局各重评43个实际局面，完整与中断／恢复合计139次实际引擎查询；各保留15个待挖掘局面，完成后重开零引擎加载且字节／修改时间保持。详细查询数量见[evidence/student-training-game-scoring-cpu-v1.json](../evidence/student-training-game-scoring-cpu-v1.json)。第一次实际控制在只有上下界的新选招处失败，其输出和冻结源码保留；两个相同模式回归在旧源码失败，在新源码通过。

机械重评接受明确标注的脚本学生控制，以便独立验证引擎链路。即使附带原生棋局清单，本入口也不据此认证神经学生实际执行或完整模型资产：`source_neural_execution_verified`及`source_model_assets_verified`保持false。

后续训练池准入仍需独立核验真实神经执行证据、相同父权重、保留位置及后续变化的隔离，并经实际搜索、教师汇总和严格收取。这里选出的局面不能直接计作训练标签、合格蒸馏轮次或棋力提升。现役记录棋谱来源、对弈评测、教师试批和学生权重保持各自冻结配置。
