# 当前学生对引擎的训练对局来源

`xqgeneral.training_play`生成训练专用的学生／Pikafish对局上下文。每局在提问之前，由冻结策略、学生资产身份和棋局序号确定唯一`student-train-...`标识与`train`归属；来源明确标作`student_engine_training_game`。从标准初始局面出发，不读取保留验证／测试开局，不产出教学标签。

实现参考作者固定[c372da2的训练对局代码](https://github.com/queen-project/queen/blob/c372da22ca75e95c3c79bee8a83c220fc54d0dcb/datagen/self_distill/get_play_positions.py)：双方颜色按序号交替，每局对手节点预算在100至100000间稳定地按对数均匀采样；学生使用温度0.6、top-k20、top-p0.95、最多4096词元。公开[32局试批配置](../configs/student-training-play-v1.json)采用四请求解码批量，最多八个活跃棋局；这不是论文训练规模或已执行的神经对局成绩。

## 走法与原文记录

训练采样依次尝试合法的根`move`、主变化第一着和独立`candidates`，其后使用带固定种子的随机合法回退。只有位于JSON根对象开头的完整`move`字段可以在后续JSON未完成时单独保留；不从变化后续走法或讲解正文中拼凑根建议。该适配使用中国象棋的UCCI与完整历史规则。

每次实际执行的走法与学生原始回答分别保存，`move_source`、原始声明是否合法、整个JSON能否解析都明确记录。随机回退不调用棋局评判引擎选招；对手引擎的完整原答、实际预算和输入历史分别保留。这是训练数据生成策略。实战评测继续按原始回答记录非法弃负，生成的训练对局不计入棋力评测。

## 执行与恢复

```bash
python -m xqgeneral.training_play --checkpoint MODEL_DIR --policy configs/student-training-play-v1.json --output runs/student-training-play-v1
python -m xqgeneral.training_play --checkpoint MODEL_DIR --policy configs/student-training-play-v1.json --output runs/student-training-play-v1 --resume
```

实际长任务应在tmux或调度器中运行，并使用冻结源码与输入核验记录。输入、源码、策略和学生身份改变时使用新输出目录。原始批次账本`student-batches.partial.jsonl`先于走法提交落盘；恢复会重演完整历史并复用已保存回答，随后只生成尚未记录的请求。恢复前核对所属棋局文件及已完成覆盖，缺失或多出棋局会提前拒绝；零次学生调用也保留空账本。同一输出的CLI生产者由文件锁互斥。已完成任务重新打开时不加载双方、不改写已发布文件。不同真实GPU批量形状可能改变后续数值输出，不承诺重新执行整个独立对局得到逐字节相同的引擎日志。

原生输出包含每局`games/game-*.json`、完整`games.jsonl`、策略／资产合同和摘要。将死、困毙及历史规则按原生棋规结束；达到半回合上限标作`censored`，不记成和棋。记录包含完整原始历史，未把学生回退后的合法着法标为原始预测正确。

## 当前证据与后续处理

35项新增检查及83项相关CPU检查通过。另以脚本学生和真实单线程Pikafish完成4局短控制，实际44次引擎查询；断点前原始批次及已提交走法前缀保持，全部棋局重演通过，完成后重开不调用双方。1局原生结束、3局按上限截断。该控制的真实神经学生调用、教师调用、SGD和CUDA初始化均为0，见[摘要](../evidence/student-training-games-cpu-v1.json)。早期控制错误要求独立引擎执行的计时日志也逐字节相同；该失败保留，后续控制明确核验保存前缀和原生历史。

真实学生试批尚待执行。生成的原始棋局还需独立强搜索打分、错误聚焦筛选、跨划分根及后续局面排除，以及新来源的正式挖掘／合并合同，才能进入训练。当前既有实战棋谱池与四卡采集计划保持原来源，不能直接把这批原始上下文混入已经冻结的训练数据。最终棋力、战略语义和教练效果仍待验收。
