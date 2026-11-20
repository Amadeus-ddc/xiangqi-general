# 搜索蒸馏中的原始讲解回放

默认六类回放只包含四课、数值和单走法，不包含旧讲解。
可选类型 `original_coaching_courses_and_oracle_policy_replay_v1` 在同一子集中
追加原始讲解训练文件的确定性子序列。它保留原标签、教师身份、完整历史和回答字节，
不生成新标注，也不改变正在执行的旧实验。

## 明确来源与配额

七阶段语料可能已经混入其他轮次的新讲解，不能把它的所有讲解当成原始教师数据。
`--coaching-data` 必须单独指定具有完整语义接纳及引擎资格证明的原始数据集。
准备时重新核对其上游资格合同及输入、输出身份，只解析其训练文件。
验证和测试文件只校验身份；这些文件不会被复制进回放子集，也不参与分词或选样。

```bash
python -m xqgeneral.auxiliary_search_replay \
  --data data/completed-seven-stage-corpus \
  --coaching-data data/original-engine-qualified-explanations \
  --coaching-train-rows 2048 --train-per-stage 2048 --seed 20261107 \
  --workers 8 --output data/search-replay-with-original-coaching
```

每类六任务仍按原算法选择，生成的训练字节是旧六类子集的原样前缀；随后按
原讲解训练文件的顺序追加至多指定配额的讲解。抽样按种子、阶段和原始 ID 的
SHA256 排名，重开时从绑定的来源重新构造全部字节，不能通过改写 manifest 的
完成标志、计数或输出哈希跳过核验。所有选中行都须重新通过原生历史和答案检查；
旧讲解还核对原教师身份、分支、事实和正文中的明确走法。

`--coaching-data` 与正整数 `--coaching-train-rows` 必须同时声明。
省略两者保留原六类类型和原输出合同。已有输出只能按同一来源、配额和种子重开。

## 训练与验证计数

新的分词预检须同时声明：

```text
--replay-data data/search-replay-with-original-coaching
--replay-profile original_coaching_courses_and_oracle_policy_replay_v1
```

使用[单卡回放配置](../configs/search-distillation-sft-one-gpu-cpu-offload-with-coaching-v1.json)，
并将数据路径替换为实际完成目录。它保留全源有限遍历、按词元计算损失、CPU Adam
状态卸载及完整原始验证。旧配方保持原有类型，不能静默接受带旧讲解的新子集。

新讲解和旧讲解共享原生 `explanation` 阶段，训练编译器对两者的行数相加，确保
两者每次完整遍历都被选入；预算仍单列新增接纳行数和旧讲解回放数。
旧讲解不计入论文新增蒸馏标签规模，也不能视为新独立教师标注。

例如，新讲解 971 条、旧讲解 2048 条、其他六类各 2048 条时，每遍共 15307 条，
全局批量 30 对应 511 次更新。原始讲解验证仍仅 384 条，绝不追加另一份 384 条。
其余六类验证保持全部原始行，按实际来源计数。

## 新实验的执行边界

准备子集不等于可以训练：新讲解、旧讲解、六类回放和受保护分割必须联合通过
完整历史、未来分支、镜像位置隔离及专家特征覆盖检查，所有实际训练和验证行都要
重新分词。训练前还须以最终数据完成真实 GPU 连续四步与二步保存后续两步的恢复预检。
已有其他配方的容量证明不能替代新配方的预检。

旧讲解回放旨在检验能力保留，代码和数据检查不能证明它改善了棋力或讲解。
若同时修正新增标签，应把它视为新的组合配方；单独归因于回放需要使用相同标签的
对照。选模仍使用保护验证集，正式晋级仍需完整能力、原始对局及讲解评估。
