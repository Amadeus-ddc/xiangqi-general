# 搜索蒸馏中的四课、数值与走法回放

`xqgeneral.auxiliary_search_replay` 从已经完成的七类语料选择六类非讲解回放，供后续搜索蒸馏复习。
它使用原始训练分割；讲解训练仍来自实际合格搜索标签，讲解验证沿用原始保留题。
辅助训练本身继续使用完整七类语料，回放子集不替代它。

## 可重建的数据选择

每个非讲解题型默认最多选择2048条训练行，以固定种子20261107和原始ID的哈希排名选择，
内存随配额增长。输出恢复原始来源顺序，保留每一行的原始字节。每个题型不足配额时全部保留，
不重复补齐。验证包含来源中全部六类原始行，保留字节和相对顺序。

当前实际来源含训练363445题、验证3749题；默认子集应为六类各2048条训练，共12288条，
验证为四课2600条、数值381条、单走法384条，共3365条。配上原始384条讲解验证后仍是3749题。
这些数字由实际来源重建，程序不接受模板中的固定数量作为完成证据。问题数与独立历史数分别计数。

```bash
python -m xqgeneral.auxiliary_search_replay \
  --data ACTUAL_COMPLETED_SEVEN_STAGE_CORPUS \
  --output FRESH_SIX_STAGE_REPLAY --train-per-stage 2048 --seed 20261107
```

消费者重新从完整来源计算相同选择，核对所有原始训练／验证数量，再逐字节比较子集。重新哈希的
漏行、改题、改答案、改顺序及错误验证数量均不能替代原选择。课程答案使用原生规则重判；数值
答案检查四字段、完整历史下合法变化、候选及评分合同，单走法必须是合法UCCI。没有新引擎或
教师查询，也不解析最终独立测试答案。完成后的同合同`--resume`只核验，保持输出字节及mtime。

课程题的未来局面与数值题的目标PV使用各自的原生合同：数值变化允许在六步之前以合法终局结束，
包括AXF和棋；不能将课程题对未来局面的额外限制套在它上面。数值PV仍逐步检查完整历史，不能在
终局后继续走；单走法只监督当前首着。所有题型仍核验完整已发生历史，课程题原有终局限制保留。
这一区分不会改写原始行、数值目标、模型输入或验证分母。

## 明确启用七类训练与验证

搜索数据分词预检须声明回放类型：

```bash
python -m xqgeneral.distillation_preflight \
  --init ACTUAL_SELECTED_AUXILIARY_MODEL/sft/adapter.pt \
  --parent-completion ACTUAL_SELECTED_AUXILIARY_MODEL/manifest.json \
  --data ACTUAL_ACCEPTED_SEARCH_LABELS --features ACTUAL_EXTENDED_FEATURES/manifest.json \
  --foundation-preflight ACTUAL_PINNED_FOUNDATION_PREFLIGHT \
  --replay-data FRESH_SIX_STAGE_REPLAY \
  --replay-profile original_courses_and_oracle_policy_replay_v1 \
  --output FRESH_SEARCH_TOKEN_PREFLIGHT
```

使用[三卡七类配置](../configs/search-distillation-sft-three-gpu-with-auxiliary-v1.json)，将其中
`replay_data_paths`填为预检返回的真实绝对路径。此配置明确三进程／批量30／微批量1／FSDP2，
以回答词元平均损失训练。分词预检和正式轮次CLI均支持显式`XQGENERAL_VERIFIED_INPUT_RECEIPT`，
仅复用已核验且签名保持的字节哈希；没有收据时完整哈希。模型、缓存、标签和输出目录仍须绑定。

每轮完整遍历全部合格新讲解标签和子集中每一行，比例按实际数量编译；至少一整轮后才能提前停止，
最多四轮且不超过30000步，尾部重复另行计数。新数值与单走法回放不计作新增在策略搜索标签。
数值目标PV也进入合并来源的原生分割隔离检查。正式训练前仍需本轮真实三卡容量和精确续跑预检，
物理GPU2按服务器分配策略排除。

验证选模必须包含原始讲解及全部六类回放验证；训练前的选择计划与实际训练保存的验证ID完整核对。
遗漏数值／走法题不能发布完成清单。完成时沿用辅助轮次数，仅增加实际完成的搜索训练轮次。
旧五类配方保留原有合同；六类回放需要新类型和显式配置。

## 能力验收

CPU受控生产者验证完整原生预检、有限采样、三卡轮次、验证ID、完成模型及续读接口，不代表真实
学生搜索SGD或能力收益。训练后仍须使用固定原始数值／走法诊断、四课保持率、讲解、教师辅助评审
和原始实战分别验收。保留训练任务与轮次数本身不证明没有遗忘，最终独立测试继续留到模型锁定后。
