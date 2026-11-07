# 学生训练对弈来源的准入、搜索与合并

`xqgeneral.student_game_inputs`把[已重评的训练棋局](STUDENT_TRAINING_GAME_SCORING.md)转换为独立类型的搜索输入。原始棋局须在生成前已归属训练，并提供实际神经学生执行、完成后重开及原始输入绑定的证据。脚本学生／CPU控制不能作为该来源准入。

## 准入检查

入口同时核对以下记录，不根据文件名或单个成功字段认定完成：

- 棋局生成清单、执行证明、全部归属棋局、原始回答账本和逐步原生历史重演。真实32局旧执行证明有明确兼容入口；通用证明须给出相同的完整合同。
- 实际父权重、专家、棋局生成策略、引擎及NNUE文件。旧检查点还需核对其内部配置指定的底座、分词器、全部权重片和固定revision，以及实际神经入口保存的完整输入哈希会话；便携模型则核对目录内全部绑定资产。核验不构造模型。
- 强搜索全部原始查询和逐步评分，按原规则重新计算选取结果；重开时禁止新增引擎查询。
- 已完成的真实棋谱搜索池所绑定的基础训练／验证／测试数据及讲解保留文件。新来源复用同一个保留位置集合，每个根及其最多8个实际后续半步连同红黑镜像均须避开该集合。不会把已有训练棋局重新划为独立测试。

新池使用`prepared_student_training_game_roots`类型。每个根保留原始学生棋局ID、演员资产身份、实际出现位置、原始回答及执行走法来源；`student_game_continuation_is_best_move_label`固定为false。实际走过的后续不是最佳着标签，原始非法建议也不会改写为合法建议。

## 生成搜索输入

以下为真实32局及其后续评分完成后的入口示例。长批次放入tmux或调度器，模型及清单参数须指向该次实际执行使用的文件；大输入可显式启用[已核验哈希会话](VERIFIED_INPUT_DIGESTS.md)。

```bash
python -m xqgeneral.student_game_inputs \
  --games runs/on-policy-training-play32-v1/native \
  --scores runs/student-training-game-scoring32-v1/native \
  --execution-proof runs/on-policy-training-play32-v1/verification.json \
  --authority data/unused-recorded-paper-search-inputs-v4 \
  --data data/paper-courses-v5 \
  --heldout-data data/astra-explanations-engine-qualified-v4 \
  --checkpoint runs/paper-sft-fsdp2-v1/pipeline/sft/adapter.pt \
  --expert-weights models/px0-latest.pb.gz \
  --gameplay-policy runs/on-policy-training-play32-v1/policy.json \
  --scoring-policy runs/student-training-game-scoring32-v1/policy.json \
  --decoder-model models/qwen3-4b \
  --decoder-manifest runs/full-decoder-available-capacity-v2/pinned-decoder/manifest.json \
  --input-digest-receipt runs/batched-strict-production-inputs-v1/verified-inputs.json \
  --executable vendor/pikafish/src/pikafish \
  --weights vendor/pikafish/src/pikafish.nnue \
  --output runs/student-training-search-inputs
```

已有完成池用`prepared_student_roots`重开。它重新构建来源、评分、根顺序、完整历史及保留位置覆盖，不能只重算修改后文件的哈希来绕过检查。输出只包含`manifest.json`、`roots.jsonl`和`counts.json`，保留集合仍引用原权威池，不另复制一份会漂移的集合。

## 搜索和收取

搜索CLI的`--student-game-inputs`与`--recorded-inputs`、`--prepared-inputs`互斥，支持原顺序的`--root-offset`分段。请求的检查点须与棋局演员相同，实际推理显式使用其原专家。原有完整历史、所有子分支、可达原始变化前缀、选中分支评分和即时将死检查继续生效。

```bash
python -m xqgeneral.search_distillation \
  --checkpoint runs/paper-sft-fsdp2-v1/pipeline/sft/adapter.pt \
  --data data/paper-courses-v5 \
  --features data/paper-courses-v5/feature-store/manifest.json \
  --student-game-inputs runs/student-training-search-inputs \
  --student-generation-config configs/paper-search-generation-v1.json \
  --nodes 100000 --max-depth 5 --limit 128 --root-offset 0 \
  --output runs/student-training-search-pilot/mining
```

`limit`与偏移不得超过实际池容量。上述入口保留棋局中的原答供来源复核，但当前搜索会重新生成根分析；没有把保存原答谎报为新查询或已启用根缓存复用。教师若需要子讲解原文，使用[显式原文配置](CHILD_PROSE_CONSOLIDATION.md)。

`collect_search --student-game-inputs`重新核对相同来源及每个查询的原根／后代身份，输出单独的学生棋局隔离证据。`search_pilot`配置可使用同名键自动接搜索、实际教师和收取三阶段；完成清单类型为`clean_student_game_search_yield_pilot`，已有真实棋谱试批类型保持。

## 显式多来源合并

只有显式选择以下版本化配置，才允许合并学生对弈与真实棋谱来源：

```bash
python -m xqgeneral.merge_search \
  --collections runs/recorded-search/labels runs/student-training-search-pilot/labels \
  --checkpoint runs/paper-sft-fsdp2-v1/pipeline/sft/adapter.pt \
  --source-profile shared_canonical_sources_v1 \
  --output runs/mixed-search-labels
```

所有来源必须共享精确的保留位置权威清单、基础数据和保留讲解字节、相同父权重、完整教师身份、搜索接纳策略及即时将死检查要求。每个来源的实际学生采样与教师输入配置继续单独保存。默认合并仍限原真实棋谱来源。

合并重建全部接受／拒收记录及原始教师部署，按完整历史去重，保留原行字节和全部重复来源，再执行跨划分原生变化隔离；全局至少32个不同历史，不能用重复记录凑数。包含学生棋局时不宣称全部来源都是原始棋谱。训练前重开会再次完整重建来源与合并产物。

## 当前证据

34项新增及123项相关CPU检查已覆盖来源准入、原专家传递、分段、搜索／教师／收取的受控CLI链、显式混合合并和重开；受控清单不等于实际神经执行。实际1792步检查点与11个底座／分词器文件已另行读回，三片权重8044982000字节，全部11文件8060907830字节，使用既有已核验哈希而未声称重新读取全部权重。

真实32局仍须实际完成，之后再完成强搜索、来源准入、原生挖掘和实际完整BF16教师标签验证。软件检查不提供新训练样本数、合格蒸馏轮次、战略正文语义或棋力改善结论。当前学生权重、现役四卡搜索与教师对照保持各自冻结配置。
