# 最终独立棋力与讲解测试

[固定配置](../configs/final-independent-evaluation-v1.json)把最终测试与[开发验证](PAPER_POST_CURRICULUM.md#评测与蒸馏)分开。先完成新四课、初始讲解和真实搜索蒸馏，根据验证集和开发对弈选定交付模型，再锁定检查点和完成证明，最后生成独立测试回答。准备测试输入不加载模型、不生成成绩，也不代表最终验收通过。

## 固定范围

| 项目 | 数据与预算 | 报告内容 |
| --- | --- | --- |
| 四课问答 | `paper-courses-v5` 的全部26000条测试项，26类各1000条；正常、清零、错配专家特征各一遍 | 原始标签正确率、各题型及最弱题型、格式有效率；三种问法按固定顺序分配，每条只问一次 |
| 结构化讲解 | 修订参考集全部512条测试项；同样三组专家特征；每条最多768个新词元，独立引擎每局面100万节点 | 原文、解析与走法合法率、首着及完整变化质量、所有失败与条件统计的分母 |
| 对弈 | 64个独立测试棋局的12半回合开局，交换红黑，Pikafish每步100／1000／10000节点，共384局；最多新增256半回合 | 每档预算及每种颜色的胜、负、和、截断与非法输出；截断单列 |
| 正文辅助评审 | 正常特征下全部512条原始讲解；完整BF16汇总教师匿名评审 | 事实、战略、清晰、遵循要求、无依据声明，以及无效或截断评分 |

问答与讲解的三组特征使用相同样本、顺序和生成预算；错配组每批至少两条。独立测试标签不进入训练索引。讲解项中的颜色派生仍按派生项报告，不把512条视为512个独立人类棋局。

对弈开局直接从本次 `paper-courses-v5` 测试项的原谱历史读取。准备和读取时都重新核对完成的课程清单、文件绑定、测试归属及实际历史；继承生产者已经复核的棋局归属，保留旧数据集的直接哈希分割入口。开局只含已发生的历史，不含参考答案和原谱未来。标准开局可能在其他棋局出现；这项控制证明棋局隔离，不声明开局几何形状从未出现过。

本服务器已完成[输入准备与隔离读回](../evidence/final-independent-evaluation-inputs-v1.json)：26000题、512条讲解、64个不同测试棋局及不同开局均已核对，所选棋局与当前课程／讲解训练和验证范围的重叠为0。复用已完成的全量来源隔离证明，没有重新全读训练题库或专家特征。最终成绩另行保存；软件检查、准备清单或训练损失均不能替代棋力与教学结果。

## 测试前锁定模型

在第一次生成测试回答前，将以下内容写入该次运行的只读选择记录，并保存哈希：

- 实际选定检查点的路径、SHA256、选择步数，以及完整课程、初始SFT、蒸馏轮次和父模型完成证明。
- 验证集能力结果、开发对弈结果及选定该模型的理由；这些结果只能来自开发分割。
- 本配置、测试数据清单、开局清单、实际执行源码、底座／专家／引擎／评审教师的身份与哈希。

当前没有已完成这些训练阶段的正式交付模型，因此尚不能生成这份模型选择记录或执行以下模型评测。选择记录完成后只测试该模型，测试结果用于报告与验收，不回流选模。若依据测试结果修改训练，本套测试视为已暴露，下一版独立结论需要新的保留测试。

## 可复用入口

从实际包含模型、数据和引擎的项目根目录运行；使用相同的路径写法。长评测放在tmux中，保留检查点选择记录及每个阶段的清单。以下命令逐项对应固定配置，不会自动选择检查点或判定模型合格。

先准备开局。已存在的完成目录应读回复用；不要覆盖输入准备时已经保存的开局。

```bash
.venv/bin/python scripts/freeze_run.py --output runs/final-independent-inputs/source-run -- \
  .venv/bin/python -m xqgeneral.match_openings \
  --data data/paper-courses-v5 \
  --split test --count 64 --plies 12 --seed 20261079 \
  --output runs/final-independent-evaluation-v1/openings-v3
```

完成并核对模型选择记录后，设置实际检查点与新的评测目录：

```bash
export XQ_FINAL_CHECKPOINT=/path/to/selected-completed-distillation/adapter.pt
export XQ_FINAL_OUTPUT=runs/final-independent-model-v1
export XQ_FINAL_SOURCE=runs/final-independent-model-v1/source-run

for memory in normal zero shuffled; do
  .venv/bin/python scripts/freeze_run.py --output "$XQ_FINAL_SOURCE" -- \
    .venv/bin/python -m xqgeneral.evaluate_qa \
    --checkpoint "$XQ_FINAL_CHECKPOINT" --data data/paper-courses-v5 \
    --features data/paper-courses-v5/feature-store/manifest.json \
    --split test --per-task 1000 --question-formats 3 --batch-size 8 \
    --max-new-tokens 384 --memory "$memory" --seed 20261015 \
    --output "$XQ_FINAL_OUTPUT/qa-$memory"
  .venv/bin/python scripts/freeze_run.py --output "$XQ_FINAL_SOURCE" -- \
    .venv/bin/python -m xqgeneral.evaluate_explanations \
    --checkpoint "$XQ_FINAL_CHECKPOINT" --data data/astra-explanations-reference-corrected-v3 \
    --features data/paper-courses-v5/feature-store/manifest.json \
    --split test --limit 512 --batch-size 4 --max-new-tokens 768 \
    --memory "$memory" --nodes 1000000 --workers 8 --seed 20261015 \
    --executable vendor/pikafish/src/pikafish --weights vendor/pikafish/src/pikafish.nnue \
    --output "$XQ_FINAL_OUTPUT/explanations-$memory"
done

.venv/bin/python scripts/freeze_run.py --output "$XQ_FINAL_SOURCE" -- \
  .venv/bin/python -m xqgeneral.evaluate_games \
  --checkpoint "$XQ_FINAL_CHECKPOINT" \
  --opening-suite runs/final-independent-evaluation-v1/openings-v3 \
  --nodes 100 1000 10000 --max-plies 256 --action-mode explanation --move-beams 4 \
  --expert-weights models/px0-latest.pb.gz \
  --executable vendor/pikafish/src/pikafish --weights vendor/pikafish/src/pikafish.nnue \
  --output "$XQ_FINAL_OUTPUT/matches"
```

原始问答和讲解入口要求新输出目录。对弈入口支持 `--resume`，仅在原工作进程确已结束、输入和冻结源码未变化时使用；不因观察超时重启存活任务。问答入口可显式使用[已核验输入证明](VERIFIED_INPUT_DIGESTS.md)减少重复哈希；变动文件必须重新核验，各入口继续执行各自原有的输入检查。

正常特征的规则与引擎判断完成后，再生成匿名正文评审；学生和完整教师顺序加载：

```bash
.venv/bin/python scripts/freeze_run.py --output "$XQ_FINAL_SOURCE" -- \
  .venv/bin/python -m xqgeneral.review_explanations prepare \
  --predictions "$XQ_FINAL_OUTPUT/explanations-normal/judged-predictions.jsonl" \
  --limit 512 --seed 20261015 --output "$XQ_FINAL_OUTPUT/review-queries"
.venv/bin/python scripts/freeze_run.py --output "$XQ_FINAL_SOURCE" -- \
  .venv/bin/python -m xqgeneral.local_teacher --config configs/teachers.json \
  --input "$XQ_FINAL_OUTPUT/review-queries/queries.jsonl" --max-new-tokens 1536 \
  --output "$XQ_FINAL_OUTPUT/reviewer"
.venv/bin/python scripts/freeze_run.py --output "$XQ_FINAL_SOURCE" -- \
  .venv/bin/python -m xqgeneral.review_explanations collect --config configs/teachers.json \
  --queries "$XQ_FINAL_OUTPUT/review-queries/queries.jsonl" \
  --responses "$XQ_FINAL_OUTPUT/reviewer/responses.jsonl" \
  --output "$XQ_FINAL_OUTPUT/ratings"
```

## 交付判断

报告同时展示全部题型、不同引擎预算、红黑方、原始失败与讲解例子。相同开局的颜色交换和不同预算属于相关观测，不能把384局当作384个独立开局；若估计区间，按64个开局成组计算。节点预算对局不直接换算人类等级分，也不证明达到国际象棋论文的Elo。

正文评审教师同时参与训练数据汇总，其分数是辅助证据。教学判断还要逐条核对实际回答中的棋子、走法、变化、理由及风险，并展示对学习者有用的成功例子和错误例子；流畅或平均分较高不能掩盖事实错误。最终模型卡应给出实测适用范围及仍不可靠的场景，不预先设定“99%总分／95%最弱题型”这类论文未确认的合格线。

当前阶段只准备输入与协议。实际模型选择记录、独立成绩、正文核查、推理导出及训练好的教练模型仍待完成。
