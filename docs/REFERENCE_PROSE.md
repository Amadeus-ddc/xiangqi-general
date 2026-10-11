# 讲解参考正文修订

训练正文继续使用 `xqgeneral.revise_prose`，其默认合同只允许训练项，验证与独立测试原文件保持不变。`xqgeneral.reference_prose` 是另一个入口：在新学生实验之前，依据原教师、原生规则和已有引擎记录，修正旧验证／独立测试参考正文。审稿不使用学生预测，独立测试不进入训练或选模。

## 原文、拒收与接受

审查包声明为 `independently_reviewed_heldout_reference_prose`，以完整文件 SHA256 和字节数绑定原文包、逐条决策、真实修订稿及独立接受。`references.jsonl` 每条含 `id`、`source_label_line_sha256`、`source_annotation_sha256` 和 `reviewed_annotation_sha256`，必须覆盖包内全部原始 ID。

正文哈希使用 UTF-8、排序键、紧凑 JSON，内容只含 `id`、`explanation` 和三项教师身份；完整原始 JSONL 行的哈希另行绑定颜色派生、历史和其他元数据。这个哈希格式与旧训练正文入口的格式不同，不能互换。

每份修订记录须绑定被拒收正文的哈希、对应拒收文件的哈希和原标签行的哈希。最终决策须绑定实际作者文件，审阅者与作者不同；四个维度均至少 4 分且无未支持声明才接受。修订需要 80–250 个汉字，保留原文和失败稿，不允许以自我接受、缺失父稿或重新计算声明哈希替代独立审稿。这是辅助神经审查，不是人工评级或棋力评测。

## 生成新版本

```bash
python scripts/freeze_run.py --output runs/reference-correction/source-run -- \
  python -m xqgeneral.reference_prose \
  --data data/astra-explanations-clean-v1 \
  --audit data/legacy-heldout-reference-reviewed-v1 \
  --output data/astra-explanations-reference-corrected-new --workers 8
```

入口要求审查恰好覆盖源数据中全部缺少内联复核记录的验证／独立测试项，不能混入训练项。原始合并数据必须已有完整历史、所有变化和跨分割隔离的完成证明。

命令使用一个新的输出目录。本服务器已完成的版本是 `data/astra-explanations-reference-corrected-v3`，不能覆盖它或原始标签。

新版本保留训练文件的全部原字节，未修订参考记录也保持原字节。修改项只更换 `answer.explanation` 并增加复核元数据；问题、推荐走法、评分、候选变化、历史、特征键、棋局归属和分割均不变。原教师字段保留初始来源，新增 `prose_review.independent_teacher_revision` 记录这次独立重写；颜色派生的局面仍不增加独立局面容量。

所有修改项再次检查完整原生历史和变化。未修改项复用同字节源数据的原生证明。新清单为 `heldout_reference_corrected_explanation_dataset`，明确报告训练字节保留、参考正文修改数及新接受；不能声称全部来源标签字节保持不变。

## 后续训练

`xqgeneral.explanation_preflight` 接受完成的新清单，继续核验全部特征身份，只对训练／验证做词元检查，独立测试不做训练词元索引。新数据需要新的索引、冻结配置和输出目录；已启动的实验及其等待脚本不能热改。

课程数据和专家特征不因正文修订改变。讲解仍继承实际完成的四门课程，完整解码器的真实多卡更新、恢复和显存预检通过后，才启动正式 SFT。参考修订、原生核验或预检通过都不证明学生棋力或讲解收益。

## 已完成的数据版本

GPT-6-Astra Low 独立审查全部 448 条旧验证／独立测试参考：284 条原文接受，164 条拒收后由另一作者实际重写，再由独立审阅者接受。改动为 73 条验证正文和 91 条独立测试正文，其中 148 条属于已有颜色派生局面，16 条属于原始局面；没有新增独立局面容量。

新版本仍为训练／验证／独立测试 10758／384／512 条。完整 11654 条读回确认问题、结构化答案、全部历史、特征键和棋局分割相同；训练文件逐字节相同，未修订参考行保留原字节。164 条新正文及其实际作者、拒收链、独立接受和原生复验均已绑定。

实际固定词元器编码了全部 11142 条训练／验证记录，最长为 649／625 tokens，上限 1024，全部答案完整；没有编码独立测试项。它没有核验尚未完成的新课程特征缓存，四课之后仍须执行完整词元／特征预检。全套 1177 项检查通过，163 份源码／测试不变。实际新等待队列及来源边界见[后续流程](PAPER_POST_CURRICULUM.md)，读回摘要见[轻量证据](../evidence/heldout-reference-prose-v1.json)。
