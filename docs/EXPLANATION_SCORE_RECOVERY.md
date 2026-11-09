# 讲解评分缺失与原始回答恢复

引擎在固定节点预算耗尽时，可能返回合法的 `bestmove`，但只报告该走法的分数上下界。上下界不能当作精确评分。原解析器因此拒收该次分析；旧批量评分流程会随单题异常退出，即使模型已经完整生成并保存了全部回答。

现在这类情况用 `UnscoredBestMove` 明确表示。讲解评分保留实际引擎输出、请求节点数、完整历史、查询阶段及变化中的位置，继续检查原始走法的合法性。模型的非法走法、格式错误和错误讲解仍然保留；未测量的评分单独记录，不算作模型错误。其他引擎进程错误仍会使该次执行失败。

没有精确评分时，不补造分数、不把边界值当作精确值，也不增加节点预算或反复重试。已有根分析的失败与成功一样复用。`pv_move_losses` 只含实际测得的损失，未评分位置在 `engine_judgments` 中标为 `score_unavailable`。

发生评分缺失时，讲解汇总的 `oracle_scoring_coverage` 给出未评分问题数、引擎请求数、合法首步数与合法变化半回合数。`first_move_no_mistake_rate` 保持全部回答作为分母，未测量首步不贡献成功，因此会显式标为下界；它不是确认的模型错误率。走法和规划汇总也会标注对应的未评分数量及下界。神经讲解裁判收到未测量的范围和说明，不能从缺少评分推断原回答有错。

## 从已保存回答评分

`xqgeneral.score_saved_explanations` 只执行 CPU 评分，不加载模型、教师或特征张量。它要求一份明确标记为 `preserved_raw_explanation_generation`、状态为 `complete_raw_answers_only` 的 JSON 收据，包含：

- `generation_config`：原始模型、数据和特征路径，验证分割、样本数、随机种子、记忆模式、批量、词元预算，以及原评分引擎、权重、节点数和工作线程数。
- `raw_predictions`、`validation`：路径、SHA256 与字节数。
- `origin_artifacts`：原始执行档案的路径及哈希、字节数；`generation_code` 保留原生成源码身份。
- `engine_artifacts`：本次 CPU 评分使用的原引擎与权重文件的哈希、字节数。

```bash
CUDA_VISIBLE_DEVICES= python -m xqgeneral.score_saved_explanations \
  --receipt /path/to/raw-generation-receipt.json \
  --output /path/to/fresh-scoring-output
```

入口重新构造全部验证问题及原始顺序，逐项核对 ID、问题和记录，拒收缺失、重复或换题。原始答案按字节复制，所有错误回答仍进入分母。每个 CPU 工作线程保存自己的评分行；若执行异常，保留这些行和原始回答，修复后使用新的输出目录重新评分。已有输出目录不会被覆盖。

输出类型是 `recovered_independent_explanation_evaluation`，明确区分原始生成源码与恢复评分源码；它不伪装成旧失败运行的完成清单，也不证明上游模型、特征或训练已经重新执行。评分阶段不会重新读取并哈希生成模型和特征张量，因此该边界在指标中保留。最终棋力、可靠教学能力和独立测试仍需各自评测。
