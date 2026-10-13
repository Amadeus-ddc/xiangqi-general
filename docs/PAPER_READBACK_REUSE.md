# 复用已完成的全量问答复核

`foundation_readback` 可以显式接收同一题库的已完成 `paper_curriculum --readback` 清单。它重新哈希该清单绑定的全部来源、三份题库和验证产物，确认与原数据生产者的源码身份相同后，沿用已经完成的逐条原生重建结果，避免再次生成同一整库答案。

没有提供此选项时仍执行完整重建。提供的证明若未完成、属于另一题库、缺失输入、统计不一致或任何文件字节改变，直接拒收；重新哈希后的错误标签集合也不能借用此前的完成证明。

## 使用

先完成本次题库的全量金标复核、专家缓存和训练预检。下列路径替换为本次准备的配方和完成清单：

```bash
python -m xqgeneral.foundation_readback \
  --config runs/preparation/foundation.config.json \
  --preflight runs/preparation/preflight \
  --base-proof runs/qwen-base-identity/manifest.json \
  --completed-question-readback runs/preparation/question-readback/manifest.json \
  --output runs/source-cache-readback-v1 \
  --public-evidence runs/source-cache-readback-v1/summary.json
```

输出目录必须是新的。只有论文题库合同支持复用；旧实战前缀合同继续使用原来的来源重建。

## 验证边界

完成证明必须绑定当前题库清单、全部生产输入与三份问答文件，且验证文件和清单中的全量统计相同。原生完整检查、棋局／根／未来隔离、累计答案频率、位置预算、难例来源混合及容量不足记录按原声明核对。来源与题库同一生产者版本要求明确记录；新的读回程序可以与该生产者版本不同。

全部问答历史与缓存身份、预检结果、官方语言基座资产以及旧特征前缀的逐位比较继续执行。已验证的源文件签名会检查到新完成清单发布前，所有继承的输入与输出文件也写入新清单。训练预检仍检查所有26类题型，复用原生复核不替代题型覆盖或训练验收。

报告中的 `paper_questions_regenerated` 表示完成证明覆盖的总题数。新增 `paper_question_readback_reused` 和 `paper_question_regeneration_executed_this_readback` 明确区分本次重建与复用；复用时记录原清单哈希及全部绑定已重新哈希的事实。

源码／标签改动、缺失绑定、错误题库、部分检查、生产版本不匹配和复用后文件改变均有拒收测试。目录、预算、频率及实际短缺记录另有原生控制。完整命令对照与实测边界见[轻量证据](../evidence/paper-readback-reuse-v1.json)；小规模控制不能证明全库提速倍数、正式训练完成或学生棋力。
