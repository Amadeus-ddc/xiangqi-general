# 讲解、旧课和引擎监督的组合语料

本入口组合七类监督：中文讲解、四门原课程回放、`move_quality` 单走法，以及
`engine_numeric` 的走法、六步变化、候选和评分。后两类是原始引擎查询的不同题目视图，
不算新增中文讲解，也不算新的学生对弈或一次已训练的论文蒸馏轮次。

## 输入与隔离

输入必须带有已完成的原生证明和精确文件绑定：引擎复查后的初始讲解、严格搜索讲解
汇总、从原课程训练分割逐字节抽取的回放、[原始查询监督](ORACLE_POLICY_DATA.md)，
以及原讲解验证局面的独立引擎参考。

初始讲解的验证行和课程验证行保持原始字节。相同 ID 只有来自不同文件且原始行完全
相同时才合并；同文件重复、冲突内容及跨分割 ID 都拒绝。训练逐行记录来源，拒收行
和重复行分别保存，原始来源总数必须等于保留、拒收和重复三者之和。

新增走法和数值验证题沿用原讲解验证历史；它们增加题型，不增加独立验证局面。根查询
参考保留原始引擎文本、固定预算及资产身份。所有已保存候选变化和颜色对应局面都加入
训练隔离范围。新增验证题的目标若进入其他保留位置则单独拒收，原有验证题不按表现筛选。
独立测试文件仅参与身份绑定和既有位置隔离，测试答案不解析、不编码，也不写入组合语料。

来源已完成的完整历史、原生答案及分割证明被明确复用；入口重新检查行身份、实际监督
后续和新增隔离。引擎监督还按原查询的文件、字节位置及行哈希读回全部保存候选变化。
它不重复执行数十万次完整历史裁定，也不生成新的引擎或语言模型回答。

## 命令

先准备原讲解验证局面的参考。兼容的历史查询可以复用；解析器改变导致解释不同的查询
会记录拒收原因并重新搜索，使用相同解析器却与原文矛盾的保存结果直接报错。

```bash
python -m xqgeneral.oracle_policy_reference \
  --data data/astra-explanations-engine-qualified-v4 \
  --reuse-reference runs/clean-teacher-label-engine-reference-v1 \
  --executable vendor/pikafish/src/pikafish --weights vendor/pikafish/src/pikafish.nnue \
  --nodes 1000000 --workers 4 --output runs/oracle-policy-validation-reference-v1/references
```

随后组合已完成来源，长任务通过 tmux 或调度器执行，并从冻结源码启动：

```bash
python -m xqgeneral.oracle_policy_corpus \
  --explanations data/astra-explanations-engine-qualified-v4 runs/strict-search-available-merge-v1/dataset \
  --replay data/course-replay-balanced256-v1 data/course-replay-balanced2048-v1 \
  --policies runs/oracle-policy-views-production-v1/dataset \
  --references runs/oracle-policy-validation-reference-v1/references \
  --workers 8 --output data/oracle-policy-corpus-v1
```

同一命令加 `--resume` 只接受已完成结果：核验原输入、源码和输出后，在临时目录完整重建
并比较所有语料、来源、拒收、重复及计数文件，原产物的字节和时间不变。未完成目录保留，
改用新输出目录。参考查询的完成重读同样不新增搜索。

## 当前已测边界

2026-11-07，原始查询监督已实际完成生产及完整重读：148157 个历史上下文、293089 条题目，
其中走法 148157、数值 144932。这些上下文来自 4381 个声明来源棋局标识，143609 个是
搜索后续；不能把问题行数或历史数当作独立真人棋局数。全部题目已有原生词元长度检查
和专家特征。384 个验证根参考复用了 371 次查询，另补 13 次真实百万节点搜索。

旧课扩充池已从原始 10804561 行训练文件重新流式抽取并哈希，26 类题各 2048 行，合计
53248 条训练、2600 条原始验证；原生答案、完整历史与后续及无截断编码检查通过。
组合后的最终行数以本入口实际输出为准。新语料尚未证明学生棋力或讲解收益；正式混合
比例、训练预算、GPU 预检、实际优化更新和独立验收仍须分别记录。
