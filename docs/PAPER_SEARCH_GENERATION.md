# 搜索生成与论文递归规则

固定作者版本 `c372da22ca75e95c3c79bee8a83c220fc54d0dcb` 的 [mine.yaml](https://github.com/queen-project/queen/blob/c372da22ca75e95c3c79bee8a83c220fc54d0dcb/configs/self_distill/iteration/mine.yaml) 声明学生温度 0.6、top-k 20、top-p 0.95，最多 4096 个输出 token。[本地生成配置](../configs/paper-search-generation-v1.json)沿用这些数值，显式启用采样。

## 入口与来源

`Predictor.generate`／`generate_batch` 可显式接收 `sampling` 和 `seed`。没有指定采样时保留原贪心行为；原始非法文本继续照实记录，采样入口不限制为引擎走法或代写正文。[Transformers 生成参数](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)决定词元采样；棋力专家特征仍由原推理入口提供。

搜索 CLI 可使用新配置：

```bash
.venv/bin/python scripts/freeze_run.py --output runs/sampled-search/source-run -- \
  .venv/bin/python -m xqgeneral.search_distillation \
  --checkpoint "$XQ_COMPLETED_SFT_CHECKPOINT" \
  --data "$XQ_COMPLETED_COURSE_DATA" --features "$XQ_VERIFIED_FEATURE_STORE" \
  --recorded-inputs "$XQ_FRESH_ISOLATED_SEARCH_INPUTS" --seed 20260828 \
  --student-generation-config configs/paper-search-generation-v1.json --max-root-retries 3 \
  --root-candidate-policy paper_fill --reuse-child-analysis \
  --acceptance-policy paper_recursive --min-clean-ply 1 --child-contract move_eval \
  --output runs/sampled-search/mining
```

路径须指向实际完成的四课、讲解 SFT 及重新隔离的搜索来源。[递归接受模板](../configs/search-paper-recursive-pilot-v1.json)通过现有 `search_pilot` 串联搜索、教师汇总和收取；它等待对应讲解能力验证，数据／缓存／父权重及实际队列路径需要统一。原[候选／复用模板](../configs/search-paper-candidates-pilot-v1.json)、[根重试模板](../configs/search-paper-retry-pilot-v1.json)与[采样模板](../configs/search-paper-sampled-pilot-v1.json)保留。模板仍只有 512 根，汇总输出预算仍是 1536，不能把它当成正式论文整轮。

配置文件原字节参与搜索与试批合同。搜索输入包括特征清单、完整历史键目录和全部存储分片；配置、分片或执行来源变动不能续写原实验。新采样实验还绑定实际 Torch／Transformers 版本。已有冻结课程和等待入口不随新配置修改。

## 单根续跑

本地按基础种子、原根 ID、完整历史键和根内生成序号派生 63 位种子，逐次保留设置、序号及十进制种子文本。每个根重新从零编号，跳过其他已完成根不消耗该根的随机流；根与子节点排序后仍携带原生成序号。该种子方案是本地续跑约定，不是作者 vLLM 随机流的逐位重放。

指定种子时只暂存 CPU 和实际推理 CUDA 设备的随机状态，结束或抛错后恢复；不调用对所有 GPU 重设种子的接口。随机作用域依据 [PyTorch fork_rng](https://docs.pytorch.org/docs/stable/random.html)。当前是单进程串行搜索；相同权重、完整历史和调用次序下的采样复现，需要与实际学生和引擎另作控制。

## 根评估重试

作者 [recursion.py](https://github.com/queen-project/queen/blob/c372da22ca75e95c3c79bee8a83c220fc54d0dcb/datagen/self_distill/recursion.py#L217) 只在数值评估无法解析时重新生成根分析，首次生成后最多再试三次。新 `--max-root-retries 3` 沿用这个次数与条件；本地结构化答案须有合法的 `side_to_move` 整数 cp／mate 评估并能完成概率换算。显式零预算只检查一次；省略参数保留旧实验合同，正预算要求采样。

每次生成的原文、完整结构检查、独立数值接受检查和生成序号全部写入 `root_generation_attempts`。第四次仍无可用数值评估则拒收，未进入引擎搜索。合法数值评估配上非法候选时仍记录候选错误；子节点错误按原递归逻辑处理。这些错误不会被数值重试条件混淆。改变重试预算会在加载学生之前拒绝续跑。

## 候选补齐与子分析复用

作者 [候选准备](https://github.com/queen-project/queen/blob/c372da22ca75e95c3c79bee8a83c220fc54d0dcb/datagen/self_distill/recursion.py#L247)保留至多三个不同的合法模型候选，再用引擎 MultiPV 补至 `min(3, 合法走法数)`。显式 `paper_fill` 按本地结构化答案的首选、候选列表、原变化顺序筛选；补入的走法、最佳走法注入和按引擎质量排序分别记录来源。生成原文、原候选列表与结构错误不变，原始模型评测仍直接评分学生输出。缺少足够的实际引擎候选时直接报错，不生成虚构走法或评分。

`--reuse-child-analysis` 对照作者[缓存子分析](https://github.com/queen-project/queen/blob/c372da22ca75e95c3c79bee8a83c220fc54d0dcb/datagen/self_distill/recursion.py#L660)：递归到该子节点时，先核对 ID、起点、全部走子、历史、最终 FEN 和特征键。数值可解析的原子分析作为第零次根尝试复用，不消耗新采样序号；原子节点失败与结构检查保留。缓存数值不可用时也保留第零次失败，随后按首次加三次重试重新生成。同一 FEN 的另一条历史不能复用。

两项选项都要求显式数值根检查预算，并进入试批配置与续跑合同；改变设置会在加载学生之前拒收。省略选项保留旧实验行为。当前课程与等待队列仍使用其原冻结源码。

真实控制中还修正了 Pikafish 的中断 MultiPV 解析：部分迭代中两个名次指向同一走法时，按已观察的精确评分补回不同候选；上下界结果不当作精确评分，原日志和各候选实际深度保留。失败控制及其冻结源码保留，原生同一条日志可复现旧解析器只有两个候选、修正后得到三个。

## 变化核验与递归选择

显式 `paper_recursive` 对照作者 [子变化核验](https://github.com/queen-project/queen/blob/c372da22ca75e95c3c79bee8a83c220fc54d0dcb/datagen/self_distill/recursion.py#L327)、[变化比较](https://github.com/queen-project/queen/blob/c372da22ca75e95c3c79bee8a83c220fc54d0dcb/datagen/self_distill/recursion.py#L463)及[递归决策](https://github.com/queen-project/queen/blob/c372da22ca75e95c3c79bee8a83c220fc54d0dcb/datagen/self_distill/recursion.py#L566)。每个子节点只生成一次，独立解析数值评估、复演主要变化，并对所有合法步实际查询引擎。首个非法走法或胜率下降达到 0.10 的走法之后的后缀不进入汇总；至少保留一个核验正确的子走法，真正终局可以保留零步。完整原文与结构失败仍留在轨迹里，未选分支的全部可达历史也先检查保留集隔离。

模型从原生质量排序的子节点中推断最优变化。错误排序导致走法质量下降至少 0.05 时，递归到数值误差最大的子节点；选中子评估距真实根达到 0.10 时，递归到原生最优子节点。数值缺失、无可用正确前缀和变化没有改进也各自选择对应子节点递归，原子分析可按完整历史复用。

新旧变化在首个分歧处比较：合法变化修复非法变化可以接受，两条合法变化用相同完整历史和节点预算的受限引擎查询比较评分。变化一致或只相差合法前缀时，要求新数值评估严格更接近真实根。被截掉的首个坏走法保留为比较边界，不能用截断掩盖新变化的错误。汇总候选按推断排序，使建议走法与候选首位一致；原生排序另留在轨迹里。

本地使用中国象棋完整历史、结构化 JSON 和 Pikafish 的材料条件 WDL；不是作者的国际象棋文本解析或 Stockfish 校准。最终 JSON 变化最多 32 步，超长目标直接拒收，不静默裁剪。该策略要求显式候选补齐、数值根检查与 `move_eval` 子节点合同；省略选项仍运行旧严格策略。策略或最短正确前缀设置变更会在续跑加载学生之前拒收。

## 已完成的控制

固定 90 份执行模块，以随机初始化的小型 GPT-2 在 CPU 上实际调用 `Predictor`；没有加载预训练学生、专家或汇总教师。64 个种子的八 token 结果均与独立的完整前缀推理、温度／top-k／top-p 筛选和显式随机生成器逐字一致。32 个根的连续处理、跳过已完成前 11 根及倒序处理结果一致，全部原始非法输出保持拒收。另一实例实际生成了 4096 个输出 token，未截为旧 768 预算。

两个实际 CUDA 设备只运行随机张量控制：选定设备的结果与独立生成器一致，正常返回及异常后 CPU／两卡随机状态和当前设备均恢复；没有在 GPU 加载神经模型。特征分片篡改的续跑测试在加载学生之前拒收。这些控制证明本地采样与输入合同，尚未产生蒸馏标签或证明学生实力。机器可读结果见 [轻量证据](../evidence/paper-search-generation-v1.json)。

根重试另用冻结的 90 份模块执行：随机 CPU 神经解码的 32 根共 128 次短输出均与独立采样逐字一致，前 11 根完成后的续跑与倒序结果相同；另一实例实际执行四次各 4096 token，共 16384 token，保留原文并以数值失败拒收，没有调用引擎、预训练学生或汇总教师。受控结构化输出还覆盖第四次成功、预算耗尽、数值溢出、候选与子节点错误及预算变更拒收。见 [重试证据](../evidence/paper-search-root-retry-v1.json)。

候选与复用控制使用 32 盘不同真实棋谱的训练侧历史，实际执行 64 次完整历史 Pikafish 查询，每次 10 万节点，补入 62 个实际 MultiPV 候选，32 个数值子分析成功复用。独立读回重建全部 64 条历史、核对原生精确评分与保留原文，并重新绑定十份输入、三份输出与 90 份源码。模型候选错误和递归触发是构造夹具，子分析取自实际引擎；没有加载预训练学生或讲解教师，也没有产生蒸馏标签。受控多层递归另覆盖数值缓存失败、新生成、不同历史拒收、保留集隔离和设置变更拒收。见 [候选与复用证据](../evidence/paper-search-candidates-reuse-v1.json)。

新策略以作者未修改的函数 AST 做了 256 组决策、256 组前缀截断及 256 组比较边界对照，全部一致；另六组首处分歧对照使用中国象棋棋盘适配器，不宣称复现原国际象棋规则引擎。32 盘真实棋谱训练侧控制实际查询 Pikafish 154 次，保留 144 条构造生成原文；8 次递归决策中实际复用 6 条子分析。30 条控制查询通过汇总条件、2 条达到控制递归上限，没有调用学生或讲解教师。这些是引擎构造夹具，不是学生产出率或训练标签。独立读回 154 条完整查询历史、449 份原生精确评分、全部原文及正确前缀；全套 1272 项通过，169 份源码／测试不变。见 [递归接受证据](../evidence/paper-search-recursive-acceptance-v1.json)。

## 后续验收

工作批量／并发、汇总上下文 16384 与输出 8192，以及正式合格标签产出仍待执行。先完成新四课、讲解 SFT、能力与对弈基线，再测合格产出并扩大搜索、训练实际蒸馏轮；最终棋力与教学验收仍使用独立保留集。
