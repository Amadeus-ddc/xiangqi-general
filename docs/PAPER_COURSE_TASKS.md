# 论文课程题型的象棋实现

`paper_xiangqi_v1` 显式声明七类静态题和六类动态题，当前／未来两组共 26 类。正在运行的 clean-v3 仍使用原来的 22 类题及冻结源码；这份实现没有修改它的配方、输入、检查点或等待队列。训练比例与其余论文差异仍见 [训练对照](PAPER_REPRODUCTION.md)。

## 题型与原生语义

核对作者固定 revision `c372da22ca75e95c3c79bee8a83c220fc54d0dcb` 的十三个题型模块、局面特征及第二课配置，共十五份来源。下面是中国象棋的对应合同，英文名称保留来源中的任务名称。

| 作者题型 | 本地 `task_type` | 中国象棋答案 |
|---|---|---|
| `piece_on_square` | `piece` | 指定格的棋子或空 |
| `square_of_piece` | `locate` | 某种棋子的全部坐标 |
| `piece_on_file` | `file` | 一列的各类型数量 |
| `piece_on_rank` | `rank` | 一行的各类型数量 |
| `piece_on_diagonal` | `diagonal` | 长度至少二的完整对角线上的各类型数量 |
| `piece_count` | `counts` | 红黑双方各类型数量 |
| `material_count` | `materials` | 红、黑分别计子力总值 |
| `piece_moves` | `moves` | 某枚行棋方棋子的所有合法着法，可为空 |
| `square_attackers` | `controllers` | 攻击／保护指定格的棋子和坐标 |
| `capture_moves` | `captures` | 全部合法吃子着法 |
| `checking_moves` | `checks` | 全部合法将军着法 |
| `check_parries` | `parries` | 正在被将军时的全部合法解将 |
| `is_checkmate` | `mate` | 行棋方是否已经被将死 |

子力按车 9、马炮 4、仕士相象 2、兵卒 1、帅将 0 计算。9×10 棋盘有 32 条长度至少二的完整对角线，按长度采样。行／列／对角线答案是棋子类型计数；不沿用旧 `rank` 题的坐标清单语义。

攻击／保护使用几何控制，受牵制棋子也计入。炮必须有一个炮架，马腿、象眼、过河兵、九宫及阻挡均参与计算。占用格的同色控制者是保护者，异色为攻击者；空格的双方控制者均列为攻击者。该关系对齐 Pikafish 的 `Position::attackers_to`，飞将检查由将军／合法走法接口处理。

`mate` 判断当前是否正在被将军且没有合法解将，并不要求模型寻找一步杀。象棋困毙也判负，但未被将军的困毙在此判断题中回答“否”，依据正文明确说明判负原因。`parries` 只从被将军的目标局面生成，将死时答案为空。

未来题使用一至八步实际合法变化。`moves` 追踪最终行棋方的存活棋子，以变化前的坐标识别它；不会把变化后的坐标或最终占用者泄漏到问题中。终局棋盘可以作为题目，实际变化可以最后一步到达将死／困毙；禁止越过终局，也拒绝仍有合法着法的 AXF 历史终止局面。

## 生成与隔离

生成器读取已完成生产清单绑定的完整根历史。源棋局必须已经分割，生成题目不会改分割。保留真实来源和未知过往标记；颜色镜像沿用同一棋局身份。检查新根、最大实际未来及颜色镜像与全部旧课程分割的归属，发现跨分割重叠即拒绝。

新配置必须在配方顶层和 `raw_qa_gates` 中同时声明 `"task_profile": "paper_xiangqi_v1"`。省略字段保持旧合同；续训、课程交接和讲解训练拒绝偷偷切换题型。完整预检要求新生产者、旧数据隔离证明、训练题型覆盖、验证题型样本量、特征缓存和词元检查。

长生成任务使用新目录，在 tmux 中运行并冻结源码。例如：

```bash
python scripts/freeze_run.py --output runs/paper-data-v1/source-run -- \
  python -m xqgeneral.paper_curriculum \
    --roots PREPARED_ROOTS.jsonl --source-manifests COMPLETED_SOURCE_MANIFEST.json \
    --reference-data data/research-human-engine-v1 \
    --sampling-profile author_answer_frequency_xiangqi_v1 \
    --position-sampling-profile author_task_position_budget_mix_xiangqi_v1 \
    --position-workers 8 \
    --output data/paper-courses-v1

python -m xqgeneral.paper_curriculum --readback data/paper-courses-v1 \
  --output runs/paper-data-v1/full-source-readback
```

读回重新核对来源／产物字节，从全部源根逐条重建金标、问法和镜像，检查输出顺序及完整未来隔离。基础课程的 `foundation_preflight`、`foundation_readback` 和 `foundation_handoff` 已支持该合同。实际缓存扩展、全量词元预检与训练必须另外完成；小规模规则检查不能替代它们。

## 原始答案评分

新金标给出原生事实依据，最后一行以 `答案：` 标记结构化内容。评测保留整个模型原文，解析最后的标签并比较坐标／着法集合、计数或判断；顺序变化允许，重复、遗漏、非法坐标、缺标签和多标签保持错误。不会调用引擎替学生补答案或修走法。

同时保存内容正确率、格式有效率、规范标签相同率及整段原文相同率。正文没有通过这份标签评分而获得事实／战略正确性，`prose_graded` 始终为 `false`；可靠教学仍需要独立讲解评测。旧题仍按原来的整段标准化文本相等评分。

## 累计答案频率与多问题

显式选择 `author_answer_frequency_xiangqi_v1` 后，频率计数按分割、当前／未来课程和题型独立维护。格子题同时按实际答案棋子类型、查询坐标及局面内类型数量降权；未来格子题先按累计频率选择改变／未改变组。定位题按答案坐标的累计频率加权，并把缺席棋子的空答案作为同一类。走法题按行棋方存活棋子类型平衡，攻击／保护题保持约一半空格、一半按占用类型平衡。行、列、对角线、计数及其他动态题不读取这些频率。

训练集每根的请求数照作者四份实际 YAML：格子四题、定位／行／列各两题、对角线三题、走法及攻击／保护各四题，其余各一题；验证和测试每类一题。每根同类问题不重复查询实体，并按实际可用实体数截断，终局仍可查询不能移动的将帅。计数在该根的不同问题及颜色镜像全部写入后更新，镜像根据实际最终棋盘重新计算答案类别。完整读回重建每条记录以及全部计数，拒绝被改动的采样配置或频率摘要。

未传该参数的已完成第一版题型数据保持原来的单题、局面内平衡合同。累计频率选项本身只控制答案分布与每根问题数；每题型根数量和难例混合使用下面的独立选项。

## 每题型根预算与难例混合

同时声明 `author_task_position_budget_mix_xiangqi_v1` 和累计答案频率选项后，按作者四份固定 YAML 的每题型根预算选择局面，然后只生成该题型的请求问题。当前／未来课程各自使用同一份根预算；每类验证请求 100 根、测试请求 1000 根。

| 题型 | 每门相应训练课的请求根数 |
|---|---:|
| `piece` | 457143 |
| `locate`、`file`、`rank` | 各 914286 |
| `diagonal` | 609524 |
| `counts`、`materials` | 各 1828572 |
| `moves`、`controllers` | 各 457143 |
| `captures`、`checks`、`parries` | 各 1828572 |
| `mate` | 360000 |

将军题混合普通来源 20% 与存在合法将军着法的来源 80%；解将题全部来自被将军的目标；将死判断混合已将死、记录中近将死且被将军、普通被将军、困毙和普通来源，比例为 30%／20%／10%／5%／35%。类别由原生规则和明确的真实／生成终局依据重新计算。普通来源也可能有将军等性质，来源比例不等于正例答案比例。

未来课为每根选择一个符合来源类别的合法目标步数，随后保持该步数，不再随机裁到另一类目标。同一给定局面的不同合法一步终局，由保留的父根与后继完整历史重新绑定，可供未来将死／困毙来源选择；生成变化仍沿用原局面身份和未知前史标记。

来源按作者的平滑加权顺序交织，耗尽即移除，余下来源按相对权重继续；不循环或有放回填满预算。普通单来源先按有限文件顺序取根，再打乱选中根的顺序，之后才生成受累计答案频率影响的问题。混合来源保持交织顺序。报告每类请求／实际根数、缺口、实际来源数量、去重及耗尽来源；配置比例被应用不意味着耗尽后仍能满足原比例。

每门课每个分割的不同题型不重复使用同一查询棋盘及其颜色对应；作者按完整查询 FEN 去重，本地按棋盘加行棋方去重，时钟差异不增加独立容量。完整源未来、全部分支、终局依据与颜色范围在选样前参与旧分割隔离。来源 JSON 存在临时 SQLite 目录，原生类别可按有界并行工人计算，单／多工人输出一致，结束删除临时目录。读回重新建库、选根、生成全部问题并核对精确顺序及报告；共享采样和原生规则实现，不宣称独立重写。

可用 `--task-position-budgets BUDGETS.json` 显式适配实际容量。文件必须完整声明三个分割、四门课及其全部题型的正整数根数；省略则请求作者预算。实际容量不足保持缺口，正式训练仍需要完整题型覆盖、验证容量、缓存和词元预检。独立测试不参与训练或选模。

## 真实终局与近将死来源

`recorded_course_pools` 从已完成原生导入的完整棋谱恢复实际将死／困毙终局，以及此前至多十四个半回合的真实历史根。既有 `recorded_curriculum` 为旧合法走法题裁掉了终局，不能直接用它的根池统计将死容量。新入口读取旧根清单中预先确定的棋局分割，源档案的默认分割不能覆盖它；未进入该清单的棋局不自动加入。

每根最多提供八步实际未来，并逐步记录对应目标的 `generic`、`have_check`、`in_check`、`mate`、`stalemate` 等类别。`near_mate_check` 要求该目标正在被将军，且原棋谱确实在接下来一至六个半回合内到达将死；保留真实主线作为依据。这表示记录中距将死的距离，不能据此断言最优应对下必然被将死。将死与困毙都需要原生确认没有合法着法；仍有合法棋盘着法的历史重复等终止另行计数，不充作终局难例。

完整历史、每个前缀、最大未来和将死依据均从原谱复演。最大未来、整个将死依据和颜色对应共同参与旧分割隔离；结构化旧讲解的主变化及所有分支也可显式保留。旧根重复和跨分割候选记入排除清单，不挪动棋局分割。完整读回重新读取所有输入／产物、重建每个根、目标类别与排除理由，并比较精确顺序；选样实现和原生规则共享，不宣称独立重写。目标出现次数不等于独立局面或完整棋局数。

```bash
python scripts/freeze_run.py --output runs/terminal-pools-v1/source-run -- \
  python -m xqgeneral.recorded_course_pools \
    --games IMPORTED_GAMES.jsonl --game-manifests COMPLETED_GAME_MANIFEST.json \
    --owners PREASSIGNED_RECORDED_ROOTS.jsonl --owner-manifest COMPLETED_ROOT_MANIFEST.json \
    --reference-data PRIOR_COURSE_DATA \
    --explanation-reference-data PRIOR_STRUCTURED_LABEL_DATA \
    --workers 8 --output data/recorded-terminal-pools-v1

python -m xqgeneral.recorded_course_pools --readback data/recorded-terminal-pools-v1 \
  --workers 8 --output runs/terminal-pools-v1/full-source-readback
```

该数据池仍是原生来源准备。按题型分配根预算、锁定所选未来目标后兑现难例比例、补齐来源容量、生成正式课程题和扩展缓存必须随后完成；原棋谱走法不作为最优走法标签，也没有生成神经讲解。

[来源容量与 CPU 合同证据](../evidence/paper-course-source-pools-v1.json) 完整核查原有 45834 个记录根及其未来、22720 局实战档案，以及 2259 条战术主线和 9683 个独立战术局面。实战档案有 257 个将死／3 个困毙，已分割旧根对应的棋局仅提供 83／1；战术主线另有 238／1，独立战术局面没有终局。不同来源尚未共同隔离，不能直接相加为独立容量；困毙来源尤其不足。24 项新增案例和标准 883 项 CPU 检查通过；正式新根池生产／完整读回仍需另行完成。

## 战术主线与合法终局延伸

`tactical_course_pools` 接收完成原生导入、按固定种子预分割的战术主线和独立战术局面。主线终局沿用真实记录的走子和将死依据，并保留给定起点、棋手缺失和片段前历史是否可用。独立局面枚举所有合法一步着法，只有实际到达将死／困毙且原生胜方一致时才收取；不读取源解答来制造标签。

生成延伸沿用原局面身份和分割，声明 `extension_is_recorded_source_move=false`，使用 `paper_generated_terminal_witness` 和 `plies_before_generated_mate`，不冒充 `paper_recorded_mate_witness`。同一父局面的多种终局只保留一个重复历史根，其他终局后继可分别收取；重复父根和隔离冲突逐项记录。每个生成终局的给定父局面也参与颜色／分割保留。棋谱中的距将死步数和生成主线的距离均不证明最优防守下强制将死。

已完成的 `recorded_coach_footprints` 可以复用：重新绑定全部原始课程／讲解文件及清单字节，核对组成集合、合并集合和计数，再显式补全颜色对应。原缓存的原生未来解析不在这里重跑。额外旧课程和结构化讲解照常复演未来；所有声明的原生来源档案完整历史和颜色也参与保留，沿用已有棋局归属，否则使用既定分割种子。不同分割共享的原谱位置标为歧义并排除新候选，不擅自选择归属。完整读回重新导出全部原生战术目标、来源字段、顺序、尾部和排除原因。

全部 9683 个给定局面的原生一步枚举发现 238 个将死和 51 个困毙后继；这些是隔离前观察，多个后继可能来自同一个局面，不能相加为新增独立棋局。已有战术主线的 238 个实际将死与一个困毙也须共同隔离后才能计入新池。正式生产／读回的完成状态见 [状态](../STATUS.md)。

```bash
python scripts/freeze_run.py --output runs/tactical-terminal-pools-v1/source-run -- \
  python -m xqgeneral.tactical_course_pools \
    --games data/recorded-tactical-lines-ccpd-v1 data/recorded-tactical-positions-pwa-v1 \
    --footprints COMPLETED_NATIVE_FORECAST_MANIFEST.json \
    --reference-data OTHER_PRIOR_COURSE_DATA \
    --explanation-reference-data OTHER_PRIOR_STRUCTURED_LABEL_DATA \
    --reservation-games ALL_COMPLETED_CANONICAL_IMPORT_DIRECTORIES \
    --owners PREASSIGNED_RECORDED_ROOTS.jsonl --owner-manifest COMPLETED_ROOT_MANIFEST.json \
    --workers 8 --output data/tactical-terminal-pools-v1
python -m xqgeneral.tactical_course_pools --readback data/tactical-terminal-pools-v1 \
  --workers 8 --output runs/tactical-terminal-pools-v1/full-source-readback
```

这一步提供有来源的终局根；按题型预算和所选未来步数与难例类别绑定由上面的课程选样入口完成。正式数据和特征缓存须另行验收。

## 验证与剩余差异

[原生控制对照](../evidence/paper-course-controllers-native-v1.json) 检查 773 个局面的全部格子，共 69570 次比较，零差异；[八个控制局面](../evidence/paper-course-controller-fixtures-v1.json) 可在无模型、无 vendor 的 CPU CI 中重算。运行入口为 `scripts/build_native_controllers.py` 与 `scripts/verify_native_controllers.py`，编译使用固定、未修改的 Pikafish，运行不加载权重。失败的构造局面检查保留在忽略的运行目录中。

题型语义、生成／隔离／读回、原始评测、累计答案频率、多问题以及每题型根预算／难例来源混合已实现，见 [采样合同证据](../evidence/paper-task-position-sampler-v1.json)。混合比例来自实际 YAML，原题型文件的旧注释有不同数字。CPU 合同和构造原生局面控制不等于已制作完整正式数据集；来源扩容、实际容量和分布仍需验收。

正式大数据、训练中的有限数据遍历、按词元的训练损失、新课程的验证选模策略，以及全宽桥接的完整四课训练仍需完成。作者额外的静态全实体测试模式尚未加入本采样合同。没有用这份新合同训练学生，也没有证明棋力或讲解收益。

来源：[作者题型源码](https://github.com/queen-project/queen/tree/c372da22ca75e95c3c79bee8a83c220fc54d0dcb/datagen/tasks)、[第二课实际配置](https://github.com/queen-project/queen/blob/c372da22ca75e95c3c79bee8a83c220fc54d0dcb/configs/sample_instances/stage2.yaml)、[Pikafish 控制关系](https://github.com/official-pikafish/Pikafish/blob/1c66b9b21cf2f280ce3b3ffa80c1c6609f2b29ff/src/position.cpp)。
