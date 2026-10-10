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

未传该参数的已完成第一版题型数据保持原来的单题、局面内平衡合同。新选项完成了累计频率与每根问题数，不代表已经实现作者每题型不同的根数量、难例来源混合或完整采样分布。

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

## 验证与剩余差异

[原生控制对照](../evidence/paper-course-controllers-native-v1.json) 检查 773 个局面的全部格子，共 69570 次比较，零差异；[八个控制局面](../evidence/paper-course-controller-fixtures-v1.json) 可在无模型、无 vendor 的 CPU CI 中重算。运行入口为 `scripts/build_native_controllers.py` 与 `scripts/verify_native_controllers.py`，编译使用固定、未修改的 Pikafish，运行不加载权重。失败的构造局面检查保留在忽略的运行目录中。

题型语义、生成／隔离／读回、原始评测及显式累计频率／多题采样已实现。完整采样分布仍待制作：尚未按题型分配不同的根数量，也尚未按作者配置混合 80% 有将军着法、100% 被将军，以及将死／近将死将军／普通将军／困毙／随机 30%／20%／10%／5%／35% 的难例来源池。这个比例来自实际 YAML，原题型文件的旧注释有不同数字。

正式大数据、有限数据遍历、按词元的训练损失、新课程的验证选模策略，以及全宽桥接的完整四课训练仍需完成。没有用这份新合同训练学生，也没有证明棋力或讲解收益。

来源：[作者题型源码](https://github.com/queen-project/queen/tree/c372da22ca75e95c3c79bee8a83c220fc54d0dcb/datagen/tasks)、[第二课实际配置](https://github.com/queen-project/queen/blob/c372da22ca75e95c3c79bee8a83c220fc54d0dcb/configs/sample_instances/stage2.yaml)、[Pikafish 控制关系](https://github.com/official-pikafish/Pikafish/blob/1c66b9b21cf2f280ce3b3ffa80c1c6609f2b29ff/src/position.cpp)。
