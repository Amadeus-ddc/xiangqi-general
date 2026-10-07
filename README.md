# Xiangqi General（象棋将军）

将 [Queen 论文](https://arxiv.org/abs/2610.03695v1) 的棋类专家与语言模型桥接路线迁移到中国象棋，目标是同时提供强走法和可核对的中文讲解。当前能力与实测限制见 [STATUS.md](STATUS.md)，模型用途与评测边界见 [模型说明](docs/MODEL_CARD.md)。

专家为冻结的 Px0 20 层、512 维 Transformer；语言模型为固定 revision 的 Qwen3-4B-Instruct-2507。可训练交叉注意力读取 90 个棋盘 token。首轮原型为四个 384 宽桥接块；扩大方案为 16 个 768 宽桥接块与 105 个棋盘词元。完整配置、分割、检查点选择和输出校验由各实验 manifest 记录。

## 安装与 CPU 验证

Python 3.11+。先安装适合机器的 PyTorch，再安装项目：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,train,reference]'
python -m pytest -q
python -m build
python scripts/check_release.py
```

测试不下载模型、不要求 GPU。正式训练需要 CUDA GPU；依赖只安装在项目环境。

## 公共源码与模型

```bash
python scripts/fetch_sources.py
python scripts/bootstrap.py
python scripts/build_native_reference.py
python scripts/build_native_network.py
python -m xqgeneral.verify_network
```

下载源码与权重的身份固定在 `configs/sources.json` 和权重 manifest。权重、数据、上游副本和运行产物不进入 Git。完整网络对照使用官方 C++ 输入编码器、未改动的 C++ ONNX 导出器和 ONNX CPU 后端，比较全部策略、胜和负、剩余步数及桥接特征。

## 数据与训练

```bash
python -m xqgeneral.data --games 120 --root-sampling legacy_black_only --output data/research-v1
python -m xqgeneral.symmetry --data data/research-v1 --output data/research-balanced-v1
python -m xqgeneral.cache_features --data data/research-balanced-v1 \
  --output data/research-balanced-v1/features-16.pt --depths 0 1 2 3 4 5 6 8 9 10 11 12 14 15 17 19
python scripts/freeze_run.py --output runs/research-v2/bridge/source-run -- \
  python -m xqgeneral.curriculum --config configs/research-v2.json --mode bridge
```

这些命令重建本次对称增强实验。新数据生成默认使用红黑平衡采样；`legacy_black_only` 仅用于重建第一版及其对称派生数据。新实验必须使用新的输出目录。课程按棋局预先划分训练、验证、独立测试，检查当前与未来局面重叠。课程间继承验证集选出的最佳检查点，并混合之前课程。长任务放在 tmux；每次运行保存配置、输入与输出哈希及冻结执行源码。

相同预算的纯语言基线使用 `--mode text_lora`。原始回答的平衡题评测入口：

```bash
python -m xqgeneral.evaluate_qa --checkpoint CHECKPOINT.pt \
  --data data/research-balanced-v1 --features data/research-balanced-v1/features-16.pt \
  --split validation --output runs/qa-validation
```

验证集用于选模型；独立测试不参与选模型。评测保留原始错误，不通过引擎修复答案。

## 讲解教师

教师身份固定在 `configs/teachers.json`：初始标注为用户授权的 GPT-6-Astra Low 子代理；搜索汇总为本地官方 Qwen3.8-27B 完整权重，BF16、无量化。后者只做推理。

```bash
python -m xqgeneral.prepare_explanations --data data/research-v1 --output data/astra-seed-v1
# 显式教师生成 annotations-*.jsonl 后，核对全部 ID、身份、结构与走法。
python -m xqgeneral.collect_teacher --input data/astra-seed-v1 --color-mirror
pip install -e '.[teacher]'
python scripts/fetch_teacher.py
python scripts/verify_teacher_weights.py
```

标注文件每行包含 `id`、真实 `explanation`、`teacher_model`、`reasoning_effort` 和 `backend`。不生成模板替代缺失教师。颜色派生项会记录来源，不计为独立教师调用。完整教师加载与实际蒸馏的完成状态见 `STATUS.md`。

追加训练标注时，`prepare_explanations --exclude-queries` 排除已有上下文；新增批次只增加训练项。教师完成全部标注后，可保留原标签与验证/测试集并合并：

```bash
python -m xqgeneral.merge_teacher --inputs data/astra-seed-v1 data/astra-additional-v1 \
  --output data/astra-seed-full-v1
python -m xqgeneral.collect_teacher --input data/astra-seed-full-v1 \
  --output data/astra-explanations-full-v1 --color-mirror
```

已合并的批次可再次作为 `merge_teacher --inputs` 的首项，使用其 `manifest.json`，继续保留原验证和测试项。颜色镜像同时转换坐标、棋子名称和明确的 `rank0` 至 `rank9` 横线编号；正文中的方向、称谓及战略理由仍须复核。新的实战标注本身红黑平衡，可直接使用原始讲解。

## 讲解训练与搜索蒸馏

对已有训练讲解的审查与修订要保存原文、拒收、修订稿和独立复核。完整审查包声明其输入、最终 `annotations.jsonl` 和哈希；修订链须逐项绑定被拒收正文，最终正文须独立接受。仅修订训练正文，已有走法、评分、历史及验证／测试标签保持不变：

```bash
python -m xqgeneral.revise_prose --data data/astra-explanations-full-v4 \
  --reviews data/old-mirror-prose-reviewed-v1 data/old-original-prose-reviewed-v1 \
  --output data/astra-explanations-full-v5
```

使用新输出目录；抽样复核不代表其余训练正文已审查，数据修订完成也不证明学生收益。新的训练须另行冻结配置与源码。

`configs/explanation-sft.json` 从已完成、哈希核验的最佳课程检查点初始化。讲解训练同时更新桥接块、棋盘词元和完整解码器，使用 FP32 参数、BF16 计算与梯度累积，并保留课程问答回放。验证集只选讲解检查点。全量优化器和检查点较大，需要为运行目录保留磁盘空间。

```bash
python scripts/freeze_run.py --output runs/explanation-v1/source-run -- \
  python -m xqgeneral.sft --init COURSE_CHECKPOINT.pt --output runs/explanation-v1/training
python -m xqgeneral.search_distillation --checkpoint SFT_CHECKPOINT.pt \
  --child-contract move_eval --output runs/search-v1/mining
python -m xqgeneral.local_teacher --input runs/search-v1/mining/queries.jsonl \
  --output runs/search-v1/consolidation
python -m xqgeneral.collect_search --queries runs/search-v1/mining/queries.jsonl \
  --responses runs/search-v1/consolidation/responses.jsonl \
  --validation-data data/astra-explanations-full-v1 --output data/search-v1
```

搜索使用真实学生根节点和子节点回答，依据引擎核验递归进入有问题的子节点，并要求主变化的首个差异确实改善。`--child-contract move_eval` 按推荐着法和评分决定子节点是否递归，保留原始回答及完整结构错误；默认 `full` 仍要求整份子分析通过。汇总教师接收实际使用的变化、逐步事实、原生记谱与核验结果；学生原始正文保留在轨迹中，未经语义保证的正文不传给汇总教师。用于标签的主线和全部分支仍须通过完整历史、终局、保留集隔离及严格改进检查。训练根节点及新子节点排除保留集局面。Qwen 汇总教师只生成讲解正文，经过验证的结构与根评分另行保留。根评分与子分析反号后的分支估计标明来源；中文记谱须紧邻对应坐标并通过原生规则核验。机械着法检查不能替代战略语义复核。后续挖掘逐次保存所有成功的自由和指定着法搜索，包括完整历史与原始引擎回答；旧轨迹缺失的回答仍按缺失记录。空产出不算完成蒸馏。新数据需重新缓存专家特征，再以新配置和输出目录训练；不能覆盖旧实验。

`configs/research-v3.json` 提供一个可选对照：专家与纯语言两组都读取同一份从输入局面得到的 90 格棋盘字典。这改变了论文仅通过专家输入棋盘的条件，应单独报告结果和专家消融。

根据中间模型的非法走法实测，增加一门引擎监督走法课程。新标签来自真实根局面与重新搜索的变化子局面，保持原棋局分割并排除保留集当前及未来局面：

```bash
python -m xqgeneral.policy_data --input data/astra-seed-full-v1 \
  --replay-data data/research-balanced-v1 --output data/move-quality-v1
python -m xqgeneral.cache_features --data data/move-quality-v1 \
  --output data/move-quality-v1/features-16.pt --depths 0 1 2 3 4 5 6 8 9 10 11 12 14 15 17 19
python -m xqgeneral.sft --recipe configs/move-quality-v2.json \
  --init DICTIONARY_COURSE_CHECKPOINT.pt --output runs/move-quality-v2/bridge/training
```

冻结解码器时，`trainable_parameter_dtype: float32` 将桥接、棋盘词元或 LoRA 参数保留为 FP32，计算继续使用 BF16；棋盘词元输出匹配冻结解码器精度。`configs/move-quality-v2.json` 启用此设置。旧配置默认 `base`，续训不能更改精度合同。

实战分布补充通过引擎自对弈生成完整历史上下文。前 24 步从分数接近的 MultiPV 候选中抽样，之后选择引擎最佳走法；每局分别抽取双方局面。保留集根局面和未来局面会被排除，步数上限记为截尾。生成结果没有神经讲解或走法训练标签，仍须独立搜索：

```bash
python scripts/freeze_run.py --output runs/selfplay/source-run -- \
  python -m xqgeneral.engine_selfplay --games 128 --workers 8 \
  --reserved-data data/move-quality-v1 --output data/engine-selfplay-train-v2
python -m xqgeneral.policy_data --input data/astra-seed-full-v1 \
  --replay-data data/research-balanced-v1 \
  --extra-contexts data/engine-selfplay-train-v2/contexts.jsonl \
  --limit 12288 --output data/move-quality-selfplay-v1
```

长任务放入 tmux。自对弈按完成棋局原子保存，用相同冻结源码、参数和输入可续跑；配置、源码或输入哈希改变时拒绝续跑。1000 节点停止若使最佳走法缺少完整评分，会真实重试一次 10000 节点搜索并保留失败输出。`--extra-contexts` 拒绝保留集棋局归属和不一致历史，去除重复或与保留集当前/未来局面重叠的训练项；它不改变验证、测试标签。

已有规划或讲解保留集时，`policy_data --reserved-data DATASET...` 将其棋局、根局面和全部未来分支加入排除范围，仅使用保留信息，不追加这些标签到回放集。相关文件哈希纳入续跑合同，新训练标签及颜色镜像也检查这组保留局面。

若输入目录已包含全部旧搜索结果，追加 `--no-descendants` 可只搜索新补充上下文，跳过旧 PV 的再次派生；原查询和走法标签仍保留。

多步规划课程从这些已完成搜索的主变化及候选分支生成最多六步的走法序列。每步按完整历史检查终局，训练项与保留集根及全部变化局面重叠时剔除；颜色派生项保留原分割。新课程复用原根局面的专家缓存，不生成神经讲解。

```bash
python -m xqgeneral.planning_data --data data/move-quality-selfplay-v2 \
  --workers 8 --chunk-size 64 --isolation-workers 8 --output data/move-planning-v1
python scripts/freeze_run.py --output runs/move-planning-v1/source-run -- \
  python -m xqgeneral.sft --recipe configs/move-planning-v1.json \
  --init WARM_MOVE_CHECKPOINT.pt --output runs/move-planning-v1/bridge/training
python -m xqgeneral.evaluate_plans --checkpoint PLAN_CHECKPOINT.pt \
  --data data/move-planning-v1 --features data/move-quality-selfplay-v2/features-16.pt \
  --split validation --output runs/planning-validation
```

生成和隔离默认单进程；`--workers 8` 并行生成标签，`--chunk-size` 控制生成批次，`--isolation-workers 8` 并行复演隔离所需的当前与未来局面。隔离进程只接收局面与变化，重复上下文共用结果；标签顺序和拒收计数保持一致。分割检查覆盖全部未来分支；颜色镜像也转换分支走法。只在新输出目录运行，生成、隔离、校验和写盘会分别报告进度。

规划评测保留原始序列，分别检查整条合法性、条件首着、参考长度和逐步走法损失。指定候选的首着不计入最优选招指标，后续每步独立搜索；终局后续着仍是错误。短答案不会因前缀合法就通过长度合同。课程的有效训练配置由 `sft` 入口计算步数；改配置须新建实验。

也可用已完成独立搜索的实战局面补充四门规则问答：

```bash
python -m xqgeneral.selfplay_grounding --data data/move-quality-selfplay-v3 \
  --reserved-data data/move-planning-v1 data/astra-explanations-full-v5 \
  --game-prefix engine-selfplay-20261029- --train-roots 8192 \
  --workers 8 --chunk-size 16 --output data/research-selfplay-v1
```

每个原始根局面生成当前静态、当前动态、未来静态和未来动态问题，并生成颜色派生项。未来前缀从引擎最佳变化中抽取一至六步，保留初始局面与完整历史；历史已终局的根局面和未来局面都拒收。原始根局面按双方平衡采样，派生项不算新独立棋局。已有课程及其验证、测试标签保持原样，所有新问题的根局面、所问未来变化和镜像排除保留局面。

扩充时可指定 `--max-future-plies 8 --question-formats 3 --split-workers 16`：从实际存在的合法变化中抽取最多八步，每道题选择一种提问方式，镜像使用同一方式，答案仍为原生规则计算的规范结果。改写不增加独立根数，也不把题目重复三遍。默认六步和原提问方式保持兼容。有限节点搜索的最佳着法是引擎监督；规则题的答案根据所问棋盘重新计算。自对弈的步数截尾不能当成和棋，完整历史与保留变化仍须检查。

并行生成按查询顺序返回，预取数量有界，各查询使用固定独立随机种子；相同参数的单进程与多进程生成保持标签一致。这个数据集只有规则问答，不生成神经讲解或重新搜索。训练前还须检查词元长度和完成的特征缓存，数据生成本身不证明模型改善。长任务用冻结源码在 tmux 内运行，并选择新输出目录。

可在另一个 tmux 会话中，按实际走法与变化结果持续选检查点：

```bash
python scripts/freeze_run.py --output runs/planning-selection/source-run -- \
  python -m xqgeneral.select_checkpoint --training runs/move-planning-v1/bridge/training \
  --training-session xqgeneral-move-planning-v1-bridge --every 2000 \
  --output runs/planning-selection/models
```

选择器每隔 2000 步及训练结束时固定真实检查点，评测 192 道原始走法题及 96 道规划题，以 `0.6 × 走法无明显失误率 + 0.4 × 完整规划合同通过率` 排序。训练器的 NLL 最佳模型仍保留；选择器另存 `selected.pt`、候选原始输出及指标。只允许验证集、无修复且无合法性强制的结果参与选择；续跑验证候选哈希和训练合同。这个选择分数不是棋力、Elo 或讲解质量结论，最终模型仍须独立测试及完整对弈。

讲解训练可额外指定 `--explanation-data DATASET`，加入 96 条原始讲解，以 `0.4 × 走法无明显失误率 + 0.2 × 规划合同通过率 + 0.4 × 完整讲解合同通过率` 选模。三项必须来自同一固定检查点；验证文件、特征缓存和各项评测证据均核验哈希。完整讲解合同包含规则、变化及声明事实，中文战略质量仍须另行评审。省略此选项时保留原走法与规划分数。

基础课程补训可指定 `--qa-data DATASET --qa-per-task 12`，在同一检查点上加入 264 道原始棋盘问答，覆盖四门课程的全部 22 类题，以 `0.4 × 走法无明显失误率 + 0.2 × 规划合同通过率 + 0.4 × 平衡问答正确率` 选模。各类题数量必须相等，独立测试、引擎代答、强制合法输出和变动证据均拒收；此选项与讲解选模互斥。类别指标与全部原回答分别保存，问答分数不能替代正文审查或对弈。

原 `configs/selfplay-foundation-v3.json` 混合补训等待任务已在开始训练前保留并撤换。其 811684 条混合记录及 799424 条训练／验证编码的预检证据仍保留，不能覆盖新扩充数据。课程训练量复核见 `evidence/paper-foundation-volume-audit.json`：本地旧四门实际共 5750 步、92000 次样本呈现；论文配置上限为 240000 步、有效批量 256，且允许早停，不能当成全部实际完成步数或独立样本数。

主线现在改为 `configs/foundation-human-engine-clean-v1.json`：冻结固定版本的预训练 Px0 专家和官方 Qwen 基座，重新初始化桥接及 105 个棋盘词元，不加载旧课程或讲解权重。论文的四门 SC → DC → SF → DF 本身就是桥接训练，必须依次完成，再从最终课程权重做讲解 SFT。棋谱下载、规则出题和教师标注可以先准备。

旧 v4 继承旧四门权重，首门实际更新 210 步后按用户要求停止，未到首次问答验收；原数据、源码、特征和日志保留。其 1155116 条规则题的完整历史、未来隔离、特征及原生答案抽样已再次读回，见 `evidence/foundation-curriculum-v4-training-readback.json`。这些引擎来源题将与经过规则检查的多样真实棋谱合并，合并后重新预检，不改写旧实验输入。

新训练计划使用四卡全局 256／微批量 4，四门上限 50000／60000／30000／100000 步，实际步数由原始问答验收决定。每 512 步对已引入课程的每类题检查 128 条原回答、三种提问方式、总正确率及最差题型；未达标不得进入下一门。训练中重置各门优化器，回放旧门数据；独立测试不选模。正式启动须等待新增棋谱、全部未来分割、词元与特征缓存预检完成。

四卡新初始化的连续四步与二加二恢复已经逐位核验，见 `evidence/four-gpu-clean-foundation-resume.json`。该配置使用 `ddp_find_unused_parameters: true` 保持同卡恢复的归约合同；切换这个设置不能恢复旧优化器。可用以下命令复查真实训练输出；这是执行检查，未完成课程：

```bash
.venv/bin/python scripts/verify_exact_resume.py \
  --continuous runs/four-gpu-clean-resume-v4/continuous \
  --resumed runs/four-gpu-clean-resume-v4/resumed
```

真实棋谱导入先固定来源和许可证，再逐着检查完整历史，按规范化棋局身份预先划分，之后才生成规则题。默认保留人类、电脑及人机来源类别，类别和棋手名称都是源记录声明；不把实战走法当作最优着法，不使用网站原讲解冒充神经标注。CCPD 首批 512 局已通过，见 `evidence/human-master-games-native-import-v2.json`；全库多棋手导入仍在进行；近期另完成 93 局、6989 个半回合、2019—2023 年的 18 个参赛者名称，见 `evidence/recent-recorded-games-native-import-v2.json`。后续训练抽样使用双方棋手与赛事上限，同时覆盖年代、来源和胜负；镜像及改写不增加真实棋局数。

```bash
python -m xqgeneral.recorded_sources --source data/sources/ccpd-v1 \
  --revision 368a47a947773dd8692c026e286dd19b6277b993 \
  --output runs/new-recorded-source-catalogue
python -m xqgeneral.human_games --source data/sources/ccpd-v1 \
  --revision 368a47a947773dd8692c026e286dd19b6277b993 \
  --candidates runs/new-recorded-source-catalogue/candidates.json \
  --max-games 60000 --workers 32 --seed 20261051 \
  --output data/new-recorded-games \
  --public-evidence evidence/new-recorded-games-import.json
```

长任务须在 tmux 中使用 `scripts/freeze_run.py` 固定源码。当前全库导入的命令与检查方式在 `runs/recorded-games-ccpd-v3-import/plan.json`，近期棋谱采集在 `runs/recent-recorded-games-import-v1/plan.json`。中断保留输出、另建目录；未完成的数据不启动正式课程。完整棋谱来源与发布边界见 `THIRD_PARTY_NOTICES.md`。

`configs/explanation-sft-v3.json` 在走法课程后混合讲解、走法与基础课程回放；其中的新增 Astra 数据集必须先完成全量标注、复核与收集，不能以未完成分片替代。该配置按每条样本的平均监督损失训练（`loss_normalization: example`），使短走法题保留配置中的回放比例。已有配置默认仍按词元归一化；验证和检查点选择继续使用词元平均 NLL。

`configs/explanation-sft-v4.json` 从完成四门课程的正式走法检查点开始，单独以 15% 比例采样 384 条全量交叉复核的实战原始讲解，并保留 20% 走法、10% 多步规划及 5% 基础问答。其余 50% 使用原讲解集；实战重采样池只改变 `stage`，保留题目、答案、分割和完整历史。该自适应实验同时改变初始化、数据和回放。实际训练与五候选选模均已完成：9216 步，按原始能力选第 9216 步，NLL 第 6144 步另存；实际权重及十五份原始评测已独立读回。完整主线仅 23/96 合法、完整讲解合同仅 8/96 通过，仍未达到可靠教学要求，见 `evidence/explanation-v4-functional-selection-readback.json`。`configs/move-planning-v2.json` 使用同一规划课程与预算，从正式四门课程走法检查点初始化，与第二门课程后的 v1 分别保存。 两个规划训练与最终选模已完成并读回；v1 选择第 10000 步，v2 选择第 11500 步，完整规划合同分别为 33/96 和 37/96，见 `evidence/move-planning-completed-readback-v1.json`。这些指标不证明完整棋力或可靠正文。

`configs/explanation-sft-v5.json` 继续已完成的全量讲解模型，使用审查后数据、新优化器与 15% 规划回放。45% 主讲解池已包含 384 条实战原始讲解，另以 15% 重采样这些条目；它们不是新的独立标注。该实验已训练 3584 步，最终能力选模在三个真实候选中选择第 3584 步，NLL 第 512 步另存。全部权重与原始能力产物已读回，初始化导出保留原权重。输入固定为当时的 v5 数据；正文语义、完整对弈和独立测试仍须验证，见 `STATUS.md`。

旧 `configs/explanation-sft-v6.json` 从 v5 继续训练已复核 v7 讲解，已按用户要求停止，不自动续跑。最后记录更新 1878，最新可恢复优化器步数 1536；第 1024 步原始能力分数 0.3875，父模型为 0.395833，未显示收益。真实祖先停在旧第二门课程，并非新的四门完成模型。其输入、权重和中途评测保留，见 `evidence/explanation-v6-interim-1024.json`、`evidence/foundation-clean-route-switch.json`。后续主线 SFT 必须从新四门全部验收后的权重开始；尚无已训练的搜索蒸馏轮次。

大型预检可调用 `verify_splits(rows, workers=N)` 并行检查全部唯一未来分支；默认 `workers=1` 保持单进程接口。预取限制为每个进程两批，跨棋局归属在去重前登记，子进程的非法未来错误向主进程传播。真实 1152 条规划上下文的串行／八进程结果一致，该样本耗时由 25.81 秒降至 6.04 秒；完整数据准备的加速尚未测量，见 `evidence/parallel-split-verification.json`。

新讲解配置启用 `deterministic_training`，在创建 CUDA 上下文前配置 CuBLAS 工作区并要求 PyTorch 使用确定性算法。续训核对损失归一化、确定性设置和工作区合同，不能中途切换；不支持的确定性操作会报错。精确复现仍要求相同源码、输入、环境和 GPU 配置，不保证跨平台逐位一致。

搜索蒸馏检查子分析的主线和全部候选分支，按初始局面与完整历史拒绝终局后的续着，并排除更深的保留集局面。汇总后的标签保存主线及分支的未来走法，分割检查覆盖全部变化；保留讲解答案中的主线和分支也计入隔离范围。实际没有严格改进目标时，不执行空数据蒸馏。

## 原始输出与对弈评测

```bash
python -m xqgeneral.evaluate_explanations --checkpoint CHECKPOINT.pt \
  --data data/astra-explanations-full-v1 --split validation --limit 96 \
  --output runs/explanation-validation
python -m xqgeneral.evaluate_games --checkpoint CHECKPOINT.pt \
  --nodes 100 1000 10000 --output runs/matches
```

讲解评测使用更高预算、独立执行的皮卡鱼检查推荐走法、变化、评分方向和棋盘事实；中文战略正文仍需单独评审。对弈从固定开局交换红黑，完整保留历史。非法模型走法判负，达到步数上限记为截尾，结果不自动换算 Elo。先用验证集完成开发，最终选择的模型才进入独立测试。

对弈对手只读取固定节点预算返回的合法走法并保存完整引擎输出，不要求附带完整评分，也不追加搜索节点。独立质量评分和训练标签继续使用严格评分接口，缺少完整评分时不会被视为有效评分。

走法课程采用相同的独立引擎质量判断，同时统计原始 UCCI 格式、合法率和相对教师标签的完全匹配率。非法回答不会被引擎替换：

```bash
python -m xqgeneral.evaluate_moves --checkpoint MOVE_CHECKPOINT.pt \
  --data data/move-quality-v1 --split validation --output runs/move-validation
```

专家消融使用 `--memory zero` 或 `--memory shuffled`，保持验证题及独立裁判预算一致。颜色派生样本会单独计数；合法样本上的平均质量损失必须与覆盖全部样本的合法率及无明显失误率一起解读。

训练过走法课程的检查点可使用规则约束解码。模型概率在合法走法词元树中选招，不调用引擎提供走法；该模式与原始生成分别记录，合法率由约束保证，棋力仍需独立评分和完整对弈验证：

```bash
python -m xqgeneral.evaluate_moves --checkpoint MOVE_CHECKPOINT.pt \
  --decoding legal --beams 4 --output runs/legal-move-validation
python -m xqgeneral.evaluate_games --checkpoint MOVE_CHECKPOINT.pt \
  --action-mode legal_move --move-beams 4 --output runs/legal-move-matches
```

`legal_move` 对弈只评测走法，不评测讲解。默认对弈入口继续使用原始 JSON 讲解答案。

可将已完成引擎评测的原始讲解提交给本地 BF16 教师作盲评，检查事实、战略理由、教学清晰度与要求完成度：

```bash
python -m xqgeneral.review_explanations prepare \
  --predictions runs/explanation-validation/judged-predictions.jsonl --output runs/prose-review/queries
python -m xqgeneral.local_teacher --input runs/prose-review/queries/queries.jsonl \
  --output runs/prose-review/inference
python -m xqgeneral.review_explanations collect --queries runs/prose-review/queries/queries.jsonl \
  --responses runs/prose-review/inference/responses.jsonl --output runs/prose-review/ratings
```

裁判输入不带检查点身份，附根局面的棋子坐标表、每步前后棋盘、规则事实及回答声明的评分，保留具体无依据声明和理由。神经裁判与训练汇总教师使用同一基础模型，评分存在相关偏差；它不计为人工评价，也不能替代机械事实和独立走法检查。

## 原型推理

已有本地首轮检查点时：

```bash
python -m xqgeneral.ask --adapter runs/pilot-v1/adapter.pt \
  --moves b0c2 --question 'c2 上是什么棋子？只回答棋子名称。'
```

原型检查点只完成过 80 步训练。它的回答不调用皮卡鱼，当前不具备成熟教练的验证证据。

完成训练后，可使用终端学习入口显示棋盘、查看推荐与主要变化，并保存完整棋局历史：

```bash
python -m xqgeneral.coach --checkpoint CHECKPOINT.pt --interactive --output runs/study.json
python -m xqgeneral.coach --checkpoint CHECKPOINT.pt --resume runs/study.json
```

学习入口核验推荐、主线和全部候选分支，按保存的完整历史拒绝终局后的续着。检查失败时保留原始答案和错误；规则检查通过仍需独立验证讲解理由与棋力。

交互命令为 UCCI 走法、`hint`、`undo`、`board` 与 `quit`。学习入口保留原始模型回答并执行规则核验；推荐本身不调用引擎。各检查点的实测质量以状态与评测证据为准。

## 开源与贡献

源码采用 GPL-3.0-or-later，保留 Px0 派生实现与协议的来源声明。依赖、参考代码与权重的边界见 [第三方说明](THIRD_PARTY_NOTICES.md)，贡献步骤见 [CONTRIBUTING.md](CONTRIBUTING.md)。本项目独立于 Queen、Px0、Qwen 和皮卡鱼官方项目。训练完成后的棋力与讲解结论以独立评测为准。
