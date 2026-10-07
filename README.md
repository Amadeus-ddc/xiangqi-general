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
  --output runs/search-v1/mining
python -m xqgeneral.local_teacher --input runs/search-v1/mining/queries.jsonl \
  --output runs/search-v1/consolidation
python -m xqgeneral.collect_search --queries runs/search-v1/mining/queries.jsonl \
  --responses runs/search-v1/consolidation/responses.jsonl \
  --validation-data data/astra-explanations-full-v1 --output data/search-v1
```

搜索使用真实学生根节点和子节点回答，依据引擎核验递归进入有问题的子节点，并要求主变化的首个差异确实改善。训练根节点及新子节点排除保留集局面。Qwen 汇总教师只生成讲解正文，经过验证的结构与根评分另行保留。空产出不算完成蒸馏。新数据需重新缓存专家特征，再以新配置和输出目录训练；不能覆盖旧实验。

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
  --workers 8 --chunk-size 64 --output data/move-planning-v1
python scripts/freeze_run.py --output runs/move-planning-v1/source-run -- \
  python -m xqgeneral.sft --recipe configs/move-planning-v1.json \
  --init WARM_MOVE_CHECKPOINT.pt --output runs/move-planning-v1/bridge/training
python -m xqgeneral.evaluate_plans --checkpoint PLAN_CHECKPOINT.pt \
  --data data/move-planning-v1 --features data/move-quality-selfplay-v2/features-16.pt \
  --split validation --output runs/planning-validation
```

生成器默认单进程；`--workers 8` 保持输入与标签顺序，`--chunk-size` 控制任务批次。分割检查覆盖全部未来分支；颜色镜像也转换分支走法。只在新输出目录运行，生成、隔离、校验和写盘会分别报告进度。

规划评测保留原始序列，分别检查整条合法性、条件首着、参考长度和逐步走法损失。指定候选的首着不计入最优选招指标，后续每步独立搜索；终局后续着仍是错误。短答案不会因前缀合法就通过长度合同。课程的有效训练配置由 `sft` 入口计算步数；改配置须新建实验。

也可用已完成独立搜索的实战局面补充四门规则问答：

```bash
python -m xqgeneral.selfplay_grounding --data data/move-quality-selfplay-v3 \
  --reserved-data data/move-planning-v1 data/astra-explanations-full-v5 \
  --game-prefix engine-selfplay-20261029- --train-roots 8192 \
  --workers 8 --chunk-size 16 --output data/research-selfplay-v1
```

每个原始根局面生成当前静态、当前动态、未来静态和未来动态问题，并生成颜色派生项。未来前缀从引擎最佳变化中抽取一至六步，保留初始局面与完整历史；历史已终局的根局面和未来局面都拒收。原始根局面按双方平衡采样，派生项不算新独立棋局。已有课程及其验证、测试标签保持原样，所有新问题的根局面、所问未来变化和镜像排除保留局面。

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

`configs/selfplay-foundation-v2.json` 是后续实战规则补训配置：90% 四门问答、5% 走法与 5% 多步规划。回放目录只保留走法和规划项，避免重复采样旧规则题；先在两组已完成规划训练的全部候选上评测规则问答，再按同份权重的问答、走法和规划分数选择初始化；使用新优化器、FP32 可训练参数、32 条全局批量及 4 条微批量。桥接与棋盘词元学习率分别为 `1e-4` 和 `1e-5`，最多 24000 步，至少 6000 步后允许按验证损失提前停止。实际数据、特征与全部训练/验证词元长度通过核验后才启动；这是自适应补训，尚未证明收益。

`configs/explanation-sft-v3.json` 在走法课程后混合讲解、走法与基础课程回放；其中的新增 Astra 数据集必须先完成全量标注、复核与收集，不能以未完成分片替代。该配置按每条样本的平均监督损失训练（`loss_normalization: example`），使短走法题保留配置中的回放比例。已有配置默认仍按词元归一化；验证和检查点选择继续使用词元平均 NLL。

`configs/explanation-sft-v4.json` 从完成四门课程的正式走法检查点开始，单独以 15% 比例采样 384 条全量交叉复核的实战原始讲解，并保留 20% 走法、10% 多步规划及 5% 基础问答。其余 50% 使用原讲解集；实战重采样池只改变 `stage`，保留题目、答案、分割和完整历史。该自适应实验同时改变初始化、数据和回放，质量必须另行实测。`configs/move-planning-v2.json` 使用同一规划课程与预算，从正式四门课程走法检查点初始化，与第二门课程后的 v1 分别保存。

`configs/explanation-sft-v5.json` 继续已完成的全量讲解模型，使用审查后数据、新优化器与 15% 规划回放。45% 主讲解池已包含 384 条实战原始讲解，另以 15% 重采样这些条目；它们不是新的独立标注。该实验保留先前训练及其失败，待初始化检查点实际完成并核验后启动。新数据并未全量审查，教学质量须继续评测。

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
