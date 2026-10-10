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

当前初始讲解集为 10758 条训练、384 条验证和 512 条独立测试。重新计数并哈希三份原文件：训练项中 7619 条没有颜色派生标记、3139 条为已有颜色派生；验证为 288＋96，测试为 384＋128，见 `evidence/clean-explanation-original-capacity-v1.json`。未标记项也不能全部视为独立人类棋局。原论文初始教师标注 15000 个局面，经走法筛选保留 8602 个，其中 8402 个训练、200 个验证，训练四轮；我们的四轮配方为 2690 次更新。数量相近不代表标签质量或覆盖相同。

全部 384 条现有教师验证标签已由独立引擎以每局面 100 万节点复查：381 条推荐走法无明显失误，352 条完整主变化每步无明显失误，推荐走法及主变化全部合法。标签原文和错误保留，独立读回重算统计，见 `evidence/clean-teacher-validation-engine-reference-v1.json`。这不是学生成绩、正文语义评价或全部训练标签的质量证明；独立测试答案没有参与。

后续搜索蒸馏的规模更大：[论文附录 B.1—B.2](https://arxiv.org/html/2610.03695v1#A2) 每轮使用 21 万个人类对局局面、11.5 万个战术题及 6250 局学生与引擎对弈产生的约 5万—10万个局面，搜索筛选后得到约 15万—28.5万条训练样本，共七轮。当前八来源重新隔离后的混合池实际提供 495306 个历史，其中 8982 个来自战术主线、7754 个是独立战术局面；它们尚未成为合格蒸馏标签，不能保证七轮独立棋局和战术容量。继续补充战术、残局及不同水平实战；首轮优先使用现有保留集隔离后的棋谱和真实学生对弈，先测有效搜索产出率，再扩充合格样本。轮数依据验证棋力、完整变化和讲解质量决定，重复问法或颜色镜像不增加独立局面容量。

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
python scripts/freeze_run.py --output runs/search-inputs-v1/source-run -- \
  python -m xqgeneral.search_inputs --data data/research-human-engine-v1 \
  --output data/search-inputs-v1
python scripts/freeze_run.py --output runs/search-v1/source-run -- \
  python -m xqgeneral.search_distillation --checkpoint SFT_CHECKPOINT.pt \
  --data data/research-human-engine-v1 \
  --features data/research-human-engine-v1/features-16.pt \
  --prepared-inputs data/search-inputs-v1 --limit 512 \
  --child-contract move_eval --output runs/search-v1/mining
python -m xqgeneral.local_teacher --input runs/search-v1/mining/queries.jsonl \
  --output runs/search-v1/consolidation
python -m xqgeneral.collect_search --queries runs/search-v1/mining/queries.jsonl \
  --responses runs/search-v1/consolidation/responses.jsonl \
  --validation-data data/astra-explanations-full-v1 --output data/search-v1
```

`local_teacher` 默认逐条生成，支持 `--batch-size 4` 的左侧填充批量推理；按每条回答的实际结束词元裁切，提示词元数不包含填充。批量和生成预算写入续跑合同，改动后须使用新输出目录；中断续跑跳过已保存的查询。`search_pilot` 的新配置可显式设置正整数 `teacher_batch_size`，默认逐条调用及旧合同保持兼容。23 项新增检查、相关 41 项及全部 724 项 CPU 测试通过；完整 BF16 教师的实际批量吞吐和回答差异尚未测量，当前已冻结队列继续使用原入口。

`search_inputs` 逐行读训练文件，保存每个完整历史的首份原记录，再按原搜索种子排序；可用 `--source-preflight COMPLETE_MANIFEST.json` 核对已完成预检中的训练文件身份。候选保留原始字段、棋局分割和已有颜色派生标记，不产生蒸馏标签。搜索入口支持上述完整候选池，核验数据、种子、输出哈希与源码合同后取前 `--limit` 条；省略候选池时仍逐行提取，选取顺序相同。旧搜索任务继续使用各自冻结源码；变动源码、数据或预算须使用新输出。

现有课程训练集的完整提取已完成：2831004 条／约 17.96GB 生成 128682 个完整历史候选，包含已有 64341 个颜色派生历史。生产约 202 秒、峰值进程内存约 2.37GiB；独立入口重读全部训练行、重算 SHA256，逐字段匹配全部首记录和完整排序，检查缓存键／头部，并抽样复演 320 个原生历史。见 `evidence/search-inputs-human-engine-full-v1.json`。这份候选池只覆盖当前课程用过的局面；未用棋谱的新增候选见下段，学生对局和搜索筛选后的合格标签另行计数。新训练仍按四门 → 初始讲解 → 搜索蒸馏的顺序执行。

已有棋谱可通过 `recorded_search_inputs` 扩充：使用完成原生导入的目录、完成的课程／旧讲解足迹、现有候选和当前讲解集，每局最多取六个不同原始历史，交替检查早／中／晚阶段。排除已用棋局及历史、全部来源的保留棋局历史，以及课程／讲解的根和未来变化及颜色对应位置。保留原来源与缺失字段，不生成镜像、神经讲解或最佳走法标签。SQLite 暂存排序，完整候选按种子与历史键稳定排序；搜索入口的 `--recorded-inputs` 会核验完整产物、输入和分割，并把附加保留位置用于全部递归分支。它与 `--prepared-inputs` 互斥。

```bash
python scripts/freeze_run.py --output runs/recorded-search-inputs-v1/source-run -- \
  python -m xqgeneral.recorded_search_inputs \
  --games data/recorded-games-ccpd-v3 data/recent-recorded-games-v2 \
          data/recorded-games-additional-v1 data/modern-recorded-games-v2 \
          data/playstrategy-portable-real-import-v1-parallel \
          data/playstrategy-public-users-native-import-v1 \
  --data data/research-human-engine-v1 \
  --footprints runs/recorded-coach-portable-footprints-v1/footprints/manifest.json \
  --used-inputs data/search-inputs-human-engine-full-v1 \
  --heldout-data data/astra-explanations-clean-v1 \
  --output data/recorded-search-inputs-v1 --per-game 6 --workers 8
python scripts/freeze_run.py --output runs/recorded-search-v1/source-run -- \
  python -m xqgeneral.search_distillation --checkpoint SFT_CHECKPOINT.pt \
  --data data/research-human-engine-v1 \
  --features data/research-human-engine-v1/features-16.pt \
  --recorded-inputs data/recorded-search-inputs-v1 --limit 512 \
  --child-contract move_eval --output runs/recorded-search-v1/mining
python scripts/freeze_run.py --output runs/recorded-consolidation-v1/source-run -- \
  python -m xqgeneral.local_teacher --input runs/recorded-search-v1/mining/queries.jsonl \
  --output runs/recorded-consolidation-v1
python scripts/freeze_run.py --output runs/recorded-collection-v1/source-run -- \
  python -m xqgeneral.collect_search --queries runs/recorded-search-v1/mining/queries.jsonl \
  --responses runs/recorded-consolidation-v1/responses.jsonl \
  --mining-manifest runs/recorded-search-v1/mining/manifest.json \
  --recorded-inputs data/recorded-search-inputs-v1 \
  --heldout-data data/research-human-engine-v1 \
  --validation-data data/astra-explanations-clean-v1 --output data/recorded-distillation-v1
```

新历史可由固定专家现场计算与训练缓存精度一致的特征；合格蒸馏数据仍需另行构建训练缓存。已有完整原生导入用于准备阶段，新增候选的实际搜索继续核验历史与终局。保留集答案只用于未来隔离，不用于训练、选模或测试成绩；原棋谱主线用于保留未来范围，不保证人类着法最优。

最终 `collect_search` 也会检查完整隔离范围：自动识别查询同目录的搜索清单，或使用显式 `--mining-manifest`。新棋谱模式要求已完成的真实搜索清单，核对查询原字节、候选池、三份课程文件和原选中根及其递归后代，再检查全部讲解主线和备选分支。额外保留范围用于最终收取，不只用于搜索。验证／测试标签必须与候选准备时的保留文件字节相同，输出直接复制原文件；来源不完整或不匹配时拒收。收取不会重新执行学生搜索或教师推理。

全量新增候选已完成：六个原生导入来源中，80117 盘未用训练棋局提供 478650 个原始历史，未新增镜像。来源声明为人类实战 87544、公开实战 387678、人机 3428；红／黑为 240740／237910，早／中／晚为 103461／179575／195614。全部候选逐字段对照原棋谱并排除 2715684 个保留位置，另抽样复演 192 个完整历史及未来。正式读取入口实际读取 512 条，检查完整候选尾部，并重新计算三份课程文件的 SHA256。见 `evidence/unused-recorded-search-inputs-full-v2.json`。生产耗时约 561 秒、峰值进程内存约 1.07GiB；首次因平台棋谱缺少可选来源说明而失败的记录保留，修正后使用新目录。全部 431 项 CPU 测试通过。棋手身份与来源声明未经认证，原生全量导入证明沿用已有产物；这仍是候选池，没有新增合格标签、有效产出率或模型收益。

战术片段的独立候选池 `data/tactical-search-inputs-full-v1` 已完成：1601 条训练主线提供 8982 个历史，红／黑为 4601／4381；其中 7744 个没有片段前历史、1238 个从标准初始局面开始。全部候选逐字段匹配原始主线、复演给定起点之后的完整历史及记录后续；七份来源的保留历史、现有课程及讲解的未来分支和颜色局面共 2738116 个位置被排除。正式读取接口实际选择 512 条，并检查全池尾部和三份课程文件的 SHA256，见 `evidence/tactical-search-inputs-full-v1.json`。原棋谱后续仍不是最佳着法标签，缺失的前序历史不会补造。该上下文保留到搜索查询、教师汇总提示与最终标签；收取拒收来源字段或布尔类型的变动。

只抽战术候选时，`--games` 仍需列出全部来源供隔离，`--candidate-kinds` 只限制候选类型。`--min-ply 0` 包含短片段起点；最后一步之前的合法决策点也保留，终局本身不作为根。阶段桶相对于给定主线，不表示片段位于原对局的哪个阶段。长批次用 tmux 和固定源码执行，输出目录须新建：

```bash
python scripts/freeze_run.py --output runs/tactical-candidates-v1/source-run -- \
  python -m xqgeneral.recorded_search_inputs \
  --games data/recorded-games-ccpd-v3 data/recent-recorded-games-v2 \
    data/recorded-games-additional-v1 data/modern-recorded-games-v2 \
    data/playstrategy-portable-real-import-v1-parallel \
    data/playstrategy-public-users-native-import-v1 data/recorded-tactical-lines-ccpd-v1 \
  --data data/research-human-engine-v1 \
  --footprints runs/recorded-coach-portable-footprints-v1/footprints/manifest.json \
  --used-inputs data/search-inputs-human-engine-full-v1 \
  --heldout-data data/astra-explanations-clean-v1 \
  --output data/tactical-candidates-v1 --candidate-kinds recorded_tactical_line \
  --min-ply 0 --per-game 6 --workers 8
```

两份旧池的库存读回曾得到 487592 个不同历史键，并发现 541 个旧根几何命中新增保留范围，不能直接拼接。七来源重新生成的旧混合池保留在 `data/recorded-tactical-search-inputs-full-v1`。现已从全部八份原生来源重新构建 `data/recorded-all-sources-search-inputs-full-v1`，对全部根及记录未来做统一隔离，实际保留 495306 个历史、89448 个训练标识。来源声明为人类实战 87390、公开实战 387693、人机 3487、战术主线 8982、独立战术局面 7754；红／黑为 251451／243855，没有新增镜像，标识不都代表完整对局或独立题目。全部候选逐字段匹配，八份原谱重新哈希，2741974 个保留位置及完整未来／颜色范围重新核验；全部 8982 个战术历史和其记录未来、7754 个给定局面再次原生检查，普通实战另抽样复演 192 个，其余沿用已完成全量导入证明。独立局面零步入选条件另行重算，完整对局与片段的采样算法未独立重写，原生规则共享。正式接口实际读取 512 条（战术主线 4 条、独立局面 9 条），检查完整尾部并重新哈希三份课程文件，见 `evidence/recorded-all-sources-search-inputs-full-v1.json`。生产约 672 秒、读回约 692 秒、正式读取约 51 秒，均未加载模型或特征缓存。来源分割未证明相对先前训练独立；现有课程及已排队的首批搜索仍使用各自冻结输入。合格新主线蒸馏标签及已训练轮次仍为零。

公开战术局面库 [xiangqi-pwa-offline](https://github.com/dffge552/xiangqi-pwa-offline) 的固定版本 `3ff21f4502a03f30bb0df55db6f3814ceeb989f5` 已全量导入：9990 条源记录保留 9683 个原生局面（训练／验证／测试 7754／966／963），306 条源内或先前原谱历史／颜色重复、1 条拒收。独立入口重算全部原文件身份，重建七份原谱的历史颜色轨道，重新解析全部局面并逐字段匹配保留、重复及拒收，见 `evidence/public-tactical-positions-native-import-v1.json`。`tactical_positions` 保留原始 FEN 和缺失计数状态，规范化局面只有零步记录；即便棋盘等于开局，此前历史也标为缺失，源解答不作为走法或讲解标签。来源许可声明与发布边界见 `THIRD_PARTY_NOTICES.md`。

`data/tactical-position-search-inputs-full-v1` 的 7754 条独立局面候选由全部八份来源共同隔离，排除根、记录未来和颜色保留范围；独立入口重算全部入选条件、逐字段匹配全部候选并再次原生检查。正式读取接口实际选择 512 条，检查完整尾部及三份课程文件 SHA256，见 `evidence/tactical-position-search-inputs-full-v1.json`。红／黑为 7750／4，保持源方向，没有新镜像；阶段桶只是零步局面的接口字段，不代表真实对局阶段。这些是候选而非合格标签，没有搜索产出率、教师生成或模型收益。旧候选与已排队首批搜索不变，新旧池不直接拼接；后续联合搜索须用统一的八来源保留范围。

导入需要已完成并哈希绑定的固定版本采集目录；输出目录必须新建。独立局面由 `recorded_tactical_position` 标识，搜索候选仍列出全部来源供隔离，并使用 `--candidate-kinds recorded_tactical_position --min-ply 0 --per-game 1`。长批次用 tmux 和 `freeze_run.py`：

```bash
python scripts/freeze_run.py --output runs/tactical-positions-new/source-run -- \
  python -m xqgeneral.tactical_positions \
  --source data/sources/public-tactical-xiangqi-pwa-v1 \
  --acquisition runs/public-tactical-source-acquisition-v1 \
  --previous data/recorded-games-ccpd-v3 data/recent-recorded-games-v2 \
    data/recorded-games-additional-v1 data/modern-recorded-games-v2 \
    data/playstrategy-portable-real-import-v1-parallel \
    data/playstrategy-public-users-native-import-v1 data/recorded-tactical-lines-ccpd-v1 \
  --output data/tactical-positions-new --seed 20261051
```

`search_pilot` 已排在现有干净 SFT 能力验证队列之后，自动执行实际学生搜索、完整 BF16 教师汇总和最终隔离收取。它核对被评测的真实 SFT 权重与搜索父模型相同；等待时不加载学生或教师。当前 `configs/search-clean-sft-pilot-v2.json` 使用八来源统一候选池，包含实战、战术主线和零步独立局面；预算仍为 512 个未用训练历史、10 万引擎节点、最多五层递归及 `move_eval` 子节点合同。新队列冻结 72 份源码，保留缺失此前历史的边界；旧六来源 v1 等待队列已在执行搜索前退役，原配置、源码和日志保留。实际等待检查、切换读回及全部 531 项 CPU 测试通过，见 `evidence/clean-search-pilot-queue-v2.json`。这没有生成新的搜索标签或证明蒸馏收益。

```bash
python scripts/freeze_run.py --output runs/clean-search-pilot-launch-v2/source-run -- \
  python -u -m xqgeneral.search_pilot \
  --pipeline runs/clean-sft-sequential-launch-v2/pipeline \
  --validation runs/clean-sft-capability-launch-v2/validation \
  --producer-session xqgeneral-clean-sft-capability-launch-v2 \
  --config configs/search-clean-sft-pilot-v2.json \
  --output runs/clean-search-pilot-launch-v2/pilot
```

长任务放在 tmux，记录相同冻结入口。查看 `pilot/state.json`、`commands.jsonl`、`log.txt` 和各阶段清单；现有进程存活时继续观察，退出后可用原入口加 `--resume`。变动源码、配方或输入必须另用新目录；未完成的收取目录保留并拒绝覆盖。搜索零产出时不加载汇总教师或启动空训练。入口统计原始尝试、改进查询、最终结构合格标签和两种产出率；小批次完成不算已训练一轮蒸馏，战略正文质量及后续特征缓存、训练和收益另行验证。

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

干净主线从 `configs/foundation-human-engine-clean-v2.json` 开始：冻结固定版本的预训练 Px0 专家和官方 Qwen 基座，重新初始化桥接及 105 个棋盘词元，不加载旧课程或讲解权重。解码器不再直接读取完整 FEN 或 90 格字典，棋盘状态通过专家特征进入桥接。论文的四门 SC → DC → SF → DF 本身就是桥接训练，必须依次完成，再从最终课程权重做讲解 SFT。棋谱下载、规则出题和教师标注可以先准备；旧字典配方保留作对照。

当前选模配方为 `configs/foundation-human-engine-clean-v3.json`，输出到新目录。它只导入上述干净第一门至第 7168 步的完整原始验收和冻结源码，不修改或恢复旧优化器；按总正确率、最弱题型、较早步数依次选择，继承第 5632 步的原始权重。每 512 步检查所有已引入题型；三次检查无改善时早停，各门至少执行 3072／4096／2048／6144 步以覆盖预热。步数上限、混合比例及训练率保留。早停次数、最少步数及排序方式是本地选择，不是论文公布的参数；99%／98%／98%／97% 与最弱题型 95% 保留为诊断。第一门原始结果为 97.9167%／94.5313%，旧通过字段仍为假；按最佳验证选择完成该门，不宣称达到原门槛。

第二门已在第 8704 步按三次无改善策略结束，并选用第 7168 步最佳权重：582／640 动态题、743／768 静态回放、1325／1408 综合（94.11%）。末次动态题为 590／640（92.19%），完整走法列表 100／128（78.13%），静态回放 726／768（94.53%），综合 1316／1408（93.47%）；相对最佳，动态净增 8 题，静态净减 17 题。十七次同题、原始正确数、原生答案及候选身份已读回，见 `evidence/dynamic-current-seventeenth-raw-validation-v1.json`；更新曲线见 `evidence/dynamic-current-learning-curve-v6.json`，最后完整 512 步平均损失为 0.017760。近期继续训练的动态收益被旧课回落抵消，实际控制器已按原配方交接，没有手动停训或改训练率。

第三门静态未来从第二门第 7168 步最佳权重启动，初始化逐位相同及新优化器已核验。完成课程仍为 2／4。最新独立复核至第 7168 步：未来静态 763／768（99.35%）、静态回放 758／768（98.70%）、动态回放 559／640（87.34%），综合 2080／2176（95.59%）；完整走法枚举为 82／128（64.06%）。全部十四次原始回答、候选身份与同题合同读回，最新 2176 个原生答案及 1907 个历史、768 个未来答案及 738 个未来上下文重新核验，见 `evidence/static-future-fourteen-checks-raw-validation-v1.json`。旧第 4096 步观察因计数条件错误退出，失败记录保留；新观察按配方声明的检查步序核验，没有重启训练或改配方。下一次第 7680 步复核会话为 `xqgeneral-static-future-fifteenth-raw-validation-readback-v1`。未来题高分未证明旧课保留、强棋力或讲解能力；正式 SFT、合格新蒸馏标签及已训练蒸馏轮次仍待前序完成。

走法枚举诊断覆盖 19 个检查点、2432 份回答，并重新核验同批 128 个完整历史及原生答案，见 `evidence/curriculum-move-enumeration-diagnostic-v1.json`。第二门末次 28 道错误均有实际漏招或多招；第三门第 1024 步 41 道错误只有 1 道纯顺序问题，炮题净回落 7 道。按棋子／答案长度分组和逐走法精确率只作诊断，保留原始严格评分；这不证明错误原因或实际棋力。这份历史诊断止于第 1024 步；最新三类诊断及原生验证见下文及上文证明。

CPU 入口 `xqgeneral.diagnose_moves` 现支持指定棋子走法、全盘吃子和全盘将军枚举。第三门第 2048／2560 步三类各 128 题，共 768 份原始回答已真实诊断，重新核验 384 个原生答案并三次扫描／哈希完整验证文件，见 `evidence/enumeration-task-diagnostic-real-v1.json`。吃子题末次 42 个错误中 41 个有集合错误，1 个仅顺序不同；漏掉的吃子招从 31 增至 47，车的漏招从 5 增至 21。多报 27 个招中 25 个非法、2 个合法但不吃子。走法题虽净多对 4 题，漏招仍从 50 增至 60；将军题为 123→120／128。原始严格评分保持不变；这些分组不能证明遗忘原因或实战棋力。默认走法接口在第 2048 步的摘要和分组与旧证明完全一致。32 项新增检查及全部 624 项 CPU 测试通过，原 129 份其他源码／测试绑定和训练配方未变；历史走法诊断保留于 `evidence/move-enumeration-portable-real-v1.json`。

以下命令只分析保存的验证回答，不加载模型；多个文件必须具有相同题目、问法变体和参考答案，按检查点顺序提供。吃子／将军题加 `--task-type captures`／`--task-type checks`（默认 `moves`）；未来动态题另加 `--stage dynamic_future`，并使用新输出目录。结果包含 `analysis.json`、`item-diagnostics.jsonl` 和绑定输入／源码身份的 `manifest.json`。

```bash
python -m xqgeneral.diagnose_moves --validation DATA/validation.jsonl \
  --predictions FIRST_QA/predictions.jsonl SECOND_QA/predictions.jsonl \
  --output runs/move-diagnostics-new
```

原课程训练使用以下冻结入口：

```bash
python scripts/freeze_run.py --output runs/recorded-foundation-clean-launch-v2/source-run -- \
  python -m xqgeneral.validation_curriculum --config configs/foundation-human-engine-clean-v3.json
```

相同冻结入口可加 `--resume` 恢复已完成课程和当前门的精确优化器合同；配置、源码或输入变动须使用新目录。`--prepare-only` 核验已完成／导入课程并准备下一门配置，不启动新的课程训练。课程交接重新核验完整候选历史、实际最佳权重、原始题目与答案、逐门父权重和全部四门完成状态；原始参考目标未达标的事实保留。新讲解／能力队列使用 `configs/explanation-sft-clean-v2.json` 和 `configs/evaluation-clean-sft-v2.json`，数据及预算不变，仍等待四门完成。

旧 v4 继承旧四门权重，首门实际更新 210 步后按用户要求停止，未到首次问答验收；原数据、源码、特征和日志保留。其 1155116 条规则题的完整历史、未来隔离、特征及原生答案抽样已再次读回，见 `evidence/foundation-curriculum-v4-training-readback.json`。这些引擎来源题将与经过规则检查的多样真实棋谱合并，合并后重新预检，不改写旧实验输入。

训练使用四卡全局 256／微批量 4，四门上限 50000／60000／30000／100000 步，每 512 步对已引入课程的每类题检查 128 条原回答和三种提问方式；实际步数按上述最佳验证早停策略决定。各门重置优化器并回放旧门数据，独立测试不选模。原第一门训练及验收保留在 `runs/curriculum-human-engine-clean-v2/static_current/`；新目录延续其干净起点与最佳权重。

原始问答门槛是本项目另加的验收策略：四门总正确率分别为 99%／98%／98%／97%，各题型至少 95%，不是论文公布的早停规则，也未经过系统校准。[论文第 3.2 节及附录 A.2](https://arxiv.org/html/2610.03695v1) 说明按验证集早停并继承各阶段最佳检查点，没有公开这组门槛或逐题型通关线。表 10 的 99.97%／98.97%／99.97%／96.07% 是当前静态／当前动态／未来静态／未来动态的最终测试结果，不能当作预设验收线。当前实验配置与历次原始结果保留；如调整策略，须记录新合同与依据。

四卡无棋盘字典的新初始化、映射缓存及逐行数据读取，连续四步与二加二恢复已经逐位核验，见 `evidence/four-gpu-clean-latent-streamed-resume-v2.json`。该配置使用 `ddp_find_unused_parameters: true` 保持同卡恢复的归约合同；切换这个设置或 `feature_cache_mmap` 不能恢复旧优化器。逐行读取保留全部源字段、文件顺序与随机抽样结果。可用以下命令复查真实训练输出；这是执行检查，未完成课程：

```bash
.venv/bin/python scripts/verify_exact_resume.py \
  --continuous runs/four-gpu-clean-latent-resume-v2/continuous \
  --resumed runs/four-gpu-clean-latent-resume-v2/resumed
```

公开平台导入入口 `xqgeneral.platform_games` 接受已完成的 PlayStrategy 原始 NDJSON 采集清单。清单的 `kind` 为 `public_platform_recorded_game_acquisition`，`verification.captures` 含每份原文件的 `path`、公开 `url`、`sha256`、`bytes`、非空记录数 `records` 和带时区的 `captured_utc`；文件同时绑定在 `outputs`。采集保存原响应字节，按[官方 API](https://playstrategy.org/api)串行请求；清单及数据不随代码发布。导入核对原 JSON／PGN 主线、结果、登记棋手及 BOT 标签、UTC 日期和原生完整历史，明确将平台 1—10 横线转成项目 0—9 坐标。默认至少 20 个半回合、仅标准初始局面及双侧登记账号；匿名 AI、未结束及合同不合记录保留隔离原因，站点文字不作神经标签。

```bash
python -m xqgeneral.platform_games \
  --acquisition data/sources/playstrategy-public-users-v1/manifest.json \
  --output data/playstrategy-new-import --workers 8 \
  --previous data/recorded-games-ccpd-v3 \
  --previous data/recent-recorded-games-v2 \
  --previous data/recorded-games-additional-v1 \
  --previous data/modern-recorded-games-v2
```

真实 37 条试导入的串行／八 CPU 文件逐字节相同，18 局／1082 个半回合保留、19 条隔离，并绑定此前第二入口完整复演，见 `evidence/playstrategy-recorded-games-portable-real-readback-v1.json`。随后从 16 个公开账号捕获 3784 条：3163 条原生通过、621 条隔离、174 条重复，新增 2989 局／165010 个半回合；平台声明的人类／人机为 2251／738 局，394 个参赛者标签。第二入口已复演全部通过历史、复现全部隔离原因并匹配重复的首份来源，见 `evidence/playstrategy-public-users-independent-readback-v1.json`。六来源重新按规范化身份计数为 107678 局、8305866 个半回合、约 866MB；训练的人类／公开实战候选 85530 局，论文式取样上限 513180 个位置、两个完整轮次，见 `evidence/recorded-dataset-paper-capacity-v3.json`。平台声明不证明真实身份或未借助软件；当前课程输入和新 SFT 状态未改变。

真实棋谱导入先固定来源和许可证，再逐着检查完整历史，按规范化棋局身份预先划分，之后才生成规则题。默认保留人类、电脑及人机来源类别，类别和棋手名称都是源记录声明；不把实战走法当作最优着法，不使用网站原讲解冒充神经标注。CCPD 全库已完成 22628 个独立有效棋局、1867475 个半回合，其中电脑／人机为 22／21 局，见 `evidence/recorded-games-ccpd-native-import-v3.json`；首批 512 局保留作历史子集。近期另完成 93 局、6989 个半回合、2019—2023 年的 18 个参赛者名称，见 `evidence/recent-recorded-games-native-import-v2.json`。训练抽样限制双方棋手及来源赛事组，并覆盖年代、来源和胜负；赛事组是元数据启发式，镜像及改写不增加真实棋局数。

当前课程使用的旧两来源去重为 22720 局、约 187 万个半回合；317 万条规则题合计约 20GB，特征缓存另约 251GB。规则题、独立局面、棋局及讲解样例须分别计数；不能用镜像、改写或重复训练宣称论文同级容量，旧容量审计见 `evidence/recorded-dataset-paper-capacity.json`。实际第 256 步参数、优化器及四卡随机状态读回见 `evidence/recorded-foundation-clean-first-optimizer.json`。第一门同一组 768 道原始验证题，第 512／1024／1536／2048／2560／3072／3584／4096／4608／5120 步正确率为 51.8%／54.0%／61.2%／67.4%／74.9%／81.1%／90.5%／92.8%／94.8%／96.7%，十次均未达到当时的本地门槛。原始输出、原生答案及检查点合同均已独立读回，见 `evidence/recorded-foundation-clean-tenth-raw-gate.json`；这是截至第 5120 步的历史快照，当前选择与训练阶段见 [STATUS.md](STATUS.md)。

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

CCPD 的中局、残局、全盘战术及杀法目录另由 `tactical_games` 导入，允许源谱缺少棋手姓名，保留原始 FEN、主线和来源别名；未提供的片段前历史明确标为缺失。原棋谱主线只做规则核验，不作为最优走法或神经讲解标签。固定版本共 3614 个文件、2818 份不同内容；对六个已有来源去重后保留 2259 条主线、68565 个半回合，来源分割为 1805／236／218，另有 505 条重复和 54 份隔离内容。1988 条未提供给定起点之前的历史，349 条缺少一方或双方棋手字段。全部不同内容已由独立入口重新提取、原生复演，并逐字段匹配保留、重复分割及隔离原因；71 份执行源码及 2930 份产物／源码绑定重新哈希。见 `evidence/recorded-tactical-lines-ccpd-native-import-v1.json`；导入里程碑的全部 479 项 CPU 测试通过。不同主线只有 1929 个起始几何局面，22 个在来源分割之间重复，430 条训练主线不超过 12 半回合。来源分割不是相对已有训练集独立的新测试集；已完成的独立战术候选及全局隔离用法见上方搜索候选部分，不能直接追加到旧候选池。执行入口为：

```bash
python scripts/freeze_run.py --output runs/tactical-import/source-run -- \
  python -m xqgeneral.tactical_games \
  --source data/sources/ccpd-v1 --revision 368a47a947773dd8692c026e286dd19b6277b993 \
  --previous data/recorded-games-ccpd-v3 data/recent-recorded-games-v2 \
             data/recorded-games-additional-v1 data/modern-recorded-games-v2 \
             data/playstrategy-portable-real-import-v1-parallel \
             data/playstrategy-public-users-native-import-v1 \
  --output data/tactical-import --workers 8 --seed 20261051
```

另有两份公开 PGN 档案已固定下载版本和实际字节，清点为 141514 个原始棋局头，其中一份比声明少 42 个；这不是去重后的有效棋局数，见 `evidence/additional-recorded-source-acquisition-v1.json`。`bundled_games` 直接读取压缩成员、保留记录序号和原字节哈希，逐着检查完整历史，隔离非法／终局后着法，并对已有棋局去重和保留不同来源署名。来源身份及许可证独立记录，不继承 CCPD 的声明。八 CPU 全量生产已完成：81711 个新主线、6246714 个半回合、46826 条重复及 12977 条隔离，见 `evidence/additional-recorded-games-native-import-v1.json`；全部 141514 条原档案候选的第二入口原生复演已完成，逐条匹配规范化保留主线、分割、原字节／序号、重复署名和隔离原因；完成后的 15 份声明产物及 53 份实际冻结源码重新核验，见 `evidence/additional-recorded-games-independent-readback-v1.json`。第二入口沿用固定的分段／原生解析合同，不代表另一套规则实现或棋手真实性认证。此前各 64 局的实际串行／四 CPU 结果仍只是有界检查。新棋谱尚未改变当前课程输入。使用固定源码、tmux 及新输出目录运行：

```bash
python scripts/freeze_run.py --output runs/new-bundled-import/source-run -- \
  python -m xqgeneral.bundled_games \
  --acquisition runs/additional-recorded-source-acquisition-v1/manifest.json \
  --previous-data data/recorded-games-ccpd-v3 data/recent-recorded-games-v2 \
  --workers 8 --seed 20261051 --output data/new-bundled-recorded-games \
  --public-evidence evidence/new-bundled-recorded-games.json
```

省略 `--limit-records` 才处理全部候选；该选项只供有界执行检查。中断后保留部分产物，在新目录用原固定源码重放，不覆盖旧输出。记录数、重复、署名差异和规则拒收原因以完成清单为准；原始记录及档案不随代码发布。

另一次近期来源收集检查了 910 个实际候选页面，原生导入保留 240 局、18641 个半回合、245 个参赛者名称，源记录年份为 2024—2026。原批次另有 80 局的日期写为 2029 年、赛事却写为 2026 年；这些记录按原文隔离，没有代改日期。独立离线重放、全部保留历史与隔离原因读回已完成；一条原已隔离的重复页面也重归日期异常。见 `evidence/modern-recorded-games-native-import-v2.json`、`evidence/modern-recorded-games-native-readback-v3.json`、`evidence/recorded-source-future-date-audit-v1.json`。与完整档案及旧池去重后，这 240 局无额外重复；四来源共 104671 局、8139774 个半回合、约 846MB 棋谱 JSONL，见 `evidence/recorded-dataset-paper-capacity-v2.json`。训练分割的人类／公开实战候选为 83726 局，按论文每局最多六个位置，宽松上限为 502356 个位置，尚不足其七轮互异人类棋局预算；根／变化筛选和真实性检查还会减小容量，新增档案的独立全量复演现已完成，原容量审计的完成时快照保留。新棋谱尚未加入当前课程。重新解析已固定页面时可显式限制来源年份：

```bash
python scripts/freeze_run.py --output runs/new-recent-reparse/source-run -- \
  python -m xqgeneral.collect_recorded \
  --cached-data data/modern-recorded-games-v1 --max-games 1024 \
  --min-year 2024 --max-year 2026 --seed 20261051 \
  --output data/new-recent-recorded-games \
  --public-evidence evidence/new-recent-recorded-games.json
```

`--max-year` 是源记录年份的可选上限，默认不设上限以保留历史运行方式。检查使用棋谱日期，未提供日期时才按解析合同使用赛事年份；不会用赛事名称改写已有日期。`--cached-data` 校验完成清单和页面哈希，不发新请求，旧页面、失败读回与原始批次保留。上面示例使用已完成的本地缓存，新下载仍须提供公开索引或棋局 URL。

规则课程从实际记录的 1—8 步后续变化出题，保留完整历史和原棋局身份。先隔离此前所有训练／保留局面及未来分支，再合并旧引擎题，新文件保留旧文件的逐字节前缀。完整生产已从选择的 6144 局隔离出 45834 个原始根，新增 2016696 条规则题；合计 3171812 条，训练／验证／测试为 2831004／173272／167536。见 `evidence/recorded-curriculum-human-engine-full-v1.json`；真实小规模执行证据单独保留。两来源去重共 22720 局，一份主线的棋手／赛事声明有冲突且未入选，不能据合法性认定姓名真实。

```bash
python -m xqgeneral.recorded_curriculum \
  --games data/recorded-games-ccpd-v3 data/recent-recorded-games-v2 \
  --footprints runs/recorded-curriculum-reserved-footprints-v1 \
  --base-data data/research-selfplay-v2 --game-budget 6144 --roots-per-game 8 \
  --participant-cap 256 --event-cap 128 --modern-weight 3 \
  --output data/new-recorded-engine-courses \
  --public-evidence evidence/new-recorded-engine-courses.json
python -m xqgeneral.extend_features \
  --roots data/new-recorded-engine-courses/recorded-roots.jsonl \
  --runtime runs/new-recorded-feature-extension \
  --output data/new-recorded-engine-courses/features-16.pt --gpu-indices 0 1 2 3 \
  --history-workers 16
```

特征扩展核验旧专家权重及编码器身份、原缓存哈希、分片键互斥、旧键及数值逐位保留，颜色镜像按原生完整着法回放以保持正确历史键；课程文件不会被缓存程序改写。真实四 GPU 小样本及专家重新计算见 `evidence/recorded-expert-cache-extension-real-fixture-v2.json`。CPU 历史准备保序、有界并行，真实 183 键缓存与串行结果逐位相同，见 `evidence/recorded-native-history-parallel-feature-readback.json`。

全量预检逐条核验数据哈希、特征键及任务合同，编码全部训练／验证及两种验收改写，原生复演所有唯一根历史、抽样重算两来源全部题型答案，并重新检查全部未来隔离和旧引擎文件字节前缀。四条历史独立测试终局探针保留，不能混入训练／验证。完整 3171812 条课程已完成预检及独立读回，见 `evidence/recorded-foundation-full-training-ready.json`、`evidence/recorded-foundation-full-training-readback.json`；早期 3963 条执行检查另保留。第一门正式模型及首次 20 次更新读回见 `evidence/recorded-foundation-clean-startup-twenty-updates.json`，这尚不是课程完成或棋力证明。使用新输出目录执行：

```bash
python -m xqgeneral.foundation_preflight \
  --config configs/foundation-human-engine-clean-v2.json \
  --output runs/new-recorded-foundation-preflight --workers 16 \
  --answer-samples-per-task 32 \
  --public-evidence evidence/new-recorded-foundation-training-ready.json
```

完成预检后，可用独立读回入口重新核验全部预检产物、固定基座、原始棋谱前缀和最大未来、完整镜像历史键、旧／新保留足迹及旧专家缓存逐位保留。输入须是已完成的预检及固定基座身份清单；它只读取数据，不训练或选择模型。真实 3963 条执行及读回见 `evidence/recorded-foundation-source-readback-real-v1.json`。

```bash
python -m xqgeneral.foundation_readback \
  --config configs/foundation-human-engine-clean-v2.json \
  --preflight runs/new-recorded-foundation-preflight \
  --base-proof runs/recorded-foundation-qwen-base-identity-v1/manifest.json \
  --output runs/new-recorded-foundation-readback \
  --public-evidence evidence/new-recorded-foundation-readback.json
```

长任务须在 tmux 中使用 `scripts/freeze_run.py` 固定源码。完整课程准备命令在 `runs/recorded-curriculum-full-v1/plan.json`；缓存、预检、独立读回和正式训练等待入口分别在 `runs/recorded-expert-cache-full-v3/`、`runs/recorded-foundation-full-preflight-v3/`、`runs/recorded-foundation-full-readback-v2/`、`runs/recorded-foundation-clean-launch-v1/` 的 `plan.json`。各目录 `log.txt` 记录实际阶段；全部前置检查及 CI 里程碑通过后，等待入口才自动启动新四门课程。中断保留输出、按固定源码及合同续跑；改变输入或配置另建实验。完整棋谱来源与发布边界见 `THIRD_PARTY_NOTICES.md`。

实战初始讲解查询从 `recorded-roots.jsonl` 选择原始根，保留棋局分割、双方平衡及棋手／赛事上限，不把整份 20GB 规则题载入内存。3744 个候选根产生 3740 个事实查询，4 个无评分首选拒收已复现；隔离全部现有课程和旧讲解的完整变化后，选定 2496 个原始查询（2048／192／256），每局一个。串行原流程与八 CPU 入口的查询及三片文件逐字节一致；最终独立读回复用已完成的完整历史证明，重新计算合法变化和颜色对应几何足迹，跨分割重叠为零。见 `evidence/recorded-coach-original-query-isolation-v1.json`、`evidence/recorded-coach-selected-query-independent-readback-v2.json`；实际 40 个根／查询的可复用入口检查另存。三个已授权的 Astra Low 代理已完成全部原始正文及相互审查：2496 条中原文接受 2363 条、拒收 133 条，见 `evidence/recorded-coach-complete-original-cross-review-v1.json`。全部 133 条拒收现已真实修订并由不同作者接受，其中 3 条经过第二轮；正式收集 2496 条新标签（2048／192／256），无镜像派生，棋局及根／未来几何足迹跨分割重叠均为零。独立读回逐字段核验原查询、结构答案、最终正文及实际修订链，见 `evidence/recorded-coach-complete-reviewed-labels-v1.json`；完整原生历史检查复用已完成生产与原查询证明。作者计划及原始身份读回保留在 `runs/astra-recorded-coach-authoring-v1/`。该批仍是初始教师材料，不代表人工评分或学生收益，新 SFT 须等四门课程按验证策略完成。使用新输出目录按以下顺序准备；长任务放在 tmux 中并由 `scripts/freeze_run.py` 固定源码：

```bash
python -m xqgeneral.recorded_coach curate \
  --data data/research-human-engine-v1 \
  --source-readback runs/recorded-foundation-full-readback-v2/manifest.json \
  --prior-labels data/astra-explanations-full-v7 --output data/new-coach-roots \
  --train-roots 3072 --validation-roots 288 --test-roots 384 --workers 8
python -m xqgeneral.prepare_explanations --data data/new-coach-roots \
  --output data/new-coach-candidates --train-roots 3072 \
  --validation-roots 288 --test-roots 384 --workers 8 --nodes 100000
python -m xqgeneral.recorded_coach_readback \
  --roots data/new-coach-roots --queries data/new-coach-candidates \
  --output runs/new-coach-native-readback --workers 8
python -m xqgeneral.recorded_coach footprints \
  --data data/research-human-engine-v1 \
  --source-readback runs/recorded-foundation-full-readback-v2/manifest.json \
  --prior-labels data/astra-explanations-full-v7 \
  --output runs/new-coach-footprints --workers 8
python -m xqgeneral.recorded_coach isolate \
  --queries data/new-coach-candidates \
  --query-readback runs/new-coach-native-readback/manifest.json \
  --footprints runs/new-coach-footprints/manifest.json \
  --output data/new-coach-seed --workers 8
python -m xqgeneral.recorded_coach_isolation_readback \
  --selected data/new-coach-seed --candidates data/new-coach-candidates \
  --native-readback runs/new-coach-native-readback/manifest.json \
  --footprints runs/new-coach-footprints/manifest.json \
  --output runs/new-coach-isolation-readback --workers 8
```

候选原生审计完成不等于跨分割隔离通过；若配额不足则拒收，不降低隔离标准。颜色对应只作几何保留，未生成镜像标注或重算评分。正文只依据学生可见局面和核验候选，不借未认证的棋手／赛事／实际后续作事实，不把有限搜索分数转换成人类真实胜率。标注仍须逐条语义复核，且四门新课程全部达标后才允许讲解 SFT。

可复用足迹入口另已实际处理全部 3171812 条课程及 9158 条旧讲解，与原流程的完整足迹文件逐字节一致，见 `evidence/recorded-coach-portable-full-footprints-v1.json`。作者完成一片后，可用 `recorded_coach_prose prepare --shard N` 冻结该片的复核输入；省略 `--shard` 要求三片全部完成。它核验作者进度、原查询及隔离证明，加入变化每步的前后棋盘，隐藏未认证的姓名／赛事／日期和实际棋谱后续。4 条真实标注的单／四 CPU 复核包对照及实际拒收检查见 `evidence/recorded-coach-prose-real-preparation-v2.json`，不能据此宣称全批已复核。

```bash
python -m xqgeneral.recorded_coach_prose prepare \
  --queries data/astra-recorded-coach-seed-portable-v1 \
  --isolation runs/recorded-coach-selected-query-independent-readback-v2/readback/manifest.json \
  --author-plan runs/astra-recorded-coach-authoring-v1/plan.json \
  --output data/new-recorded-coach-prose-review --workers 8
python -m xqgeneral.recorded_coach_prose resolve \
  --prepared data/new-recorded-coach-prose-review \
  --reviews data/new-recorded-coach-prose-review/review-0.json \
    data/new-recorded-coach-prose-review/review-1.json \
    data/new-recorded-coach-prose-review/review-2.json \
  --output data/new-recorded-coach-prose-resolved
python -m xqgeneral.recorded_coach_prose collect \
  --queries data/astra-recorded-coach-seed-portable-v1 \
  --isolation runs/recorded-coach-selected-query-independent-readback-v2/readback/manifest.json \
  --author-plan runs/astra-recorded-coach-authoring-v1/plan.json \
  --reviews data/new-recorded-coach-prose-resolved \
  --output data/new-reviewed-recorded-coach-labels --workers 8
```

审查 JSON 记录 `reviewer_model`、`reasoning_effort`、`backend`、`reviewer_agent`、`input_packet_sha256`、`human_rating: false`、`source_labels_modified: false` 及 `results`。每条决定包含 `id`、`reviewed_annotation_sha256`、`verdict`、具体中文 `reason` 和 `issues`；接受项不能残留问题，逐条审查者不能是该正文作者。继续审查可以保留多个决定文件，但同一正文的接受／拒收冲突不能被忽略。

拒收后由已授权教师真实修订，另存与原包棋盘／事实完全相同的修订包，仅替换 `teacher_annotation`、其哈希和真实作者身份。新标注含 `corrected_from_annotation_sha256` 与 `correction_review_sha256`，绑定准确被拒原文及拒收文件；修订仍须由另一位审查者接受。把修订包加入 `resolve --repair-packets FILE...`，并把原决定及修订决定都传入 `--reviews`。收集会重新核验完整接受链及整个原查询批次，保留原始分割和全部未来变化。任何未完成、未复核或被篡改的输入均拒收；这些入口只准备数据，SFT 仍须等待四门课程达标。

`prepare-repairs` 从已完成复核包和固定的实际审查 JSON 准备拒收项，支持多个 `--prepared` 分片目录。它逐字节保存拒收快照、重新检查原生历史与事实，并按原作者输出修订查询及计划；没有生成或接受新正文。输入须已完成或先固定快照，运行中改变会拒收。修订祖先使用输出快照的哈希；后续 `resolve --reviews` 也须包含这些快照，并保留后续完整决定与修订接受意见。

```bash
python -m xqgeneral.recorded_coach_prose prepare-repairs \
  --prepared data/new-recorded-coach-prose-review \
  --reviews data/new-recorded-coach-prose-review/review-0.json \
    data/new-recorded-coach-prose-review/review-1.json \
    data/new-recorded-coach-prose-review/review-2.json \
  --output data/new-recorded-coach-repair-queries --workers 8
```

实际固定前缀的 856 条神经审查决定包含 52 条拒收（训练 40、测试 12）；全部修订查询已通过串行／八 CPU CLI 和原生准备结果对照，仅输出引用路径不同，见 `evidence/recorded-coach-repair-portable-real-v1.json`。这证明拒收准备与祖先保留，不证明完成修订、全文语义接受或学生收益。

全部 2496 条原文及三个完整复核包已准备完成，核验原查询身份、顺序和精确正文，并由准备入口逐条检查完整历史、结构答案变化及正文合同；见 `evidence/recorded-coach-full-authoring-review-preparation-v1.json`。三位 Astra Low 审查者分别检查另一位作者的 832 条，分配及进度入口在 `runs/astra-recorded-coach-prose-review-v1/`；全部拒收已真实修订并由不同作者接受，最终训练标签收集已完成，见 `evidence/recorded-coach-complete-reviewed-labels-v1.json`。原文、拒收意见和后续修订分别保留。

这批 2496 个原始查询及 4992 个原生颜色对应历史键、9158 条旧讲解历史键已全部覆盖在当前课程的专家缓存中，见 `evidence/recorded-coach-expert-feature-coverage-v1.json`；无需单独运行 GPU 特征扩充或复制 251GB 缓存。该盘点检查缓存头并绑定已完成生产证明，没有再次提取或验证全部特征值。实际收集标签的合并及 CPU 预检现已完成，见下方；完成四门验收后才能准备正式 SFT 配方。此前固定 1672 条前缀的机械检查见 `evidence/recorded-coach-authored-prefix-mechanical-audit-v1.json`，不能替代完整语义接受。

`reviewed_explanations` 合并已完成的讲解标签集，原始训练／验证／测试文件分别逐字节拼接，保留原字段、分割和已有镜像。它重验完整过去、原生终局及全部答案／实战变化、走法事实和颜色对应隔离；新增实战标签要求内嵌接受身份，旧训练项的独立接受池可用 `--legacy-review-data` 逐字段绑定。此入口适用于已收集标签；`merge_teacher` 仍用于旧的原始查询／标注批次。

```bash
python scripts/freeze_run.py --output runs/new-reviewed-explanation-prep/source-run -- \
  python -m xqgeneral.reviewed_explanations \
  --inputs data/astra-explanations-full-v7 data/astra-recorded-coach-teacher-reviewed-v1 \
  --legacy-review-data data/astra-selfplay-explanation-replay-v1 \
  --output data/new-reviewed-explanations --workers 8
python scripts/freeze_run.py --output runs/new-reviewed-explanation-prep/source-run -- \
  python -m xqgeneral.explanation_preflight --data data/new-reviewed-explanations \
  --foundation-config configs/foundation-human-engine-clean-v2.json \
  --foundation-preflight runs/recorded-foundation-full-preflight-v3/preflight/manifest.json \
  --output runs/new-reviewed-explanation-preflight --max-tokens 1024 --workers 8
```

实际候选集 `data/astra-explanations-clean-v1` 为 10758／384／512 条，共 11654 个不同完整历史键；包含已有 3363 条颜色派生项，新增独立讲解仍为 2496 条。旧 384 条训练标签与已完成接受池逐字段相同；旧 448 条保留集的内容及原审查状态保留，没有补做或宣称新的语义接受。全部训练／验证共 11142 条按主线词元器及 105 个棋盘词元编码，最长 649／625、上限 1024，无截断；测试答案不参与词元编码。全部标签键存在现有 169904 键缓存中。

第二入口逐字节匹配全部原片段，复演全部过去、重算事实及所有合法未来／颜色几何足迹和逐项词元长度；61 份来源／生产产物及 60 份实际冻结源码核验，见 `evidence/clean-explanation-data-independent-preflight-v1.json`。第二入口复用生产阶段完整终局证明；缓存检查读取键、形状和精度，绑定已完成全量预检，没有重哈希全部 251GB 或重算特征值。272 项 CPU 测试通过。当前四卡课程输入不变，预检不加载模型权重，新 SFT 仍须等待四门课程按验证策略完成。

四门新课程全部完成后，先将最终原始验收候选导出为讲解训练可读取的初始化目录。`foundation_handoff` 要求完整四门清单，核验干净起点、逐门继承、实际步数、全部声明字节和完整桥接／棋盘词元状态；重新计算四次已引入课程的原始验收，并复演全部抽中历史、未来终局及原生答案。相同大文件只哈希一次，验证前后检查文件身份；只读取验证答案。使用新的输出目录，在 tmux 中固定源码执行：

```bash
python scripts/freeze_run.py --output runs/new-foundation-handoff/source-run -- \
  python -m xqgeneral.foundation_handoff \
  --curriculum runs/curriculum-human-engine-clean-v3 \
  --output runs/new-foundation-handoff/selected
```

完成目录包含与最终候选同字节的硬链接 `adapter.pt`、原配置 `config.json` 和完成清单 `manifest.json`。正式 SFT 配方应保留其模型版本、桥接、专家层与潜在棋盘输入合同，并以 `selected/adapter.pt` 初始化；`sft` 拒收不完整交接、变动配置及架构覆盖。该入口不加载基座权重或启动 SFT。31 项新增 CPU 检查覆盖受控四门合同的导出／消费及错误拒收；真实第 4608 步部分模型的 258 个张量、139920416 个有限 FP32 参数和完整形状也已核验，实际四门未完成状态确实拒收且未建立输出，见 `evidence/foundation-handoff-real-partial-guard-v1.json`。完整真实四门导出及其讲解收益仍未执行。

新主线初始讲解配方为 `configs/explanation-sft-clean-v2.json`，仅切换到新的验证选模课程，预算保留 v1 配方；强制要求上述完成四门的干净交接，拒收旧课程或旧 SFT 完成目录。它只采样现有已复核讲解，保留训练／验证／测试分割，以四卡全局 16／微批量 1 训练完整解码器；桥接／解码器／棋盘词元学习率分别为 `1e-5`／`1e-6`／`1e-6`，预热 5%，之后保持恒定，权重衰减 0.1。四轮、批量及学习率参考[论文附录 B.2 表 12](https://arxiv.org/html/2610.03695v1)，教师、棋种、长度、验证规模和采样方法为本地适配。10758 条训练项对应 2690 次更新、43040 次样本呈现；随机有放回采样不保证每行恰好出现四次。全部 384 条验证标签选模，512 条测试标签不选模。

CPU 预算预检已逐项绑定实际数据、词元预检及原独立读回，并核对第 5120 步部分模型的架构参考，见 `evidence/clean-explanation-sft-budget-preflight-v2.json`。该部分模型不能作为正式讲解起点；实际未完成四门的导出仍拒收。缓存身份复用已完成全量预检，没有再次哈希全部缓存。四卡完整解码器的真实显存、更新及精确续跑检查尚未执行，新 SFT 尚未启动。完成交接后，可以先固定源码核对正式配置：

```bash
python scripts/freeze_run.py --output runs/new-clean-sft/source-run -- \
  python -m xqgeneral.sft --recipe configs/explanation-sft-clean-v2.json \
  --init runs/new-foundation-handoff/selected/adapter.pt \
  --output runs/new-clean-sft/training --prepare-only
```

`--prepare-only` 保留相同父模型守卫，写入正式配置而不启动训练；变动的已准备配置不能覆盖。完成所需四卡 GPU 检查后，在 tmux 中用同一冻结源码、同一输出去掉该选项启动。SFT 控制器按 `ddp_world_size` 启动对应进程数，单卡仍直接调用训练入口；完整优化器续跑指向同一 `latest.pt`。25 项新增 CPU 检查覆盖配置准备、单／多卡启动、续跑及旧父模型拒收，全套 328 项 CPU 测试通过，未执行 GPU 模型训练。独立第二入口另核验预算预检及第十次原始验收的 36 份声明产物、61／60 份实际冻结源码和公开证据原字节，并重新计数实际训练项与更新预算。

`sft_pipeline` 将上述阶段顺序执行：先等待指定的现有课程进程写出四门完成清单，再导出最终权重；四卡完整解码器检查通过后才启动初始讲解训练。最长的实际 24 条训练／12 条验证标签用于连续四步和二加二续跑检查，测试答案不参与。预检验证完整 FP32 参数、Adam 状态、随机状态和逐步日志逐位一致，检查各参数组真实更新以及全部工作进程的显存记录。正式 SFT 完成后，还会读回所选完整解码器的实际参数和输入身份。它不自动重启课程；来源、配方或输入变动时停止并保留输出。长任务在 tmux 中使用以下冻结入口：

```bash
python scripts/freeze_run.py --output runs/clean-sft-sequential-launch-v2/source-run -- \
  python -u -m xqgeneral.sft_pipeline \
  --curriculum runs/curriculum-human-engine-clean-v3 \
  --producer-session xqgeneral-recorded-foundation-clean-v3 \
  --recipe configs/explanation-sft-clean-v2.json \
  --token-preflight runs/reviewed-explanation-clean-data-v1/preflight/manifest.json \
  --output runs/clean-sft-sequential-launch-v2/pipeline
```

检查 `pipeline/state.json`、各阶段的 `manifest.json` 和日志。仍存活的同一 tmux 任务继续观察；确认其已退出后，用同一冻结入口及原参数加 `--resume` 续跑，不能覆盖已有输出。单独执行预检可用 `python -m xqgeneral.sft_preflight --init HANDOFF/adapter.pt --recipe configs/explanation-sft-clean-v2.json --token-preflight runs/reviewed-explanation-clean-data-v1/preflight/manifest.json --output NEW_PREFLIGHT`。真实等待守卫见 `evidence/clean-sft-sequential-real-wait-guard-v1.json`；25 项新增受控 CPU 检查、全部 353 项测试通过。真实四卡完整解码器检查和新 SFT 尚未执行，当前能力进度以 `STATUS.md` 为准。

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

讲解评测使用更高预算、独立执行的皮卡鱼检查推荐走法、变化、评分方向和棋盘事实；中文战略正文仍需单独评审。对弈可从固定开局或保留集棋谱前缀交换红黑，完整保留历史。非法模型走法判负，达到步数上限记为截尾，结果不自动换算 Elo。先用验证集完成开发，最终选择的模型才进入独立测试。

`match_openings` 从已完成课程的验证文件选择不同原始实战棋局的固定前缀，不读取训练文件或使用问答标签、记录未来；默认取前 12 半回合，去重完整历史及最终几何局面。完整源文件重新哈希，实际加载时逐项对照源行并原生复演。测试分割仅用于最终选定模型的独立验收。每个开局、节点预算均交换红黑；对弈记录实际初始 FEN、完整过去、来源分割和开局身份：

```bash
python -m xqgeneral.match_openings --data data/research-human-engine-v1 \
  --split validation --count 64 --plies 12 --output NEW_OPENING_SUITE
python scripts/freeze_run.py --output NEW_MATCH_SOURCE -- \
  python -m xqgeneral.evaluate_games --checkpoint CHECKPOINT.pt \
  --opening-suite NEW_OPENING_SUITE --nodes 100 1000 10000 --output NEW_MATCHES
```

对弈入口还核验讲解主线和分支是否越过完整历史终局。合法首着用于对弈，变化错误保留在逐步核验；非法首着判负。检查点、引擎、专家权重及开局输入在运行中改变时拒绝完成，保留已生成记录。`--expert-weights` 可指定实际 Px0 权重，默认沿用项目路径。每回合原始记录写入 `progress/game-NNN.json`，完成局另存为 `game-NNN.json`。原任务退出后，在同一冻结入口的原命令末尾加 `--resume`；它先核验配置、输入、执行源码、原始回答和完整历史，再复用完成局及最后保存回合。已完成的同一评测只读回核验，不再加载模型或引擎。`metrics.json` 的 `seconds` 只记录最后一次调用耗时。

实际 64 个验证开局来自 64 个不同原始棋局（源声明为 63 个人类实战、1 个人机），来源和原生读回见 `evidence/recorded-match-openings-real-v1.json`。新的 256 局真实引擎 CPU 控制在第二局第三回合保存后中断，续跑复用一局、11 回合并保留原字节和回答前缀；总计 2048 次双方调用与最终回合数一致。再次续读未加载双方或改变产物，515 份输出重新哈希，见 `evidence/match-resume-real-v1.json`；全部 669 项 CPU 测试通过。这些每局八个新增半回合的控制方是脚本，学生棋力与讲解质量仍待正式验收；来源真实性、开局几何未见性及棋手身份未独立证明。

新主线可将能力评测排在已启动的干净 SFT 队列后，在 tmux 中使用以下冻结入口：

```bash
python scripts/freeze_run.py --output runs/clean-sft-capability-launch-v2/source-run -- \
  python -u -m xqgeneral.clean_sft_evaluation \
  --pipeline runs/clean-sft-sequential-launch-v2/pipeline \
  --producer-session xqgeneral-clean-sft-sequential-launch-v2 \
  --config configs/evaluation-clean-sft-v2.json \
  --output runs/clean-sft-capability-launch-v2/validation
```

该配方固定全部 384 条不同完整历史键的验证记录、批量 4 和每局面 100 万节点独立引擎预算；三组均保留原始生成，以正常、清零和错配专家特征进行配对比较。`evaluate_explanations --memory normal|zero|shuffled` 也可单独使用；每个错配批次须至少两条样本，不能用单条尾批充当消融。正常组全部接受有规则事实和独立评分依据的完整 BF16 教师盲评，原始拒收与条件均分分别保留；不把神经评分视为人工评价或棋力证明。

检查 `validation/state.json`、`commands.jsonl`、各组 `raw-predictions.jsonl`／`judged-predictions.jsonl`、盲评接受／拒收及阶段清单。现有任务存活时继续观察；已退出后可用同一冻结入口与原参数加 `--resume`，复用经哈希核验的完整阶段及教师逐条缓存。未完成的原始评测目录保留并拒绝覆盖，此时须用新输出重做该实验。真实等待守卫见 `evidence/clean-sft-capability-real-wait-guard-v1.json`；17 项新增受控检查及全部 370 项 CPU 测试通过。当前入口只在未完成的新 SFT 前等待，实际能力评测和盲评尚未执行。

干净初始讲解模型的完整对弈基线排在上述配对能力评测之后。`configs/matches-clean-sft-baseline-v1.json` 固定已有 64 个保留验证开局、双方颜色和 100／1000／10000 节点预算，共 384 局，每局最多新增 256 半回合。保留原始讲解首着和错误，非法首着判负，截尾单列；GPU 2 与使用 GPU 0 的搜索试批分开。

```bash
python scripts/freeze_run.py --output runs/clean-sft-match-baseline-launch-v1/source-run -- \
  python -u -m xqgeneral.clean_sft_matches \
  --pipeline runs/clean-sft-sequential-launch-v2/pipeline \
  --validation runs/clean-sft-capability-launch-v2/validation \
  --producer-session xqgeneral-clean-sft-capability-launch-v2 \
  --config configs/matches-clean-sft-baseline-v1.json \
  --output runs/clean-sft-match-baseline-launch-v1/baseline
```

该队列核验实际完成的四门后全参数 SFT、同份模型的三组原始验证及开局输入，再启动完整对弈。检查 `baseline/state.json`、`commands.jsonl` 和 `matches/`；原任务退出后，同一冻结命令加 `--resume` 可复用逐回合记录。32 项新增检查及全部 701 项 CPU 测试通过。实际冻结进程已完成输入检查并等待，尚未执行训练学生对弈，见 `evidence/clean-sft-full-match-baseline-real-wait-v1.json`。这份基线用于后续蒸馏比较，最终独立测试仍待最终选模。

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
