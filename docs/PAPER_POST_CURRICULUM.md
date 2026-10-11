# 新四课之后的讲解与评测

这套流程接在[论文对齐四课](PAPER_FOUNDATION_EXPERIMENT.md)之后，只接受该实验实际完成的四门交接权重。课程数据、特征缓存和四课尚未完成；当前已准备讲解索引与配置，后续队列等待四课完成证明。它没有提前加载学生或教师，也没有启动讲解训练。

## 初始讲解训练

[explanation-sft-paper-v1.json](../configs/explanation-sft-paper-v1.json)继承交接模型的预训练版本、全宽桥接、专家层和棋盘词元配置，不从旧讲解权重续训。全解码器与桥接使用 FP32 可训练主参数，回答 token 损失，四 GPU、全局批量 16、微批量 1。桥接／解码器／棋盘词元学习率为 `1e-5`／`1e-6`／`1e-6`，预热 5% 后恒定，权重衰减 0.1。

训练数据沿用 `data/astra-explanations-clean-v1` 的 10758 条训练项和 384 条验证项，包含引擎局面及真实实战讲解；512 条独立测试项不训练、选模或建立训练索引。初始标注教师为用户指定的 GPT-6-Astra Low。训练项继承已完成的语义接受；旧保留集的内容和审查状态保持原样，本次没有新增语义审查。

新配置使用有限遍历，每轮完整呈现全部训练项，共四轮。实际数据对应每轮 673 次更新、共 **2692** 次更新；尾部每轮补两条以满足分布式微批量，四轮共 43032 条有效呈现和八条补齐呈现。`max_steps`／`min_steps` 的 4096 是配方上限，`sft.prepare_config` 按实际完整轮次数编译为 2692；没有第五轮片段或有放回凑数。选模仍使用验证 NLL，之后单独检查原始走法、完整变化和讲解正文。

现有 169904 历史的原缓存覆盖全部讲解键，新课程特征存储保留这份缓存。仍须用**实际新缓存和实际基础预检**重新执行讲解词元预检；旧单文件缓存证明不能代替。完成四课交接后，先运行真实四卡全解码器连续四步／二加二续跑与最长标签显存预检，通过后才启动正式 SFT。短控制和 CPU 预算检查不能证明训练收益。

## 评测与蒸馏

[evaluation-paper-sft-v1.json](../configs/evaluation-paper-sft-v1.json)对同一正式 SFT 权重及全部 384 条验证项分别使用正常、清零和错配专家特征，保留原始回答与规则评分，再以完整 BF16、未量化 Qwen3.8-27B 进行匿名正文辅助评分。学生和教师在 GPU 0 顺序加载；教师评分不当作人工评价，也不自动构成最终验收。

[matches-paper-sft-baseline-v1.json](../configs/matches-paper-sft-baseline-v1.json)接在完整能力验证之后：64 个保留验证开局、交换红黑、Pikafish 100／1000／10000 节点，共 384 局，每局最多新增 256 半回合，使用 GPU 2。非法首着或变化照实计入失败，不由引擎代答；逐回合保存与续跑沿用现有对弈入口。这是开发基线，独立最终棋力和教学验收仍须另做。

[search-paper-sft-pilot-v1.json](../configs/search-paper-sft-pilot-v1.json)只声明后续 512 根搜索试批。新课程完成后须重新核查来源与全部新保留范围的可用容量，再依据实际有效标签产量、能力和对弈分数调整搜索量与蒸馏轮数。当前队列不会自动训练这个试批；不能把已准备搜索局面、配置或教师调用算作完成蒸馏。

## 可复用命令

以下命令使用仓库现有公开入口；路径按自己的完成产物配置，长任务放在 tmux 中。使用同一个 `source-run` 保存的源码执行整个链，变动配置或输入须使用新实验目录。先确认对应生产会话仍存活或已有完成证明，避免重复启动。

1. 制作训练和验证索引。独立测试不建立训练索引。

```bash
.venv/bin/python scripts/freeze_run.py --output runs/paper-followthrough/source-run -- \
  .venv/bin/python -m xqgeneral.training_index \
  --source data/astra-explanations-clean-v1/train.jsonl --split train \
  --output data/astra-explanations-paper-index-v1/train
.venv/bin/python scripts/freeze_run.py --output runs/paper-followthrough/source-run -- \
  .venv/bin/python -m xqgeneral.training_index \
  --source data/astra-explanations-clean-v1/validation.jsonl --split validation \
  --output data/astra-explanations-paper-index-v1/validation
```

2. 新缓存及基础预检完成后，使用实际基础配置和预检路径复查讲解词元。配置里的相对路径以项目根目录为准。

```bash
.venv/bin/python scripts/freeze_run.py --output runs/paper-followthrough/source-run -- \
  .venv/bin/python -m xqgeneral.explanation_preflight \
  --data data/astra-explanations-clean-v1 \
  --foundation-config "$XQ_FOUNDATION_CONFIG" \
  --foundation-preflight "$XQ_FOUNDATION_PREFLIGHT" \
  --max-tokens 1024 --workers 8 --output runs/paper-sft-v1/token-preflight
```

3. 四课全部完成且四卡有足够资源后，执行交接、真实全解码器预检和初始 SFT；随后才做能力验证和对弈。

```bash
.venv/bin/python scripts/freeze_run.py --output runs/paper-followthrough/source-run -- \
  .venv/bin/python -u -m xqgeneral.sft_pipeline \
  --curriculum runs/curriculum-paper-v1 \
  --producer-session xqgeneral-paper-curriculum-preparation-v3 \
  --recipe configs/explanation-sft-paper-v1.json \
  --token-preflight runs/paper-sft-v1/token-preflight/manifest.json \
  --output runs/paper-sft-v1/pipeline
.venv/bin/python scripts/freeze_run.py --output runs/paper-followthrough/source-run -- \
  .venv/bin/python -u -m xqgeneral.clean_sft_evaluation \
  --pipeline runs/paper-sft-v1/pipeline --producer-session xqgeneral-paper-followthrough-v1 \
  --config configs/evaluation-paper-sft-v1.json --output runs/paper-sft-v1/capability-validation
.venv/bin/python scripts/freeze_run.py --output runs/paper-followthrough/source-run -- \
  .venv/bin/python -u -m xqgeneral.clean_sft_matches \
  --pipeline runs/paper-sft-v1/pipeline --validation runs/paper-sft-v1/capability-validation \
  --producer-session xqgeneral-paper-followthrough-v1 \
  --config configs/matches-paper-sft-baseline-v1.json --output runs/paper-sft-v1/baseline
```

`--producer-session` 指向实际拥有对应前序工作的会话；这里的后两项会话名与本服务器队列一致，复用时改成自己的实际会话。每个阶段要求正确的完成清单、原始配置及输入字节。确认原会话退出后，已有合同的 SFT／能力／对弈入口可以加 `--resume`；已完成部分读回后复用，未完成原始评测或词元预检按入口要求保留并改用新目录。不要仅因观察超时重启仍存活的任务。

## 本服务器当前队列

保存入口位于 `runs/paper-followthrough-v1/worktree/runs/paper-followthrough-preparation-v1/entry.py`，tmux 会话为 `xqgeneral-paper-followthrough-v1`；同目录 `plan.json` 绑定入口、五份实际配置和 88 份冻结执行模块，`state.json` 保存实际等待句柄。实际输出在 `runs/paper-sft-v1`，只含已固定的输入；训练／验证索引在 `data/astra-explanations-paper-index-v1`。

只有新四课完整证明出现后，队列才复查新缓存、执行正式 SFT、三组能力验证和 384 局基线；首次 SFT 等待四卡各至少空闲 90GiB，正文评测等待 GPU 0 至少 80GiB，对弈等待 GPU 2 至少 48GiB。这些是保守启动条件，实际显存容量仍由真实预检决定。私有保存入口与运行数据不随软件包发布；复用公开命令即可执行相同阶段。

[轻量读回证据](../evidence/paper-post-curriculum-preparation-v1.json)记录了真实索引逐字段比较、四轮有限遍历及两个现场存活句柄。它证明数据准备和顺序等待，不证明新四课、讲解训练、蒸馏或最终模型能力已完成。
