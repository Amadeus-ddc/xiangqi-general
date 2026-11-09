# 七类监督的辅助联合训练

当前服务器以[两卡微批量3](TWO_GPU_FULL_DECODER.md)和[完整状态迁移](AUXILIARY_RESOURCE_TRANSITION.md)完成首轮：累计16000步、选中12000步，辅助完成1轮、真实新搜索蒸馏0轮。2026-11-09固定结果见[完成记录](AUXILIARY_ROUND_COMPLETED.md)。物理GPU1、2预留，现役任务仅允许GPU0、3；下文四卡与三卡部分保留为其他拓扑及历史合同，不是本机当前启动方案。

本流程从已完成讲解训练的完整模型继续，以中文讲解、四门原课程回放、
`move_quality` 和 `engine_numeric` 七类监督共同训练。输入来自
[已完成的组合语料](ORACLE_POLICY_CORPUS.md)，监督答案按词元平均计算损失。
这是一轮辅助监督训练；它不增加已训练的在策略搜索蒸馏轮次数，也不直接证明棋力或教学效果。

## 数据、父模型与缓存

CPU 预检要求组合语料的原生完成证明，核对全部七类训练和验证行、唯一 ID、
原始行哈希、实际分词长度和完整历史对应的专家特征。独立测试答案不解析、不编码。
父模型须有已完成的全参数训练合同。扩充特征缓存可以有多代祖先，但原父缓存的
有序历史键、张量分片、专家及编码器身份必须完整保留。

```bash
python -m xqgeneral.auxiliary_policy_preflight \
  --data runs/oracle-policy-corpus-production-v2/dataset \
  --source runs/joint-replay-initial-sft-v1/pipeline/sft/adapter.pt \
  --parent-completion runs/joint-replay-initial-sft-v1/pipeline/manifest.json \
  --features runs/oracle-policy-feature-preparation-v1/features/manifest.json \
  --pinned-base-model-manifest runs/full-decoder-available-capacity-v2/pinned-decoder/manifest.json \
  --max-tokens 2048 --workers 8 \
  --output runs/auxiliary-policy-token-preparation-v1/preflight
```

以上路径是实验目录示例，必须与实际父模型、完成证明和缓存一致。已有同名任务运行时
应检查原任务，不能重新执行入口。长任务放在 tmux 或调度器中，并冻结源码、配置和输入身份。

预检输出逐行长度、各题型监督词元数、历史数量及已核验输入收据。
`XQGENERAL_VERIFIED_INPUT_RECEIPT` 可显式引用先前已核验收据；复用文件的身份和
实际新读字节分别记录，不把收据复用声称为重新读取全部特征张量。
同一命令加 `--resume` 只接受完成结果：重新编码全部行并比较长度、计数和合同，
原输出字节和修改时间保持。未完成输出保留，使用新目录记录下一次尝试。

## 预算与选模

[模板](../configs/auxiliary-policy-sft-v1.json)固定四卡、全参数 FP32 主权重、
FSDP2、全局批量 32、每卡微批量 1，以及桥和解码器各自的学习率。
编译器根据实际七类行数生成混合比例与有限遍历预算；每一遍先完整覆盖全部不同问题行，
分布式尾部重复单独记录。允许一至四遍，模板使用四遍；至少完成一遍才允许早停。
具体步数必须以实际 `budget.json` 为准，不能沿用旧 7770 行实验的 972 步。

每次正式验证覆盖全部七类原始验证 ID，原有讲解和旧课题完整保留。
新增两个题型仍来自原验证历史，没有增加独立验证局面。选模指标是所有验证答案的
词元平均 NLL；各课准确率、首着合法性、变化与候选一致性、讲解事实和实战成绩另行评测。
起点 0 步获选的训练不能晋升为合格新轮次。最低步数限制实际作业的早停时机，并不强制最佳检查点位于完整首遍之后；本轮最佳12000步先于12115步边界，其所见360000个有效行出现次数与作业已完成全源首遍分别记录。

## 四卡容量与精确恢复（历史及可选拓扑）

正式入口首先建立原生行索引和完整验证 ID 计划，再执行真正的四卡预检：
从实际数据保留每类最长行，并补齐到 32 条训练、12 条验证样本；原行字节保持。
有界探测覆盖七类，但不冒充完整正式语料。实际连续四步与两步加两步必须得到相同的
完整参数、优化器、随机状态和数据位置，全部参数与 Adam 状态均有限且保持 FP32。
预检使用明确的 `four_gpu_auxiliary_policy_sft_preflight` 类型，不能借用旧蒸馏完成标志。

```bash
python -m xqgeneral.auxiliary_policy_training \
  --init runs/joint-replay-initial-sft-v1/pipeline/sft/adapter.pt \
  --recipe configs/auxiliary-policy-sft-v1.json \
  --token-preflight runs/auxiliary-policy-token-preparation-v1/preflight/manifest.json \
  --output runs/auxiliary-policy-full-source-joint-round-v1/round
```

运行前须取得服务器既有的完整四卡任务租约，并保持到预检和正式训练全部退出。
不能绕过共享 GPU 队列、重启存活生产者或停止来源不明的显存占用。
`state.json` 区分预检、正式 SGD 和失败；实际更新读取原生训练日志。
完整输出通过后，完成重开不新增 GPU 训练；失败及部分输出保留。

<a id="三卡资源配置"></a>

## 三卡资源配置（历史及可选拓扑）

[三卡模板](../configs/auxiliary-policy-sft-three-gpu-v1.json)使用明确的
`full_decoder_auxiliary_three_workers_v1` 配置：FSDP2、全局批量 30、每卡微批量 1，
每次完整更新每卡累计十个微批量。原生训练器要求全局批量能够被工作进程数整除，
所以不能在三卡上继续声明批量 32。当前服务器仅允许两卡；本节三卡例子不用于现役运行。学习率、词元损失、全部七类数据和验证规则保持原配方。
编译器按实际行数重新计算遍历步数，尾部只补齐一个分布式微批量，而不是补齐至 30 行。

三卡必须重新完成 `three_gpu_auxiliary_policy_sft_preflight`：每类最长行共 30 条训练、
12 条验证，连续四步与两步加两步逐位比较，三份工作进程的显存、随机状态、参数、
Adam 与有限遍历位置全部核对。四卡完成标志不能授权三卡训练。三卡完成合同明确写入
`actual_three_gpu_full_decoder_preflight_passed=true` 与四卡标志 `false`；父模型和
评测入口按同一配置核对实际工作进程数、批量和完成类型，辅助／搜索轮次仍分开计数。

物理卡号由服务器分配策略控制，不写死在通用训练模块中。2026-11-07 用户明确保留
物理 GPU2，当时本项目后续任务允许 0、1、3；随后GPU1也在第1000步保存后释放，现役仅允许0、3。旧四卡任务等第 243 步原生验证和完整
检查点落盘后释放；原四卡等待队列被拦住，不能直接恢复。完整检查点保留原四卡
优化器状态，不声称可以原样跨三卡续训。三卡新实验从声明的已完成父模型初始化，
使用新目录和自己的实际容量／精确续跑证明。CPU 语料及分词准备可以继续。

```bash
# 历史三卡示例：本机当前不启用。现役仅允许GPU0、3，见两卡与迁移文档。
CUDA_VISIBLE_DEVICES=0,1,3 python -m xqgeneral.auxiliary_policy_training \
  --init runs/joint-replay-initial-sft-v1/pipeline/sft/adapter.pt \
  --recipe configs/auxiliary-policy-sft-three-gpu-v1.json \
  --token-preflight runs/auxiliary-policy-token-preparation-v1/preflight/manifest.json \
  --output runs/auxiliary-policy-three-gpu-joint-round-v1/round
```

三卡协议的 CPU 受控产物检查不代表真实三卡完整模型通过，也不代表已重新开训。
实际语料、分词重编码、显存容量和学生棋力仍分别验收。

## 早期证据边界（2026-11-07历史记录）

2026-11-07，J2304 父权重、2386161 个特征历史的完整祖先关系及固定底座 11 个文件已完成
实际 CPU 核验；本次新核验 38 个文件、23706047624 字节，其余明确复用旧收据。
新增流程有 20 项预检和 7 项模拟训练协议控制，另有 102 项旧 SFT 相关检查通过记录。
模拟训练不加载真实 GPU。完整新源码回归、实际七类组合语料与分词重读、四卡预检和
新辅助 SGD 尚须各自验收；旧严格讲解实验的 GPU 进度不能代替这条新路线。

完成时只增加 `trained_auxiliary_policy_rounds`，保持 `trained_search_distillation_rounds`。
最终交付仍要求真实棋力、可靠中文讲解、隔离的最终评测和可搬动的模型目录。
