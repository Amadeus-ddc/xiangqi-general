# 可搬动的模型交付目录

`python -m xqgeneral.model_bundle` 将已完成的训练记录导出为推理目录。`xqgeneral-ask --adapter` 和 `xqgeneral-coach --checkpoint` 可直接使用该目录；默认加载目录中的 Px0，另指定权重时必须与训练专家的 SHA256 一致。

目录包括 `adapter.pt`、`config.json`、`manifest.json`、`decoder/` 和 `expert/px0.pb.gz`。底座配置、分词器、完整原底座权重和已有模型卡／许可证文件复制到 `decoder/`；所有已训练参数保留原值和精度。完整解码器训练的参数映射也完整保留，不只保留连接桥。优化器、随机状态、数据集、特征缓存和下载记录不进入目录，运行配置省去私有训练路径。目录可以整体搬动，加载前重新核对每个文件和九份推理／编码源码。

[完整初始SFT模型的真实核验](FULL_SFT_INFERENCE_BUNDLE.md)已覆盖全部624个FP32训练张量及搬动后的3例现场推理；这是交付组件证据，最终模型尚待棋力与教学验收。显式`autocast_cache_enabled`作为布尔设置保留到目录并用于推理，未声明时保持默认启用。

导出时须保留已完成训练的 `manifest.json`、实际选中的 `adapter.pt`、原底座，以及训练引用的特征缓存完成清单。清单用来核对专家权重、深度和编码代码；导出后的推理直接按完整棋局历史计算特征。新建目录用于每次导出，已有交付和源实验均保留。

正式选模完成后，把 `--training` 指向含有完成的 `model_training` 清单及 `adapter.pt` 的目录：

```bash
.venv/bin/python -m xqgeneral.model_bundle \
  --training runs/completed-model-training \
  --weights models/px0-latest.pb.gz \
  --output runs/xiangqi-general-inference-v1

xqgeneral-ask --adapter runs/xiangqi-general-inference-v1 \
  --question '请分析当前局面并说明选招理由。'
xqgeneral-coach --checkpoint runs/xiangqi-general-inference-v1 --interactive
```

`runs/completed-model-training` 是待替换的实际选模目录。当前新四课、真实全模型门禁及[完整初始SFT](INITIAL_SFT_COMPLETED.md)已完成；足量蒸馏、最终独立验收和正式交付仍待完成。原 `adapter.pt` 文件入口继续可用。模型和数据不提交到源码仓库；权重来源与用途仍依照 [模型说明](MODEL_CARD.md) 和 [第三方说明](../THIRD_PARTY_NOTICES.md)。

## 评测交付目录

问答、讲解、对弈评测，以及单着和主线辅助诊断的 `--checkpoint` 均可使用交付目录，也继续接受原 `adapter.pt` 文件。评测输入清单绑定目录中的模型参数、底座、分词器、配置、专家权重和交付清单；损坏或缺失的目录资产在生成前拒收。运行中输入或源码发生改变时，不发布完成清单，保留已生成的原始回答。对弈续跑还会核对全部目录资产与原合同，即使重新绑定过的目录内部自洽，也不能替换已开始评测的模型。

交付目录可以通过符号链接选择。入口先固定解析后的实际目录，再核验资产和加载模型；运行中切换该链接不会让实际模型与输入清单指向不同版本。

单着和主线诊断要求每批生成数量与请求数量相等，缺失回答会拒收，避免只按已返回的回答计算完成成绩。失败批次的原始回答与请求ID另存于`generation-failure.json`，不为它们推断配对关系或计分；此前完整批次的原文保留。单着的原始生成和规则约束解码继续分别记录；最终验收采用固定的[独立测试协议](FINAL_INDEPENDENT_EVALUATION.md)。

对弈默认使用交付目录内的专家权重；显式指定 `--expert-weights` 时，仍须与训练专家的 SHA256 一致。数据集、特征缓存和对弈引擎由评测环境另行提供。

以下开发验证命令中的目录须替换为实际导出目录；最终测试须先按[独立评测协议](FINAL_INDEPENDENT_EVALUATION.md)锁定模型和执行源码。

```bash
.venv/bin/python -m xqgeneral.evaluate_qa \
  --checkpoint runs/xiangqi-general-inference-v1 \
  --data data/paper-courses-v5 \
  --features data/paper-courses-v5/feature-store/manifest.json \
  --split validation --per-task 100 \
  --output runs/inference-bundle-development-qa
```

验证使用现有两步、五个完整历史的容量控制检查点：搬动 14 个文件、约 10.90 GB 的目录后，真实 Qwen／Px0 的 624 个参数张量、4694274592 个运行时参数以及全部缓冲一致；其中只有 671806496 个连接桥／棋盘参数参与了那两步容量更新，底座仍冻结。35 道原生题的 token、16 份现场专家特征、首步全部概率对应的 logits 和最多 8 个新 token 的原始回答均逐值相同。GPU 0 设备使用量最高观测为 23299 MiB。

独立 CPU 入口重新哈希 39 份约 21.80 GB 的产物，重建运行配置、对照全部 226 个 FP32 训练张量及 35 道金标，检查完整目录和 144 次显存记录。现场 GPU 等价结果继承自已经完成的真实控制，CPU 入口没有重新前向。全套 1473 项通过，188 份源码／测试／测试配置不变。小模型单测另覆盖完整解码器参数、缺失分片、损坏文件、未绑定权重、错误专家、搬动和遗失源目录。

本验证没有新增课程、初始讲解或蒸馏训练，没有评判完整讲解质量或棋力。控制产物保留在 `runs/model-inference-bundle-v1/worktree/runs/`，轻量依据见 [交付验证记录](../evidence/model-inference-bundle-v1.json)。

## CPU 推理

Python 接口 `Predictor(checkpoint, device='cpu')` 使用 FP32 模型和桥接层，无需外加 CPU 自动混合精度上下文。专家特征继续先经过训练使用的 FP16 缓存舍入和 BF16 转换；送入 CPU 桥接层之前再精确扩为 FP32。这不改写缓存、参数或原始回答，也不改变 GPU 的生成分支。

旧版默认 CPU 路径曾在密集桥接层的线性投影处因 BF16 特征与 FP32 权重不匹配而失败。另行显式启用 CPU BF16 计算的控制仍属于不同执行配置，其输出不能与默认 FP32 计算混作同一结果。缓存与现场特征、记忆消融和完整模型推理各自需要验证；CPU 组件检查不代表棋力或讲解质量通过。

交付目录绑定推理源码。使用本次修正时，应由匹配版本的原生导出器生成新目录；已有目录仍保留原运行时要求，不能修改其清单绕过检查。原检查点文件入口继续可用，模型选择与独立最终评测要求不变。

完整2892项CPU套件、构建与发布检查通过。真实12000步开发权重使用两条验证问题，原检查点与新导出迁移目录共四次默认CPU生成、26个新词元逐值一致；每种加载方式的624个FP32训练张量和16层专家特征均核对。这仅验证默认CPU加载与迁移推理，不证明棋力或讲解质量提升。见[轻量验证记录](../evidence/cpu-inference-memory-precision-v1.json)。
