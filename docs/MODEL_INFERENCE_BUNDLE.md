# 可搬动的模型交付目录

`python -m xqgeneral.model_bundle` 将已完成的训练记录导出为推理目录。`xqgeneral-ask --adapter` 和 `xqgeneral-coach --checkpoint` 可直接使用该目录；默认加载目录中的 Px0，另指定权重时必须与训练专家的 SHA256 一致。

目录包括 `adapter.pt`、`config.json`、`manifest.json`、`decoder/` 和 `expert/px0.pb.gz`。底座配置、分词器、完整原底座权重和已有模型卡／许可证文件复制到 `decoder/`；所有已训练参数保留原值和精度。完整解码器训练的参数映射也完整保留，不只保留连接桥。优化器、随机状态、数据集、特征缓存和下载记录不进入目录，运行配置省去私有训练路径。目录可以整体搬动，加载前重新核对每个文件和九份推理／编码源码。

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

`runs/completed-model-training` 是待替换的实际选模目录，当前新四课和后续 SFT 尚未完成。原 `adapter.pt` 文件入口继续可用。模型和数据不提交到源码仓库；权重来源与用途仍依照 [模型说明](MODEL_CARD.md) 和 [第三方说明](../THIRD_PARTY_NOTICES.md)。

验证使用现有两步、五个完整历史的容量控制检查点：搬动 14 个文件、约 10.90 GB 的目录后，真实 Qwen／Px0 的 624 个参数张量、4694274592 个运行时参数以及全部缓冲一致；其中只有 671806496 个连接桥／棋盘参数参与了那两步容量更新，底座仍冻结。35 道原生题的 token、16 份现场专家特征、首步全部概率对应的 logits 和最多 8 个新 token 的原始回答均逐值相同。GPU 0 设备使用量最高观测为 23299 MiB。

独立 CPU 入口重新哈希 39 份约 21.80 GB 的产物，重建运行配置、对照全部 226 个 FP32 训练张量及 35 道金标，检查完整目录和 144 次显存记录。现场 GPU 等价结果继承自已经完成的真实控制，CPU 入口没有重新前向。全套 1473 项通过，188 份源码／测试／测试配置不变。小模型单测另覆盖完整解码器参数、缺失分片、损坏文件、未绑定权重、错误专家、搬动和遗失源目录。

本验证没有新增课程、初始讲解或蒸馏训练，没有评判完整讲解质量或棋力。控制产物保留在 `runs/model-inference-bundle-v1/worktree/runs/`，轻量依据见 [交付验证记录](../evidence/model-inference-bundle-v1.json)。
