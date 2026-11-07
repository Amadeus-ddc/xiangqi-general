# 辅助训练的三卡到两卡完整状态迁移

用户预留GPU1、2后，当前三卡辅助任务在第1000步完整检查点保存后退出，后续只允许物理GPU0、3。显式迁移入口保留已经完成的更新、全部FP32参数、Adam状态及数据进度，再从第1001步继续同一辅助轮次。

| 产物 | 原生类型与用途 |
| --- | --- |
| 释放后的源状态清单 | `released_partial_auxiliary_training_state`：核验已经退出的三rank、原轮次合同、完整模型及优化器、随机状态、原始训练前缀和全部验证ID |
| 两卡迁移清单 | `three_to_two_worker_auxiliary_state_transition`：写出并逐张量读回原生两卡可消费状态，绑定源证据和目标配置 |
| CPU准备完成状态 | `prepared_auxiliary_round_without_GPU_or_SGD`：建立完整索引、验证选择和迁移状态，尚未启动GPU预检或正式SGD |
| 最终辅助完成清单 | 记录原三卡更新、实际新增两卡更新及总全局步数；完整一轮完成后才增加一次辅助轮数 |

源状态清单完成表示快照检查完成。部分训练状态不能作为已完成模型的父权重，不能提前增加辅助或搜索蒸馏轮数。

## 保持和重新声明的内容

迁移核对所有参数名、顺序、形状、有限FP32值及完整Adam整数索引、参数组、两个矩估计和逐参数更新计数。全局步数、累计监督词元、初始／最佳验证损失、最佳步数、耐心计数、Python和CPU随机状态均保留；旧最佳adapter即使仍是0步也如实保留。

同一全局批量30、原始数据、混合比例、学习率、预热、验证选取和有限遍历预算继续使用。程序从原生索引重建已经消费的每个全局批次及下一批有效行的顺序；全部原验证ID和顺序必须一致。当前配置允许原第一遍尚未消费尾部填充的状态，第1000步低于每遍12115步，符合此合同。

原三卡RNG按显式`--retained-ranks 0 2`保留，映射到新两卡rank0／1，对应服务器上继续使用的物理0／3。卡数及可选微批量改变会改变之后的随机与归约轨迹；不声明与继续三卡逐位相同。普通原生准确续跑仍要求相同的源码、world、输入和计算合同。迁移清单单独记录原身份与目标身份，并核对原生`resumed_from_step`实际延续保存步数。

原保存执行源码必须完整存在。训练器、模型、模型输入、有限采样、参数分片和Adam恢复等计算文件须逐字节相同；只有明确列出的编排、验证、拓扑模块和新增迁移模块允许已审查的差异。不会执行产物中保存的源码。

## 可复用入口

先等真实释放证明成功，再仅用CPU捕获原状态。下面使用本服务器已有路径；源冻结清单属于原三卡执行，后续执行须使用此次合格的冻结源码和固定revision。

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python -m xqgeneral.auxiliary_resource_transition capture \
  --round runs/auxiliary-policy-three-gpu-joint-round-v1/round \
  --release runs/gpu1-release-request-20261108-v1 \
  --source-freeze runs/auxiliary-three-gpu-development-v1/source-freeze.json \
  --output runs/released-three-worker-auxiliary-source-v1
```

在允许的两卡配置、真实完成的分词证明及已核验输入会话下，可以先完成CPU准备。该阶段不申请GPU，适合在等待实际数据或资源准入时准备。

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python -m xqgeneral.auxiliary_policy_training \
  --init runs/joint-replay-initial-sft-v1/pipeline/sft/adapter.pt \
  --recipe configs/auxiliary-policy-sft-two-gpu-v1.json \
  --token-preflight runs/auxiliary-policy-token-preparation-v1/preflight/manifest.json \
  --resource-source runs/released-three-worker-auxiliary-source-v1/manifest.json \
  --retained-ranks 0 2 --prepare-only \
  --output runs/resumed-two-worker-auxiliary-round-v1
```

实际数据、源码和资源准入通过后，在tmux中使用同一配置、源状态和输出目录，加`--resume`并去掉`--prepare-only`。运行环境显式设置`CUDA_VISIBLE_DEVICES=0,3`。程序先完成真实两卡容量、全部worker资源及连续四步对两步加两步的完整参数／Adam／RNG／数据游标比较，再加载转换状态执行正式SGD。最终完成入口和后续父模型读取都会复核原生执行日志、安装记录和跨资源更新账本。

可选[微批量3配置](../configs/auxiliary-policy-sft-two-gpu-micro3-v1.json)仍为全局批量30，须单独通过真实最长序列容量和准确续跑预检后使用。它是速度候选，尚无实测速率；失败的尝试应保留，使用微批量1时建立新目录。两种配置都完整覆盖363445条实际训练及3749条原验证，至少12115步，四遍上限48460步。未来每遍尾部重复行分别为微批量1的1行、微批量3的5行，原三卡为2行；差异在迁移证据中明确记录。

## 检查点与证据保存

原三卡文件不修改。新两卡参数及优化器逐张量读回；初始状态安装使用硬链接，而训练日志复制原始完整前缀。受控链接导致的ctime变化只有在文件身份和SHA256重新核验后才由该安装阶段确认；正常输入身份检查仍严格执行。后续原生原子替换不会改变初始迁移文件的字节，完成后可重新读回。

保存步数之后尚未落盘的更新不宣称已恢复，原日志完整保留。现有输出目录和原始失败不能覆盖。相同卡数下的继续运行与跨卡数迁移分别验收；不能仅因观察超时重启存活任务。

当前166项定向CPU检查通过，包含实际CPU Adam状态、原生索引、完整受控续训和父模型重读；实际归档三卡step4的624参数／Adam格式和顺序也已只读核对。这些证据不替代真实step1000捕获、全套CPU／构建／发布检查、真实两卡GPU预检或模型棋力和中文讲解评测。
