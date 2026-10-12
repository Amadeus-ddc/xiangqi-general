# 分片专家特征缓存

核对日期：2026-10-12。实现位于 `feature_store.py` 和 `cache_feature_store.py`。格式为 `sharded_expert_features_xiangqi_v1`；当前课程准备队列将在完整问答核验后构建正式分片缓存。

## 构建和训练

先完成问题生成、棋局分割和原生核验；来源清单必须绑定每份问题文件的原始 SHA256 和字节数。构建器逐行读取实际问题历史、核对棋盘与历史键，同一历史只提取一次。默认每片 512 个历史、每次 GPU 提取 32 个；16 层、90 格、512 维 FP16 的单片张量约 720MiB，不拼接完整缓存。WDL 保持 FP32，FP16 转换后非有限值也会拒收。

```bash
python -m xqgeneral.cache_feature_store \
  --source data/new-paper/train.jsonl \
  --source data/new-paper/validation.jsonl \
  --source data/new-paper/test.jsonl \
  --source-proof data/new-paper/manifest.json \
  --weights models/px0-latest.pb.gz \
  --depths 0 1 2 3 4 5 6 8 9 10 11 12 14 15 17 19 \
  --base-cache data/research-human-engine-v1/features-16.pt \
  --base-proof data/research-human-engine-v1/features-16.manifest.json \
  --output data/new-paper/feature-store
```

这些是新实验的路径示例。已有基缓存的字节、专家权重和编码器源码必须与完成证明相同；只引用其原文件，保持键顺序和张量字节。新增片禁止历史键重叠。完全由已有缓存覆盖的来源不会加载专家。提取使用固定完整 Px0，FP32 计算后按原缓存方式转成 FP16；没有量化专家或解码器。

后续蒸馏可直接继承完成的分片缓存。把 `--base-cache` 和 `--base-proof` 都指向前一份 `feature-store/manifest.json`，构建器重新核对其清单、完整键目录和所有分片字节，逐片检查实际历史顺序、深度与精度，再直接引用原张量文件。新清单保留原键前缀并追加新历史；只为原缓存没有的历史调用专家。可以继续继承上一轮的新清单，无需拼接或复制完整张量。

```bash
python -m xqgeneral.cache_feature_store \
  --source data/distillation-round-1/train.jsonl \
  --source data/distillation-round-1/validation.jsonl \
  --source data/distillation-round-1/test.jsonl \
  --source-proof data/distillation-round-1/manifest.json \
  --weights models/px0-latest.pb.gz \
  --depths 0 1 2 3 4 5 6 8 9 10 11 12 14 15 17 19 \
  --base-cache data/new-paper/feature-store/manifest.json \
  --base-proof data/new-paper/feature-store/manifest.json \
  --output data/distillation-round-1/feature-store
```

示例里的蒸馏标签须先实际挖掘、汇总并完成隔离收取。新的增量入口供后续蒸馏使用，正在运行的课程准备继续按保存计划执行。

新配方指向完成清单：

```json
{
  "feature_path": "data/new-paper/feature-store/manifest.json",
  "feature_cache_mmap": true
}
```

可同时使用 [有限课程遍历](PAPER_FINITE_TRAINING.md) 和 [问答行索引](PAPER_INDEXED_TRAINING.md)。读取入口对训练、推理和预检返回相同顺序的原始特征；一次批次同时取各深度，避免逐层重复打开文件。最多保留八个分片映射，重复索引和跨片顺序保留。已有基文件作为一个引用片，其驻留文件页取决于实际访问；映射不等于全部页已读入内存。

清单、键目录、每片实际字节均纳入训练输入哈希、课程交接及讲解预检。每次读取核对清单、键目录和实际使用片的文件身份；保存检查点和预检结束时核对完整存储。精确续跑拒绝变动的片或目录。长样本 SFT 预检使用对应子数据的行索引，并保存全部索引产物，不复用全库偏移。

长任务放在 tmux 并用 `scripts/freeze_run.py` 冻结源码；读取 `progress.json` 检查提取进度。只有完成 `manifest.json` 可用，中断目录保留并换新目录重建；已有缓存、问题文件和原冻结队列保持原字节。

## 验证范围

54 项新增缓存检查和一项缺失随机状态拒收检查通过；全套 1,113 项通过，159 份源码／测试绑定不变。检查包括源分割与历史、顺序和重复读取、张量精度、跨片末段、错误来源／专家、FP16 溢出、运行中变动与完成后不可写。四课交接及讲解预检的相关检查也通过。

实际 CUDA 训练入口用构造张量模型比较原单文件、分片和第四步中断后恢复到第八步：参数、优化器、CPU／CUDA 与 Python 随机状态、监督词元和批次轨迹逐位一致。没有加载预训练学生；真实多卡神经恢复、正式 26 类新问答及完整新增缓存、四课训练和棋力／讲解验收仍待执行。真实控制完整比较旧缓存 169904 个历史的 16 层、125266821120 个 FP16 特征值及 509712 个 FP32 WDL 值，全部逐位一致且有限；16 个新增训练历史由实际 Px0 GPU 提取成两片，每片八条，再用相同固定专家、相同批量独立重算，全部特征与 WDL 逐位一致。控制约 520 秒，进程峰值 RSS 32166520KiB（约 30.68GiB），包含完整旧层映射的驻留页，不是神经训练的内存或吞吐。两组各冻结 88 份模块。完整新缓存仍待生成；见 [paper-feature-store-v1.json](../evidence/paper-feature-store-v1.json)。
最终发布前重新哈希两组控制声明的全部实际存储与输出文件，并复核各 88 份冻结模块、159 份当前源码／测试及原生产 67 份冻结模块；全部身份保持一致。

增量继承另以 64 盘不同真实棋谱的训练历史执行固定 Px0 GPU 提取：32 条基缓存先后追加 16 和 16 条，与相同专家、相同批量从头提取的 64 条逐项比对。三次读回分别比较 23592960／35389440／47185920 个 FP16 特征值及 96／144／192 个 FP32 WDL 值，全部逐位一致且有限；两次扩展各只写一份新张量，所有原文件和清单重新哈希不变。13 项新增检查覆盖连续扩展、旧单文件祖先、完成合同、文件篡改、实际片键错配与已有历史免提取，全套 1377 项通过。独立入口另读回全部 24 份约 3.20GB 绑定，并直接加载每片张量复算三组比对，不调用分片读取器。见 [增量缓存证据](../evidence/paper-distillation-cache-extension-v1.json)。这项控制没有生成蒸馏标签或训练学生，正式蒸馏收益仍须验证。
