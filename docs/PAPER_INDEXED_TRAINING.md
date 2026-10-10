# 大规模问答的磁盘读取

核对日期：2026-10-11。实现见 [training_index.py](../src/xqgeneral/training_index.py)。有限课程混合仍遵守 [完整遍历合同](PAPER_FINITE_TRAINING.md)，磁盘读取保留原始文件顺序、所选行身份和每步批次；当前生产训练保持原冻结版本。

## 准备与使用

在完整问答生成和隔离检查完成后，分别建立训练、验证索引；多来源按训练入口相同的顺序重复 `--source`。输出目录须为新目录。

```bash
python -m xqgeneral.training_index --source data/new-curriculum/train.jsonl --split train --output data/new-curriculum/train-index
python -m xqgeneral.training_index --source data/new-curriculum/validation.jsonl --split validation --output data/new-curriculum/validation-index
```

新训练配方显式声明：

```json
{
  "training_data_profile": "author_finite_mixture_epochs_xiangqi_v1",
  "mixture_seed": 0,
  "row_index_paths": {
    "train": "data/new-curriculum/train-index",
    "validation": "data/new-curriculum/validation-index"
  }
}
```

训练索引是必需项，验证索引可省略，届时验证仍用原内存读取。每张卡通过只读映射访问紧凑的偏移／身份元数据：题型、特征覆盖、固定课程子集和 epoch 顺序无需读入全部问题及答案；实际训练批次和抽中的验证行才读取原始问答。原题目、答案、历史和来源字段完整保留。

构建器逐字节计算来源哈希，检查分割、唯一行 ID、题型合同和特征键。偏移文件、身份文件、元数据和完成清单纳入训练输入哈希；路径、来源顺序、分割或内容变化会拒绝使用／精确续跑。同长度改动也会被校验。每个批次读取前后核对文件身份与时间戳。独立测试不能建立训练索引。

长任务放在 tmux，并用 `scripts/freeze_run.py` 冻结源码。构建进度在索引目录的 `progress.json`；只有完整 `manifest.json` 出现后才可使用。完成索引只读复用；中断后的不完整目录保留，换新目录重建。训练续跑沿用原 `--resume`、源码／输入／优化器／数据位置合同，修改配方使用新实验目录。索引、SQLite 检查文件和语料均不提交 Git。

## 实际验证

54 项新增检查覆盖 Unicode、CRLF、空行、末行无换行、多来源顺序、课程过滤、重复 ID、三种卡数和 epoch 的样本一致性、验证抽样、源文件和索引变动、偏移损坏与续跑拒收。全套 1,058 项通过，156 份源码／测试绑定前后不变。

现有四课语料的 2,831,004 条训练问答及 173,272 条验证问答已全部建立索引，并通过另一条顺序读取路径逐条比较全部原始字段、元数据和字节位置。来源字节与既有生产清单一致。训练构建／读回约 157／473 秒，验证约 9／28 秒；含实际四课有限混合准备的完整控制约 691 秒，进程峰值 RSS 1,510,540KiB（约 1.44GiB）。这不是完整模型训练的内存或吞吐，也没有重新计算棋规金标或复演全部历史；既有原生验证以相同来源字节沿用。

实际 CUDA 训练入口用构造张量模型比较内存版连续八次更新、磁盘版连续八次更新和第四次中断后恢复：参数、优化器、CPU／CUDA 随机状态、监督词元、损失、梯度范数与批次轨迹逐位一致。两项控制各冻结 86 份模块；最后再次计算全部声明输入／输出、实际源码及原 67 份生产模块的哈希，共 451 份文件一致。未加载预训练学生或教师；大规模问答读取通过不代表完成新的 26 类语料、正式新增专家特征缓存、多卡神经优化轨迹、课程训练、棋力或讲解验收。完整证据见 [paper-indexed-training-v1.json](../evidence/paper-indexed-training-v1.json)。

后续 [分片专家特征存储](PAPER_FEATURE_STORE.md) 已通过完整旧缓存与真实 Px0 新片控制，并与行索引共同纳入训练／交接／讲解预检输入绑定。正式新语料、完整新增特征和真实多卡神经执行仍待完成。
