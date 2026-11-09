# 走法与数值分析的固定开发诊断

2026-11-09的首轮辅助模型已完成765题原始执行与逐条CPU复算；结果、基线可比性及剩余错误见[本轮实测](AUXILIARY_ROUND_COMPLETED.md)。下文定义通用执行和评分合同。

`xqgeneral.oracle_policy_evaluation` 用于比较辅助训练及后续模型在原始两类引擎监督题上的
变化。数据来自已经完成生产的七类组合语料，完整保留原验证题、字节和顺序。当前真实
输入为381道数值分析、384道单走法题，共765个问题、384个历史上下文；这些问题不代表
765个独立局面。训练与最终独立测试答案不由选题入口读取或解析。

```bash
python -m xqgeneral.oracle_policy_evaluation prepare \
  --data ACTUAL_COMPLETED_AUXILIARY_CORPUS --output FRESH_DIAGNOSTIC_INPUTS
```

模型加载前，消费入口从原组合语料独立重建完整选择，并核对原生解码合同、原始字节、
全部题型数量和唯一ID。自行重新计算哈希的删行、改题或改答案不能替代原始验证输入。

## 分开报告的指标

- 单走法：只接受一个原始UCCI走法，分别报告格式、合法性及与保存引擎参考首着的吻合。
  从解释段落中抽取一个走法不算格式正确，不使用引擎修复。
- 数值分析：严格检查四字段JSON、首着、候选、完整历史下的主要变化，以及六个半回合
  或提前结束的约定。分别报告各项指标及完整参考吻合，不要求模型生成讲解正文。
- 估值：以固定Pikafish材料相关WDL变换计算相对参考的误差，仅在合法估值回答中计算
  条件均值，同时报告这个条件的回答数量。无效回答仍计入全部格式／合同指标的分母。

这些是相对保存参考的诊断，不是实战胜率。不同但同样好的走法或变化没有在这里重新
搜索，不能仅因与参考不一致便断言棋力差。棋力和中文教学质量仍依赖完整对弈与讲解评测。

## 原始模型执行

```bash
# 先取得当前服务器资源租约。GPU1、2为用户保留；此例使用物理GPU3。
CUDA_VISIBLE_DEVICES=3 python -m xqgeneral.oracle_policy_evaluation evaluate \
  --checkpoint ACTUAL_COMPLETED_MODEL/sft/adapter.pt \
  --completion ACTUAL_COMPLETED_MODEL/manifest.json \
  --features ACTUAL_COMPLETED_FEATURE_STORE/manifest.json \
  --expert-weights ACTUAL_TRAINED_EXPERT.pb.gz \
  --inputs FRESH_DIAGNOSTIC_INPUTS --output FRESH_DIAGNOSTIC_OUTPUT
```

检查完整父模型完成记录、固定底座和专家、缓存原始张量前缀及全部验证历史后，才加载
模型。默认batch8，单走法最多64个生成词元，数值题768；真实执行前应以同一原生分词器
核对参考答案和提示预算。接口不暴露EOS／触顶原因，输出明确保留这个限制。

传给生成器的记录删除答案、真实未来、教师与参考字段；实际用户消息保持一致。所有
原始答案、包括非法输出，随批次写入日志。接口也支持zero／shuffled专家记忆；错配批次
必须有至少两个不同历史。不要将不同记忆条件合并成一个准确率。

长任务放入tmux并冻结源码、配置和输入。在同一冻结合同中断后加`--resume`，保留已完成
的原始批次；重新核对原始行与评分。完成结果的续读不再次加载或生成模型回答。更改模型、
预算、缓存、输出目录或源码时使用新的运行。真实模型结果与受控CPU生产者的接口检查
分别记录。接口检查、实际训练完成和神经诊断结果分别验收；本轮两卡训练与实测见上面的固定记录。
