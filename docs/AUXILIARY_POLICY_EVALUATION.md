# 七类辅助训练模型的开发评测

评测入口接收已完成的[七类辅助联合训练](AUXILIARY_POLICY_TRAINING.md)，
输出 `auxiliary_policy_capability_validation` 和 `auxiliary_policy_full_match_baseline`。
辅助监督轮次与在策略搜索蒸馏轮次分别记录，不把辅助模型标成初始模型或搜索蒸馏模型。

## 完成模型与原始验证题

消费者核对实际选中权重、原生训练与四卡预检、完整参数集合、有限 FP32 状态、
父模型、数据、缓存、源码和配置身份。起点 0 步不作为合格新模型。
选中模型的辅助轮次须等于其父模型辅助轮次加一，搜索蒸馏轮次数保持原值。

[讲解配置](../configs/evaluation-auxiliary-policy-v1.json)的 `sft_recipe` 指向训练输出的
实际 `round/recipe.json`，`token_preflight` 指向已完成七类分词预检。
`data` 则指向原始讲解验证目录，当前实验为原 384 道讲解题。
原始讲解验证行必须与七类组合语料中 `explanation` 子序列的字节及顺序完全相同。
验证子集、重排、改写答案、换缓存或读取独立测试的声明均拒绝。
新增走法／数值题增加题型，不增加独立讲解验证历史。

目录和教师设置是当前实验示例，其他服务器须填写其已核验资源。
模型和教师执行环境须满足各自依赖；当前教师使用完整、未量化 BF16 权重。
服务器的自动流程须先核验完成证据与输入收据，再取得共享 GPU 资源租约；
已有同名任务存活时查看原任务，不重复启动。

## 配对讲解与辅助盲评

```bash
python -m xqgeneral.clean_sft_evaluation \
  --pipeline runs/auxiliary-policy-full-source-joint-round-v1/round \
  --producer-session xqgeneral-auxiliary-policy-full-source-joint-round-v1 \
  --config configs/evaluation-auxiliary-policy-v1.json \
  --output runs/auxiliary-policy-development-evaluation-v1/lecture
```

全部原始讲解题分别使用正常、置零及不同历史的专家记忆，保留未经修复的学生原答。
机械检查单独报告解析、首着合法性、引擎错误门槛、主变化、候选和完整结构。
完整 BF16 教师在原始回答和核实的局面信息上作辅助盲评，保留无效或被拒评分；
有效评分的条件均值不当作全部请求的均值，辅助评分也不等于人工教学验收。

同一合同的完成重开只读取已保存产物，不新增学生或教师推理。
输入、源码或原始回答变化时保留旧结果，使用新输出目录。
现有固定 2600 道课程保持率复核仍单独执行，讲解分数不能替代它。

## 完整原始对弈

```bash
python -m xqgeneral.clean_sft_matches \
  --pipeline runs/auxiliary-policy-full-source-joint-round-v1/round \
  --validation runs/auxiliary-policy-development-evaluation-v1/lecture \
  --producer-session xqgeneral-auxiliary-policy-full-source-joint-round-v1 \
  --config configs/matches-auxiliary-policy-v1.json \
  --output runs/auxiliary-policy-development-evaluation-v1/matches
```

[对弈配置](../configs/matches-auxiliary-policy-v1.json)沿用原 64 个验证开局、双方执色和
100／1000／10000 节点对手，合计 384 局。完整对弈要求配对讲解评测已经完成，
且两者绑定同一个实际选中模型、辅助轮次及搜索蒸馏轮次。
保留非法走子判负、原始讲解合同错误、截断和未完成棋局；不施加合法动作约束，
不以引擎替换学生走法。原生完整棋局和逐步记录须支持结果重读。

## 已测范围

2026-11-07，15 项新增控制与 69 项旧初始／搜索蒸馏评测及完整对弈回归通过。
新控制实际读取原始验证文件，检查字节与顺序、禁止打开独立测试，消费受控辅助训练
生成的真实文件和张量，并执行受控配对原答及完整对弈。这里的模型、教师和对手为
受控替身，没有实际 GPU 神经推理，也没有新辅助模型的棋力或讲解成绩。

真实辅助训练、完整源码资格、实际固定开发复核与最终独立验收各有自己的完成证明。
此入口不宣称独立最终测试已经执行，也不宣称已经获得强棋力或可靠教练模型。
