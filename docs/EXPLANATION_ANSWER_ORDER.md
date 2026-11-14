# 可选的讲解字段前置

`explanation_answer_profile: "explanation_before_move_json_v1"` 将讲解监督答案中的
`explanation` 键放到 `move` 前。省略或设为 `null` 时，答案保持原有字节。
它只作用于 `stage: explanation`；课程题、走法题与数值题保持原样。

这个实验比较两种监督序列。配置只规定 JSON 字段次序，正文仍是原文，可能在开头就提到着法；
实际推理顺序、棋力和讲解收益需要训练后测量。

## 不改源标注的编码

格式化发生在答案分词前。原数据文件、问题、ID、完整历史、数据分割与特征键不变。
每个原始 JSON 成员（包括嵌套分支、数值写法、中文转义及正文）的字节都保留，
其余成员的相对顺序和外部空白也保留。重复键、缺少文本型 `move`／`explanation`
或非有限数值会在新格式中报错。

训练批量、验证的正常／清零／错配记忆条件、CPU 分词预检及生成样例的参考答案使用同一配置。
真实 GPU 预检从同一训练配方继承该配置。精确续训禁止切换格式；切换时使用新的实验目录。

## 运行前的同格式预检

搜索训练示例为
[`search-distillation-sft-one-gpu-cpu-offload-with-coaching-explanation-first-v1.json`](../configs/search-distillation-sft-one-gpu-cpu-offload-with-coaching-explanation-first-v1.json)。
除格式声明和用途说明外，它与原单卡旧讲解回放示例使用相同设置。数据路径仍需绑定实际完成的产物。

在原有 `xqgeneral.distillation_preflight` 命令中增加：

```bash
--explanation-answer-profile explanation_before_move_json_v1
```

`xqgeneral.explanation_preflight` 也提供相同选项。新格式的预检会保存独立的类型、
配置声明与验证声明；旧版本据此拒绝该产物。新入口显式识别此类型，训练配方与预检不一致时会报错。
旧预检需要重新运行分词才能用于新格式。

新格式与原格式的 NLL 对应不同目标序列，跨格式不能直接据其高低宣称能力提升。
开发模型仍须用相同局面的原始走子、完整讲解事实和对弈结果验收；输出键顺序不修补非法着法。
最终独立测试继续只在模型锁定后使用。

## 证据范围

字段内容保留、原默认编码、CPU 分词／训练批量一致性、验证及续训合同有专项 CPU 控制。
现有标注的完整重排审计仅证明内容保持；不增加独立标签数量。
此配置尚未完成实际 GPU 训练，也没有模型收益或最终教练质量结论。
