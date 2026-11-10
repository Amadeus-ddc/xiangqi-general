# 搜索汇总教师的独立部署

`xqgeneral.search_pilot` 可以为搜索汇总指定独立的 `teacher_deployment_config`。它读取同一个已完成验证的完整 BF16 教师，允许改变 `tensor_parallel_size` 和 `gpu_memory_utilization`。原验证配置和原始讲解复核记录仍是父模型的验收输入；新配置另行绑定到搜索合同、教师推理和标签收取产物。

使用时，从验证阶段的教师配置复制一份，只修改 `search_consolidator` 下的这两个部署字段。例如，原教师配置为 TP2、显存配额 0.45，副本可改为：

```json
{
  "tensor_parallel_size": 1,
  "gpu_memory_utilization": 0.70
}
```

这段 JSON 表示副本中的两个字段，并非完整教师配置。仓库、revision、模型路径、精度、量化、上下文上限、提示和其他字段都必须与原教师配置一致。搜索配置增加：

```json
{
  "teacher_backend": "vllm",
  "teacher_deployment_config": "runs/my-search/teachers-tp1.json",
  "teacher_max_context_tokens": 16384,
  "teacher_max_new_tokens": 8192,
  "teacher_batch_size": 2
}
```

其余搜索配置继续使用已验证的父模型、课程数据、隔离候选和收数条件。显卡租约、物理设备绑定及学生／教师串行仍由外层运行控制器负责。部署覆盖目前要求显式 vLLM 后端；未指定该字段时沿用原教师配置。

独立部署文件与原验证教师文件都会按内容哈希绑定。教师生成和标签收取使用独立部署文件，结果中的 `consolidation_teacher_deployment` 记录原验证教师、新汇总教师及实际是否执行过汇总。改变任何绑定文件后，必须使用新的输出目录；不能续跑旧合同。TP1 与 TP2 的输出数值等价尚未证明，产物会明确记录这一边界。

## 已完成的单卡容量实测

同一固定 Qwen3.8-27B 完整 BF16 权重已在一张 H20 上执行 TP1、显存配额 0.70、上下文 16384、输出预算 8192、批量 2 的实际推理。两条已有盲评查询分别含 5220 和 5958 个输入词元，实际生成 2727 和 2796 个词元，均完成推理、未触发生成上限。运行时读回确认无量化，加载到推理图的参数为 27356726256 个 BF16 和 2304 个 FP32；教师引擎及创建的工作进程随后退出，显存恢复到此前占用。精确权重身份、运行时版本和证据摘要见[容量记录](../evidence/full-bf16-teacher-single-gpu-capacity-v1.json)。

这次实测覆盖两条已有验证查询，没有生成训练标签或执行 SGD。完整递归搜索提示、实际搜索产出率、跨 TP 输出差异及学生训练收益仍需各自实测；两条教师推理不能证明最终象棋模型的棋力或讲解质量。
