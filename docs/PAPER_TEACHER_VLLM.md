# 搜索汇总教师的 vLLM 入口

搜索汇总可以显式选择 `vllm`，复用已校验的完整官方 Qwen3.8-27B 权重。它采用 BF16、无量化、单卡、最多 16 个请求、1024 token 分块预填充和 16384 token 上下文。后续搜索配置可声明 `teacher_backend: "vllm"`；同时须提供明确上下文，并移除 HF 专用的 `teacher_prefill_chunk_size`。改变配置必须使用新的输出目录。

## 执行与恢复

教师环境可安装 `.[teacher-vllm]`，固定 vLLM 0.19.1。原始权重由 `configs/teachers.json` 指定，须已有官方分片哈希清单；入口重新核验全部分片与支持文件，再加载模型。

```bash
CUDA_VISIBLE_DEVICES=0 python -m xqgeneral.local_teacher \
  --config configs/teachers.json --input runs/search/mining/queries.jsonl \
  --output runs/search/consolidation-vllm \
  --backend vllm --batch-size 16 \
  --max-context-tokens 16384 --max-new-tokens 8192 \
  --generation-config configs/paper-consolidation-generation-v1.json \
  --seed 20261072
```

正式批处理应通过 `scripts/freeze_run.py` 在 tmux 中执行，并在启动清单中记录环境、源码、配置、输入和输出哈希。服务器控制记录将编译缓存放在项目内，另用经过长度检查的短目录存放 IPC 临时文件。不要把很长的工作区路径直接用作 IPC 临时目录。

每个请求按自己的完整输入长度获得输出预算；超过上下文的输入直接失败，不截断。原始 token IDs、正文、思考结束标志、输出上限和结束原因均保留。随机性记录既有的有序批次 seed，另由批次 seed 与请求序号生成独立请求 seed；这不承诺与 HF 相同输出。中断后可复用同字节已写响应；更换输入、解码配方、后端或请求 seed 会拒绝续跑。已完成批次不重复执行。正常结束和异常均关闭已创建的推理引擎。

## 实测范围

[独立读回证据](../evidence/paper-teacher-vllm-v1.json) 对完整官方分片重新哈希，复算原始 token IDs、正文、预算和 seed，并核对磁盘张量与实际模型参数数。

| 同一组 16 条构造查询 | HF | vLLM |
|---|---:|---:|
| 每条最多输出 128 token | 2048 token / 37.27 秒 | 2048 token / 21.17 秒 |
| 每条最多输出 8192 token | 56718 token / 909.68 秒 | 60268 token / 205.90 秒 |
| 完整预算的总 token/s | 62.35 | 292.71 |

vLLM 每次测量前重置前缀缓存，计时不包括模型加载。两种实现的采样 seed 和实际输出不同；约 4.69 倍是这一控制的 token 吞吐比，不能直接推成实际蒸馏所需总时间。vLLM 完整预算控制中 15/16 条结束思考；未结束的原始输出也保留。项目实际 CLI 使用另一组可追踪请求 seed，完成同一组 16 条查询、生成 50404 token，16 条均结束思考；它的 341.29 秒包含校验和部署时间，不与上表的纯生成计时混比。

单张 H20 另完成 16 个不同的 16382-token 构造输入，各保留两个输出 token，全部完整输入逐项读回。它证明该请求容量，不证明长文本棋局讲解质量。

磁盘包含 27,781,427,952 个 BF16 参数元素、1199 个张量。标准生成模型使用 27,356,728,560 个参数元素；另 424,699,392 个属于 15 个 `mtp.*` 辅助预测张量，在 HF 和 vLLM 的标准生成架构中都默认不加载。全部官方分片，包括 MTP，仍完整保存并核验。vLLM 主模型中 2304 个标量为 FP32，其余为 BF16，实际量化方法均为 `Unquantized*`。这一区别由磁盘张量名、两种实际加载参数计数与固定版本加载器核对；加载器处理分别见 [vLLM 0.19.1](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/model_executor/models/qwen3_5.py) 和 [Transformers 5.8.1](https://github.com/huggingface/transformers/blob/v5.8.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py)。

全部 1452 项本地检查通过，185 份源码／测试字节不变；CPU 控制覆盖预算越界、原始输出损坏、权重改变、量化拒收、异常释放及中断续跑。实际 GPU 控制不是训练后的学生搜索。真实学生标签接受率、战略讲解质量、正式蒸馏和最终棋力仍未证明，新四课正式更新仍为零。
