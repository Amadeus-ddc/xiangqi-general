# 两卡完整参数训练

两卡辅助监督和搜索蒸馏各有显式配置。逻辑进程数与物理卡号分开：配置声明两进程，运行时由资源策略分配允许的两张卡。本服务器的新任务仅允许物理GPU0、3；GPU1、2预留，正在运行的三卡任务有等待第1000步检查点保存后退出的临时例外。

| 阶段 | 配置 | 原生预检类型 |
| --- | --- | --- |
| 七类辅助监督 | [auxiliary-policy-sft-two-gpu-v1.json](../configs/auxiliary-policy-sft-two-gpu-v1.json) | `two_gpu_auxiliary_policy_sft_preflight` |
| 新搜索讲解和六类回放 | [search-distillation-sft-two-gpu-with-auxiliary-v1.json](../configs/search-distillation-sft-two-gpu-with-auxiliary-v1.json) | `two_gpu_search_distillation_sft_preflight` |

两种配置均要求完整FP32可训练参数、FSDP2参数及梯度分片、全局批量30、微批量1。辅助监督从已完成七类分词证明取得完整来源比例；搜索配置从真实接受讲解及[六类回放](AUXILIARY_SEARCH_REPLAY.md)的实际数量编译完整遍历。两者都用回答词元平均损失，验证覆盖原始全部七类，不读取独立测试答案。

## 原生执行与完成证明

训练入口先执行真实两卡容量与续跑预检，再启动正式SGD。预检从完整父权重初始化，比较连续四步与两步加两步恢复后的全部训练张量、Adam状态、随机状态和有限数据游标；所有实际worker的资源证明都须存在。缺少第二个rank、实际world不符、非有限优化器状态、错误数据游标或随机状态都会拒收。

完成证明记录实际world为2，以及`actual_two_gpu_full_decoder_preflight_passed=true`和相应两卡profile。其他卡数的预检标记不能同时声明为真。后续父模型读取和开发评测会核对原生训练、全局批量、选中权重、七类验证ID与对应预检；辅助监督与搜索蒸馏的轮次数分别累计。

从已完成的父模型新建辅助轮次时，原生入口如下。长任务放入tmux，并先完成资源准入；服务器上当前三卡任务的检查点保留和释放须先完成。

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python -m xqgeneral.auxiliary_policy_training \
  --init runs/joint-replay-initial-sft-v1/pipeline/sft/adapter.pt \
  --recipe configs/auxiliary-policy-sft-two-gpu-v1.json \
  --token-preflight runs/auxiliary-policy-token-preparation-v1/preflight/manifest.json \
  --output runs/new-two-gpu-auxiliary-round-v1
```

本次实际完成的分词证明已纯CPU编译为363445条训练、3749条验证、每遍12115步、至少一遍／最多四遍48460步。两卡尾部填充重复1行；原三卡重复2行。所有真实训练行仍须完整选中。编译证明没有执行GPU预检、模型生成或训练更新。

## 三卡检查点的后续处理

当前三卡任务在第1000步保存完整模型、优化器、随机状态及数据游标后退出。切换为两卡需要单独核验状态迁移；上面的新轮次命令不会自行完成这种迁移。现有原生准确续跑检查仍要求相同的执行源码、world、输入与计算合同。

状态迁移应保留全部参数与Adam状态、全局更新步数及已消费数据，并明确记录卡数改变带来的随机与分布式执行差异；相同卡数下的准确续跑证明不能替代跨卡数证明。当前两卡配置及受控CPU生产／消费检查通过，不代表三卡状态已迁移、真实两卡容量已通过、辅助或搜索轮次已完成，也不证明棋力或教学收益。

初始新增检查曾在未实现两卡profile时出现18项失败／12项通过；实现后125项定向控制通过，覆盖两卡完整流程及原三／四卡合同。完整CPU套件、构建、发布检查，以及真实GPU执行分别验收。
