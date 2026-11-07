# 三卡完整解码器搜索蒸馏

`full_decoder_search_three_workers_v1` 是正式搜索蒸馏的独立资源配置，与辅助监督的三卡类型
分别记录。全局批量为30，每卡微批量为1，使用 FSDP2；物理卡由服务器分配，当前明确
排除 GPU2。原四卡搜索配置仍为默认四进程／批量32，三卡必须明确声明自身配置。

## 从实际标签编译预算

配置 [search-distillation-sft-three-gpu-v1.json](../configs/search-distillation-sft-three-gpu-v1.json)
使用回答词元平均损失，并要求实际合格搜索标签及四门课程回放的完整遍历。`mixture`
在模板中声明题型，编译器按完成数据预检的真实数量生成比例；不沿用历史1114条标签的
固定权重。所有实际来源的每一行每轮都必须被选择，训练步数为每轮实际总行数除以30
向上取整。分布式尾部重复按三卡微批量补齐，并单独记录，不作为新增独立样本。

验证保留全部原讲解题和四课原验证题，以完整验证损失选模；至少完成一整轮数据才允许
提前停止。默认最多4轮、30000次更新，实际较小的预算先达到即停止。标签规模只计新
讲解标签，旧课回放不充作论文的15万条新蒸馏标签。模板中的回放路径必须与数据预检
完全一致；改变数据、权重、长度或资源配置时使用新的目录和完整预检。

这个配方保留讲解和四课回放的历史合同。新增[六类回放及七类验证](AUXILIARY_SEARCH_REPLAY.md)
使用单独的显式配置，将数值与单走法监督一并复习，并完整核对实际验证ID。原辅助模型
的验证参考隔离由[后续模型搜索准入](DESCENDANT_SEARCH_PILOT.md)强制继承。

## 实际容量和恢复检查

正式 SGD 前，以这份配方和实际最长的30条训练、12条验证讲解检查三卡完整模型。
连续4步与2步加恢复2步必须在完整参数、Adam状态、随机状态、有限数据位置及日志上
一致；全部三卡都须有实际资源和完成记录。父模型的所有参数必须原值继承，解码器、
桥和棋盘参数都须真实更新。完成类型为 `three_gpu_search_distillation_sft_preflight`。

```bash
# 取得满足分配策略的服务器资源租约后运行；路径须指向真实完成的输入。
CUDA_VISIBLE_DEVICES=0,1,3 python -m xqgeneral.distillation_training \
  --init ACTUAL_COMPLETED_PARENT/sft/adapter.pt \
  --recipe configs/search-distillation-sft-three-gpu-v1.json \
  --token-preflight ACTUAL_SEARCH_TOKEN_PREFLIGHT/manifest.json \
  --output FRESH_THREE_GPU_SEARCH_ROUND
```

这里是独立三卡新轮次，四卡优化器检查点另行保留，不宣称跨资源数的精确续训。同一
冻结三卡合同中断后，可对同一原生入口和目录加 `--resume`；变更合同必须保留旧目录，
从明确的完成父模型创建新轮次。长任务使用 tmux，并保存冻结配置及查看、恢复命令。

## 完成与能力验证

原生完成检查核对真实三卡配方、全局批量、选中完整权重和训练数据状态。三卡完成清单
同时声明三卡容量检查通过、四卡容量声明为false。只有完成实际搜索 SGD 才增加
`trained_search_distillation_rounds`，`trained_auxiliary_policy_rounds`沿用父模型；试批挖掘
不增加这两类轮次数。后续评测及再次作为父模型使用时重新核对这些声明。

CPU 控制使用模拟的 GPU／模型生产者，检查实际有限采样、三卡原生产物消费、原始验证
和完整教师辅助盲评的接口。真实三卡搜索训练、数值与走法能力保留、新讲解与对弈成绩
仍须实际运行后验收；保存轮次数不证明能力未遗忘。最终独立测试不用于训练或选模。
