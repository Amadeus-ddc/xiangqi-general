# 搜索学生的采样设置

固定作者版本 `c372da22ca75e95c3c79bee8a83c220fc54d0dcb` 的 [mine.yaml](https://github.com/queen-project/queen/blob/c372da22ca75e95c3c79bee8a83c220fc54d0dcb/configs/self_distill/iteration/mine.yaml) 声明学生温度 0.6、top-k 20、top-p 0.95，最多 4096 个输出 token。[本地生成配置](../configs/paper-search-generation-v1.json)沿用这些数值，显式启用采样。

## 入口与来源

`Predictor.generate`／`generate_batch` 可显式接收 `sampling` 和 `seed`。没有指定采样时保留原贪心行为；原始非法文本继续照实记录，采样入口不限制为引擎走法或代写正文。[Transformers 生成参数](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)决定词元采样；棋力专家特征仍由原推理入口提供。

搜索 CLI 可使用新配置：

```bash
.venv/bin/python scripts/freeze_run.py --output runs/sampled-search/source-run -- \
  .venv/bin/python -m xqgeneral.search_distillation \
  --checkpoint "$XQ_COMPLETED_SFT_CHECKPOINT" \
  --data "$XQ_COMPLETED_COURSE_DATA" --features "$XQ_VERIFIED_FEATURE_STORE" \
  --recorded-inputs "$XQ_FRESH_ISOLATED_SEARCH_INPUTS" --seed 20260828 \
  --student-generation-config configs/paper-search-generation-v1.json --max-root-retries 3 \
  --output runs/sampled-search/mining
```

路径须指向实际完成的四课、讲解 SFT 及重新隔离的搜索来源。[带根重试的试批模板](../configs/search-paper-retry-pilot-v1.json)通过现有 `search_pilot` 串联搜索、教师汇总和收取；它等待对应讲解能力验证，数据／缓存／父权重及实际队列路径需要统一。原[采样模板](../configs/search-paper-sampled-pilot-v1.json)保留。模板仍只有 512 根，汇总输出预算仍是 1536，不能把它当成正式论文整轮。

配置文件原字节参与搜索与试批合同。搜索输入包括特征清单、完整历史键目录和全部存储分片；配置、分片或执行来源变动不能续写原实验。新采样实验还绑定实际 Torch／Transformers 版本。已有冻结课程和等待入口不随新配置修改。

## 单根续跑

本地按基础种子、原根 ID、完整历史键和根内生成序号派生 63 位种子，逐次保留设置、序号及十进制种子文本。每个根重新从零编号，跳过其他已完成根不消耗该根的随机流；根与子节点排序后仍携带原生成序号。该种子方案是本地续跑约定，不是作者 vLLM 随机流的逐位重放。

指定种子时只暂存 CPU 和实际推理 CUDA 设备的随机状态，结束或抛错后恢复；不调用对所有 GPU 重设种子的接口。随机作用域依据 [PyTorch fork_rng](https://docs.pytorch.org/docs/stable/random.html)。当前是单进程串行搜索；相同权重、完整历史和调用次序下的采样复现，需要与实际学生和引擎另作控制。

## 根评估重试

作者 [recursion.py](https://github.com/queen-project/queen/blob/c372da22ca75e95c3c79bee8a83c220fc54d0dcb/datagen/self_distill/recursion.py#L217) 只在数值评估无法解析时重新生成根分析，首次生成后最多再试三次。新 `--max-root-retries 3` 沿用这个次数与条件；本地结构化答案须有合法的 `side_to_move` 整数 cp／mate 评估并能完成概率换算。显式零预算只检查一次；省略参数保留旧实验合同，正预算要求采样。

每次生成的原文、完整结构检查、独立数值接受检查和生成序号全部写入 `root_generation_attempts`。第四次仍无可用数值评估则拒收，未进入引擎搜索。合法数值评估配上非法候选时仍记录候选错误；子节点错误按原递归逻辑处理。这些错误不会被数值重试条件混淆。改变重试预算会在加载学生之前拒绝续跑。

## 已完成的控制

固定 90 份执行模块，以随机初始化的小型 GPT-2 在 CPU 上实际调用 `Predictor`；没有加载预训练学生、专家或汇总教师。64 个种子的八 token 结果均与独立的完整前缀推理、温度／top-k／top-p 筛选和显式随机生成器逐字一致。32 个根的连续处理、跳过已完成前 11 根及倒序处理结果一致，全部原始非法输出保持拒收。另一实例实际生成了 4096 个输出 token，未截为旧 768 预算。

两个实际 CUDA 设备只运行随机张量控制：选定设备的结果与独立生成器一致，正常返回及异常后 CPU／两卡随机状态和当前设备均恢复；没有在 GPU 加载神经模型。特征分片篡改的续跑测试在加载学生之前拒收。这些控制证明本地采样与输入合同，尚未产生蒸馏标签或证明学生实力。机器可读结果见 [轻量证据](../evidence/paper-search-generation-v1.json)。

根重试另用冻结的 90 份模块执行：随机 CPU 神经解码的 32 根共 128 次短输出均与独立采样逐字一致，前 11 根完成后的续跑与倒序结果相同；另一实例实际执行四次各 4096 token，共 16384 token，保留原文并以数值失败拒收，没有调用引擎、预训练学生或汇总教师。受控结构化输出还覆盖第四次成功、预算耗尽、数值溢出、候选与子节点错误及预算变更拒收。见 [重试证据](../evidence/paper-search-root-retry-v1.json)。

## 后续验收

采样设置、根评估重试与可复现编号只是整轮准备的一部分。作者的候选补齐、子分析复用、工作批量／并发、汇总上下文 16384 与输出 8192，以及正式合格标签产出仍须继续核对和实际执行。先完成新四课、讲解 SFT、能力与对弈基线，再测合格产出并扩大搜索、训练实际蒸馏轮；最终棋力与教学验收仍使用独立保留集。
