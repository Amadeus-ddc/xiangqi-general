# 复用已验证的实战局面目录

`native_catalog` 将完整实战根池的 SQLite 分类目录与完成的来源／全量历史复验证明绑定。复核逐项比较全部根 JSON、抽样用表头、棋局归属及每个未来步数的分类位；目录中的保存分类不能作为自己的正确性依据。

完整过去与分割隔离沿用相同字节的已完成来源证明，分类仍由原生规则重算。仅接受完整实战根池；生成式战术终局扩展使用原建库入口。未使用的 `positions` 表不重新计算，后续选样只读取已核验的根与目标表。

## 使用

先取得已经完成且没有 SQLite 日志文件的目录快照。变量指向根、来源、全量历史读回和旧参考集；源码、目录与大文件保持在自己的运行目录中。长任务放在 tmux，通过 `scripts/freeze_run.py` 保存执行源码。

```bash
.venv/bin/python -m xqgeneral.native_catalog \
  --catalog "$XQ_CATALOG" \
  --roots "$XQ_ROOT_FILE" --source-manifests "$XQ_ROOT_MANIFEST" \
  --history-readbacks "$XQ_HISTORY_READBACK" \
  --reference-data "$XQ_REFERENCE_DATA" \
  --workers 64 --batch-size 16 --output runs/catalog-readback
```

完整读回产生 `manifest.json` 后，向[并行问答生成](PAPER_QUESTION_RENDERING.md)命令添加：

```bash
--catalog-proof runs/catalog-readback/manifest.json
```

此选项要求已声明的位置采样。加载前核验目录、原始输入、来源证明和参考集字节，SQLite 以只读模式打开；抽题顺序、预算、难例比例、累计答案频率及分割沿用原合同。每条最终问答仍完整复验历史并生成原生金标。生成期间的输入改动拒收，完成清单绑定目录与证明。课程完整读回使用保存的同一目录选项，不能通过 CLI 覆盖。

省略选项时继续使用原来的临时建库流程，旧默认清单不新增字段。配置、输入或未完成输出改变时使用新目录；中断后的部分目录复核不能作为可用训练证明。

## 已测范围

真实六盘棋各分割两盘，原记录与分割未改，完整历史及所有合格前缀经生产与第二入口复演。392 个根、3312 个分类目标完整核对；212 条问答覆盖每个分割的全部 26 类题型。重新建库与只读复用的题目、答案、顺序、累计频率及选样报告一致，复用数据完整读回通过，目录字节不变。

独立入口另外检查 26 份约 200.92MB 绑定，按原完整历史路径重算全部 392 根及 3312 个目标，再复验 212 条问答。规则与答案辅助函数共享，不代表另一套独立棋规。95 份执行模块与含测试配置的 178 份源码／测试核验；全部 1364 项检查通过。见[轻量证据](../evidence/native-position-catalog-v1.json)。

正式目录已完成全量复核：55110930432 字节目录、6301835 个根、52964340 个目标与 105352 盘来源棋局逐项匹配；训练／验证／独立测试根为 5099559／602056／600220。48 份约 126.27GB 输入绑定已核验，完成清单固定目录身份。六盘对照只证明接口；全量证明另见 [生产启动证据](../evidence/native-position-catalog-production-v1.json)。

新的 `xqgeneral-paper-curriculum-preparation-v7` 使用该目录与 32 个有序原生问答进程，语料写入 `data/paper-courses-v5`，后续四课写入 `runs/curriculum-paper-v5`。CPU 准备固定 95 份模块，GPU 提取／课程固定原四卡控制的 88 份模块；讲解等待器 `xqgeneral-paper-followthrough-catalog-v1` 保留自己的 88 份 GPU 模块与 89 份参考预检模块。训练配方的非路径字段、题目预算及 10758 条讲解训练标签未改。

入口、固定配置与源码分别保存在 `runs/paper-position-catalog-v1/worktree/runs/paper-curriculum-preparation-v7` 和同级 `paper-followthrough-catalog-v1`；检查 `state.json`、`plan.json` 与生成日志。交接记录在同级 `native-catalog-handoff-v1`，必须确认新工作进程及两次持续文件增长后，才停止仅属本项目的旧准备／讲解等待任务；旧部分语料、配置和日志保留。新队列的 GPU 阶段等待完成的交接证明，讲解任务另等实际四门课程完成。不要重复启动存活会话；中断后的不可续跑阶段使用新输出，检查各 `plan.json` 的恢复边界。

交接已完成：32 个原生问答工作进程的两次输出观察从 2079245 增至 21909926 字节，旧准备与旧讲解等待会话及其六个进程确认终止；旧会话退役后再次确认产出增至 23998656 字节。旧部分语料、配置及日志均保留。独立入口再复验三个原分割各 128 条完整前缀的历史与原生金标，384 条一致；这只是启动检查，不能代替完整语料读回。见 [交接证据](../evidence/native-position-catalog-handoff-v1.json)。

新的完整问答、正式四课、讲解、蒸馏及最终棋力／教学验收尚未完成。当前进度以 [STATUS.md](../STATUS.md) 为准。
