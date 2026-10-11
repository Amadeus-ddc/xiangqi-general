# 并行原生课程问答

`paper_curriculum` 的显式 `--question-workers` 把完整历史复验与原生金标生成交给 CPU 工作进程。父进程仍按来源顺序锁定课程／题型／未来步数、抽取问题并更新累计答案频率；结果按相同顺序写入。每条历史仍检查所有终局前缀，失败不会变成可用训练标签。等待队列最多保留两倍工作进程数的批次，避免预先装入全部题目。

默认一个问答进程，保留原串行生成、颜色镜像和旧清单字段。并行模式要求显式位置采样且关闭颜色镜像；它不修改四课根预算、难例混合、抽题权重、未来步数、棋局归属或验证／独立测试隔离。`--position-workers` 与 `--position-batch-size` 仍控制此前的原生分类，两组参数职责不同。

## 使用

输入变量指向自己完成并绑定的根文件、来源清单及旧参考集。使用新数据与运行目录，长任务放在 tmux 中；不要替换正在运行的冻结源码。

```bash
.venv/bin/python scripts/freeze_run.py --output runs/paper-parallel/source-run -- \
  .venv/bin/python -m xqgeneral.paper_curriculum \
  --roots "$XQ_ROOT_FILE" --source-manifests "$XQ_ROOT_MANIFEST" \
  --reference-data "$XQ_REFERENCE_DATA" \
  --sampling-profile author_answer_frequency_xiangqi_v1 \
  --position-sampling-profile author_task_position_budget_mix_xiangqi_v1 \
  --task-order-profile scarce_dynamic_tasks_first_xiangqi_v1 \
  --task-position-budgets configs/paper-task-position-budgets-v1.json \
  --position-workers 64 --position-batch-size 16 \
  --question-workers 32 --question-batch-size 4 --no-color-mirror \
  --seed 20261071 --output data/paper-courses-parallel-v1
```

完成后按清单保存的抽样与工作进程设置重新生成每条问题和金标，不用 CLI 覆盖这些设置：

```bash
.venv/bin/python scripts/freeze_run.py --output runs/paper-parallel/source-run -- \
  .venv/bin/python -m xqgeneral.paper_curriculum \
  --readback data/paper-courses-parallel-v1 --output runs/paper-parallel/readback
```

## 实测边界

固定的 64 盘实战、64 个训练侧完整历史（最长 161 个局面）产生 256 条控制题。串行／16 进程每批 4 根／32 进程每批 2 根耗时 33.63／4.82／3.60 秒，32 进程局部约快 9.34 倍；实际执行原生复验的进程为 1／15／23，配置数量为上限。192 条复验记录覆盖三次全部 64 个历史。该时间仅测这批生成，不能外推为全库、课程训练或整个流程的加速。

另把同批根用于全部 26 类题型的构造选择流，3340 条问答与此前 93 模块冻结实现逐字节一致，独立重算问题、金标、身份、每根不同查询及累计答案频率。跨题型重复根仅属于这个控制，正式目录仍执行原课程内查询局面与颜色去重。61 份约 60.60MB 的输入／输出绑定读回，94 份新模块和含测试配置的 175 份源码／测试核验；全套 1339 项检查通过。见 [轻量证据](../evidence/paper-question-render-v1.json)。

当前生产任务仍使用原冻结版本。完整局面目录的来源／目标读回及新准备交接尚未完成；保存的只读目录快照不能单独作为可训练数据证明。完整新问答、专家特征与预检完成后，才启动正式四门课，再进行讲解、蒸馏及独立棋力／教学验收。
