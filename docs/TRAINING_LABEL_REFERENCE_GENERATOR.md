# 原生训练标签引擎参考生成器

`xqgeneral.training_label_reference`对已接受的原始讲解训练标签逐项运行Pikafish评分，然后由独立Python子进程重新解析留存请求、核对完整历史并重算损失。它直接生成[讲解数据筛选入口](ENGINE_QUALIFIED_EXPLANATIONS.md)所需的参考包，不依赖服务器上的实验脚本。

输入须具有已完成的原始语义接受清单。生成器只打开清单和`train.jsonl`，不打开验证或独立测试文件；训练字节、原始记录、完整历史、推荐走法和主变化均保留。已有语义接受声明沿用，正文不重新打分。该入口使用POSIX文件锁和CPU引擎，不加载学生、语言教师或专家特征缓存。

## 生成与筛选

长任务放在tmux或调度器中。完整参考覆盖原训练文件的全部行，默认每次引擎请求100万节点、两个工作进程、每个引擎一个线程：

```bash
python -m xqgeneral.training_label_reference \
  --data data/reviewed-original \
  --output runs/teacher-label-reference-attempt-1 \
  --executable vendor/pikafish/src/pikafish \
  --weights vendor/pikafish/src/pikafish.nnue \
  --nodes 1000000 --workers 2 --threads 1 \
  --artifact-root .
```

成功后，`full/manifest.json`绑定全部原训练行及逐项结果，`full/independent-readback.json`保存独立重算结果，`full-readback-exit.json`记录实际子进程退出和结果哈希，`full-pipeline-exit.json`证明两阶段完成。随后可生成保持原文和保留集字节的训练子集：

```bash
python -m xqgeneral.engine_qualified_explanations \
  --data data/reviewed-original \
  --reference runs/teacher-label-reference-attempt-1 \
  --artifact-root . --workers 2 \
  --output data/engine-qualified-attempt-1
```

筛选沿用严格的预期得分损失小于0.10规则，推荐走法和完整主变化每一步都须通过。100万节点是每次请求的预算；一条标签可能需要多个不限招及限制走法请求。

## 留存输出与续跑

`state.json`记录当前阶段和完成数量；逐项原记录、原答案、评分、按顺序保留的引擎请求在`full/rows/`。每个请求保留原始响应、初始FEN、完整历史、节点预算和限制走法。`attempts/`保留实际读回子进程、退出码、日志以及失败请求信息，不覆盖失败尝试。

若不限招请求结束时没有原`bestmove`的完整评分，只允许再对同一合法最佳招进行一次相同预算的限制搜索。原响应、额外请求及其严格评分均保留；额外评分不得改变原最佳招，也不改教师答案。其他协议错误会终止任务。

续跑时重复原命令并添加`--resume`。输入、引擎／权重、节点预算、工作数量、源码及读回入口都必须与原计划相同；仍存活的文件锁所有者会阻止重复启动。已经完成的行先按原始请求重算验证，再跳过评分；完整生产结果及读回证明可重新绑定，完成输出不重写。配置变更使用新目录。独立读回子进程继承原所有权锁；父进程异常退出后，只要读回仍活着就不能启动重复任务。捕获到父进程中断时，会终止并等待本次拥有的读回子进程退出。真实隔离子进程检查覆盖父进程退出后的锁保留和中断清理；实际8条读回也现场捕获了继承的锁描述符。

中断发生在完成清单或退出证明发布期间也能续跑：已有行、指标和读回字节会被核对并保留。再次调用已经完成的`readback.py --reference <参考目录>`只复核已完成证明及制品绑定，不宣称又做了一次全量数学重算。

## 实际验证范围

2026-10-20，公开的107份源码与冻结执行副本逐字节相同。原生CLI从原10758条训练标签中选定8条进行`capability`运行，共46个主变化半回合、58次真实引擎请求，包含1次上述同招额外评分。实际独立读回子进程退出成功；另一次阻止任何子进程启动的读回重新核对全部请求和损失。原标签、训练／保留集版本和正式等待队列不变。见[机器可读证据](../evidence/native-training-label-reference-capability-v1.json)。

小范围检查可显式使用`--scope capability --indices 0 326 329 1024 4096 8192 10000 10757`。索引属于输入的原训练行；`capability/`不满足正式筛选所要求的`full/`覆盖，不能用它代替完整参考。

40项针对性检查覆盖完整生成到筛选接入、实际独立子进程、保留集不读取、断点续跑、文件锁、完成证明写入中断、参数／文件改动拒收和错误协议拒收。其中25项是本入口的新检查；受控标签和引擎响应只证明程序合同。

原10758条标签的[全量引擎参考](TRAINING_TEACHER_ENGINE_REFERENCE.md)已由此前的独立实验完成并读回。本次没有用新生成器重跑该全集；8条的全部通过率不代表全库通过率、学生泛化、讲解语义质量或实战棋力。正式讲解仍等待四课完成及实际四卡全参数预检。
