# 辅助训练之后的搜索保留位置

七类辅助语料从原验证历史增加了数值与走法题。其百万节点参考搜索产生的全部候选变化，
包括没有进入最终数值答案的长尾，均已用于隔离该语料。后续学生搜索、汇总和标签合并
必须保留同一范围，不能退回原始保留位置文件。

## 从已完成语料重建

```bash
python -m xqgeneral.search_protection \
  --data runs/oracle-policy-corpus-production-v2/dataset \
  --output runs/oracle-policy-corpus-search-protection-v1/artifact
```

入口只接受原生完成的七类语料。它核对生产清单、计数、原生解码源码闭包和原始保留位置，
逐字节重建所有来源的验证行及去重顺序，重新解析原验证参考的全部候选变化，并重建新增
验证题。原始语料已排除的参考变化也进入保护集；不能只从最后保留的六步数值答案提取。
最终位置及红黑镜像的并集，必须与语料生产时的扩展保留计数一致。

导出器不打开原始独立测试标签，不读取训练答案，不运行引擎、学生或教师。原始大范围
保留覆盖复用已完成权威文件；实际重新计算的范围是全部验证行和原始参考变化。
输出 `protected-positions.json`、`counts.json`、`contract.json` 和完成清单。
完成后加 `--resume` 会重建并核对，保留原输出字节及修改时间；部分或失败输出不覆盖。

## 搜索、收取与合并

原生搜索入口可显式添加：

```bash
python -m xqgeneral.search_distillation \
  --checkpoint ACTUAL_SELECTED_CHECKPOINT --data ORIGINAL_BASE_DATA \
  --features EXACT_FEATURE_CACHE --recorded-inputs COMPLETED_RECORDED_POOL \
  --protection-manifest runs/oracle-policy-corpus-search-protection-v1/artifact/manifest.json \
  --output FRESH_SEARCH_OUTPUT
```

以上是参数示例，须补齐实际引擎、采样、资源租约及冻结运行配置；GPU2仍按用户要求排除。
相同扩展也支持原生 `--student-game-inputs`。保留证据必须属于该原始棋谱权威，学生棋局
必须声明同一原始权威；直接未认证输入池不能使用该选项。后续模型准入还须核对实际
选中检查点及完整配对开发评测，不能用这个位置清单替代模型完成证据。

搜索在模型生成前完成保护重建，将原范围与扩展位置一起用于根、子节点、原始可达变化
和推断目标检查。原范围计数单独保留。证据与全部依赖进入搜索合同，改变后不能原样续跑。

`collect_search` 从原矿工声明读取同一完成证据，独立重建保护范围，重新检查原始查询和
教师回答；缺失依赖或伪改范围拒收。`merge_search` 重放相同收取检查，保留扩展保护身份，
同时保持原始共享权威的原始计数。具有不同保护证据的叶子，以及扩展保护与旧无声明叶子，
不能静默合并。需要混合旧数据时应重新在同一范围内完成收取验证。

旧初始模型入口未声明该选项时保持历史合同。[后续模型试批](DESCENDANT_SEARCH_PILOT.md)
会核对实际完整模型、父模型链、完整配对评测和匹配辅助语料的保护证据，再将该选项
实际传入矿工。保护导出本身不能替代模型准入；三卡正式搜索蒸馏及真实新标签产量另行验收。

## 证据边界

CPU控制使用真实规则回放验证全部原始参考长尾和镜像，另以受控上游证据检查原生搜索、
标签收取及合并边界。它们不是实际新学生搜索、真实教师正文或训练收益。完整生产导出
仍必须等实际七类语料及原生复建完成，并在独立运行目录读回。最终独立评测不训练、不选模。
