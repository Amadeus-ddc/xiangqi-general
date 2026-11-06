# 有界的学生搜索批量生成

这条可选路径让多个起始局面分别保留搜索、棋规和引擎状态，再将待生成的模型请求合为一批。原批量分支的2140项完整CPU检查及110项相关检查保留在[原始证据](../evidence/batched-student-search-cpu-v1.json)中。本次将它与严格标签收集、显式推理权重缓存配置组合；当前提交的软件检查以CI和验证回执为准，真实挖掘产量另行测量。

## 启用方式

默认串行行为及原有参数清单保持。显式传入`--student-batch-size`才启用新路径，且必须提供原有`--student-generation-config`随机采样配置。

| 参数／试批配置字段 | 含义 | 缺省 |
| --- | --- | --- |
| `--student-batch-size`／`student_batch_size` | 一次模型解码最多合并的请求数 | 不启用 |
| `--student-workers`／`student_workers` | 独立搜索和引擎状态的线程数，须不少于批量 | 等于显式批量 |
| `--student-batch-wait-ms`／`student_batch_wait_ms` | 首个请求到达后收集同批请求的等待上限，毫秒 | 10 |
| `--student-autocast-cache on/off`／`student_autocast_cache_enabled` | 显式设置学生推理的混合精度权重缓存；试批配置必须是布尔值 | 继承训练检查点 |

批量与工作线程须为正整数，等待须有限且非负。缺少显式批量时不能单独给线程或等待参数。相同配置可经`search_pilot`传入实际矿工，并重新核对原生批量清单。

推理权重缓存可独立使用，也可与批量组合。它不修改检查点配置或参数，在真实Torch上下文中生效；矿工保存实际模式，试批在模式证明不匹配时拒绝进入教师阶段。缓存占用更多显存；它与transformer KV缓存是两个不同设置。新收集显式设置`require_immediate_mate_consistency: true`，与批量和缓存开关共同传递，所有原有接受、拒收和保留隔离门槛继续执行。

新路径使用`bounded_independent_root_workers_v1`调度与`independent_request_multinomial_v1`采样。源局面／生成序号的种子派生规则保持，每个请求独占随机数生成器。不同GPU批次形状和填充可能改变浮点计算，因此不承诺跨真实GPU重分批的逐比特输出一致；这与有限训练的精确恢复是不同范围。

## 搜索与输出

每个工作线程独占矿工和引擎，其构造、使用和关闭均发生在同一线程。已完成但尚未按原顺序写出的根也计入在途上限，最多为线程数的两倍。模型和专家特征的实际批量计算始终由调用者线程执行，工作线程只提交请求。接受、拒收、递归、保留分割与原始错误规则保持，不用引擎替换模型评测输出。

原`results.partial.jsonl`仍按输入顺序落盘；成功引擎查询在同一原始账本中串行写入。新增`student-batches.partial.jsonl`在任何工作线程收到回答前，先写出实际同批输入、问题、63位种子字符串及原始回答。错误或数量不配对的输出使该次采集失败，不能作为成功清单。取消、模型错误、账本写入失败及上下文关闭错误均经过控制；续跑必须保留原批量账本并检查其策略、序号与配对。

原始回答和共同批次必须连同配置、源码及输入绑定一起保存。一次失败可能留下尚未提交完整根的查询，不能将它们算作已接受标签。新的输入或执行源码需要新的试验目录。

## 已完成的真实权重测量与剩余范围

原冻结批量源码已使用同一1792步完整FP32学生和四个原开发历史，执行每次128个新词元、两次重复的真实测量。BF16计算下，单条关闭权重缓存的中位吞吐为21.80词元/秒；四条开启缓存为105.11词元/秒。单条兼容接口和私有随机流、单条缓存开关分别逐词元相同；单条与四条的实际文本不逐词元相同，批次形状带来的数值差异保留。

同一冻结分支还通过有界压力检查：四条请求掩码左填充到2048输入宽度，各强制生成4096个词元，实际16384个词元、峰值34.64 GiB；实际KV缓存长度增长正常。这些强制输出不作为标签。实测与[训练质量诊断](EXPANDED_DISTILLATION_DIAGNOSIS.md)的[共同证据](../evidence/expanded-distillation-diagnosis-v1.json)分别记录源码和产物绑定。

短解码速度及有界容量不等于整根搜索／教师汇总／严格收取吞吐，也不证明棋力或讲解改善。当前集成源码新增配置入口后，仍须通过完整软件回归和实际挖掘试批。最终棋力、可靠教学及独立测试仍待完成。

The [actual integrated 128-root pilot](BATCHED_STRICT_SEARCH_PILOT.md) completed mining, full-BF16 consolidation and strict collection with 89 accepted records. This closes the integrated-generation qualification for this software profile; model-quality acceptance and the 4096-root production output remain separate.
