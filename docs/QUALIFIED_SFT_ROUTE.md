# 筛选后讲解数据的训练接续

2026-11-05（UTC+8），v2开发控制器在已完成正常组384条模型／引擎结果后，因缺少4040个缓存依赖绑定而退出。原始数据保留并完整复核，[修复后的显式复用](PAIRED_RAW_REUSE.md)继续要求所有输入精确一致。清零／错配由`runs/initial-sft-remaining-raw-validation-v1`执行，v3串行拥有者位于`runs/post-sft-full-bf16-development-v3`，在完整源码检查和三组原始结果就绪后继续盲评／384局／512根试批。实际完整27B教师已有[单条四卡原生推理证明](FULL_BF16_TEACHER_NATIVE.md)。旧v2错误及各自源码、配置、输出均保留。

2026-11-05（UTC+8），[同设置课程能力核验](INITIAL_SFT_RETENTION.md)完成。训前GPU原始回答在`runs/initial-sft-retention-control-v2/before-normal`，凭据会话内完整收尾在`runs/initial-sft-retention-control-v3-readback`，五组再次评分在`runs/initial-sft-retention-readback-v2`。旧CPU收尾及首版等待器已在确认拥有权和无子进程后退役，输出保留；未知GPU任务保持。原v8课程复查正常结束，开发串行v2已接续讲解评测；2456步权重的独立诊断在`runs/initial-sft-terminal-retention-v1`，未更换当前1792步基线。

2026-11-04 22:57（UTC+8），[完整初始讲解SFT已结束](INITIAL_SFT_COMPLETED.md)，实际2456次更新、选1792步。评测遗漏用户课程前缀传递的问题已修复，1906项完整CPU检查和真实选中模型CPU校验通过；新的复查v8、开发串行v2使用112模块冻结评测来源。以下保留准备期、开训和失败的历史快照，现役路径统一见[当前运行定位](#本服务器运行定位与恢复)。

2026-11-04 17:43（UTC+8）首读回：真实全模型容量及完整状态4对2+2门禁已通过，v7已进入正式讲解，完成前三次更新。训练预算仍为四轮2456次更新，后续按小时观测。课程复查和开发／对弈／真实试批各有独立拥有者，按顺序在实际SFT完成后启动，来源未知任务保持。见[真实开训证据](FULL_DECODER_SFT_START.md)；以下准备期与失败记录保留其原时间范围。

2026-11-04分片接续：当前v7主队列在`runs/pinned-base-postcourse-v7`，模型流水线在`runs/paper-sft-fsdp2-v1/pipeline`，使用109份独立冻结源码。1889项当前源码和1790项实际消费者完整检查、构建／发布已通过，原始全模型4对2+2门禁保持，通过才执行正式讲解。9823条训练数据、四轮2456次更新、完整FP32主参数与BF16计算、所选42k父权重及学习率保持。按`queue-worker.json`核验拥有者，查看`queue-state.json`、模型`state.json`及`monitoring/hourly.jsonl`；失败后保留旧输出，再按原合同恢复或新建配置实验。后续开发评测、对弈和真实搜索使用独立接续合同，不能复用v6中指向旧配方及固定单卡的配置。

2026-11-04更新：v6主队列已失败退出，当前无正式讲解训练。真实连续4步和恢复前2步均保存完整检查点，恢复第3步反向显存不足。恢复额外复制本地Adam状态的问题已修复并完成软件与四卡小型存储对照；来源未知GPU占用增长后空闲约36GiB，低于此前约64.57GiB实测峰值，因此先适配参数／梯度分片，再重新执行完整模型容量及4对2+2门禁。以下v6目录表是已保留的历史定位，不是仍在运行的拥有者。

2026-11-03，[四门课程完成读回](PAPER_FOUR_COURSES_COMPLETED.md)已确认结束第四课并选择42000步。原生CPU导出与四份所选权重核验已完成。[基础资产绑定修复](SFT_PINNED_DECODER_ASSETS.md)增加显式权重清单；新的108模块主队列已完成拥有权交接，须通过其完整CPU检查及原生四卡容量／续跑证明后才能正式训练讲解。旧90GiB空闲门槛等待器已退役，容量由原生最长样本与4步对2+2实测核验；本记录未提供GPU容量或正式SFT完成结论。讲解后四课复查的109模块源码保持，其拥有者也已交接。以下带日期的早期交接记录保留其历史范围。

当前主控制器位于 `runs/pinned-base-postcourse-v6/`，回放控制器位于 `runs/pinned-base-post-sft-replay-v6/auto-initial-replay/`；分别查看 `queue-state.json`／`queue-handoff-verification.json` 和 `state.json`／`handoff.json`。完整四课、9823条讲解训练、四轮2456次更新、原词元与特征数据、开发评测／完整对弈／真实搜索pilot的配置保持。前三次真实预检均完成1次更新后显存不足；随后Adam分片试跑完成2次更新，但检查点收集超时。当前CPU收集实现的[显存控制](FULL_DECODER_MEMORY_CONTROLS.md)须另行通过完整软件检查和真实容量／恢复证明，随后才能正式训练。旧控制器和失败的诊断尝试均保留原文件。恢复须读取实际拥有者和模型阶段合同，避免重复启动仍存活的拥有者。

2026-10-26，接续目标已改为[用户直接指定的三课前缀](USER_AUTHORIZED_COURSE_PREFIX.md)：第一课24000步、第二课30000步、第三课26000步；第四课已从选定权重实际开训。第二课原最低32000步和第三课原耐心停止规则未满足的事实保留，源第三课额外564次更新单列为弃用续训。新的108模块消费者通过1738项CPU检查，并绑定该三课声明；正式讲解仍须等待真实四课、原生导出及全解码器容量／恢复证明。见[实际交接](PAPER_FOURTH_COURSE_START_26000.md)和[核验记录](../evidence/direct-third-course-26000-handoff-v1.json)。

2026-10-20的[首次筛选数据迁移](../evidence/qualified-sft-route-real-v1.json)保留为历史记录。筛选数据、原生词元预检、原索引及完整四轮遍历安排继续复用；四课完成清单已核验；仍须在严格四课导出及新的四卡全解码器容量／恢复证明通过之后才训练讲解。课程GPU执行保持原89份冻结源码。

## 实际数据与批次

讲解数据为9823条训练、384条验证、512条独立测试；推荐首步和完整主变化继续使用[原严格筛选规则](ENGINE_QUALIFIED_EXPLANATIONS.md)。原生预检检查全部10719个特征上下文，缺失0；仅编码10207条训练／验证，最长649／625词元，在1024上限内完整保留监督。独立测试没有分词、训练或选模。

使用真实训练索引及原生有限遍历器逐批核对四个rank、四轮的完整索引顺序：每轮9823条有效样本各出现一次，每轮614次更新及1条分布式填充。四轮编译为2456次更新，39292条有效样本呈现、4条填充，共39296条全局样本。批量16、微批量1、四卡、FP32全解码器、词元损失和学习率等原设置保留。这是数据安排读回；四课现已结束，选中42000步父权重的字节已另行核对；另行原生导出已通过，现役接续入口仍会在自身流水线目录建立同源交接产物。

## 缓存凭据与读取范围

实际4041份缓存文件合计3298928763121字节，全部匹配此前完整字节验证形成的凭据及当前文件身份。原生预检实际新哈希276356735字节。复用计数可能多次累计同一文件，不能当作新读取量或独立数据量。新报告明确不声称本次重新读取全部缓存，也没有重新计算或读取特征数组值。

讲解预检及原始讲解评测CLI显式读取`XQGENERAL_VERIFIED_INPUT_RECEIPT`。该变量应指向实际完成字节验证后形成的不可变输入凭据；缺省仍执行原有新哈希。文件身份改变时拒收，不能把当前签名直接附到未验证的预期哈希。评测入口在控制夹具中实际打开分片缓存，保留原始非法回答，未读取已验证分片的二进制字节。控制夹具不测棋力。

原生数据预检使用的104模块历史快照保持原样。当前108模块消费者另增加显式前缀交接及讲解入口的绑定检查，已独立冻结并通过1738项CPU检查。已完成的GPU课程使用自己的89模块快照，不能以模块数代替实际字节核验；真实全解码器预检仍在四课完成后执行。

## 顺序与验收边界

新等待器先确认四课完成及项目完整检查，再要求四卡容量与全参数连续／续跑证明，随后训练讲解。开发验证仍比较正常、清零和错配记忆，并保留全文原始错误；辅助正文教师为官方完整BF16 Qwen3.8-27B，未替换量化权重。对弈基线为64个验证开局、红黑交换和三档引擎预算，共384局，步数截断单列。

512根搜索保持既有未用棋谱池、递归规则与完整教师设置，用于量测真实可用标签产量。后续正式搜索预算与蒸馏轮次须由实际产量、开发棋力和讲解结果决定。六项受控门禁检查覆盖真实隔离tmux拥有者、错误进程身份、未完成四课、缺失／失败检查及变化的检查日志；它们不等于真实四课或训练完成。

四课及初始讲解已完成；足量真实蒸馏、最终独立棋力／教学验收及训练模型交付仍待完成。原始数据、运行等待器与机器路径保留在服务器运行目录；公开发布的数据／权重范围和许可证仍按模型卡及第三方说明核定。

## 本服务器运行定位与恢复

初始讲解已完成，原拥有队列`runs/pinned-base-postcourse-v7/queue-exit.json`为0。实际模型是`runs/paper-sft-fsdp2-v1/pipeline/sft/adapter.pt`，选1792步、完整4694274592参数；当前尚未通过教练交付验收。

| 项目 | 当前定位 |
| --- | --- |
| 训练完成证明 | `runs/paper-sft-fsdp2-v1/pipeline/manifest.json`、`sft/metrics.json` |
| 原实际训练来源 | `runs/full-decoder-fsdp2-v1/consumer-source-run/source`，109模块，完整1790项检查 |
| 修复后的评测来源与检查 | `runs/post-sft-evaluation-prefix-v1/source-run`，112模块，完整1906项检查 |
| 四课复查拥有者 | `runs/pinned-base-post-sft-replay-v8/auto-initial-replay`；会话`xqgeneral-pinned-base-course-replay-v8` |
| 四课复查输出 | `runs/pinned-base-post-sft-replay-v8/actual-initial-sft-validation` |
| 后续串行拥有者 | `runs/post-sft-full-bf16-development-v2`；会话`xqgeneral-post-sft-full-bf16-development-v2` |
| 开发／对弈／试批配置 | `runs/paper-sft-fsdp2-v1/development-inputs-v1` |
| 配对讲解与对弈输出 | `runs/paper-sft-fsdp2-v1/capability-validation`、`baseline` |
| 真实搜索试批输出 | `runs/fresh-paper-student-search-yield-pilot-fsdp2-v1` |
| 小时记录与轻量阶段变化 | `runs/post-sft-full-bf16-development-v2/monitoring` |

按各拥有者的`owner.json`核对PID、启动时间、父进程和命令哈希；查看`state.json`、`entry.log`与`exit.json`。当前课程复查、配对讲解、完整BF16教师、384局及512根试批依次使用GPU；没有沿用旧的未实测80／48GiB门槛。实际推理失败仍须保存原输出并按同一合同恢复，配置或源码改变时使用新目录。

SFT原始109模块来源、权重、授权证明和完成记录保持；首次复查v7及开发等待器v1已失败终止，日志保留。后续评测不重跑已经完成的SFT。外层拥有者仍须明确交接，不能因观察超时重复启动存活任务；子阶段是否可续跑以该阶段原合同为准。

## 历史运行定位（v6，已失败）

直接26000步路线使用新的单进程拥有者与108份冻结消费者源码，并绑定实际三课声明及1738项完整CPU检查。原两课等待器随父课程被最新指令退役而结束，其实际退出与历史输入保留；新拥有者须单独确认旧队列终止和四课门禁为假。2026-10-22的1708项检查及2026-10-20的[只读核对](../evidence/qualified-postcourse-runbook-readback-v1.json)均作为历史路线保留。

| 项目 | 现役位置或合同 |
| --- | --- |
| 等待器目录 | `runs/pinned-base-postcourse-v6` |
| 入口与会话 | `postcourse-entry.py`；`xqgeneral-pinned-base-postcourse-v6` |
| 父课程 | `runs/curriculum-paper-user-direct26000-v1`；会话`xqgeneral-course4-from-26000-v1` |
| 显式前缀声明 | `runs/user-third-course-direct-prefix-v1/authorized-prefix.json`及其锁定SHA256 |
| 讲解输出与配置 | `runs/paper-sft-zero1-cpu-checkpoint-v1`及其`inputs/` |
| 训练／验证索引 | `data/astra-explanations-engine-qualified-paper-index-v1/train`及`validation` |
| 复用原生词元预检 | `runs/paper-sft-engine-qualified-v1/token-preflight/manifest.json` |
| 搜索试批配置 | `runs/user-direct26000-postcourse-v1/pilot-owned/pilot.json` |
| 搜索试批输出 | `runs/fresh-paper-student-search-yield-pilot-zero1-cpu-checkpoint-v1` |

`queue-plan.json`绑定不可变源码、配置、输入凭据及阶段合同；`queue-state.json`报告当前阶段；`queue-handoff.json`保存实际拥有权交接。诊断查看`queue-entry.log`和`queue-exit.json`。状态文件或日志不增长不能单独证明进程已经结束，应按保存的PID、启动时间、父进程及命令哈希核对实际进程与会话。

```bash
.venv/bin/python -c 'import json; print(json.load(open("runs/pinned-base-postcourse-v6/queue-state.json")))'
tmux capture-pane -p -t '=xqgeneral-pinned-base-postcourse-v6'
```

外层拥有者没有可直接重复调用的`--resume`。原拥有者确已终止后，需要新的明确拥有权启动／交接，并只复用源码、输入和合同都不变的可续跑子阶段；不可续跑部分改用新输出。所有部分输出和旧交接证据保留。不得直接重跑旧`paper-followthrough-catalog-v1`入口，也不覆盖活任务的`queue-worker.json`。

已完成课程使用89份冻结源码，新讲解消费者使用108份；原104份消费者快照作为历史保留。恢复时核对各自实际字节，不能只比较模块数或测试条数。讲解容量以原生四卡实测为准，旧的每卡92160MiB空闲等待门槛已退役；后续正文验证GPU 0的81920MiB、对弈GPU 2的49152MiB和试批GPU 0的81920MiB等待门槛仍存在，须在各阶段结合真实部署容量另行确认。显存门槛不能代替真实容量／续跑预检。

完整BF16教师的18个分片与1199个张量头已经核对，均为BF16且没有量化配置。本次只读取150872字节头信息，没有读取55562855904字节权重载荷或重新计算全部载荷哈希；这是一项资产就绪核对，不是新的教师推理或模型质量结果。

等待器完成初始讲解、三组开发验证、384局基线和512根真实试批后，仍需依据实际产量安排正式搜索与蒸馏，再锁定交付模型完成[独立验收](FINAL_INDEPENDENT_EVALUATION.md)。`queue-completion.json`不代表整个项目或最终教练模型已经合格。


讲解后四课复查的独立等待器位于`runs/pinned-base-post-sft-replay-v6/auto-initial-replay`，会话为`xqgeneral-pinned-base-course-replay-v6`；按`owner.json`保存的完整进程身份核验存活，读取`state.json`、`completion.json`和`exit.json`判断进展。它使用自己的109份冻结源码及1761项已完成CPU检查，实际讲解模型完成后才在GPU 3空闲至少49152MiB时执行三组2600题。主队列讲解开发验证／试批使用GPU 0，对弈使用GPU 2。当前题目顺序、问法和金标已与42k原始验证逐条核对；实际讲解后成绩仍待产生。
