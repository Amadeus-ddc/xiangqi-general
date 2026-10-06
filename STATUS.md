# Xiangqi General 当前状态

2026-10-07。最终目标是能用于个人象棋学习的强走法与可靠中文讲解。训练课程与蒸馏轮数依据独立评测调整；当前不能宣称已达到该目标。

## 已验证

- 原型基线保留：冻结 Px0 与 Qwen，四个 384 宽桥接块，80 步静态课程；12 条生成样例中 8 条正确。这仅证明链路可执行。
- 原生输入编码：原始 102 个局面逐平面一致。
- 完整网络数值对照：34 个局面，2062 个策略输出、胜和负、剩余步数与四层特征均一致。胜和负最大误差 `6.85e-7`，策略最大误差 `5.60e-6`。参考为官方 C++ 编码器与未修改的 C++ ONNX 导出器，通过 ONNX Runtime CPU 执行。证据见 `evidence/native-network.json`。
- 43 项 CPU 测试通过，覆盖规则与完整历史终局、分割、梯度、重复、长将与长捉，以及新增词元冻结、对称数据、教师身份、引擎 MultiPV 中断、全量 SFT 初始化、搜索递归、追加标注合并、盲评协议和引擎走法课程的保留集隔离。新增检查验证不同答案长度下的样本损失权重、梯度累积和续训确定性合同。独立走法评测拒收非法或终局后走法，并按原始输出统计全样本与合法样本条件指标。裁定锁定 `pyffish 0.0.90 xiangqi AXF`，保留完整历史。
- 课程数据生成器加入空格、零数量、棋子位置、子力、吃子、将军与未来非法反例；训练/验证/测试按棋局划分并检查当前及未来局面重叠。
- Git 已建立，原始原型有独立基线提交。软件包改名 `xqgeneral`；本地目录已同步改名为 `xiangqi-general`，项目环境与可编辑安装路径已修复。
- [GitHub 仓库](https://github.com/Amadeus-ddc/xiangqi-general) 已同步并合并首个源码里程碑；Python 3.11/3.12 的 CPU CI 均通过。
- 第一版专家桥接与纯语言 LoRA 均完成四门课程，每门 800 步。264 道平衡验证题的原始生成正确率分别为 **50.8% / 52.7%**，置零专家为 **43.6%**，打乱专家为 **52.3%**。没有证明专家带来稳定收益，也没有证明强棋力。证据见 `evidence/curriculum-validation-v1.json`。
- 发现并修正第一版根局面全为黑方行棋的采样缺陷。红黑对称增强后为 73,772 条课程题、3,354 个完整历史上下文，双方各 1,677 个；棋局及当前/未来局面跨分割重叠均为零。派生样本不算新独立棋局，见 `evidence/balanced-data.json`。
- 新方案的 16 个 768 宽桥接块与 105 个专用棋盘词元可在 GPU 上训练，合计 139,920,416 个可训练参数。连续 4 步与 2 步后续跑到 4 步的 258 个可训练张量逐位相同，初始损失、84 个监督词元和执行源码身份一致，见 `evidence/wide-training-resume.json`。四步验证不代表已完成新课程。
- 用户指定的 **GPT-6-Astra Low 子代理**生成了 736 条初始中文讲解（512 训练、96 验证、128 测试）。结构、走法、变化和声明事实全部通过机械校验；另有 72 条训练/验证讲解经交叉神经抽检通过。加上明确标记的颜色对称派生项，共 1,472 条；派生项没有再次调用教师或重算引擎分数。见 `evidence/initial-teacher.json`。抽检不是完整战略质量证明或人工评分。
- 扩大专家桥接模型完成四门课程：执行步数为 2500/1750/1250/1750，各门恢复验证集最佳检查点。最终课程验证 NLL 为 0.8505；24 条抽样生成仅 12 条完全正确，264 道平衡验证问答原始正确率为 46.6%，置零专家为 0%，打乱专家为 43.6%。见 `evidence/curriculum-v2-bridge.json`。课程损失下降尚不足以证明棋力。
- 全量讲解训练通过两步 GPU 执行检查：41.62 亿可训练参数，FP32 参数、BF16 计算，解码器参数实际变化，峰值 77.66GiB；见 `evidence/full-decoder-sft-verification.json`。这不是完成的讲解模型。全局批量 4、微批量 1 的连续与续跑结果逐位一致，见 `evidence/microbatch-resume.json`。
- 专家组初始讲解 SFT 已执行 1408 步，选择验证集最佳第 896 步；NLL 从 8.5569 降至 0.6584，峰值 77.59GiB。见 `evidence/explanation-sft-v2-bridge.json`。损失不能代替走法与正文质量评测。
- 同预算纯语言组讲解 SFT 已执行 1408 步，选择第 896 步；NLL 从 2.0752 降至 0.6413，峰值 75.65GiB，输出哈希已读回。见 `evidence/explanation-sft-v2-text.json`。同样的 96 局原始讲解评测中，38 个推荐走法合法、17 个无明显失误，完整变化合法率为零；见 `evidence/explanation-v2-text-validation.json`。
- 中间检查点已完成 32 个验证局面的独立 100 万节点评测：31 个可解析，7 个推荐走法合法，4 个未出现引擎判定的明显失误，完整结构及主变化合法率均为零。原始错误和引擎证据均保留；见 `evidence/interim-explanation-v2-001.json`。这解释了追加走法课程的必要性，不是最终模型结论。
- 已完成 SFT 的最佳检查点另经 96 局验证：全部可解析，22 个推荐走法合法，14 个无明显失误；完整结构与主变化合法率仍为零。见 `evidence/explanation-v2-bridge-validation.json`。该模型不能作为可靠教练交付。
- 本地完整 BF16 教师已实际完成 8 条中间讲解盲评，事实与战略维度平均均为 1/5，全部指出了无依据声明。首次被截断的 1 条经更大输出预算真实重试，最终评分全量覆盖、零拒收。见 `evidence/interim-neural-prose-review.json`。该神经裁判与汇总教师共享基础模型，评分不是人工评价或绝对事实标准。
- 官方 **Qwen/Qwen3.8-27B 未量化完整权重**已部署：18 个分片、55,563,006,776 字节，全部官方 SHA256 匹配；真实加载 27,356,728,560 个 BF16 参数，一张 H20 占用约 50.96GiB，完成真实生成。见 `evidence/full-teacher-weights.json` 与 `evidence/full-teacher-deployment.json`。已有 Q4_K_M GGUF 不用于本项目。
- 新增 7100 个独立引擎搜索局面，形成 16,382 条训练走法题、192 条验证题、256 条测试题，并保留四门课程回放；棋局与当前/未来局面跨分割重叠均为零。16 层专家特征已缓存，共 17,554 个完整历史上下文。见 `evidence/move-quality-data.json` 与 `evidence/move-quality-feature-cache.json`。这些是引擎标签，不是神经讲解。
- 新走法评测已实际执行：字典课程第二门检查点在补训前的 32 道验证题中，20 个格式有效、9 个走法合法、8 个无明显失误。独立裁判每局使用 100 万节点，保留全部原始回答；见 `evidence/move-quality-before-training.json`。抽样含颜色派生项，不是 32 个新独立棋局。
- Astra 新增 2048 条原始训练讲解已完成，另交叉抽查 72 条。一处无依据的“车炮交换”表述由原作者修正后独立复核；原标注及拒收意见保留。合并后共 3363 条原始标注和 3363 条颜色派生项，训练/验证/测试分别为 6278/192/256；全部输出哈希已读回，验证/测试与原数据逐字节一致。累计神经抽查 216 条，见 `evidence/initial-teacher-expanded.json`。本批抽查来自固定的已完成作者前缀，不能据此估计全体正文正确率。
- 扩充讲解已通过数据检查：最长训练样本 935 个词元，没有超过 1024 上限，全部完整历史专家特征存在；检查 146,042 步训练变化，与保留集根局面、未来及分支局面重叠为零。见 `evidence/expanded-teacher-data-readiness.json`。测试标签仅用于长度与隔离检查，未参与模型选择。
- 新样本平均损失在实际 BF16 桥接训练中通过连续 4 步与 2＋2 步续训对照：258 个可训练张量逐位一致，4290 个监督词元相同，优化器与随机状态也一致；见 `evidence/example-normalized-resume.json`。首次默认计算的验证失败，重复完整图检查确认前向相同但反向梯度存在差异；启用确定性算法后消失。失败记录保留，该证明不涵盖完整解码器或多卡新损失续训。
- 已完成讲解检查点实际搜索 16 个训练根局面，全部因根候选无效而未被接收；保留了 17 次学生根回答和 1 次递归下降。见 `evidence/search-v2-diagnostic.json`。未执行汇总或蒸馏训练，空产出不算完成蒸馏。
- 新样本损失完成全量解码器两步 GPU 检查：41.62 亿 FP32 参数、BF16 计算，参数实际更新，峰值 79.59GiB，输出哈希全部读回。见 `evidence/example-normalized-full-sft-verification.json`。这是执行检查，不是训练完成或完整解码器续训证明。
- 冻结阶段新增可选 FP32 可训练参数。实际两步检查确认 40.22 亿冻结解码器参数保持 BF16，1.40 亿可训练参数为 FP32；棋盘参数约 `2e-6` 的更新被保存。见 `evidence/frozen-float-master-verification.json`。首次精度报告选错新增参数的记录及检查失败已保留；后续正式走法配置为 `configs/move-quality-v2.json`，旧运行不改变。
- 两门基础课程后的走法预训练完成 2000 步，选择第 1500 步；NLL 从 3.1929 降至 0.9849。192 道验证题原始合法率 87.0%、无明显失误率 77.1%；置零专家两者为零，打乱专家分别为 56.3% 与 29.7%。见 `evidence/move-quality-warm-{training,normal-validation,zero-validation,shuffled-validation}.json`。这组含 96 个原始局面及 96 个颜色派生项，是两门课程初始化的额外实验，不是四门课程匹配对照。
- 冻结 Px0 自身的原生合法策略在相同 192 个验证局面中有 191 个无明显失误。见 `evidence/frozen-expert-move-validation.json`。它显式使用规则合法走法掩码，没有语言解码器；该结果用于定位专家能力与语言模型之间的差距，不能算成桥接模型棋力。
- 搜索挖掘、真实 BF16 汇总收集及完整对弈入口已实现并通过受控测试，真实搜索蒸馏与对弈结果尚待产生。
- 终端学习入口支持棋盘显示、推荐讲解、悔棋和完整历史保存/续读；命令行入口已检查，真实训练模型的学习体验仍待验证。

## 正在完成

- 红黑平衡数据上的扩大纯语言基线已完成四门课程，执行 4000/2750/4000/3250 步，264 道平衡验证问答正确率为 51.9%；见 `evidence/curriculum-v2-text.json`。完整讲解 SFT 与 96 根局面评测均已完成。
- Astra 已生成新增 579 个训练根局面的初始讲解，并对新增训练项抽检 72 条；一处“回吃”表述已修正并复核。合并后的 1,315 条原始标注与颜色派生项共 2,630 条（2182 训练、192 验证、256 测试），全部结构校验与输出哈希读回通过；验证/测试文件与原版本字节一致。共 144 条神经抽检，见 `evidence/initial-teacher-full.json`。
- 棋盘字典对照两组正在执行四门课程，配置 `configs/research-v3.json`。双方第一门课程均在 24 条生成样例中答对 22 条，不能推广为整体正确率。这是论文专家输入条件以外的改动，结果与专家消融需单独报告。
- 四门字典课程完成后，两组将自动进入 `configs/move-quality-v2.json` 的真实引擎走法课程，再进行讲解训练。课程按验证集选检查点。
- 第二门字典课程后的走法预训练及全量消融均已完成。其选定检查点已开始 `configs/explanation-sft-v3.json` 的扩充讲解 SFT；这是自适应的额外实验，四门课程的匹配对照仍单独进行。
- 扩充讲解收集与数据检查已完成。后续 `configs/explanation-sft-v3.json` 按样本平均损失混合讲解、走法题及四门基础课程回放，完整解码器两步 GPU 执行检查已通过。
- 扩充初始讲解 SFT 与实际分支搜索蒸馏。
- 相同数据预算的纯语言模型基线、专家桥接与蒸馏比较。
- 独立走法质量、解释事实和完整对弈评测。
- 开源发布资产、可复现入口、GitHub CI 与里程碑同步。

## 结果解释

所有新训练结果属于新生成的中国象棋基线，不是论文的西洋棋历史检查点。模型输出必须先核验，再报告非法率、走法损失和解释错误。对弈到达步数上限时记为截尾，不能自动当作和棋或据此换算 Elo。

原始 `runs/pilot-v1`、下载权重、数据与失败日志均保留。运行清单和大文件在本地，轻量可分享证据放入 `evidence/`。尚未完成的训练与评测会保留为未验证状态。

## 当前运行与续跑

- 完整教师验证已完成，日志与输出位于 `runs/consolidator-weight-verification.log`、`runs/consolidator-verification-v1/`。GPU 3 字典纯语言课程已从保存的第 1000 步恢复。
- 字典对照：`xqgeneral-curriculum-v3-bridge`（GPU 2）、`xqgeneral-v3-text-resume`（GPU 3），日志 `runs/research-v3/{bridge,text_lora}/curriculum/*.log`。
- 两组首轮讲解训练及原始输出评测均已完成。扩充讲解训练：`xqgeneral-explanation-v3-warm-bridge`（GPU 1），日志 `runs/explanation-v3-warm-bridge.log`；初始化与验证门槛见 `runs/explanation-v3-warm/bridge/plan.json`。
- 走法课程排队：`xqgeneral-move-quality-v2-bridge`（GPU 2）、`xqgeneral-move-quality-v2-text_lora`（GPU 3），日志 `runs/move-quality-v2-{bridge,text_lora}.log`，等待对应四门课程完成。
- 完成的课程纯语言基线问答与专家最佳讲解检查点验证日志：`runs/research-v2-text-qa-validation.log` 与 `runs/explanation-v2-bridge-validation.log`。完成的走法裁判检查日志：`runs/move-quality-v1-pre-curriculum-validation.log`。
- 走法早期实验与消融已完成：`runs/move-quality-warm-v1.log` 与 `runs/move-quality-warm-v1/final-validation/`；初始化范围见 `runs/move-quality-warm-v1/plan.json`。
- 新损失桥接续训检查已完成：`runs/example-normalization-verification-v2/log.txt`。完整解码器检查与 FP32 主参数检查分别已完成于 `runs/example-normalization-full-sft-verification/`、`runs/frozen-float-master-verification-v2/`。
- 搜索诊断已完成：`runs/search-v2-diagnostic.log` 与 `runs/search-v2-diagnostic/mining/`。新增 Astra 收集与数据检查已完成：`runs/astra-full-v2-collection/verification.json` 与 `runs/astra-full-v2-preflight/verification.json`。
- 完成的神经盲评与重试输出位于 `runs/interim-explanation-v2-001/prose-review/`；新增交叉复核输入、修改记录及原始拒收意见分别位于 `data/astra-engine-seed-v1/review-inputs/`、`data/astra-engine-seed-v1/corrections-2.jsonl` 与 `runs/astra-engine-seed-v1/pre-review/`。
- 训练通过 `scripts/freeze_run.py` 保存执行源码；开发中的后续修改不会改变已启动的训练。课程入口自动从 `latest.pt` 续跑，并核对输入哈希；改变数据或配置必须新建实验。
