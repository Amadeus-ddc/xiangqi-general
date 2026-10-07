# Xiangqi General 当前状态

2026-10-07。最终目标是能用于个人象棋学习的强走法与可靠中文讲解。训练课程与蒸馏轮数依据独立评测调整；当前不能宣称已达到该目标。

## 已验证

- 新增规则基础能力选模，要求同份权重完成原始走法、完整规划及四门课程全部 22 类平衡问答；拒收测试、代答、强制合法及变动证据。补训前已完成实际基线：264 道问答正确率 68.9%，当前吃子为 2/12、未来吃子为 0/12；同份模型的原始走法无明显失误率 77.1%，完整规划合同为零，加权基础能力分数为 0.5841。三份清单与全部输出哈希已读回，见 `evidence/foundation-functional-baseline-warm.json`。这不是完成新补训或可靠教学证明。

- 原型基线保留：冻结 Px0 与 Qwen，四个 384 宽桥接块，80 步静态课程；12 条生成样例中 8 条正确。这仅证明链路可执行。
- 原生输入编码：原始 102 个局面逐平面一致。
- 完整网络数值对照：34 个局面，2062 个策略输出、胜和负、剩余步数与四层特征均一致。胜和负最大误差 `6.85e-7`，策略最大误差 `5.60e-6`。参考为官方 C++ 编码器与未修改的 C++ ONNX 导出器，通过 ONNX Runtime CPU 执行。证据见 `evidence/native-network.json`。
- 83 项 CPU 测试通过，覆盖规则与完整历史终局、分割、梯度、重复、长将与长捉，以及新增词元冻结、对称数据、教师身份、引擎 MultiPV 中断、全量 SFT 初始化、搜索递归、追加标注合并、盲评协议和引擎走法课程的保留集隔离。新增检查验证样本损失权重、梯度累积、续训精度合同、合法词元树、自对弈截尾与完整历史、补充局面归属及未来局面隔离。规划数据和评测检查完整历史终局、条件首着、变化长度及保留集第二步以后局面的隔离。搜索汇总标签保存全部分支未来局面，排除更深的保留集变化及完整历史终局后的走法。选模快照固定原子检查点，拒绝同一步数的不同权重、篡改证据、独立测试或强制合法结果。新增检查验证并行规划标签逐字节一致、终局拒收计数一致，以及非主线未来局面的隔离和颜色镜像。新增隔离多进程检查验证完整未来及镜像分支的拒收、标签顺序/计数/字节一致，以及非法未来向主进程传播；真实追加合并清单与正文 rank 编号错误均有先失败后通过的回归检查。正文修订拒收保留集、异源教师、变动哈希和不完整修订链，保留结构标签并检查完整历史终局。实战规则问答检查未来棋盘答案、完整历史终局、深层未来与镜像隔离、缓存根键及单进程/多进程标签和拒收顺序一致。对弈选招不消耗评分，严格评分查询仍拒绝缺少完整评分的结果，两者均保持请求节点预算。独立走法评测拒收非法或终局后走法，并按原始输出统计全样本与合法样本条件指标。裁定锁定 `pyffish 0.0.90 xiangqi AXF`，保留完整历史。
- 课程数据生成器加入空格、零数量、棋子位置、子力、吃子、将军与未来非法反例；训练/验证/测试按棋局划分并检查当前及未来局面重叠。
- Git 已建立，原始原型有独立基线提交。软件包改名 `xqgeneral`；本地目录已同步改名为 `xiangqi-general`，项目环境与可编辑安装路径已修复。
- [GitHub 仓库](https://github.com/Amadeus-ddc/xiangqi-general) 已同步并合并首个源码里程碑；Python 3.11/3.12 的 CPU CI 均通过。
- 源码包实际内容已检查：第三方说明、贡献指南、模型说明、训练配置、冻结运行脚本及轻量证据均被打包，wheel 包含源码许可证与第三方说明；权重、数据、日志和 vendor 未入包。见 `evidence/release-package-contents.json`。完整 Qwen 教师的固定版本官方 Apache-2.0 许可证已列入第三方说明；这些检查不等同于模型权重发布。
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
- 两门课程后走法模型的可选规则约束解码已实测：192 题全部输出合法，167 个无明显失误（87.0%）。模型在合法词元树内按自身概率选招，没有引擎选招或替换。见 `evidence/move-quality-warm-legal-validation.json`。合法率由规则保证，该模式与原始生成分别报告，不评测讲解。
- 同一走法模型完成 12 局规则约束对弈：两种固定开局交换红黑，对皮卡鱼 100/1000/10000 节点各四局，总计 1 胜 11 负，零非法弃权、零截尾；唯一胜局来自 100 节点组。完整历史、逐步原始答案及终局均保存，输出哈希已读回；见 `evidence/move-quality-warm-legal-matches.json`。这证明合法输出不足以带来强棋力，未换算 Elo。
- 棋盘字典对照两组均完成四门课程，专家组执行 1500/1375/1500/1375 步，纯语言组均为 1500 步。264 道平衡验证题的原始正确率分别为 73.9% / 72.7%；专家打乱后为 68.9%。吃子题仍弱，专家组当前吃子题仅 1/12 正确、未来吃子题为 0/12，纯语言组两者均为零。原始回答显示了实际走法错误，不能据总体问答正确率宣称强棋力。见 `evidence/curriculum-v3-{bridge,text}.json` 与对应问答证据；字典输入属于论文条件以外的改动。
- 128 局引擎自对弈已完成，112 局裁定终局、16 局截尾；去除 851 个重复完整历史后保留 2937 个训练上下文，红黑分别 1466/1471。全部历史及未来变化已重新回放，与保留集根/未来局面重叠为零，输出哈希全部读回。实际触发 23 次缺评分搜索重试，失败输出均保留。见 `evidence/engine-selfplay-contexts.json`。这些是上下文，尚不证明补训收益或讲解质量。
- 扩充讲解 SFT 第 512 步中间检查点完成相同 96 个验证局面的原始评测：74 个可解析、60 个推荐走法合法、50 个无明显失误；只有 2 个完整主变化合法，完整结构合同为零。见 `evidence/explanation-v3-interim-512.json`。走法较前一版改善，但正文仍出现原地走法及虚构交换；训练预算、数据和初始化同时改变，不能归因于单项改动。
- 同一扩充讲解实验第 1024 步中间检查点：96 条中 94 条可解析、72 个推荐走法合法、57 个无明显失误，但完整主变化合法只有 4 条，完整结构合同仍为零。见 `evidence/explanation-v3-interim-1024.json`。尚不能作为可靠教学模型交付。
- 同实验第 2048 步：96 条全部可解析，78 个推荐走法合法、63 个无明显失误，5 条完整主变化合法、1 条通过完整结构合同。见 `evidence/explanation-v3-after-2048.json`。原始输出及哈希已读回；增长仍不足以证明可靠教学，讲解语义未评测。
- 正式四门课程后走法实验的第 1500 步中间检查点：192 条原始回答中 153 个合法、134 个无明显失误，低于第二门课程后走法预训练的对应验证结果。见 `evidence/move-quality-v2-interim-bridge.json`。初始化和课程经历不同，不能把差异归因于主参数精度；后续自适应课程选用验证表现较好的预训练检查点，匹配对照继续保留。
- 新自对弈上下文的独立搜索及缓存已完成：2933 个新局面有完整评分，训练走法题增至 22248 条；缓存包含 23420 个完整历史根局面及 16 层特征。验证、测试与旧走法课程字节一致，输出哈希全部读回。见 `evidence/selfplay-move-curriculum.json`。新增数据尚不证明模型收益。
- 冻结 Px0 原生策略完成同一 12 局开局与节点预算的诊断对弈：7 胜、2 负、2 和、1 局截尾；1000 节点对手组为四胜。见 `evidence/frozen-expert-matches.json`。没有语言解码器或专家搜索，结果不能算成桥接模型棋力。首次评分接口在小预算缺评分时中断，原始失败保留；后续对弈仅获取同预算合法走法，评分查询不放宽。
- 完整 BF16 教师完成 8 对原始学生讲解与保留 Astra 标注的盲评，16 个评分全部完成且无截断。教师参考的事实/战略均分为 3.5/5，5/8 四项至少 4 分；学生对应两项均为 1/5，0/8 四项至少 4 分。参考的机械结构、推荐走法和 48 步变化全部通过独立规则及引擎检查。见 `evidence/explanation-v3-paired-prose-review.json`。这些神经评分不是人工评价。
- 两位 Astra 审查者已分别复演裁判质疑的三条参考：9 个分支、54 步变化全部合法且逐步事实吻合，11 项主要战术及评分指控均未获支持，原因包括坐标、炮架、后续棋盘和搜索预算混淆；其中一位另建议规范两个颜色镜像项的棋子称谓。见 `evidence/teacher-reference-review-audit.json`。原标注及评分未修改。审查者共享初始教师基础模型，这不是全语料质量证明；后续裁判输入须强化规则事实，不能直接采信上述神经评分。
- 加入坐标棋子表、逐步前后棋盘和声明评分后，完整 BF16 裁判重评同一 8 对原始答案：参考的事实/战略均分为 4.5/4.125，7/8 四项至少 4 分；学生均为 1.125，仍为 0/8。见 `evidence/explanation-v3-grounded-prose-review.json`。Astra 对剩余受指控参考复演了主线和三分支共 24 步，吃子事实与评分方向支持原文，仅有兵/卒称谓建议；见 `evidence/grounded-review-reference-audit.json`。评分及标注保持原样，提示改进不是总体裁判准确率证明。
- 搜索挖掘、真实 BF16 汇总收集及完整对弈入口已实现并通过受控测试，真实搜索蒸馏与讲解模式对弈结果尚待产生。
- 搜索隔离回归先复现了两个错误：更深的子变化进入保留集仍被接收，以及仅按 FEN 合法的变化越过完整历史重复终局。修复后 12 项集中检查及全套 60 项测试通过，见 `evidence/search-history-isolation.json`。这是合同验证，实际搜索收益仍须单独测量。
- 2048 步学生的 32 个训练根局面搜索试批已完成：零个目标通过，31 条最终止于非法根候选、1 条达到递归上限；原始学生生成与递归轨迹保留。见 `evidence/search-v3-after-2048-pilot.json`。没有调用汇总教师或进行蒸馏训练，空结果不计为一轮蒸馏。
- 规划数据已完成：88,676 条真实引擎主变化及条件变化题，其中 86,892 条训练项；共 185,144 条规划、走法及四门问答回放。完整历史终局与棋局、根及未来局面分割检查通过，原非规划标签全分割逐字节保留，全部特征键存在。见 `evidence/move-planning-data.json`。颜色派生项不是独立棋局，数据就绪不代表规划能力。
- 同一 256 个真实查询的原实现、新单进程与八进程共生成 1994 条标签，标签及隔离后记录逐字节一致；生成阶段耗时约 83.15/83.54/20.83 秒，八进程父进程的隔离阶段约 32.40 秒。见 `evidence/planning-generation-performance.json`。这不是全数据吞吐或共享缓存收益的测量。后续生成支持保序多进程、复用镜像历史与未来复演，并报告各阶段进度；已完成旧运行不改变。
- 实战规划隔离的原生规则复演耗时已测量，新增可选 `--isolation-workers`：真实 256 个查询的 1968 条规划标签、4096 条分散训练记录及全部 26216 条旧保留记录，原冻结代码、新单进程与八进程保留的 32280 条记录逐字节一致，拒收计数一致；隔离耗时约 77.28/77.57/15.04 秒。实际输出哈希读回，见 `evidence/planning-isolation-parallel-equivalence.json`。这是有限 CPU 样本上的阶段耗时，未测全数据加速；当前旧全量运行的冻结代码和输出不修改。
- 未学习规划课程的走法预训练基线已评测同一 96 条规划验证题：95 条只输出一步，完整规划合同通过率为零，条件首着匹配率 45.8%。见 `evidence/planning-baseline-warm.json`。合法短前缀不算合格规划，后续补训须比较完整合同与逐步质量。
- 四门课程后的正式专家走法训练已完成 4250 步，按 NLL 选择 2750 步。192 道同题原始验证中 169 个走法合法、156 个无明显失误（88.0%/81.3%）；置零专家均为零，打乱专家为 60.4%/33.9%。规则约束解码得到 178 个无明显失误（92.7%），合法性由约束保证。见 `evidence/move-quality-v2-final-bridge.json`。原始输出、全部裁判证据和输出哈希已读回。随后同协议 12 局规则约束对弈仅 1 胜、10 负、1 和，1000 和 10000 节点组均四负；见 `evidence/move-quality-v2-final-bridge-matches.json`。局面指标尚未转化为强棋力，未换算 Elo 或评测正文。
- 正式纯语言走法组完成 5500 步，选择第 4000 步。同一 192 条原始验证中 153 个合法、110 个无明显失误（79.7%/57.3%）；规则约束得到 140 个无明显失误（72.9%），相同 12 局对弈全负。见 `evidence/move-quality-v2-final-text.json`。与桥接组同走法数据和预算上限，但实际早停步数、参数量及结构不同；这是开发验证比较，未证明独立测试棋力或单一因果收益。
- 扩充讲解第 3072 步完成同一 96 条原始验证：79 个推荐走法合法、67 个无明显失误，11 条完整主变化合法、3 条通过完整结构合同。见 `evidence/explanation-v3-after-3072.json`。输出哈希已读回，正文语义未评分，仍不能作为可靠教练交付。
- 同实验第 4096 步原始验证：80 个推荐走法合法、62 个无明显失误，22 条完整主变化合法、7 条通过完整结构合同。见 `evidence/explanation-v3-after-4096.json`。全部输出哈希已读回；首着质量较第 3072 步回退，不能仅凭 NLL 下降选取教学模型；正文的后续抽样审查如下。
- 新增 384 条 Astra Low 实战原始讲解已全部交叉复核：345 条原文通过，39 条修正后独立复核通过；原始标注、拒收和修正各自保留。选择红黑各 192 条原始讲解，保留旧训练项后为 6662/192/256，全部特征键可用、长度不超过 943 个词元；完整历史隔离继承已复演的父集合，验证/测试字节不变。见 `evidence/initial-teacher-selfplay.json`。新增镜像候选存在两处横线编号及一处未限定颜色的仕/士称谓问题，保留在单独候选中，未用于新训练；见 `evidence/initial-teacher-selfplay-candidate.json`。后续镜像已修正明确 `rank` 编号，既有实验不改写。全量神经复核不等同于人工评分或学生质量证明。
- 实际走法检查点的选模快照已验证：258 个张量、139920416 个可训练参数逐位一致，快照与原检查点 SHA256 相同；见 `evidence/functional-checkpoint-snapshot.json`。这验证参数保存，不证明选模收益。
- 第 4096 步真实检查点已完成三项能力分数基线：192 条原始走法的无明显失误率为 70.8%，96 条规划完整合同通过率为零，96 条讲解完整合同通过率为 7.3%；同份权重的加权分数为 0.3125。三份评测清单、权重身份及输出哈希已读回，见 `evidence/explanation-functional-baseline-v3.json`。这是选择器的真实评测基线，不是已完成全量训练选模；独立测试未使用，强棋力与正文语义仍未证明。
- 真实已合并教师批次已再次追加成功：3747 条原始标注，3523 训练、96 验证、128 测试；全部输出哈希读回，验证/测试字节不变。见 `evidence/teacher-batch-extension.json`。这是原始标注和审查记录的合并，不产生新讲解或镜像标签。
- 终端学习入口已用真实第 3072 步检查点完成保存、续读、用户走棋、提示、悔棋和退出：三条原始回答及完整历史均读回，旧分析随续读保留，悔棋恢复原历史。见 `evidence/coach-real-checkpoint.json`。三条回答均未通过完整讲解合同，界面保留原输出并显示核验失败；这证明操作链路，不证明教学质量。
- 学习入口的完整历史检查先复现了两种终局遗漏：主线和候选分支都可能在重复终局后继续，即使每步按 FEN 合法。修复后两种情况均拒收，恰好到达终局的合法变化仍通过；原始答案不修复。补充走法生成也支持并集保留多个数据集的棋局及全部未来分支，相关输入哈希纳入续跑合同。
- 旧训练讲解的 384 个原始／镜像配对样本共 768 条已神经复核：原始 27 条、镜像 34 条要求修订，全部修订经另一位 Astra Low 独立接受；镜像一条首次修订仍写错进退，第二次才通过。两组各 1123 分支、6658 步由根代理再次机械复演，原文与各轮意见保留。见 `evidence/old-{original,mirror}-prose-sample-review.json`。这不是全体语料或人工质量证明。
- 新数据 `data/astra-explanations-full-v5` 已实际应用上述 61 条正文修订；保持 6662/192/256 条，红黑各 3331 条训练项。所有结构标签、提示、历史、特征键及验证／测试字节不变，最长训练项仍为 943 个词元。见 `evidence/reviewed-prose-dataset-v5.json`。完整哈希、修订祖先和拒收／接受链已核验，未测学生收益。首次长度检查调用错误及字段措辞修正分别归档，完成核验位于 `runs/teacher-prose-data-v5-readback-v2/`。
- 第 4096 步旧学生的 32 条均匀原始讲解抽样经 Astra Low 逐条评分：事实均分 1.59/5、战略 1.44/5，零条四项均至少 4 分。见 `evidence/explanation-v3-4096-astra-prose-review.json`。输入包含独立引擎和逐步事实、没有检查点身份；审查者仍有项目上下文并共享初始教师基础模型，这不是完全盲测或人工评测。
- 正式四门初始化的讲解 v4 在第 512 步完成 96 条原始验证：39 条可解析、37 个推荐合法、28 个无明显失误，只有 1 条完整主变化合法，完整讲解合同为零。见 `evidence/explanation-v4-interim-512.json`。实际原输出大量重复或正文未闭合，保留失败；它尚未达到教学要求，不能因 NLL 为 0.8769 就判定优于旧实验。
- 新 512 局引擎自对弈完成：286 局终局、226 局截尾，保留 27432 个完整历史训练上下文、排除 2375 个重复历史；全部上下文和输出哈希已读回，保留局面重叠为零。见 `evidence/engine-selfplay-contexts-v3.json`。后续独立搜索生成 27408 条新查询；11348 条旧查询逐条完整保留，验证/测试字节不变。原检查把按分割重组误判为记录改变，失败运行保留；新检查验证分组后的完整记录。16 层缓存现已完成，包含 78236 个完整历史根；115370410864 字节的实际缓存、全部输入与输出哈希、形状/类型、所有请求根及旧 23420 个键均经核验。见 `evidence/selfplay-move-curriculum-v3.json`、`evidence/selfplay-cache-contract-v3.json`；全量基础补训词元预检尚须完成，没有据此宣称训练收益或神经讲解。
- 规划 v1 的首个真实第 2000 步候选已评测：192 条原始走法中 158 条合法、141 条无明显失误；96 条规划均输出六步、条件首着全部匹配，但只有 10 条完整合法，逐步质量全尝试无明显失误率为 45.5%。同份检查点的两项分数为 0.4823，见 `evidence/move-planning-v1-interim-2000.json`。全部输出和权重哈希已读回；课程与选模仍在运行，这不是强棋力或独立测试结论。
- 同一早期起点规划训练的第 4000 步候选完成原始验证：192 条走法中 177 条合法、152 条无明显失误，96 条规划中 21 条完整合同通过；第 2000 步对应为 158/141/10，三项计数均增加。两项加权分数从 0.4823 升至 0.5625；逐步质量全尝试无明显失误率为 53.6%，仍有 74 条非法规划及 1 条终局后续着。实际权重步数与全部原始输出哈希读回，见 `evidence/move-planning-v1-interim-4000.json`。课程和选模未完成，正文与完整对弈未在本项测量，没有强棋力结论。
- 正式起点规划训练的第 4000 步候选也完成同题原始验证：192 条走法中 170 条合法、160 条无明显失误，96 条规划有 30 条完整合同通过；同组第 2000 步为 165/152/17，两项分数从 0.5458 升至 0.6250。仍有 65 条非法规划及 1 条终局后续着；权重实际步数、训练合同及全部原始输出哈希已读回，见 `evidence/move-planning-v2-interim-4000.json`。这是未完成训练的中间验证，没有正文、完整对弈或独立测试棋力结论。
- 第 4096 步讲解模型还在训练标签池中抽查了红黑各 16 条：32 条全部可解析、30 个推荐合法、28 个无明显失误，但只有 5 条主变化合法、1 条完整合同通过。见 `evidence/explanation-v3-training-pool-diagnostic.json`。这是训练标签池诊断，不是泛化评测；未证明各抽中样本实际被训练采样，也未用于选模或测试。错误不能只归因于陌生局面，正文语义在这项检查中未评分。
- 第 3072 步模型的实战训练根局面搜索试批完成：128 根、168 次根尝试、40 次递归下降、10 次按既定条件注入候选，零条严格改进通过；最终均止于非法根候选。见 `evidence/search-v3-after-3072-selfplay-pilot.json`。原始回答、递归轨迹与全部输出哈希保留；没有汇总或蒸馏训练，仍不计为一轮蒸馏。
- 新实战规则问答生成器已完成真实 512 根验证：红黑各 256 个原始根、108 个训练棋局，生成 22528 道当前/未来规则题；每条新答案独立按规则复核，70532 条训练记录词元化后最长为 521，全部根特征键存在。原验证/测试字节不变、棋局及当前/未来跨分割重叠为零，见 `evidence/selfplay-grounding-data-ready.json`。同参数单进程和八进程的三个分割文件及拒收计数完全一致，见 `evidence/selfplay-grounding-parallel-equivalence.json`。全量 8192 新根也已生成 360448 道规则题，双方各 4096 根、499 个新训练棋局；所有产物哈希和旧保留文件字节已读回，见 `evidence/selfplay-grounding-data-v1.json`。全量词元/特征预检仍须完成，没有训练收益或神经讲解结论。

## 正在完成

- 扩充讲解 SFT 已实际完成 6656 步，按验证 NLL 选择第 4608 步，最终 NLL 为 0.5271；41.62 亿 FP32 可训练参数、BF16 计算。全部产物哈希及实际权重中的步数、参数量和类型已读回，见 `evidence/explanation-sft-v3-warm-completed.json`。该选择依据损失，尚非最终能力选定模型；中间输出的走法与讲解缺陷继续保留，不能据此宣称可靠教练。
- 上述实际第 4608 步完成同份权重三项原始能力补测：192 条走法中 167 条合法、150 条无明显失误；96 条规划完整合同为零；96 条讲解中 78 个推荐合法、65 个无明显失误、17 条主变化合法、6 条完整合同通过。三项分数为 0.3375；41.62 亿 FP32 参数、实际步数、权重哈希及所有原始产物再次读回，见 `evidence/explanation-v3-selected-4608-validation.json`。该模型已完成训练，但按损失选取，尚非能力选定教练；正文语义、完整对弈和独立测试未在此项测量。
- 正式起点的全量讲解训练第 2048 步完成同份权重三项原始验证：192 条走法中 167 条合法、154 条无明显失误；96 条规划有 9 条完整合同；96 条讲解有 84 个推荐合法、72 个无明显失误、9 条主变化合法，但仅 2 条完整合同通过。三项分数为 0.3479，权重及全部输出哈希已读回，见 `evidence/explanation-v4-after-2048.json`。训练和后续选模仍在进行，正文语义未评分，不能交付为可靠教练。
- 同一正式起点讲解训练第 4096 步完成三项原始验证：192 条走法中 170 条合法、157 条无明显失误；96 条规划有 15 条完整合同；96 条讲解有 87 个推荐合法、77 个无明显失误、21 条主变化合法，完整讲解仍只有 2 条通过。三项分数从第 2048 步的 0.3479 升至 0.3667；实际 41.62 亿 FP32 可训练参数、权重、训练合同及全部原始产物哈希核验，见 `evidence/explanation-v4-after-4096.json`。连续推演与完整讲解仍弱，训练和选模未完成，正文语义与完整对弈未在本项测量。
- 两组第 2000 步规划候选已完成同题比较：早期/正式初始化的原始走法无明显失误数为 141/152（192 题），完整规划合同为 10/17（96 题），平衡规则问答为 69.7%/74.2%（264 题）。正式组仍有 79 条非法规划，当前/未来吃子仅 1/12、2/12；没有强棋力结论。所有权重和输出哈希已读回，见 `evidence/move-planning-v2-interim-2000.json`、`evidence/planning-2000-rule-retention.json`。后续初始化改为比较两组训练实际完成后的全部候选，不能拿中间模型冒充完成父模型。
- 正式第 2000 步的 24 条吃子验证答案已按原生规则逐条重算并复演完整历史，全部参考正确、无历史终局。忽略排列顺序后仍仅当前 1/12、未来 2/12 正确；两组各 9 条漏掉吃子，分别 7/4 条混入非法走法、6/1 条混入合法非吃子走法。生成文本最长 20 个词元，参考最长 28，预算 384；这是实际动作错误，见 `evidence/planning-capture-error-audit.json`。原始回答和指标保持原样；24 条诊断不能估计总体错误率，文本长度也不独立证明停止生成的原因。
- 其余 5510 条旧训练讲解（2755 原始、2755 镜像）已完成真实逐条神经审查：5143 条原文接受、367 条要求修订；审查身份、原标注及决定哈希全部核验，见 `evidence/old-prose-full-review-completed.json`。此前已解决其中 98 条修订，余下 269 条真实修订已生成并全部由不同审查者独立接受，精确拒收祖先和最终接受哈希核验通过；现有训练任务保持固定输入。神经审查不是人工评价或学生收益证明。
- 首批实际复核的 532 条旧正文已整理为可追溯审查包：502 条原文接受，30 条修订后由不同教师独立接受，其中 1 条经过第二轮修正。精确原标注、拒收、修订祖先和最终接受哈希全部核验，见 `evidence/old-prose-first-prefix-resolved.json`。另 1280 条实际复核范围也已整理完成：1212 条原文接受，68 条真实修订经不同审查者全部独立接受；与前 532 条没有重复，见 `evidence/old-prose-second-prefix-resolved.json`。两包累计 1812 条已解决，完整修订祖先和接受哈希均通过核验；未估计全体错误率，也未修改现有训练数据。其余 3698 条也已解决：3429 条原文接受、269 条真实修订全部独立接受；三个互不重复包共覆盖 5510 条，见 `evidence/old-prose-remaining-resolved.json`。新数据版本的组装和词元预检仍在进行，尚未测学生收益。
- 新实战初始教师输入已完成：2048 个原始训练根局面，来自 485 个训练棋局，红黑各 1024；三个分片为 683/683/682 条。完整历史和未来候选分支复演、保留集隔离及产物哈希均核验。Astra Low 已真实完成前两片 1366 条正文，身份、ID、结构和准备产物哈希核验通过，见 `evidence/astra-selfplay-shards01-authored-v3.json`。首片 683 条独立真审与修订已完成：670 条原文接受、13 条真实修订全部由不同审查者接受，精确标注、继承前缀及全部修订链核验，见 `evidence/astra-selfplay-shard0-resolved-v3.json`；第二片正独立审查，第三片继续生成。全 2048 条的生成、独立接受和训练均未完成，这批仍是初始讲解补充，不计作蒸馏。

- 红黑平衡数据上的扩大纯语言基线已完成四门课程，执行 4000/2750/4000/3250 步，264 道平衡验证问答正确率为 51.9%；见 `evidence/curriculum-v2-text.json`。完整讲解 SFT 与 96 根局面评测均已完成。
- Astra 已生成新增 579 个训练根局面的初始讲解，并对新增训练项抽检 72 条；一处“回吃”表述已修正并复核。合并后的 1,315 条原始标注与颜色派生项共 2,630 条（2182 训练、192 验证、256 测试），全部结构校验与输出哈希读回通过；验证/测试文件与原版本字节一致。共 144 条神经抽检，见 `evidence/initial-teacher-full.json`。
- 四门字典课程后的专家与纯语言正式走法训练、同题验证及完整诊断对弈均已完成，后续比较加入规划与讲解后的实际收益。
- 第二门字典课程后的走法预训练及全量消融均已完成。其选定检查点已开始 `configs/explanation-sft-v3.json` 的扩充讲解 SFT；这是自适应的额外实验，四门课程的匹配对照仍单独进行。
- `configs/move-planning-v1.json` 加入真实自对弈走法与最多六步主变化、条件变化课程，保留四门问答回放；预算上限 12000 步、最少 4000 步，再按验证集早停。完整历史规划数据与裁判校准已完成，GPU 0 正在实际训练；独立选模队列随后测量走法与完整规划合同。
- 扩充讲解收集与数据检查已完成。后续 `configs/explanation-sft-v3.json` 按样本平均损失混合讲解、走法题及四门基础课程回放，完整解码器两步 GPU 执行检查已通过。
- `configs/move-planning-v2.json` 已在 GPU 2 启动，从正式四门课程走法模型初始化，其余数据、混合比例、优化器和随机种子与 v1 相同；两者有不同课程经历，不能据此隔离单个课程的因果收益。每组另按真实走法和完整规划合同选模。
- `configs/explanation-sft-v4.json` 已在 GPU 3 实际训练；从正式走法模型开始全量解码器训练，单独以 15% 比例采样全量复核的 384 条实战原始讲解，保留走法与多步规划回放。验证讲解仍用原保留集。该批是初始标注补充，未计作搜索蒸馏；另以同一检查点的原始走法、规划和完整讲解合同选模，正文语义仍单独评审。
- 扩充初始讲解 SFT 与实际分支搜索蒸馏。
- 相同数据预算的纯语言模型基线、专家桥接与蒸馏比较。
- 独立走法质量、解释事实和完整对弈评测。
- 根据弱对手对弈失败补充引擎自对弈的开局、中局训练上下文；此前随机合法棋局及其搜索子局面不足以代表实战分布。生成器仅产生训练上下文，不生成神经讲解；补充局面再经独立搜索才能形成走法标签。
- 开源发布资产、可复现入口、GitHub CI 与里程碑同步。

## 结果解释

所有新训练结果属于新生成的中国象棋基线，不是论文的西洋棋历史检查点。模型输出必须先核验，再报告非法率、走法损失和解释错误。对弈到达步数上限时记为截尾，不能自动当作和棋或据此换算 Elo。

原始 `runs/pilot-v1`、下载权重、数据与失败日志均保留。运行清单和大文件在本地，轻量可分享证据放入 `evidence/`。尚未完成的训练与评测会保留为未验证状态。

## 当前运行与续跑

- 完整教师验证已完成，日志与输出位于 `runs/consolidator-weight-verification.log`、`runs/consolidator-verification-v1/`。
- 字典专家组和纯语言组四门课程及问答验证均已完成，日志 `runs/research-v3/{bridge,text_lora}/curriculum/*.log` 与 `runs/research-v3-{bridge,text}-qa-validation.log`。
- 两组首轮讲解训练及原始输出评测均已完成。扩充讲解训练也已完成 6656 步，选择第 4608 步；日志 `runs/explanation-v3-warm-bridge.log`，初始化与验证门槛见 `runs/explanation-v3-warm/bridge/plan.json`，完整产物与实际权重读回位于 `runs/explanation-v3-completion-readback-v1/`。
- 正式走法课程与匹配评测均已完成。训练日志 `runs/move-quality-v2-{bridge,text_lora}.log`；专家最终消融日志 `runs/move-quality-v2-final-bridge-v2/log.txt`，纯语言同题与对弈日志 `runs/move-quality-v2-final-text-v1/log.txt`。首次评测启动错误和虚拟环境路径修正另存。
- 讲解中间检查：`xqgeneral-explanation-v3-interim-512`（GPU 0），日志 `runs/explanation-v3-interim-512.log`；原子保存的最佳检查点通过硬链接固定，后续训练不会替换该评测输入。
- 自对弈上下文、独立搜索及特征缓存均已完成。日志 `runs/engine-selfplay-train-v2.log`、`runs/selfplay-policy-v2.log`，精确命令及续跑说明在对应 `runs/*/plan.json`。旧准备遍历额外的历史 PV，在真正引擎搜索开始前停止，合同和源码均保留。新批次用 `--no-descendants` 保留全部旧搜索结果，仅搜索新自对弈上下文。首次自对弈缺评分错误及真实重试证据保留于 `runs/engine-selfplay-debug-v1/`。
- 第 512、1024 步讲解检查及两轮成对盲评均已完成，输出位于 `runs/explanation-v3-interim-{512,1024}/`、`runs/explanation-v3-paired-prose-review/`、`runs/explanation-v3-grounded-prose-review/`；Astra 复核不改已有标注和评分。
- 规划数据及输出哈希检查已完成，日志 `runs/move-planning-data-v1.log`，验证 `runs/move-planning-data-v1/verification.json`；新课程正在训练：`xqgeneral-move-planning-v1-bridge`（GPU 0），日志 `runs/move-planning-v1-bridge.log`，配置、等待条件及精确续跑命令在 `runs/move-planning-v1/bridge/plan.json`。使用精确 tmux 会话匹配，避免相似名称互相等待。
- 补训前原始规划基线已完成，日志 `runs/planning-baseline-warm-v1/log.txt`；新旧生成器保序验证已完成，日志 `runs/planning-generation-benchmark-v1/log.txt`。
- 实际能力选模：`xqgeneral-move-planning-functional-selection`（GPU 0），日志 `runs/move-planning-v1/functional-selection/log.txt`；每 2000 步固定真实参数，评测同一 192 道原始走法及 96 道规划验证题，另存功能分数最佳模型。NLL 训练选择保留；独立测试集不参与。配置、分数定义及续跑入口在同目录 `plan.json`，首个候选及其分数已完成读回，最终选择仍未完成。
- 可选约束解码的验证与 12 局对弈已完成：`runs/move-quality-warm-v1/legal-validation/` 与 `runs/move-quality-warm-v1/legal-matches/`。
- 完成的课程纯语言基线问答与专家最佳讲解检查点验证日志：`runs/research-v2-text-qa-validation.log` 与 `runs/explanation-v2-bridge-validation.log`。完成的走法裁判检查日志：`runs/move-quality-v1-pre-curriculum-validation.log`。
- 走法早期实验与消融已完成：`runs/move-quality-warm-v1.log` 与 `runs/move-quality-warm-v1/final-validation/`；初始化范围见 `runs/move-quality-warm-v1/plan.json`。
- 新损失桥接续训检查已完成：`runs/example-normalization-verification-v2/log.txt`。完整解码器检查与 FP32 主参数检查分别已完成于 `runs/example-normalization-full-sft-verification/`、`runs/frozen-float-master-verification-v2/`。
- 搜索诊断已完成：`runs/search-v2-diagnostic.log` 与 `runs/search-v2-diagnostic/mining/`。新增 Astra 收集与数据检查已完成：`runs/astra-full-v2-collection/verification.json` 与 `runs/astra-full-v2-preflight/verification.json`。
- 2048 步后讲解检查：`xqgeneral-explanation-v3-after-2048`（GPU 2），日志 `runs/explanation-v3-after-2048/log.txt`；选定检查点通过硬链接固定，真实选定步数写入同目录 `pinned-checkpoint.json`。实际搜索试批：`xqgeneral-search-v3-after-2048`（GPU 3），日志 `runs/search-v3-after-2048/log.txt`；仅挖掘 32 个训练根局面，符合严格改进条件才调用完整 BF16 教师汇总，不等同于完成蒸馏训练。
- 第 3072 步讲解检查已完成，日志 `runs/explanation-v3-after-3072/log.txt`；真实选定步数、哈希与原子硬链接在 `pinned-checkpoint.json`，使用相同 96 道原始验证题。原计划 GPU 2，但 tmux 服务未继承启动进程的环境，实际运行 GPU 0，记录于 `execution-device.json`；正式走法同题消融也实际在 GPU 0 完成。后续启动命令显式包含 `env CUDA_VISIBLE_DEVICES=...`。
- Astra 实战标注与原拒收审查在 `data/astra-selfplay-seed-v1/`；修正后输入在 `data/astra-selfplay-seed-v2/`，最终数据检查在 `runs/astra-selfplay-final-data-v1/verification.json`。全量复核输入、实际逐步检查与修正哈希保留，训练只采纳已通过的原始讲解。
- 正式初始化规划组：`xqgeneral-move-planning-v2-bridge`（GPU 2），日志 `runs/move-planning-v2-bridge.log`；其实际能力选模会话 `xqgeneral-move-planning-v2-functional-selection`，配置及续跑命令在对应 `runs/move-planning-v2/*/plan.json`。
- 新讲解训练：`xqgeneral-explanation-v4-formal-bridge`（GPU 3），日志 `runs/explanation-v4-formal/bridge/log.txt`。三项能力选模：`xqgeneral-explanation-v4-functional-selection-gpu0`（GPU 0），每 2048 步及完成时检查实际权重。原 GPU 1 等待任务尚未评测便迁移，原记录保留于 `superseded.json`；GPU 1 留给后续全量 SFT。NLL 选择另存，精确参数、守卫哈希与续跑命令见各目录 `plan.json`。
- 512 局实战上下文已完成，日志 `runs/engine-selfplay-train-v3/log.txt`；每局最多 64 个平衡上下文，排除完整保留变化。新独立走法标签和缓存也已完成。
- 新独立走法搜索及 16 层特征缓存均已完成；原 `runs/selfplay-policy-v3/log.txt` 保留错误全局顺序断言的失败。旧记录按训练/验证/测试重组后逐条不变，旧保留文件字节一致；缓存恢复在 `runs/selfplay-policy-v3-cache-recovery-v1/`，实际缓存合同的独立读回在 `runs/selfplay-cache-contract-v3/`。两者清单与输出均通过检查；`runs/selfplay-policy-v3/verification.json` 保留依赖入口，后续新训练仍须通过规划与词元预检。
- 实战根局面的新搜索试批：`xqgeneral-search-v3-after-3072-selfplay`（GPU 2），日志 `runs/search-v3-after-3072-selfplay/log.txt`；128 个原始训练根局面，保留原回答与所有分支轨迹，仅严格改进项调用完整 BF16 教师汇总。学习入口的真实模型检查：`xqgeneral-coach-real-checkpoint`（GPU 0），日志 `runs/coach-real-checkpoint-v1/log.txt`；检查保存、续读、用户走棋、提示和悔棋。
- 第 4096 步原始讲解及三项能力基线：`xqgeneral-explanation-v3-after-4096` 与 `xqgeneral-explanation-v3-functional-baseline`（GPU 0），日志在各自 `runs/*/log.txt`；同一真实原子硬链接检查点，保留原始结果并核验三项分数，不是完成新一轮训练选模。
- 旧讲解配对抽样及修订已完成，输入与意见在 `data/old-{original,mirror}-prose-review-v1/`；两个接受后的审查包在相应 `*-prose-reviewed-v1/`，其余旧标注的完整审查已完成并保留原文。 全量剩余正文的复核包在 `data/old-prose-full-review-v1/review-{0,1}.jsonl`，实际审查进度在相应 `review-*.progress.jsonl`；任务与模型调用配置见 `runs/old-prose-full-review-v1/plan.json`。
- 首批 532 条的完整审查与修订祖先在 `data/old-prose-repair-prefix-reviewed-v1/`；全部输入/输出与精确接受链在 `runs/old-prose-repair-prefix-resolved-v1/` 读回。第二批 1280 条及其中 68 条独立接受的修订保留于 `data/old-prose-repair-prefix-reviewed-v2/`，在 `runs/old-prose-repair-prefix-resolved-v2/` 核验精确祖先与无重复。原文全量审查已完成，修订包不直接覆盖训练数据。
- 原已完成全量讲解训练的实际第 4608 步权重三项原始能力补测已完成，日志 `runs/explanation-v3-selected-4608-validation-v1/log.txt`；实际权重、三份原始产物及完整训练身份的再次读回在同目录 `root-readback.json`，测试未参与。
- 审查后讲解续训 `xqgeneral-explanation-v5-reviewed-bridge`（GPU 1）已通过初始化守卫：实际父权重为完成的第 4608 步，重置优化器，采用已复核的旧正文修订；新训练已实际开始，执行日志 `runs/explanation-v5-reviewed/bridge/log.txt`；尚未完成或证明收益。其三项能力选模 `xqgeneral-explanation-v5-functional-selection` 使用 GPU 2，日志在同目录。配置为 `configs/explanation-sft-v5.json`，守卫与续跑命令见各自 `plan.json`。
- 全量旧正文 5510 条逐条审查的最终独立读回在 `runs/old-prose-full-review-readback-v1/`。其余 3698 条及其中 269 条拒收输入已冻结于 `data/old-prose-remaining-repair-v1/`；真实 269 条修订已完成。独立接受准备日志与冻结守卫在 `runs/old-prose-remaining-repair-v1/acceptance-preparation/`，已核验结构、完整历史及精确拒收祖先，269 条修订均已真实神经接受。完整剩余审查包在 `data/old-prose-remaining-reviewed-v1/`，独立读回在 `runs/old-prose-remaining-resolved-v1/`。新数据 `data/astra-explanations-full-v6` 的组装和全训练项词元预检由 `xqgeneral-teacher-prose-data-v6` 执行，精确参数及固定源码在 `runs/teacher-prose-data-v6/`；预检未完成，不改变活跃训练的输入。原文审查续接的身份、范围及前缀哈希在 `runs/old-prose-full-review-v1/reviewer-1*-continuation-plan.json`；修订作者范围在准备目录及 `author-progress.json`。
- 新实战初始教师输入准备已完成，日志 `runs/astra-selfplay-seed-v3/log.txt`；实际教师调用与全 2048 条范围见同目录 `author-plan.json`。前两片 1366 条正文已完成，第三片仍在生成。首片审查和独立读回在 `data/astra-selfplay-prose-review-v3/` 与 `runs/astra-selfplay-prose-review-v3-preparation/`，13 条真实修订及拒收祖先已冻结于 `data/astra-selfplay-prose-repair-v3-shard0/`，13 条均已由不同审查者逐条接受，完整可追溯包在 `data/astra-selfplay-prose-reviewed-v3-shard0/`，独立读回在 `runs/astra-selfplay-prose-shard0-resolved-v3/`。第二片输入与准备证明分别在 `data/astra-selfplay-prose-review-v3-shard1/` 和 `runs/astra-selfplay-prose-review-v3-shard1-preparation/`。只有全查询覆盖、独立接受、结构和哈希检查完成后才组装新训练数据。
- 四门实战规则问答已生成全 8192 个平衡原始根的 360448 道新题，日志 `runs/selfplay-grounding-data-v2/log.txt`，全部产物哈希与保留文件字节另经 `root-readback.json` 读回。原单进程等待任务尚未生成数据即迁移，源码与 `superseded.json` 保留；512 根真实执行与并行字节对照分别记录在 `runs/selfplay-grounding-verification-v1/` 和 `runs/selfplay-grounding-parallel-verification-v1/`。全量词元长度与缓存由基础补训预检继续核验。
- 实战基础补训当前队列：准备会话 `xqgeneral-selfplay-foundation-preparation` 保持不变；训练会话 `xqgeneral-selfplay-foundation-v2-bridge`、能力选模 `xqgeneral-selfplay-foundation-v2-functional-selection`（均 GPU 0）。等待实际数据/特征/词元预检及两组规划训练与能力选模完成，并等待相关 GPU 会话释放。随后在两组全部真实候选上补测 264 道平衡规则问答，以同份权重的问答、走法和规划分数选择父模型，再重置优化器补训。每 2000 步及完成时选模，独立测试不参与。配置 `configs/selfplay-foundation-v2.json`，日志与精确续跑参数在 `runs/selfplay-foundation-v2/{bridge,functional-selection}/`；共用预检在 `runs/selfplay-foundation-v1/preparation/`。原 v1 两个等待任务在未评测父模型、未开始训练时替换，源码与 `superseded.json` 保留；目前没有新训练收益结论。
- 扩充规划数据准备：`xqgeneral-move-planning-data-v2`，日志 `runs/move-planning-data-v2/log.txt`；正在生成 `data/move-planning-selfplay-v3`，核验全部产物哈希、旧规划验证/测试字节与非规划标签；特征与长度就绪前不启动新训练。
- 完成的神经盲评与重试输出位于 `runs/interim-explanation-v2-001/prose-review/`；新增交叉复核输入、修改记录及原始拒收意见分别位于 `data/astra-engine-seed-v1/review-inputs/`、`data/astra-engine-seed-v1/corrections-2.jsonl` 与 `runs/astra-engine-seed-v1/pre-review/`。
- 训练通过 `scripts/freeze_run.py` 保存执行源码；开发中的后续修改不会改变已启动的训练。课程入口自动从 `latest.pt` 续跑，并核对输入哈希；改变数据或配置必须新建实验。
