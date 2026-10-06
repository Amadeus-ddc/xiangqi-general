# Xiangqi General 当前状态

2026-10-07。最终目标是能用于个人象棋学习的强走法与可靠中文讲解。训练课程与蒸馏轮数依据独立评测调整；当前不能宣称已达到该目标。

## 已验证

- 原型基线保留：冻结 Px0 与 Qwen，四个 384 宽桥接块，80 步静态课程；12 条生成样例中 8 条正确。这仅证明链路可执行。
- 原生输入编码：原始 102 个局面逐平面一致。
- 完整网络数值对照：34 个局面，2062 个策略输出、胜和负、剩余步数与四层特征均一致。胜和负最大误差 `6.85e-7`，策略最大误差 `5.60e-6`。参考为官方 C++ 编码器与未修改的 C++ ONNX 导出器，通过 ONNX Runtime CPU 执行。证据见 `evidence/native-network.json`。
- 35 项 CPU 测试通过，含马腿、象眼、炮架、将帅照面、分割、历史、梯度、重复、长将与长捉，以及新增词元冻结、对称数据、教师身份、引擎 MultiPV 中断、全量 SFT 初始化、搜索递归、保留集不变的追加标注合并与评测协议。新增神经讲解裁判的盲化、评分与完整 BF16 身份协议测试通过；实际神经评分仍待执行。裁定锁定 `pyffish 0.0.90 xiangqi AXF`，保留完整历史。
- 课程数据生成器加入空格、零数量、棋子位置、子力、吃子、将军与未来非法反例；训练/验证/测试按棋局划分并检查当前及未来局面重叠。
- Git 已建立，原始原型有独立基线提交。软件包改名 `xqgeneral`；本地目录已同步改名为 `xiangqi-general`，项目环境与可编辑安装路径已修复。
- [GitHub 仓库](https://github.com/Amadeus-ddc/xiangqi-general) 已同步并合并首个源码里程碑；Python 3.11/3.12 的 CPU CI 均通过。
- 第一版专家桥接与纯语言 LoRA 均完成四门课程，每门 800 步。264 道平衡验证题的原始生成正确率分别为 **50.8% / 52.7%**，置零专家为 **43.6%**，打乱专家为 **52.3%**。没有证明专家带来稳定收益，也没有证明强棋力。证据见 `evidence/curriculum-validation-v1.json`。
- 发现并修正第一版根局面全为黑方行棋的采样缺陷。红黑对称增强后为 73,772 条课程题、3,354 个完整历史上下文，双方各 1,677 个；棋局及当前/未来局面跨分割重叠均为零。派生样本不算新独立棋局，见 `evidence/balanced-data.json`。
- 新方案的 16 个 768 宽桥接块与 105 个专用棋盘词元可在 GPU 上训练，合计 139,920,416 个可训练参数。连续 4 步与 2 步后续跑到 4 步的 258 个可训练张量逐位相同，初始损失、84 个监督词元和执行源码身份一致，见 `evidence/wide-training-resume.json`。四步验证不代表已完成新课程。
- 用户指定的 **GPT-6-Astra Low 子代理**生成了 736 条初始中文讲解（512 训练、96 验证、128 测试）。结构、走法、变化和声明事实全部通过机械校验；另有 72 条训练/验证讲解经交叉神经抽检通过。加上明确标记的颜色对称派生项，共 1,472 条；派生项没有再次调用教师或重算引擎分数。见 `evidence/initial-teacher.json`。抽检不是完整战略质量证明或人工评分。
- 扩大专家桥接模型完成四门课程：执行步数为 2500/1750/1250/1750，各门恢复验证集最佳检查点。最终课程验证 NLL 为 0.8505；24 条抽样生成仅 12 条完全正确，264 道平衡验证问答原始正确率为 46.6%，置零专家为 0%，打乱专家为 43.6%。见 `evidence/curriculum-v2-bridge.json`。课程损失下降尚不足以证明棋力。
- 全量讲解训练通过两步 GPU 执行检查：41.62 亿可训练参数，FP32 参数、BF16 计算，解码器参数实际变化，峰值 77.66GiB；见 `evidence/full-decoder-sft-verification.json`。这不是完成的讲解模型。全局批量 4、微批量 1 的连续与续跑结果逐位一致，见 `evidence/microbatch-resume.json`。
- 搜索挖掘、真实 BF16 汇总收集、独立讲解裁判及完整对弈入口已实现并通过受控测试，真实模型运行结果尚待产生。
- 终端学习入口支持棋盘显示、推荐讲解、悔棋和完整历史保存/续读；命令行入口已检查，真实训练模型的学习体验仍待验证。

## 正在完成

- 红黑平衡数据上的扩大纯语言基线继续训练；专家组已完成。配置为 `configs/research-v2.json`，按验证集早停。
- Astra 已生成新增 579 个训练根局面的初始讲解，并对新增训练项抽检 72 条；一处“回吃”表述已修正并复核。合并后的 1,315 条原始标注与颜色派生项共 2,630 条（2182 训练、192 验证、256 测试），全部结构校验与输出哈希读回通过；验证/测试文件与原版本字节一致。共 144 条神经抽检，见 `evidence/initial-teacher-full.json`。
- 棋盘字典对照两组正在执行四门课程，配置 `configs/research-v3.json`。这是论文专家输入条件以外的改动，结果与专家消融需单独报告。
- 32 个验证局面的讲解中间检查点评测正在执行，使用独立 100 万节点引擎预算，不接触独立测试。它不代替完成训练后的模型评测。
- 初始讲解 SFT 与实际分支搜索蒸馏。
- 下载并核验官方 **Qwen/Qwen3.8-27B 未量化完整权重**，固定 revision，以 BF16 部署。已有 Q4_K_M GGUF 不用于本项目的汇总教师。完整教师实际加载、精度与汇总生成仍待验证。
- 相同数据预算的纯语言模型基线、专家桥接与蒸馏比较。
- 独立走法质量、解释事实和完整对弈评测。
- 开源发布资产、可复现入口、GitHub CI 与里程碑同步。

## 结果解释

所有新训练结果属于新生成的中国象棋基线，不是论文的西洋棋历史检查点。模型输出必须先核验，再报告非法率、走法损失和解释错误。对弈到达步数上限时记为截尾，不能自动当作和棋或据此换算 Elo。

原始 `runs/pilot-v1`、下载权重、数据与失败日志均保留。运行清单和大文件在本地，轻量可分享证据放入 `evidence/`。尚未完成的训练与评测会保留为未验证状态。

## 当前运行与续跑

- 下载：`tmux attach -t xqgeneral-consolidator-download`；日志 `runs/consolidator-download-resume.log`。完成后 `xqgeneral-teacher-deployment` 自动校验官方 SHA256，待 GPU 3 上的字典纯语言课程释放后执行 BF16 推理验证。日志位于 `runs/consolidator-weight-verification.log` 与 `runs/consolidator-verification-v1/`。
- 专家课程与问答验证已完成；纯语言课程：`xqgeneral-curriculum-v2-text`，日志 `runs/research-v2/text_lora/curriculum/*.log`。
- 字典对照：`xqgeneral-curriculum-v3-bridge`（GPU 2）、`xqgeneral-curriculum-v3-text`（GPU 3），日志 `runs/research-v3/{bridge,text_lora}/curriculum/*.log`。
- 标注收集：`xqgeneral-collect-full-astra`，日志 `runs/astra-full-collection.log`。讲解训练：`xqgeneral-explanation-v2-bridge`（GPU 0），已启动完整解码器 SFT，日志 `runs/explanation-v2-bridge.log`。
- 中间讲解检查：`xqgeneral-interim-validation`（与 GPU 1 上的冻结课程共用），日志 `runs/interim-explanation-v2-001/evaluation.log`，保留了独立检查点副本。
- 训练通过 `scripts/freeze_run.py` 保存执行源码；开发中的后续修改不会改变已启动的训练。课程入口自动从 `latest.pt` 续跑，并核对输入哈希；改变数据或配置必须新建实验。
