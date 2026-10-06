# Xiangqi General（象棋将军）

将 [Queen 论文](https://arxiv.org/abs/2610.03695v1) 的棋类专家与语言模型桥接路线迁移到中国象棋，目标是同时提供强走法和可核对的中文讲解。当前能力与实测限制见 [STATUS.md](STATUS.md)，模型用途与评测边界见 [模型说明](docs/MODEL_CARD.md)。

专家为冻结的 Px0 20 层、512 维 Transformer；语言模型为固定 revision 的 Qwen3-4B-Instruct-2507。四个可训练交叉注意力块读取 90 个棋盘 token。首轮原型使用宽度 384，完整配置、分割、检查点选择和输出校验由各实验 manifest 记录。

## 安装与 CPU 验证

Python 3.11+。先安装适合机器的 PyTorch，再安装项目：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,train,reference]'
python -m pytest -q
python -m build
python scripts/check_release.py
```

测试不下载模型、不要求 GPU。正式训练需要 CUDA GPU；依赖只安装在项目环境。

## 公共源码与模型

```bash
python scripts/fetch_sources.py
python scripts/bootstrap.py
python scripts/build_native_reference.py
python scripts/build_native_network.py
python -m xqgeneral.verify_network
```

下载源码与权重的身份固定在 `configs/sources.json` 和权重 manifest。权重、数据、上游副本和运行产物不进入 Git。完整网络对照使用官方 C++ 输入编码器、未改动的 C++ ONNX 导出器和 ONNX CPU 后端，比较全部策略、胜和负、剩余步数及桥接特征。

## 数据与训练

```bash
python -m xqgeneral.data --games 120 --output data/research-v1
python -m xqgeneral.cache_features --data data/research-v1 --output data/research-v1/features.pt
python -m xqgeneral.train --config configs/pilot.json
```

新实验必须使用新的输出目录。课程按棋局预先划分训练、验证、独立测试，检查当前与未来局面重叠。课程间继承验证集选出的最佳检查点，并混合之前课程。每次运行保存配置、输入与输出哈希、源代码身份和验证结果。长任务放在 tmux，完整研究配置和后续评测入口随已验证里程碑补齐。

## 原型推理

已有本地首轮检查点时：

```bash
python -m xqgeneral.ask --adapter runs/pilot-v1/adapter.pt \
  --moves b0c2 --question 'c2 上是什么棋子？只回答棋子名称。'
```

原型检查点只完成过 80 步训练。它的回答不调用皮卡鱼，当前不具备成熟教练的验证证据。

## 开源与贡献

源码采用 GPL-3.0-or-later，保留 Px0 派生实现与协议的来源声明。依赖、参考代码与权重的边界见 [第三方说明](THIRD_PARTY_NOTICES.md)，贡献步骤见 [CONTRIBUTING.md](CONTRIBUTING.md)。本项目独立于 Queen、Px0、Qwen 和皮卡鱼官方项目。训练完成后的棋力与讲解结论以独立评测为准。
