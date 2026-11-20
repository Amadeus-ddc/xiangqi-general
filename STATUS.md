# Xiangqi General 当前状态入口

2026-11-20核对：**原目标保持暂停，本项目不允许使用任何GPU或恢复旧队列；可靠象棋教练模型尚未交付。** 源码仓库现为公开，训练权重与原始／衍生数据未随源码发布。

现役运行与交接说明统一在[开发线状态页](https://github.com/Amadeus-ddc/xiangqi-general/blob/d75e7b892f28a020bb7f73a3e427d56415984ca7/STATUS.md)，模型质量、已完成阶段、实际1000步诊断与尚未证明项统一在[开发线模型卡](https://github.com/Amadeus-ddc/xiangqi-general/blob/d75e7b892f28a020bb7f73a3e427d56415984ca7/docs/MODEL_CARD.md)。本分支保留已合并源码；更新文档入口不会把未完成的开发功能集成到main。[PR265](https://github.com/Amadeus-ddc/xiangqi-general/pull/265)仍为开放草稿。

服务器GPU许可来自`runs/GPU_ALLOCATION_POLICY_CURRENT.json`，当前允许列表为空。旧兼容策略、历史“仅GPU0”条目和旧tmux会话不提供恢复授权。用户明确恢复任务后仍需新的资源准入；本次文档整理没有新模型调用、搜索或训练。

原main状态和模型卡可从[固定状态版本](https://github.com/Amadeus-ddc/xiangqi-general/blob/2ffefae5e7bb9c7e50e812b84f57fa27695ab08e/STATUS.md)和[固定模型卡版本](https://github.com/Amadeus-ddc/xiangqi-general/blob/2ffefae5e7bb9c7e50e812b84f57fa27695ab08e/docs/MODEL_CARD.md)查证；更早记录见[状态档案](STATUS_HISTORY.md)和[模型卡档案](docs/MODEL_CARD_HISTORY.md)。这些档案解释当时证据，不代表当前进度。分支、worktree、模型、检查点与未完评测现场保留；本次未清场。
