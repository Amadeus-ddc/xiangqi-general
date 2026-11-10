# 对弈推理转换缓存

`xqgeneral.evaluate_games` 支持可选参数 `--autocast-cache on` 或 `--autocast-cache off`，转发给已有的 `Predictor` 推理覆盖接口。省略时沿用检查点的设置，参数合同中不增加空字段。

```bash
.venv/bin/python -m xqgeneral.evaluate_games \
  --checkpoint /path/to/selected/adapter.pt \
  --autocast-cache on --output runs/new-cached-matches
```

该设置控制自动混合精度中的临时权重转换缓存，保持原参数精度与检查点文件。启用缓存可能增加显存占用；CPU接口检查不能证明实际加速、完整上下文容量或原始回答一致。改变运行设置前应在同一实际模型上比较原始输出、生成耗时和显存。

显式开关写入对弈合同；续跑须使用相同冻结源码和相同开关，省略与显式关闭也属于不同合同。改变设置须使用新的输出目录。既有运行继续使用其原冻结源码与合同。

加载器实际采用覆盖设置后才开始对弈；未采用则报错。完成统计中的 `inference_autocast_cache` 区分请求值和本次实际加载后的观察值。如果中断发生在所有对局已保存、完成清单尚未写入时，续跑无需加载模型，观察值为 `null`。已完成运行的再次读回同样不会加载模型。

原始回答、逐回合保存、非法走法判负、终局和截断规则保持；缓存开关不修正模型输出，也不构成棋力或讲解质量证明。
