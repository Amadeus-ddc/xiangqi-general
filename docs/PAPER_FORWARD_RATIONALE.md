# 未来课程的逐着叙述监督

可选配置`paper_forward_moves_v1`为第三／第四课答案加入按先后顺序的走子叙述，再保留最终任务理由和原来的`答案：`标签。作者的`datagen/tasks/__init__.py`将未来题交给`prose.compose_forward`，后者也在最终理由前加入逐着叙述。本地原简短模板缺少这段监督；这个差异是否影响82%的第四课成绩，尚未通过同条件训练对照验证。

实现读取题目所呈现的局面和已经给定的1—8半回合，用原生规则逐着核对移动棋子、起止格、被吃棋子、将军、将死和困毙。描述区分象棋的困毙与将死。非法给定走子会使准备失败，所有事实来自监督数据，没有调用语言模型。

例如，标准初始局面中红炮走`b2b9`、黑车走`a9b9`时，前缀为“首先，红炮从b2走到b9，吃掉黑马。最后，黑车从a9走到b9，吃掉红炮。”第二着的棋子身份和吃子来自第一着之后的实际棋盘。

## 显式生成新数据

先将`XQ_RATIONALE_ROOTS`、`XQ_RATIONALE_SOURCES`、`XQ_RATIONALE_PRIOR`设置为既有完整源根、完成清单及保留分割目录，`XQ_RATIONALE_OUTPUT`和`XQ_RATIONALE_READBACK`使用新的输出目录。其他采样、根预算、难例比例和工作进程设置沿用准备实验所需的显式选项。

```bash
.venv/bin/python -m xqgeneral.paper_curriculum \
  --roots "$XQ_RATIONALE_ROOTS" --source-manifests "$XQ_RATIONALE_SOURCES" \
  --reference-data "$XQ_RATIONALE_PRIOR" \
  --rationale-profile paper_forward_moves_v1 --output "$XQ_RATIONALE_OUTPUT"
.venv/bin/python -m xqgeneral.paper_curriculum \
  --readback "$XQ_RATIONALE_OUTPUT" --output "$XQ_RATIONALE_READBACK"
```

未指定该选项时，原有答案、问题ID和清单约定保持兼容。启用后只改变未来题的监督正文和身份；`rationale_source_question_id`保留原题ID。题目、原始专家历史、`feature_key`、给定未来、金标标签和预分配棋局归属保持一致。当前局面课程的题目和答案保持原样。采样和颜色镜像先完成，再确定性地加入前缀，因此不消耗原抽题随机数。

生产清单和完整读回都绑定`rationale_profile`。读回按相同源根重新生成答案；不能把该声明删掉后沿用已变更的监督文本。新的数据仍需要索引、特征身份、完整词元长度及生成预算预检。相同历史键支持核对特征复用，不能据此省略对应的输入验证。

## 使用边界

真实输入检查及独立读回覆盖1600题、6928个给定走子步骤，核对了全部最终标签、原始问题、专家历史与参考引擎事实。使用现役分词器后，抽查的新答案最长196 token，提示加完整监督最长325 token，样本内未超过384生成预算或2048训练上限。它不代表完整题库的最大长度，详见[软件与真实输入证据](../evidence/paper-forward-rationale-v1.json)。

现役89模块课程训练仍使用原简短监督。这个选项未回溯改变已报告的验证结果，也没有启动新的学生训练。它首先支持控制父模型、批次与预算之后的文本监督对照；软件和输入检查不证明模型收益。

逐着叙述配置只增加给定变化的过程说明。另有可独立或组合启用的[应将机制说明](PAPER_PARRY_RATIONALE.md)，适配象棋的炮架、马腿及同时解除多将。现役训练尚未使用这些配置。完整复刻差异见[论文对照](PAPER_REPRODUCTION.md)，实际模型状态见[模型卡](MODEL_CARD.md)。
