# 用户指定停课后的四课交接

最新指令直接采用第三课26000步权重进入第四课，已替代此前的无提升条件。26000步本课694/700（99.14%，最弱题型95%），旧课回放1273/1300（97.92%）。第四课实际输入固定为该检查点，前八次更新已核验；2026-10-26 18:49（UTC+8）的计划观测为70步。见[实际交接记录](PAPER_FOURTH_COURSE_START_26000.md)与[证据](../evidence/direct-third-course-26000-handoff-v1.json)。此前的[条件路线记录](../evidence/conditional-third-course-handoff-v1.json)和[24000步报告](PAPER_THIRD_COURSE_24000.md)保留。

本次接入保留第一课选中的24000步与第二课选中的30000步。第二课原最低停止预算是32000步，本次没有达到；完成依据是用户的直接选择。原始分支后续未用于接续的更新和已取消的31000步检查路线保留在各自运行目录，不计入所选30000步父模型的有效训练前缀，也没有生成31000步验证成绩。

## 严格的前缀声明

原普通四课导出仍要求各课满足原停止规则。新的`user_boundary_four_course_curriculum`必须额外提供明确的两课前缀声明，不能仅增加一个通过标记。声明固定三个文件的真实SHA256：

```json
{
  "kind": "explicit_user_authorized_course_prefix",
  "prefix_stages": ["static_current", "dynamic_current"],
  "admission": {"path": "runs/selected-prefix/admission.json", "sha256": "填写该文件的真实64位SHA256"},
  "source_recipe": {"path": "runs/original-foundation.json", "sha256": "填写该文件的真实64位SHA256"},
  "authorization": {"path": "runs/selected-prefix/authorization.json", "sha256": "填写该文件的真实64位SHA256"}
}
```

第三课按条件或明确直接选择结束时，使用有序三课声明：`prefix_stages`为`["static_current", "dynamic_current", "static_future"]`，增加`prior_prefix: {"path": "原两课声明的路径", "sha256": "该声明的真实SHA256"}`。其`admission`固定实际三课来源，`source_recipe`固定第三课所用原配方，`authorization`固定最新明确指令。前三课的导入文件、每次验证、父权重与既有两课授权逐层核对。新配方只允许改变输出位置，第四课必须从所选第三课权重开始。

```bash
python -m xqgeneral.foundation_handoff \
  --curriculum runs/user-curriculum \
  --authorized-prefix runs/selected-prefix.json \
  --output runs/user-curriculum-handoff
```

导出检查原第一课的完整证明、各课全部计划验证、实际快照及原始回答，重新计算当前课选模结果，并保留第二课原停止决定为`false`。两课路线的第三、第四课仍须满足原停止策略。三课条件路线只在本课正确数没有严格增加、授权阈值和原始判定一致时接受。三课直接路线要求最新明确授权，阶段为`static_future`、动作为`select_completed_course_checkpoint_and_continue`，固定真实最佳检查点及SHA256，不要求伪造无提升条件。第三课最低步数已满足，原耐心停止规则的`false`决定仍保留。第四课须满足原停止策略；缺任一课均不能导出。

直接边界分别记录保留的已验证模型分支、源运行完成更新总数、弃用续训数量和范围。终止证明须匹配所选权重；源优化更新日志重新核对SHA256、连续步数和有限损失／梯度／学习率，并进入导出和讲解消费者的输入绑定。本次保留26000步模型分支，源运行实际完成26564步，额外564步及原始日志保留但不用于第四课初始化。

成功导出使用独立类型`user_boundary_foundation_handoff`，明确标记采用原规则或用户指定边界，而不声称四门课都满足原停止规则。讲解配方也必须绑定相同声明：

```json
{
  "require_clean_foundation_handoff": true,
  "authorized_foundation_prefix": {
    "path": "runs/selected-prefix.json",
    "sha256": "填写声明文件的真实64位SHA256"
  }
}
```

未显式绑定、声明改变、只完成两课或后续课程来源不匹配时，讲解入口拒绝启动模型。普通四课路线不需要该字段。当前课选模的起点与最低停止步数分别检查；已经训练到最低停止步数的课程，可以保留更早且符合`selection_start_step`的最佳检查点。

## 实际接续与等待队列

当前GPU课程目录为`runs/curriculum-paper-user-direct26000-v1`，控制会话为`xqgeneral-course4-from-26000-v1`。第四课沿用原89份GPU源码、数据和训练配方，混合第一、第二、第三、第四课的比例为6%、6%、8%、80%；最低48000、最多100000步，每2000步验证，6000步起可参与选模。第三课选定26000步权重，原耐心停止决定保留为`false`。

旧第三课训练树和六个被新指令替代的CPU控制器已按精确身份及进程描述符终止确认。本服务器tmux场景的独立CPU对照无法维持可观测的`SIGSTOP`暂停；新直接指令通过终止确切旧进程树后接续，不补写旧暂停成功记录。独立CPU与小时观测分别位于`runs/user-course4-from-26000-readback-v1`及`runs/user-course4-from-26000-hourly-v1`。

直接26000步路线的讲解消费者使用`runs/user-direct26000-postcourse-v1`及三课声明，冻结108份源码并绑定独立验证记录。其正式模型阶段仍要求实际四课完成及全套本地检查通过。原两课等待器因其课程控制器被最新指令退役而结束，旧源码、退出状态与输入均保留。服务器操作与恢复见[运行定位](QUALIFIED_SFT_ROUTE.md#本服务器运行定位与恢复)。

三课声明、原两课授权、后续入口与所用源码必须一致。实际三课声明及弃用源日志已用新校验器复核；本次元数据复读沿用实际生产者的原生验证与权重核验。四课完成前继续保持讲解门禁为假。

讲解仍使用原筛选后的9823条训练、384条验证和512条独立测试。四轮2456次更新的完整rank顺序与旧安排一致：39292条有效呈现、4条分布式填充，共39296条全局样本。词元预检和4041份约3.30TB缓存的既有验证凭据复用，本次不重读全部缓存载荷。完整BF16教师、开发验证、384局基线及512根搜索产量试验保持原设置；512根仅是试验，不能代替足量蒸馏和最终验收。

## 验证范围

原两课路线的1708项及三课条件扩展的1724项CPU检查证据保留。最新直接三课扩展通过1738项完整CPU检查，其中前缀模块45项检查覆盖原生问答／受控小型张量的四课导出、讲解流程串联、两层授权锁定、弃用源日志与条件阈值重算；缺少第四课或记录变化均拒收。这些夹具没有训练真实GPU模型，见[本次记录](../evidence/direct-third-course-26000-handoff-v1.json)。实际两课预检曾重新核对各15轮问题、问法、原始判分、快照头与训练输入来源；先前各轮完整CPU参数读回保留。真实四课导出仍须在第三、第四课结束后执行。

完整四课父模型导出和新的四卡全解码器容量／连续四步对二加二恢复对照通过之后，才能开始正式讲解训练。第三课26000步验证与第四课实际启动已完成；最终模型的棋力与教学验收仍未完成。
