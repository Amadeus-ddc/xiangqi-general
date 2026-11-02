# 恢复实验使用的原始源码提交

GitHub连接器同步已验证文件树时会生成新的提交身份。实验使用的原始本地提交通过[33提交基包](https://github.com/Amadeus-ddc/xiangqi-general/raw/main/reproducibility/source-history-through-502a9be.bundle)和[后续增量包](https://github.com/Amadeus-ddc/xiangqi-general/raw/main/reproducibility/source-history-after-502a9be-through-3d632e7.bundle)恢复。

基包需要公共基础提交`14dff1e0ad4de54109123dabc213ca2758b5006e`，恢复到`502a9beba2191182b5dbf2ebcdfa3c546e9dfd0a`的33个原始提交。增量包需要该末次提交，再引入两个后续提交，恢复到`3d632e740bfb962e7be91253c5c2deb06aeb91af`，累计35个，顺序和身份保持。末次源码包含单着与主线诊断的目录输入、批次覆盖和故障原文保存修复。源码包没有模型、数据、特征缓存、上游副本或运行日志，只保存在Git仓库，不随sdist或wheel分发。

在正常完整克隆的仓库根目录按顺序执行：

```bash
sha256sum reproducibility/source-history-through-502a9be.bundle reproducibility/source-history-after-502a9be-through-3d632e7.bundle
git bundle verify reproducibility/source-history-through-502a9be.bundle
git fetch reproducibility/source-history-through-502a9be.bundle HEAD:refs/remotes/verified-history/base-source
git bundle verify reproducibility/source-history-after-502a9be-through-3d632e7.bundle
git fetch reproducibility/source-history-after-502a9be-through-3d632e7.bundle HEAD:refs/remotes/verified-history/current-source
git show 3d632e740bfb962e7be91253c5c2deb06aeb91af:src/xqgeneral/evaluate_moves.py
git show e8744cf80386d66928174aff823ce0b65a3735fe:src/xqgeneral/paper_curriculum.py
```

基包SHA256为`fdaf5aba65c94004f74ec3a3c62ed93338ef4647dc28b4e1586b0d7375756517`，增量包SHA256为`c2b931f770fc23d2d4a6bf2c1c0b85a356df48f2e4a7cdb56f833b9b588b3332`。这些包需要上述前序提交，不是独立完整仓库备份。

验证使用没有alternates的独立Git仓库，先只导入公共基础提交，再验证、导入基包和增量包。基包导入后尚不存在两个后续提交；增量导入后全部35个原始提交及顺序恢复。当前源码和原生语料实验分别恢复111份模块，并逐字节匹配。全部历史树检查排除运行资产；新增文本对象扫描凭据模式，新增历史中携带的原基包二进制按已验证SHA256复用其审计证明。

此前[截至b9e402a的30提交对象包](https://github.com/Amadeus-ddc/xiangqi-general/raw/main/reproducibility/source-history-through-b9e402a.bundle)和上述33提交基包原字节保留。30提交包SHA256仍为`b14c6ce9f8375113550c96fd9e44cae41546c6af3f087162be179ec61d10da19`，旧`refs/heads/main`入口与[原核验证据](../evidence/paper-rationale-corpus-complete-native-v1.json)继续有效。

这项验证恢复执行源码；训练数据、专家权重、环境和模型身份仍按实验清单准备，不提供棋力或讲解能力结论。[本次软件核验证据](../evidence/model-bundle-auxiliary-evaluations-v1.json)绑定受控回归、完整CPU检查和源码历史链。
