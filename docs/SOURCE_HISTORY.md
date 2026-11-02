# 恢复实验使用的原始源码提交

GitHub连接器同步已验证文件树时会生成新的提交身份。为保留实验引用的本地提交，仓库提供[当前源码历史Git对象包](https://github.com/Amadeus-ddc/xiangqi-general/raw/main/reproducibility/source-history-through-502a9be.bundle)。

当前包包含公共基础提交`14dff1e0ad4de54109123dabc213ca2758b5006e`之后、截至`502a9beba2191182b5dbf2ebcdfa3c546e9dfd0a`的33个原始提交，保留顺序和身份。末次提交包含模型交付目录评测修复；原生语料实验的`e8744cf80386d66928174aff823ce0b65a3735fe`也包含在内。包中没有模型、数据、特征缓存、上游副本或运行日志。二进制对象包保存在Git仓库，sdist和wheel不携带该包。

在已克隆的仓库根目录执行：

```bash
sha256sum reproducibility/source-history-through-502a9be.bundle
git bundle verify reproducibility/source-history-through-502a9be.bundle
git fetch reproducibility/source-history-through-502a9be.bundle HEAD:refs/remotes/verified-history/bundle-evaluation
git show 502a9beba2191182b5dbf2ebcdfa3c546e9dfd0a:src/xqgeneral/model_bundle.py
git show e8744cf80386d66928174aff823ce0b65a3735fe:src/xqgeneral/paper_curriculum.py
```

期望SHA256为`fdaf5aba65c94004f74ec3a3c62ed93338ef4647dc28b4e1586b0d7375756517`。对象包需要上述基础提交，正常克隆包含该提交的本仓库即可满足；它不是独立完整仓库备份。

验证使用没有alternates的独立Git仓库，先只导入公共基础提交，并确认其中不存在上述两个实验提交，再验证、导入对象包。33个原始提交及顺序均恢复；末次提交和原生语料实验分别恢复111份源码并逐字节匹配。所有新增历史树均检查排除大型运行资产；新增文本对象扫描凭据模式，旧源码包二进制按原已验证SHA256复用其审计证明。

此前[截至b9e402a的30提交对象包](https://github.com/Amadeus-ddc/xiangqi-general/raw/main/reproducibility/source-history-through-b9e402a.bundle)原字节保留，SHA256仍为`b14c6ce9f8375113550c96fd9e44cae41546c6af3f087162be179ec61d10da19`。旧包的`refs/heads/main`导入入口与[原核验证据](../evidence/paper-rationale-corpus-complete-native-v1.json)继续有效。

本项验证恢复执行源码；训练数据、专家权重、环境和模型身份仍须按实验清单准备，不提供棋力或讲解能力结论。[当前软件核验摘要](../evidence/model-bundle-evaluation-inputs-v1.json)记录目录评测修复、测试和源码历史绑定。
