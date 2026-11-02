# 恢复实验使用的原始源码提交

GitHub连接器能够同步已验证文件树，但会生成新的提交身份。为保留原生实验引用的本地提交，本仓库另提供[源码历史Git对象包](https://github.com/Amadeus-ddc/xiangqi-general/raw/main/reproducibility/source-history-through-b9e402a.bundle)。

它包含远端原有基础提交`14dff1e0ad4de54109123dabc213ca2758b5006e`之后、截至`b9e402abb2d0f261bad986c63eb837fa29289c42`的30个原始提交，保留其顺序和身份。它不包含模型、数据、特征缓存、上游副本或运行日志。二进制对象包保存在Git仓库，sdist和wheel不携带该包；需要精确复现实验时从仓库获取。

在已克隆的仓库根目录执行：

```bash
sha256sum reproducibility/source-history-through-b9e402a.bundle
git bundle verify reproducibility/source-history-through-b9e402a.bundle
git fetch reproducibility/source-history-through-b9e402a.bundle refs/heads/main:refs/remotes/verified-history/main
git show e8744cf80386d66928174aff823ce0b65a3735fe:src/xqgeneral/paper_curriculum.py
```

期望SHA256为`b14c6ce9f8375113550c96fd9e44cae41546c6af3f087162be179ec61d10da19`。对象包需要上述基础提交，正常克隆包含该提交的本仓库即可满足；它不是独立完整仓库备份。

验证使用不共享本地对象、没有alternates的独立Git仓库：先仅导入远端已有基础提交，确认其中不存在实验提交，再验证并导入对象包。30个原始提交均恢复，`e8744cf80386d66928174aff823ce0b65a3735fe`中的111份源码与原生完整语料核验清单逐份匹配。验证还扫描了新增Git对象，并检查所有历史树不包含被排除的大型资产。

这项验证恢复的是执行源码。训练数据、专家权重、环境与模型身份仍须按相应实验清单分别准备；它不提供棋力或讲解能力结论。核验摘要见[完整语料证据](../evidence/paper-rationale-corpus-complete-native-v1.json)。
