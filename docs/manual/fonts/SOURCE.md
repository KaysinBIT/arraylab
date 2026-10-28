# 手册字体来源

本目录的 NotoSansSC-Regular.ttf 由 Google Fonts 官方 Noto Sans SC 可变字体提取 wght=400 静态实例，供PDF生成器使用。原始名称不含OFL保留名称“Source”；保留字体原版权及OFL1.1许可。不是从Windows系统字体复制。

- 原始字体：https://raw.githubusercontent.com/google/fonts/main/ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf
- 取得日期：2026-10-28。
- 原始文件SHA-256：`a3041811a78c361b1de50f953c805e0244951c21c5bd412f7232ef0d899af0da`。
- 官方许可：https://raw.githubusercontent.com/google/fonts/main/ofl/notosanssc/OFL.txt
- 许可文件SHA-256：`1c05c68c34f9708415aada51f17e1b0092d2cea709bf4a94cd38114f9e73d7d9`。

转换使用fontTools.varLib.instancer.instantiateVariableFont，轴值wght=400，其余轴保持默认；生成字体的内部家族/样式名称改为ArrayLab Manual Sans SC / Regular，避免沿用变量字体默认的Thin样式名称。PDF由ReportLab自动嵌入实际使用字形子集。OFL允许按其条件嵌入和分发；完整许可见同目录OFL.txt。

构建说明书只需本目录静态字体和文档专用的ReportLab、pypdf。无需安装字体到操作系统，也无需在软件运行时加载此文件。
