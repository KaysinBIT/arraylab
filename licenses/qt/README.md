# Qt / PySide / WebEngine 许可补充材料

本目录随 ArrayLab 源码包和运行包一起分发，补充所用 wheel 未携带的开源许可文本及 WebEngine 第三方声明。原 wheel 的商业许可文件仍保留在运行包原有的 `licenses/PySide6-*`、`licenses/shiboken6` 目录，不以本目录替换或删除。

## 核实的版本与内容

- PySide6-Essentials、PySide6-Addons、shiboken6：6.12.0；PySide6-WebEngine：6.12.0.140。
- 上述四个发行包实际元数据均标注 `LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only`，但 `.dist-info/licenses` 仅提供 `LicenseRef-Qt-Commercial.txt`。
- 安装运行库报告 Qt 6.12.0、Qt WebEngine 6.140.0、Chromium 140.0.7339.225，以及安全补丁版本 152.0.7977.83。后一个字段按运行库原样记录，不用它推定完整 Chromium 源码版本。
- `pyside-6.12.0/` 保存官方 PySide v6.12.0 的8份许可文本，包括 LGPLv3、GPLv2/GPLv3及 Qt GPL 例外；`qtbase-6.12.0/` 补充官方 QtBase v6.12.0 的 GPLv2-or-later、LGPLv2.1-or-later 文本。
- `qtwebengine-6.140/` 保存官方 Qt WebEngine 6.140 分支固定提交的12份许可证及 `LICENSE.Chromium`。
- `webengine-third-party/` 保存 Qt 官方 WebEngine 6.140.0 文档列出的125份组件许可/通知正文，保留各自版权声明和条款。不是用单独的 Qt LGPL 文本覆盖 Chromium、FFmpeg、WebKit 等各组件的许可。

每份文件的来源URL、源页面摘要、输出摘要和取得日期见 `SOURCES.json`。许可文件逐字保存；第三方文档中的预格式化许可/通知正文经HTML实体解码后保存，文件开头记录标题及来源。未改写条款。每一页还在 `webengine-third-party/html/` 保留未改动的原始HTML，包含预格式化正文之外的作者、版权、归属及上下文；打包只归档这些文件，不执行网页。文档内容归属 Qt 文档及其原作者，相关 GFDL 文本包含在目录内；各组件自己的许可证独立适用。

## 源码入口与动态库

- [PySide / Shiboken v6.12.0](https://github.com/qtproject/pyside-pyside-setup/tree/v6.12.0)
- [QtBase v6.12.0](https://github.com/qt/qtbase/tree/v6.12.0)
- [Qt各模块官方源代码](https://code.qt.io/cgit/qt/)
- [Qt源码发行目录](https://download.qt.io/official_releases/qt/6.12/6.12.0/)
- [Qt WebEngine固定分支提交](https://github.com/qt/qtwebengine/tree/e4d8df13c71ea53ac2de3c75d47585bcc0da815d)
- [Qt WebEngine 6.140.0许可与逐组件索引](https://doc-snapshots.qt.io/qtwebengine/qtwebengine-licensing.html)

ArrayLab原创代码采用MIT许可。随包Qt动态库未作修改，以独立文件保留在 `_internal/PySide6`，并保留本项目构建脚本；该布局不限制接收方按相关组件许可证使用兼容修改版重新构建或替换动态库。本目录不是Qt/Chromium完整源码包。

## 证据范围

本次用全新的普通 `QWebEnginePage` / 内存配置尝试打开 `chrome://credits/` 和 `qtwebengine://qtwebengine-credits/` 均失败，未经过ArrayLab场景拦截器。官方6.140分支的WebUI工厂也未注册credits路由。检查交付库PAK资源中的gzip/原始HTML未取得完整credits页面。因此，本目录第三方正文来自官方6.140.0文档，**不是运行库内嵌credits的成功导出，也不是已证明逐DLL、逐平台精确匹配的SBOM**。

取得材料时，官方公开仓库未提供 `v6.140.0` 正式标签，仅见预发布标签及6.140分支；故WebEngine源许可文件固定为 `e4d8df13c71ea53ac2de3c75d47585bcc0da815d`，不宣称已证明该提交与wheel构建完全一致。实际版本、来源和这些限制一并保留，后续升级依赖时应重新核对本目录。
