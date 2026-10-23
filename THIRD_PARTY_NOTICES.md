# 第三方组件说明

ArrayLab原创代码采用MIT许可，第三方组件继续适用各自的许可证。运行包的`licenses`文件夹保留所安装发行包中的许可文件，`versions.json`记录实际版本。

| 组件 | 上游与源码入口 |
| --- | --- |
| Python | [Python源代码](https://www.python.org/downloads/source/) |
| PySide6 Essentials 与 Shiboken6 | [Qt for Python许可说明](https://doc.qt.io/qtforpython-6/)、[pyside-setup源代码](https://code.qt.io/cgit/pyside/pyside-setup.git/)、[Qt发布源码](https://download.qt.io/official_releases/qt/) |
| NumPy | [源代码与许可](https://github.com/numpy/numpy) |
| SciPy | [源代码与许可](https://github.com/scipy/scipy) |
| Matplotlib | [源代码与许可](https://github.com/matplotlib/matplotlib) |
| PyInstaller 构建工具 | [源代码与许可例外](https://github.com/pyinstaller/pyinstaller) |

Qt/PySide采用开放源码许可选项，发行包中的Qt动态库未作修改。其他随附依赖及NumPy/SciPy所含底层数学库的声明见其许可文件。字体由Matplotlib附带或使用操作系统已有字体，未复制Windows系统字体。

构建工具会从已安装发行包收集许可元数据。本仓库发布原创源码；二进制分发前仍应按实际发行内容核对全部通知和对应源码提供方式。

PCAAD7Manual.pdf仅用作用户提供的需求参考，不打包到运行包或源代码压缩包中。程序界面、图标、几何示意和帮助文字为本项目独立实现。

## 二维稀疏模块来源

`arraylab/sparse2d`由用户工作区的MIT许可Sparse2D 1.3.0迁入；原计算模块保留，仅调整包内导入。迁移及适配代码沿用本仓库MIT许可。未包含原运行数据、商业说明书或第三方训练模型。数值依赖仍为已列出的NumPy、SciPy、Matplotlib和PySide6，未新增外部数值依赖。迁移范围见`docs/sparse-migration.md`。
