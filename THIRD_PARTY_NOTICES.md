# 第三方组件说明

ArrayLab原创代码采用MIT许可，第三方组件继续适用各自的许可证。运行包的`licenses`文件夹保留所安装发行包中的许可文件，`versions.json`记录实际版本。

| 组件 | 上游与源码入口 |
| --- | --- |
| Python | [Python源代码](https://www.python.org/downloads/source/) |
| PySide6 Essentials、Addons 与 Shiboken6 | [Qt for Python许可说明](https://doc.qt.io/qtforpython-6/)、[pyside-setup源代码](https://code.qt.io/cgit/pyside/pyside-setup.git/)、[Qt发布源码](https://download.qt.io/official_releases/qt/) |
| Qt WebEngine / Chromium | [Qt WebEngine源码](https://code.qt.io/cgit/qt/qtwebengine.git/)；用于桌面内嵌离线三维视口，随Qt动态库及许可元信息分发 |
| Three.js 0.180.0 | [源码与MIT许可](https://github.com/mrdoob/three.js/tree/r180)；场景资源保留`THREE-LICENSE.txt`，运行包另附`licenses/three/LICENSE.txt` |
| NumPy | [源代码与许可](https://github.com/numpy/numpy) |
| SciPy | [源代码与许可](https://github.com/scipy/scipy) |
| threadpoolctl | [线程数控制与BSD-3-Clause许可](https://github.com/joblib/threadpoolctl) |
| psutil | [源码与许可](https://github.com/giampaolo/psutil) |
| Rasterio / GDAL | [Rasterio](https://github.com/rasterio/rasterio)、[GDAL](https://github.com/OSGeo/gdal)；地形栅格读入与重投影 |
| pyproj / PROJ | [pyproj](https://github.com/pyproj4/pyproj)、[PROJ](https://github.com/OSGeo/PROJ)；坐标转换，随包保留发行元数据中的许可证 |
| Matplotlib | [源代码与许可](https://github.com/matplotlib/matplotlib) |
| PyInstaller 构建工具 | [源代码与许可例外](https://github.com/pyinstaller/pyinstaller) |

Qt/PySide采用开放源码许可选项，发行包中的Qt动态库未作修改。其他随附依赖及NumPy/SciPy所含底层数学库的声明见其许可文件。字体由Matplotlib附带或使用操作系统已有字体，未复制Windows系统字体。

构建工具会从已安装发行包收集许可元数据。本仓库发布原创源码；二进制分发前仍应按实际发行内容核对全部通知和对应源码提供方式。

0.26.0正式包另附`licenses/qt/`：Qt/PySide官方开源许可文本、Qt GPL例外、Qt WebEngine 6.140.0官方文档中的125份第三方通知，以及可核对的来源和摘要清单。保留原wheel内的商业许可文件；不把一份Qt许可证视作全部Chromium依赖的许可证。运行库credits页面未能导出，官方WebEngine正式源码标签未公开，故这份归档不声称是与各DLL精确对应的内嵌credits或SBOM；具体版本、固定分支提交及证据边界见[许可补充说明](licenses/qt/README.md)。

脉冲雷达三维场景保留原PulseLab的Three.js渲染形式，仅把离线场景视口嵌入ArrayLab；原项目服务器与整站界面不随包运行。Qt 6.12将WebEngine改为独立发行包，本项目使用`PySide6-WebEngine 6.12.0.140`及其要求的`PySide6-Addons 6.12.0`；参见[Qt官方发行说明](https://www.qt.io/blog/qt-for-python-release-6.12-is-here)及[该wheel的依赖元数据](https://pypi.org/project/PySide6-WebEngine/6.12.0.140/)。其辅助进程、渲染资源及本地化文件按PyInstaller的Qt规则打包，CLI启动代码仍不导入Qt。场景可修改源码、构建入口与来源说明一并随源码包保留，运行时不需要Node.js或联网加载脚本。

PCAAD7Manual.pdf仅用作用户提供的需求参考，不打包到运行包或源代码压缩包中。程序界面、图标、几何示意和帮助文字为本项目独立实现。

0.26.0.dev0 集成用户工作区交付的 ArrayStudio 1.2.1、EquationLab 1.0、PulseLab 1.0.1。来源、固定快照摘要及改动边界见 `docs/radar-components.md`。仅收录明确的代码、参数示例与人工合成 DEM；不包含原运行目录、用户工程或独立服务。传播模型数据的依据随 `arraylab/features/equation/data/PROPAGATION_SOURCES.md` 保留。附属地理库和其依赖遵循各自许可。

0.26.1将相控阵与雷达方程来源更新为ArrayStudio 1.2.2、EquationLab 1.0.1的已完成修复交付包；PulseLab 1.0.1、WaveLab 1.4保持。固定摘要与迁移范围见`docs/upstream-sync-0261.md`。本轮未新增或升级第三方运行依赖，既有Qt、Three.js和地理库许可材料继续适用，不改写原项目或纳入其运行数据。

## 二维稀疏模块来源

`arraylab/sparse2d`由用户工作区的MIT许可Sparse2D 1.3.0迁入；0.9.0迁移时原计算模块保留，仅调整包内导入。0.12.0增加有界并行复核和取消检查，优化算法与数学定义保持。迁移及适配代码沿用本仓库MIT许可。未包含原运行数据、商业说明书或第三方训练模型。0.12.0新增已列出的threadpoolctl，用于控制数值库线程数。迁移范围见`docs/sparse-migration.md`。
