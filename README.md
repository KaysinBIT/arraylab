# ArrayLab 阵列分析

用于均匀直线阵分析的开源Python工具，提供Qt桌面工作台与无头命令行。当前版本0.4.0，原创源码采用MIT许可。

桌面采用参考CST的工程工作台布局：顶部功能区、左侧导航树、中央模型与结果视图、右侧参数与指标、底部消息窗口。图标和模型示意独立绘制。0.4.0新增分析方案保存与加载，桌面与无头模式使用同一份JSON参数文件；计算模型仍为均匀直线阵。

![ArrayLab工作台](docs/images/workbench.png)

[下一阶段计划](ROADMAP.md) · [验证记录](docs/validation.md) · [参与开发](CONTRIBUTING.md)

功能分析参考PCAAD说明书，参考PDF不随仓库分发；界面风格参考[CST官方设计环境说明](https://www.3ds.com/products/simulia/cst-studio-suite/electromagnetic-design-environment)及[官方宣传册第3页](https://www.3ds.com/assets/invest/2024-03/simulia-cst-studio-suite-brochure.pdf)。本项目为独立实现。

## 启动

仓库提供源码，不包含预构建EXE。克隆后可在Windows PowerShell中创建环境并启动桌面版：

```powershell
git clone https://github.com/KaysinBIT/arraylab.git
cd arraylab
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install ".[desktop]" pytest
.\.venv\Scripts\python.exe -m arraylab
```

已记录的本地构建环境为Python 3.14.2 x64；`requirements-lock.txt`保留其依赖版本。其他版本的兼容性以对应测试结果为准。

Windows运行包构建成功后，桌面入口为`dist/ArrayLab-0.4.0/ArrayLab.exe`，无头入口为同目录的`ArrayLab-cli.exe`。移动时须保留整个目录及`_internal`；运行包用户无需另装Python。

## 无头运行

从源码运行（已按上节安装依赖）：

```powershell
.\.venv\Scripts\python.exe -m arraylab --headless --elements 20 --frequency-ghz 3 --spacing 0.5 --scan-deg 30 --output-dir ".\results\run01" --plot both
```

构建运行包后的等价入口：

```powershell
.\dist\ArrayLab-0.4.0\ArrayLab-cli.exe --elements 20 --scan-deg 30 --output-dir ".\results\run02" --plot both
```

也可运行`python -m arraylab.cli`或安装后的`arraylab-cli`。独立命令行EXE的启动逻辑不包含Qt；源码无头路径在导入任何Qt模块前分派。无需显示器、桌面会话或`QT_QPA_PLATFORM=offscreen`；不会打开图形窗口。当前实际测试环境仍为Windows，未声称完成Linux/macOS验收。

| 参数 | 作用 |
| --- | --- |
| `--elements` | 阵元数，默认8 |
| `--frequency-ghz` | 工作频率GHz，默认3 |
| `--spacing`、`--spacing-unit` | 间距及单位`lambda`或`mm`，默认0.5λ |
| `--scan-deg` | 扫描角，默认0° |
| `--step-deg` | 绘图/导出采样步长，默认0.1° |
| `--config` | UTF-8 JSON分析方案，例如`examples/analysis.json`；兼容旧版`examples/ula.json`，显式命令行参数覆盖文件值 |
| `--output-dir` | 创建一个新结果目录；已有目录（包括空目录）不覆盖，退出码3 |
| `--include-phase` | DAT文件增加相位列 |
| `--plot` | `none`、`rectangular`、`polar`或`both`，默认none；需要同时指定输出目录 |
| `--help`、`--version` | 查看帮助或版本，不启动界面 |

不指定输出目录时，只向标准输出写一行JSON指标。指定目录时生成：

- `summary.json`：参数、模型/坐标约定、方向性、主瓣指向、HPBW、采样数、提示信息和输出文件名。
- `pattern.dat`：角度°、幅度dB、可选相位°，可在桌面版直接读入。
- `excitation.csv`：阵元编号、x坐标mm、相对幅度、激励相位°。
- `rectangular.png`和/或`polar.png`：按需生成，使用无交互Agg渲染；显示下限−40 dB不影响数据文件。图中文字使用通用英文字体，避免依赖系统中文字体。

结果先在临时目录内全部生成，成功后再发布为目标目录。不要让多项任务共用一个输出目录；批处理脚本为每次调用分配不同路径即可。配置文件的格式及6项参数见[分析方案约定](docs/analysis-format.md)；未知字段、重复字段、错误类型、未知版本和非有限数值会被拒绝。

成功退出码为0，参数/配置内容错误为2，文件读写错误为3，其他运行异常为1。失败时错误JSON写入标准错误，标准输出不混入成功信息；`--help`和`--version`输出普通文本。无唯一主瓣或无完整半功率宽度时，相关字段为JSON的`null`，仍属正常计算结果。

GUI入口也支持`ArrayLab.exe --headless --output-dir ...`，但Windows窗口版EXE不保证标准输出/错误流可用。脚本、重定向和调度任务应使用`ArrayLab-cli.exe`。

仅安装源码计算功能可使用`pip install .`，基础依赖为NumPy和SciPy；导出PNG另安装`pip install ".[plotting]"`；桌面功能安装`pip install ".[desktop]"`。完整锁定清单包含本次构建与测试工具，并非无头计算的最小依赖清单。

## 使用

1. 在右侧参数面板输入频率GHz、阵元数、中心间距（λ或mm），设置正侧向或指定扫描角。
2. 在顶部“主页”点击“开始计算”，或按F5，查看方向图、主瓣指向、半功率波束宽度和方向性。
3. 选择直角坐标或极坐标。在图中移动鼠标读取幅度，点击定位游标；图下工具栏可缩放、平移及保存图片。
4. 勾选“叠加比较”后重算，可保留最新5条曲线。曲线下拉框选定指标和保存的对象。
5. “采样数据”显示方向图数值，“阵元激励”显示位置、单位幅度和馈电相位。
6. 功能区的“保存数据”导出所选曲线的无表头DAT文件；右侧可勾选相位列。“读取数据”可导入2/3列文件作对比。
7. “说明书验证例题”载入p.48的5阵元、0.4λ间距、60°扫描例题并计算。按F1可打开详细说明。
8. 左侧导航树可切换模型、数据及历史计算记录；中央“模型示意”支持透视/x-z视角、参考网格、坐标轴和滚轮缩放。
9. 导航、参数与消息面板可以拖动、浮动、关闭；使用“视图 → 恢复工作台布局”恢复默认位置。底部消息保留本次会话最近300条运行信息。

修改输入不会篡改已有结果。状态条会提示未重算；图形、指标和“保存数据”始终对应选中的结果快照。

## 保存与恢复分析方案

在主页点击“保存方案”（Ctrl+Shift+S），保存右侧当前输入的6项有效参数，即使这些参数尚未计算也可保存。重开程序后点击“打开方案”（Ctrl+Shift+O），载入参数并按F5计算。加载不会自动计算或清除已有曲线，界面会提示结果仍属于此前计算。

方案包含阵元数、频率、间距、间距单位、扫描角和采样步长；不包含结果曲线、导入数据、窗口布局和显示选项。保存已有文件时由文件对话框确认覆盖；校验或写入失败时保留原文件。旧版平铺参数JSON可直接打开，新保存统一使用版本1格式。

桌面保存的JSON可直接用于无头计算，也可显式覆盖其中的参数：

```powershell
python -m arraylab --headless --config "examples/analysis.json" --scan-deg 30 --output-dir "results/scheme01"
```

覆盖仅作用于本次运行，不改动方案文件。详见[完整格式、兼容策略与失败行为](docs/analysis-format.md)。

## 首版范围与坐标

- 仅均匀直线阵、相同各向同性单元、均匀幅度、无互耦、无地平面。阵元数1–200。
- 阵列沿x轴，阵列中心为相位参考。显示x-z前半空间切面，θ从+z朝+x为正，显示范围−90°至90°。
- 各向同性直线阵在三维空间绕x轴旋转对称，图中的单个峰不意味着三维只有一个笔形波束；前后半空间具有对应辐射。
- 方向性D使用全空间总辐射功率，不是二维切面积分，也不等于包含损耗的增益。
- 功率峰值一半的边界用于计算HPBW，对应−3.0103 dB。多等高主瓣、全向单元或边界未落在所示范围内时不报告唯一宽度。
- 主瓣与宽度不依赖绘图采样步长。绘图过粗时提示减小步长；不会用粗采样的最大值假装真实主瓣峰值。
- 新软件首版输入范围：0.001–1000 GHz，间距0.000001–10λ，绘图步长0.01°–5°。除200阵元外，这些范围为本版约束，不是原软件承诺。
- 方向图按理论峰值归一化0 dB；保存数值零点以−300 dB表示。该数值只是有限精度下限，零点的相位不作物理解释。显示下限不改变保存的数据。
- 导入文件只显示数据，不由单个切面反推方向性，也不假定其角度定义与当前模型相同。允许的导入角度范围为−180°至180°，角度必须严格递增。

尚未实现：非均匀幅度、非各向同性单元、互耦、任意位置优化、二维阵列、三维方向图、阻抗与匹配、包含曲线及布局的完整项目持久化。

## 计算与核对

对阵元坐标xₙ与扫描方向u₀=sinθ₀，归一化场为各复指数项之和除以N。首版对均匀阵使用有限几何级数及可去奇点的解析极限；总功率使用球面积分的解析结果。数值测试采用独立逐阵元复数求和和SciPy自适应积分交叉检查。

参考：用户提供说明书IV.D.1，p.45–49；[MathWorks的方向图和方向性定义](https://www.mathworks.com/help/phased/ug/element-and-array-radiation-patterns-and-responses.html)。工具嵌入参考[Matplotlib Qt后端文档](https://matplotlib.org/stable/api/backend_qt_api.html)。外部文档仅用于实现核对，不用于扩大原软件功能范围。

## 测试与构建

- 使用上节的虚拟环境；以下命令中的`python`可替换为`.\.venv\Scripts\python.exe`。
- 数值与文件测试：`python -m pytest`
- Qt交互集成检查：`python -m arraylab --smoke-test tmp/qt-smoke`
- Windows构建依赖：`python -m pip install ".[desktop]" pyinstaller`
- Windows独立运行包构建：`python tools/build_release.py`
- 运行包检查：`dist/ArrayLab-0.4.0/ArrayLab.exe --smoke-test tmp/frozen-smoke`
- 便携包与源码包：`python tools/package_release.py`（先完成运行包构建）。

检查报告区分数学测试、Qt自动交互、真实桌面操作和打包运行结果。打包运行通过不自动证明在未安装Python的另一台干净电脑上通过。

GitHub Actions仅保留手动触发入口，推送和拉取请求不会自动运行测试。本轮验证均安排在推送前，推送后不再测试。手动工作流覆盖Windows/Linux上的数值、文件与无头测试，不替代桌面视觉验收。下一步开发按[ROADMAP](ROADMAP.md)推进。

## 许可

原创项目文件使用MIT许可，见`LICENSE`。第三方依赖及参考说明书不受该MIT许可重新授权，详见`THIRD_PARTY_NOTICES.md`。
