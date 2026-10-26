# 参与开发

先查看[当前功能与运行方法](README.md)、[下一阶段计划](ROADMAP.md)和[验证边界](docs/validation.md)。

## 代码组织与依赖

实际实现按[结构说明](docs/structure-refactor.md)放在`arraylab/framework`、`application`、`features`和`shared`。新业务代码进入对应feature，不再加入根目录前缀文件。根目录旧文件只用于公开导入兼容，不应成为内部依赖；跨业务使用feature包的显式公开入口，数值入口不应提前加载Qt。

单元测试放在对应模块的`tests/`，原`tests/test_*.py`保留转发以兼容既有命令，并只默认收集一份。`verification/`是可在独立程序中运行的桌面验收代码，不能与不进入wheel的单元测试混淆。新增任务在`application/desktop/task_registry.py`注册；新增CLI在`application/cli.py`按明确优先级注册。不要让基础框架反向导入业务，也不要把没有实际复用关系的代码放入shared。

结构修改可执行`python tools/check_architecture.py --report tmp/architecture.json`检查边界与循环，再选择受影响入口、记录和界面路径验证。源码测试可直接运行就近文件，旧测试命令继续有效。现有JSON/SQLite格式、AppData位置和环境变量属于兼容范围，不随文件迁移改名。

## 开发检查

桌面尺寸响应使用`tests/test_resize_canvas.py`及`--smoke-suite resize`检查；共享3D画布继承变化时选择受影响交互测试，避免扩展为全量数值回归。记录真实Dock宽度、完整绘制次数、最终尺寸/数据/导出，以及计时环境，不能只测延迟计时器是否触发。

任务树导航调整使用`tests/test_navigation.py`和`--smoke-suite navigation`定向验证，覆盖全部已有叶节点、双向选择与记录入口，不运行全套业务数值回归。应用层`application/desktop/navigation.py`只组合现有视图/参数控件，不启动计算或创建新历史库。

建议使用Python 3.14创建独立环境。仅修改计算/无头功能时安装`.[plotting]`与pytest；修改桌面界面时安装`.[desktop]`与pytest。

```text
python -m pip install -e ".[desktop]" pytest
python -m pytest -q
python -m arraylab --smoke-test tmp/qt-smoke
```

以上完整命令保留供明确需要时使用，不作为每次修改的默认步骤。按用户维护要求，每次只测试修改或新增的功能及其直接影响路径，不自动执行全量回归。Qt自动交互检查不等同于手工视觉或干净电脑验收；根据改动选择相关输入、单位、边界、GUI/CLI或文件读写检查。

修改二维稀疏的图形、菜单或历史联动时，可先用专项入口缩短反馈周期：

```text
python -m pytest -q tests/test_sparse_overview_plot.py tests/test_sparse_plot_cache.py tests/test_sparse_history_store.py tests/test_smoke_suites.py tests/test_plot_layout.py tests/test_footer_layout_cache.py
python -m arraylab --smoke-test tmp/sparse-view --smoke-suite sparse-view
```

专项包含并排图的真实Qt交互、全部稀疏导航和自动历史；基本综合/导出/馈电交接保留在完整检查。删除重复的“仅判断存在坐标轴”循环，由导航组统一覆盖入口。`smoke.json`列出本次范围、每组通过数量与耗时，失败组也留下已完成检查。数值测试负责场/指标一致性和图元边界，Qt检查负责实际选择、平移、放大、图片导出、小窗口及恢复，避免在多个层次反复启动相同综合。

独立包可向`tools/check_headless_release.py`传入`--gui-smoke --gui-suite sparse-view`。该模式只包含基础无头验收和稀疏专项，不运行需要全量桌面产物的GUI/CLI联验；报告明确标记范围。跨模块修改时只补充直接受影响的检查，完整回归需用户明确要求。不要把专项通过数量作为全软件验收结果。

MIMO功能可先执行`python -m pytest -q tests/test_mimo.py tests/test_mimo_io.py tests/test_mimo_cli.py`与`python -m arraylab --smoke-test tmp/mimo --smoke-suite mimo`。MIMO专项使用独立窗口/记录和实际后台计算，涵盖阵列交接及失败恢复；完整冻结验收额外比较一维、二维桌面快照与无头输出，并验证PNG导出。相关说明见[雷达MIMO](docs/mimo-arrays.md)。

波形功能使用`tests/test_waveform_core.py`、`test_waveform_ab.py`、`test_waveform_ambiguity.py`、`test_waveform_pri.py`与`test_waveform_integration.py`，按修改范围选择。1.4新增领域分别使用`test_waveform_doa.py`、`test_waveform_gain.py`、`test_waveform_antenna.py`、`test_waveform_sparse_design.py`、`test_radar_sync.py`和`test_waveform_radar_desktop.py`，不要因版本同步重跑未改的四个旧引擎。桌面专项为`python -m arraylab --smoke-test tmp/waveform --smoke-suite waveform`；独立程序使用`python tools/check_waveform_release.py <运行包目录> <报告路径>`，只检查雷达波形/天线及直接入口，不顺带运行其他工作台。原WaveLab项目保持只读，开发对照只使用固定快照副本。

按当前维护约定，验证在推送前完成，推送后不再测试。GitHub Actions仅接受手动触发，不随push或pull_request自动运行。方案格式变更须保持旧版有效文件可读，并验证失败时原参数、文件和结果不受损。

## 提交内容

数学或模型变更应说明假设、输出含义及独立验证方法。界面变更附清晰的真实运行截图，并说明验证尺寸。文档或实现必须区分方向性与增益、二维切面与全空间积分、随机实现与平均统计量。

请勿提交商业参考说明书、第三方程序或图标、环境目录、构建产物、本地结果、日志、凭据或个人路径。图片应为项目自身界面的干净截图。原创代码适用MIT许可，第三方依赖遵循各自许可。
