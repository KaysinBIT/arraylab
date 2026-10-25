# 参与开发

先查看[当前功能与运行方法](README.md)、[下一阶段计划](ROADMAP.md)和[验证边界](docs/validation.md)。

## 开发检查

建议使用Python 3.14创建独立环境。仅修改计算/无头功能时安装`.[plotting]`与pytest；修改桌面界面时安装`.[desktop]`与pytest。

```text
python -m pip install -e ".[desktop]" pytest
python -m pytest -q
python -m arraylab --smoke-test tmp/qt-smoke
```

最后一项是Qt自动交互检查，不等同于手工视觉或干净电脑验收。提交前检查输入、单位、特殊条件、GUI/CLI结果一致性，以及相应文件读写。

修改二维稀疏的图形、菜单或历史联动时，可先用专项入口缩短反馈周期：

```text
python -m pytest -q tests/test_sparse_overview_plot.py tests/test_sparse_plot_cache.py tests/test_sparse_history_store.py tests/test_smoke_suites.py tests/test_plot_layout.py tests/test_footer_layout_cache.py
python -m arraylab --smoke-test tmp/sparse-view --smoke-suite sparse-view
```

专项包含并排图的真实Qt交互、全部稀疏导航和自动历史；基本综合/导出/馈电交接保留在完整检查。删除重复的“仅判断存在坐标轴”循环，由导航组统一覆盖入口。`smoke.json`列出本次范围、每组通过数量与耗时，失败组也留下已完成检查。数值测试负责场/指标一致性和图元边界，Qt检查负责实际选择、平移、放大、图片导出、小窗口及恢复，避免在多个层次反复启动相同综合。

独立包可向`tools/check_headless_release.py`传入`--gui-smoke --gui-suite sparse-view`。该模式只包含基础无头验收和稀疏专项，不运行需要全量桌面产物的GUI/CLI联验；报告明确标记范围。发布或改动跨模块逻辑时，省略suite参数执行默认完整检查。不要把专项通过数量作为全软件验收结果。

MIMO功能可先执行`python -m pytest -q tests/test_mimo.py tests/test_mimo_io.py tests/test_mimo_cli.py`与`python -m arraylab --smoke-test tmp/mimo --smoke-suite mimo`。MIMO专项使用独立窗口/记录和实际后台计算，涵盖阵列交接及失败恢复；完整冻结验收额外比较一维、二维桌面快照与无头输出，并验证PNG导出。相关说明见[雷达MIMO](docs/mimo-arrays.md)。

按当前维护约定，验证在推送前完成，推送后不再测试。GitHub Actions仅接受手动触发，不随push或pull_request自动运行。方案格式变更须保持旧版有效文件可读，并验证失败时原参数、文件和结果不受损。

## 提交内容

数学或模型变更应说明假设、输出含义及独立验证方法。界面变更附清晰的真实运行截图，并说明验证尺寸。文档或实现必须区分方向性与增益、二维切面与全空间积分、随机实现与平均统计量。

请勿提交商业参考说明书、第三方程序或图标、环境目录、构建产物、本地结果、日志、凭据或个人路径。图片应为项目自身界面的干净截图。原创代码适用MIT许可，第三方依赖遵循各自许可。
