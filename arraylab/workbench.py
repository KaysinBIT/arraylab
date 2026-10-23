"""CST-inspired dockable engineering workbench; original controls and artwork."""
from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget, QFrame, QToolButton,
    QGroupBox, QFormLayout, QLineEdit, QComboBox, QCheckBox, QPushButton,
    QDockWidget, QScrollArea, QTreeWidget, QTreeWidgetItem, QPlainTextEdit,
)
from .icons import workbench_icon
from .widgets import GeometryView, PatternPlot
from . import __version__


STYLE = """
QWidget { font-family:'Microsoft YaHei','Segoe UI'; font-size:12px; color:#25333d; }
QMainWindow, QWidget#central { background:#e5e9ed; }
QMenuBar { background:#185986; color:white; padding:1px 4px; }
QFrame#menuHeader { background:#185986; }
QMenuBar::item { padding:5px 12px; color:white; background:transparent; }
QMenuBar::item:selected { background:#3276a4; }
QMenu { background:#fbfcfd; border:1px solid #a9b8c6; }
QMenu::item:selected { background:#d7e9f7; }
QLabel#appCaption { color:#eff7fc; padding-right:12px; font-weight:600; }
QTabWidget#ribbon::pane { border:0; border-bottom:1px solid #aebdca; background:#f5f7f9; }
QTabWidget#ribbon QTabBar::tab { background:#e5ebf1; border:0; padding:6px 22px; min-width:46px; }
QTabWidget#ribbon QTabBar::tab:selected { background:#f8fafb; color:#155a89; border-top:2px solid #287bab; }
QFrame#ribbonGroup { border-right:1px solid #cbd3da; background:#f5f7f9; }
QLabel#ribbonCaption { color:#697b89; font-size:11px; }
QToolButton { border:1px solid transparent; border-radius:2px; background:transparent; padding:3px 9px; }
QToolButton:hover { border-color:#91b5d1; background:#e0edf8; }
QToolButton:pressed, QToolButton:checked { border-color:#73a0c3; background:#cadff0; }
QToolButton:disabled { color:#a0aab2; }
QDockWidget { font-weight:600; }
QDockWidget::title { background:#dce6ee; text-align:left; padding:5px 7px; border-bottom:1px solid #b8c8d4; }
QDockWidget::close-button, QDockWidget::float-button { border:0; padding:2px; }
QDockWidget::close-button:hover, QDockWidget::float-button:hover { background:#c1d5e4; }
QMainWindow::separator { background:#ced7df; width:4px; height:4px; }
QMainWindow::separator:hover { background:#69a5cb; }
QTreeWidget { border:0; background:#fbfcfd; selection-background-color:#d3e8f7; selection-color:#174969; }
QTreeWidget::item { height:25px; padding:1px 2px; }
QTreeWidget::item:hover { background:#e9f1f7; }
QGroupBox { border:1px solid #c4d0da; border-radius:0; margin-top:10px; padding-top:9px; background:#f7f9fb; }
QGroupBox::title { subcontrol-origin:margin; left:8px; padding:0 4px; color:#345672; font-weight:600; }
QLineEdit, QComboBox { background:white; border:1px solid #aebdca; border-radius:0; padding:4px 6px; min-height:18px; selection-background-color:#297db0; }
QLineEdit:focus, QComboBox:focus { border-color:#3288bd; }
QLineEdit:disabled { color:#8996a1; background:#edf1f5; }
QPushButton { background:#f5f7f9; border:1px solid #aebecb; border-radius:2px; padding:5px 9px; }
QPushButton:hover { background:#deedf8; border-color:#6fa0c3; }
QPushButton:disabled { color:#a1acb6; border-color:#d4dce2; }
QTabWidget#documents::pane { background:white; border:1px solid #afc1cf; }
QTabBar::tab { background:#e2e9ef; border:1px solid #b6c7d4; border-bottom:0; padding:6px 13px; }
QTabBar::tab:selected { background:white; color:#155d8b; border-top:2px solid #3889ba; }
QTabBar::tab:hover { background:#edf5fa; }
QTableView { border:0; background:white; alternate-background-color:#f2f6fa; gridline-color:#dfe6ec; }
QHeaderView::section { background:#e7eef4; padding:6px; border:0; border-right:1px solid #c9d5de; color:#33536d; }
QPlainTextEdit { border:0; background:#fff; font-family:'Consolas','Microsoft YaHei'; font-size:11px; color:#475967; }
QScrollArea { border:0; background:#edf2f6; }
QLabel#scope, QLabel#viewHint { color:#657c8f; font-size:11px; }
QLabel#state { padding:5px 8px; background:#eef4f8; border-bottom:1px solid #c6d5e1; color:#37617e; font-size:11px; }
QLabel#notice { padding:6px; background:#fff7e5; color:#88621b; border:1px solid #e1cd9e; }
QStatusBar { background:#e4edf4; border-top:1px solid #c1d0dc; }
"""


def ribbon_button(text, icon, callback):
    button = QToolButton()
    button.setText(text)
    button.setToolTip(text.replace("\n", " "))
    button.setIcon(workbench_icon(icon))
    button.setIconSize(QSize(29, 29))
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
    button.setMinimumSize(64, 66)
    button.clicked.connect(callback)
    return button


def ribbon_group(layout, caption, widgets):
    frame = QFrame()
    frame.setObjectName("ribbonGroup")
    v = QVBoxLayout(frame)
    v.setContentsMargins(6, 1, 6, 1)
    v.setSpacing(1)
    row = QHBoxLayout()
    row.setSpacing(2)
    for widget in widgets:
        row.addWidget(widget)
    v.addLayout(row)
    name = QLabel(caption)
    name.setObjectName("ribbonCaption")
    name.setAlignment(Qt.AlignmentFlag.AlignCenter)
    v.addWidget(name)
    layout.addWidget(frame)


def dock(window, title, name, area, content):
    result = QDockWidget(title, window)
    result.setObjectName(name)
    result.setWidget(content)
    result.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
    window.addDockWidget(area, result)
    return result


def build_workbench(w):
    w.setDockNestingEnabled(True)
    w.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea)
    w.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)
    w.setWindowIcon(workbench_icon("model"))

    # A menu widget remains above all dock areas, like a desktop ribbon.
    menu_host = QWidget()
    menu_layout = QVBoxLayout(menu_host)
    menu_layout.setContentsMargins(0, 0, 0, 0)
    menu_layout.setSpacing(0)
    from PySide6.QtWidgets import QMenuBar
    w.workbench_menu = QMenuBar()
    caption = QLabel(f"ArrayLab   |   {__version__}")
    caption.setObjectName("appCaption")
    caption.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    header = QFrame(); header.setObjectName("menuHeader")
    header_row = QHBoxLayout(header); header_row.setContentsMargins(0, 0, 0, 0)
    header_row.addWidget(w.workbench_menu, 1); header_row.addWidget(caption)
    menu_layout.addWidget(header)
    w.ribbon = QTabWidget()
    w.ribbon.setObjectName("ribbon")
    w.ribbon.setFixedHeight(121)
    w.ribbon.setAccessibleName("工作台功能区")
    menu_layout.addWidget(w.ribbon)
    w.setMenuWidget(menu_host)

    pages = []
    for name in ("主页", "分析", "后处理", "视图"):
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(4, 2, 4, 1)
        row.setSpacing(0)
        w.ribbon.addTab(page, name)
        pages.append(row)
    w.save_button = ribbon_button("保存数据", "save", w.save_dialog)
    w.compute_button = ribbon_button("开始计算\nF5", "run", w.compute)
    w.compute_button.setObjectName("compute")
    w.compute_button.setAccessibleName("计算阵列")
    w.example_button = ribbon_button("说明书\n验证例题", "open", w.load_example)
    w.model_button = ribbon_button("模型视图", "model", lambda: w.show_view(3))
    w.result_button = ribbon_button("方向图", "plot", lambda: w.show_view(0))
    w.load_scheme_button = ribbon_button("打开方案", "open", w.open_configuration_dialog)
    w.save_scheme_button = ribbon_button("保存方案", "save", w.save_configuration_dialog)
    w.load_scheme_button.setToolTip("加载分析参数，保留已有结果；加载后请重新计算（Ctrl+Shift+O）")
    w.save_scheme_button.setToolTip("保存右侧当前输入参数为JSON分析方案（Ctrl+Shift+S）")
    ribbon_group(pages[0], "分析方案", [w.load_scheme_button, w.save_scheme_button])
    ribbon_group(pages[0], "数据", [ribbon_button("读取数据", "open", w.open_dialog), w.save_button])
    ribbon_group(pages[0], "模型与参数", [w.model_button, ribbon_button("分析参数", "parameters", w.focus_parameters), ribbon_button("重置参数", "reset", w.reset_parameters)])
    ribbon_group(pages[0], "分析", [w.compute_button, w.example_button])
    ribbon_group(pages[0], "结果", [w.result_button, ribbon_button("数据列表", "table", lambda: w.show_view(1))])
    ribbon_group(pages[1], "计算设置", [ribbon_button("分析参数", "parameters", w.focus_parameters), ribbon_button("开始计算\nF5", "run", w.compute)])
    ribbon_group(pages[1], "模型检查", [ribbon_button("模型视图", "model", lambda: w.show_view(3)), ribbon_button("阵元激励", "table", lambda: w.show_view(2)), ribbon_button("验证例题", "open", w.load_example)])
    ribbon_group(pages[2], "方向图", [ribbon_button("直角坐标", "plot", lambda: w.set_plot_view(0)), ribbon_button("极坐标", "polar", lambda: w.set_plot_view(1)), ribbon_button("完整范围", "reset", w.reset_plot_view)])
    ribbon_group(pages[2], "输出", [ribbon_button("保存数据", "save", w.save_dialog), ribbon_button("保存图片", "plot", w.save_plot), ribbon_button("复制图形", "copy", w.copy_plot)])
    w.restore_layout_button = ribbon_button("恢复布局", "layout", w.restore_workbench)
    ribbon_group(pages[3], "面板", [ribbon_button("导航树", "model", lambda: w.nav_dock.show()), ribbon_button("参数面板", "parameters", w.focus_parameters), ribbon_button("消息窗口", "table", lambda: w.messages_dock.show()), w.restore_layout_button])
    ribbon_group(pages[3], "说明", [ribbon_button("使用帮助\nF1", "help", w.show_help), ribbon_button("关于", "help", w.show_about)])
    for row in pages:
        row.addStretch()

    central = QWidget()
    central.setObjectName("central")
    w.setCentralWidget(central)
    layout = QVBoxLayout(central)
    layout.setContentsMargins(2, 2, 2, 0)
    layout.setSpacing(4)
    controls = QHBoxLayout()
    controls.addWidget(QLabel("结果"))
    w.selection = QComboBox()
    w.selection.setAccessibleName("所选结果曲线")
    w.selection.setMinimumWidth(130)
    w.selection.currentIndexChanged.connect(w.update_selection)
    controls.addWidget(w.selection, 1)
    w.keep_history = QCheckBox("叠加比较（最多5条）")
    controls.addWidget(w.keep_history)
    w.remove_button = QPushButton("移除")
    w.remove_button.clicked.connect(w.remove_selected)
    controls.addWidget(w.remove_button)
    layout.addLayout(controls)
    w.tabs = QTabWidget()
    w.tabs.setObjectName("documents")
    w.tabs.setAccessibleName("结果视图")
    plot_page = QWidget()
    pl = QVBoxLayout(plot_page)
    pl.setContentsMargins(5, 5, 5, 2)
    toolbar = QHBoxLayout()
    w.plot_type = QComboBox()
    w.plot_type.setAccessibleName("方向图类型")
    w.plot_type.addItems(["直角坐标", "极坐标"])
    w.plot_type.currentIndexChanged.connect(w.redraw)
    w.floor_combo = QComboBox()
    w.floor_combo.setAccessibleName("显示下限")
    w.floor_combo.addItems(["−30 dB", "−40 dB", "−60 dB", "−80 dB"])
    w.floor_combo.setCurrentIndex(1)
    w.floor_combo.currentIndexChanged.connect(w.redraw)
    toolbar.addWidget(QLabel("显示方式")); toolbar.addWidget(w.plot_type)
    toolbar.addWidget(QLabel("下限")); toolbar.addWidget(w.floor_combo)
    toolbar.addStretch()
    hint = QLabel("x-z 切面  |  峰值归一化")
    hint.setObjectName("viewHint")
    toolbar.addWidget(hint)
    pl.addLayout(toolbar)
    w.plot = PatternPlot()
    pl.addWidget(w.plot, 1)
    w.cursor = QLabel("移动鼠标读数，单击定位角度游标。")
    w.cursor.setObjectName("scope")
    w.cursor.setWordWrap(True)
    w.plot.cursor_changed.connect(w.cursor.setText)
    pl.addWidget(w.cursor)
    w.tabs.addTab(plot_page, workbench_icon("plot"), "1D 方向图")
    w.pattern_table = w.make_table()
    w.excitation_table = w.make_table()
    w.tabs.addTab(w.pattern_table, workbench_icon("table"), "方向图数据")
    w.tabs.addTab(w.excitation_table, workbench_icon("table"), "阵元激励")
    model_page = QWidget()
    ml = QVBoxLayout(model_page)
    ml.setContentsMargins(0, 0, 0, 0)
    w.geometry = GeometryView()
    model_tools = QHBoxLayout()
    model_tools.setContentsMargins(7, 5, 7, 2)
    model_tools.addWidget(QLabel("视角"))
    w.geometry_view = QComboBox()
    w.geometry_view.addItems(["透视示意", "x-z 切面"])
    w.geometry_view.currentIndexChanged.connect(w.geometry.set_view)
    model_tools.addWidget(w.geometry_view)
    grid = QCheckBox("参考网格"); grid.setChecked(True); grid.toggled.connect(w.geometry.set_grid)
    axes = QCheckBox("坐标轴"); axes.setChecked(True); axes.toggled.connect(w.geometry.set_axes)
    model_tools.addWidget(grid); model_tools.addWidget(axes)
    fit = QPushButton("适配视图"); fit.clicked.connect(w.geometry.reset_view)
    model_tools.addStretch(); model_tools.addWidget(fit)
    ml.addLayout(model_tools)
    ml.addWidget(w.geometry, 1)
    model_note = QLabel("  所选计算结果的阵列示意 · 点标记表示各向同性单元 · 不代表天线实体尺寸")
    model_note.setObjectName("scope")
    model_note.setWordWrap(True)
    ml.addWidget(model_note)
    w.tabs.addTab(model_page, workbench_icon("model"), "模型示意")
    layout.addWidget(w.tabs, 1)

    w.tree = QTreeWidget()
    w.tree.setHeaderHidden(True)
    w.tree.setAccessibleName("工程导航树")
    w.tree.setIndentation(17)
    root = QTreeWidgetItem(w.tree, ["当前分析 · 均匀直线阵"])
    root.setIcon(0, workbench_icon("model"))
    def node(parent, text, kind, data):
        item = QTreeWidgetItem(parent, [text])
        item.setIcon(0, workbench_icon(kind))
        item.setData(0, Qt.ItemDataRole.UserRole, data)
        return item
    model = node(root, "模型", "model", ("view", 3))
    w.model_tree_item = node(model, "均匀直线阵 / 各向同性", "array", ("view", 3))
    node(root, "分析参数", "parameters", ("parameters", 0))
    results = node(root, "结果", "open", ("view", 0))
    w.plot_tree_item = node(results, "1D 方向图", "plot", ("view", 0))
    w.data_tree_item = node(results, "方向图采样数据", "table", ("view", 1))
    node(results, "阵元激励列表", "table", ("view", 2))
    w.history_tree_item = node(root, "计算记录", "open", None)
    w.tree.expandAll()
    w.tree.currentItemChanged.connect(w.select_result_tab)
    w.nav_dock = dock(w, "导航树", "navigationDock", Qt.DockWidgetArea.LeftDockWidgetArea, w.tree)
    w.nav_dock.setMinimumWidth(210)

    inspector = QWidget()
    iv = QVBoxLayout(inspector)
    iv.setContentsMargins(7, 4, 7, 7)
    iv.setSpacing(7)
    parameters_box = QGroupBox("分析参数")
    form = QFormLayout(parameters_box)
    form.setContentsMargins(8, 12, 8, 8)
    form.setVerticalSpacing(6)
    w.inputs = {}
    for key, text, value in [
        ("frequency_ghz", "频率 (GHz)", "3"), ("elements", "阵元数量 N", "8"),
        ("spacing", "中心间距 d", "0.5"), ("scan_deg", "扫描角 (°)", "0"),
        ("step_deg", "采样步长 (°)", "0.1"),
    ]:
        edit = QLineEdit(value)
        edit.setObjectName(key); edit.setAccessibleName(text)
        edit.setMinimumWidth(60)
        w.inputs[key] = edit
        label = QLabel(text); label.setBuddy(edit)
        if key == "spacing":
            box = QWidget(); row = QHBoxLayout(box); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(3)
            row.addWidget(edit, 1)
            w.unit = QComboBox(); w.unit.setAccessibleName("间距单位")
            w.unit.addItem("λ", "lambda"); w.unit.addItem("mm", "mm")
            w.unit.currentIndexChanged.connect(w.change_unit)
            row.addWidget(w.unit)
            form.addRow(label, box)
        else:
            form.addRow(label, edit)
        edit.textChanged.connect(w.mark_dirty)
    w.phase_mode = QComboBox()
    w.phase_mode.setAccessibleName("相位设置方式")
    w.phase_mode.addItems(["正侧向（0°）", "指定扫描角"])
    w.phase_mode.currentIndexChanged.connect(w.phase_mode_changed)
    form.insertRow(3, "相位设置", w.phase_mode)
    form.addRow("单元 / 幅度", QLabel("各向同性 / 均匀"))
    iv.addWidget(parameters_box)
    w.physical = QLabel()
    w.physical.setObjectName("scope"); w.physical.setWordWrap(True)
    iv.addWidget(w.physical)
    metrics = QGroupBox("所选结果摘要")
    mf = QFormLayout(metrics); mf.setContentsMargins(8, 12, 8, 8); mf.setVerticalSpacing(7)
    w.metrics = {}
    for key, text in [("peak", "主瓣指向"), ("beamwidth", "半功率宽度"), ("directivity", "方向性"),
                      ("scan", "设定扫描角"), ("phase", "渐进相移"), ("samples", "采样点数")]:
        field = QLabel("—"); field.setObjectName("metric_" + key)
        field.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        field.setStyleSheet("color:#185d8c; font-weight:600;")
        w.metrics[key] = field; mf.addRow(text, field)
    iv.addWidget(metrics)
    w.notice = QLabel(); w.notice.setObjectName("notice"); w.notice.setWordWrap(True)
    iv.addWidget(w.notice)
    w.phase_export = QCheckBox("导出数据时包含相位列")
    iv.addWidget(w.phase_export)
    scope = QLabel("模型范围：无互耦、无地平面。\n方向性按全空间积分计算。")
    scope.setObjectName("scope"); scope.setWordWrap(True)
    iv.addWidget(scope)
    iv.addStretch()
    scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(inspector)
    w.inspector_dock = dock(w, "参数与结果", "inspectorDock", Qt.DockWidgetArea.RightDockWidgetArea, scroll)
    w.inspector_dock.setMinimumWidth(265)

    messages = QWidget(); mv = QVBoxLayout(messages)
    mv.setContentsMargins(0, 0, 0, 0); mv.setSpacing(0)
    w.state_label = QLabel("准备就绪"); w.state_label.setObjectName("state"); w.state_label.setWordWrap(True)
    mv.addWidget(w.state_label)
    w.message_log = QPlainTextEdit()
    w.message_log.setReadOnly(True); w.message_log.setMaximumBlockCount(300)
    w.message_log.setAccessibleName("计算消息日志")
    mv.addWidget(w.message_log)
    w.messages_dock = dock(w, "消息", "messagesDock", Qt.DockWidgetArea.BottomDockWidgetArea, messages)
    w.messages_dock.setMinimumHeight(95)
    w.statusBar().messageChanged.connect(w.record_message)
    w.statusBar().addPermanentWidget(QLabel("均匀直线阵  |  本地计算"))
    w.statusBar().showMessage("就绪。可设置参数，或载入说明书例题。")
    w.resizeDocks([w.nav_dock, w.inspector_dock], [235, 285], Qt.Orientation.Horizontal)
    w.resizeDocks([w.messages_dock], [135], Qt.Orientation.Vertical)
