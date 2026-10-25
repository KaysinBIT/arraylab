"""Linear result views and parameters owned by the linear feature."""
"""CST-inspired dockable engineering workbench; original controls and artwork."""
from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget, QFrame, QToolButton,
    QGroupBox, QFormLayout, QLineEdit, QComboBox, QCheckBox, QPushButton,
    QDockWidget, QScrollArea, QTreeWidget, QTreeWidgetItem, QPlainTextEdit, QStackedWidget,
    QSizePolicy, QMenu,
)
from arraylab.shared.qt.icons import application_icon, workbench_icon
from arraylab.features.linear.widgets import GeometryView, PatternPlot
from arraylab import __version__


def build_linear_views(w):
    plot_page = QWidget()
    pl = QVBoxLayout(plot_page)
    pl.setContentsMargins(5, 5, 5, 2)
    toolbar = QHBoxLayout()
    w.plot_type = QComboBox()
    w.plot_type.setAccessibleName("方向图类型")
    w.plot_type.addItems(["直角坐标", "极坐标", "三维方向图"])
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
    w.tabs.addTab(plot_page, workbench_icon("plot"), "方向图")
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


def build_linear_parameters(w, iv):
    parameters_box = QGroupBox("分析参数")
    w.parameters_box=parameters_box
    form = QFormLayout(parameters_box);w.parameter_form=form
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
    w.linear_element_label=QLabel("各向同性 / 均匀");form.addRow("单元 / 幅度",w.linear_element_label)
    iv.addWidget(parameters_box)
    w.physical = QLabel()
    w.physical.setObjectName("scope"); w.physical.setWordWrap(True)
    iv.addWidget(w.physical)
    metrics = QGroupBox("所选阵列结果摘要")
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
    w.linear_only_widgets=[w.physical,metrics,w.notice,w.phase_export,scope]
