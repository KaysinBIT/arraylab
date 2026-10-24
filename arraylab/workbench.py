"""CST-inspired dockable engineering workbench; original controls and artwork."""
from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget, QFrame, QToolButton,
    QGroupBox, QFormLayout, QLineEdit, QComboBox, QCheckBox, QPushButton,
    QDockWidget, QScrollArea, QTreeWidget, QTreeWidgetItem, QPlainTextEdit, QStackedWidget,
    QSizePolicy,
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


def sidebar_grid(page):
    """Stack label/control pairs into one column before using a former wide form."""
    grid=page.layout();items=[]
    if not hasattr(grid,'getItemPosition'):return
    for i in range(grid.count()):
        row,col,rs,cs=grid.getItemPosition(i);items.append((row,col,rs,cs,grid.itemAt(i).widget()))
    while grid.count():grid.takeAt(0)
    for col in range(4):grid.setColumnStretch(col,0);grid.setColumnMinimumWidth(col,0)
    items.sort(key=lambda item:(item[0],item[1]));row=0;i=0
    while i<len(items):
        old_r,col,rs,cs,widget=items[i]
        if cs==1 and col%2==0 and i+1<len(items) and items[i+1][0]==old_r and items[i+1][1]==col+1:
            if isinstance(widget,QLabel):widget.setWordWrap(True)
            grid.addWidget(widget,row,0);grid.addWidget(items[i+1][4],row,1);i+=2
        else:
            if isinstance(widget,QLabel):widget.setWordWrap(True)
            grid.addWidget(widget,row,0,1,2);i+=1
        row+=1
    grid.setColumnStretch(1,1);grid.setVerticalSpacing(5)


class TaskTabs(QTabWidget):
    """Only the visible result page contributes to the scrollable minimum."""
    def __init__(self):
        super().__init__()
        self.currentChanged.connect(self._sync_page_sizes)

    def tabInserted(self, index):
        super().tabInserted(index)
        self._sync_page_sizes()

    def setCurrentIndex(self, index):
        super().setCurrentIndex(index)
        # Task routing deliberately blocks signals while changing visible tabs.
        self._sync_page_sizes()

    def _sync_page_sizes(self, *_):
        for index in range(self.count()):
            page = self.widget(index)
            if not hasattr(page, '_task_tab_size_policy'):
                page._task_tab_size_policy = page.sizePolicy()
            page.setSizePolicy(page._task_tab_size_policy if index == self.currentIndex()
                               else QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored))
        stack = self.findChild(QStackedWidget)
        if stack is not None:
            stack.layout().invalidate()
            stack.updateGeometry()
        self.updateGeometry()


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
    # Let Windows font metrics and the two-line run button determine the height.
    # A fixed ribbon height clips buttons into their group captions at native DPI.
    w.ribbon.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    w.ribbon.setAccessibleName("工作台功能区")
    menu_layout.addWidget(w.ribbon)
    w.setMenuWidget(menu_host)

    pages = []
    for name in ('工作', '图形与布局'):
        page=QWidget();row=QHBoxLayout(page);row.setContentsMargins(4,2,4,1);row.setSpacing(0)
        w.ribbon.addTab(page,name);pages.append(row)
    w.home_button=ribbon_button('开始','model',lambda:w.activate_task('home'))
    w.compute_button=ribbon_button('运行计算\nF5','run',w.compute)
    w.compute_button.setObjectName('compute');w.compute_button.setAccessibleName('运行当前任务')
    w.load_scheme_button=ribbon_button('打开参数','open',w.open_configuration_dialog)
    w.save_scheme_button=ribbon_button('保存参数','save',w.save_configuration_dialog)
    w.load_scheme_button.setToolTip('打开当前任务的参数文件；不恢复计算结果')
    w.save_scheme_button.setToolTip('只保存当前输入，下次打开需重新计算')
    w.history_open_button=ribbon_button('打开记录','open',w.load_history_dialog)
    w.history_save_button=ribbon_button('保存记录','save',w.save_history_dialog)
    w.history_open_button.setToolTip('恢复直线阵/馈电计算记录，或打开馈电导出的feed-network.json')
    w.history_save_button.setToolTip('保存已有计算记录、标签及选中项；不含未计算的输入')
    w.save_button=ribbon_button('导出当前结果','save',w.save_dialog)
    w.scan_button=ribbon_button('多角度评估','array',w.run_scan)
    w.cancel_button=ribbon_button('取消计算','reset',w.cancel_current)
    ribbon_group(pages[0],'任务',[w.home_button])
    ribbon_group(pages[0],'1. 参数',[w.load_scheme_button,w.save_scheme_button])
    ribbon_group(pages[0],'2. 计算',[w.compute_button,w.cancel_button,w.scan_button])
    ribbon_group(pages[0],'3. 保存与继续',[w.history_save_button,w.history_open_button,w.save_button])
    w.result_button=ribbon_button('方向图','plot',lambda:w.show_view(0))
    w.model_button=ribbon_button('模型视图','model',lambda:w.show_view(3))
    w.rect_button=ribbon_button('直角坐标','plot',lambda:w.set_plot_view(0))
    w.polar_button=ribbon_button('极坐标','polar',lambda:w.set_plot_view(1))
    w.reset_view_button=ribbon_button('重置图形','reset',w.reset_plot_view)
    w.image_button=ribbon_button('保存当前图','plot',w.save_plot)
    w.copy_button=ribbon_button('复制当前图','copy',w.copy_plot)
    w.restore_layout_button=ribbon_button('恢复布局','layout',w.restore_workbench)
    w.example_button=ribbon_button('直线阵例题','open',w.load_example)
    ribbon_group(pages[1],'直线阵视图',[w.result_button,w.model_button,w.rect_button,w.polar_button])
    ribbon_group(pages[1],'当前图形',[w.reset_view_button,w.image_button,w.copy_button])
    ribbon_group(pages[1],'布局与帮助',[ribbon_button('显示参数','parameters',w.focus_parameters),w.restore_layout_button,w.example_button,ribbon_button('本页帮助','help',w.show_help)])
    for row in pages:row.addStretch()

    central = QWidget()
    central.setObjectName("central")
    w.setCentralWidget(central)
    layout = QVBoxLayout(central)
    layout.setContentsMargins(2, 2, 2, 0)
    layout.setSpacing(4)
    w.task_header=QWidget();task_layout=QVBoxLayout(w.task_header);task_layout.setContentsMargins(10,5,10,3);task_layout.setSpacing(3)
    w.task_title=QLabel();w.task_title.setStyleSheet('font-size:18px;font-weight:600;color:#185986;')
    w.task_hint=QLabel();w.task_hint.setWordWrap(True);w.task_hint.setObjectName('scope')
    task_layout.addWidget(w.task_title);task_layout.addWidget(w.task_hint);layout.addWidget(w.task_header)
    controls = QHBoxLayout()
    controls.addWidget(QLabel("结果"))
    w.selection = QComboBox()
    w.selection.setAccessibleName("所选结果曲线")
    w.selection.setMinimumWidth(130)
    w.selection.currentIndexChanged.connect(w.update_selection)
    controls.addWidget(w.selection, 1)
    w.keep_history = QCheckBox("保留并比较（最多5条）")
    controls.addWidget(w.keep_history)
    w.remove_button = QPushButton("移除")
    w.remove_button.clicked.connect(w.remove_selected)
    controls.addWidget(w.remove_button)
    w.result_bar = QWidget(); w.result_bar.setLayout(controls); layout.addWidget(w.result_bar)
    w.tabs = TaskTabs()
    w.tabs.setObjectName("documents")
    w.tabs.setAccessibleName("结果视图")
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
    from .feed_widgets import FeedPanel
    w.feed_panel = FeedPanel(w)
    w.tabs.addTab(w.feed_panel, workbench_icon("parameters"), "馈电电气设计")
    from .planar_widgets import PlanarPanel
    w.planar_panel = PlanarPanel(w)
    w.tabs.addTab(w.planar_panel, workbench_icon("array"), "平面阵电气")
    from .sparse_widgets import SparsePanel
    w.sparse_panel = SparsePanel(w)
    w.sparse_panel.planar_requested.connect(w.receive_sparse_layout)
    w.tabs.addTab(w.sparse_panel, workbench_icon("array"), "二维稀疏")
    from .start_page import build_start_page
    w.start_page=build_start_page(w);w.tabs.addTab(w.start_page,'开始')
    w.feed_button=w.task_cards['feed'];w.planar_button=w.task_cards['planar'];w.sparse_button=w.task_cards['sparse']
    # Result controls and plots keep their readable minimum sizes when the
    # work area is short or a dock is widened. The viewport scrolls instead.
    w.results_scroll=QScrollArea();w.results_scroll.setWidgetResizable(True)
    w.results_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    w.results_scroll.setWidget(w.tabs);layout.addWidget(w.results_scroll, 1)

    w.tree=QTreeWidget();w.tree.setHeaderHidden(True);w.tree.setAccessibleName('工作任务导航');w.tree.setIndentation(13)
    root=QTreeWidgetItem(w.tree,['工作任务']);w.project_tree_root=root
    def node(parent,text,kind,data):
        item=QTreeWidgetItem(parent,[text]);item.setIcon(0,workbench_icon(kind));item.setData(0,Qt.ItemDataRole.UserRole,data);return item
    w.home_tree_item=node(root,'开始与操作流程','open',('view',7))
    w.linear_tree_item=node(root,'直线阵分析','array',('view',0))
    w.plot_tree_item=node(w.linear_tree_item,'方向图','plot',('view',0))
    w.data_tree_item=node(w.linear_tree_item,'方向图数据','table',('view',1))
    node(w.linear_tree_item,'阵元激励','table',('view',2))
    w.model_tree_item=node(w.linear_tree_item,'阵列模型','model',('view',3))
    w.history_tree_item=node(w.linear_tree_item,'当前计算记录','open',None)
    w.feed_tree_item=node(root,'直线阵馈电','parameters',('view',4))
    w.planar_tree_item=node(root,'平面阵馈电','array',('view',5))
    w.sparse_tree_item=node(root,'二维稀疏设计','array',('view',6))
    root.setExpanded(True);w.tree.currentItemChanged.connect(w.select_result_tab)
    w.nav_dock=dock(w,'选择任务','navigationDock',Qt.DockWidgetArea.LeftDockWidgetArea,w.tree)
    w.nav_dock.setMinimumWidth(175)

    inspector = QWidget()
    iv = QVBoxLayout(inspector)
    iv.setContentsMargins(7, 4, 7, 7)
    iv.setSpacing(7)
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
    w.feed_parameter_host=QWidget();feed_layout=QVBoxLayout(w.feed_parameter_host);feed_layout.setContentsMargins(0,0,0,0)
    w.feed_panel.parameters_in_sidebar=True
    w.feed_panel.layout().removeWidget(w.feed_panel.beam_row);w.feed_panel.layout().removeWidget(w.feed_panel.settings_tabs)
    w.feed_panel.beam_row.layout().removeWidget(w.feed_panel.beam_mode_hint)
    feed_layout.addWidget(QLabel('2. 馈电与目标指向'));feed_layout.addWidget(w.feed_panel.beam_row)
    feed_layout.addWidget(w.feed_panel.beam_mode_hint);feed_layout.addWidget(w.feed_panel.settings_tabs)
    w.feed_panel.settings_tabs.setMaximumHeight(16777215)
    w.feed_panel.settings_tabs.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    for i,title in enumerate(('馈电网络','幅相加权','误差失效')):w.feed_panel.settings_tabs.setTabText(i,title)
    iv.addWidget(w.feed_parameter_host)
    iv.addStretch()
    scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(inspector)
    w.inspector_stack = QStackedWidget(); w.inspector_stack.addWidget(scroll); w.inspector_stack.addWidget(w.sparse_panel.parameter_widget)
    planar_parameters=QWidget();pv=QVBoxLayout(planar_parameters);pv.setContentsMargins(7,4,7,7)
    label=QLabel('平面阵参数独立设置；可先选布局，再设置目标角与馈电。');label.setWordWrap(True);pv.addWidget(label)
    w.planar_panel.parameters_in_sidebar=True
    w.planar_panel.layout().removeWidget(w.planar_panel.beam_row);w.planar_panel.layout().removeWidget(w.planar_panel.settings_tabs)
    w.planar_panel.beam_row.layout().removeWidget(w.planar_panel.beam_mode_hint)
    pv.addWidget(w.planar_panel.beam_row);pv.addWidget(w.planar_panel.beam_mode_hint);pv.addWidget(w.planar_panel.settings_tabs);pv.addStretch()
    w.planar_panel.settings_tabs.setMaximumHeight(16777215)
    w.planar_panel.settings_tabs.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    for i,title in enumerate(('布局','切面','阻抗','幅相','误差')):w.planar_panel.settings_tabs.setTabText(i,title)
    planar_scroll=QScrollArea();planar_scroll.setWidgetResizable(True);planar_scroll.setWidget(planar_parameters)
    w.inspector_stack.addWidget(planar_scroll)
    for panel in (w.feed_panel,w.planar_panel):
        for i in range(panel.settings_tabs.count()):sidebar_grid(panel.settings_tabs.widget(i))
        for widget in (panel.intro,panel.pointing_button,panel.design_button,panel.open_button,panel.save_button,panel.export_button,panel.toggle_settings,panel.study_button):widget.hide()
    w.feed_panel.state.setText('设置右侧参数后，点击顶部“生成馈电设计”。')
    w.planar_panel.state.setText('设置右侧参数后，点击顶部“生成平面阵设计”。')
    for widget in (w.sparse_panel.run_button,w.sparse_panel.cancel_button,w.sparse_panel.export_button,w.sparse_panel.load_settings_button,w.sparse_panel.save_settings_button):widget.hide()
    w.sparse_panel.planar_button.setText('下一步：设计馈电')
    w.inspector_dock = dock(w, "当前任务参数", "inspectorDock", Qt.DockWidgetArea.RightDockWidgetArea, w.inspector_stack)
    w.inspector_dock.setMinimumWidth(340)

    messages = QWidget(); mv = QVBoxLayout(messages)
    mv.setContentsMargins(0, 0, 0, 0); mv.setSpacing(0)
    w.state_label = QLabel("准备就绪"); w.state_label.setObjectName("state"); w.state_label.setWordWrap(True)
    w.task_header.layout().addWidget(w.state_label)
    w.message_log = QPlainTextEdit()
    w.message_log.setReadOnly(True); w.message_log.setMaximumBlockCount(300)
    w.message_log.setAccessibleName("计算消息日志")
    mv.addWidget(w.message_log)
    w.messages_dock = dock(w, "消息", "messagesDock", Qt.DockWidgetArea.BottomDockWidgetArea, messages)
    w.messages_dock.setMinimumHeight(95)
    w.statusBar().messageChanged.connect(w.record_message)
    w.context_caption = QLabel("均匀直线阵  |  本地计算"); w.statusBar().addPermanentWidget(w.context_caption)
    from .parallel import set_workers, requested_workers
    w.worker_choice = QComboBox(); w.worker_choice.setAccessibleName("计算并行数")
    w.worker_choice.addItem("自动（最多4路）",0)
    for count in range(1,9): w.worker_choice.addItem("单核" if count==1 else f"{count}路并行",count)
    w.worker_choice.setCurrentIndex(w.worker_choice.findData(requested_workers()))
    w.worker_choice.setToolTip("用于之后启动的扫描、平面阵方向图和稀疏复核；自动模式小任务串行。运行设置不写入阵列方案。")
    w.worker_choice.currentIndexChanged.connect(lambda:set_workers(w.worker_choice.currentData()))
    w.statusBar().addPermanentWidget(w.worker_choice)
    w.statusBar().showMessage("就绪。可设置参数，或载入说明书例题。")
    w.resizeDocks([w.nav_dock, w.inspector_dock], [190, 360], Qt.Orientation.Horizontal)
    w.resizeDocks([w.messages_dock], [135], Qt.Orientation.Vertical)
    w.tabs.currentChanged.connect(w.change_workbench_context)
    w._workflow_ready=True
