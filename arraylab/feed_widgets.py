"""Desktop panel for impedance synthesis; computation lives in feed.py."""
from datetime import datetime
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import matplotlib

from PySide6.QtCore import QByteArray, Qt, QSignalBlocker
from PySide6.QtGui import QColor, QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtSvgWidgets import QGraphicsSvgItem
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QComboBox, QLineEdit,
    QPushButton, QTabWidget, QTableWidget, QTableWidgetItem, QAbstractItemView,
    QHeaderView, QFileDialog, QMessageBox, QGraphicsScene, QGraphicsView, QCheckBox, QInputDialog,
)
from .feed import FeedSettings, design_feed, settings_payload, save_feed_configuration, load_feed_configuration, export_feed_directory, write_text_atomic, synthesize_impedance
from .feed_diagram import diagram_svg, section_lengths, port_lengths
from .electrical_widgets import ElectricalControls
from .widgets import PatternPlot, PatternSet


class NetworkView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(QColor("#f7fafc"))
        self.renderer = None
        self.focus_point = (400, 180)

    def set_report(self, report):
        self.scene().clear()
        if self.renderer is not None:
            self.renderer.deleteLater()
        self.renderer = QSvgRenderer(QByteArray(diagram_svg(report).encode("utf-8")), self)
        item = QGraphicsSvgItem()
        item.setSharedRenderer(self.renderer)
        self.scene().addItem(item)
        self.scene().setSceneRect(item.boundingRect())
        n = report["metrics"]["ports"]
        self.focus_point = (430, 180 if report["settings"]["topology"] == "series" else 120 + (n - 1) * 70)
        if report["settings"]["topology"] == "symmetric_series":
            self.focus_point = (1400, 250)
        self.readable_view()

    def readable_view(self):
        self.resetTransform()
        self.centerOn(*self.focus_point)

    def fit(self):
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        scale = self.transform().m11() * factor
        if .015 <= scale <= 4:
            self.scale(factor, factor)
        event.accept()


class FeedPanel(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.report = None
        self._changing = False
        self.last_export = None
        self.history = []
        self.sequence = 0
        self._compact_mode = None
        self.branch_delays_deg = ()
        self.branch_delay_elements = 0
        self.branch_delay_topology = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        intro = QLabel("按目标幅度分配功率 · 纯电阻负载 · 匹配相位补偿。阵列参数取右侧当前输入。")
        intro.setWordWrap(True)
        self.intro = intro
        layout.addWidget(intro)
        self.settings_tabs = QTabWidget()
        self.settings_tabs.setMaximumHeight(152)
        impedance_page = QWidget()
        grid = QGridLayout(impedance_page); grid.setContentsMargins(6, 3, 6, 3)
        self.topology = QComboBox()
        self.topology.addItem("并联馈电（树形）", "corporate")
        self.topology.addItem("串行馈电（逐级抽头）", "series")
        self.topology.addItem("1分2＋对称串馈", "symmetric_series")
        self.topology.setItemData(2, "偶数阵元2–200；先等分到左右两臂，再从中心向外逐级抽头。", Qt.ItemDataRole.ToolTipRole)
        grid.addWidget(QLabel("馈电拓扑"), 0, 0); grid.addWidget(self.topology, 0, 1)
        self.fields = {}
        for index, (key, label, value) in enumerate([
            ("input_ohm", "输入阻抗 (Ω)", 50), ("junction_ohm", "节点阻抗 (Ω)", 50),
            ("load_ohm", "阵元负载 (Ω)", 50), ("min_line_ohm", "线阻抗提示下限 (Ω)", 20),
            ("max_line_ohm", "线阻抗提示上限 (Ω)", 120),
        ], 1):
            edit = QLineEdit(str(value)); edit.setAccessibleName(label)
            edit.setMinimumWidth(60)
            self.fields[key] = edit
            row, col = index // 2, (index % 2) * 2
            grid.addWidget(QLabel(label), row, col); grid.addWidget(edit, row, col + 1)
            edit.textChanged.connect(self.mark_dirty)
        self.topology.currentIndexChanged.connect(self.mark_dirty)
        self.branch_delay_button = QPushButton("支路电长度…（全部为0°）")
        self.branch_delay_button.setAccessibleName("编辑支路匹配延迟电长度")
        self.branch_delay_button.clicked.connect(self.edit_branch_delays)
        grid.addWidget(self.branch_delay_button, 3, 0, 1, 4)
        grid.setColumnStretch(1, 1); grid.setColumnStretch(3, 1)
        self.settings_tabs.addTab(impedance_page, "阻抗与拓扑")
        self.electrical = ElectricalControls(self, self.settings_tabs)
        self.beam_row = QWidget(); beam = QHBoxLayout(self.beam_row); beam.setContentsMargins(0, 0, 0, 0)
        beam.addWidget(QLabel("目标波束 θ (°)")); self.target_theta = QLineEdit("0"); self.target_theta.setMaximumWidth(90)
        self.target_theta.setAccessibleName("馈电目标波束指向角")
        self.target_theta.setToolTip("−90°至90°，从+z朝+x为正；与右侧直线阵扫描角同步。")
        beam.addWidget(self.target_theta)
        self.pointing_button = QPushButton("按目标生成")
        self.pointing_button.setToolTip("启用按目标指向补偿，生成阻抗、端口相位、电长度和输出方向图。")
        self.pointing_button.clicked.connect(lambda: self.generate(True)); beam.addWidget(self.pointing_button)
        self.beam_mode_hint = QLabel(); self.beam_mode_hint.setWordWrap(True); beam.addWidget(self.beam_mode_hint, 1)
        self.electrical.phase_mode.currentIndexChanged.connect(self.update_beam_hint)
        self.update_beam_hint(); layout.addWidget(self.beam_row)
        layout.addWidget(self.settings_tabs)
        actions = QHBoxLayout()
        self.design_button = QPushButton("生成电气设计")
        self.design_button.clicked.connect(self.generate)
        self.open_button = QPushButton("打开设置")
        self.open_button.clicked.connect(self.open_dialog)
        self.save_button = QPushButton("保存设置")
        self.save_button.clicked.connect(self.save_dialog)
        self.export_button = QPushButton("导出结果")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_dialog)
        for item in (self.design_button, self.open_button, self.save_button, self.export_button):
            actions.addWidget(item)
        self.toggle_settings = QPushButton("收起设置")
        self.toggle_settings.clicked.connect(self.toggle_controls)
        actions.addWidget(self.toggle_settings)
        self.study_button = QPushButton("扫描评估")
        self.study_button.setToolTip("以所选已生成设计为基础，逐扫描角计算电气响应")
        self.study_button.clicked.connect(self.open_study)
        actions.addWidget(self.study_button)
        actions.addStretch()
        layout.addLayout(actions)
        self.state = QLabel("请设置参数后点击“生成电气设计”。50 Ω为初值，请按设计条件修改。")
        self.state.setWordWrap(True); self.state.setObjectName("state")
        self.state.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.state)
        history_row = QHBoxLayout()
        history_row.addWidget(QLabel("设计记录"))
        self.history_selection = QComboBox(); self.history_selection.setMinimumWidth(100)
        self.history_selection.setAccessibleName("选择并恢复馈电设计记录")
        self.history_selection.setPlaceholderText("暂无设计记录，请先生成或打开记录")
        self.history_selection.setToolTip("选择记录会恢复保存的阵列参数、馈电设置及结果，不重新计算。未运行的输入修改请先用“保存参数”保存。")
        self.history_selection.setEnabled(False)
        self.history_selection.currentIndexChanged.connect(self.select_history)
        self.history_selection.activated.connect(self.activate_history)
        history_row.addWidget(self.history_selection, 1)
        rename = QPushButton("改名"); rename.clicked.connect(self.rename_history); history_row.addWidget(rename)
        self.compare_history = QCheckBox("比较历史输出"); self.compare_history.toggled.connect(self.redraw_patterns)
        history_row.addWidget(self.compare_history)
        layout.addLayout(history_row)
        self.result_tabs = QTabWidget()
        self.sections = QTableWidget()
        self.ports = QTableWidget()
        for table in (self.sections, self.ports):
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            table.setAlternatingRowColors(True)
            table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.result_tabs.addTab(self.sections, "支路参数")
        diagram_page = QWidget(); dl = QVBoxLayout(diagram_page); dl.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        hint = QLabel("滚轮缩放，拖动平移；点击适配查看全图。连接图不按尺寸绘制。")
        hint.setWordWrap(True); controls.addWidget(hint, 1)
        self.diagram = NetworkView()
        fit = QPushButton("适配"); fit.clicked.connect(self.diagram.fit); controls.addWidget(fit)
        actual = QPushButton("100%"); actual.clicked.connect(self.diagram.readable_view); controls.addWidget(actual)
        dl.addLayout(controls); dl.addWidget(self.diagram)
        self.result_tabs.addTab(diagram_page, "连接示意")
        self.result_tabs.addTab(self.ports, "端口与相位")
        pattern_page = QWidget(); pattern_layout = QVBoxLayout(pattern_page); pattern_layout.setContentsMargins(0, 0, 0, 0)
        view_row = QHBoxLayout()
        self.pattern_view = QComboBox(); self.pattern_view.addItems(["直角坐标", "极坐标", "三维方向图"])
        self.pattern_source = QComboBox()
        for label, key in [("目标激励", "target"), ("网络原始输出（未补偿）", "uncompensated"), ("电气模型输出（最终）", "output")]:
            self.pattern_source.addItem(label, key)
        self.pattern_source.setCurrentIndex(2)
        view_row.addWidget(QLabel("显示方式")); view_row.addWidget(self.pattern_view)
        view_row.addWidget(QLabel("指标／三维来源")); view_row.addWidget(self.pattern_source, 1)
        pattern_layout.addLayout(view_row)
        self.pattern_view.currentIndexChanged.connect(self.change_pattern_view)
        self.pattern_source.currentIndexChanged.connect(self.redraw_patterns)
        self.pattern_metrics = QLabel(); self.pattern_metrics.setWordWrap(True)
        self.pattern_metrics.hide()  # Summary now lives inside the exported figure.
        self.pattern_plot = PatternPlot(); pattern_layout.addWidget(self.pattern_plot, 1)
        with matplotlib.rc_context({"figure.autolayout": False, "figure.constrained_layout.use": False}):
            self.pattern_plot.figure.set_layout_engine(None)
        self.pattern_plot.figure.subplots_adjust(left=.09, right=.98, bottom=.22, top=.95)
        self.pattern_hint = QLabel("共同0 dB参考：当前目标激励的相干幅度和；未单独抬高各曲线峰值。")
        self.pattern_hint.setWordWrap(True); pattern_layout.addWidget(self.pattern_hint)
        self.result_tabs.addTab(pattern_page, "方向图核对")
        self.result_tabs.currentChanged.connect(self.result_tab_changed)
        layout.addWidget(self.result_tabs, 1)
        self.notice = QLabel("理想电气模型；物理尺寸暂缓。误差为单次实现，失效按匹配吸收、辐射置零。")
        self.notice.setWordWrap(True); self.notice.setObjectName("notice")
        layout.addWidget(self.notice)

    def connect_shared_steering(self):
        self._syncing_beam = False
        self.target_theta.textChanged.connect(self.edit_target_theta)
        self.window.inputs["scan_deg"].textChanged.connect(self.sync_target_theta)
        self.window.phase_mode.currentIndexChanged.connect(self.sync_target_theta)
        self.window.inputs["elements"].textChanged.connect(self.mark_dirty)
        self.sync_target_theta()

    def edit_target_theta(self, text):
        if self._syncing_beam: return
        self._syncing_beam = True
        try:
            self.window.phase_mode.setCurrentIndex(1)
            self.window.inputs["scan_deg"].setText(text)
        finally: self._syncing_beam = False
        self.mark_dirty()

    def sync_target_theta(self, *_):
        if self._syncing_beam: return
        text = self.window.inputs["scan_deg"].text() if self.window.phase_mode.currentIndex() == 1 else "0"
        with QSignalBlocker(self.target_theta): self.target_theta.setText(text)

    def update_beam_hint(self):
        mode = self.electrical.phase_mode.currentData()
        self.beam_mode_hint.setText({"scan": "θ从+z朝+x；按目标指向补偿。", "equalize": "当前为正侧向0°；在“幅相加权”中切换指向补偿。",
                                     "none": "当前不补偿；可在“幅相加权”中启用指向补偿。"}[mode])

    def read_settings(self, phase_mode=None, include_branch_delays=True):
        values = {key: float(edit.text().strip()) for key, edit in self.fields.items()}
        electrical = self.electrical.values()
        if phase_mode is not None: electrical["phase_mode"] = phase_mode
        branch = dict(branch_delays_deg=self.branch_delays_deg,
                      branch_delay_elements=self.branch_delay_elements,
                      branch_delay_topology=self.branch_delay_topology) if include_branch_delays else {}
        settings = FeedSettings(topology=self.topology.currentData(), **values, **electrical, **branch)
        settings.validate()
        return settings

    def import_linear_result(self, data):
        """Load a selected analysis snapshot as inputs, preserving old results."""
        from .linear_feed_import import prepare_linear_feed_import
        # Ignore stale/invalid electrical controls which this handoff resets.
        # Validate the retained impedances before changing any live widget.
        impedances = {}
        for key, edit in self.fields.items():
            try:
                impedances[key] = float(edit.text().strip())
            except ValueError as exc:
                raise ValueError(f"当前馈电设置“{edit.accessibleName()}”无效，请修正后导入。") from exc
        network = FeedSettings(topology=self.topology.currentData(), **impedances)
        plan = prepare_linear_feed_import(data, network)
        changing = self._changing
        self._changing = True
        try:
            self.window.apply_parameters(plan.parameters)
            self.topology.setCurrentIndex(self.topology.findData(plan.settings.topology))
            for key, edit in self.fields.items():
                edit.setText(str(getattr(plan.settings, key)))
            self.electrical.apply(plan.settings)
            self.apply_branch_delays(plan.settings)
        finally:
            self._changing = changing
        self.linear_import_source = plan
        self._linear_import_select_output = True
        p = plan.parameters
        label = " ".join(plan.source_label.split())[:100]
        unit = "λ" if p.spacing_unit == "lambda" else "mm"
        message = (f"已导入直线阵记录：{label}；N={p.elements}，d={p.spacing:g}{unit}，θ={p.scan_deg:g}°。"
                   "已设为均匀幅度、按目标指向连续补偿，并清除旧幅相扰动和支路附加延迟。"
                   + " ".join(plan.notes)
                   + " 请按F5生成馈电设计；原阵列对应目标激励，最终指向看电气模型输出。已有表格和图形仍是原设计结果。")
        self.state.setText(message)
        self.window.statusBar().showMessage("已导入所选直线阵记录；请按F5生成馈电设计。")
        self.window.record_message(message)
        self.window.refresh_actions()
        return plan

    def apply_branch_delays(self, settings):
        self.branch_delays_deg = tuple(settings.branch_delays_deg)
        self.branch_delay_elements = settings.branch_delay_elements
        self.branch_delay_topology = settings.branch_delay_topology
        self.update_branch_hint()

    def branch_binding_matches(self):
        if not self.branch_delays_deg:
            return True
        try:
            return (int(self.window.inputs["elements"].text()) == self.branch_delay_elements
                    and self.topology.currentData() == self.branch_delay_topology)
        except (ValueError, AttributeError):
            return False

    def update_branch_hint(self):
        if not hasattr(self, "branch_delay_button"):
            return
        count = sum(value != 0 for value in self.branch_delays_deg)
        suffix = "需重新设置" if not self.branch_binding_matches() else f"{count}条非零" if count else "全部为0°"
        self.branch_delay_button.setText(f"支路电长度…（{suffix}）")
        self.branch_delay_button.setToolTip(
            "每条支路：90°变换段＋独立匹配延迟段；端口自动补偿另外计算。"
            + ("阵元数或拓扑已改变，原T编号不再适用；请打开后重设或清零。" if not self.branch_binding_matches() else "")
        )

    def edit_branch_delays(self):
        from .branch_delay_widgets import BranchDelayDialog
        from .feed import with_branch_delays
        try:
            parameters = self.window.read_parameters()
            # Generate only the topology; a stale binding must not block resetting it.
            base = self.read_settings(include_branch_delays=False)
            network = synthesize_impedance(parameters, base)
            matches = self.branch_binding_matches()
            values = self.branch_delays_deg if matches and self.branch_delays_deg else (0.0,) * len(network["sections"])
            reason = ("阵元数或拓扑与原支路设置不匹配。本次从零重新设置，不能复用原T编号；取消会保留原设置。"
                      if not matches else "")
            dialog = BranchDelayDialog(network, values, self, reason)
            if dialog.exec():
                delays = dialog.result_delays if any(dialog.result_delays) else ()
                settings = with_branch_delays(base, parameters.elements, delays)
                self.apply_branch_delays(settings)
                self.mark_dirty()
        except (ValueError, ArithmeticError) as exc:
            QMessageBox.warning(self, "支路电长度", str(exc))

    def open_study(self):
        if self.report is None:
            QMessageBox.warning(self, "扫描评估", "请先生成并选择一项电气设计。")
            return
        from .study_widgets import ScanStudyDialog
        if getattr(self,'_study_source_report',None) is not self.report:
            self.study_dialog = ScanStudyDialog(self.report, self)
            self._study_source_report = self.report
        if self.study_dialog.worker is None:self.study_dialog._close_pending=False
        self.study_dialog.exec()

    def mark_dirty(self):
        self.update_branch_hint()
        if self._changing or not hasattr(self, "state"):
            return
        self.state.setText("输入已改变，尚未重新生成。表格、方向图和导出仍对应所选设计；保存设置使用当前输入。")

    def generate(self, force_pointing=False):
        try:
            report = design_feed(self.window.read_parameters(), self.read_settings("scan" if force_pointing else None))
        except (ValueError, ArithmeticError) as exc:
            QMessageBox.warning(self, "馈电阻抗设计失败", str(exc))
            return
        if force_pointing:
            self._changing = True
            try: self.electrical.phase_mode.setCurrentIndex(self.electrical.phase_mode.findData("scan"))
            finally: self._changing = False
        if getattr(self, "_linear_import_select_output", False):
            # Switch the view only when the imported inputs have produced a new
            # result. Until then, old controls and old figures remain consistent.
            with QSignalBlocker(self.compare_history), QSignalBlocker(self.pattern_source):
                self.compare_history.setChecked(False)
                self.pattern_source.setCurrentIndex(self.pattern_source.findData("output"))
            self._linear_import_select_output = False
        self.report = report
        self.sequence += 1
        s, p = report["settings"], report["array_parameters"]
        label = f"#{self.sequence} {s['topology']} · {s['weighting']} · N={p['elements']} · θ={report['electrical']['metrics']['target_scan_deg']:g}°"
        self.history.append({"label": label, "report": report})
        self.history = self.history[-5:]
        self.history_selection.blockSignals(True)
        self.history_selection.clear(); self.history_selection.addItems([record["label"] for record in self.history])
        self.history_selection.setCurrentIndex(len(self.history) - 1)
        self.history_selection.blockSignals(False)
        self.update_history_available()
        self.display_report(report)

    def display_report(self, report):
        self.report = report
        columns = [("id", "线段"), ("parent", "起点"), ("child", "终点"),
                   ("line_ohm", "变换线 Zₜ (Ω)"),
                   ("electrical_length_deg", "变换段 (°)"), ("matched_delay_ohm", "匹配延迟 Zc (Ω)"),
                   ("matched_delay_deg", "附加延迟 (°)"), ("total_electrical_length_deg", "支路合计 (°)"),
                   ("local_power_fraction", "局部分配比"), ("branch_input_ohm", "支路输入 (Ω)"),
                   ("child_equivalent_ohm", "下游等效 (Ω)"),
                   ("outside_range", "超提示范围")]
        self.fill_table(self.sections, columns, [section_lengths(line) for line in report["sections"]])
        port_columns = [("port", "端口"), ("load_ohm", "负载 (Ω)"),
                        ("power_fraction", "总功率占比"), ("quarter_wave_sections", "λ/4段数"),
                        ("transform_path_deg", "变换路径 (°)"), ("branch_delay_path_deg", "支路延迟累计 (°)"),
                        ("network_electrical_length_deg", "网络路径合计 (°)"),
                        ("network_phase_deg", "网络相位 (°)"), ("target_phase_deg", "目标相位 (°)"),
                        ("applied_delay_deg", "端口补偿 (°)"), ("total_electrical_length_deg", "完整路径 (°)"),
                        ("output_phase_deg", "输出相位 (°)"),
                        ("output_amplitude", "输出幅度"), ("failed", "失效")]
        if report["settings"]["topology"] == "symmetric_series":
            port_columns.extend([("arm", "支臂"), ("mirror_port", "镜像端口")])
        self.fill_table(self.ports, port_columns, port_lengths(report))
        self.diagram.set_report(report)
        m, p = report["metrics"], report["array_parameters"]
        self.state.setText(f"所选设计：{report['settings']['topology']} · {m['ports']}端口 · {p['frequency_ghz']:g} GHz · "
                           f"Zₜ {m['line_min_ohm']:.3f}–{m['line_max_ohm']:.3f} Ω · 理想输入 {m['input_equivalent_ohm']:g} Ω")
        self.notice.setText(f"阻抗超提示范围：{len(m['out_of_range_sections'])}条。变换段、支路匹配延迟和端口补偿分别记录；电长度均在所选设计频率处。理想无损模型，物理尺寸暂缓。")
        self.export_button.setEnabled(True)
        self.redraw_patterns()
        if not self.inputs_match_report(report):
            self.state.setText(self.state.text() + "；与当前输入不同。")
        self.window.statusBar().showMessage(self.state.text())
        self.window.refresh_actions()

    def toggle_controls(self):
        visible = not self.settings_tabs.isVisible()
        self.settings_tabs.setVisible(visible)
        self.toggle_settings.setText("收起设置" if visible else "展开设置")

    def result_tab_changed(self, index):
        if index == 1 and self.report:
            self.diagram.readable_view()
        elif index == 3 and self.settings_tabs.isVisible() and not getattr(self,'parameters_in_sidebar',False):
            self.toggle_controls()
        self.window.refresh_actions()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_plot_layout()

    def update_plot_layout(self, force=False):
        if not hasattr(self, "pattern_hint"):
            return
        compact = self.width() < 700 or self.window.height() < 820
        if not force and compact == self._compact_mode:
            return
        self._compact_mode = compact
        self.intro.setVisible(not compact and not getattr(self,'parameters_in_sidebar',False))
        self.pattern_plot.toolbar.setVisible(not compact)
        self.pattern_hint.setVisible(not compact)
        self.pattern_plot.setToolTip(self.pattern_hint.text() + " 放大窗口可显示完整图形工具栏。")
        # The shared footer owns the plot's lower margin, including PNG export.
        ax = self.pattern_plot.axes
        if ax is not None and not self.pattern_plot.three_dimensional:
            from matplotlib.ticker import MaxNLocator
            ax.tick_params(labelsize=7 if compact else 9)
            ax.xaxis.set_major_locator(MaxNLocator(5 if compact else 9))
            ax.yaxis.set_major_locator(MaxNLocator(4 if compact else 7))
            ax.xaxis.label.set_fontsize(8 if compact else 10)
            ax.yaxis.label.set_fontsize(8 if compact else 10)
            if ax.get_legend() is not None:
                for text in ax.get_legend().get_texts():
                    text.set_fontsize(7 if compact else 8)
        self.pattern_plot.canvas.draw_idle()

    def inputs_match_report(self, report):
        try:
            return (asdict(self.window.read_parameters()) == report["array_parameters"]
                    and settings_payload(self.read_settings()) == settings_payload(FeedSettings(**report["settings"])))
        except (ValueError, ArithmeticError):
            return False

    def update_history_available(self):
        self.history_selection.setEnabled(bool(self.history))
        for index, record in enumerate(self.history):
            self.history_selection.setItemData(index, record["label"], Qt.ItemDataRole.ToolTipRole)

    @staticmethod
    def prepare_restore_report(report):
        from .configuration import parameters_from_values
        from .excitation import validate_electrical
        from .feed import settings_from_payload, normalized_branch_delays
        # External files have already passed complete history validation. Keep
        # the trusted report object (and its plot caches), checking only inputs
        # before applying them rather than copying all sampled result arrays.
        parameters = parameters_from_values(report["array_parameters"])
        settings = settings_from_payload(report["settings"], report["schema_version"])
        validate_electrical(settings, parameters)
        normalized_branch_delays(settings, parameters.elements)
        return parameters, settings

    def restore_report(self, report, prepared=None):
        parameters, settings = prepared or self.prepare_restore_report(report)
        self._linear_import_select_output = False
        changing = self._changing
        self._changing = True
        try:
            self.window.apply_parameters(parameters)
            self.topology.setCurrentIndex(self.topology.findData(settings.topology))
            for key, edit in self.fields.items():
                edit.setText(str(getattr(settings, key)))
            self.electrical.apply(settings)
            self.apply_branch_delays(settings)
            self.display_report(report)
            self.state.setText(self.state.text() + "；已恢复记录参数和结果，未重新计算。")
            self.window.statusBar().showMessage(self.state.text())
        finally:
            self._changing = changing

    def select_history(self, index):
        if not 0 <= index < len(self.history):
            return
        previous = next((i for i, record in enumerate(self.history) if record["report"] is self.report), -1)
        try:
            self.restore_report(self.history[index]["report"])
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            with QSignalBlocker(self.history_selection):
                self.history_selection.setCurrentIndex(previous)
            QMessageBox.warning(self, "设计记录恢复失败", str(exc))

    def activate_history(self, index):
        if not 0 <= index < len(self.history) or index != self.history_selection.currentIndex():
            return
        report = self.history[index]["report"]
        # A changed index was just restored by currentIndexChanged. Activating
        # the same item after editing inputs must also restore that saved design.
        if self.report is not report or not self.inputs_match_report(report):
            self.select_history(index)

    def rename_history(self):
        index = self.history_selection.currentIndex()
        if not 0 <= index < len(self.history):
            return
        label, accepted = QInputDialog.getText(self, "设计标签", "标签（最多80字）", text=self.history[index]["label"])
        if accepted and label.strip():
            self.history[index]["label"] = label.strip()[:80]
            self.history_selection.setItemText(index, self.history[index]["label"])
            self.redraw_patterns()

    def change_pattern_view(self, index):
        self.pattern_polar = index == 1
        self.redraw_patterns()

    def redraw_patterns(self):
        if self.report is None or not hasattr(self, "pattern_plot"):
            return
        patterns = self.report["electrical"]["patterns"]
        if self.compare_history.isChecked():
            data = [PatternSet(record["label"], np.array(record["report"]["electrical"]["patterns"]["output"]["angles_deg"]),
                               np.array(record["report"]["electrical"]["patterns"]["output"]["pattern_db"]),
                               feed_report=record["report"]) for record in self.history]
            selected = self.history_selection.currentIndex()
            self.pattern_hint.setText("历史比较：每条曲线使用其目标相干和作参考；配置与归一化定义随记录一同导出。")
        else:
            data = [PatternSet(label, np.array(patterns[key]["angles_deg"]), np.array(patterns[key]["pattern_db"]),
                               feed_report=self.report, pattern_key=key)
                    for key, label in [("target", "目标激励"), ("uncompensated", "网络原始输出（未补偿）"), ("output", "电气模型输出（最终）")]]
            selected = self.pattern_source.currentIndex()
            self.pattern_hint.setText("目标激励对应待实现阵列；最终指向看电气模型输出，网络原始输出仅为未补偿对照。各曲线共用目标相干幅度和作0 dB参考。")
        self.pattern_source.setEnabled(not self.compare_history.isChecked())
        self.pattern_plot.draw_sets(data, selected, polar=getattr(self,'pattern_polar',False), floor=-60,
                                   view="3d" if self.pattern_view.currentIndex() == 2 else None)
        target, output = patterns["target"]["metrics"], patterns["output"]["metrics"]
        def number(value):
            return "未定义" if value is None else f"{0 if abs(value)<1e-9 else value:.4g}"
        pointing = self.report["electrical"]["pointing"]
        peaks = ', '.join(f'{x:.4g}°' for x in pointing['output_cut_peak_angles_deg'])
        peak_text = f"切面峰 {peaks}" if peaks else "切面峰未定义"
        self.pattern_metrics.setText(f"有效目标 θ={number(pointing['target_theta_deg'])}°  |  {peak_text}  |  "
                                     f"单峰偏差 {number(pointing['output_cut_offset_deg'])}°\n"
                                     f"目标D {number(target['directivity_dbi'])} / 输出D {number(output['directivity_dbi'])} dBi  |  HPBW {number(output['half_power_beamwidth_deg'])}°")
        self.update_plot_layout(force=True)

    def export_history_path(self, path):
        if not self.history:
            raise ValueError("尚无设计记录。")
        write_text_atomic(path, json.dumps({"format": "arraylab.feed_comparison", "schema_version": 1,
                                           "records": self.history}, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    def export_history_dialog(self):
        dialog = QFileDialog(self, "导出设计比较记录", self.window.last_directory, "比较记录 (*.json)")
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog); dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setDefaultSuffix("json"); dialog.selectFile("feed-comparison.json")
        if dialog.exec():
            try:
                self.export_history_path(dialog.selectedFiles()[0])
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "导出失败", str(exc))

    @staticmethod
    def fill_table(table, columns, rows):
        table.clear(); table.setColumnCount(len(columns)); table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels([label for key, label in columns])
        for i, row in enumerate(rows):
            for j, (key, _) in enumerate(columns):
                value = row[key]
                text = ("是" if value else "否") if isinstance(value, bool) else f"{value:.7g}" if isinstance(value, float) else str(value)
                if value is None:
                    text = "—"
                if key == "arm":
                    text = {"left": "左", "right": "右"}[value]
                item = QTableWidgetItem(text)
                if row.get("outside_range"):
                    item.setForeground(QColor("#a45c14"))
                table.setItem(i, j, item)

    def save_path(self, path):
        save_feed_configuration(path, self.window.read_parameters(), self.read_settings())

    def load_path(self, path):
        p, settings = load_feed_configuration(path)
        self._changing = True
        try:
            self.window.apply_parameters(p)
            self.topology.setCurrentIndex(self.topology.findData(settings.topology))
            for key, edit in self.fields.items():
                edit.setText(str(getattr(settings, key)))
            self.electrical.apply(settings)
            self.apply_branch_delays(settings)
        finally:
            self._changing = False
        self.mark_dirty()
        self.state.setText("馈电设置已加载，请重新生成阻抗。已有阵列曲线和馈电结果均保留原快照。")

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开馈电设置", self.window.last_directory, "馈电设置 (*.json)",
                                            options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            try:
                self.load_path(path)
                self.window.last_directory = str(Path(path).parent)
            except (OSError, ValueError, OverflowError) as exc:
                QMessageBox.warning(self, "读取失败", str(exc))

    def save_dialog(self):
        dialog = QFileDialog(self, "保存当前馈电设置", self.window.last_directory, "馈电设置 (*.json)")
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave); dialog.setDefaultSuffix("json")
        dialog.selectFile("feed-config.json")
        if dialog.exec():
            try:
                self.save_path(dialog.selectedFiles()[0])
                self.window.last_directory = str(Path(dialog.selectedFiles()[0]).parent)
            except (OSError, ValueError, OverflowError) as exc:
                QMessageBox.warning(self, "保存失败", str(exc))

    def export_path(self, path):
        if self.report is None:
            raise ValueError("请先生成电气设计。")
        export_feed_directory(path, self.report)
        self.last_export = Path(path)

    def export_dialog(self):
        parent = QFileDialog.getExistingDirectory(self, "选择导出位置（将在其下新建结果文件夹）", self.window.last_directory,
                                                  options=QFileDialog.Option.DontUseNativeDialog)
        if parent:
            target = Path(parent) / ("feed-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
            try:
                self.export_path(target)
                self.window.last_directory = parent
                self.window.statusBar().showMessage(f"已导出所显示的馈电设计：{target}")
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "导出失败", str(exc))
