"""Desktop panel for impedance synthesis; computation lives in feed.py."""
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtSvgWidgets import QGraphicsSvgItem
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QComboBox, QLineEdit,
    QPushButton, QTabWidget, QTableWidget, QTableWidgetItem, QAbstractItemView,
    QHeaderView, QFileDialog, QMessageBox, QGraphicsScene, QGraphicsView,
)
from .feed import FeedSettings, synthesize_impedance, save_feed_configuration, load_feed_configuration, export_feed_directory
from .feed_diagram import diagram_svg


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
        self.focus_point = (400, 180 if report["settings"]["topology"] == "series" else 100 + (n - 1) * 45)
        if report["settings"]["topology"] == "symmetric_series":
            self.focus_point = (730, 180)
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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        intro = QLabel("等功率分配 · 纯电阻负载 · 单频点 λ/4 变换线。阵元数与频率取右侧当前输入。")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        grid = QGridLayout()
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
        grid.setColumnStretch(1, 1); grid.setColumnStretch(3, 1)
        layout.addLayout(grid)
        actions = QHBoxLayout()
        self.design_button = QPushButton("生成阻抗")
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
        actions.addStretch()
        layout.addLayout(actions)
        self.state = QLabel("请设置阻抗后点击“生成阻抗”。50 Ω为初值，请按设计条件修改。")
        self.state.setWordWrap(True); self.state.setObjectName("state")
        layout.addWidget(self.state)
        self.result_tabs = QTabWidget()
        self.sections = QTableWidget()
        self.ports = QTableWidget()
        for table in (self.sections, self.ports):
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            table.setAlternatingRowColors(True)
            table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.result_tabs.addTab(self.sections, "支路阻抗")
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
        self.result_tabs.currentChanged.connect(lambda index: self.diagram.readable_view() if index == 1 and self.report else None)
        layout.addWidget(self.result_tabs, 1)
        self.notice = QLabel("仅进行理想阻抗综合；相位均衡、扫描馈相、线宽、基板与尺寸在后续设计。")
        self.notice.setWordWrap(True); self.notice.setObjectName("notice")
        layout.addWidget(self.notice)

    def read_settings(self):
        values = {key: float(edit.text().strip()) for key, edit in self.fields.items()}
        settings = FeedSettings(topology=self.topology.currentData(), **values)
        settings.validate()
        return settings

    def mark_dirty(self):
        if self._changing or not hasattr(self, "state"):
            return
        self.state.setText("输入已改变，尚未重新生成阻抗。表格和导出仍对应上一次设计；保存设置使用当前输入。")

    def generate(self):
        try:
            report = synthesize_impedance(self.window.read_parameters(), self.read_settings())
        except (ValueError, ArithmeticError) as exc:
            QMessageBox.warning(self, "馈电阻抗设计失败", str(exc))
            return
        self.report = report
        columns = [("id", "线段"), ("parent", "起点"), ("child", "终点"),
                   ("local_power_fraction", "局部分配比"), ("branch_input_ohm", "支路输入 (Ω)"),
                   ("child_equivalent_ohm", "下游等效 (Ω)"), ("line_ohm", "变换线 Zₜ (Ω)"),
                   ("electrical_length_deg", "电长度 (°)"), ("outside_range", "超提示范围")]
        self.fill_table(self.sections, columns, report["sections"])
        port_columns = [("port", "端口"), ("load_ohm", "负载 (Ω)"),
                        ("power_fraction", "总功率占比"), ("quarter_wave_sections", "λ/4段数"),
                        ("network_phase_deg", "未补偿相位 (°)")]
        if report["settings"]["topology"] == "symmetric_series":
            port_columns.extend([("arm", "支臂"), ("mirror_port", "镜像端口")])
        self.fill_table(self.ports, port_columns, report["ports"])
        self.diagram.set_report(report)
        m, p = report["metrics"], report["array_parameters"]
        self.state.setText(f"已生成：{self.topology.currentText()} · {m['ports']}端口 · {p['frequency_ghz']:g} GHz · "
                           f"Zₜ {m['line_min_ohm']:.3f}–{m['line_max_ohm']:.3f} Ω · 理想输入 {m['input_equivalent_ohm']:g} Ω")
        unequal = not m["equal_port_phase_at_design_frequency"]
        self.notice.setText(f"超提示范围：{len(m['out_of_range_sections'])}条。" +
                            ("端口相位不等，需后续均衡。" if unequal else "中心频点端口等相，扫描馈相未设计。") +
                            "仅理想T型网络，无输出隔离保证；尚未设计线宽、基板与物理尺寸。")
        self.export_button.setEnabled(True)
        self.window.statusBar().showMessage(self.state.text())

    @staticmethod
    def fill_table(table, columns, rows):
        table.clear(); table.setColumnCount(len(columns)); table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels([label for key, label in columns])
        for i, row in enumerate(rows):
            for j, (key, _) in enumerate(columns):
                value = row[key]
                text = ("是" if value else "否") if isinstance(value, bool) else f"{value:.7g}" if isinstance(value, float) else str(value)
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
            raise ValueError("请先生成阻抗设计。")
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
