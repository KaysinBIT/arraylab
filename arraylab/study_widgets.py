"""Desktop scan studies with immutable inputs, progress and cooperative cancellation."""
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QComboBox,
                               QSpinBox, QLineEdit, QPushButton, QProgressBar, QTableWidget,
                               QTableWidgetItem, QTabWidget, QWidget, QFileDialog, QMessageBox)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from .configuration import parameters_from_values
from .feed import FeedSettings
from .planar import PlanarParameters
from .study import (StudySettings, run_study, validate_study, StudyCancelled,
                    load_study_configuration, save_study_configuration, export_study_directory)
from .study_plot import draw_study
from .parallel import requested_workers


class ScanWorker(QThread):
    progress = Signal(int, int)

    def __init__(self, parameters, settings, study, parent=None):
        super().__init__(parent)
        self.parameters, self.settings, self.study = parameters, settings, study
        self.workers = requested_workers()
        self.result = None; self.error = None; self.cancelled = False

    def run(self):
        try:
            self.result = run_study(self.parameters, self.settings, self.study,
                                    progress=self.progress.emit, cancelled=self.isInterruptionRequested, workers=self.workers)
        except StudyCancelled:
            self.cancelled = True
        except Exception as exc:
            self.error = str(exc)


class ScanStudyDialog(QDialog):
    def __init__(self, source_report, parent=None):
        super().__init__(parent)
        self.setWindowTitle("多扫描角电气评估")
        self.resize(1000, 720)
        self.report = None; self.worker = None; self._close_pending = False; self._cancel_pending = False
        if "geometry" in source_report:
            self.parameters = PlanarParameters(**source_report["geometry"])
        else:
            self.parameters = parameters_from_values(source_report["array_parameters"])
        self.feed_settings = FeedSettings(**source_report["settings"])
        layout = QVBoxLayout(self)
        self.source_label = QLabel(); self.source_label.setWordWrap(True); layout.addWidget(self.source_label)
        self.controls = QWidget(); form = QFormLayout(self.controls)
        self.axis = QComboBox(); self.axis.addItem("θ", "theta"); self.axis.addItem("φ（平面阵）", "phi")
        self.start_angle = QLineEdit("0"); self.stop_angle = QLineEdit("60")
        for widget in (self.start_angle, self.stop_angle):
            widget.setMaximumWidth(120)
        self.count = QSpinBox(); self.count.setRange(2, 200); self.count.setValue(13)
        row = QHBoxLayout()
        for label, widget in [("扫描轴", self.axis), ("起始", self.start_angle), ("终止", self.stop_angle), ("点数", self.count)]:
            row.addWidget(QLabel(label)); row.addWidget(widget)
        form.addRow(row); layout.addWidget(self.controls)
        actions = QHBoxLayout()
        self.run_button = QPushButton("开始扫描"); self.run_button.clicked.connect(self.start)
        self.cancel_button = QPushButton("取消计算"); self.cancel_button.setEnabled(False); self.cancel_button.clicked.connect(self.cancel)
        self.open_button = QPushButton("打开评估设置"); self.open_button.clicked.connect(self.open_dialog)
        self.save_button = QPushButton("保存评估设置"); self.save_button.clicked.connect(self.save_dialog)
        self.export_button = QPushButton("导出扫描结果"); self.export_button.setEnabled(False); self.export_button.clicked.connect(self.export_dialog)
        for button in (self.run_button, self.cancel_button, self.open_button, self.save_button, self.export_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.progress = QProgressBar(); self.progress.setRange(0, 13); self.progress.setValue(0); layout.addWidget(self.progress)
        self.status = QLabel("从所选已生成设计创建独立评估；主页面尚未计算的新输入不参与。")
        self.status.setWordWrap(True); layout.addWidget(self.status)
        self.tabs = QTabWidget(); layout.addWidget(self.tabs, 1)
        plot_page = QWidget(); plot_layout = QVBoxLayout(plot_page)
        view_row = QHBoxLayout(); self.view = QComboBox()
        for label, key in [("目标方向响应", "response"), ("扫描方向图总览", "map"), ("选定扫描点方向图", "pattern")]:
            self.view.addItem(label, key)
        self.selection = QComboBox(); self.selection.setMinimumContentsLength(15)
        view_row.addWidget(self.view); view_row.addWidget(self.selection, 1)
        layout.insertLayout(layout.indexOf(self.tabs), view_row)
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure); plot_layout.addWidget(self.canvas, 1)
        plot_layout.addWidget(NavigationToolbar2QT(self.canvas, self)); self.tabs.addTab(plot_page, "曲线与总览")
        self.table = QTableWidget(); self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True); self.tabs.addTab(self.table, "逐扫描点指标")
        self.ports = QTableWidget(); self.ports.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.ports.setAlternatingRowColors(True); self.tabs.addTab(self.ports, "选定扫描点的端口相位")
        note = QLabel("目标方向方向性 ≠ 全空间最大方向性或增益。每点重新设置补偿相位，固定同一组误差和失效；不进行误差统计。平面阵切面φ保持原值。")
        note.setWordWrap(True); layout.addWidget(note)
        self.view.currentIndexChanged.connect(self.redraw); self.selection.currentIndexChanged.connect(self.redraw)
        for widget in (self.start_angle, self.stop_angle):
            widget.textChanged.connect(self.mark_dirty)
        self.count.valueChanged.connect(self.mark_dirty)
        self.axis.currentIndexChanged.connect(self.mark_dirty)
        self.update_source_label()

    def update_source_label(self):
        planar = isinstance(self.parameters, PlanarParameters)
        self.axis.setEnabled(planar)
        kind = "直线阵" if not planar else ("规则平面阵" if self.parameters.geometry == "rectangular" else "任意坐标平面阵")
        self.source_label.setText(f"{kind} · {self.parameters.elements}阵元 · {self.feed_settings.topology} · {self.feed_settings.weighting} · {self.parameters.frequency_ghz:g} GHz · 相位模式 {self.feed_settings.phase_mode}")

    def read_study(self):
        return StudySettings(count=self.count.value(), scan_axis=self.axis.currentData(),
                             start_deg=float(self.start_angle.text()), stop_deg=float(self.stop_angle.text()))

    def mark_dirty(self):
        self.status.setText("评估设置已改变；图表和导出仍对应上次成功扫描。")

    def set_busy(self, busy):
        self.controls.setEnabled(not busy)
        for button in (self.run_button, self.open_button, self.save_button):
            button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.export_button.setEnabled(not busy and self.report is not None)

    def start(self):
        if self.worker is not None:
            return
        try:
            study = self.read_study()
            validate_study(self.parameters, self.feed_settings, study)
        except (ValueError, ArithmeticError) as exc:
            QMessageBox.warning(self, "扫描设置无效", str(exc)); return
        self._cancel_pending = False
        self.set_busy(True); self.progress.setRange(0, study.count); self.progress.setValue(0)
        self.status.setText("正在逐角度计算；可取消，原结果保留。")
        self.worker = ScanWorker(self.parameters, self.feed_settings, study, self)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.completed)
        self.worker.start()

    def update_progress(self, value, total):
        self.progress.setRange(0, total); self.progress.setValue(value)

    def cancel(self):
        if self.worker is not None:
            self._cancel_pending = True
            self.worker.requestInterruption()
            self.status.setText("正在取消；当前扫描点结束后停止，原结果保留。")

    def completed(self):
        worker = self.worker
        self.worker = None
        # A close/cancel request arriving after the last numerical step still wins.
        if worker.cancelled or self._cancel_pending:
            self.status.setText("已取消，原结果保留。")
        elif worker.error is not None:
            self.status.setText("扫描失败，原结果保留：" + worker.error)
        else:
            self.report = worker.result
            self.selection.blockSignals(True); self.selection.clear()
            self.selection.addItems([f"#{row['index']}  θ={row['scan_theta_deg']:g}°  φ={row['scan_phi_deg']:g}°" for row in self.report["records"]])
            self.selection.blockSignals(False)
            columns = [("index", "序号"), ("scan_theta_deg", "扫描θ°"), ("scan_phi_deg", "扫描φ°"),
                       ("target_direction_directivity_dbi", "目标方向 D/dBi"), ("target_direction_level_db", "目标方向幅度/dB"),
                       ("output_to_nominal_power_ratio", "输出/名义功率"), ("max_active_phase_error_deg", "最大相位偏差°")]
            self.fill_table(self.table, columns, [{**row, **row["metrics"]} for row in self.report["records"]])
            self.redraw(); self.status.setText(f"已完成{len(self.report['records'])}个扫描点 · {self.report['execution']['workers']}路计算；指标、相位表与导出均对应此快照。")
        worker.deleteLater(); self.set_busy(False)
        if self._close_pending:
            super().reject()

    @staticmethod
    def fill_table(table, columns, rows):
        table.setColumnCount(len(columns)); table.setHorizontalHeaderLabels([label for _, label in columns])
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, (key, _) in enumerate(columns):
                value = row[key]
                text = "—" if value is None else (f"{value:.7g}" if isinstance(value, float) else str(value))
                table.setItem(i, j, QTableWidgetItem(text))
        table.resizeColumnsToContents()

    def redraw(self):
        if self.report is None:
            return
        index = max(0, self.selection.currentIndex())
        draw_study(self.figure, self.report, self.view.currentData(), index); self.canvas.draw_idle()
        self.fill_table(self.ports, [("port", "端口"), ("target_phase_deg", "目标相位°"),
            ("desired_delay_deg", "理想补偿电长度°"), ("applied_delay_deg", "实际补偿电长度°"),
            ("matched_line_wavelengths", "导波波长比例"), ("output_amplitude", "输出幅度"),
            ("output_phase_deg", "输出相位°")], self.report["records"][index]["ports"])

    def save_path(self, path):
        save_study_configuration(path, self.parameters, self.feed_settings, self.read_study())

    def load_path(self, path):
        p, settings, study = load_study_configuration(path)
        self.parameters, self.feed_settings = p, settings
        self.axis.setCurrentIndex(self.axis.findData(study.scan_axis))
        self.start_angle.setText(str(study.start_deg)); self.stop_angle.setText(str(study.stop_deg)); self.count.setValue(study.count)
        self.update_source_label(); self.mark_dirty()

    def export_path(self, path):
        if self.report is None:
            raise ValueError("请先完成扫描。")
        export_study_directory(path, self.report, plot=True)

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开扫描评估设置", "", "JSON (*.json)")
        if path:
            try: self.load_path(path)
            except (ValueError, OSError, UnicodeError) as exc: QMessageBox.warning(self, "打开失败", str(exc))

    def save_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存扫描评估设置", "scan-study.json", "JSON (*.json)")
        if path:
            try: self.save_path(path)
            except (ValueError, OSError, UnicodeError) as exc: QMessageBox.warning(self, "保存失败", str(exc))

    def export_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出到新的扫描结果目录", "scan-results", "目录 (*)")
        if path:
            try: self.export_path(path)
            except (ValueError, OSError) as exc: QMessageBox.warning(self, "导出失败", str(exc))

    def reject(self):
        if self.worker is not None:
            self._close_pending = True; self.cancel(); return
        super().reject()

    def closeEvent(self, event):
        if self.worker is not None:
            self._close_pending = True; self.cancel(); event.ignore()
        else:
            event.accept()
