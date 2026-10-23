"""Electrical input controls, kept separate from the immutable result snapshots."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QGridLayout, QHBoxLayout, QLabel, QComboBox,
                              QLineEdit, QPushButton, QFileDialog, QMessageBox, QDialog, QVBoxLayout, QTableWidget)
from .excitation import read_excitation_csv


class ElectricalControls:
    def __init__(self, panel, tabs):
        self.panel = panel
        self.custom_amplitudes = ()
        self.custom_phases_deg = ()
        self.phase_mode = QComboBox()
        for text, value in [("不补偿（保留网络相位）", "none"), ("正侧向补偿（0°）", "equalize"), ("扫描补偿（右侧扫描角）", "scan")]:
            self.phase_mode.addItem(text, value)
        self.phase_mode.setCurrentIndex(2)
        self.weighting = QComboBox()
        for text, value in [("均匀幅度", "uniform"), ("Chebyshev", "chebyshev"), ("Taylor", "taylor"), ("自定义CSV", "custom")]:
            self.weighting.addItem(text, value)
        self.fields = {}
        amplitudes = QWidget(); grid = QGridLayout(amplitudes)
        grid.setContentsMargins(6, 3, 6, 3)
        grid.addWidget(QLabel("相位设计"), 0, 0); grid.addWidget(self.phase_mode, 0, 1)
        grid.addWidget(QLabel("幅度分布"), 0, 2); grid.addWidget(self.weighting, 0, 3)
        for key, label, value, col in [("sidelobe_db", "旁瓣参数 (dB)", "45", 0), ("taylor_nbar", "Taylor nbar", "4", 2)]:
            edit = QLineEdit(value); edit.setMinimumWidth(55); edit.setAccessibleName(label)
            self.fields[key] = edit; grid.addWidget(QLabel(label), 1, col); grid.addWidget(edit, 1, col + 1)
        self.import_button = QPushButton("导入幅相CSV"); self.import_button.clicked.connect(self.import_dialog)
        self.preview_button = QPushButton("预览输入"); self.preview_button.clicked.connect(self.preview)
        buttons = QWidget(); row = QHBoxLayout(buttons); row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.import_button); row.addWidget(self.preview_button)
        grid.addWidget(buttons, 2, 0, 1, 2)
        self.custom_label = QLabel("相位列为附加偏置（度）")
        self.custom_label.setWordWrap(True); grid.addWidget(self.custom_label, 2, 2, 1, 2)
        grid.setColumnStretch(1, 1); grid.setColumnStretch(3, 1)
        tabs.addTab(amplitudes, "幅相与加权")
        errors = QWidget(); form = QGridLayout(errors); form.setContentsMargins(6, 3, 6, 3)
        for index, (key, label, value) in enumerate([
            ("phase_bits", "移相位数（0连续）", "0"), ("seed", "随机种子", "0"),
            ("amplitude_error_db", "幅度误差σ (dB)", "0"), ("phase_error_deg", "相位误差σ (°)", "0"),
        ]):
            edit = QLineEdit(value); edit.setMinimumWidth(55); edit.setAccessibleName(label)
            self.fields[key] = edit
            row, col = index // 2, 2 * (index % 2)
            form.addWidget(QLabel(label), row, col); form.addWidget(edit, row, col + 1)
        self.failed = QLineEdit(); self.failed.setPlaceholderText("逗号分隔，如1,3；空白为无失效")
        form.addWidget(QLabel("失效阵元"), 2, 0); form.addWidget(self.failed, 2, 1, 1, 3)
        form.setColumnStretch(1, 1); form.setColumnStretch(3, 1)
        tabs.addTab(errors, "误差与失效")
        self.phase_mode.currentIndexChanged.connect(panel.mark_dirty)
        self.weighting.currentIndexChanged.connect(panel.mark_dirty)
        self.failed.textChanged.connect(panel.mark_dirty)
        for edit in self.fields.values():
            edit.textChanged.connect(panel.mark_dirty)

    def values(self):
        values = {key: (int(edit.text().strip()) if key in ("taylor_nbar", "seed", "phase_bits") else float(edit.text().strip()))
                  for key, edit in self.fields.items()}
        failed = self.failed.text().strip()
        values.update(phase_mode=self.phase_mode.currentData(), weighting=self.weighting.currentData(),
                      failed_ports=tuple(int(part.strip()) for part in failed.split(",")) if failed else ())
        custom = values["weighting"] == "custom"
        values.update(custom_amplitudes=self.custom_amplitudes if custom else (),
                      custom_phases_deg=self.custom_phases_deg if custom else ())
        return values

    def apply(self, settings):
        self.phase_mode.setCurrentIndex(self.phase_mode.findData(settings.phase_mode))
        self.weighting.setCurrentIndex(self.weighting.findData(settings.weighting))
        for key, edit in self.fields.items():
            edit.setText(str(getattr(settings, key)))
        self.failed.setText(",".join(map(str, settings.failed_ports)))
        self.custom_amplitudes, self.custom_phases_deg = settings.custom_amplitudes, settings.custom_phases_deg
        self.custom_label.setText(f"自定义激励：{len(self.custom_amplitudes)}行；相位为附加偏置")

    def import_path(self, path):
        from dataclasses import replace
        from .feed import validate_feed_inputs
        p = self.panel.array_parameters() if hasattr(self.panel, "array_parameters") else self.panel.window.read_parameters()
        amplitudes, phases = read_excitation_csv(path, p.elements)
        settings = replace(self.panel.read_settings(), weighting="custom", custom_amplitudes=amplitudes, custom_phases_deg=phases)
        validate_feed_inputs(p, settings)
        self.custom_amplitudes, self.custom_phases_deg = amplitudes, phases
        self.weighting.setCurrentIndex(self.weighting.findData("custom"))
        self.custom_label.setText(f"已导入{len(amplitudes)}行；生成设计后生效")
        self.panel.mark_dirty()

    def import_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self.panel, "导入激励：element,amplitude,phase_deg", self.panel.window.last_directory,
                                            "激励CSV (*.csv)", options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            try:
                self.import_path(path)
            except (OSError, ValueError, OverflowError) as exc:
                QMessageBox.warning(self.panel, "激励导入失败", str(exc))

    def preview(self):
        try:
            from .excitation import target_amplitudes
            p = self.panel.array_parameters() if hasattr(self.panel, "array_parameters") else self.panel.window.read_parameters()
            s = self.panel.read_settings()
            amplitudes = self.panel.preview_amplitudes() if hasattr(self.panel, "preview_amplitudes") else target_amplitudes(p, s)
            phases = s.custom_phases_deg if s.weighting == "custom" else [0] * p.elements
            dialog = QDialog(self.panel); dialog.setWindowTitle("目标幅度与附加相位预览（幅度峰值归一化）"); dialog.resize(480, 450)
            layout = QVBoxLayout(dialog); table = QTableWidget()
            rows = [dict(port=i + 1, amplitude=float(a), phase=float(ph)) for i, (a, ph) in enumerate(zip(amplitudes, phases))]
            self.panel.fill_table(table, [("port", "阵元"), ("amplitude", "相对幅度"), ("phase", "附加相位 (°)")], rows)
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            layout.addWidget(table); dialog.exec()
        except (ValueError, ArithmeticError) as exc:
            QMessageBox.warning(self.panel, "预览失败", str(exc))
