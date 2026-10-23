from dataclasses import asdict
from pathlib import Path
import math
import time
from datetime import datetime

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHeaderView, QMainWindow, QMessageBox, QTableView,
)

from . import __version__
from .core import ArrayParameters, InputError, calculate
from .configuration import load_configuration, save_configuration
from .dataio import read_pattern, write_pattern
from .widgets import NumericTableModel, PatternSet


from .workbench import STYLE, build_workbench
from .icons import workbench_icon


class MainWindow(QMainWindow):
    def __init__(self, auto_compute=True):
        super().__init__()
        self.setWindowTitle(f"ArrayLab {__version__} · 阵列分析")
        self.resize(1440, 900)
        self.setMinimumSize(1100, 740)
        self.datasets: list[PatternSet] = []
        self.sequence = 0
        self.last_directory = str(Path.home())
        self._changing = False
        self._spacing_unit = "lambda"
        self.setStyleSheet(STYLE)
        self.build_ui()
        self.build_menus()
        self.reset_parameters()
        self.update_selection()
        if auto_compute:
            QTimer.singleShot(60, self.compute)

    def build_ui(self):
        build_workbench(self)

    def show_view(self, index):
        self.tabs.setCurrentIndex(index)
        if index == 4 and self.feed_panel.report is None:
            self.feed_panel.generate()

    def set_plot_view(self, kind):
        self.plot_type.setCurrentIndex(kind)
        self.show_view(0)

    def reset_plot_view(self):
        self.redraw()
        self.show_view(0)

    def save_plot(self):
        if self.selected_dataset() is not None:
            self.show_view(0)
            self.plot.toolbar.save_figure()
        else:
            self.statusBar().showMessage("请先计算或读取方向图。")

    def focus_parameters(self):
        self.inspector_dock.show()
        self.inspector_dock.raise_()
        self.inputs["frequency_ghz"].setFocus()

    def restore_workbench(self):
        for item, area in [(self.nav_dock, Qt.DockWidgetArea.LeftDockWidgetArea),
                           (self.inspector_dock, Qt.DockWidgetArea.RightDockWidgetArea),
                           (self.messages_dock, Qt.DockWidgetArea.BottomDockWidgetArea)]:
            item.setFloating(False)
            self.addDockWidget(area, item)
            item.show()
        self.resizeDocks([self.nav_dock, self.inspector_dock], [235, 285], Qt.Orientation.Horizontal)
        self.resizeDocks([self.messages_dock], [135], Qt.Orientation.Vertical)
        self.statusBar().showMessage("工作台布局已恢复。")

    def record_message(self, text):
        if text and hasattr(self, "message_log"):
            self.message_log.appendPlainText(f"{datetime.now():%H:%M:%S}  {text}")

    def sync_history_tree(self):
        if not hasattr(self, "history_tree_item"):
            return
        from PySide6.QtWidgets import QTreeWidgetItem
        self.tree.blockSignals(True)
        signature = tuple((id(data), data.label) for data in self.datasets)
        if signature != getattr(self, "_history_signature", None):
            self.history_tree_item.takeChildren()
            for i, data in enumerate(self.datasets):
                item = QTreeWidgetItem(self.history_tree_item, [data.label])
                item.setIcon(0, workbench_icon("plot"))
                item.setData(0, Qt.ItemDataRole.UserRole, ("result", i))
                item.setToolTip(0, data.label)
            self._history_signature = signature
        if 0 <= self.selection.currentIndex() < len(self.datasets):
            self.tree.setCurrentItem(self.history_tree_item.child(self.selection.currentIndex()))
        self.history_tree_item.setExpanded(True)
        self.tree.blockSignals(False)

    def make_table(self):
        view = QTableView()
        view.setAlternatingRowColors(True)
        view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        return view

    def build_menus(self):
        file_menu = self.workbench_menu.addMenu("文件(&F)")
        plot_menu = self.workbench_menu.addMenu("绘图(&P)")
        array_menu = self.workbench_menu.addMenu("阵列(&A)")
        help_menu = self.workbench_menu.addMenu("帮助(&H)")
        view_menu = self.workbench_menu.addMenu("视图(&V)")

        def action(menu, title, callback, shortcut=None):
            a = QAction(title, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(callback)
            menu.addAction(a)
            return a

        action(file_menu, "打开分析方案…", self.open_configuration_dialog, "Ctrl+Shift+O")
        action(file_menu, "保存当前参数为方案…", self.save_configuration_dialog, "Ctrl+Shift+S")
        file_menu.addSeparator()
        action(file_menu, "读取方向图…", self.open_dialog, "Ctrl+O")
        self.save_action = action(file_menu, "保存所选方向图…", self.save_dialog, "Ctrl+S")
        action(file_menu, "退出", self.close, "Ctrl+Q")
        action(plot_menu, "直角坐标方向图", lambda: self.plot_type.setCurrentIndex(0))
        action(plot_menu, "极坐标方向图", lambda: self.plot_type.setCurrentIndex(1))
        action(plot_menu, "复制当前图形", self.copy_plot)
        action(array_menu, "计算均匀直线阵", self.compute, "F5")
        action(array_menu, "馈电网络阻抗设计", lambda: self.show_view(4), "F6")
        action(array_menu, "载入说明书验证例题", self.load_example)
        action(help_menu, "本例程帮助", self.show_help, "F1")
        action(help_menu, "关于 ArrayLab", self.show_about)
        for item in (self.nav_dock, self.inspector_dock, self.messages_dock):
            view_menu.addAction(item.toggleViewAction())
        action(view_menu, "恢复工作台布局", self.restore_workbench)

    def read_parameters(self):
        values = {}
        for key, edit in self.inputs.items():
            if key == "scan_deg" and self.phase_mode.currentIndex() == 0:
                values[key] = 0.0
                continue
            try:
                values[key] = int(edit.text().strip()) if key == "elements" else float(edit.text().strip())
            except ValueError as exc:
                raise InputError(key, f"“{edit.accessibleName()}”须填写有效数值。") from exc
        if self.phase_mode.currentIndex() == 0:
            values["scan_deg"] = 0.0
        p = ArrayParameters(**values, spacing_unit=self.unit.currentData())
        p.validate()
        return p

    def reset_parameters(self):
        self.apply_parameters(ArrayParameters())

    def apply_parameters(self, parameters):
        # Unit changes must not convert values while restoring a complete snapshot.
        self._changing = True
        self.unit.setCurrentIndex(self.unit.findData(parameters.spacing_unit))
        self._spacing_unit = parameters.spacing_unit
        self.phase_mode.setCurrentIndex(1 if parameters.scan_deg != 0 else 0)
        for key, value in asdict(parameters).items():
            if key in self.inputs:
                self.inputs[key].setText(str(value))
        self.inputs["scan_deg"].setEnabled(self.phase_mode.currentIndex() == 1)
        self._changing = False
        self.mark_dirty()

    def load_configuration_path(self, path):
        parameters = load_configuration(path)  # Validate everything before changing widgets.
        self.apply_parameters(parameters)
        self.last_directory = str(Path(path).parent)
        self.state_label.setText("方案已加载，尚未重新计算。已有图形和指标仍对应所选历史结果；点击“开始计算”生成新结果。")
        self.statusBar().showMessage(f"已加载分析方案：{Path(path).name}；请点击“开始计算”。")
        self.record_message(f"已加载分析方案：{Path(path).name}（待计算）。")
        self.focus_parameters()

    def save_configuration_path(self, path):
        save_configuration(path, self.read_parameters())
        self.last_directory = str(Path(path).parent)
        self.statusBar().showMessage(f"已保存当前输入参数：{Path(path).name}。")
        self.record_message(f"已保存分析方案：{Path(path).name}（当前输入参数）。")

    def open_configuration_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开分析方案（加载后需重新计算）", self.last_directory,
                                            "分析方案 (*.json)", options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            try:
                self.load_configuration_path(path)
            except (OSError, ValueError, OverflowError) as exc:
                QMessageBox.warning(self, "方案加载失败", str(exc))

    def save_configuration_dialog(self):
        # A default suffix lets Qt check the exact destination for overwrite confirmation.
        dialog = QFileDialog(self, "保存当前输入参数为分析方案", self.last_directory, "分析方案 (*.json)")
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setDefaultSuffix("json")
        dialog.selectFile("analysis.json")
        if dialog.exec():
            try:
                self.save_configuration_path(dialog.selectedFiles()[0])
            except (OSError, ValueError, OverflowError) as exc:
                QMessageBox.warning(self, "方案保存失败", str(exc))

    def load_example(self):
        self.reset_parameters()
        self.inputs["elements"].setText("5")
        self.inputs["spacing"].setText("0.4")
        self.phase_mode.setCurrentIndex(1)
        self.inputs["scan_deg"].setText("60")
        self.compute()
        self.statusBar().showMessage("说明书 p.48 例题：5 阵元、0.4 λ、扫描 60°，参考方向性 7.1 dB。频率取 3 GHz 以实例化尺寸。")

    def phase_mode_changed(self):
        if not hasattr(self, "inputs"):
            return
        enabled = self.phase_mode.currentIndex() == 1
        self.inputs["scan_deg"].setEnabled(enabled)
        self.mark_dirty()

    def change_unit(self):
        new = self.unit.currentData()
        if self._changing:
            self._spacing_unit = new
            return
        try:
            f = float(self.inputs["frequency_ghz"].text())
            value = float(self.inputs["spacing"].text())
            if math.isfinite(f) and f > 0 and math.isfinite(value):
                wavelength = 299.792458 / f
                value = value * wavelength if new == "mm" else value / wavelength
                self.inputs["spacing"].setText(f"{value:.12g}")
        except ValueError:
            pass
        self._spacing_unit = new
        self.mark_dirty()

    def mark_dirty(self):
        if self._changing or not hasattr(self, "state_label"):
            return
        if hasattr(self, "feed_panel"):
            self.feed_panel.mark_dirty()
        if self.datasets:
            self.state_label.setText("参数已修改，尚未重新计算。图形、指标和“保存数据”仍对应所选历史结果；“保存方案”保存当前输入。")
        else:
            self.state_label.setText("设置参数后点击“计算”。扫描角从 +z 朝 +x 为正。")

    def compute(self):
        try:
            p = self.read_parameters()
            start = time.perf_counter()
            result = calculate(p)
            elapsed = (time.perf_counter() - start) * 1000
        except (ValueError, ArithmeticError) as exc:
            self.record_message("输入错误：" + str(exc))
            QMessageBox.warning(self, "请检查输入", str(exc))
            if isinstance(exc, InputError) and exc.field in self.inputs:
                self.inputs[exc.field].setFocus()
                self.inputs[exc.field].selectAll()
            return
        self.sequence += 1
        label = f"#{self.sequence}  N={p.elements} · d={p.spacing_lambda:.4g}λ · θ₀={p.scan_deg:g}°"
        self.add_dataset(PatternSet(label, result.angles_deg, result.pattern_db, result.phase_deg, result))
        self.tabs.setCurrentIndex(0)
        self.state_label.setText(f"计算完成  {label}   |   峰值归一化为 0 dB · 方向性使用全空间积分")
        self.statusBar().showMessage(f"计算耗时 {elapsed:.1f} ms（不含绘图）  ·  {len(result.angles_deg)} 个采样点  ·  最多保留最近 5 条曲线")
        self.record_message(f"结果 {label}：方向性 {result.directivity_dbi:.4f} dBi。")
        for notice in result.notices:
            self.record_message(notice)

    def add_dataset(self, data):
        if not self.keep_history.isChecked():
            self.datasets.clear()
        self.datasets.append(data)
        self.datasets = self.datasets[-5:]
        self.selection.blockSignals(True)
        self.selection.clear()
        self.selection.addItems([d.label for d in self.datasets])
        self.selection.setCurrentIndex(len(self.datasets) - 1)
        self.selection.blockSignals(False)
        self.update_selection()

    def remove_selected(self):
        i = self.selection.currentIndex()
        if 0 <= i < len(self.datasets):
            self.datasets.pop(i)
            self.selection.blockSignals(True)
            self.selection.clear()
            self.selection.addItems([d.label for d in self.datasets])
            self.selection.setCurrentIndex(min(i, len(self.datasets) - 1))
            self.selection.blockSignals(False)
            self.update_selection()

    def selected_dataset(self):
        i = self.selection.currentIndex()
        return self.datasets[i] if 0 <= i < len(self.datasets) else None

    def update_selection(self):
        if not hasattr(self, "plot"):
            return
        data = self.selected_dataset()
        for field in self.metrics.values():
            field.setText("—")
        self.save_button.setEnabled(data is not None)
        if hasattr(self, "save_action"):
            self.save_action.setEnabled(data is not None)
        self.remove_button.setEnabled(data is not None)
        self.phase_export.setEnabled(data is not None and data.phase is not None)
        self.notice.hide()
        self.physical.setText("波长与物理间距将在计算后显示")
        columns, headers = [], ["θ (°)", "幅度 (dB)"]
        excitation = []
        if data:
            columns = [data.angles, data.db]
            if data.phase is not None:
                headers.append("相位 (°)")
                columns.append(data.phase)
            self.metrics["samples"].setText(str(len(data.angles)))
            if data.result:
                r, p = data.result, data.result.parameters
                self.geometry.set_parameters(p)
                self.geometry.show()
                self.metrics["peak"].setText(f"{r.peaks_deg[0]:.3f}°" if len(r.peaks_deg) == 1 else "无唯一主瓣")
                self.metrics["beamwidth"].setText(f"{r.beamwidth_deg:.3f}°" if r.beamwidth_deg is not None else "不报告")
                self.metrics["directivity"].setText(f"{r.directivity_dbi:.4f} dBi")
                self.metrics["scan"].setText(f"{p.scan_deg:g}°")
                phase_step = -360 * p.spacing_lambda * math.sin(math.radians(p.scan_deg))
                self.metrics["phase"].setText(f"{phase_step:.3f}°")
                self.physical.setText(f"所选结果：λ = {p.wavelength_mm:.5g} mm   d = {p.spacing_mm:.5g} mm")
                excitation = [np.arange(1, p.elements + 1), r.positions_mm, np.ones(p.elements), r.excitation_phase_deg]
                if r.notices:
                    self.notice.setText("\n".join(r.notices))
                    self.notice.show()
                try:
                    matches = self.read_parameters() == p
                except ValueError:
                    matches = False
                self.state_label.setText(("显示所选计算结果：" if matches else "当前输入与所选结果不同；“保存数据”仅针对所选结果：") + data.label)
            else:
                self.geometry.hide()
                self.notice.setText("外部曲线按原始 dB 值显示，不重新归一化。没有完整模型，故不推算方向性或主瓣指标。")
                self.notice.show()
                self.state_label.setText("显示外部数据：" + data.label + "；未将其作为当前输入的计算结果。")
        else:
            self.geometry.hide()
            self.state_label.setText("尚无结果。点击“计算”或读取方向图文件。")
        for table, names, values in [
            (self.pattern_table, headers, columns),
            (self.excitation_table, ["阵元", "x (mm)", "相对幅度", "馈电相位 (°)"], excitation),
        ]:
            old_model = table.model()
            table.setModel(NumericTableModel(names, values, table))
            if old_model is not None:
                old_model.deleteLater()
        for field in self.metrics.values():
            field.setAccessibleName(field.text())
        self.sync_history_tree()
        self.redraw()

    def redraw(self):
        if hasattr(self, "plot"):
            self.plot.draw_sets(self.datasets, self.selection.currentIndex(),
                                self.plot_type.currentIndex() == 1,
                                [-30, -40, -60, -80][self.floor_combo.currentIndex()])

    def select_result_tab(self, current, previous):
        payload = current.data(0, Qt.ItemDataRole.UserRole) if current else None
        if not payload:
            return
        kind, index = payload
        if kind == "view":
            self.show_view(index)
        elif kind == "parameters":
            self.focus_parameters()
        elif kind == "result":
            self.selection.setCurrentIndex(index)
            self.show_view(0)

    def import_path(self, path):
        angles, db, phase = read_pattern(path)
        self.add_dataset(PatternSet(Path(path).name, angles, db, phase))
        self.last_directory = str(Path(path).parent)
        self.tabs.setCurrentIndex(0)
        self.statusBar().showMessage("已读取方向图：" + str(path))

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "读取方向图（角度、dB幅度、可选相位）", self.last_directory,
                                            "方向图数据 (*.dat *.txt *.csv);;所有文件 (*)",
                                            options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            try:
                self.import_path(path)
            except (OSError, ValueError, UnicodeError) as exc:
                QMessageBox.warning(self, "读取失败", str(exc))

    def export_path(self, path):
        data = self.selected_dataset()
        if data is None:
            raise ValueError("请先计算或读取方向图。")
        write_pattern(path, data.angles, data.db, data.phase if self.phase_export.isChecked() else None)
        self.last_directory = str(Path(path).parent)

    def save_dialog(self):
        if self.selected_dataset() is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存所选方向图", str(Path(self.last_directory) / "pattern.dat"),
                                            "方向图数据 (*.dat);;数值文本 (*.txt)",
                                            options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            try:
                self.export_path(path)
                self.statusBar().showMessage(f"已保存所选结果：{path}")
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "保存失败", str(exc))

    def copy_plot(self):
        if self.selected_dataset():
            QApplication.clipboard().setPixmap(self.plot.canvas.grab())
            self.statusBar().showMessage("当前图形已复制到剪贴板。")

    def show_help(self):
        QMessageBox.information(self, "均匀直线阵 · 使用帮助",
            "1. 输入频率、阵元数、间距和绘图步长。间距可用 λ 或 mm。\n"
            "2. 选择正侧向或指定扫描角，点击计算（F5）。\n"
            "3. 切换直角/极坐标；在图中移动鼠标读数、点击定位。\n"
            "4. 勾选叠加比较后计算可保留最近5条曲线。所选曲线决定指标和保存内容。\n"
            "5. 文件菜单支持数值方向图读写，图下工具栏支持缩放及图片保存。\n\n"
            "分析方案：主页的“保存方案”保存当前输入的6项参数；“打开方案”加载后需重新计算。"
            "方案不包含历史曲线或窗口布局；也可通过无头入口的--config读取。\n\n"
            "馈电阻抗（F6）：支持并联树形、串行抽头和1分2后对称串馈（偶数阵元），输入阻抗、节点阻抗和阵元纯电阻负载可改。"
            "生成λ/4变换线阻抗，支持独立馈电设置保存/加载及JSON、CSV、SVG导出。"
            "当前只做单频点理想阻抗综合，不包含相位均衡、线宽或实际基板。\n\n"
            "坐标：阵列沿x轴，以阵列中心为相位参考。图为x-z前半空间切面，θ从+z向+x为正，范围−90°至90°。"
            "各向同性单元没有极化；不提供同/交叉极化标识。\n"
            "方向性采用全空间功率积分，不等于增益。半功率宽度按功率峰值的一半计算（−3.0103 dB）。"
            "出现多个等高主瓣或边界不完整时不报告宽度。\n\n"
            "首版：1–200阵元，0.001–1000 GHz，间距0.000001–10 λ，绘图步长0.01°–5°。"
            "除阵元数上限外，这些是本版实现范围，不是说明书给出的原软件限制。\n"
            "DAT格式：角度(°)、幅度(dB)、可选相位(°)，无表头。数据下限−300 dB用于表达数值零点；零点相位无物理意义。\n\n"
            "功能参考说明书III和IV.A、D.1。界面采用功能区、导航树、可停靠面板和消息窗口。\n"
            "面板关闭或拖动后，可通过“视图 → 恢复工作台布局”恢复；模型示意可切换视角、网格和缩放。")

    def show_about(self):
        QMessageBox.about(self, "关于 ArrayLab", f"ArrayLab {__version__}\n均匀直线阵分析 · 本地桌面版\n\n"
                          "Python / PySide6 / NumPy / SciPy / Matplotlib\n"
                          "本版范围：各向同性单元、均匀幅度、无互耦。\n"
                          "本软件为独立实现，与PCAAD原软件无隶属关系。\n"
                          "项目源码按MIT许可；依赖许可见随包THIRD_PARTY_NOTICES。")
