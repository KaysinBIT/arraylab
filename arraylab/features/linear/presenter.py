"""Linear-array presenter and automatic-history coordination for a supplied workbench host."""
from dataclasses import asdict
from pathlib import Path
import math, os, time
import numpy as np
from PySide6.QtCore import Qt, QTimer, QSignalBlocker, QStandardPaths
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
from arraylab.features.linear.model import ArrayParameters, InputError, calculate
from arraylab.features.linear.configuration import load_configuration, save_configuration
from arraylab.features.linear.dataio import read_pattern, write_pattern
from arraylab.features.linear.widgets import NumericTableModel, PatternSet
from arraylab.shared.qt.icons import workbench_icon
from arraylab.features.linear.records import array_result_from_snapshot

def pattern_from_record(record):
    result = array_result_from_snapshot(record["result"]) if record["result"] is not None else None
    # A validated calculated record duplicates these arrays in its JSON envelope;
    # share the immutable restored arrays instead of keeping both copies in RAM.
    return PatternSet(record["label"], result.angles_deg if result is not None else np.array(record["angles"], dtype=float),
                      result.pattern_db if result is not None else np.array(record["db"], dtype=float),
                      result.phase_deg if result is not None else (
                          None if record["phase"] is None else np.array(record["phase"], dtype=float)), result)


class LinearAnalysisActions:

    def _initialize_linear_history(self, explicit_path):
        """Normal launches restore history; smoke/probe windows stay isolated."""
        if not self._history_requested:
            self._history_loading = False
            return
        directory = os.environ.get("ARRAYLAB_HISTORY_DIR")
        self._history_path = Path(explicit_path) if explicit_path is not None else (
            Path(directory) / "linear-history.sqlite3" if directory else
            Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)) / "history" / "linear-history.sqlite3")
        try:
            from arraylab.features.linear.history_store import LinearHistoryStore
            store = LinearHistoryStore(self._history_path)
            entries, sequence, selected_id = store.load()
            restored = []
            for row_id, record in entries:
                data = pattern_from_record(record)
                data._archive_id = row_id
                restored.append(data)
            self._history_store = store
            self.datasets = restored
            self.sequence = sequence
            selected = next((i for i, data in enumerate(restored) if data._archive_id == selected_id), len(restored) - 1)
            with QSignalBlocker(self.selection):
                self.selection.clear()
                self.selection.addItems([data.label for data in restored])
                self.selection.setCurrentIndex(selected)
            if selected >= 0 and restored[selected].result is not None:
                self.apply_parameters(restored[selected].result.parameters)
            self._persisted_selection = restored[selected]._archive_id if selected >= 0 else None
            self.update_selection()
            if restored:
                message = f"已自动恢复{len(restored)}条历史计算记录，未重新计算。"
                self.statusBar().showMessage(message)
                self.record_message(message)
        except (OSError, ValueError, ArithmeticError) as exc:
            self._history_warning = f"自动历史无法读取，原档案未覆盖：{exc}"
            self.statusBar().showMessage(self._history_warning)
            self.record_message(self._history_warning)
        finally:
            self._history_loading = False
            self.history_tree_item.setToolTip(0, f"自动保存位置：{self._history_path}" + (
                "\n" + self._history_warning if self._history_warning else ""))

    def _persist_pending_history(self):
        if not self._history_requested or self._history_loading:
            return
        pending = [(index, data) for index, data in enumerate(self.datasets) if getattr(data, "_archive_id", None) is None]
        if not pending:
            return
        try:
            if self._history_store is None:
                from arraylab.features.linear.history_store import LinearHistoryStore
                store = LinearHistoryStore(self._history_path)
                store.load()  # Do not append to a damaged, unreadable archive.
                self._history_store = store
            for index, data in pending:
                record = dict(label=data.label, angles=data.angles, db=data.db, phase=data.phase, result=data.result)
                row_id, sequence, label = self._history_store.append(record,
                    getattr(data, "_history_sequence", self.sequence),
                    generated=getattr(data, "_history_generated", False))
                data._archive_id = row_id
                data._history_sequence = sequence
                data.label = label
                if self.selection.count() == len(self.datasets):
                    with QSignalBlocker(self.selection):
                        self.selection.setItemText(index, label)
                self.sequence = max(self.sequence, sequence)
                self._persisted_selection = row_id
            self._history_warning = ""
        except (OSError, ValueError, ArithmeticError) as exc:
            self._history_warning = f"历史自动保存失败；未保存结果仍保留在本次会话，请用“保存记录”备份。{exc}"
            self.statusBar().showMessage(self._history_warning)
            self.record_message(self._history_warning)

    def _persist_history_selection(self):
        if self._history_loading or self._history_store is None:
            return
        data = self.selected_dataset()
        row_id = getattr(data, "_archive_id", None)
        if row_id is None or row_id == self._persisted_selection:
            return
        try:
            self._history_store.select(row_id)
            self._persisted_selection = row_id
        except (OSError, ValueError) as exc:
            self._history_warning = str(exc)
            self.record_message(str(exc))

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
        if self.active_task()=="linear" and 0 <= self.selection.currentIndex() < len(self.datasets):
            self.tree.setCurrentItem(self.history_tree_item.child(self.selection.currentIndex()))
        self.history_tree_item.setExpanded(True)
        self.tree.blockSignals(False)

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

    def open_linear_configuration_dialog(self):
        if self.sparse_active():
            self.sparse_panel.open_settings_dialog(); return
        path, _ = QFileDialog.getOpenFileName(self, "打开直线阵参数（加载后需重新计算）", self.last_directory,
                                            "分析方案 (*.json)", options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            try:
                self.load_configuration_path(path)
            except (OSError, ValueError, OverflowError) as exc:
                QMessageBox.warning(self, "方案加载失败", str(exc))

    def save_linear_configuration_dialog(self):
        if self.sparse_active():
            self.sparse_panel.save_settings_dialog(); return
        # A default suffix lets Qt check the exact destination for overwrite confirmation.
        dialog = QFileDialog(self, "保存当前直线阵参数", self.last_directory, "分析方案 (*.json)")
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
        self.activate_task("linear")
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
            self.state_label.setText("参数已修改，尚未重新计算。图形、指标和“导出曲线”仍对应所选历史结果；“保存参数”保存当前输入。")
        else:
            self.state_label.setText("设置参数后点击“计算”。扫描角从 +z 朝 +x 为正。")

    def compute_linear(self):
        if self.sparse_active():
            self.sparse_panel.start(); return
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
        data = PatternSet(label, result.angles_deg, result.pattern_db, result.phase_deg, result)
        self.add_dataset(data, generated=True)
        label = data.label
        self.tabs.setCurrentIndex(0)
        self.state_label.setText(f"计算完成  {label}   |   峰值归一化为 0 dB · 方向性使用全空间积分" + (
            "；" + self._history_warning if self._history_warning else ""))
        self.statusBar().showMessage(f"计算耗时 {elapsed:.1f} ms（不含绘图）  ·  {len(result.angles_deg)} 个采样点  ·  已保留{len(self.datasets)}条历史" + (
            "；" + self._history_warning if self._history_warning else ""))
        self.record_message(f"结果 {label}：方向性 {result.directivity_dbi:.4f} dBi。")
        for notice in result.notices:
            self.record_message(notice)

    def add_dataset(self, data, *, generated=False):
        data._history_generated = generated
        data._history_sequence = self.sequence
        self.datasets.append(data)
        self._persist_pending_history()
        self.selection.blockSignals(True)
        self.selection.clear()
        self.selection.addItems([d.label for d in self.datasets])
        self.selection.setCurrentIndex(len(self.datasets) - 1)
        self.selection.blockSignals(False)
        self.update_selection()

    def remove_selected(self):
        i = self.selection.currentIndex()
        if 0 <= i < len(self.datasets):
            row_id = getattr(self.datasets[i], "_archive_id", None)
            if row_id is not None and self._history_store is not None:
                next_index = i - 1 if i > 0 else 1
                next_id = getattr(self.datasets[next_index], "_archive_id", -1) if next_index < len(self.datasets) else -1
                try:
                    self._history_store.remove(row_id, next_id)
                except (OSError, ValueError) as exc:
                    QMessageBox.warning(self, "历史删除失败", str(exc))
                    return
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
                self.state_label.setText(("显示所选计算结果：" if matches else "当前输入与所选结果不同；“导出曲线”仅针对所选结果：") + data.label)
            else:
                self.geometry.hide()
                self.notice.setText("外部曲线按原始 dB 值显示，仅提取采样切面指标。缺少完整模型，无法推算三维方向图、方向性或增益。")
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
        self._persist_history_selection()
        self.redraw()
        self.refresh_actions()

    def redraw(self):
        if hasattr(self, "plot"):
            selected = self.selection.currentIndex()
            indices = ([selected] if 0 <= selected < len(self.datasets) else [])
            if self.keep_history.isChecked() and indices:
                # Bound graph work independently of the complete archive.
                indices = sorted(set(range(max(0, len(self.datasets) - 4), len(self.datasets))) | {selected})
            displayed = [self.datasets[index] for index in indices]
            self.plot.draw_sets(displayed, indices.index(selected) if selected in indices else -1,
                                self.plot_type.currentIndex() == 1,
                                [-30, -40, -60, -80][self.floor_combo.currentIndex()],
                                view="3d" if self.plot_type.currentIndex() == 2 else None)

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

    def save_linear_dialog(self):
        if self.sparse_active():
            self.sparse_panel.export(); return
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

    def selected_linear_array_snapshot(self):
        """Return owned coordinates/amplitudes from the selected computed result."""
        selected = self.selected_dataset()
        if selected is None or selected.result is None:
            raise ValueError('请先选择具有阵列参数的已计算直线阵。')
        result = selected.result
        return {'positions_lambda': np.column_stack((result.positions_mm / result.parameters.wavelength_mm,
                    np.zeros(result.parameters.elements))),
                'weights': np.ones(result.parameters.elements),
                'frequency_ghz': result.parameters.frequency_ghz,
                'element_model': 'isotropic',
                'origin': '直线阵：' + selected.label}
