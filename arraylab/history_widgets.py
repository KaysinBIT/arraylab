"""Restore complete analysis snapshots without invoking an analysis engine."""
from pathlib import Path
import re

import numpy as np
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QFileDialog, QMessageBox

from .core import ArrayParameters
from .feed import FeedSettings
from .history_io import array_result_from_snapshot, load_history, save_history
from .widgets import PatternSet


def _history_sequence(records):
    """Continue generated labels even when older entries have been discarded."""
    numbers = [len(records)]
    for record in records:
        label = record.label if isinstance(record, PatternSet) else record["label"]
        match = re.match(r"^#(\d+)(?:\s|$)", label)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers)


class HistoryActions:
    """MainWindow actions; the window owns all widgets and task routing."""

    def _require_history_idle(self):
        subarray = getattr(self, 'subarray_panel', None)
        if subarray is not None and (subarray._busy or subarray.thread is not None):
            raise ValueError('子阵任务仍在计算或导出，请完成后再恢复历史记录。')
        sparse = getattr(self, "sparse_panel", None)
        if sparse is not None and (getattr(sparse, "_busy", False) or getattr(sparse, "thread", None) is not None):
            raise ValueError("二维稀疏任务仍在计算或导出，请完成后再恢复历史记录。")

    def save_history_path(self, path):
        kind = self.active_task()
        if kind == "linear":
            records = [dict(label=data.label, angles=data.angles, db=data.db,
                            phase=data.phase, result=data.result) for data in self.datasets]
            selected = self.selection.currentIndex()
        elif kind == "feed":
            records = self.feed_panel.history
            selected = self.feed_panel.history_selection.currentIndex()
        else:
            raise ValueError("当前工作区不支持完整历史保存；请选择直线阵或馈电设计。")
        if not records:
            raise ValueError("当前工作区尚无可保存的历史结果，请先计算或读取数据。")
        save_history(path, kind, records, selected)
        self.last_directory = str(Path(path).parent)
        message = f"已保存{len(records)}条历史快照：{Path(path).name}；包含所选项与已有结果，未计算的输入不在其中。"
        self.statusBar().showMessage(message)
        self.record_message(message)

    def load_history_path(self, path):
        self._require_history_idle()
        # Complete validation and object construction before touching live widgets.
        document = load_history(path)
        kind, selected = document["kind"], document["selected"]
        parameters = settings = None
        if kind == "linear":
            records = []
            for record in document["records"]:
                result = (array_result_from_snapshot(record["result"])
                          if record["result"] is not None else None)
                records.append(PatternSet(record["label"], np.array(record["angles"], dtype=float),
                                          np.array(record["db"], dtype=float),
                                          None if record["phase"] is None else np.array(record["phase"], dtype=float),
                                          result))
            if records[selected].result is not None:
                parameters = records[selected].result.parameters
        elif kind == "feed":
            records = document["records"]
            selected_report = records[selected]["report"]
            parameters = ArrayParameters(**selected_report["array_parameters"])
            settings = FeedSettings(**selected_report["settings"])
            parameters.validate()
            settings.validate()
        else:
            raise ValueError("此文件没有可恢复的直线阵或馈电历史。")
        sequence = _history_sequence(records)

        if kind == "linear":
            if parameters is not None:
                self.apply_parameters(parameters)
            self.datasets = records
            self.sequence = sequence
            with QSignalBlocker(self.selection):
                self.selection.clear()
                self.selection.addItems([record.label for record in records])
                self.selection.setCurrentIndex(selected)
            self.keep_history.setChecked(True)
            self.activate_task(kind)
            self.update_selection()
        else:
            panel = self.feed_panel
            panel._changing = True
            try:
                self.apply_parameters(parameters)
                panel.topology.setCurrentIndex(panel.topology.findData(settings.topology))
                for key, edit in panel.fields.items():
                    edit.setText(str(getattr(settings, key)))
                panel.electrical.apply(settings)
                panel.apply_branch_delays(settings)
                panel.history = records
                panel.sequence = sequence
                with QSignalBlocker(panel.history_selection):
                    panel.history_selection.clear()
                    panel.history_selection.addItems([record["label"] for record in records])
                    panel.history_selection.setCurrentIndex(selected)
                self.activate_task(kind)
                panel.display_report(selected_report)
            finally:
                panel._changing = False
        self.last_directory = str(Path(path).parent)
        self.change_workbench_context(self.tabs.currentIndex())
        message = f"已恢复{len(records)}条历史快照：{Path(path).name}；显示保存时的结果，未重新计算。"
        if kind == "linear":
            self.state_label.setText(message + (" 所选外部曲线不含参数，当前输入已保留。" if parameters is None else ""))
        else:
            self.feed_panel.state.setText(self.feed_panel.state.text() + "；已恢复历史快照，未重新计算。")
        self.statusBar().showMessage(message)
        self.record_message(message)

    def save_history_dialog(self):
        kind = self.active_task()
        if kind not in ("linear", "feed"):
            QMessageBox.warning(self, "历史保存", "请选择直线阵或馈电设计后保存完整历史。")
            return
        dialog = QFileDialog(self, "保存完整历史快照", self.last_directory,
                             "ArrayLab历史 (*.arraylab.json *.json)")
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setDefaultSuffix("json")
        dialog.selectFile(f"{kind}-history.arraylab.json")
        if dialog.exec():
            try:
                self.save_history_path(dialog.selectedFiles()[0])
            except (OSError, ValueError, OverflowError) as exc:
                QMessageBox.warning(self, "历史保存失败", str(exc))

    def load_history_dialog(self):
        try:
            self._require_history_idle()
        except ValueError as exc:
            self.statusBar().showMessage(str(exc))
            self.record_message(str(exc))
            return
        path, _ = QFileDialog.getOpenFileName(self, "恢复历史快照（不重新计算）", self.last_directory,
                                            "ArrayLab历史或旧馈电对比 (*.json)",
                                            options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            try:
                self.load_history_path(path)
            except (OSError, ValueError, OverflowError) as exc:
                QMessageBox.warning(self, "历史恢复失败", str(exc))
