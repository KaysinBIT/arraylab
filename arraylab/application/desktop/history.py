"""Restore complete analysis snapshots without invoking an analysis engine."""
from pathlib import Path
import re

import numpy as np
from PySide6.QtCore import QSignalBlocker,QTimer
from PySide6.QtWidgets import QFileDialog, QMessageBox

from arraylab.application.records import array_result_from_snapshot, load_history, save_history
from arraylab.features.linear import PatternSet
from arraylab.features.linear import pattern_from_record




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
        if getattr(self,'_record_reader',None) is not None:raise ValueError('正在读取完整记录，请稍候。')
        if any(panel._busy or panel.thread is not None for panel in getattr(self,'simulation_panels',{}).values()):
            raise ValueError('雷达任务仍在运行或保存，请结束后再打开记录。')
        waveform = getattr(self, 'waveform_panel', None)
        if waveform is not None and (waveform._busy or waveform.thread is not None):
            raise ValueError('波形任务仍在计算或保存，请完成后再恢复记录。')
        mimo = getattr(self, 'mimo_panel', None)
        if mimo is not None and (mimo._busy or mimo.thread is not None):
            raise ValueError('MIMO任务仍在计算或保存，请完成后再恢复记录。')
        subarray = getattr(self, 'subarray_panel', None)
        if subarray is not None and (subarray._busy or subarray.thread is not None):
            raise ValueError('子阵任务仍在计算或导出，请完成后再恢复历史记录。')
        sparse = getattr(self, "sparse_panel", None)
        if sparse is not None and (getattr(sparse, "_busy", False) or getattr(sparse, "thread", None) is not None):
            raise ValueError("二维稀疏任务仍在计算或导出，请完成后再恢复历史记录。")

    def save_history_path(self, path):
        kind = self.active_task()
        if kind in getattr(self,'simulation_panels',{}):
            self.simulation_panels[kind].save_record_path(path);return
        if kind == 'waveform':
            self.waveform_panel.save_record_path(path); return
        if kind == 'mimo':
            self.mimo_panel.save_record_path(path); return
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
        if Path(path).suffix.lower()=='.npz':
            from arraylab.application.simulation import load_record
            report,arrays=load_record(path)
            kind=report['feature'];self.simulation_panels[kind].accept_result(report,arrays,restore_inputs=True)
            self.activate_task(kind);return
        from arraylab.shared.simulation_io import read_json
        marker=read_json(path)
        if isinstance(marker,dict) and marker.get('format')=='arraylab.simulation.export':
            from arraylab.application.simulation import load_record
            report,arrays=load_record(path);kind=report['feature']
            self.simulation_panels[kind].accept_result(report,arrays,restore_inputs=True);self.activate_task(kind);return
        from arraylab.features.waveform import is_waveform_record
        if is_waveform_record(path):
            self.waveform_panel.load_record_path(path)
            self.activate_task('waveform'); return
        from arraylab.features.mimo import is_mimo_record
        if is_mimo_record(path):
            self.mimo_panel.load_record_path(path)
            self.activate_task('mimo'); return
        # Complete validation and object construction before touching live widgets.
        document = load_history(path)
        kind, selected = document["kind"], document["selected"]
        parameters = prepared = None
        if kind == "linear":
            records = [pattern_from_record(record) for record in document["records"]]
            if records[selected].result is not None:
                parameters = records[selected].result.parameters
        elif kind == "feed":
            records = document["records"]
            selected_report = records[selected]["report"]
            prepared = self.feed_panel.prepare_restore_report(selected_report)
        else:
            raise ValueError("此文件没有可恢复的直线阵或馈电历史。")
        sequence = _history_sequence(records)

        if kind == "linear":
            if parameters is not None:
                self.apply_parameters(parameters)
            offset = len(self.datasets)
            self.datasets.extend(records)
            self.sequence = max(self.sequence, sequence)
            self._persist_pending_history()
            with QSignalBlocker(self.selection):
                self.selection.clear()
                self.selection.addItems([record.label for record in self.datasets])
                self.selection.setCurrentIndex(offset + selected)
            self.keep_history.setChecked(True)
            self.activate_task(kind)
            self.update_selection()
        else:
            panel = self.feed_panel
            panel._changing = True
            try:
                panel.history = records
                panel.sequence = sequence
                with QSignalBlocker(panel.history_selection):
                    panel.history_selection.clear()
                    panel.history_selection.addItems([record["label"] for record in records])
                    panel.history_selection.setCurrentIndex(selected)
                panel.update_history_available()
                self.activate_task(kind)
                panel.restore_report(selected_report, prepared=prepared)
            finally:
                panel._changing = False
        self.last_directory = str(Path(path).parent)
        self.change_workbench_context(self.tabs.currentIndex())
        verb = "已追加" if kind == "linear" else "已恢复"
        message = f"{verb}{len(records)}条历史快照：{Path(path).name}；显示保存时的结果，未重新计算。"
        if kind == "linear" and self._history_warning:
            message += " " + self._history_warning
        if kind == "linear":
            self.state_label.setText(message + (" 所选外部曲线不含参数，当前输入已保留。" if parameters is None else ""))
        self.statusBar().showMessage(message)
        self.record_message(message)

    def save_history_dialog(self):
        kind = self.active_task()
        if kind in getattr(self,'simulation_panels',{}):
            self.simulation_panels[kind].save_record_dialog();return
        if kind == 'waveform':
            self.waveform_panel.save_record_dialog(); return
        if kind == 'mimo':
            self.mimo_panel.save_record_dialog(); return
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
        path, _ = QFileDialog.getOpenFileName(self, "打开完整计算记录（不重新计算）", self.last_directory,
                                            "ArrayLab计算记录 (*.json *.npz)",
                                            options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            try:
                new_record=Path(path).suffix.lower()=='.npz'
                if not new_record:
                    from arraylab.shared.simulation_io import read_json
                    marker=read_json(path)
                    new_record=isinstance(marker,dict) and marker.get('format')=='arraylab.simulation.export'
                if new_record:self.load_simulation_record_async(path)
                else:self.load_history_path(path)
            except (OSError, ValueError, OverflowError) as exc:
                QMessageBox.warning(self, "历史恢复失败", str(exc))

    def load_simulation_record_async(self,path):
        self._require_history_idle()
        from arraylab.application.simulation import load_record
        from .simulation_panel import SimulationWorker
        reader=SimulationWorker(lambda progress,cancel:load_record(path),self,cancellable=False)
        self._record_reader=reader;reader.finished.connect(self.complete_simulation_record_load)
        self.statusBar().showMessage('正在读取完整数值记录，保留原结果…');self.refresh_actions();reader.start()

    def complete_simulation_record_load(self):
        reader=self._record_reader
        try:
            if reader.error:raise ValueError(reader.error)
            report,arrays=reader.result;kind=report['feature']
            self.simulation_panels[kind].accept_result(report,arrays,restore_inputs=True);self.activate_task(kind)
        except Exception as exc:QMessageBox.warning(self,'记录未恢复，原结果保留',str(exc))
        finally:
            self._record_reader=None;reader.deleteLater();self.refresh_actions()
            if getattr(self,'_record_close_pending',False):self._record_close_pending=False;QTimer.singleShot(0,self.close)
