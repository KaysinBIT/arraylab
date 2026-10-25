from dataclasses import asdict
from pathlib import Path
import math
import os
import time
from datetime import datetime

import numpy as np
from PySide6.QtCore import Qt, QTimer, QSignalBlocker, QStandardPaths
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHeaderView, QMainWindow, QMessageBox, QTableView,
)

from arraylab import __version__
from arraylab.features.linear import ArrayParameters, InputError, calculate
from arraylab.features.linear import load_configuration, save_configuration
from arraylab.features.linear import read_pattern, write_pattern
from arraylab.features.linear import NumericTableModel, PatternSet


from arraylab.application.desktop.workbench import STYLE, build_workbench
from arraylab.shared.qt.icons import workbench_icon


from arraylab.application.desktop.workflow import WorkflowActions
from arraylab.application.desktop.history import HistoryActions
from arraylab.features.linear import LinearAnalysisActions
from arraylab.features.sparse import SparseNavigationActions
from arraylab.application.bridges.mimo import MIMOWorkflowActions, attach_mimo
from arraylab.features.waveform import attach_waveform


class MainWindow(LinearAnalysisActions, WorkflowActions, HistoryActions, SparseNavigationActions, MIMOWorkflowActions, QMainWindow):
    def __init__(self, auto_compute=True, *, history_path=None, sparse_history_path=None):
        super().__init__()
        self.setWindowTitle(f"ArrayLab {__version__} · 阵列分析")
        available = self.screen().availableGeometry()
        self.resize(min(1440, available.width()-24), min(900, available.height()-60))
        self.setMinimumSize(1100, 640)
        self.datasets: list[PatternSet] = []
        self.sequence = 0
        self._history_store = None
        self._history_path = None
        self._history_warning = ""
        self._history_loading = True
        self._history_requested = bool(auto_compute) or history_path is not None
        self._persisted_selection = None
        self.last_directory = str(Path.home())
        self._changing = False
        self._spacing_unit = "lambda"
        self.setStyleSheet(STYLE)
        self.build_ui()
        self.build_menus()
        self.feed_panel.connect_shared_steering()
        self.reset_parameters()
        self.update_selection()
        self.activate_task("home")
        self._initialize_linear_history(history_path)
        sparse_directory = os.environ.get('ARRAYLAB_SPARSE_HISTORY_DIR')
        sparse_path = (Path(sparse_history_path) if sparse_history_path is not None else
            Path(sparse_directory) / 'sparse-history.sqlite3' if sparse_directory else
            Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)) / 'history' / 'sparse-history.sqlite3')
        self.sparse_panel.initialize_history(sparse_path if auto_compute or sparse_history_path is not None else None)
        self.sync_sparse_navigation()

    def build_ui(self):
        build_workbench(self)
        from arraylab.features.subarray import attach_subarray
        attach_subarray(self)
        attach_mimo(self)
        attach_waveform(self)
        for canvas in (self.plot.canvas, self.feed_panel.pattern_plot.canvas,
                       self.planar_panel.canvas, self.sparse_panel.canvas,
                       self.subarray_panel.canvas, self.mimo_panel.canvas, self.waveform_panel.canvas):
            canvas.setMinimumSize(480, 360)



    def sparse_active(self):
        return hasattr(self, 'sparse_panel') and self.tabs.currentWidget() is self.sparse_panel


    def receive_sparse_layout(self, parameters, settings, provenance):
        self.planar_panel.apply_configuration(parameters, settings)
        self.planar_panel.sparse_source = provenance
        self.tabs.setCurrentIndex(5)
        note = (f"已接收稀疏方案：{provenance['source_method']} · {parameters.elements}阵元。"
                "坐标/相对幅度已载入；50 Ω并联初值、连续扫描补偿，尚未重新生成。旧结果保持不变。")
        self.planar_panel.state.setText(note); self.record_message(note)
        self.show(); self.raise_(); self.activateWindow()

    def closeEvent(self, event):
        waveform = getattr(self, 'waveform_panel', None)
        if waveform is not None and (waveform.thread is not None or waveform._busy):
            if waveform.thread is not None:
                self._waveform_close_pending = True; waveform.cancel()
                self.statusBar().showMessage('正在停止波形计算，随后关闭窗口。')
            else:
                self.statusBar().showMessage('波形结果正在保存或导出，请完成后关闭。')
            event.ignore(); return
        mimo = getattr(self, 'mimo_panel', None)
        if mimo is not None and (mimo.thread is not None or mimo._busy):
            if mimo.thread is not None:
                self._mimo_close_pending = True; mimo.cancel()
                self.statusBar().showMessage('正在停止MIMO计算，随后关闭窗口。')
            else:
                self.statusBar().showMessage('MIMO结果正在保存或导出，请完成后关闭。')
            event.ignore(); return
        subarray = getattr(self, 'subarray_panel', None)
        if subarray is not None and (subarray.thread is not None or subarray._busy):
            self.statusBar().showMessage('子阵任务仍在计算或导出，请完成后关闭。')
            event.ignore(); return
        sparse = getattr(self, 'sparse_panel', None)
        if sparse is not None and (sparse.thread is not None or sparse._busy):
            if sparse.thread is not None:
                self._sparse_close_pending = True; sparse.cancel()
                self.statusBar().showMessage('正在停止二维稀疏计算并回收资源，随后关闭窗口。')
            else:
                self.statusBar().showMessage('二维稀疏结果正在导出，请完成后关闭。')
            event.ignore(); return
        self._persist_pending_history()
        self._persist_history_selection()
        if not self.sparse_panel.can_close_history():
            event.ignore(); return
        if self._history_requested and any(getattr(data, "_archive_id", None) is None for data in self.datasets):
            answer = QMessageBox.question(self, "计算历史尚未自动保存",
                "自动保存未成功，部分结果目前仅在本次会话内。\n"
                "请选择“否”返回并使用“保存记录”备份；选择“是”将退出并放弃尚未自动保存的结果。\n\n"
                + self._history_warning + "\n\n仍要退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore(); return
        super().closeEvent(event)








    def restore_workbench(self):
        for item, area in [(self.nav_dock, Qt.DockWidgetArea.LeftDockWidgetArea),
                           (self.inspector_dock, Qt.DockWidgetArea.RightDockWidgetArea),
                           (self.messages_dock, Qt.DockWidgetArea.BottomDockWidgetArea)]:
            item.setFloating(False)
            self.addDockWidget(area, item)
            item.show()
        self.resizeDocks([self.nav_dock, self.inspector_dock], [190, 360], Qt.Orientation.Horizontal)
        self.resizeDocks([self.messages_dock], [135], Qt.Orientation.Vertical)
        if self.active_task()=='home':self.inspector_dock.hide();self.messages_dock.hide()
        self.statusBar().showMessage("工作台布局已恢复。")

    def record_message(self, text):
        if text and hasattr(self, "message_log"):
            self.message_log.appendPlainText(f"{datetime.now():%H:%M:%S}  {text}")


    def make_table(self):
        view = QTableView()
        view.setAlternatingRowColors(True)
        view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        return view



















    def select_result_tab(self, current, previous):
        payload = current.data(0, Qt.ItemDataRole.UserRole) if current else None
        if not payload:
            return
        kind, index = payload
        if kind in ('sparse_view', 'sparse_tab', 'sparse_scheme', 'sparse_history'):
            self.navigate_sparse(kind, index)
            return
        if kind == "view":
            self.show_view(index)
        elif kind == "parameters":
            self.focus_parameters()
        elif kind == "sparse":
            self.open_sparse()
        elif kind == "result":
            self.selection.setCurrentIndex(index)
            self.show_view(0)







    def show_about(self):
        QMessageBox.about(self, "关于 ArrayLab", f"ArrayLab {__version__}\n直线阵 / 平面阵 / 二维稀疏 / 馈电电气设计\n\n"
                          "Python / PySide6 / NumPy / SciPy / Matplotlib\n"
                          "各页保留各自模型边界；无互耦、无全波求解，馈线物理尺寸暂缓。\n"
                          "本软件为独立实现，与PCAAD原软件无隶属关系。\n"
                          "项目源码按MIT许可；依赖许可见随包THIRD_PARTY_NOTICES。")
