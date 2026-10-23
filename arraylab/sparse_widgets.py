"""Integrated sparse-array workspace and checked handoff to planar feed design."""
from datetime import datetime
from pathlib import Path

import matplotlib
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QApplication, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton

from . import __version__
from .sparse2d.gui import Window
from .sparse2d.headless import save_bundle
from .sparse2d.runtime_paths import document_directory
from .sparse_bridge import planar_from_sparse
from .workbench import STYLE


class SparseWindow(Window):
    planar_requested = Signal(object, object, object)

    def __init__(self, main_window):
        with matplotlib.rc_context():
            super().__init__()
        self.setParent(main_window, Qt.WindowType.Window)
        self.setWindowTitle(f'ArrayLab {__version__} · 二维稀疏阵列')
        self.setStyleSheet(STYLE + "QLabel#title {font-size:20px;font-weight:600;color:#185986;} QTableWidget::item:selected {background:#d5e8f6;color:#174969;}")
        self.setMinimumSize(1100, 740)
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1460, screen.width() - 60), min(950, screen.height() - 80))
        for label in self.findChildren(QLabel):
            if '独立研究 Demo' in label.text():
                label.setText('稀疏选点 · 中心禁布 · 最小间距 · 共同角频综合 · 选中布局可送入平面阵电气设计')
        row = QHBoxLayout()
        self.planar_button = QPushButton('将所选理想布局送入平面阵电气页')
        self.planar_button.setToolTip('传递所选方案和工况的毫米坐标、相对幅度及扫描角；新建50 Ω并联初始网络，连续移相。原量化码和非点源单元不迁移。')
        self.planar_button.clicked.connect(self.send_to_planar)
        row.addWidget(self.planar_button); row.addStretch()
        self.centralWidget().layout().insertLayout(0, row)
        self.statusBar().showMessage('ArrayLab 二维稀疏模块 · 请设置参数后计算')
        self.update_workflow()

    def update_workflow(self):
        super().update_workflow()
        if hasattr(self, 'planar_button'):
            self.planar_button.setEnabled(not self._busy and self.bundle is not None and self.table.currentRow() >= 0)

    def receive(self, bundle):
        bundle['integration'] = {'arraylab_version': __version__, 'source_engine': 'Sparse2D 1.3.0'}
        super().receive(bundle)

    def send_to_planar(self):
        if self._busy or self.bundle is None:
            return
        try:
            p, settings, provenance = planar_from_sparse(self.bundle, self.table.currentRow(), self.case_selector.currentIndex())
        except (ValueError, TypeError, KeyError) as exc:
            QMessageBox.warning(self, '无法传递此方案', str(exc)); return
        self.planar_requested.emit(p, settings, provenance)
        self.hide()

    def export_path(self, path):
        if not self.bundle:
            raise ValueError('请先完成稀疏计算。')
        self.last_export = save_bundle(self.bundle, path, report=True)

    def export(self):
        if not self.bundle or self._busy:
            return
        parent = QFileDialog.getExistingDirectory(self, '选择稀疏结果父目录', str(document_directory()))
        if not parent:
            return
        target = Path(parent) / datetime.now().strftime('sparse2d_%Y%m%d_%H%M%S_%f')
        self._job_kind = 'export'; self.set_busy(True)
        try:
            self.export_path(target)
            self.statusBar().showMessage('已导出完整离线报告：' + str(target / 'report.html'))
        except Exception as exc:
            QMessageBox.critical(self, '导出失败', str(exc))
        finally:
            self._job_kind = None; self.set_busy(False)
