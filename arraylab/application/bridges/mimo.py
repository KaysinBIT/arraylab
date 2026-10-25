"""Native MIMO task wiring and explicit array-snapshot handoff."""
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTreeWidgetItem, QMessageBox
from arraylab.shared.qt.icons import workbench_icon


def attach_mimo(window):
    from arraylab.features.mimo import MIMOPanel
    window.tabs.blockSignals(True)
    try:
        window.mimo_panel = MIMOPanel(window)
        window.tabs.addTab(window.mimo_panel, workbench_icon('mimo'), '稀疏MIMO分析')
        window.inspector_stack.addWidget(window.mimo_panel.parameter_widget)
        item = QTreeWidgetItem(window.project_tree_root, ['稀疏MIMO分析'])
        item.setData(0, Qt.ItemDataRole.UserRole, ('view', window.task_page('mimo')))
        item.setIcon(0, workbench_icon('mimo'))
        window.mimo_tree_item = item
    finally:
        window.tabs.blockSignals(False)


class MIMOWorkflowActions:
    def import_selected_to_mimo(self, side):
        if side not in ('tx', 'rx'):
            raise ValueError('MIMO导入侧须为Tx或Rx。')
        panel = self.mimo_panel
        if panel._busy or panel.thread is not None:
            self.statusBar().showMessage('MIMO计算进行中，完成后再导入阵列。'); return
        try:
            kind = self.active_task()
            if kind == 'linear':
                snapshot = self.selected_linear_array_snapshot()
            elif kind == 'sparse':
                snapshot = self.sparse_panel.selected_array_snapshot()
            else:
                raise ValueError('请从直线阵或二维稀疏的已计算结果导入。')
            if snapshot['element_model'] != 'isotropic':
                raise ValueError('MIMO首版只支持理想点源；当前稀疏单元模型不能直接导入。')
            panel.set_aperture(side, snapshot['positions_lambda'], snapshot['weights'],
                               snapshot['frequency_ghz'], snapshot['origin'])
            self.activate_task('mimo')
            self.statusBar().showMessage(f'已导入{side.upper()}坐标与相对幅度；目标指向在MIMO页设置，按F5计算。')
        except (ValueError, OSError, ArithmeticError) as exc:
            QMessageBox.warning(self, 'MIMO阵列导入', str(exc))
