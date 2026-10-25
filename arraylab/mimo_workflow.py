"""Native MIMO task wiring and explicit array-snapshot handoff."""
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTreeWidgetItem, QMessageBox
from .icons import workbench_icon


def attach_mimo(window):
    from .mimo_widgets import MIMOPanel
    window.tabs.blockSignals(True)
    try:
        window.mimo_panel = MIMOPanel(window)
        window.tabs.addTab(window.mimo_panel, workbench_icon('mimo'), '稀疏MIMO分析')
        window.inspector_stack.addWidget(window.mimo_panel.parameter_widget)
        item = QTreeWidgetItem(window.project_tree_root, ['稀疏MIMO分析'])
        item.setData(0, Qt.ItemDataRole.UserRole, ('view', 9))
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
                selected = self.selected_dataset()
                if selected is None or selected.result is None:
                    raise ValueError('请先选择具有阵列参数的已计算直线阵。')
                result = selected.result; frequency = result.parameters.frequency_ghz
                coordinates = np.column_stack((result.positions_mm / result.parameters.wavelength_mm,
                                                np.zeros(result.parameters.elements)))
                weights = np.ones(result.parameters.elements)
                origin = '直线阵：' + selected.label
            elif kind == 'sparse':
                source = self.sparse_panel
                if source._busy or source.thread is not None:
                    raise ValueError('稀疏计算进行中，完成后再导入阵列。')
                result = source.selected_result()
                if result is None:
                    raise ValueError('请先选择一项已计算的稀疏方案。')
                config = source.bundle['config']
                if config['element_model'] != 'isotropic':
                    raise ValueError('MIMO首版只支持理想点源；当前稀疏单元模型不能直接导入。')
                case = source.bundle['scenarios'][max(0, source.case_selector.currentIndex())]
                frequency = case['frequency_ghz']
                coordinates = np.asarray(result['xy']) * frequency / config['reference_ghz']
                weights = np.asarray(result['weights'])
                from .sparse2d.plots import result_name
                origin = '二维稀疏：' + result_name(result)
            else:
                raise ValueError('请从直线阵或二维稀疏的已计算结果导入。')
            panel.set_aperture(side, coordinates, weights, frequency, origin)
            self.activate_task('mimo')
            self.statusBar().showMessage(f'已导入{side.upper()}坐标与相对幅度；目标指向在MIMO页设置，按F5计算。')
        except (ValueError, OSError, ArithmeticError) as exc:
            QMessageBox.warning(self, 'MIMO阵列导入', str(exc))
