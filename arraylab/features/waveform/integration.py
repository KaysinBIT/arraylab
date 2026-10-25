"""Attach the native waveform task without any source-project dependency."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTreeWidgetItem
from arraylab.shared.qt.icons import workbench_icon


def attach_waveform(window):
    from arraylab.features.waveform.desktop import WaveformPanel
    window.tabs.blockSignals(True)
    try:
        window.waveform_panel = WaveformPanel(window)
        window.tabs.addTab(window.waveform_panel, workbench_icon('waveform'), '雷达波形设计')
        window.inspector_stack.addWidget(window.waveform_panel.parameter_widget)
        item = QTreeWidgetItem(window.project_tree_root, ['雷达波形设计'])
        item.setData(0, Qt.ItemDataRole.UserRole, ('view', window.task_page('waveform')))
        item.setIcon(0, workbench_icon('waveform')); window.waveform_tree_item = item
    finally:
        window.tabs.blockSignals(False)
