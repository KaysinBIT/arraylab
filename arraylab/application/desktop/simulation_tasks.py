"""Compose new capability panels and their existing-array handoffs."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTreeWidgetItem,QPushButton,QMenu,QMessageBox
from arraylab.shared.qt.icons import workbench_icon
from .task_registry import task_spec


def attach_simulations(window):
    from .simulation_panel import SimulationPanel
    window.simulation_panels={}
    with_signals=window.tabs.blockSignals(True)
    try:
        for feature in ('phased','equation','pulse'):
            panel=SimulationPanel(window,feature);window.simulation_panels[feature]=panel
            setattr(window,feature+'_panel',panel)
            spec=task_spec(feature)
            window.tabs.addTab(panel,workbench_icon(feature),spec.title)
            window.inspector_stack.addWidget(panel.parameter_widget)
            item=QTreeWidgetItem(window.project_tree_root,[spec.title]);item.setIcon(0,workbench_icon(feature))
            item.setData(0,Qt.ItemDataRole.UserRole,('view',spec.page));setattr(window,spec.tree_attr,item)
            if feature in ('equation','pulse'):
                button=QPushButton('导入已算阵列');menu=QMenu(button)
                for source,title in [('linear','所选直线阵'),('sparse','所选二维稀疏阵'),('mimo','当前MIMO阵列'),('phased','当前相控阵结果')]:
                    for side in ('tx','rx'):
                        action=menu.addAction(title+' → '+side.upper())
                        action.triggered.connect(lambda checked=False,d=feature,s=side,k=source:import_array(window,d,s,k))
                button.setMenu(menu);panel.add_integration_action(button,0)
                panel.array_import_button=button
                panel.array_import_actions={source:[] for source in ('linear','sparse','mimo','phased')}
                for i,action in enumerate(menu.actions()):
                    panel.array_import_actions[('linear','sparse','mimo','phased')[i//2]].append(action)
            if feature=='equation':
                button=QPushButton('导入已算FMCW波形');button.clicked.connect(lambda:import_waveform(window))
                panel.add_integration_action(button,1)
                panel.waveform_import_button=button
            if feature=='phased':
                button=QPushButton('将所选工况送入平面馈电')
                button.clicked.connect(lambda:send_to_planar(window));panel.add_integration_action(button,0)
                panel.planar_transfer_button=button
    finally:window.tabs.blockSignals(with_signals)


def refresh_handoffs(window):
    """Reflect available completed snapshots without evaluating their models."""
    panels=getattr(window,'simulation_panels',{})
    source=window.selected_dataset()
    ready={'linear':source is not None and source.result is not None,
           'sparse':window.sparse_panel.selected_result() is not None,
           'mimo':bool(window.mimo_panel.report),
           'phased':bool(getattr(panels.get('phased'),'report',None))}
    busy=any(getattr(p,'_busy',False) or getattr(p,'thread',None) is not None for p in
             (window.sparse_panel,window.subarray_panel,window.mimo_panel,window.waveform_panel,*panels.values()))
    busy=busy or getattr(window,'_record_reader',None) is not None
    for panel in panels.values():
        for key,actions in getattr(panel,'array_import_actions',{}).items():
            for action in actions:action.setEnabled(not busy and ready[key])
        if hasattr(panel,'array_import_button'):
            panel.array_import_button.setEnabled(not busy and any(ready.values()))
            panel.array_import_button.setToolTip('从已成功计算的阵列结果读取几何；先在来源模块计算，再选择Tx或Rx。')
        if hasattr(panel,'waveform_import_button'):
            panel.waveform_import_button.setEnabled(not busy and bool(window.waveform_panel.report))
        if hasattr(panel,'planar_transfer_button'):
            panel.planar_transfer_button.setEnabled(not busy and bool(panel.report))


def import_array(window,destination,side,source):
    panel=window.simulation_panels[destination]
    if panel._busy:return
    try:
        from arraylab.application.bridges.radar import import_array as prepare
        configuration,note=prepare(window,destination,side,source)
        panel.apply_configuration(configuration);panel.state.setText(note)
    except (ValueError,TypeError,KeyError,AttributeError) as exc:QMessageBox.warning(window,'阵列未导入',str(exc))


def import_waveform(window):
    if window.equation_panel._busy:return
    try:
        from arraylab.application.bridges.radar import import_waveform as prepare
        configuration,note=prepare(window)
        window.equation_panel.apply_configuration(configuration);window.equation_panel.state.setText(note)
    except (ValueError,TypeError,KeyError,AttributeError) as exc:QMessageBox.warning(window,'波形未导入',str(exc))


def send_to_planar(window):
    if window.phased_panel._busy:return
    try:
        from arraylab.application.bridges.radar import apply_phased_to_planar
        prepared=apply_phased_to_planar(window)
        window.planar_panel.apply_configuration(prepared['parameters'],prepared['settings'])
        window.activate_task('planar');window.planar_panel.state.setText(prepared['note'])
    except (ValueError,TypeError,KeyError,AttributeError) as exc:QMessageBox.warning(window,'未送入馈电设计',str(exc))
