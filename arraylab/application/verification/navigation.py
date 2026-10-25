"""Focused navigation acceptance; never synthesize unrelated workbenches."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from PySide6.QtCore import QEventLoop, QTimer, Qt, QSignalBlocker
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QInputDialog, QLineEdit,
                              QPlainTextEdit, QMessageBox)


TASKS = ('linear', 'feed', 'planar', 'sparse', 'subarray', 'mimo', 'waveform')


def settle(app, milliseconds=8):
    loop = QEventLoop(); QTimer.singleShot(milliseconds, loop.quit); loop.exec()
    app.processEvents()


def input_signature(window):
    """Editable scientific controls only; selected tabs are navigation state."""
    host = window.inspector_stack
    rows = []
    for kind, read in ((QLineEdit, lambda w: w.text()), (QPlainTextEdit, lambda w: w.toPlainText()),
                       (QCheckBox, lambda w: w.isChecked()), (QComboBox, lambda w: w.currentIndex())):
        rows.extend((id(control), read(control)) for control in host.findChildren(kind))
    return sorted(rows)


def report_signature(window):
    return ([id(row) for row in window.datasets], id(window.sparse_panel.bundle),
            *[id(getattr(window, name+'_panel').report) for name in ('feed','planar','subarray','mimo','waveform')])


@contextmanager
def no_solvers(window):
    with ExitStack() as stack:
        mocks = []
        for owner, names in [(window, ('compute', 'compute_linear')),
                             (window.feed_panel, ('generate',)), (window.planar_panel, ('generate',)),
                             (window.sparse_panel, ('start',)), (window.subarray_panel, ('start',)),
                             (window.mimo_panel, ('start',)), (window.waveform_panel, ('start',))]:
            for name in names:
                mocks.append(stack.enter_context(patch.object(owner, name, side_effect=AssertionError('Navigation started a solver'))))
        for name in ('warning', 'critical', 'information', 'question'):
            stack.enter_context(patch.object(QMessageBox, name, side_effect=AssertionError('Navigation opened a modal dialog')))
        for name in ('getOpenFileName', 'getSaveFileName', 'getExistingDirectory'):
            stack.enter_context(patch.object(QFileDialog, name, side_effect=AssertionError('Navigation opened a file dialog')))
        stack.enter_context(patch.object(QInputDialog, 'getText', side_effect=AssertionError('Navigation opened an input dialog')))
        yield
        assert all(mock.call_count == 0 for mock in mocks)


def select_node(app, window, item):
    parent = item.parent()
    while parent is not None:
        parent.setExpanded(True); parent = parent.parent()
    # Clearing with blocked signals ensures selecting the current leaf is also
    # exercised through the real currentItemChanged connection.
    with QSignalBlocker(window.tree): window.tree.setCurrentItem(None)
    window.tree.setCurrentItem(item); window.tree.scrollToItem(item)
    settle(app)


def assert_target(window, key):
    task, section, target = key
    assert window.active_task() == task, key
    panel = getattr(window, task+'_panel', None)
    if section == 'result':
        if task == 'linear': assert window.tabs.currentIndex() == target, key
        elif task == 'feed': assert panel.result_tabs.currentIndex() == target, key
        else: assert panel.view.currentData() == target, key
    elif section == 'feed_pattern':
        assert panel.result_tabs.currentIndex() == 3 and panel.pattern_view.currentIndex() == target, key
    elif section == 'radar_parameter':
        assert panel.parameter_widget.currentIndex() == 5 and panel.radar_controls.pages.currentIndex() == target, key
    elif section == 'parameter':
        assert window.inspector_dock.isVisible(), key
        if task == 'feed' and target >= 0: assert panel.settings_tabs.currentIndex() == target, key
        elif task in ('planar','sparse'): assert panel.settings_tabs.currentIndex() == target, key
        elif task not in ('linear','feed'): assert panel.parameter_widget.currentIndex() == target, key
    elif section == 'history':
        assert panel.history_selection.currentIndex() == target, key


def check_all_leaves(app, window, output, report, check):
    before = input_signature(window); snapshots = report_signature(window)
    nodes = [(key, item) for key, item in window._task_nav_nodes.items() if key[1] != 'history']
    assert nodes
    with no_solvers(window):
        for key, item in nodes:
            select_node(app, window, item); assert_target(window, key)
            check(window.tree.currentItem() is item, 'navigation_selects_leaf_'+'_'.join(map(str,key)))
            check(input_signature(window) == before and report_signature(window) == snapshots,
                  'navigation_preserves_inputs_and_snapshots_'+'_'.join(map(str,key)))
    report['navigation']['leaf_count'] = len(nodes)


def check_reverse_and_refresh(app, window, output, report, check):
    with no_solvers(window):
        for task, values in [('planar',('geometry','sections')), ('subarray',('layout','cuts')),
                             ('mimo',('layout','unique')), ('waveform',('overview','gain_budget'))]:
            window.activate_task(task); panel = getattr(window, task+'_panel')
            for value in values:
                panel.view.setCurrentIndex(panel.view.findData(value)); settle(app)
                item = window._task_nav_nodes[(task,'result',value)]
                check(window.tree.currentItem() is item, 'navigation_reverse_result_'+task+'_'+value)
                window.refresh_actions(); window.sync_task_navigation(); settle(app)
                check(window.tree.currentItem() is item, 'navigation_refresh_keeps_result_'+task+'_'+value)
        window.activate_task('feed')
        for value in (0,2):
            window.feed_panel.result_tabs.setCurrentIndex(value); settle(app)
            check(window.tree.currentItem() is window._task_nav_nodes[('feed','result',value)],
                  'navigation_reverse_feed_tab_'+str(value))
        for task in ('feed','planar','subarray','mimo','waveform'):
            window.activate_task(task); panel = getattr(window, task+'_panel')
            tabs = panel.settings_tabs if task in ('feed','planar') else panel.parameter_widget
            target = 1
            tabs.setCurrentIndex(0); tabs.setCurrentIndex(target); settle(app)
            item = window._task_nav_nodes[(task,'parameter',target)]
            check(window.tree.currentItem() is item, 'navigation_reverse_parameter_'+task)
            window.refresh_actions(); window.sync_task_navigation(); settle(app)
            check(window.tree.currentItem() is item, 'navigation_refresh_keeps_parameter_'+task)
        window.navigate_task_node('waveform','radar_parameter',0); settle(app)
        window.waveform_panel.radar_controls.pages.setCurrentIndex(2); settle(app)
        item = window._task_nav_nodes[('waveform','radar_parameter',2)]
        check(window.tree.currentItem() is item, 'navigation_reverse_nested_radar_parameter')
        window.refresh_actions(); window.sync_task_navigation(); settle(app)
        check(window.tree.currentItem() is item, 'navigation_refresh_keeps_nested_radar_parameter')


def check_expansion(app, window, output, report, check):
    with no_solvers(window):
        for task in TASKS:
            item = getattr(window, task+'_tree_item'); item.setExpanded(False)
            select_node(app, window, item)
            check(item.isExpanded() and item.childCount() > 0 and window.active_task() == task,
                  'navigation_module_expands_'+task)
        window.linear_tree_item.setExpanded(True); window.planar_tree_item.setExpanded(True)
        window.mimo_tree_item.setExpanded(False)
        select_node(app, window, window.waveform_tree_item)
        check(window.linear_tree_item.isExpanded() and window.planar_tree_item.isExpanded()
              and not window.mimo_tree_item.isExpanded(), 'navigation_keeps_other_manual_expansion_states')


def check_legacy(app, window, output, report, check):
    from arraylab.features.linear import PatternSet
    # This is an imported three-point curve, not a synthetic antenna solution.
    if not window.datasets:
        window.add_dataset(PatternSet('导航验收导入曲线', np.array([-90.,0.,90.]), np.array([-20.,0.,-20.]), None))
    with no_solvers(window):
        for page, item in ((0,window.plot_tree_item), (1,window.data_tree_item), (3,window.model_tree_item)):
            select_node(app, window, item)
            check(window.tabs.currentIndex() == page and window.active_task() == 'linear',
                  'navigation_legacy_linear_view_'+str(page))
        select_node(app, window, window.history_tree_item.child(0))
        check(window.selection.currentIndex() == 0 and window.selected_dataset().label == '导航验收导入曲线',
              'navigation_legacy_linear_record')
        for payload in (('sparse_view','overview'),('sparse_view','cuts'),('sparse_tab',1)):
            item = window._sparse_nav_nodes[payload]; select_node(app, window, item)
            check(window.active_task() == 'sparse' and
                  (window.sparse_panel.view.currentData() == payload[1] if payload[0] == 'sparse_view'
                   else window.sparse_panel.tabs.currentIndex() == payload[1]),
                  'navigation_legacy_'+'_'.join(map(str,payload)))
        check(window.sparse_history_tree_item.parent() is window.sparse_tree_item
              and window.sparse_schemes_tree_item.parent() is window.sparse_tree_item,
              'navigation_keeps_sparse_history_and_scheme_roots')


def check_feed_history(app, window, output, report, check):
    window.activate_task('feed'); window.inputs['elements'].setText('4'); window.inputs['step_deg'].setText('2')
    panel = window.feed_panel
    if not panel.history:
        panel.generate(); settle(app)
    check(len(panel.history) == 1 and panel.report is not None, 'navigation_small_feed_fixture_created')
    reference = panel.history[0]['report']; before = deepcopy(reference)
    window.sync_task_navigation(); settle(app)
    item = window._task_nav_nodes[('feed','history',0)]
    check(panel.history[0]['label'] in item.text(0), 'navigation_feed_record_appears')
    with patch.object(QInputDialog, 'getText', return_value=('导航验收已改名',True)):
        panel.rename_history(); settle(app)
    item = window._task_nav_nodes[('feed','history',0)]
    check('导航验收已改名' in item.text(0), 'navigation_feed_rename_syncs')
    with no_solvers(window):
        window.inputs['elements'].setText('9')
        select_node(app, window, item)
        check(window.inputs['elements'].text() == '4' and panel.report is reference,
              'navigation_feed_record_restores_inputs_without_design')
        window.inputs['elements'].setText('7')
        window.tree.setFocus(); QTest.keyClick(window.tree, Qt.Key.Key_Return); settle(app)
        check(window.inputs['elements'].text() == '4' and panel.report is reference,
              'navigation_feed_same_record_enter_restores_again')
        check(panel.report == before and len(panel.history) == 1,
              'navigation_feed_record_selection_does_not_append_or_change_report')
    report['navigation']['small_feed_computations'] = 1


def check_layout(app, window, output, report, check, width=1100, height=640):
    window.resize(width,height)
    for task in TASKS: getattr(window,task+'_tree_item').setExpanded(True)
    item=window._task_nav_nodes[('waveform','radar_parameter',2)]
    with no_solvers(window): select_node(app,window,item)
    settle(app,70)
    viewport=window.tree.viewport().rect(); selected=window.tree.visualItemRect(item)
    check(not selected.isEmpty() and viewport.intersects(selected), f'navigation_selected_leaf_scrolls_into_view_{width}x{height}')
    check(window.nav_dock.geometry().right() < window.inspector_dock.geometry().left(),
          f'navigation_docks_do_not_overlap_{width}x{height}')
    visible=[]
    iterator=window.tree.itemAbove(item)
    if iterator is not None:
        box=window.tree.visualItemRect(iterator)
        if viewport.intersects(box): visible.append(box)
    iterator=window.tree.itemBelow(item)
    if iterator is not None:
        box=window.tree.visualItemRect(iterator)
        if viewport.intersects(box): visible.append(box)
    check(all(not box.intersects(selected) for box in visible), f'navigation_adjacent_rows_do_not_overlap_{width}x{height}')
    filename=Path(output)/f'navigation-{width}x{height}.png'
    check(window.grab().save(str(filename)), f'navigation_screenshot_{width}x{height}')
    report['navigation']['screenshots'].append(str(filename))


def check_linear_reverse_history(app, window, output, report, check):
    from arraylab.features.linear import PatternSet
    if not window.datasets:
        window.add_dataset(PatternSet('导航验收导入曲线',np.array([-90.,0.,90.]),np.array([-20.,0.,-20.]),None))
    with no_solvers(window):
        select_node(app,window,window.history_tree_item.child(0))
        window.tabs.setCurrentIndex(1);settle(app)
        check(window.tree.currentItem() is window.data_tree_item,
              'navigation_native_linear_tabs_override_history_preference')
        select_node(app,window,window.history_tree_item.child(0))
        window.show_view(3);settle(app)
        check(window.tree.currentItem() is window.model_tree_item,
              'navigation_linear_toolbar_overrides_history_preference')


def check_sparse_busy_preserves_parameter(app, window, output, report, check):
    panel=window.sparse_panel
    with no_solvers(window):
        window.navigate_task_node('sparse','parameter',0);settle(app)
        selected=window._task_nav_nodes[('sparse','parameter',0)]
        previous=panel._busy
        try:
            panel._busy=True;panel.navigation_changed.emit();settle(app)
            check(window.tree.currentItem() is selected,
                  'navigation_sparse_busy_signal_keeps_parameter_selection')
        finally:panel._busy=previous;panel.navigation_changed.emit();settle(app)
        panel.tabs.setCurrentIndex(0)
        panel.view.setCurrentIndex(panel.view.findData('overview'))
        panel.view.setCurrentIndex(panel.view.findData('cuts'));settle(app)
        check(window.tree.currentItem() is window._sparse_nav_nodes[('sparse_view','cuts')],
              'navigation_sparse_actual_result_change_updates_selection')


def check_feed_bad_record_rolls_back(app, window, output, report, check):
    panel=window.feed_panel;window.activate_task('feed')
    if not panel.history:
        window.inputs['elements'].setText('4');window.inputs['step_deg'].setText('2');panel.generate();settle(app)
    panel.history_selection.setCurrentIndex(0)
    original=panel.history[0]['report'];bad=deepcopy(original)
    bad['array_parameters']['elements']=-4
    panel.history.append({'label':'故障注入记录','report':bad})
    panel.history_selection.addItem('故障注入记录');window.sync_task_navigation();settle(app)
    try:
        with no_solvers(window),patch.object(QMessageBox,'warning',return_value=QMessageBox.StandardButton.Ok) as warning:
            select_node(app,window,window._task_nav_nodes[('feed','history',1)])
            check(warning.call_count>0 and panel.history_selection.currentIndex()==0 and panel.report is original,
                  'navigation_invalid_feed_record_keeps_original_result')
            check(window.tree.currentItem() is window._task_nav_nodes[('feed','history',0)],
                  'navigation_invalid_feed_record_highlights_actual_restored_item')
    finally:
        panel.history.pop();panel.history_selection.removeItem(1);window.sync_task_navigation()


def exercise_navigation(app, window, output, report, check):
    """Use a separate window with isolated stores; preserve caller snapshots."""
    from arraylab.application.desktop.window import MainWindow
    from arraylab.framework.parallel import requested_workers, set_workers
    output=Path(output)/('navigation-ui-'+uuid4().hex[:8]); output.mkdir(parents=True)
    details={'scope':'Task tree, views, existing records and parameters only',
             'qt_platform':app.platformName(), 'screenshots':[], 'directory':str(output),
             'uses_real_user_history':False, 'small_feed_computations':0}
    report['navigation']=details
    quit_before=app.quitOnLastWindowClosed();workers=requested_workers()
    app.setQuitOnLastWindowClosed(False);set_workers(1)
    probe=MainWindow(auto_compute=False,history_path=output/'linear.sqlite3',sparse_history_path=output/'sparse.sqlite3')
    probe.resize(1536,900);probe.show();settle(app)
    try:
        for helper in (check_all_leaves,check_reverse_and_refresh,check_expansion,check_legacy,check_feed_history,
                       check_linear_reverse_history,check_sparse_busy_preserves_parameter,check_feed_bad_record_rolls_back):
            helper(app,probe,output,report,check)
        for width,height in ((1100,640),(1536,900)):
            check_layout(app,probe,output,report,check,width,height)
    finally:
        probe.close();probe.deleteLater();app.processEvents()
        set_workers(workers);app.setQuitOnLastWindowClosed(quit_before)
    (output/'details.json').write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding='utf-8')
