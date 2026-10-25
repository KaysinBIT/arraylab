"""Exercise sparse tree/view/selection synchronization in an isolated window."""
from copy import deepcopy
from pathlib import Path
import json
from unittest.mock import patch
import uuid

from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QTimer, Qt
from PySide6.QtTest import QTest

from .sparse2d.core import Config, run
from .sparse_history_store import SparseHistoryStore, snapshot_bytes
from .window import MainWindow


def exercise_sparse_navigation(app, output, report, check):
    output = Path(output) / ('sparse-navigation-' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    path = output / '导航 历史.sqlite3'
    bundle = run(Config(nx=4, ny=4, k=8, iterations=1, opt_grid=25, eval_grid=81, phase_bits=3), workers=1)
    second = deepcopy(bundle)
    second['audit_note'] = 'second immutable snapshot'
    store = SparseHistoryStore(path)
    first = store.append(bundle, '综合 A · 4×4 · K=8')
    last = store.append(second, '综合 B · 4×4 · K=8')
    original_payloads = {entry['id']: snapshot_bytes(store.load_record(entry['id'])['bundle']) for entry in (first, last)}
    loads = []
    real_load = SparseHistoryStore.load_record

    def counted_load(storage, record_id):
        loads.append(record_id)
        return real_load(storage, record_id)

    def settle(delay=45):
        loop = QEventLoop(); QTimer.singleShot(delay, loop.quit); loop.exec(); app.processEvents()

    def selected_payload(window):
        item = window.tree.currentItem()
        return tuple(item.data(0, Qt.ItemDataRole.UserRole) or ()) if item else ()

    def click_item(window, item):
        parent = item.parent()
        while parent is not None:
            parent.setExpanded(True); parent = parent.parent()
        window.tree.scrollToItem(item); settle()
        point = window.tree.visualItemRect(item).center()
        QTest.mouseClick(window.tree.viewport(), Qt.MouseButton.LeftButton, pos=point); settle()

    old_quit = app.quitOnLastWindowClosed(); app.setQuitOnLastWindowClosed(False)
    window = None
    details = {'qt_platform': app.platformName(), 'checks_scope': 'Qt event-driven navigation; isolated archive', 'view_nodes': []}
    try:
        with patch.object(SparseHistoryStore, 'load_record', counted_load):
            window = MainWindow(auto_compute=False, sparse_history_path=path)
            window.show(); settle()
            panel = window.sparse_panel
            check(not loads and panel.bundle is None and len(panel.history_entries) == 2,
                  'sparse_navigation_startup_reads_history_index_without_payloads')
            click_item(window, window._sparse_history_nodes[first['id']])
            check(loads == [first['id']] and panel.current_history_id == first['id'],
                  'sparse_navigation_direct_history_click_loads_only_requested_record')
            check(selected_payload(window) == ('sparse_history', first['id'])
                  and panel.config_from_inputs() == Config(**bundle['config']),
                  'sparse_navigation_history_selection_and_parameters_match')
            before = len(loads)
            window.activate_sparse_tree_item(window._sparse_history_nodes[first['id']], 0)
            check(len(loads) == before, 'sparse_navigation_double_activation_does_not_reload_clean_snapshot')
            panel.inputs['nx'].setValue(6)
            window.activate_sparse_tree_item(window._sparse_history_nodes[first['id']], 0)
            check(len(loads) == before + 1 and panel.config_from_inputs().nx == 4,
                  'sparse_navigation_reactivation_restores_dirty_parameters')
            scientific = snapshot_bytes(panel.bundle)
            current = panel.bundle

            with patch.object(panel, 'start', side_effect=AssertionError('Navigation must not synthesize')), \
                 patch.object(panel, 'apply_weighting', side_effect=AssertionError('Navigation must not apply weights')):
                for key, item in window._sparse_nav_nodes.items():
                    if key[0] == 'sparse_preview': continue
                    click_item(window, item)
                    if key[0] == 'sparse_view':
                        good = panel.tabs.currentIndex() == 0 and panel.view.currentData() == key[1]
                    else:
                        good = panel.tabs.currentIndex() == key[1]
                    check(good and selected_payload(window) == key,
                          'sparse_navigation_tree_opens_' + str(key[1]))
                    details['view_nodes'].append(key)
                check(panel.bundle is current and panel.thread is None and not panel._busy
                      and snapshot_bytes(panel.bundle) == scientific,
                      'sparse_navigation_views_preserve_scientific_snapshot_without_new_job')

            panel.tabs.setCurrentIndex(0); panel.view.setCurrentIndex(panel.view.findData('cuts')); settle()
            check(selected_payload(window) == ('sparse_view', 'cuts'), 'sparse_navigation_middle_view_syncs_left_node')
            panel.tabs.setCurrentIndex(1); settle()
            check(selected_payload(window) == ('sparse_tab', 1) and not window.image_button.isEnabled(),
                  'sparse_navigation_table_tab_syncs_tree_and_picture_actions')
            target = (panel.table.currentRow() + 1) % len(panel.bundle['results'])
            panel.scheme_selector.setCurrentIndex(target); settle()
            check(selected_payload(window) == ('sparse_scheme', target)
                  and panel.case_selector.currentIndex() == panel.selected_result()['metrics']['worst_case_index'],
                  'sparse_navigation_middle_scheme_syncs_left_and_worst_case')
            target = (target + 1) % len(panel.bundle['results'])
            click_item(window, window._sparse_scheme_nodes[target])
            check(panel.table.currentRow() == target and panel.scheme_selector.currentIndex() == target
                  and panel.tabs.currentIndex() == 0 and panel.view.currentData() == 'layout',
                  'sparse_navigation_left_scheme_selects_same_result')

            window.navigate_sparse('sparse_view', 'layout'); panel.preview_element(); settle()
            check(panel.preview_active and window.tree.currentItem() is window.sparse_preview_tree_item,
                  'sparse_navigation_preview_has_distinct_unsolved_identity')
            click_item(window, window._sparse_nav_nodes[('sparse_view', 'layout')])
            check(not panel.preview_active and panel.view.currentData() == 'layout'
                  and panel.bundle is current and '已返回已计算快照' in panel.state.text(),
                  'sparse_navigation_same_view_leaves_preview_for_saved_result')

            panel.set_busy(True); old_id = panel.current_history_id; old_row = panel.table.currentRow()
            window.navigate_sparse('sparse_history', last['id'])
            window.navigate_sparse('sparse_scheme', (old_row + 1) % len(panel.bundle['results']))
            check(panel.current_history_id == old_id and panel.table.currentRow() == old_row
                  and panel.bundle is current, 'sparse_navigation_busy_guard_preserves_current_result')
            panel.set_busy(False)
            click_item(window, window._sparse_history_nodes[last['id']])
            check(panel.current_history_id == last['id'] and panel.bundle.get('audit_note') == 'second immutable snapshot',
                  'sparse_navigation_can_open_another_saved_run')
            window.navigate_sparse('sparse_view', 'pattern'); settle(220)
            window.activate_task('linear'); window.activate_task('sparse'); settle()
            check(selected_payload(window) == ('sparse_view', 'pattern') and panel.current_history_id == last['id'],
                  'sparse_navigation_return_to_task_keeps_active_result_view')
            for entry in (first, last):
                check(snapshot_bytes(real_load(store, entry['id'])['bundle']) == original_payloads[entry['id']],
                      'sparse_navigation_does_not_modify_archived_payload_' + str(entry['id']))
            window.sparse_schemes_tree_item.setExpanded(False)
            window.sparse_more_tree_item.setExpanded(False)
            for width, height in ((1440, 900), (1100, 740)):
                window.resize(width, height); settle()
                check(window.grab().save(str(output / f'sparse-navigation-{width}.png')),
                      'sparse_navigation_screenshot_' + str(width))
            details['loads'] = loads
    finally:
        if window is not None:
            window.sparse_panel.set_busy(False)
            window.close()
            for timer in window.findChildren(QTimer): timer.stop()
            window.deleteLater(); QCoreApplication.sendPostedEvents(window, QEvent.Type.DeferredDelete)
        app.setQuitOnLastWindowClosed(old_quit); app.processEvents()
    report['sparse_navigation'] = details
    (output / 'navigation-details.json').write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding='utf-8')
