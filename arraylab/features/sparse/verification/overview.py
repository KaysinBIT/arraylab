"""Focused Qt acceptance for the paired sparse layout and pattern view."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QTimer, Qt, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog

from arraylab.features.sparse.engine.core import Config, run
from arraylab.features.sparse.engine.weighting import add_weighting, Weighting
from arraylab.features.sparse.history_store import SparseHistoryStore, snapshot_bytes
from arraylab.application.desktop.window import MainWindow


def exercise(app, window, output, check, click, report):
    output = Path(output) / ('sparse-overview-' + uuid4().hex[:8])
    output.mkdir(parents=True)
    config = Config(nx=4, ny=4, k=8, iterations=1, opt_grid=25, eval_grid=81,
                    scan_angles='0,0;25,30', frequencies_ghz='9,10', phase_bits=3)
    bundle = run(config, workers=1)
    source = next(i for i, row in enumerate(bundle['results']) if row['name'] == 'Simulated annealing')
    weighted = add_weighting(deepcopy(bundle), source, Weighting(method='gaussian', sigma=.5), workers=1)
    path = output / '并排 历史.sqlite3'
    store = SparseHistoryStore(path)
    first = store.append(bundle, '未加权', view_state={'result_index': source, 'view': 'overview'})
    last = store.append(weighted, '高斯加权', view_state={'view': 'overview', 'case_index': 3})
    original = {item['id']: snapshot_bytes(store.load_record(item['id'])['bundle']) for item in (first, last)}
    probe = None
    old_quit = app.quitOnLastWindowClosed(); app.setQuitOnLastWindowClosed(False)
    details = {'scope': 'Native Qt paired-view interactions; isolated saved results',
               'directory': str(output), 'screenshots': []}

    def settle(milliseconds=100):
        loop = QEventLoop(); QTimer.singleShot(milliseconds, loop.quit); loop.exec(); app.processEvents()

    def current_pair(panel):
        panel.canvas.draw()
        axes = panel.figure.axes
        check(len(axes) == 3 and len(axes[1].images) == 1,
              'sparse_overview_two_panels_and_one_colorbar_' + str(len(report['checks'])))
        np.testing.assert_array_equal(axes[0].collections[-1].get_offsets(), panel.selected_result()['xy'])
        return axes[0], axes[1]

    try:
        probe = MainWindow(auto_compute=False, sparse_history_path=path)
        probe.resize(1440, 900); probe.show(); settle()
        panel = probe.sparse_panel
        check(panel.view.currentData() == 'overview' and panel.bundle is None,
              'sparse_overview_is_default_without_startup_calculation')
        probe.activate_task('sparse'); settle()
        check(panel.current_history_id == last['id'] and panel.view.currentData() == 'overview',
              'sparse_overview_restores_saved_paired_view')
        layout, pattern = current_pair(panel)
        check(np.allclose(layout.collections[-1].get_sizes(),
                          20 + 60 * panel.selected_result()['weights'] / panel.selected_result()['weights'].max()),
              'sparse_overview_layout_shows_selected_weighted_amplitudes')
        initial_xy = layout.collections[-1].get_offsets().copy()
        initial_image = pattern.images[0].get_array().copy()
        panel.case_selector.setCurrentIndex(0); settle()
        layout, pattern = current_pair(panel)
        check(np.array_equal(layout.collections[-1].get_offsets(), initial_xy)
              and not np.ma.allequal(pattern.images[0].get_array(), initial_image)
              and '9 GHz' in pattern.get_title(),
              'sparse_overview_case_changes_pattern_and_preserves_layout')
        panel.case_selector.setCurrentIndex(3); panel.phase_view.setCurrentIndex(panel.phase_view.findData('ideal')); settle()
        before = panel.figure.axes[1].images[0].get_array().copy()
        panel.phase_view.setCurrentIndex(panel.phase_view.findData('quantized')); settle()
        check(not np.ma.allequal(panel.figure.axes[1].images[0].get_array(), before),
              'sparse_overview_phase_control_updates_pattern')
        panel.pattern_view.setCurrentIndex(panel.pattern_view.findData('element')); settle()
        values = panel.figure.axes[1].images[0].get_array().compressed()
        check(np.allclose(values, 0) and np.array_equal(panel.figure.axes[0].collections[-1].get_offsets(), initial_xy),
              'sparse_overview_element_view_keeps_layout_and_shows_isotropic_field')
        panel.pattern_view.setCurrentIndex(panel.pattern_view.findData('total')); settle()

        with patch.object(panel, 'start', side_effect=AssertionError('View must not synthesize')) as start:
            probe.navigate_sparse('sparse_scheme', 0); settle()
            layout, pattern = current_pair(panel)
            check(len(layout.collections[-1].get_offsets()) == 16 and panel.view.currentData() == 'overview',
                  'sparse_overview_scheme_selection_updates_both_panels')
            probe.navigate_sparse('sparse_history', first['id']); settle()
            layout, pattern = current_pair(panel)
            check(len(layout.collections[-1].get_offsets()) == 8 and panel.current_history_id == first['id'],
                  'sparse_overview_history_selection_updates_both_panels')
            check(start.call_count == 0, 'sparse_overview_selection_never_starts_synthesis')

        # A native toolbar operation affects one axes; Home restores its limits.
        limits = layout.get_xlim(); other = pattern.get_xlim(); box = layout.get_window_extent()
        ratio = panel.canvas.device_pixel_ratio
        point = QPoint(int((box.x0 + box.width / 2) / ratio),
                       int(panel.canvas.height() - (box.y0 + box.height / 2) / ratio))
        panel.toolbar.pan()
        QTest.mousePress(panel.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
        QTest.mouseMove(panel.canvas, point + QPoint(25, 0), 30)
        QTest.mouseRelease(panel.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point + QPoint(25, 0))
        check(not np.allclose(layout.get_xlim(), limits) and np.allclose(pattern.get_xlim(), other),
              'sparse_overview_pan_left_does_not_move_right')
        panel.toolbar.pan(); panel.toolbar.home(); settle()
        check(np.allclose(layout.get_xlim(), limits), 'sparse_overview_toolbar_home_restores_layout')

        figure = panel.figure
        with patch('arraylab.sparse2d.plots._plot_response', side_effect=AssertionError('Resize must not sample field')) as sample:
            for width, height in ((1100, 740), (1440, 900)):
                probe.resize(width, height); settle(180); panel.canvas.draw()
                left, right = panel.figure.axes[:2]
                check(panel.figure is figure and left.get_window_extent().x1 < right.get_window_extent().x0
                      and min(left.get_window_extent().width, right.get_window_extent().width) / panel.canvas.device_pixel_ratio > 140,
                      'sparse_overview_resize_keeps_readable_pair_without_resampling_' + str(width))
                probe.results_scroll.ensureWidgetVisible(panel.canvas)
                settle()
                destination = output / f'sparse-overview-{width}.png'
                check(probe.grab().save(str(destination)), 'sparse_overview_screenshot_' + str(width))
                details['screenshots'].append(str(destination))
            check(sample.call_count == 0, 'sparse_overview_resize_never_samples_field_in_qt_callbacks')
        image_path = output / '并排 当前图.png'
        with patch.object(QFileDialog, 'getSaveFileName', return_value=(str(image_path), 'PNG (*.png)')):
            probe.save_plot()
        check(image_path.is_file() and image_path.stat().st_size > 5000,
              'sparse_overview_current_picture_export_contains_rendered_pair')

        enlarged = []; dialog_errors = []
        def close_enlarged():
            dialog = QApplication.activeModalWidget()
            if isinstance(dialog, QDialog):
                try:
                    canvas = dialog.findChild(FigureCanvasQTAgg)
                    if canvas is not None:
                        canvas.draw(); axes = canvas.figure.axes
                        enlarged.append(len(axes) == 3 and axes[0].get_window_extent().x1 < axes[1].get_window_extent().x0
                                        and dialog.grab().save(str(output / 'sparse-overview-enlarged.png')))
                except Exception as exc:
                    dialog_errors.append(str(exc))
                finally:
                    dialog.accept()
        dialog_timer = QTimer(); dialog_timer.setInterval(250); dialog_timer.timeout.connect(close_enlarged)
        dialog_timer.start()
        try: panel.maximize_plot()
        finally: dialog_timer.stop()
        check(enlarged == [True] and not dialog_errors, 'sparse_overview_enlarged_dialog_keeps_both_panels')
        for item in (first, last):
            check(snapshot_bytes(store.load_record(item['id'])['bundle']) == original[item['id']],
                  'sparse_overview_display_does_not_change_saved_payload_' + str(item['id']))
    finally:
        if probe is not None:
            probe.close()
            for timer in probe.findChildren(QTimer): timer.stop()
            probe.deleteLater(); QCoreApplication.sendPostedEvents(probe, QEvent.Type.DeferredDelete)
        app.setQuitOnLastWindowClosed(old_quit); app.processEvents()
        report['sparse_overview'] = details
