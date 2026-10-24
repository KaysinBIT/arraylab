"""Real Qt gestures and scientific-snapshot checks for the shared 3D canvas."""
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import time

import numpy as np
from PySide6.QtCore import QEventLoop, QPoint, QPointF, QTimer, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit


def exercise_3d_interaction(app, window, output, report, check):
    """Exercise real native mouse events without changing stored calculation data."""
    from .core import ArrayParameters, calculate
    from .feed import FeedSettings, design_feed
    from .planar import design_planar
    from .subarray import SubarrayConfig, optimize_subarrays
    from .widgets import PatternPlot, PatternSet
    from .window import MainWindow

    destination = output / 'interaction-3d'
    destination.mkdir(parents=True, exist_ok=True)
    events = []
    started = time.perf_counter()
    stage = 'prepare'
    details = {'qt_platform': app.platformName(), 'cases': []}
    report['interaction_3d'] = details
    log = (destination / 'events.jsonl').open('w', encoding='utf-8')
    independent = None
    probe = None
    connections = []
    held_axes = []  # Deliberately prevent GC from hiding stale axes callbacks.
    old_task = window.active_task()
    old_size = window.size()
    old_selected = window.selected_dataset()
    old_sequence = window.sequence
    old_datasets = tuple(map(id, window.datasets))
    old_inputs = [(control, control.text()) for control in window.findChildren(QLineEdit)]
    old_choices = [(control, control.currentIndex()) for control in window.findChildren(QComboBox)]
    caller_panels = (window.feed_panel, window.planar_panel, window.subarray_panel)
    old_reports = [(panel, panel.report) for panel in caller_panels]
    old_views = [(panel, panel.view.currentIndex()) for panel in caller_panels[1:]]
    old_case = window.subarray_panel.case_selector.currentIndex()

    def serializable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if is_dataclass(value):
            return asdict(value)
        raise TypeError(type(value).__name__)

    def digest(value):
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=serializable).encode('utf-8')
        return hashlib.sha256(encoded).hexdigest()

    old_report_hashes = [digest(value) for _, value in old_reports]

    def record(kind, **values):
        item = dict(t=time.perf_counter()-started, stage=stage, kind=kind, **values)
        events.append(item)
        log.write(json.dumps(item, ensure_ascii=False)+'\n')
        log.flush()

    heartbeat = QTimer()
    heartbeat.setTimerType(Qt.TimerType.PreciseTimer)
    heartbeat.setInterval(20)
    heartbeat.timeout.connect(lambda: record('heartbeat'))

    def pause(milliseconds):
        loop = QEventLoop()
        QTimer.singleShot(milliseconds, loop.quit)
        loop.exec()

    def counts(canvas):
        return {name: len(items) for name, items in canvas.callbacks.callbacks.items()}

    def surface_state(canvas, ax):
        states = []
        for item in canvas._surfaces.get(ax, ()):
            full = item.full()
            preview = item.preview() if item.preview else None
            states.append((full is not None and full.get_visible(),
                           preview is not None and preview.get_visible()))
        return states

    def full_visible(canvas, ax):
        states = surface_state(canvas, ax)
        return bool(states) and all(full and not preview for full, preview in states)

    def preview_visible(canvas, ax):
        states = surface_state(canvas, ax)
        return bool(states) and all(not full and preview for full, preview in states)

    def camera(ax):
        return {'azim': float(ax.azim), 'elev': float(ax.elev),
                'limits': [list(ax.get_xlim3d()), list(ax.get_ylim3d()), list(ax.get_zlim3d())]}

    def center(canvas, ax):
        bbox = ax.bbox
        ratio = canvas.device_pixel_ratio
        return QPoint(round((bbox.x0+bbox.x1)/2/ratio),
                      round(canvas.height()-(bbox.y0+bbox.y1)/2/ratio))

    def event_summary(kind, begin, end):
        times = [event['t'] for event in events if event['kind'] == kind and begin <= event['t'] <= end]
        gaps = np.diff(times)
        return {'count': len(times), 'max_interval_seconds': float(max(gaps)) if len(gaps) else None,
                'mean_interval_seconds': float(np.mean(gaps)) if len(gaps) else None}

    def begin_preview(canvas, ax, label):
        point = center(canvas, ax)
        QTest.mouseMove(canvas, point)
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
        QTest.mouseMove(canvas, point+QPoint(25, 12), delay=-1)
        pause(70)
        check(canvas.interaction_active and preview_visible(canvas, ax), label+'_preview_visible')
        return point+QPoint(25, 12)

    def drag(canvas, ax, label, button, duration):
        nonlocal stage
        stage = label
        record('gesture_start')
        before = camera(ax)
        origin = center(canvas, ax)
        QTest.mouseMove(canvas, origin)
        QTest.mousePress(canvas, button, Qt.KeyboardModifier.NoModifier, origin)
        check(canvas.interaction_active and preview_visible(canvas, ax), label+'_uses_preview')
        begin = time.perf_counter()-started
        initial_frames = canvas.interaction_draw_count
        driver_start = time.perf_counter()
        sent = []
        errors = []
        last = origin
        loop = QEventLoop()
        driver = QTimer()
        driver.setTimerType(Qt.TimerType.PreciseTimer)
        driver.setInterval(20)

        def move():
            nonlocal last
            elapsed = time.perf_counter()-driver_start
            if elapsed >= duration:
                driver.stop()
                loop.quit()
                return
            try:
                position = origin+QPoint(round(60*math.sin(elapsed*2.3)), round(38*math.sin(elapsed*2.8)))
                record('send_motion', x=position.x(), y=position.y())
                QTest.mouseMove(canvas, position, delay=-1)
                sent.append(elapsed)
                last = position
                if not canvas.interaction_active or not preview_visible(canvas, ax):
                    raise AssertionError(label+'_lost_preview_during_drag')
            except Exception as error:
                errors.append(str(error))
                driver.stop()
                loop.quit()

        driver.timeout.connect(move)
        driver.start()
        try:
            loop.exec()
        finally:
            driver.stop()
            end = time.perf_counter()-started
            frame_count = canvas.interaction_draw_count-initial_frames
            QTest.mouseRelease(canvas, button, Qt.KeyboardModifier.NoModifier, last)
        if errors:
            raise AssertionError(errors[0])
        check(not canvas.interaction_active and full_visible(canvas, ax), label+'_full_restored_on_release')
        after = camera(ax)
        if button == Qt.MouseButton.LeftButton:
            check((before['azim'], before['elev']) != (after['azim'], after['elev']), label+'_camera_rotated')
        else:
            check(before['limits'] != after['limits'], label+'_limits_zoomed')
        stage = label+'_settle'
        pause(250)
        idle_begin = time.perf_counter()-started
        settled_frames = canvas.interaction_draw_count
        pause(250)
        idle_end = time.perf_counter()-started
        check(not canvas._frame_timer.isActive() and not canvas._wheel_timer.isActive()
              and canvas.interaction_draw_count == settled_frames
              and event_summary('draw', idle_begin, idle_end)['count'] == 0, label+'_idle_stops_drawing')
        elapsed = end-begin
        result = {'elapsed_seconds': elapsed, 'sent_moves': len(sent), 'interaction_frames': frame_count,
                  'interaction_fps': frame_count/elapsed, 'before': before, 'after': after,
                  'motion': event_summary('motion', begin, end), 'draw': event_summary('draw', begin, end),
                  'heartbeat': event_summary('heartbeat', begin, end)}
        # Save real measurements before enforcing the acceptance limits.
        details.setdefault('gestures', {})[label] = result
        (destination/'performance.json').write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding='utf-8')
        gap = result['heartbeat']['max_interval_seconds']
        check(gap is not None and gap < .5, label+'_heartbeat_under_500ms')
        check(result['interaction_fps'] >= 12, label+'_at_least_12fps')
        return result

    def wheel(canvas, ax, label):
        nonlocal stage
        stage = label
        before = camera(ax)
        point = center(canvas, ax)
        for _ in range(3):
            event = QWheelEvent(QPointF(point), QPointF(canvas.mapToGlobal(point)), QPoint(), QPoint(0, 120),
                                Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                                Qt.ScrollPhase.NoScrollPhase, False)
            QApplication.sendEvent(canvas, event)
            check(canvas.interaction_active and preview_visible(canvas, ax), label+'_uses_preview')
            pause(40)
        after = camera(ax)
        check(before['limits'] != after['limits'], label+'_changes_limits')
        pause(canvas.wheel_settle_ms+70)
        check(not canvas.interaction_active and full_visible(canvas, ax), label+'_settled_full_surface')
        return {'before': before, 'after': after}

    def exercise(kind, canvas, toolbar, figure, redraw, science, duration):
        nonlocal stage
        stage = kind+'_prepare'
        redraw(True)
        pause(200)
        canvas.draw()
        ax = next(axis for axis in figure.axes if axis.name == '3d')
        check(canvas.isVisible() and full_visible(canvas, ax), kind+'_3d_full_surface_ready')
        scientific_hash = digest(science)
        footer_lines = list(figure._arraylab_footer_lines)
        footer = figure._arraylab_metric_footer
        check(footer in figure.texts and bool(footer.get_text()) and len(footer_lines) >= 2,
              kind+'_metrics_below_plot_present')
        footer_box = footer.get_window_extent(canvas.get_renderer())
        figure_box = figure.bbox
        check(footer_box.x0 >= figure_box.x0-2 and footer_box.y0 >= figure_box.y0-2
              and footer_box.x1 <= figure_box.x1+2 and footer_box.y1 <= figure_box.y1+2,
              kind+'_metrics_fit_inside_image')
        connections.extend([(canvas, canvas.mpl_connect('motion_notify_event',
            lambda event: record('motion', button=str(event.button), inaxes=event.inaxes is not None))),
                            (canvas, canvas.mpl_connect('draw_event', lambda event: record('draw')))])
        baseline_callbacks = counts(canvas)
        case = {'kind': kind, 'canvas_pixels': [canvas.width(), canvas.height()],
                'device_pixel_ratio': canvas.device_pixel_ratio, 'callbacks_before': baseline_callbacks,
                'left_drag': drag(canvas, ax, kind+'_rotate', Qt.MouseButton.LeftButton, duration),
                'right_drag': drag(canvas, ax, kind+'_zoom', Qt.MouseButton.RightButton, duration),
                'wheel': wheel(canvas, ax, kind+'_wheel')}

        stage = kind+'_preview_screenshot'
        point = begin_preview(canvas, ax, kind+'_screenshot')
        check(canvas.grab().save(str(destination/(kind+'-during-preview.png'))), kind+'_preview_screenshot_saved')
        check(preview_visible(canvas, ax), kind+'_screenshot_keeps_preview')
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
        pause(200)
        check(full_visible(canvas, ax), kind+'_screenshot_release_full')
        check(canvas.grab().save(str(destination/(kind+'-after-full.png'))), kind+'_full_screenshot_saved')

        stage = kind+'_repeated_switches'
        for _ in range(10):
            held_axes.extend(axis for axis in figure.axes if axis.name == '3d')
            redraw(False)
            pause(20)
            redraw(True)
            pause(20)
        pause(150)
        ax = next(axis for axis in figure.axes if axis.name == '3d')
        case['callbacks_after_ten_cycles'] = counts(canvas)
        check(case['callbacks_after_ten_cycles'] == baseline_callbacks, kind+'_callbacks_stable_with_old_axes_retained')
        case['rotate_after_switches'] = drag(canvas, ax, kind+'_rotate_after_switches', Qt.MouseButton.LeftButton, duration)

        stage = kind+'_home'
        point = begin_preview(canvas, ax, kind+'_home')
        toolbar.home()
        check(not canvas.interaction_active and full_visible(canvas, ax), kind+'_home_restores_full_surface')
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
        pause(200)

        stage = kind+'_export'
        point = begin_preview(canvas, ax, kind+'_export')
        export_states = []
        export_cid = canvas.mpl_connect('draw_event', lambda event: export_states.append(full_visible(canvas, ax)))
        try:
            path = destination/(kind+'-export-full.png')
            figure.savefig(path, dpi=100)
        finally:
            canvas.mpl_disconnect(export_cid)
            QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
        pause(150)
        check(path.exists() and path.stat().st_size > 1000 and bool(export_states) and all(export_states)
              and not canvas.interaction_active and full_visible(canvas, ax), kind+'_png_export_uses_full_surface')
        check(list(figure._arraylab_footer_lines) == footer_lines
              and figure._arraylab_metric_footer in figure.texts and bool(figure._arraylab_metric_footer.get_text()),
              kind+'_metrics_unchanged_after_interaction_home_export')
        check(digest(science) == scientific_hash, kind+'_scientific_snapshot_unchanged')
        check(canvas.last_interaction_error is None, kind+'_no_silent_interaction_fallback')
        case['scientific_sha256'] = scientific_hash
        case['callbacks_final'] = counts(canvas)
        details['cases'].append(case)

    heartbeat.start()
    try:
        # Other suite stages use their own windows. Prepare real results here,
        # independent of the caller's history, inputs, and prior suite ordering.
        probe = MainWindow()
        planar = probe.planar_panel
        subarray = probe.subarray_panel
        planar.report = design_planar(planar.read_parameters(), planar.read_settings(), workers=1)
        planar.export_button.setEnabled(True)
        planar.redraw()
        subarray_config = SubarrayConfig(count=4, nx=2, ny=2, iterations=4,
                                         restarts=1, opt_grid=21, eval_grid=41)
        subarray.apply_config(subarray_config)
        subarray.accept_result(optimize_subarrays(subarray_config, workers=1))
        prepared_reports = [(panel, panel.report, digest(panel.report)) for panel in (planar, subarray)]
        check(planar.report is not None and subarray.report is not None, 'interaction_independent_planar_subarray_results_prepared')
        parameters = ArrayParameters(elements=200, spacing=.49)
        linear = calculate(parameters)
        feed = design_feed(parameters, FeedSettings())
        for kind in ('linear', 'feed'):
            independent = PatternPlot()
            independent.setWindowTitle('ArrayLab 3D interaction acceptance - '+kind)
            independent.resize(1000, 760)
            if kind == 'linear':
                data = PatternSet('200元线阵验收', linear.angles_deg, linear.pattern_db, result=linear)
                science = linear
            else:
                pattern = feed['electrical']['patterns']['output']
                data = PatternSet('200元馈电验收', np.asarray(pattern['angles_deg']),
                                  np.asarray(pattern['pattern_db']), feed_report=feed)
                science = feed
            independent.show()
            exercise(kind, independent.canvas, independent.toolbar, independent.figure,
                     lambda three_d: independent.draw_sets([data], 0, view='3d' if three_d else None), science, 2.0)
            independent.close()
            independent.deleteLater()
            independent = None
            pause(50)
        probe.resize(1200, 850)
        probe.show()
        for kind, panel, flat in [('planar', planar, 'cut'), ('subarray', subarray, 'cuts')]:
            probe.activate_task(kind)
            def redraw(three_d, panel=panel, flat=flat):
                index = panel.view.findData('3d' if three_d else flat)
                if panel.view.currentIndex() == index:
                    panel.redraw()
                else:
                    panel.view.setCurrentIndex(index)
            exercise(kind, panel.canvas, panel.toolbar, panel.figure, redraw, panel.report, 1.0)
        for kind, (panel, original, original_hash) in zip(('planar', 'subarray'), prepared_reports):
            check(panel.report is original and digest(panel.report) == original_hash,
                  kind+'_interaction_preserves_prepared_report')
        check(window.sequence == old_sequence and tuple(map(id, window.datasets)) == old_datasets
              and window.selected_dataset() is old_selected, 'interaction_preserves_history_and_selected_snapshot')
        check(window.active_task() == old_task and window.size() == old_size
              and all(panel.view.currentIndex() == index for panel, index in old_views)
              and window.subarray_panel.case_selector.currentIndex() == old_case,
              'interaction_preserves_caller_view_state')
        check(all(control.text() == value for control, value in old_inputs)
              and all(control.currentIndex() == index for control, index in old_choices),
              'interaction_preserves_caller_inputs')
        for index, (panel, original) in enumerate(old_reports):
            check(panel.report is original and digest(panel.report) == old_report_hashes[index],
                  ('feed', 'planar', 'subarray')[index]+'_interaction_preserves_saved_report')
    finally:
        heartbeat.stop()
        for canvas, cid in connections:
            canvas.mpl_disconnect(cid)
        if independent is not None:
            independent.close()
            independent.deleteLater()
        if probe is not None:
            for panel in (probe.planar_panel, probe.subarray_panel):
                panel.canvas.finish_interaction(redraw=False)
            probe.close()
            probe.deleteLater()
        (destination/'performance.json').write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding='utf-8')
        log.close()
