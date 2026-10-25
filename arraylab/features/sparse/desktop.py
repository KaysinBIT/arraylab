"""Native ArrayLab sparse page. Uses numerical and drawing functions, not the legacy GUI."""
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
import copy
import json
import time
import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSignalBlocker
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTabWidget, QScrollArea, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox, QDialog, QTextBrowser)
from arraylab import __version__
from arraylab.framework.parallel import ComputationCancelled, requested_workers
from arraylab.features.sparse.engine.core import Config
from arraylab.features.sparse.engine.resources import estimate_resources, ensure_resources, format_bytes
from arraylab.features.sparse.engine.weighting import METHODS, Weighting
from arraylab.features.sparse.process import run_sparse_process
from arraylab.features.sparse.engine.elements import MODELS, build_element, write_example_csv
from arraylab.features.sparse.engine.scenarios import scenarios
from arraylab.features.sparse.engine.headless import save_bundle, read_json, validate_request
from arraylab.shared.io import write_text_atomic
from arraylab.features.sparse.engine.phase_quantization import excitation_phases
from arraylab.features.sparse.engine.runtime_paths import resource_path
from arraylab.features.sparse.engine.plots import style, draw_detail_plot, draw_layout_pattern, draw_case_results, draw_weight_comparison, draw_phase_comparison, draw_element, comparison_figure, structure_figure, result_name
from arraylab.features.sparse.planar_bridge import planar_from_sparse


SparseCancelled = ComputationCancelled


class SparseIntegerInput(QLineEdit):
    """Exact integer entry without Qt's signed 32-bit spin-box range."""

    def __init__(self, value, label):
        super().__init__(str(value))
        self.label = label
        self.setMinimumWidth(65)
        self.setAccessibleName(label)
        self.setToolTip('输入正整数，不限制为偶数；开始计算前检查对称条件和本机资源预算。')

    def value(self):
        text = self.text().strip()
        if not text or not text.isascii() or not text.isdecimal():
            raise ValueError(f'{self.label}须填写正整数。')
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(f'{self.label}的数字过长，无法读取。') from exc
        if value <= 0:
            raise ValueError(f'{self.label}须大于0。')
        return value

    def setValue(self, value):
        if type(value) is not int:
            raise ValueError(f'{self.label}须使用整数，不接受自动取整。')
        self.setText(str(value))


class SparseWorker(QThread):
    progress = Signal(str)

    def __init__(self, config=None, weighting=None, parent=None):
        super().__init__(parent)
        self.config, self.weighting = config, weighting
        self.workers = requested_workers()
        self.result = None; self.error = None; self.cancelled = False

    def tell(self, message):
        if self.isInterruptionRequested(): raise SparseCancelled()
        self.progress.emit(message)

    def run(self):
        try:
            self.result = run_sparse_process(self.config, self.weighting, workers=self.workers,
                progress=self.tell, cancelled=self.isInterruptionRequested)
            self.tell('完成')
        except SparseCancelled: self.cancelled = True
        except Exception as exc: self.error = f'{type(exc).__name__}: {exc}'


class SparsePlotToolbar(NavigationToolbar2QT):
    """Restore data limits while keeping layout sized for the current window."""
    def __init__(self, canvas, panel):
        self.panel = panel
        super().__init__(canvas, panel)

    def _update_view(self):
        super()._update_view()
        # Matplotlib also restores the old axes rectangles on Home/Back/Forward.
        # Those rectangles may predate a window resize, especially for colorbars.
        self.panel.reflow_current_figure()
        self.canvas.draw_idle()


class SparsePanel(QWidget):
    planar_requested = Signal(object, object, object)
    navigation_changed = Signal()

    def __init__(self, host):
        super().__init__()
        self.host = host; self.bundle = None; self.thread = None; self.last_export = None
        self._busy = False; self._last_error = None; self._initializing = True; self._last_selected_row = None
        self._cancel_pending = False; self._job_kind = None; self.inputs = {}
        self._job_started = None; self._progress_stage = ''
        self.preview_active = False
        self.history_entries = []; self.current_history_id = None
        self._history_path = None; self._history_store = None; self._history_loading = False
        self._history_restore_pending = False; self._history_restore_id = None
        self._history_pending = []; self._history_current_pending = None
        self._history_view_dirty = False; self._history_saved_view = None; self._history_warning = ''
        self.history_timer = QTimer(self); self.history_timer.setSingleShot(True)
        self.history_timer.timeout.connect(self._persist_history_view)
        self.progress_timer = QTimer(self); self.progress_timer.setInterval(250)
        self.progress_timer.timeout.connect(self.refresh_progress)
        self.build_parameters()
        layout = QVBoxLayout(self); layout.setContentsMargins(8, 6, 8, 6)
        actions = QHBoxLayout()
        self.run_button = QPushButton('开始综合'); self.run_button.clicked.connect(self.start)
        self.cancel_button = QPushButton('取消'); self.cancel_button.clicked.connect(self.cancel)
        self.weight_open_button = QPushButton('稀疏后加权'); self.weight_open_button.clicked.connect(self.open_weighting)
        self.export_button = QPushButton('导出结果'); self.export_button.clicked.connect(self.export)
        self.planar_button = QPushButton('送入平面阵'); self.planar_button.clicked.connect(self.send_to_planar)
        from arraylab.shared.qt.icons import workbench_icon
        self.planar_button.setIcon(workbench_icon('planar'))
        self.planar_button.setToolTip('所选理想布局与相对幅度 → 平面阵电气输入；50 Ω并联初值、连续扫描补偿。')
        for button in (self.run_button, self.cancel_button, self.weight_open_button, self.export_button, self.planar_button): actions.addWidget(button)
        actions.addStretch(); layout.addLayout(actions)
        self.state = QLabel('在右侧设置稀疏参数，然后开始综合。'); self.state.setWordWrap(True); self.state.setObjectName('state'); layout.addWidget(self.state)
        self.table = QTableWidget(); self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection); self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(65); self.table.setMaximumHeight(220)
        self.table.setStyleSheet('QTableWidget::item:selected {background:#d5e8f6;color:#174969;}')
        self.table.itemSelectionChanged.connect(self.select_result)
        self.scheme_box = QWidget(); schemes = QHBoxLayout(self.scheme_box); schemes.setContentsMargins(0, 0, 0, 0)
        schemes.addWidget(QLabel('方案')); self.scheme_selector = QComboBox(); schemes.addWidget(self.scheme_selector, 1)
        self.scheme_selector.currentIndexChanged.connect(self.table.selectRow); layout.addWidget(self.scheme_box); self.scheme_box.hide()
        self.result_splitter = QSplitter(Qt.Orientation.Vertical); self.result_splitter.addWidget(self.table)
        results = QWidget(); body = QVBoxLayout(results); body.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout(); row.addWidget(QLabel('工况'))
        self.case_selector = QComboBox(); self.case_selector.setMinimumWidth(90); row.addWidget(self.case_selector, 1)
        self.pattern_view = QComboBox()
        for label, key in [('总场', 'total'), ('阵因子', 'array_factor'), ('单元', 'element')]: self.pattern_view.addItem(label, key)
        self.phase_view = QComboBox(); self.phase_view.addItem('理想相位', 'ideal'); self.phase_view.addItem('量化相位', 'quantized')
        row.addWidget(self.pattern_view); row.addWidget(self.phase_view); body.addLayout(row)
        self.summary = QLabel('尚无稀疏方案。'); self.summary.setWordWrap(True); body.addWidget(self.summary)
        self.tabs = QTabWidget(); body.addWidget(self.tabs, 1)
        plot = QWidget(); pl = QVBoxLayout(plot); pl.setContentsMargins(0, 0, 0, 0); self.plot_layout = pl
        views = QHBoxLayout(); self.view = QComboBox()
        for label, key in [('布局与二维方向图', 'overview'), ('阵列布局（单图）', 'layout'), ('二维方向图（单图）', 'pattern'), ('扫描切线', 'cuts'), ('差分共阵', 'coarray'),
                           ('工况比较', 'cases'), ('加权前后', 'weights'), ('量化复核', 'quantization'), ('单元快照', 'element'),
                           ('方法比较', 'comparison'), ('结构化共阵', 'structures')]: self.view.addItem(label, key)
        views.addWidget(QLabel('图形')); views.addWidget(self.view, 1)
        self.max_button = QPushButton('放大图形'); self.max_button.clicked.connect(self.maximize_plot); views.addWidget(self.max_button)
        pl.addLayout(views); self.figure = Figure(figsize=(7, 4), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure); self.canvas.setMinimumSize(100, 100); pl.addWidget(self.canvas, 1)
        self.toolbar = SparsePlotToolbar(self.canvas, self); pl.addWidget(self.toolbar); self.tabs.addTab(plot, '图形')
        self.port_table = QTableWidget(); self.case_table = QTableWidget()
        for table in (self.port_table, self.case_table):
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers); table.setAlternatingRowColors(True)
        self.tabs.addTab(self.port_table, '阵元幅相'); self.tabs.addTab(self.case_table, '工况指标'); self.build_weighting()
        help_page = QTextBrowser(); help_page.setOpenExternalLinks(True)
        help_page.setMarkdown(resource_path('MANUAL.md').read_text(encoding='utf-8') + '\n\n---\n\n' + resource_path('RESEARCH.md').read_text(encoding='utf-8'))
        self.manual_tab = self.tabs.addTab(help_page, '说明')
        self.result_splitter.addWidget(results); self.result_splitter.setStretchFactor(0, 0); self.result_splitter.setStretchFactor(1, 1)
        self.result_splitter.setSizes([105, 450]); layout.addWidget(self.result_splitter, 1)
        for combo in (self.view, self.case_selector, self.pattern_view, self.phase_view): combo.currentIndexChanged.connect(self.redraw)
        self.tabs.currentChanged.connect(self._history_ui_changed)
        self.resize_timer = QTimer(self); self.resize_timer.setSingleShot(True); self.resize_timer.timeout.connect(self.refresh_layout)
        self._compact = None; self._initializing = False; self.set_busy(False); self.update_controls()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not hasattr(self, 'resize_timer'): return
        compact = self.height() < 520
        if compact != self._compact:
            self._compact = compact; self.table.setVisible(not compact); self.scheme_box.setVisible(compact)
            if not compact: self.result_splitter.setSizes([100, max(250, self.height() - 100)])
        self.resize_timer.start(60)

    def refresh_layout(self):
        if self.table.isVisible() and self.table.currentRow() >= 0:
            self.table.scrollToItem(self.table.item(self.table.currentRow(), 0))
        # Keep axes and the toolbar's navigation stack during delayed resize
        # events, including events delivered in the middle of a pan gesture.
        self.reflow_current_figure()
        self.canvas.draw_idle()

    def initialize_history(self, path=None):
        """Read only the archive index; the selected bundle is loaded on entry."""
        self._history_path = Path(path) if path is not None else None
        self._history_store = None; self.history_entries = []
        self.current_history_id = None; self._history_restore_id = None
        self._history_restore_pending = False; self._history_saved_view = None
        if self._history_path is not None:
            try:
                from arraylab.features.sparse.history_store import SparseHistoryStore
                self._history_store = SparseHistoryStore(self._history_path)
                entries, selected = self._history_store.list_records()
                self.history_entries = entries
                self._history_restore_id = selected if selected is not None and selected >= 0 else None
                self._history_restore_pending = self._history_restore_id is not None
            except (OSError, ValueError, ArithmeticError) as exc:
                self._history_failure('稀疏历史无法读取；原档案未覆盖', exc)
        self.navigation_changed.emit()

    def ensure_history_restored(self):
        if not self._history_restore_pending or self._busy or self.thread is not None:
            return False
        self._history_restore_pending = False
        if self.bundle is not None:
            return False
        return self.restore_history(self._history_restore_id)

    def history_view_state(self):
        names = ('plot', 'ports', 'cases', 'weight', 'manual')
        return {'result_index': max(0, self.table.currentRow()),
                'case_index': max(0, self.case_selector.currentIndex()),
                'view': self.view.currentData(), 'tab': names[self.tabs.currentIndex()],
                'pattern_view': self.pattern_view.currentData(), 'phase_view': self.phase_view.currentData()}

    def _history_ui_changed(self, *_):
        if self._initializing or self._history_loading:
            return
        self.navigation_changed.emit()
        self.host.refresh_actions()
        if self._history_path is None or self.bundle is None or self.preview_active:
            return
        state = self.history_view_state()
        if self._history_current_pending is not None:
            self._history_current_pending['view_state'] = state
        elif self.current_history_id is not None and state != self._history_saved_view:
            self._history_view_dirty = True
            if not self.history_timer.isActive():
                self.history_timer.start(180)

    def _history_failure(self, message, error):
        self._history_warning = f'{message}：{error}'
        self.set_state(self._history_warning)
        self.host.statusBar().showMessage(self._history_warning)
        self.host.record_message(self._history_warning)

    def _persist_history_view(self, *, force=False):
        if (self._history_loading or self.preview_active or self.bundle is None
                or self._history_store is None or self.current_history_id is None
                or self._history_current_pending is not None):
            return
        if not force and not self._history_view_dirty:
            return
        state = self.history_view_state()
        try:
            self._history_store.select(self.current_history_id, view_state=state)
            self._history_saved_view = state; self._history_view_dirty = False
            if not self._history_pending:
                self._history_warning = ''
        except (OSError, ValueError, ArithmeticError) as exc:
            self._history_view_dirty = True
            self._history_failure('稀疏历史视图状态未保存，已有计算快照保留', exc)

    def _record_completed_history(self, kind):
        if self._history_path is None:
            self.navigation_changed.emit()
            return
        c = self.bundle['config']
        label = f"{'综合' if kind == 'synthesis' else '加权'} · {c['nx']}×{c['ny']} · K={c['k']} · {datetime.now():%m-%d %H:%M:%S}"
        pending = {'bundle': self.bundle, 'label': label, 'kind': kind,
                   'view_state': self.history_view_state()}
        self._history_pending.append(pending); self._history_current_pending = pending
        self.current_history_id = None; self._history_saved_view = None
        self.retry_history_save()

    def retry_history_save(self):
        """Persist only new successful snapshots; never flush a stale view."""
        if self._history_path is None:
            return True
        self.history_timer.stop()
        written = False
        try:
            if self._history_store is None and self._history_pending:
                from arraylab.features.sparse.history_store import SparseHistoryStore
                self._history_store = SparseHistoryStore(self._history_path)
            while self._history_pending:
                item = self._history_pending[0]
                entry = self._history_store.append(item['bundle'], item['label'],
                                                  kind=item['kind'], view_state=item['view_state'])
                self._history_pending.pop(0); written = True
                if self._history_current_pending is item:
                    self.current_history_id = entry['id']; self._history_current_pending = None
                    self._history_saved_view = dict(item['view_state'])
                # The append is committed even if a later index refresh fails.
                if not any(existing['id'] == entry['id'] for existing in self.history_entries):
                    self.history_entries.append(entry)
            if written:
                self.history_entries, _ = self._history_store.list_records()
                self._history_warning = ''
        except (OSError, ValueError, ArithmeticError) as exc:
            self._history_failure('自动保存稀疏历史失败，新结果仍在内存；请导出当前结果备份', exc)
        if not self._busy:
            self._persist_history_view(force=written)
        self.navigation_changed.emit()
        return not self._history_pending

    def can_close_history(self):
        self.retry_history_save()
        if not self._history_pending:
            return True
        answer = QMessageBox.question(self, '稀疏历史尚未自动保存',
            f'有{len(self._history_pending)}份成功计算快照尚未自动保存。\n'
            '“导出当前结果”只备份当前显示的一份；其他待存快照需在存储恢复后重试保存。\n'
            '选择“否”返回；选择“是”将退出并放弃全部尚未保存的快照。\n\n'
            + self._history_warning + '\n\n仍要退出吗？',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    def _capture_input_state(self):
        controls = list(self.inputs.values()) + [self.hole_shape, self.element_model, self.element_axis,
            self.element_import_mode, self.lp_solver, self.element_file, self.joint_scans, self.joint_frequencies,
            self.weight_method, self.weight_dr, self.weight_sll, self.weight_nbar, self.weight_sigma]
        return [(control, control.currentIndex() if isinstance(control, QComboBox) else
                 control.text() if isinstance(control, QLineEdit) else control.value()) for control in controls]

    def _restore_input_state(self, states):
        previous = self._initializing; self._initializing = True
        try:
            for control, value in states:
                if isinstance(control, QComboBox): control.setCurrentIndex(value)
                elif isinstance(control, QLineEdit): control.setText(value)
                else: control.setValue(value)
        finally:
            self._initializing = previous
        self.update_controls()

    def _apply_history_view(self, state):
        with QSignalBlocker(self.view), QSignalBlocker(self.pattern_view), QSignalBlocker(self.phase_view), QSignalBlocker(self.tabs):
            for control, key in ((self.view, 'view'), (self.pattern_view, 'pattern_view'), (self.phase_view, 'phase_view')):
                control.setCurrentIndex(control.findData(state[key]))
            tabs = {'plot': 0, 'ports': 1, 'cases': 2, 'weight': self.weight_tab, 'manual': self.manual_tab}
            self.tabs.setCurrentIndex(tabs[state['tab']])

    def restore_history(self, record_id):
        if self._busy or self.thread is not None or self._history_store is None:
            return False
        self.history_timer.stop()
        self._persist_history_view()
        try:
            record = self._history_store.load_record(record_id)
            bundle, state = record['bundle'], record['view_state']
            config = Config(**bundle['config'])
        except (OSError, ValueError, ArithmeticError) as exc:
            self._history_failure('稀疏历史恢复失败；当前输入和结果保留', exc)
            return False
        old = (self.bundle, self.history_view_state(), self.figure, self.preview_active,
               self.current_history_id, self._history_current_pending, self._history_saved_view)
        inputs = self._capture_input_state()
        self.history_timer.stop(); self._history_loading = True
        try:
            self.set_config(config)
            self.receive(bundle, draw=False, selected_index=state['result_index'], case_index=state['case_index'], preserve_snapshot=True)
            self._apply_history_view(state)
            selected = self.selected_result()
            weighting = selected.get('weighting') if selected is not None else None
            if weighting:
                self.weight_method.setCurrentIndex(self.weight_method.findData(weighting['method']))
                for control, key in ((self.weight_dr, 'dr_db'), (self.weight_sll, 'sll_db'),
                                     (self.weight_nbar, 'nbar'), (self.weight_sigma, 'sigma')):
                    control.setValue(weighting[key])
            self.redraw()
            self.current_history_id = record['id']; self._history_current_pending = None
            self._history_saved_view = None; self._history_view_dirty = True
            self._history_restore_pending = False
        except Exception as exc:
            self.bundle, previous_view, old_figure, self.preview_active, self.current_history_id, self._history_current_pending, self._history_saved_view = old
            self._restore_input_state(inputs)
            try:
                if self.bundle is not None:
                    self.receive(self.bundle, draw=False, selected_index=previous_view['result_index'],
                                 case_index=previous_view['case_index'], preserve_snapshot=True)
                else:
                    self.table.setRowCount(0); self.scheme_selector.clear(); self.case_selector.clear()
                    self.summary.setText('尚无稀疏方案。'); self.port_table.setRowCount(0); self.case_table.setRowCount(0)
                self._apply_history_view(previous_view)
                self.refresh_result_tables()
                self.install_figure(old_figure); self.preview_active = old[3]
            except Exception:
                self.summary.setText('历史显示失败；原计算数据仍保留。')
            self._history_failure('稀疏历史显示失败；当前输入和结果已恢复', exc)
            return False
        finally:
            self._history_loading = False
            self.navigation_changed.emit(); self.host.refresh_actions()
        self._persist_history_view(force=True)
        suffix = ('；' + self._history_warning) if self._history_view_dirty else ''
        self.set_state(f"已恢复稀疏历史：{record['label']}；显示保存的结果，未重新综合或加权。" + suffix)
        self.host.statusBar().showMessage('稀疏历史已恢复；参数、方案和工况来自所选快照。')
        self.set_busy(False)
        return True

    def build_parameters(self):
        self.parameter_widget = QWidget(); layout = QVBoxLayout(self.parameter_widget); layout.setContentsMargins(5, 5, 5, 5)
        presets = QHBoxLayout()
        for text, callback in [('默认', lambda: self.set_config(Config())), ('中心孔', self.hollow_preset), ('角频', self.joint_preset)]:
            button = QPushButton(text); button.clicked.connect(callback); presets.addWidget(button)
        layout.addLayout(presets); self.settings_tabs = QTabWidget(); layout.addWidget(self.settings_tabs, 1)
        def page(name):
            widget = QWidget(); body = QVBoxLayout(widget); body.setContentsMargins(6, 8, 6, 6)
            form = QFormLayout(); form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows); body.addLayout(form)
            scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(widget)
            self.settings_tabs.addTab(scroll, name); return body, form
        defaults = Config()
        def number(form, key, label, low, high, step=1):
            if key == 'seed':
                box = QLineEdit(str(defaults.seed)); box.textChanged.connect(self.mark_stale)
                box.setAccessibleName(label); self.inputs[key] = box; form.addRow(label, box); return
            if key in ('nx', 'ny', 'k'):
                box = SparseIntegerInput(getattr(defaults, key), label)
                box.textChanged.connect(self.mark_stale); self.inputs[key] = box; form.addRow(label, box); return
            box = QSpinBox() if type(getattr(defaults, key)) is int else QDoubleSpinBox()
            if isinstance(box, QDoubleSpinBox): box.setDecimals(8)
            box.setRange(low, high); box.setSingleStep(step); box.setValue(getattr(defaults, key)); box.setMinimumWidth(65)
            box.setAccessibleName(label); box.valueChanged.connect(self.mark_stale); self.inputs[key] = box; form.addRow(label, box)
        def combo(form, label, pairs):
            box = QComboBox()
            for text, value in pairs: box.addItem(text, value)
            box.currentIndexChanged.connect(self.mark_stale); form.addRow(label, box); return box
        body, form = page('阵列')
        for spec in [('nx', 'X候选数', None, None), ('ny', 'Y候选数', None, None), ('k', '目标阵元K', None, None),
                     ('spacing', '间距/λ0', .1, 2, .1), ('reference_ghz', '参考GHz', .1, 1000, 1), ('min_spacing', '最小距/λ0', 0, 100, .1)]: number(form, *spec)
        self.hole_shape = combo(form, '中心禁布', [('无', 'none'), ('矩形', 'rectangle'), ('圆形', 'circle')])
        number(form, 'hole_x', '孔宽X/直径λ0', .01, 100, .25); number(form, 'hole_y', '孔高Y/λ0', .01, 100, .25)
        self.availability = QLabel(); self.availability.setWordWrap(True); body.addWidget(self.availability)
        label = QLabel('X/Y候选数和目标阵元K均可填写正整数。中心对称布局中，奇数K需要X/Y均为奇数且中心阵元不在禁布区；其余阵元成对选取。实际可计算规模取决于资源预算。')
        label.setWordWrap(True); body.addWidget(label); body.addStretch()
        body, form = page('单元')
        self.element_model = combo(form, '模型', [(v, k) for k, v in MODELS.items()])
        self.element_axis = combo(form, '偶极子轴', [(v, v) for v in ('x', 'y', 'z')])
        number(form, 'element_dipole_length', '长度/λ0', .05, 1, .05); number(form, 'element_patch_hpbw', '贴片HPBW°', 30, 150, 5)
        self.element_import_mode = combo(form, 'CSV角域', [('二维网格', 'grid'), ('轴对称', 'axisymmetric')])
        self.element_file = QLineEdit(); self.element_file.textChanged.connect(self.mark_stale); form.addRow('CSV文件', self.element_file)
        for text, callback in [('导入单元CSV', self.import_element), ('载入示例CSV', self.load_example_element),
                               ('保存CSV模板', self.save_element_template), ('预览待计算单元', self.preview_element)]:
            button = QPushButton(text); button.clicked.connect(callback); body.addWidget(button)
        body.addStretch(); body, form = page('工况')
        number(form, 'theta', '扫描θ°', 0, 60, 5); number(form, 'phi', '扫描φ°', -360, 360, 15)
        self.joint_scans = QLineEdit(); self.joint_scans.setPlaceholderText('0,0;30,0;30,90')
        self.joint_frequencies = QLineEdit(); self.joint_frequencies.setPlaceholderText('9,10,11')
        for label, box in [('角度θ,φ', self.joint_scans), ('频率GHz', self.joint_frequencies)]:
            form.addRow(label, box); box.textChanged.connect(self.mark_stale)
        number(form, 'phase_bits', '移相位数', 0, 12)
        label = QLabel('列表留空时使用单扫描角/参考频率。共同布局与幅度，最多25工况；每频率重新设置扫描相位。')
        label.setWordWrap(True); body.addWidget(label); body.addStretch(); body, form = page('高级')
        for spec in [('seed', '随机种子', 0, 2147483647, 1), ('iterations', '退火步数', 1, 20000, 100),
                     ('opt_grid', '优化网格', 25, 101, 2), ('eval_grid', '复核网格', 81, 401, 20),
                     ('amplitude_dr_db', '调幅范围dB', 0, 40, 5)]: number(form, *spec)
        self.lp_solver = combo(form, '线性规划', [('自动（大网格内点法）', 'auto'), ('原求解器', 'highs'), ('内点法', 'highs-ipm')])
        self.lp_solver.setToolTip('自动模式在候选对不少于128时使用内点法。约束和采样不变；多重最优解可能使L1布局不同。需要原算法时选择原求解器。')
        label = QLabel('PSL为固定主瓣掩膜外采样峰值，不是连续角域保证。HPBW采用方向余弦宽度。误差统计及馈线尺寸暂缓。')
        label.setWordWrap(True); body.addWidget(label); body.addStretch()
        self.restore_button = QPushButton('恢复已计算参数'); self.restore_button.clicked.connect(self.restore_parameters); layout.addWidget(self.restore_button)
        files = QHBoxLayout(); self.load_settings_button = QPushButton('打开设置'); self.save_settings_button = QPushButton('保存设置')
        self.load_settings_button.clicked.connect(self.open_settings_dialog); self.save_settings_button.clicked.connect(self.save_settings_dialog)
        files.addWidget(self.load_settings_button); files.addWidget(self.save_settings_button); layout.addLayout(files)

    def build_weighting(self):
        page = QWidget(); body = QVBoxLayout(page); form = QFormLayout(); body.addLayout(form)
        self.weight_method = QComboBox()
        for key, label in METHODS.items(): self.weight_method.addItem(label, key)
        form.addRow('加权方法', self.weight_method)
        self.weight_dr = QDoubleSpinBox(); self.weight_dr.setRange(0, 40); self.weight_dr.setValue(20)
        self.weight_sll = QDoubleSpinBox(); self.weight_sll.setRange(20, 80); self.weight_sll.setValue(30)
        self.weight_nbar = QSpinBox(); self.weight_nbar.setRange(2, 10); self.weight_nbar.setValue(4)
        self.weight_sigma = QDoubleSpinBox(); self.weight_sigma.setRange(.05, 1); self.weight_sigma.setValue(.35); self.weight_sigma.setSingleStep(.05)
        for label, box in [('幅度范围dB', self.weight_dr), ('参考旁瓣dB', self.weight_sll), ('Taylor nbar', self.weight_nbar), ('Gaussian σ', self.weight_sigma)]: form.addRow(label, box)
        self.weight_button = QPushButton('应用到所选方案'); self.weight_button.clicked.connect(self.apply_weighting); body.addWidget(self.weight_button)
        note = QLabel('保留所选方案的位置和数量，追加新的权重结果。修改右侧输入不会改变加权源。完成后可在“图形→加权前后”和“阵元幅相”查看。')
        note.setWordWrap(True); body.addWidget(note); body.addStretch()
        self.weight_tab = self.tabs.addTab(page, '加权'); self.weight_method.currentIndexChanged.connect(self.update_controls)

    def config_from_inputs(self):
        return Config(**{key: int(box.text()) if key == 'seed' else box.value() for key, box in self.inputs.items()}, hole_shape=self.hole_shape.currentData(),
            element_model=self.element_model.currentData(), element_dipole_axis=self.element_axis.currentData(),
            element_import_mode=self.element_import_mode.currentData(), element_file=self.element_file.text().strip(),
            scan_angles=self.joint_scans.text(), frequencies_ghz=self.joint_frequencies.text(), lp_solver=self.lp_solver.currentData())

    def set_config(self, config):
        if self._busy: return
        self._initializing = True
        try:
            for key, box in self.inputs.items():
                if key == 'seed': box.setText(str(config.seed))
                else: box.setValue(getattr(config, key))
            for box, value in [(self.hole_shape, config.hole_shape), (self.element_model, config.element_model),
                               (self.element_axis, config.element_dipole_axis), (self.element_import_mode, config.element_import_mode),
                               (self.lp_solver, config.lp_solver)]: box.setCurrentIndex(box.findData(value))
            self.element_file.setText(config.element_file); self.joint_scans.setText(config.scan_angles); self.joint_frequencies.setText(config.frequencies_ghz)
        finally: self._initializing = False
        self.mark_stale()

    def hollow_preset(self): self.set_config(replace(Config(), hole_shape='rectangle', hole_x=2, hole_y=2))
    def joint_preset(self): self.set_config(replace(Config(), k=24, min_spacing=.6, scan_angles='0,0;30,0;30,90', frequencies_ghz='9,10,11'))
    def restore_parameters(self):
        if self.bundle: self.set_config(Config(**self.bundle['config']))

    def save_settings_path(self, path):
        request = {'schema_version': 1, 'config': asdict(self.config_from_inputs()), 'weighting': []}
        checked = validate_request(request)
        write_text_atomic(path, json.dumps(checked, ensure_ascii=False, indent=2) + '\n')

    def load_settings_path(self, path):
        request = validate_request(read_json(path), Path(path).absolute().parent)
        if request['weighting']:
            raise ValueError('此文件带有自动加权任务，请通过无头入口执行；界面设置文件的weighting须为空。')
        self.set_config(Config(**request['config']))

    def open_settings_dialog(self):
        if self._busy: return
        path, _ = QFileDialog.getOpenFileName(self, '打开稀疏参数', self.host.last_directory, 'JSON (*.json)')
        if path:
            try: self.load_settings_path(path)
            except (ValueError, OSError, TypeError) as exc: QMessageBox.warning(self, '设置读取失败', str(exc))

    def save_settings_dialog(self):
        if self._busy: return
        path, _ = QFileDialog.getSaveFileName(self, '保存当前稀疏参数', 'sparse-config.json', 'JSON (*.json)')
        if path:
            try: self.save_settings_path(path)
            except (ValueError, OSError, TypeError) as exc: QMessageBox.warning(self, '设置保存失败', str(exc))

    def mark_stale(self):
        if self._initializing: return
        self.update_controls(); self.set_state('参数已改变，尚未综合；图形、加权和导出仍使用已计算快照。' if self.bundle else '请设置右侧参数后开始综合。')

    def set_state(self, message):
        self.state.setText(message)

    def update_controls(self):
        if self._initializing: return
        self.inputs['hole_x'].setEnabled(self.hole_shape.currentData() != 'none'); self.inputs['hole_y'].setEnabled(self.hole_shape.currentData() == 'rectangle')
        model = self.element_model.currentData(); self.element_axis.setEnabled(model == 'dipole'); self.inputs['element_dipole_length'].setEnabled(model == 'dipole')
        self.inputs['element_patch_hpbw'].setEnabled(model == 'patch'); self.element_import_mode.setEnabled(model == 'imported'); self.element_file.setEnabled(model == 'imported')
        try:
            config = self.config_from_inputs()
            count = config.nx * config.ny
            candidates = f'可用候选位置：{count:,}' if config.hole_shape == 'none' else f'完整网格：{count:,}；禁布后可用数量在综合前核实'
            resources = estimate_resources(config, workers=requested_workers())
            verdict = '预算内' if resources['within_budget'] else '超出当前预算，开始综合时会提示'
            self.availability.setText(f"{candidates}。\n预计峰值 {format_bytes(resources['estimated_peak_bytes'])} / 当前预算 {format_bytes(resources['budget_bytes'])}；{verdict}。")
        except (ValueError, OverflowError) as exc:
            self.availability.setText('请检查参数：' + str(exc))
        method = self.weight_method.currentData(); self.weight_dr.setEnabled(method != 'uniform')
        self.weight_sll.setEnabled(method in ('taylor', 'chebyshev')); self.weight_nbar.setEnabled(method == 'taylor'); self.weight_sigma.setEnabled(method == 'gaussian')

    def set_busy(self, busy):
        self._busy = busy; self.parameter_widget.setEnabled(not busy); self.run_button.setEnabled(not busy); self.cancel_button.setEnabled(busy)
        ready = self.bundle is not None and not busy
        for widget in (self.weight_open_button, self.weight_button, self.export_button, self.planar_button, self.restore_button): widget.setEnabled(ready)
        for widget in (self.table, self.scheme_selector, self.case_selector, self.pattern_view): widget.setEnabled(not busy)
        self.phase_view.setEnabled(not busy and bool(self.bundle and self.bundle['config']['phase_bits']))
        if hasattr(self.host, 'save_button') and self.host.tabs.currentWidget() is self:
            self.host.save_button.setEnabled(ready)
            if hasattr(self.host, 'save_action'): self.host.save_action.setEnabled(ready)
        self.host.refresh_actions()
        self.navigation_changed.emit()

    def start(self):
        if self.thread is not None or self._busy: return
        try:
            config = self.config_from_inputs().validate()
            ensure_resources(config, workers=requested_workers())
            element = build_element(config)
            for case in scenarios(config): element.steering_response(case)
        except (ValueError, OSError) as exc:
            self._last_error = str(exc); QMessageBox.warning(self, '稀疏参数检查', str(exc)); return
        self.begin_worker(SparseWorker(config=config, parent=self), 'synthesis')

    def begin_worker(self, worker, kind):
        self.history_timer.stop(); self._persist_history_view()
        self._last_error = None; self._cancel_pending = False; self._job_kind = kind; self.thread = worker; self.set_busy(True)
        self._job_started = time.monotonic(); self._progress_stage = '正在启动后台计算'
        self.refresh_progress(); self.progress_timer.start()
        worker.progress.connect(self.progress_message); worker.finished.connect(self.complete); worker.start()

    def progress_message(self, text):
        self._progress_stage = text; self.refresh_progress()

    def refresh_progress(self):
        if self._job_started is None: return
        elapsed = time.monotonic() - self._job_started
        stage = '正在停止后台计算并回收资源' if self._cancel_pending else self._progress_stage
        text = f'{stage} · 已用 {elapsed:.1f} 秒' + ('。' if self._cancel_pending else ' · 可取消；上一份结果保留。')
        self.set_state(text); self.host.statusBar().showMessage('稀疏 · ' + text)

    def cancel(self):
        if self.thread is not None:
            self._cancel_pending = True; self.thread.requestInterruption(); self.refresh_progress()

    def complete(self):
        worker = self.thread
        if worker is None: return
        self.progress_timer.stop(); self._job_started = None
        previous = self.bundle
        selection = (self.table.currentRow(), self.case_selector.currentIndex(), self.phase_view.currentIndex())
        previous_view, previous_tab = self.view.currentIndex(), self.tabs.currentIndex()
        previous_figure = self.figure
        previous_preview = self.preview_active
        previous_history = (self.current_history_id, self._history_current_pending, self._history_saved_view)
        kind, succeeded = self._job_kind, False
        history_loading = self._history_loading; self._history_loading = True
        try:
            if worker.cancelled or self._cancel_pending: self.set_state('已取消，保留上一份结果。')
            elif worker.error:
                self._last_error = worker.error; self.set_state('计算失败，保留上一份结果：' + worker.error)
            else:
                self.receive(worker.result)
                succeeded = True
        except Exception as exc:
            self._last_error = f'{type(exc).__name__}: {exc}'
            self.bundle = previous
            self.current_history_id, self._history_current_pending, self._history_saved_view = previous_history
            try:
                self.view.blockSignals(True); self.view.setCurrentIndex(previous_view); self.view.blockSignals(False)
                self.tabs.setCurrentIndex(previous_tab)
                if previous is not None:
                    self.receive(previous, draw=False, selected_index=selection[0], case_index=selection[1], preserve_snapshot=True)
                    self.phase_view.blockSignals(True); self.phase_view.setCurrentIndex(selection[2]); self.phase_view.blockSignals(False)
                    if previous_preview:
                        self.install_figure(previous_figure); self.preview_active = True
                    else:
                        self.redraw()
                else:
                    self.table.setRowCount(0); self.scheme_selector.clear(); self.case_selector.clear()
                    self.summary.setText('尚无稀疏方案。'); self.port_table.setRowCount(0); self.case_table.setRowCount(0)
                    self.install_figure(previous_figure)
            except Exception:
                # A renderer failure must never strand the application in its busy state.
                self.port_table.setRowCount(0); self.case_table.setRowCount(0)
                self.summary.setText('结果显示失败；已保留上一份计算数据。')
            self.set_state('结果显示失败，保留上一份结果：' + self._last_error)
        finally:
            self._history_loading = history_loading
            self.thread = None; worker.deleteLater(); self._job_kind = None; self.set_busy(False)
            if succeeded:
                self._record_completed_history(kind)
            if getattr(self.host, '_sparse_close_pending', False):
                self.host._sparse_close_pending = False; QTimer.singleShot(0, self.host.close)

    def open_weighting(self): self.tabs.setCurrentIndex(self.weight_tab)
    def apply_weighting(self):
        if self._busy or self.bundle is None or self.table.currentRow() < 0: return
        settings = Weighting(method=self.weight_method.currentData(), dr_db=self.weight_dr.value(), sll_db=self.weight_sll.value(), nbar=self.weight_nbar.value(), sigma=self.weight_sigma.value())
        try: settings.validate(Config(**self.bundle['config']))
        except ValueError as exc: QMessageBox.warning(self, '加权参数检查', str(exc)); return
        self.begin_worker(SparseWorker(weighting=(copy.deepcopy(self.bundle), self.table.currentRow(), settings), parent=self), 'weight')

    @staticmethod
    def fill_table(table, columns, rows):
        table.setColumnCount(len(columns)); table.setHorizontalHeaderLabels([label for _, label in columns]); table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, (key, _) in enumerate(columns):
                value = row.get(key); text = '—' if value is None else (f'{value:.6g}' if isinstance(value, (float, np.floating)) else str(value)); table.setItem(i, j, QTableWidgetItem(text))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    def receive(self, bundle, *, draw=True, selected_index=None, case_index=None, preserve_snapshot=False):
        if not preserve_snapshot:
            bundle['integration'] = {'arraylab_version': __version__, 'source_engine': 'Sparse2D 1.3.0'}
        if not self._history_loading and bundle is not self.bundle:
            self.current_history_id = None; self._history_current_pending = None; self._history_saved_view = None
        self.bundle = bundle; self._last_selected_row = None; self.preview_active = False
        controls = (self.table, self.scheme_selector, self.case_selector, self.phase_view, self.view)
        blocked = [control.blockSignals(True) for control in controls]
        try:
            rows = [{'name': result_name(r), **r['metrics']} for r in bundle['results']]
            self.fill_table(self.table, [('name', '方案'), ('elements', '阵元'), ('validation_psl_db', '理想PSL/dB'), ('amplitude_dynamic_range_db', '幅度范围/dB'),
                ('hpbw_u', 'HPBW u'), ('hpbw_v', 'HPBW v'), ('min_distance_lambda_ref', '最小距/λ0'), ('spacing_violations', '间距违规'), ('unique_lags', '不同lag')], rows)
            self.scheme_selector.clear(); self.scheme_selector.addItems([r['name'] for r in rows]); self.case_selector.clear()
            for case in bundle['scenarios']: self.case_selector.addItem(case['label'], case['index'])
            self.phase_view.setCurrentIndex(1 if bundle['config']['phase_bits'] else 0)
            index = selected_index if selected_index is not None and selected_index >= 0 else bundle.get('last_weighted_index')
            if index is None: index = min((i for i, r in enumerate(bundle['results']) if r['metrics']['elements'] == bundle['config']['k'] and r['name'] != 'L1 weighted solution'), key=lambda i: bundle['results'][i]['metrics']['validation_psl_db'])
            self.table.selectRow(index); self.scheme_selector.setCurrentIndex(index)
            if case_index is not None and case_index >= 0:
                self.case_selector.setCurrentIndex(case_index); self._last_selected_row = index
            if draw and self._job_kind == 'weight':
                self.view.setCurrentIndex(self.view.findData('weights')); self.tabs.setCurrentIndex(0)
        finally:
            for control, previous in zip(controls, blocked): control.blockSignals(previous)
        if draw: self.select_result()
        try: dirty = self.config_from_inputs() != Config(**bundle['config'])
        except ValueError: dirty = True
        suffix = '当前输入尚未综合，本次结果仍对应已计算参数。' if dirty else '结果与当前参数一致。'
        execution=bundle.get('execution',{})
        resources=f" · 复核{execution['workers']}路" if execution.get('scope')=='sparse_layout_validation' else ''
        self.set_state(f"已完成 {len(bundle['results'])} 个方案、{len(bundle['scenarios'])} 个工况{resources}；" + suffix)
        self.host.statusBar().showMessage('稀疏结果已更新；选中方案可送入平面阵电气设计。')
        self._history_ui_changed()

    def select_result(self):
        row = self.table.currentRow()
        if self.bundle is None or row < 0: return
        if row != self._last_selected_row:
            self.case_selector.blockSignals(True); self.case_selector.setCurrentIndex(self.bundle['results'][row]['metrics']['worst_case_index']); self.case_selector.blockSignals(False); self._last_selected_row = row
        self.scheme_selector.blockSignals(True); self.scheme_selector.setCurrentIndex(row); self.scheme_selector.blockSignals(False)
        self.redraw()

    def selected_result(self):
        return self.bundle['results'][self.table.currentRow()] if self.bundle is not None and self.table.currentRow() >= 0 else None

    def make_figure(self, preview=None):
        with matplotlib.rc_context():
            style(); fig = Figure(figsize=(8, 5), dpi=100)
            if preview is not None: draw_element(fig, *preview); return fig
            result = self.selected_result(); kind = self.view.currentData(); case = max(0, self.case_selector.currentIndex())
            if kind == 'structures': return structure_figure()
            if result is None: return fig
            if kind == 'comparison': return comparison_figure(self.bundle)
            fig._arraylab_sparse_view = kind
            if kind == 'overview': draw_layout_pattern(fig, self.bundle, result, case, self.pattern_view.currentData(), self.phase_view.currentData())
            elif kind in ('layout', 'pattern', 'cuts', 'coarray'): draw_detail_plot(fig, self.bundle, result, kind, case, self.pattern_view.currentData(), phase_mode=self.phase_view.currentData())
            elif kind == 'cases': draw_case_results(fig, self.bundle, result)
            elif kind == 'weights': draw_weight_comparison(fig, self.bundle, result, case)
            elif kind == 'quantization': draw_phase_comparison(fig, self.bundle, result, case)
            elif kind == 'element':
                c = Config(**self.bundle['config']); draw_element(fig, c, build_element(c, self.bundle['element_pattern']), self.bundle['scenarios'][case]['frequency_ghz'])
            return fig

    def install_figure(self, figure):
        if self.canvas.widgetlock.isowner(self.toolbar):
            self.canvas.widgetlock.release(self.toolbar)
        self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
        # Report figures may use 140 dpi for files. Reuse the desktop canvas's
        # DPI so those figures do not acquire enlarged, clipped annotations.
        original_dpi = self.canvas.figure._original_dpi
        figure.set_dpi(self.canvas.figure.dpi)
        figure._original_dpi = original_dpi
        self.figure = figure; self.canvas.figure = figure; figure.set_canvas(self.canvas)
        self.canvas.setMinimumWidth(500 if getattr(figure, '_arraylab_sparse_overview', False) else 100)
        self.reflow_current_figure()
        previous = self.toolbar
        self.toolbar = SparsePlotToolbar(self.canvas, self)
        self.plot_layout.replaceWidget(previous, self.toolbar)
        previous.hide(); previous.deleteLater()
        self.canvas.draw_idle()

    def reflow_current_figure(self):
        """Resize existing artists without replacing axes or navigation state."""
        figure = self.figure
        ratio = self.canvas.device_pixel_ratio
        figure.set_size_inches(max(1, self.canvas.width()) * ratio / figure.dpi,
                               max(1, self.canvas.height()) * ratio / figure.dpi, forward=False)
        from arraylab.shared.plotting.layout import reflow_plot_layout
        reflow_plot_layout(figure)
        if not getattr(figure, '_arraylab_responsive_plots', None):
            self.canvas.setMinimumHeight(360)
        if hasattr(figure, '_arraylab_footer_lines'):
            from arraylab.shared.plotting.metrics import draw_metric_footer
            draw_metric_footer(figure, figure._arraylab_footer_lines)
        if getattr(figure, '_arraylab_sparse_view', None) == 'pattern' and len(figure.axes) == 2:
            figure.set_layout_engine(None)
            height, width = max(1, self.canvas.height()), max(1, self.canvas.width())
            bottom = max(.20, getattr(figure, '_arraylab_footer_reserved', .12) + .09)
            plot_height = max(.15, .90-bottom)
            span = min(.6, plot_height * height / width)
            figure.axes[0].set_position([.48 - span / 2, bottom, span, plot_height]); figure.axes[0].set_anchor('C')
            figure.axes[1].set_position([.48 + span / 2 + .04, bottom, .018, plot_height])
            figure.axes[1].set_ylabel(''); figure.axes[1].set_xlabel('dB', fontsize=8, labelpad=1)
            figure.axes[0].title.set_fontsize(10)
            for axis in figure.axes: axis.tick_params(labelsize=8)

    def refresh_result_tables(self):
        """Update saved-result tables without replacing or redrawing the figure."""
        result = self.selected_result()
        if result is not None:
            ci = max(0, self.case_selector.currentIndex()); case = self.bundle['scenarios'][ci]; m = result['metrics']; config = Config(**self.bundle['config'])
            distance = '未定义（单阵元）' if m['min_distance_lambda_ref'] is None else f"{m['min_distance_lambda_ref']:.4g} λ0"
            self.summary.setText(f"{result_name(result)} · {m['elements']}阵元 · 理想最差PSL {m['validation_psl_db']:.2f} dB · 最小距 {distance} · {self.bundle['element_pattern']['name']}")
            phases = excitation_phases(result['xy'], case, config.phase_bits)
            ports = [{'index': int(idx), 'x': float(xy[0]), 'y': float(xy[1]), 'amplitude': float(w), 'ideal': float(ideal), 'actual': float(actual), 'code': int(code) if config.phase_bits else None}
                     for idx, xy, w, ideal, actual, code in zip(result['indices'], result['xy'], result['weights'], phases['ideal_phase_deg'], phases['command_phase_deg'], phases['phase_code'])]
            self.fill_table(self.port_table, [('index','候选编号'),('x','x/λ0'),('y','y/λ0'),('amplitude','相对幅度'),('ideal','理想相位°'),('actual','实际相位°'),('code','码字')], ports)
            cases = []
            for i, item in enumerate(result['scenario_metrics']):
                quant = result.get('phase_quantization', {}).get('scenarios', [])
                cases.append({**item, 'quantized_psl': quant[i]['validation_psl_db'] if quant else None, 'loss': quant[i]['steering_loss_db'] if quant else None})
            self.fill_table(self.case_table, [('frequency_ghz','GHz'),('theta','θ°'),('phi','φ°'),('validation_psl_db','理想PSL/dB'),('quantized_psl','量化PSL/dB'),('loss','扫描损失/dB'),('hpbw_u','HPBW u'),('hpbw_v','HPBW v')], cases)
        else:
            self.summary.setText('尚无稀疏方案。'); self.port_table.setRowCount(0); self.case_table.setRowCount(0)

    def redraw(self):
        if self._initializing: return
        was_preview = self.preview_active
        self.preview_active = False
        self.refresh_result_tables()
        self.install_figure(self.make_figure())
        if was_preview:
            if self.bundle is None:
                self.set_state('已退出待计算单元预览；尚无综合结果，请设置参数后开始综合。')
            else:
                try: dirty = self.config_from_inputs() != Config(**self.bundle['config'])
                except (ValueError, OverflowError): dirty = True
                self.set_state('已返回已计算快照；' + ('当前输入仍待综合。' if dirty else '结果与当前参数一致。'))
        self._history_ui_changed()

    def maximize_plot(self):
        dialog = QDialog(self); dialog.setWindowTitle('稀疏图形 · ' + self.view.currentText()); dialog.resize(1000, 720)
        layout = QVBoxLayout(dialog); canvas = FigureCanvasQTAgg(self.make_figure()); layout.addWidget(canvas, 1)
        layout.addWidget(NavigationToolbar2QT(canvas, dialog)); dialog.exec()

    def load_element_file(self, path):
        candidate = replace(self.config_from_inputs(), element_model='imported', element_file=str(Path(path).absolute()))
        element = build_element(candidate)
        for case in scenarios(candidate): element.steering_response(case)
        self.set_config(candidate)

    def import_element(self):
        path, _ = QFileDialog.getOpenFileName(self, '导入单元CSV', self.host.last_directory, 'CSV (*.csv)')
        if path:
            try: self.load_element_file(path)
            except (ValueError, OSError) as exc: QMessageBox.warning(self, '单元导入失败', str(exc))
    def load_example_element(self):
        try: self.load_element_file(resource_path('examples/element_grid_multifrequency.csv'))
        except (ValueError, OSError) as exc: QMessageBox.warning(self, '单元示例', str(exc))
    def save_element_template(self):
        path, _ = QFileDialog.getSaveFileName(self, '保存单元CSV模板', 'element.csv', 'CSV (*.csv)')
        if path:
            try: write_example_csv(path, self.element_import_mode.currentData() == 'axisymmetric')
            except (ValueError, OSError) as exc: QMessageBox.warning(self, '模板保存失败', str(exc))
    def preview_element(self):
        try:
            c = self.config_from_inputs().validate(); element = build_element(c)
            for case in scenarios(c): element.steering_response(case)
        except (ValueError, OSError) as exc: QMessageBox.warning(self, '单元检查失败', str(exc)); return
        self.preview_active = True
        self.tabs.setCurrentIndex(0); self.install_figure(self.make_figure((c, element, scenarios(c)[0]['frequency_ghz'])))
        self.set_state('当前图形为待计算单元预览；方案、表格和导出仍使用已计算快照。')
        self.navigation_changed.emit(); self.host.refresh_actions()

    def send_to_planar(self):
        if self._busy or self.bundle is None: return
        try: p, s, origin = planar_from_sparse(self.bundle, self.table.currentRow(), self.case_selector.currentIndex())
        except (ValueError, TypeError, KeyError) as exc: QMessageBox.warning(self, '无法传递此方案', str(exc)); return
        self.planar_requested.emit(p, s, origin)

    def export_path(self, path):
        if self.bundle is None: raise ValueError('请先完成稀疏计算。')
        self.last_export = save_bundle(self.bundle, path, report=True)

    def export(self):
        if self.bundle is None or self._busy: return
        parent = QFileDialog.getExistingDirectory(self, '选择稀疏结果父目录', self.host.last_directory)
        if not parent: return
        target = Path(parent) / datetime.now().strftime('sparse_%Y%m%d_%H%M%S_%f'); self.set_busy(True)
        try: self.export_path(target); self.host.statusBar().showMessage('稀疏报告已导出：' + str(target / 'report.html'))
        except Exception as exc: QMessageBox.warning(self, '导出失败', str(exc))
        finally: self.set_busy(False)

    def selected_array_snapshot(self):
        """Public handoff snapshot; input edits and caller writes cannot alter it."""
        if self._busy or self.thread is not None:
            raise ValueError('稀疏计算进行中，完成后再导入阵列。')
        result = self.selected_result()
        if result is None:
            raise ValueError('请先选择一项已计算的稀疏方案。')
        config = self.bundle['config']
        case = self.bundle['scenarios'][max(0, self.case_selector.currentIndex())]
        frequency = case['frequency_ghz']
        return {'positions_lambda': np.asarray(result['xy']).copy() * frequency / config['reference_ghz'],
                'weights': np.asarray(result['weights']).copy(), 'frequency_ghz': frequency,
                'element_model': config['element_model'],
                'origin': '二维稀疏：' + result_name(result)}
