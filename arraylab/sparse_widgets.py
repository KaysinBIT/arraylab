"""Native ArrayLab sparse page. Uses numerical and drawing functions, not the legacy GUI."""
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
import copy
import json
import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTabWidget, QScrollArea, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox, QDialog, QTextBrowser)
from . import __version__
from .sparse2d.core import Config, run, available_mask
from .sparse2d.weighting import METHODS, Weighting, add_weighting
from .sparse2d.elements import MODELS, build_element, write_example_csv
from .sparse2d.scenarios import scenarios
from .sparse2d.headless import save_bundle, read_json, validate_request
from .feed import write_text_atomic
from .sparse2d.phase_quantization import excitation_phases
from .sparse2d.runtime_paths import resource_path
from .sparse2d.plots import (style, draw_detail_plot, draw_case_results, draw_weight_comparison,
    draw_phase_comparison, draw_element, comparison_figure, structure_figure, result_name)
from .sparse_bridge import planar_from_sparse


class SparseCancelled(Exception):
    pass


class SparseWorker(QThread):
    progress = Signal(str)

    def __init__(self, config=None, weighting=None, parent=None):
        super().__init__(parent)
        self.config, self.weighting = config, weighting
        self.result = None; self.error = None; self.cancelled = False

    def tell(self, message):
        if self.isInterruptionRequested(): raise SparseCancelled()
        self.progress.emit(message)

    def run(self):
        try:
            self.result = run(self.config, self.tell) if self.weighting is None else add_weighting(*self.weighting, progress=self.tell)
            self.tell('完成')
        except SparseCancelled: self.cancelled = True
        except Exception as exc: self.error = f'{type(exc).__name__}: {exc}'


class SparsePanel(QWidget):
    planar_requested = Signal(object, object, object)

    def __init__(self, host):
        super().__init__()
        self.host = host; self.bundle = None; self.thread = None; self.last_export = None
        self._busy = False; self._last_error = None; self._initializing = True; self._last_selected_row = None
        self._cancel_pending = False; self._job_kind = None; self.inputs = {}
        self.build_parameters()
        layout = QVBoxLayout(self); layout.setContentsMargins(8, 6, 8, 6)
        actions = QHBoxLayout()
        self.run_button = QPushButton('开始综合'); self.run_button.clicked.connect(self.start)
        self.cancel_button = QPushButton('取消'); self.cancel_button.clicked.connect(self.cancel)
        self.weight_open_button = QPushButton('稀疏后加权'); self.weight_open_button.clicked.connect(self.open_weighting)
        self.export_button = QPushButton('导出结果'); self.export_button.clicked.connect(self.export)
        self.planar_button = QPushButton('送入平面阵'); self.planar_button.clicked.connect(self.send_to_planar)
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
        for label, key in [('阵列布局', 'layout'), ('二维方向图', 'pattern'), ('扫描切线', 'cuts'), ('差分共阵', 'coarray'),
                           ('工况比较', 'cases'), ('加权前后', 'weights'), ('量化复核', 'quantization'), ('单元快照', 'element'),
                           ('方法比较', 'comparison'), ('结构化共阵', 'structures')]: self.view.addItem(label, key)
        views.addWidget(QLabel('图形')); views.addWidget(self.view, 1)
        self.max_button = QPushButton('放大图形'); self.max_button.clicked.connect(self.maximize_plot); views.addWidget(self.max_button)
        pl.addLayout(views); self.figure = Figure(figsize=(7, 4), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure); self.canvas.setMinimumSize(100, 100); pl.addWidget(self.canvas, 1)
        self.toolbar = NavigationToolbar2QT(self.canvas, self); pl.addWidget(self.toolbar); self.tabs.addTab(plot, '图形')
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
        self.redraw()

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
            box = QSpinBox() if type(getattr(defaults, key)) is int else QDoubleSpinBox()
            if isinstance(box, QDoubleSpinBox): box.setDecimals(8)
            box.setRange(low, high); box.setSingleStep(step); box.setValue(getattr(defaults, key)); box.setMinimumWidth(65)
            box.setAccessibleName(label); box.valueChanged.connect(self.mark_stale); self.inputs[key] = box; form.addRow(label, box)
        def combo(form, label, pairs):
            box = QComboBox()
            for text, value in pairs: box.addItem(text, value)
            box.currentIndexChanged.connect(self.mark_stale); form.addRow(label, box); return box
        body, form = page('阵列')
        for spec in [('nx', 'X候选数', 4, 20, 2), ('ny', 'Y候选数', 4, 20, 2), ('k', '目标阵元K', 4, 400, 2),
                     ('spacing', '间距/λ0', .1, 2, .1), ('reference_ghz', '参考GHz', .1, 1000, 1), ('min_spacing', '最小距/λ0', 0, 100, .1)]: number(form, *spec)
        self.hole_shape = combo(form, '中心禁布', [('无', 'none'), ('矩形', 'rectangle'), ('圆形', 'circle')])
        number(form, 'hole_x', '孔宽X/直径λ0', .01, 100, .25); number(form, 'hole_y', '孔高Y/λ0', .01, 100, .25)
        self.availability = QLabel(); self.availability.setWordWrap(True); body.addWidget(self.availability); body.addStretch()
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
            scan_angles=self.joint_scans.text(), frequencies_ghz=self.joint_frequencies.text())

    def set_config(self, config):
        if self._busy: return
        self._initializing = True
        try:
            for key, box in self.inputs.items():
                if key == 'seed': box.setText(str(config.seed))
                else: box.setValue(getattr(config, key))
            for box, value in [(self.hole_shape, config.hole_shape), (self.element_model, config.element_model),
                               (self.element_axis, config.element_dipole_axis), (self.element_import_mode, config.element_import_mode)]: box.setCurrentIndex(box.findData(value))
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
        if self.host.tabs.currentWidget() is self and hasattr(self.host, 'state_label'): self.host.state_label.setText(message)

    def update_controls(self):
        if self._initializing: return
        self.inputs['hole_x'].setEnabled(self.hole_shape.currentData() != 'none'); self.inputs['hole_y'].setEnabled(self.hole_shape.currentData() == 'rectangle')
        model = self.element_model.currentData(); self.element_axis.setEnabled(model == 'dipole'); self.inputs['element_dipole_length'].setEnabled(model == 'dipole')
        self.inputs['element_patch_hpbw'].setEnabled(model == 'patch'); self.element_import_mode.setEnabled(model == 'imported'); self.element_file.setEnabled(model == 'imported')
        try: self.availability.setText(f'可用候选位置：{int(available_mask(self.config_from_inputs()).sum())}；按中心对称成对选点。')
        except ValueError: self.availability.setText('请检查候选网格与禁布区。')
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

    def start(self):
        if self.thread is not None or self._busy: return
        try:
            config = self.config_from_inputs().validate(); element = build_element(config)
            for case in scenarios(config): element.steering_response(case)
        except (ValueError, OSError) as exc:
            self._last_error = str(exc); QMessageBox.warning(self, '稀疏参数检查', str(exc)); return
        self.begin_worker(SparseWorker(config=config, parent=self), 'synthesis')

    def begin_worker(self, worker, kind):
        self._last_error = None; self._cancel_pending = False; self._job_kind = kind; self.thread = worker; self.set_busy(True)
        self.set_state('正在计算，可取消；上一份结果保留。'); worker.progress.connect(self.progress_message); worker.finished.connect(self.complete); worker.start()

    def progress_message(self, text):
        self.set_state('正在计算 · ' + text); self.host.statusBar().showMessage('稀疏 · ' + text)

    def cancel(self):
        if self.thread is not None:
            self._cancel_pending = True; self.thread.requestInterruption(); self.set_state('已请求取消，当前计算阶段结束后停止。')

    def complete(self):
        worker = self.thread; self.thread = None
        if worker.cancelled or self._cancel_pending: self.set_state('已取消，保留上一份结果。')
        elif worker.error:
            self._last_error = worker.error; self.set_state('计算失败，保留上一份结果：' + worker.error)
        else: self.receive(worker.result)
        worker.deleteLater(); self._job_kind = None; self.set_busy(False)

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

    def receive(self, bundle):
        bundle['integration'] = {'arraylab_version': __version__, 'source_engine': 'Sparse2D 1.3.0'}; self.bundle = bundle; self._last_selected_row = None
        self.table.blockSignals(True); rows = [{'name': result_name(r), **r['metrics']} for r in bundle['results']]
        self.fill_table(self.table, [('name', '方案'), ('elements', '阵元'), ('validation_psl_db', '理想PSL/dB'), ('amplitude_dynamic_range_db', '幅度范围/dB'),
            ('hpbw_u', 'HPBW u'), ('hpbw_v', 'HPBW v'), ('min_distance_lambda_ref', '最小距/λ0'), ('spacing_violations', '间距违规'), ('unique_lags', '不同lag')], rows)
        self.scheme_selector.blockSignals(True); self.scheme_selector.clear(); self.scheme_selector.addItems([r['name'] for r in rows]); self.scheme_selector.blockSignals(False)
        self.table.blockSignals(False); self.case_selector.blockSignals(True); self.case_selector.clear()
        for case in bundle['scenarios']: self.case_selector.addItem(case['label'], case['index'])
        self.case_selector.blockSignals(False); self.phase_view.blockSignals(True)
        self.phase_view.setCurrentIndex(1 if bundle['config']['phase_bits'] else 0); self.phase_view.blockSignals(False)
        index = bundle.get('last_weighted_index')
        if index is None: index = min((i for i, r in enumerate(bundle['results']) if r['metrics']['elements'] == bundle['config']['k'] and r['name'] != 'L1 weighted solution'), key=lambda i: bundle['results'][i]['metrics']['validation_psl_db'])
        self.table.selectRow(index); self.select_result()
        try: dirty = self.config_from_inputs() != Config(**bundle['config'])
        except ValueError: dirty = True
        suffix = '当前输入尚未综合，本次结果仍对应已计算参数。' if dirty else '结果与当前参数一致。'
        self.set_state(f"已完成 {len(bundle['results'])} 个方案、{len(bundle['scenarios'])} 个工况；" + suffix)
        self.host.statusBar().showMessage('稀疏结果已更新；选中方案可送入平面阵电气设计。')
        if self._job_kind == 'weight': self.view.setCurrentIndex(self.view.findData('weights')); self.tabs.setCurrentIndex(0)

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
            if kind in ('layout', 'pattern', 'cuts', 'coarray'): draw_detail_plot(fig, self.bundle, result, kind, case, self.pattern_view.currentData(), phase_mode=self.phase_view.currentData())
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
        self.figure = figure; self.canvas.figure = figure; figure.set_canvas(self.canvas)
        figure.set_size_inches(max(1, self.canvas.width()) / figure.dpi, max(1, self.canvas.height()) / figure.dpi, forward=False)
        if self.view.currentData() == 'pattern' and len(figure.axes) == 2:
            figure.set_layout_engine(None)
            height, width = max(1, self.canvas.height()), max(1, self.canvas.width())
            span = min(.6, .64 * height / width)
            figure.axes[0].set_position([.48 - span / 2, .20, span, .64]); figure.axes[0].set_anchor('C')
            figure.axes[1].set_position([.48 + span / 2 + .04, .20, .018, .64])
            figure.axes[1].set_ylabel(''); figure.axes[1].set_xlabel('dB', fontsize=8, labelpad=1)
            figure.axes[0].title.set_fontsize(10)
            for axis in figure.axes: axis.tick_params(labelsize=8)
        previous = self.toolbar
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.plot_layout.replaceWidget(previous, self.toolbar)
        previous.hide(); previous.deleteLater()
        self.canvas.draw_idle()

    def redraw(self):
        if self._initializing: return
        result = self.selected_result()
        if result is not None:
            ci = max(0, self.case_selector.currentIndex()); case = self.bundle['scenarios'][ci]; m = result['metrics']; config = Config(**self.bundle['config'])
            self.summary.setText(f"{result_name(result)} · {m['elements']}阵元 · 理想最差PSL {m['validation_psl_db']:.2f} dB · 最小距 {m['min_distance_lambda_ref']:.4g} λ0 · {self.bundle['element_pattern']['name']}")
            phases = excitation_phases(result['xy'], case, config.phase_bits)
            ports = [{'index': int(idx), 'x': float(xy[0]), 'y': float(xy[1]), 'amplitude': float(w), 'ideal': float(ideal), 'actual': float(actual), 'code': int(code) if config.phase_bits else None}
                     for idx, xy, w, ideal, actual, code in zip(result['indices'], result['xy'], result['weights'], phases['ideal_phase_deg'], phases['command_phase_deg'], phases['phase_code'])]
            self.fill_table(self.port_table, [('index','候选编号'),('x','x/λ0'),('y','y/λ0'),('amplitude','相对幅度'),('ideal','理想相位°'),('actual','实际相位°'),('code','码字')], ports)
            cases = []
            for i, item in enumerate(result['scenario_metrics']):
                quant = result.get('phase_quantization', {}).get('scenarios', [])
                cases.append({**item, 'quantized_psl': quant[i]['validation_psl_db'] if quant else None, 'loss': quant[i]['steering_loss_db'] if quant else None})
            self.fill_table(self.case_table, [('frequency_ghz','GHz'),('theta','θ°'),('phi','φ°'),('validation_psl_db','理想PSL/dB'),('quantized_psl','量化PSL/dB'),('loss','扫描损失/dB'),('hpbw_u','HPBW u'),('hpbw_v','HPBW v')], cases)
        self.install_figure(self.make_figure())

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
        self.tabs.setCurrentIndex(0); self.install_figure(self.make_figure((c, element, scenarios(c)[0]['frequency_ghz'])))
        self.set_state('当前图形为待计算单元预览；方案、表格和导出仍使用已计算快照。')

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
