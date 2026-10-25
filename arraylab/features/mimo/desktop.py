"""Native ArrayLab radar MIMO page with independent Tx and Rx apertures."""
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
import csv
import math
import threading

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSignalBlocker
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QTabWidget,
    QScrollArea, QLabel, QLineEdit, QComboBox, QPlainTextEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox)

from arraylab.features.mimo.model import MIMOParameters, mimo_parameters_from_dict, calculate_mimo
from arraylab.framework.parallel import ComputationCancelled, requested_workers
from arraylab.features.mimo.plotting import draw_mimo_figure
from arraylab.shared.plotting.layout import reflow_plot_layout
from arraylab.shared.plotting.metrics import draw_metric_footer


def parse_aperture_text(text, dimension):
    """Unambiguous columns: 1D x[,amplitude]; 2D x,y[,amplitude]."""
    if len(text) > 256_000:
        raise ValueError('坐标文本超过256000字符，请缩小阵列或检查输入。')
    positions, weights = [], []
    explicit_weight = False
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try: columns = next(csv.reader([line], strict=True))
        except csv.Error as exc: raise ValueError(f'第{line_number}行CSV格式无效。') from exc
        tokens = [item.strip() for item in columns]
        headers = (['x'], ['x', 'amplitude']) if dimension == 'linear' else (['x', 'y'], ['x', 'y', 'amplitude'])
        if not positions and tokens in headers:
            continue
        if not positions and tokens in (['x'], ['x', 'amplitude'], ['x', 'y'], ['x', 'y', 'amplitude']):
            raise ValueError('坐标文件表头与当前维度不匹配；一维为x[,amplitude]，二维为x,y[,amplitude]。')
        allowed = (1, 2) if dimension == 'linear' else (2, 3)
        if len(tokens) not in allowed:
            example = 'x,幅度' if dimension == 'linear' else 'x,y,幅度'
            raise ValueError(f'第{line_number}行列数无效；本模式为{example}，幅度可省略。')
        try:
            values = list(map(float, tokens))
        except ValueError as exc:
            raise ValueError(f'第{line_number}行须使用逗号分隔的数值。') from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f'第{line_number}行包含非有限数。')
        if dimension == 'linear':
            x, y, weight = values[0], 0., values[1] if len(values) == 2 else 1.
            explicit_weight |= len(values) == 2
        else:
            x, y, weight = values[0], values[1], values[2] if len(values) == 3 else 1.
            explicit_weight |= len(values) == 3
        if weight <= 0:
            raise ValueError(f'第{line_number}行幅度须大于0；删去不使用的阵元行即可形成稀疏布局。')
        positions.append((x, y)); weights.append(weight)
    if not positions:
        raise ValueError('至少输入一个阵元坐标。')
    return tuple(positions), tuple(weights) if explicit_weight else ()


def aperture_text(positions, weights, dimension):
    explicit_weight = bool(len(weights))
    weights = list(weights) if explicit_weight else [1.] * len(positions)
    rows = []
    for (x, y), amplitude in zip(positions, weights):
        values = ((x, amplitude) if dimension == 'linear' else (x, y, amplitude)) if explicit_weight else ((x,) if dimension == 'linear' else (x, y))
        rows.append(','.join(str(float(value)) for value in values))
    return '\n'.join(rows)


class MIMOWorker(QThread):
    progress = Signal(str)

    def __init__(self, parameters, parent=None):
        super().__init__(parent)
        self.parameters = parameters; self.workers = requested_workers()
        self.stop = threading.Event(); self.result = None; self.error = None; self.cancelled = False

    def tell(self, *args):
        if self.stop.is_set() or self.isInterruptionRequested():
            raise ComputationCancelled()
        self.progress.emit(' / '.join(map(str, args)))

    def run(self):
        try:
            self.result = calculate_mimo(self.parameters, workers=self.workers,
                cancelled=lambda: self.stop.is_set() or self.isInterruptionRequested(), progress=self.tell)
        except ComputationCancelled:
            self.cancelled = True
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'


class MIMOPlotToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, panel):
        self.panel = panel
        super().__init__(canvas, panel)

    def _update_view(self):
        super()._update_view()
        self.panel.reflow()


class MIMOPanel(QWidget):
    """Input changes never mutate the last successful result or its exports."""
    def __init__(self, window):
        super().__init__()
        self.host = window; self.report = None; self.thread = None; self._busy = False
        self._changing = False; self._cancel_pending = False; self._last_error = None
        self.last_export = None; self.provenance = {}; self.inputs = {}
        self._dimension_key = 'linear'
        self.parameter_widget = QTabWidget()
        self._build_parameters()
        body = QVBoxLayout(self); body.setContentsMargins(8, 6, 8, 6)
        self.state = QLabel('分别设置Tx与Rx坐标，按F5计算虚拟阵列与波束指标。')
        self.state.setObjectName('state'); self.state.setWordWrap(True); body.addWidget(self.state)
        self.summary = QLabel('尚无MIMO结果。'); self.summary.setWordWrap(True); body.addWidget(self.summary)
        row = QHBoxLayout(); row.addWidget(QLabel('结果视图'))
        self.view = QComboBox()
        for label, key in [('物理 / 虚拟阵列布局', 'layout'), ('Tx / Rx / 虚拟切面', 'cut'),
                           ('二维虚拟方向图', 'map'), ('Tx–Rx通道映射', 'channels'), ('独特虚拟位置', 'unique')]:
            self.view.addItem(label, key)
        self.view.setAccessibleName('MIMO结果视图'); row.addWidget(self.view, 1); body.addLayout(row)
        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure); self.canvas.setMinimumSize(500, 360)
        body.addWidget(self.canvas, 1)
        self.toolbar = MIMOPlotToolbar(self.canvas, self); body.addWidget(self.toolbar)
        self.plot_layout = body
        self.table = QTableWidget(); self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True); self.table.hide(); body.addWidget(self.table, 1)
        self.table_summary = QLabel(); self.table_summary.setWordWrap(True); self.table_summary.hide(); body.addWidget(self.table_summary)
        note = QLabel('理想正交波形、远场近共址模型。Tx+Rx为等效接收虚拟位置；通道数不等于RF链数，数字响应不等于实际辐射增益。')
        note.setWordWrap(True); note.setObjectName('scope'); body.addWidget(note)
        self.resize_timer = QTimer(self); self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.reflow)
        self.view.currentIndexChanged.connect(self.redraw)
        self.apply_configuration(MIMOParameters())
        self.state.setText('分别设置Tx与Rx坐标，按F5计算虚拟阵列与波束指标。')
        self.redraw()

    def _build_parameters(self):
        general = QWidget(); body = QVBoxLayout(general); form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows); body.addLayout(form)
        for key, label, choices in [('dimension', '阵列维度', [('一维线阵', 'linear'), ('二维平面阵', 'planar')]),
                                    ('coordinate_unit', '坐标单位', [('λ（当前频率）', 'lambda'), ('mm', 'mm')])]:
            control = QComboBox()
            for title, value in choices: control.addItem(title, value)
            control.setAccessibleName('MIMO ' + label); self.inputs[key] = control; form.addRow(label, control)
        self.inputs['coordinate_unit'].setToolTip('按所选单位解释当前数值；改频率时λ坐标代表的毫米位置随之改变。')
        for key, label, default in [('frequency_ghz', '频率 (GHz)', '10'), ('theta_deg', '目标 θ (°)', '0'),
                                     ('phi_deg', '目标 φ (°)', '0'), ('cut_phi_deg', '切面 φ (°)', '0'),
                                     ('step_deg', '切面步长 (°)', '0.1')]:
            edit = QLineEdit(default); edit.setAccessibleName('MIMO ' + label)
            self.inputs[key] = edit; form.addRow(label, edit); edit.textChanged.connect(self.mark_dirty)
        grid = QComboBox()
        for size in (91, 181, 361): grid.addItem(f'{size} × {size}', size)
        self.inputs['map_size'] = grid; form.addRow('u-v采样网格', grid)
        grid.currentIndexChanged.connect(self.mark_dirty)
        self.angle_note = QLabel(); self.angle_note.setWordWrap(True); body.addWidget(self.angle_note)
        presets = QHBoxLayout()
        button = QPushButton('二维稀疏例题')
        button.clicked.connect(lambda: self.load_example('planar')); presets.addWidget(button)
        body.addLayout(presets)
        note = QLabel('幅度是数字通道合成的可分离系数，默认均匀；并非发射机功率。阵元列表只列出实际使用位置，不要求规则网格。')
        note.setWordWrap(True); body.addWidget(note); body.addStretch()
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(general)
        self.parameter_widget.addTab(scroll, '模型与扫描')
        self.coordinate_editors = {}; self.coordinate_notes = {}
        for side, label in [('tx', 'Tx发射阵列'), ('rx', 'Rx接收阵列')]:
            page = QWidget(); content = QVBoxLayout(page)
            hint = QLabel(); hint.setWordWrap(True); content.addWidget(hint); self.coordinate_notes[side] = hint
            editor = QPlainTextEdit(); editor.setAccessibleName('MIMO ' + label + '坐标与幅度')
            editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap); editor.setMinimumHeight(200)
            editor.textChanged.connect(self.mark_dirty); self.coordinate_editors[side] = editor; content.addWidget(editor, 1)
            load = QPushButton('载入坐标CSV…'); load.clicked.connect(lambda checked=False, s=side: self.import_aperture_dialog(s))
            content.addWidget(load)
            scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(page)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.parameter_widget.addTab(scroll, label)
        self.inputs['dimension'].currentIndexChanged.connect(self.dimension_changed)
        self.inputs['coordinate_unit'].currentIndexChanged.connect(self.mark_dirty)
        self.inputs['coordinate_unit'].currentIndexChanged.connect(self.update_parameter_hints)
        self.update_parameter_hints()

    def update_parameter_hints(self):
        linear = self.inputs['dimension'].currentData() == 'linear'
        unit = 'λ' if self.inputs['coordinate_unit'].currentData() == 'lambda' else 'mm'
        unit_label = 'λ（当前频率的波长）' if unit == 'λ' else 'mm（毫米）'
        for side, hint in self.coordinate_notes.items():
            if linear:
                note = (f'每行填写一个{side.upper()}阵元的x坐标，单位：{unit_label}。\n'
                        f'例如分两行填写0和2，表示两个阵元分别位于0{unit}和2{unit}。\n'
                        '如需加权，填写“x,权重”，例如“2,0.5”；只填坐标时权重为1。\n')
            else:
                note = (f'每行填写一个{side.upper()}阵元的“x,y”坐标，用英文逗号分隔，单位：{unit_label}。\n'
                        f'例如“1,2”表示一个阵元，位置为x=1{unit}、y=2{unit}。\n'
                        '如需加权，填写“x,y,权重”，例如“1,2,0.5”；省略权重时为1。\n')
            hint.setText(note + '权重须大于0，表示数字合成的相对幅度，不是发射功率。')
        self.angle_note.setText('θ从+z朝+x，范围−90°至90°；一维φ和切面φ固定为0°。' if linear else
                               'θ从+z量起（0–90°），φ从+x朝+y（−180–180°）。切面φ可与目标φ不同。')
        for key in ('phi_deg', 'cut_phi_deg'): self.inputs[key].setEnabled(not linear)

    def dimension_changed(self):
        if self._changing: return
        new = self.inputs['dimension'].currentData(); old = self._dimension_key
        if new == old: return
        try:
            arrays = {side: parse_aperture_text(editor.toPlainText(), old) for side, editor in self.coordinate_editors.items()}
            if new == 'linear' and any(y != 0 for positions, _ in arrays.values() for x, y in positions):
                raise ValueError('一维要求Tx和Rx的全部y坐标为0；已保留二维坐标，请先编辑后再切换。')
            theta = float(self.inputs['theta_deg'].text()); phi = float(self.inputs['phi_deg'].text())
            if not math.isfinite(theta) or not math.isfinite(phi): raise ValueError('请先修正非有限指向角。')
            if new == 'planar' and theta < 0:
                theta, phi = -theta, 180.
            elif new == 'linear':
                theta, phi = math.degrees(math.asin(float(np.clip(math.sin(math.radians(theta))*math.cos(math.radians(phi)), -1, 1)))), 0.
        except ValueError as exc:
            with QSignalBlocker(self.inputs['dimension']):
                self.inputs['dimension'].setCurrentIndex(self.inputs['dimension'].findData(old))
            QMessageBox.warning(self, '维度未切换', str(exc)); return
        self._changing = True
        try:
            for side, (positions, weights) in arrays.items():
                self.coordinate_editors[side].setPlainText(aperture_text(positions, weights, new))
            self.inputs['theta_deg'].setText(str(theta)); self.inputs['phi_deg'].setText(str(phi))
            if new == 'linear': self.inputs['cut_phi_deg'].setText('0')
            self._dimension_key = new; self.update_parameter_hints()
        finally: self._changing = False
        self.mark_dirty()

    def config(self):
        values = {}
        for key, control in self.inputs.items():
            if isinstance(control, QComboBox): values[key] = control.currentData()
            else:
                try: values[key] = float(control.text().strip())
                except ValueError as exc: raise ValueError(control.accessibleName() + '须填写有效数字。') from exc
        for side, editor in self.coordinate_editors.items():
            try: positions, weights = parse_aperture_text(editor.toPlainText(), values['dimension'])
            except ValueError as exc: raise ValueError(side.upper() + '：' + str(exc)) from exc
            values[side + '_positions'] = positions; values[side + '_weights'] = weights
        return mimo_parameters_from_dict(values)

    read_parameters = config

    def apply_configuration(self, parameters):
        if not isinstance(parameters, MIMOParameters): parameters = mimo_parameters_from_dict(parameters)
        parameters.validate(); self._changing = True
        try:
            values = asdict(parameters)
            for key, control in self.inputs.items():
                if isinstance(control, QComboBox):
                    index = control.findData(values[key])
                    if index < 0 and key == 'map_size': control.addItem(f"{values[key]} × {values[key]}", values[key]); index = control.count()-1
                    control.setCurrentIndex(index)
                else: control.setText(str(float(values[key])))
            for side, editor in self.coordinate_editors.items():
                editor.setPlainText(aperture_text(values[side+'_positions'], values[side+'_weights'], values['dimension']))
            self._dimension_key = parameters.dimension; self.update_parameter_hints()
        finally: self._changing = False
        self.mark_dirty()

    apply_config = apply_configuration

    def load_example(self, kind):
        if self._busy: return
        p = MIMOParameters()
        if kind == 'planar':
            p = replace(p, dimension='planar', tx_positions=((0., 0.), (1.5, 0.), (0., 1.)),
                        rx_positions=((0., 0.), (.5, 0.), (0., .5), (.5, .5)), theta_deg=20., phi_deg=30., cut_phi_deg=30.)
        self.apply_configuration(p); self.provenance = {}

    def set_aperture(self, side, positions_lambda, weights, frequency_ghz, provenance=None):
        if side not in ('tx', 'rx'): raise ValueError('MIMO导入侧须为tx或rx。')
        if self._busy or self.thread is not None: raise ValueError('请先完成或取消MIMO计算。')
        original = self.config(); wavelength = MIMOParameters(frequency_ghz=float(frequency_ghz)).validate().wavelength_mm
        imported = np.asarray(positions_lambda, dtype=float)
        if imported.ndim != 2 or imported.shape[1] != 2: raise ValueError('导入阵列须为x-y坐标。')
        if not np.isfinite(imported).all(): raise ValueError('导入坐标含非有限数。')
        other = 'rx' if side == 'tx' else 'tx'
        other_mm = original.positions(other) * original.wavelength_mm
        imported_mm = imported * wavelength
        dimension = 'planar' if np.any(imported_mm[:, 1] != 0) or np.any(other_mm[:, 1] != 0) else original.dimension
        values = asdict(original)
        values.update(frequency_ghz=float(frequency_ghz), coordinate_unit='mm', dimension=dimension)
        if dimension == 'planar' and original.dimension == 'linear' and original.theta_deg < 0:
            values.update(theta_deg=-original.theta_deg, phi_deg=180.)
        values[side+'_positions'] = imported_mm.tolist(); values[side+'_weights'] = list(weights)
        values[other+'_positions'] = other_mm.tolist()
        p = mimo_parameters_from_dict(values)
        self.apply_configuration(p); self.provenance[side] = provenance or {}
        self.state.setText(f'已载入{side.upper()}阵列，另一侧原物理坐标保留；当前使用mm，按F5计算。原结果未改变。')

    def mark_dirty(self, *_):
        if not self._changing and hasattr(self, 'state'):
            self.state.setText('输入已改变，尚未计算；图形、表格与导出继续对应上一份成功结果。')

    def start(self):
        if self.thread is not None or self._busy: return
        try: parameters = self.config()
        except (ValueError, TypeError, ArithmeticError) as exc:
            QMessageBox.warning(self, 'MIMO参数检查', str(exc)); return
        self._busy = True; self._cancel_pending = False; self._last_error = None
        self.parameter_widget.setEnabled(False); self.state.setText('正在构建MIMO虚拟通道和方向图…')
        worker = MIMOWorker(parameters, self); self.thread = worker
        worker.progress.connect(self.state.setText); worker.finished.connect(self.complete)
        worker.start(); self.host.refresh_actions()

    def cancel(self):
        if self.thread is not None:
            self._cancel_pending = True; self.thread.stop.set(); self.thread.requestInterruption()
            self.state.setText('正在取消MIMO计算；完成回收后保留原结果。')

    def complete(self):
        worker = self.thread
        if worker is None: return
        try:
            if worker.cancelled or self._cancel_pending: self.state.setText('计算已取消，原MIMO结果保留。')
            elif worker.error:
                self._last_error = worker.error; self.state.setText('计算未完成，原结果保留：' + worker.error)
            else: self.accept_result(worker.result)
        except Exception as exc:
            self._last_error = f'{type(exc).__name__}: {exc}'
            self.state.setText('结果显示未完成，原结果保留：' + self._last_error)
        finally:
            self.thread = None; worker.deleteLater(); self._busy = False
            self.parameter_widget.setEnabled(True); self.host.refresh_actions()
            if getattr(self.host, '_mimo_close_pending', False):
                self.host._mimo_close_pending = False; QTimer.singleShot(0, self.host.close)

    def _make_figure(self, report, view):
        figure = Figure(figsize=(8, 5), dpi=100)
        if report is None:
            axis = figure.add_subplot(111); axis.set_axis_off()
            axis.text(.5, .5, '设置Tx、Rx阵列后按F5计算MIMO参数', ha='center', va='center', transform=axis.transAxes)
        else: draw_mimo_figure(figure, report, view if view in ('layout', 'cut', 'map') else 'layout', chinese=True)
        return figure

    def _install_figure(self, figure):
        if self.canvas.widgetlock.isowner(self.toolbar): self.canvas.widgetlock.release(self.toolbar)
        self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
        original_dpi = self.canvas.figure._original_dpi
        figure.set_dpi(self.canvas.figure.dpi); figure._original_dpi = original_dpi
        self.figure = figure; self.canvas.figure = figure; figure.set_canvas(self.canvas)
        old = self.toolbar; self.toolbar = MIMOPlotToolbar(self.canvas, self)
        self.plot_layout.replaceWidget(old, self.toolbar); old.hide(); old.deleteLater()
        self.reflow()

    def _input_state(self):
        return ({key: control.currentIndex() if isinstance(control, QComboBox) else control.text() for key, control in self.inputs.items()},
                {side: editor.toPlainText() for side, editor in self.coordinate_editors.items()}, self._dimension_key)

    def _restore_inputs(self, state):
        self._changing = True
        try:
            for key, value in state[0].items():
                control = self.inputs[key]
                if isinstance(control, QComboBox): control.setCurrentIndex(value)
                else: control.setText(value)
            for side, text in state[1].items(): self.coordinate_editors[side].setPlainText(text)
            self._dimension_key = state[2]; self.update_parameter_hints()
        finally: self._changing = False

    def accept_result(self, report, *, restore_inputs=False, view=None):
        chosen = view or self.view.currentData()
        if self.view.findData(chosen) < 0: raise ValueError('未知MIMO结果视图。')
        figure = self._make_figure(report, chosen)
        old = (self.report, self.figure, self.view.currentIndex(), self._input_state())
        try:
            if restore_inputs: self.apply_configuration(mimo_parameters_from_dict(report['parameters']))
            self.report = report
            with QSignalBlocker(self.view): self.view.setCurrentIndex(self.view.findData(chosen))
            self._install_figure(figure); self._show_table_or_plot(); self.update_summary()
        except Exception:
            self.report = old[0]; self._restore_inputs(old[3])
            with QSignalBlocker(self.view): self.view.setCurrentIndex(old[2])
            self._install_figure(old[1]); self._show_table_or_plot(); self.update_summary()
            raise
        self._last_error = None
        self.state.setText('已载入完整MIMO记录；结果未重新计算。' if restore_inputs else 'MIMO计算完成，图形和表格对应本次成功输入。')
        self.host.statusBar().showMessage(self.state.text()); self.host.refresh_actions()

    def update_summary(self):
        if self.report is None: self.summary.setText('尚无MIMO结果。'); return
        c = self.report['counts']; p = self.report['parameters']; g = self.report['geometry']
        self.summary.setText(f"Tx {c['tx_elements']} · Rx {c['rx_elements']} · 配对通道 {c['tx_rx_channels']} · "
                             f"独特虚拟位置 {c['unique_virtual_positions']} · 重复 {100*c['redundancy_ratio']:.3g}% · {p['frequency_ghz']:g} GHz")
        span = g['virtual']['span_lambda']; spacing = g['virtual']['min_spacing_lambda']
        self.table_summary.setText(f"虚拟孔径跨度 x={span[0]:.6g}λ，y={span[1]:.6g}λ；最小独特间距："
            + ('未定义' if spacing is None else f'{spacing:.6g}λ')
            + '。索引从1起，通道按Tx优先排列；重复虚拟位置仍保留全部通道贡献。RF链数与实际增益未推定。')

    def _show_table_or_plot(self):
        key = self.view.currentData(); table_view = key in ('channels', 'unique')
        self.table.setVisible(table_view); self.table_summary.setVisible(table_view)
        self.canvas.setVisible(not table_view); self.toolbar.setVisible(not table_view)
        if not table_view or self.report is None:
            if self.report is None: self.table.setRowCount(0)
            return
        if key == 'channels':
            columns = [('channel','通道'),('tx','Tx'),('rx','Rx'),('x_lambda','虚拟x/λ'),('y_lambda','虚拟y/λ'),('weight','数字幅度'),('unique_index','独特位置')]
            rows = self.report['geometry']['pairs']
        else:
            columns = [('index','位置编号'),('x_lambda','x/λ'),('y_lambda','y/λ'),('multiplicity','通道重数'),('weight','相干幅度和')]
            rows = self.report['geometry']['virtual_unique']
        self.table.setColumnCount(len(columns)); self.table.setHorizontalHeaderLabels([label for _, label in columns])
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, (field, _) in enumerate(columns):
                value = row[field]; text = f'{value:.8g}' if isinstance(value, float) else str(value)
                self.table.setItem(i, j, QTableWidgetItem(text))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    def redraw(self, *_):
        self._install_figure(self._make_figure(self.report, self.view.currentData()))
        self._show_table_or_plot(); self.update_summary()
        if getattr(self.host, '_workflow_ready', False) and getattr(self.host, 'mimo_panel', None) is self: self.host.refresh_actions()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'resize_timer'): self.resize_timer.start(60)

    def reflow(self):
        if not hasattr(self, 'canvas'): return
        ratio = self.canvas.device_pixel_ratio
        self.figure.set_size_inches(max(1, self.canvas.width())*ratio/self.figure.dpi,
                                    max(1, self.canvas.height())*ratio/self.figure.dpi, forward=False)
        if getattr(self.figure, '_arraylab_responsive_plots', None): reflow_plot_layout(self.figure)
        else: self.canvas.setMinimumHeight(360)
        if hasattr(self.figure, '_arraylab_footer_lines'): draw_metric_footer(self.figure, self.figure._arraylab_footer_lines)
        self.canvas.draw_idle()

    def import_aperture_dialog(self, side):
        path, _ = QFileDialog.getOpenFileName(self, '导入'+side.upper()+'坐标', self.host.last_directory, 'CSV (*.csv);;文本 (*.txt)')
        if not path: return
        try:
            if Path(path).stat().st_size > 1024*1024: raise ValueError('坐标文件超过1 MiB，请检查内容。')
            text = Path(path).read_text(encoding='utf-8-sig')
            positions, weights = parse_aperture_text(text, self.inputs['dimension'].currentData())
            # Validate all fields before replacing just this input aperture.
            p = self.config(); values = asdict(p); values[side+'_positions'] = positions; values[side+'_weights'] = weights
            mimo_parameters_from_dict(values)
            self.coordinate_editors[side].setPlainText(aperture_text(positions, weights, p.dimension))
        except (OSError, ValueError, TypeError) as exc: QMessageBox.warning(self, 'MIMO坐标导入失败', str(exc))

    def save_path(self, path):
        from arraylab.features.mimo.io import save_mimo_configuration
        save_mimo_configuration(path, self.config())

    def open_path(self, path):
        if self._busy or self.thread is not None: raise ValueError('请先完成或取消MIMO计算。')
        from arraylab.features.mimo.io import load_mimo_configuration
        self.apply_configuration(load_mimo_configuration(path))

    load_path = open_path

    def save_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, '保存MIMO参数', str(Path(self.host.last_directory)/'mimo.json'), 'JSON (*.json)')
        if path:
            try: self.save_path(path)
            except (ValueError, OSError) as exc: QMessageBox.warning(self, '保存失败', str(exc))

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, '打开MIMO参数', self.host.last_directory, 'JSON (*.json)')
        if path:
            try: self.open_path(path)
            except (ValueError, OSError) as exc: QMessageBox.warning(self, '打开失败', str(exc))

    load_dialog = open_dialog

    def save_record_path(self, path):
        if self.report is None: raise ValueError('请先完成MIMO计算。')
        from arraylab.features.mimo.io import save_mimo_record
        save_mimo_record(path, self.report, view=self.view.currentData())

    def load_record_path(self, path):
        if self._busy or self.thread is not None: raise ValueError('请先完成或取消MIMO计算。')
        from arraylab.features.mimo.io import load_mimo_record
        record = load_mimo_record(path)
        self.accept_result(record['report'], restore_inputs=True, view=record['view'])

    def save_record_dialog(self):
        if self.report is None: return
        path, _ = QFileDialog.getSaveFileName(self, '保存MIMO计算记录', str(Path(self.host.last_directory)/'mimo-record.json'), 'JSON (*.json)')
        if path:
            try: self.save_record_path(path)
            except (ValueError, OSError) as exc: QMessageBox.warning(self, '记录保存失败', str(exc))

    def load_record_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, '打开MIMO计算记录', self.host.last_directory, 'JSON (*.json)')
        if path:
            try: self.load_record_path(path)
            except (ValueError, OSError) as exc: QMessageBox.warning(self, '记录打开失败', str(exc))

    def export_path(self, path):
        if self.report is None: raise ValueError('请先完成MIMO计算。')
        from arraylab.features.mimo.io import export_mimo_directory
        export_mimo_directory(path, self.report, plot='all'); self.last_export = Path(path)

    def export_dialog(self):
        if self.report is None or self._busy: return
        parent = QFileDialog.getExistingDirectory(self, '选择MIMO结果父目录', self.host.last_directory)
        if not parent: return
        path = Path(parent)/datetime.now().strftime('mimo_%Y%m%d_%H%M%S_%f')
        self._busy = True; self.host.refresh_actions()
        try: self.export_path(path); self.host.statusBar().showMessage('MIMO结果已导出。')
        except (ValueError, OSError) as exc: QMessageBox.warning(self, '导出失败', str(exc))
        finally: self._busy = False; self.host.refresh_actions()
