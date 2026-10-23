"""Native task page for continuous subarray-center placement."""
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import threading

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QTabWidget, QScrollArea, QLabel, QLineEdit, QComboBox, QProgressBar,
    QFileDialog, QMessageBox, QTreeWidgetItem)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from .parallel import ComputationCancelled, requested_workers
from .subarray import (SubarrayConfig, optimize_subarrays,
    load_subarray_configuration, save_subarray_configuration, export_subarray)
from .subarray_plot import render_subarray_figure


class SubarrayWorker(QThread):
    progress = Signal(int, int)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, config, workers, parent=None):
        super().__init__(parent)
        self.config = config
        self.workers = workers
        self.stop = threading.Event()

    def run(self):
        try:
            result = optimize_subarrays(self.config, workers=self.workers,
                progress=self.progress.emit, cancelled=self.stop.is_set)
        except ComputationCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(result)


class SubarrayPanel(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.report = None
        self.thread = None
        self._busy = False
        self._changing = False
        self.last_export = None
        self.inputs = {}
        self.parameter_widget = QTabWidget()
        defaults = asdict(SubarrayConfig())
        groups = [
            ('子阵', [('count', '子阵数量'), ('nx', '内部阵元 X 数'), ('ny', '内部阵元 Y 数'),
                     ('spacing_x', '内部 X 间距 / λ₀'), ('spacing_y', '内部 Y 间距 / λ₀')]),
            ('孔径', [('aperture_x', '总孔径 X / λ₀'), ('aperture_y', '总孔径 Y / λ₀'),
                     ('min_spacing', '阵元最小距 / λ₀'), ('hole_shape', '中心禁布区'),
                     ('hole_x', '孔宽 X / 圆直径 λ₀'), ('hole_y', '孔高 Y / λ₀')]),
            ('扫描', [('reference_ghz', '参考频率 / GHz'), ('theta', '目标 θ / °'),
                     ('phi', '目标 φ / °'), ('frequencies_ghz', '频率列表 / GHz'),
                     ('scan_angles', '角度列表 θ,φ / °')]),
            ('优化', [('exclusion_u', '主瓣排除半宽 u'), ('exclusion_v', '主瓣排除半宽 v'),
                     ('iterations', '搜索迭代次数'), ('opt_grid', '优化网格边长'),
                     ('eval_grid', '复核网格边长'), ('restarts', '独立搜索次数'), ('seed', '随机种子')]),
        ]
        for name, fields in groups:
            widget = QWidget(); body = QVBoxLayout(widget)
            form = QFormLayout(); form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            body.addLayout(form)
            for key, label in fields:
                if key == 'hole_shape':
                    edit = QComboBox()
                    for title, value in [('无', 'none'), ('矩形', 'rectangle'), ('圆形', 'circle')]:
                        edit.addItem(title, value)
                    edit.currentIndexChanged.connect(self.mark_dirty)
                else:
                    edit = QLineEdit(str(defaults[key]))
                    edit.textChanged.connect(self.mark_dirty)
                edit.setAccessibleName('子阵 ' + label)
                self.inputs[key] = edit; form.addRow(label, edit)
            notes = {
                '子阵': '固定内部矩形模板与朝向、等幅阵元。优化连续的子阵中心位置；内部阵元随各工况理想调相。',
                '孔径': '孔径包含展开后的全部阵元中心。检查阵元间距与子阵包络不交叠；λ₀由参考频率定义，暂不设计实体尺寸。',
                '扫描': 'θ从+z量起，φ从+x朝+y。列表为空时使用上方单工况；填写列表则替代单工况。频率示例：9,10,11；角度示例：0,0;20,0;30,45。固定布局在各工况重新调相。',
                '优化': '主瓣排除范围为目标u/v周围的矩形。优化范围外最坏副瓣；网格复核为采样估计，不保证连续全局最优。',
            }
            note = QLabel(notes[name]); note.setWordWrap(True); body.addWidget(note); body.addStretch()
            scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(widget)
            self.parameter_widget.addTab(scroll, name)
        layout = QVBoxLayout(self); layout.setContentsMargins(8, 6, 8, 6)
        self.state = QLabel('设置右侧模板、孔径及扫描工况，再点击顶部“优化子阵中心”。')
        self.state.setObjectName('state'); self.state.setWordWrap(True); layout.addWidget(self.state)
        row = QHBoxLayout()
        self.view = QComboBox()
        for label, value in [('子阵与阵元布局', 'layout'), ('二维方向图', 'pattern'),
                             ('主瓣切面', 'cuts'), ('三维方向图', '3d')]:
            self.view.addItem(label, value)
        self.case_selector = QComboBox(); self.case_selector.setAccessibleName('子阵结果工况')
        row.addWidget(QLabel('结果')); row.addWidget(self.view, 1)
        row.addWidget(QLabel('扫描工况')); row.addWidget(self.case_selector, 1); layout.addLayout(row)
        self.progress = QProgressBar(); self.progress.hide(); layout.addWidget(self.progress)
        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure); layout.addWidget(self.canvas, 1)
        self.toolbar = NavigationToolbar2QT(self.canvas, self); layout.addWidget(self.toolbar)
        self.view.currentIndexChanged.connect(self.redraw)
        self.case_selector.currentIndexChanged.connect(self.redraw)
        self.redraw()

    def config(self):
        defaults = asdict(SubarrayConfig()); values = {}
        for key, control in self.inputs.items():
            if isinstance(control, QComboBox):
                values[key] = control.currentData()
            else:
                value = control.text().strip()
                try:
                    values[key] = type(defaults[key])(value)
                except ValueError as exc:
                    raise ValueError(f'{control.accessibleName()}：请输入有效数值。') from exc
        result = SubarrayConfig(**values); result.validate(); return result

    def apply_config(self, config):
        self._changing = True
        try:
            for key, value in asdict(config).items():
                control = self.inputs.get(key)
                if isinstance(control, QComboBox): control.setCurrentIndex(control.findData(value))
                elif control is not None: control.setText(str(value))
        finally:
            self._changing = False
        self.mark_dirty()

    def mark_dirty(self):
        if not self._changing and hasattr(self, 'state'):
            self.state.setText('输入已改变，尚未计算。现有图形和导出继续对应上一份结果。')

    def start(self):
        if self.thread is not None or self._busy: return
        try: config = self.config()
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, '子阵参数', str(exc)); return
        self._busy = True; self.parameter_widget.setEnabled(False)
        self.progress.setRange(0, config.restarts); self.progress.setValue(0); self.progress.show()
        self.state.setText('正在搜索可行布局并优化子阵中心位置…')
        worker = SubarrayWorker(config, requested_workers(), self)
        self.thread = worker
        worker.progress.connect(self.on_progress)
        worker.succeeded.connect(self.accept_result)
        worker.failed.connect(self.on_failure)
        worker.cancelled.connect(lambda: self.state.setText('计算已取消，原结果保留。'))
        worker.finished.connect(self.on_finished)
        worker.start(); self.window.refresh_actions()

    def cancel(self):
        if self.thread is not None:
            self.thread.stop.set(); self.state.setText('正在取消；等待当前计算结束后保留原结果。')

    def on_progress(self, done, total):
        self.progress.setRange(0, max(1, total)); self.progress.setValue(done)

    def accept_result(self, report):
        self.report = report
        self.case_selector.blockSignals(True); self.case_selector.clear()
        for case in report['best']['cases']:
            self.case_selector.addItem(f"{case['frequency_ghz']:g} GHz · {case['theta']:g}° / {case['phi']:g}°")
        self.case_selector.blockSignals(False)
        self.state.setText(f"优化完成：{report['config']['count']} 个子阵；布局及结果可导出。主瓣内外以设定排除区划分。")
        self.window.statusBar().showMessage(self.state.text())
        self.redraw()

    def on_failure(self, message):
        self.state.setText('本次未完成，原结果保留。' + message)
        QMessageBox.warning(self, '子阵优化未完成', message)

    def on_finished(self):
        worker = self.thread; self.thread = None; self._busy = False
        self.parameter_widget.setEnabled(True); self.progress.hide()
        if worker is not None: worker.deleteLater()
        self.window.refresh_actions()

    def redraw(self):
        if self.report is None:
            self.figure.clear(); ax = self.figure.add_subplot(111); ax.set_axis_off()
            ax.text(.5, .5, '设置右侧参数后运行子阵中心优化', ha='center', va='center', transform=ax.transAxes)
        else:
            render_subarray_figure(self.figure, self.report, self.view.currentData(), max(0, self.case_selector.currentIndex()), chinese=True)
        self.canvas.draw_idle()

    def save_path(self, path): save_subarray_configuration(path, self.config())

    def open_path(self, path):
        if self.thread is not None: raise ValueError('请先完成或取消当前子阵计算。')
        self.apply_config(load_subarray_configuration(path))

    def save_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, '保存子阵参数', str(Path(self.window.last_directory) / 'subarray.json'), 'JSON (*.json)')
        if path:
            try: self.save_path(path)
            except (ValueError, OSError) as exc: QMessageBox.warning(self, '保存失败', str(exc))

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, '打开子阵参数', self.window.last_directory, 'JSON (*.json)')
        if path:
            try: self.open_path(path)
            except (ValueError, OSError) as exc: QMessageBox.warning(self, '打开失败', str(exc))

    def export_path(self, path):
        if self.report is None: raise ValueError('请先完成子阵优化。')
        export_subarray(path, self.report, plot='all'); self.last_export = Path(path)

    def export_dialog(self):
        if self.report is None or self._busy: return
        parent = QFileDialog.getExistingDirectory(self, '选择子阵结果父目录', self.window.last_directory)
        if not parent: return
        path = Path(parent) / datetime.now().strftime('subarray_%Y%m%d_%H%M%S_%f')
        self._busy = True; self.window.refresh_actions()
        try:
            self.export_path(path); self.window.statusBar().showMessage('子阵结果已导出：' + str(path))
        except (ValueError, OSError) as exc: QMessageBox.warning(self, '导出失败', str(exc))
        finally:
            self._busy = False; self.window.refresh_actions()


def attach_subarray(window):
    from .icons import workbench_icon
    window.tabs.blockSignals(True)
    try:
        window.subarray_panel = SubarrayPanel(window)
        window.tabs.addTab(window.subarray_panel, workbench_icon('array'), '子阵中心优化')
        window.inspector_stack.addWidget(window.subarray_panel.parameter_widget)
        item = QTreeWidgetItem(window.project_tree_root, ['子阵中心优化'])
        item.setData(0, Qt.ItemDataRole.UserRole, ('view', 8))
        item.setIcon(0, workbench_icon('array')); window.subarray_tree_item = item
    finally:
        window.tabs.blockSignals(False)
