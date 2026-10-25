"""Native antenna-design controls composed into the radar workbench."""
from copy import deepcopy

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QTabWidget,
    QScrollArea, QLineEdit, QComboBox, QCheckBox, QPlainTextEdit, QPushButton,
    QLabel, QMessageBox)

from .settings import normalize_antenna_design, normalize_geometry


SPARSE_FIELDS = (
    ('criterion', '角分辨判据', [('常规波束形成：3 dB 谷', 'two_target'), ('HPBW代理', 'hpbw'),
        ('MUSIC', 'music'), ('Capon / MVDR', 'mvdr'), ('LS-ESPRIT', 'esprit'), ('DML', 'dml')]),
    ('targetResolutionDeg', '目标角间隔 / 宽度 (°)', None),
    ('axis', '一维设计轴', [('方位 x', 'azimuth'), ('俯仰 z', 'elevation')]),
    ('txCount', 'Tx物理通道数', None), ('rxCount', 'Rx物理通道数', None),
    ('scanMaxDeg', '扫描中心半角 ±(°)', None),
    ('maxApertureLambda', '设计轴物理孔径上限 (λ)', None),
    ('minSpacingLambda', '同类天线最小间距 (λ)', None), ('gridLambda', '坐标格点 (λ)', None),
    ('maxSidelobeDb', '旁瓣 / 栅瓣上限 (dB)', None),
    ('iterations', '搜索迭代预算', None), ('seed', '随机种子', None))
DOA_FIELDS = (
    ('snrDb', '强目标单通道 SNR (dB)', None), ('snapshots', '独立角度快拍数', None),
    ('powerDifferenceDb', '强弱目标功率差 (dB)', None), ('sourceCorrelation', '源相关系数 ρ (0–1)', None),
    ('trials', '每扫描中心验证次数', None), ('requiredSuccessRate', '最低经验通过率 (0–1)', None),
    ('angleToleranceDeg', '单目标角误差限 (°)', None), ('scanGridStepDeg', 'MUSIC / MVDR步长 (°)', None),
    ('diagonalLoading', 'MVDR相对对角加载', None),
    ('smoothing', '协方差预处理', [('无空间平滑', 'none'), ('前向空间平滑（连续ULA）', 'spatial')]),
    ('subarrayLength', '平滑子阵长度（0自动）', None))
GAIN_FIELDS = (
    ('gainMode', '增益分配', [('Tx/Rx等增益', 'equal'), ('已知Tx，反算Rx', 'solve_rx'), ('已知Rx，反算Tx', 'solve_tx')]),
    ('fixedGainDbi', '已知另一侧增益 (dBi)', None), ('rcsDbsm', '目标RCS (dBsm)', None),
    ('rangeM', '预算距离 (m)', None), ('txPowerDbm', '天线端传导功率 (dBm)', None),
    ('powerBasis', '功率口径', [('每路Tx峰值', 'per_tx'), ('同时活动Tx总功率', 'total_active')]),
    ('combining', '接收处理', [('理想Rx相干合并', 'coherent_rx'), ('单Rx / 不另计阵列收益', 'single_rx')]),
    ('radialVelocityMps', '预算目标径向速度 (m/s)', None), ('noiseFigureDb', '接收机噪声系数 (dB)', None),
    ('systemLossDb', '额外系统损耗 (dB)', None), ('requiredSnrDb', '所需处理后SNR (dB)', None),
    ('marginDb', '额外设计裕量 (dB)', None), ('temperatureK', '参考温度 (K)', None),
    ('apertureEfficiency', '等效孔径效率 (0–1)', None))


class RadarAntennaControls:
    def __init__(self, panel):
        self.panel = panel
        self.sparse = {}; self.gain = {}; self.layout = {}; self.buttons = []
        body, _ = panel._scroll_page('天线设计')
        self.pages = QTabWidget(); body.addWidget(self.pages)
        layout_body, form = self._page('布局')
        self.custom = QCheckBox('使用自定义Tx/Rx坐标'); form.addRow(self.custom)
        self.custom.toggled.connect(panel.mark_dirty)
        self.custom.toggled.connect(self.refresh_enabled)
        for key, label in [('tx', 'Tx发射坐标'), ('rx', 'Rx接收坐标')]:
            edit = QPlainTextEdit(); edit.setMinimumHeight(105); edit.setAccessibleName('雷达 '+label)
            edit.setPlaceholderText('每行：x, z\n例如：0, 0\n0.5, 0')
            edit.textChanged.connect(panel.mark_dirty); self.layout[key] = edit
            form.addRow(label, edit)
        self._note(layout_body, '每行是一根天线的位置，填写 x,z 两个数（λ）。方位沿 x，俯仰沿 z。坐标行数须与共用页的 Tx/Rx 数一致；不含幅度加权。')
        self.fixed = QCheckBox('固定物理位置（指定参考频率）'); form.addRow(self.fixed)
        self.fixed.toggled.connect(panel.mark_dirty); self.fixed.toggled.connect(self.refresh_enabled)
        for key, label in [('referenceFrequencyGHz', '参考频率 (GHz)'), ('steerAzDeg', '方位指向 (°)'), ('steerElDeg', '俯仰指向 (°)')]:
            self._field(form, self.layout, key, label)
        self._button(layout_body, '用坐标行数更新波形通道数', self.use_geometry_counts)
        self._note(layout_body, '未勾选自定义时按当前 Tx/Rx 生成默认均匀虚拟线阵。固定参考频率后，A/B切换保持物理位置，归一化坐标按波长换算。F5更新方向图。')
        layout_body.addStretch()
        sparse_body, form = self._page('反向设计')
        for key, label, options in SPARSE_FIELDS:
            self._field(form, self.sparse, key, label, options)
        self._button(sparse_body, '从共用页复制Tx/Rx数量', self.copy_counts)
        self._button(sparse_body, '开始综合', lambda: panel.start('sparse'))
        self._button(sparse_body, '验证当前阵列', lambda: panel.start('evaluate'))
        self.apply_button = self._button(sparse_body, '应用本次综合坐标到雷达布局', self.apply_result)
        self._note(sparse_body, '设置只在点击开始综合时运行。验证当前阵列使用当前布局；综合则重新搜索物理Tx/Rx坐标。未达标候选会明确保留未达标结论。')
        sparse_body.addStretch()
        self.doa_page_body, form = self._page('高分辨条件')
        for key, label, options in DOA_FIELDS:
            self._field(form, self.sparse, key, label, options)
        self.method_note = self._note(self.doa_page_body, '')
        self._note(self.doa_page_body, '目标数固定为已知2个，假设Tx已分离和校相。快拍数独立于波形chirp数；ρ为实数源相关系数。经验通过率不是Pd或置信下限。最多48个独立位置，超出会明确拒绝。')
        self.doa_page_body.addStretch()
        gain_body, form = self._page('增益预算')
        self.scope = QComboBox(); self.scope.addItem('当前分析波形', 'active'); self.scope.addItem('分别满足A/B，取较严要求', 'all')
        self.scope.currentIndexChanged.connect(panel.mark_dirty); form.addRow('参考波形', self.scope)
        for key, label, options in GAIN_FIELDS:
            self._field(form, self.gain, key, label, options)
        self._button(gain_body, '从频谱目标复制距离和速度', self.copy_target)
        self._button(gain_body, '计算波形与增益预算', lambda: panel.start())
        self._note(gain_body, '功率是天线端传导功率，不是EIRP。增益是每个RF通道所接天线/子阵在目标方向的realized gain；若已含整Rx阵列收益，请选“不另计阵列收益”。')
        self._note(gain_body, '仅累计实际ADC时长和chirp数，窗损耗分别计入；不额外添加Tx数量收益。等效口径为预算估算，不是天线或馈线尺寸设计。F5计算后查看“天线增益预算”。')
        gain_body.addStretch()
        self.sparse['criterion'].currentIndexChanged.connect(self.method_changed)
        self.sparse['smoothing'].currentIndexChanged.connect(self.refresh_enabled)
        self.pages.currentChanged.connect(self._section_changed)
        self._section = 'layout'

    def _page(self, title):
        widget = QWidget(); body = QVBoxLayout(widget)
        form = QFormLayout(); form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        body.addLayout(form)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(widget)
        self.pages.addTab(scroll, title)
        return body, form

    def _field(self, form, group, key, label, options=None):
        if options:
            item = self.panel._choice(form, group, key, label, options)
        else:
            item = self.panel._edit(form, group, key, label)
        item.setMinimumWidth(50)
        return item

    def _note(self, body, text):
        note = QLabel(text); note.setWordWrap(True); body.addWidget(note); return note

    def _button(self, body, text, callback):
        button = QPushButton(text); button.clicked.connect(callback); body.addWidget(button)
        self.buttons.append(button); return button

    def _section_changed(self, index):
        self._section = ('layout', 'sparse', 'sparse', 'gain')[index]

    def all_controls(self):
        return [self.custom, self.fixed, self.scope, *self.layout.values(), *self.sparse.values(), *self.gain.values()]

    def _values(self, group):
        return {key: control.currentData() if isinstance(control, QComboBox) else self.panel._number(control, key)
                for key, control in group.items()}

    def configuration(self):
        design = normalize_antenna_design({'version': 1, 'section': self._section,
            'sparseOptions': self._values(self.sparse), 'gainOptions': self._values(self.gain), 'gainScope': self.scope.currentData()})
        geometry = None
        if self.custom.isChecked():
            geometry = {key: self.read_coordinates(self.layout[key].toPlainText()) for key in ('tx', 'rx')}
            for key in ('steerAzDeg', 'steerElDeg'):
                geometry[key] = self.panel._number(self.layout[key], key)
            if self.fixed.isChecked():
                geometry['referenceFrequencyGHz'] = self.panel._number(self.layout['referenceFrequencyGHz'], 'referenceFrequencyGHz')
            geometry = normalize_geometry(geometry)
        return {'geometry': geometry, 'antennaDesign': design}

    @staticmethod
    def read_coordinates(text):
        result = []
        for index, line in enumerate(text.splitlines(), 1):
            if not line.strip(): continue
            pieces = line.replace('，', ',').replace(',', ' ').split()
            if len(pieces) != 2: raise ValueError(f'坐标第{index}行需要x,z两个数。')
            try: result.append([float(p) for p in pieces])
            except ValueError as exc: raise ValueError(f'坐标第{index}行含无效数字。') from exc
        return result

    def apply_configuration(self, configuration):
        from .antenna import geometry_preset
        design = normalize_antenna_design(configuration.get('antennaDesign'))
        geometry = configuration.get('geometry')
        self.custom.setChecked(geometry is not None)
        shown = geometry or {**geometry_preset('ula', configuration['params']['txCount'], configuration['params']['rxCount']),
                             'steerAzDeg': 0, 'steerElDeg': 0}
        for key in ('tx', 'rx'):
            self.layout[key].setPlainText('\n'.join(f'{p[0]:.17g}, {p[1]:.17g}' for p in shown[key]))
        self.fixed.setChecked('referenceFrequencyGHz' in shown)
        for key in ('steerAzDeg', 'steerElDeg', 'referenceFrequencyGHz'):
            self.panel._set(self.layout[key], shown.get(key, configuration['params']['startGHz'] if key == 'referenceFrequencyGHz' else 0))
        for key, group in (('sparseOptions', self.sparse), ('gainOptions', self.gain)):
            for field, control in group.items(): self.panel._set(control, design[key][field])
        self.panel._set(self.scope, design['gainScope'])
        self.pages.setCurrentIndex({'layout': 0, 'sparse': 1, 'gain': 3}[design['section']])
        self._section = design['section']; self.refresh_enabled()

    def method_changed(self):
        if not self.panel._changing and self.sparse['criterion'].currentData() == 'dml':
            self.panel._set(self.sparse['smoothing'], 'none'); self.panel._set(self.sparse['subarrayLength'], 0)
        self.refresh_enabled()

    def refresh_enabled(self, *_):
        for key, item in self.layout.items():
            item.setEnabled(self.custom.isChecked() and (key != 'referenceFrequencyGHz' or self.fixed.isChecked()))
        self.fixed.setEnabled(self.custom.isChecked())
        if 'smoothing' not in self.sparse: return
        method = self.sparse['criterion'].currentData(); high = method in ('music', 'mvdr', 'esprit', 'dml')
        for key, _, _ in DOA_FIELDS: self.sparse[key].setEnabled(high)
        self.sparse['diagonalLoading'].setEnabled(method == 'mvdr')
        self.sparse['scanGridStepDeg'].setEnabled(method in ('music', 'mvdr'))
        self.sparse['smoothing'].setEnabled(high and method != 'dml')
        self.sparse['subarrayLength'].setEnabled(high and method != 'dml' and self.sparse['smoothing'].currentData() == 'spatial')
        notes = {'music': 'MUSIC：从样本协方差分离噪声子空间；相干源可能导致秩亏。',
                 'mvdr': 'MVDR：使用加载后的协方差求逆，自适应谱不是天线物理方向图。',
                 'esprit': 'LS-ESPRIT：只用可提取的连续ULA子阵，间距≤0.5λ；不适用时不回退其他算法。',
                 'dml': 'DML：双角联合投影残差搜索，支持相干模型；不生成伪造的一维谱，不保证全局精确最优。'}
        self.method_note.setText(notes.get(method, '当前使用常规波束形成 / HPBW，无需高分辨统计条件。'))

    def copy_counts(self):
        for key in ('txCount', 'rxCount'): self.sparse[key].setText(self.panel.inputs[key].text())

    def copy_target(self):
        self.gain['rangeM'].setText(self.panel.inputs['targetRange'].text())
        self.gain['radialVelocityMps'].setText(self.panel.inputs['targetVelocity'].text())

    def use_geometry_counts(self):
        try:
            parsed={key:self.read_coordinates(self.layout[key].toPlainText()) for key in ('tx','rx')}
            normalize_geometry(parsed)
            for key, count_key in (('tx', 'txCount'), ('rx', 'rxCount')):
                self.panel.inputs[count_key].setText(str(len(parsed[key])))
        except ValueError as exc: QMessageBox.warning(self.panel, '坐标未应用', str(exc))

    def apply_result(self):
        if self.panel._busy: return
        try:
            report = self.panel.report
            result = report.get('sparse_design') if report else None
            if not result or not result.get('valid'): raise ValueError('请先完成一次有效的综合或阵列验证。')
            current = self.panel.config()
            if current['antennaDesign']['sparseOptions'] != report['configuration']['antennaDesign']['sparseOptions']:
                raise ValueError('综合条件已改变，请先重新运行。')
            from .antenna import applied_geometry
            current['geometry'] = applied_geometry(result, result['referenceFrequencyGHz'])
            for key in ('txCount', 'rxCount'): current['params'][key] = result['options'][key]
            current['antennaDesign']['section'] = 'layout'
            self.panel.apply_configuration(current)
            self.panel.view.setCurrentIndex(self.panel.view.findData('antenna'))
            self.panel.state.setText(('已应用未达标候选' if result.get('status') != 'met' else '已应用综合坐标') + '，并固定物理参考频率；按F5更新天线方向图和增益预算。')
        except ValueError as exc: QMessageBox.warning(self.panel, '坐标未应用', str(exc))
