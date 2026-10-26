"""Native waveform, A/B and distance-ambiguity task; no browser dependency."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import math
import threading

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from arraylab.shared.qt.resize_canvas import ResizeAwareFigureCanvas as FigureCanvasQTAgg
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSignalBlocker
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QTabWidget, QScrollArea, QLabel, QLineEdit, QComboBox, QCheckBox, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QTextBrowser, QFileDialog,
    QMessageBox, QInputDialog, QPlainTextEdit)

from arraylab.framework.parallel import ComputationCancelled, requested_workers
from arraylab.features.waveform.model import default_waveform_configuration, normalize_waveform_configuration, calculate_waveform
from arraylab.features.waveform.plotting import draw_waveform_figure, VIEWS, value
from arraylab.shared.plotting.layout import reflow_plot_layout
from arraylab.shared.plotting.metrics import draw_metric_footer


SHARED = ('mode','txCount','rxCount','frameMs','ddmSlots','targetRange','targetVelocity','window')
PROFILE_FIELDS = (
    ('startGHz','起始频率 / GHz'), ('slopeMHzUs','调频斜率 / MHz/μs'),
    ('rampUs','扫频时间 / μs'), ('idleUs','空闲时间 / μs'),
    ('adcStartUs','ADC延迟 / μs'), ('samples','ADC采样点数'),
    ('sampleRateMsps','采样率 / MSps'), ('loops','每Tx慢时间样本数'),
    ('ifBandwidthMHz','模拟IF带宽 / MHz'), ('rangeFft','距离FFT长度'),
    ('dopplerFft','速度FFT长度'))
INTEGER_FIELDS = {'samples','loops','txCount','rxCount','rangeFft','dopplerFft','ddmSlots'}
METRICS = (
    ('observedBandwidthMHz','有效ADC采样带宽','MHz'),('bandwidthMHz','整段扫频带宽','MHz'),
    ('rangeResolutionM','名义距离分辨率','m'),('rangeBinM','距离FFT栅格','m'),
    ('maxRangeM','规划IF距离上限（含速度余量）','m'),('maxRangeStaticM','静止目标IF距离上限','m'),
    ('maxVelocityMps','保守无模糊速度幅值','m/s'),('rawMaxVelocityMps','原始慢时间速度幅值','m/s'),
    ('velocityResolutionMps','名义速度分辨率','m/s'),('velocityBinMps','速度FFT栅格','m/s'),
    ('txIntervalUs','同组同Tx重访间隔','μs'),('coherentMs','组内相干采样时长','ms'),
    ('chirpUs','物理Chirp周期','μs'),('adcUs','ADC采样时长','μs'),
    ('centerGHz','ADC窗中点频率','GHz'),('lambdaMm','ADC中心频率波长','mm'),
    ('ifLimitMHz','规划可用IF上限','MHz'),('targetBeatMHz','目标解析拍频','MHz'),
    ('targetDopplerHz','目标多普勒','Hz'),('rangeBiasM','距离-速度耦合偏差','m'),
    ('perTxDutyPercent','单Tx扫频占空比','%'),('totalTxDutyPercent','各Tx扫频占空比之和','%'))
STATUS = {'off':'未开启','blocked':'不可解算','resolved':'范围内唯一候选',
          'ambiguous':'仍有多个候选','inconsistent':'无一致候选'}


class WaveformWorker(QThread):
    progress=Signal(str)

    def __init__(self,configuration,parent=None,radar_action='waveform'):
        super().__init__(parent)
        self.configuration=deepcopy(configuration);self.workers=requested_workers()
        self.stop=threading.Event();self.result=None;self.error=None;self.cancelled=False
        self.radar_action=radar_action

    def tell(self,*args):
        if self.stop.is_set() or self.isInterruptionRequested():raise ComputationCancelled()
        self.progress.emit(' / '.join(map(str,args)))

    def run(self):
        try:
            self.result=calculate_waveform(self.configuration,workers=self.workers,
                cancelled=lambda:self.stop.is_set() or self.isInterruptionRequested(),progress=self.tell,
                radar_action=self.radar_action)
        except ComputationCancelled:self.cancelled=True
        except Exception as exc:self.error=f'{type(exc).__name__}: {exc}'


class WaveformToolbar(NavigationToolbar2QT):
    def __init__(self,canvas,panel):
        self.panel=panel;super().__init__(canvas,panel)

    def _update_view(self):
        super()._update_view();self.panel.reflow()


class WaveformPanel(QWidget):
    """A submitted configuration is an immutable snapshot until next success."""
    def __init__(self,window):
        super().__init__()
        self.host=window;self.report=None;self.thread=None;self._busy=False
        self._changing=True;self._cancel_pending=False;self._last_error=None
        self.comparisons=[];self.last_export=None;self.inputs={};self.profile_inputs={}
        self.derived_inputs={};self.range_inputs={};self.pri_inputs={};self._derived_updating=False
        self.parameter_widget=QTabWidget();self._build_parameters()
        body=QVBoxLayout(self);body.setContentsMargins(8,6,8,6)
        self.state=QLabel();self.state.setObjectName('state');self.state.setWordWrap(True);body.addWidget(self.state)
        self.summary=QLabel('尚无波形计算结果。');self.summary.setWordWrap(True);body.addWidget(self.summary)
        controls=QHBoxLayout();controls.addWidget(QLabel('查看已计算组'))
        self.result_profile=QComboBox();self.result_profile.addItem('A','A');self.result_profile.addItem('B','B')
        self.result_profile.setAccessibleName('查看已计算波形组');controls.addWidget(self.result_profile)
        self.view=QComboBox()
        for title,key in [('波形与A/B时序','overview'),('波形指标与参数检查','metrics'),
                          ('DDM相位编码','phase'),('距离与速度频谱','spectra'),('距离-速度图','rd'),
                          ('ADC折叠解模糊','range_recovery'),('跨周期归属解模糊','pri_recovery'),
                          ('雷达天线布局与方向图','antenna'),('稀疏阵列设计 / 高分辨验证','sparse_design'),
                          ('天线增益预算','gain_budget'),
                          ('方案对比','comparisons'),('公式与适用范围','theory')]:self.view.addItem(title,key)
        self.view.setAccessibleName('波形结果视图');controls.addWidget(self.view,1);body.addLayout(controls)
        self.recovery_table_toggle=QCheckBox('候选表');self.recovery_table_toggle.setToolTip('显示全部候选区间；图中只展示前80项。')
        self.recovery_table_toggle.toggled.connect(self._candidate_display_changed);controls.addWidget(self.recovery_table_toggle)
        self.figure=Figure(figsize=(8,5),dpi=100);self.canvas=FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(480,360);body.addWidget(self.canvas,1)
        self.toolbar=WaveformToolbar(self.canvas,self);body.addWidget(self.toolbar);self.plot_layout=body
        self.table=QTableWidget();self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers);body.addWidget(self.table,1)
        self.check_table=QTableWidget();self.check_table.setAlternatingRowColors(True)
        self.check_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers);self.check_table.setMaximumHeight(190)
        body.addWidget(self.check_table)
        self.table_note=QLabel();self.table_note.setWordWrap(True);body.addWidget(self.table_note)
        self.theory=QTextBrowser();self.theory.setOpenExternalLinks(True);body.addWidget(self.theory,1)
        self.comparison_bar=QWidget();compare_row=QHBoxLayout(self.comparison_bar);compare_row.setContentsMargins(0,0,0,0)
        self.add_comparison_button=QPushButton('将当前结果加入对比');self.add_comparison_button.clicked.connect(self.add_comparison_dialog)
        self.remove_comparison_button=QPushButton('移除所选方案（含A/B）');self.remove_comparison_button.clicked.connect(self.remove_selected_comparison)
        compare_row.addWidget(self.add_comparison_button);compare_row.addWidget(self.remove_comparison_button);compare_row.addStretch()
        body.addWidget(self.comparison_bar)
        note=QLabel('WaveLab 1.4 · 理想波形、Tx/Rx反向设计与增益预算；高分辨判据是已知双目标有限快拍实验，不是硬件分辨率保证。')
        note.setWordWrap(True);note.setObjectName('scope');body.addWidget(note)
        self.resize_timer=QTimer(self);self.resize_timer.setSingleShot(True);self.resize_timer.timeout.connect(self.reflow)
        self.canvas.resize_finished.connect(self.reflow)
        self.view.currentIndexChanged.connect(self.redraw);self.result_profile.currentIndexChanged.connect(self.redraw)
        self._changing=False;self.apply_configuration(default_waveform_configuration())
        self.state.setText('右侧分别设置波形、A/B编排和解模糊实验，按F5计算。');self.redraw()

    def _scroll_page(self,label):
        page=QWidget();body=QVBoxLayout(page);body.setContentsMargins(8,8,8,8)
        form=QFormLayout();form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows);body.addLayout(form)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(page)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.parameter_widget.addTab(scroll,label)
        return body,form

    def _edit(self,form,group,key,label):
        edit=QLineEdit();edit.setMinimumWidth(70);edit.setAccessibleName('波形 '+label)
        group[key]=edit;form.addRow(label,edit);edit.textChanged.connect(self.mark_dirty)
        return edit

    def _choice(self,form,group,key,label,options):
        combo=QComboBox()
        for text,data in options:combo.addItem(text,data)
        combo.setAccessibleName('波形 '+label);group[key]=combo;form.addRow(label,combo)
        combo.currentIndexChanged.connect(self.mark_dirty)
        return combo

    def _note(self,body,text):
        note=QLabel(text);note.setWordWrap(True);body.addWidget(note)
        return note

    def _build_parameters(self):
        body,form=self._scroll_page('共用/A-B')
        self._choice(form,self.inputs,'mode','发射模式',[('TDM轮流发射','tdm'),('DDM同时相位编码','ddm')])
        for key,label in [('txCount','Tx通道数'),('rxCount','Rx通道数'),('ddmSlots','DDM总槽数（含空槽）'),
                          ('frameMs','帧周期 / ms'),('targetRange','频谱目标距离 / m'),('targetVelocity','频谱目标速度 / m/s')]:
            self._edit(form,self.inputs,key,label)
        self.velocity_label=self._note(body,'')
        self._choice(form,self.inputs,'window','窗函数',[('Hann','hann'),('矩形','rect')])
        self.inputs['abEnabled']=QCheckBox('启用A/B独立双波形');form.addRow(self.inputs['abEnabled'])
        self.inputs['abEnabled'].toggled.connect(self._options_changed)
        self._choice(form,self.inputs,'abLayout','A/B编排',[('先A后B分段','blocks'),('整轮A/B交替','alternating')])
        self._edit(form,self.inputs,'switchGapUs','每次切换间隔 / μs')
        self._choice(form,self.inputs,'activeProfile','计算后首先查看',[('波形A','A'),('波形B','B')])
        self._note(body,'A/B共享本页参数，各自的斜率、带宽、周期、ADC与FFT在独立页设置。整轮交替须有相同轮数；未启用A/B时仅计算A，保留B输入。')
        self._note(body,'DDM总槽数须不少于Tx数。空槽只表示未分配编码，当前模型没有DDM多目标盲解码。')
        body.addStretch();self.inputs['targetVelocity'].textChanged.connect(self._velocity_changed)
        self.inputs['mode'].currentIndexChanged.connect(self._options_changed)
        for profile in ('A','B'):
            body,form=self._scroll_page('波形'+profile);group={};self.profile_inputs[profile]=group
            for key,label in PROFILE_FIELDS:
                edit=self._edit(form,group,key,label)
                if key in ('slopeMHzUs','rampUs','idleUs'):
                    edit.textChanged.connect(lambda _=None,name=profile:self.refresh_derived(name))
            derived={};self.derived_inputs[profile]=derived
            self._edit(form,derived,'bandwidthMHz','整段带宽 / MHz')
            self._edit(form,derived,'chirpUs','物理Chirp周期 / μs')
            for key in derived:
                derived[key].editingFinished.connect(lambda name=profile,k=key:self.apply_derived(name,k))
            self._note(body,'可直接编辑带宽或周期：带宽按当前扫频时间反算斜率；周期保持扫频时间、调整空闲时间。只修改本组参数。')
            self._note(body,'每Tx慢时间样本数：TDM按每路Tx计；DDM按同时发射chirp数计。FFT长度不小于样本数且为2的幂，补零不提高物理分辨率。')
            body.addStretch()
        self.inputs['startGHz']=self.profile_inputs['A']['startGHz']
        body,form=self._scroll_page('ADC恢复')
        self._choice(form,self.range_inputs,'method','实验算法',[('关闭','off'),('双采样率候选一致性','dual_rate'),('双斜率候选一致性','dual_slope')])
        for key,label in [('secondSampleRateMsps','第二采样率 / MSps'),('secondSlopeMHzUs','第二斜率 / MHz/μs'),
                          ('searchMaxM','搜索上限 / m'),('knownVelocityMps','外部速度先验 / m/s'),('toleranceBins','容差 / FFT栅格')]:
            self._edit(form,self.range_inputs,key,label)
        self.range_inputs['method'].currentIndexChanged.connect(self._options_changed)
        demo=QPushButton('载入ADC演示（替换波形参数）');demo.clicked.connect(self.load_adc_demo);body.addWidget(demo)
        self._note(body,'对A/B各组单独生成两次实数ADC观测，再求候选交集。需要模拟IF保留欠采样回波；第二次采集时间未自动计入主帧。')
        self._note(body,'外部速度先验独立于频谱目标真速度，不会自动复制真值。多候选保留为多解，不自动选最近真值。');body.addStretch()
        body,form=self._scroll_page('跨周期')
        self._choice(form,self.pri_inputs,'method','实验算法',[('关闭','off'),('双重复周期候选交集','dual_period')])
        for key,label in [('period1Us','第一次重复周期 / μs'),('period2Us','第二次重复周期 / μs'),
                          ('targetRangeM','独立时延实验目标 / m'),('searchMaxM','搜索上限 / m'),('timingResolutionNs','时延量化步长 / ns')]:
            self._edit(form,self.pri_inputs,key,label)
        demo=QPushButton('载入时延实验设置');demo.clicked.connect(self.load_pri_demo);body.addWidget(demo)
        self._note(body,'独立的周期内时延观测实验，假设同一个静止目标、Tx已正确分离、时延已测得。不是主波形的原始回波链，也不表示这些距离可被真实雷达探测。')
        self._note(body,'两种重复周期分别用于均匀观测；未模拟单序列交错PRI。搜索范围与容差决定候选唯一性。');body.addStretch()
        from .antenna_controls import RadarAntennaControls
        self.radar_controls=RadarAntennaControls(self)

    @staticmethod
    def _number(control,key):
        try:number=float(control.text().strip())
        except ValueError as exc:raise ValueError(control.accessibleName()+'须填写数字。') from exc
        if not math.isfinite(number):raise ValueError(control.accessibleName()+'须为有限数值。')
        if key in INTEGER_FIELDS:
            if not number.is_integer():raise ValueError(control.accessibleName()+'须为整数。')
            return int(number)
        return number

    def config(self):
        common={key:self.inputs[key].currentData() if isinstance(self.inputs[key],QComboBox) else self._number(self.inputs[key],key) for key in SHARED}
        profiles={name:{**common,**{key:self._number(group[key],key) for key,_ in PROFILE_FIELDS}} for name,group in self.profile_inputs.items()}
        def values(group):
            return {key:control.currentData() if isinstance(control,QComboBox) else self._number(control,key) for key,control in group.items()}
        return normalize_waveform_configuration({'params':profiles['A'],'profileB':profiles['B'],
            'abOptions':{'enabled':self.inputs['abEnabled'].isChecked(),'layout':self.inputs['abLayout'].currentData(),
                         'switchGapUs':self._number(self.inputs['switchGapUs'],'switchGapUs')},
            'activeProfile':self.inputs['activeProfile'].currentData(),
            'rangeOptions':values(self.range_inputs),'priOptions':values(self.pri_inputs),
            **self.radar_controls.configuration()})

    read_parameters=config

    def apply_configuration(self,configuration):
        configuration=normalize_waveform_configuration(configuration)
        self._changing=True
        try:
            for key in SHARED:self._set(self.inputs[key],configuration['params'][key])
            for profile,key in (('A','params'),('B','profileB')):
                for field,_ in PROFILE_FIELDS:self._set(self.profile_inputs[profile][field],configuration[key][field])
            self.inputs['abEnabled'].setChecked(configuration['abOptions']['enabled'])
            self._set(self.inputs['abLayout'],configuration['abOptions']['layout'])
            self._set(self.inputs['switchGapUs'],configuration['abOptions']['switchGapUs'])
            self._set(self.inputs['activeProfile'],configuration['activeProfile'])
            for name,group in [('rangeOptions',self.range_inputs),('priOptions',self.pri_inputs)]:
                for key,control in group.items():self._set(control,configuration[name][key])
            self.radar_controls.apply_configuration(configuration)
            for profile in ('A','B'):self.refresh_derived(profile)
            self._options_changed();self._velocity_changed()
        finally:self._changing=False
        self.mark_dirty()

    @staticmethod
    def _set(control,value):
        if isinstance(control,QComboBox):
            index=control.findData(value)
            if index<0:raise ValueError('无法显示的参数选项：'+str(value))
            control.setCurrentIndex(index)
        else:control.setText(str(value))

    def _options_changed(self,*_):
        if 'abEnabled' not in self.inputs:return
        enabled=self.inputs['abEnabled'].isChecked()
        self.inputs['abLayout'].setEnabled(enabled);self.inputs['switchGapUs'].setEnabled(enabled)
        self.inputs['activeProfile'].model().item(1).setEnabled(enabled)
        if not enabled:self.inputs['activeProfile'].setCurrentIndex(0)
        self.inputs['ddmSlots'].setEnabled(self.inputs['mode'].currentData()=='ddm')
        if self.range_inputs:
            method=self.range_inputs['method'].currentData()
            for key in ('secondSampleRateMsps','secondSlopeMHzUs'):
                if key in self.range_inputs:self.range_inputs[key].setEnabled(method==('dual_rate' if key=='secondSampleRateMsps' else 'dual_slope'))
        self.mark_dirty()

    def _velocity_changed(self,*_):
        try:text=f"目标速度：{self._number(self.inputs['targetVelocity'],'targetVelocity')*3.6:.6g} km/h"
        except ValueError:text='目标速度：待填写有效数值'
        self.velocity_label.setText(text)

    def refresh_derived(self,profile):
        if self._derived_updating or profile not in self.derived_inputs:return
        group=self.profile_inputs[profile]
        self._derived_updating=True
        try:
            try:
                ramp=self._number(group['rampUs'],'rampUs');slope=self._number(group['slopeMHzUs'],'slopeMHzUs');idle=self._number(group['idleUs'],'idleUs')
                calculated={'bandwidthMHz':ramp*slope,'chirpUs':ramp+idle}
            except ValueError:calculated={'bandwidthMHz':None,'chirpUs':None}
            for key,result in calculated.items():
                with QSignalBlocker(self.derived_inputs[profile][key]):
                    self.derived_inputs[profile][key].setText('' if result is None else f'{result:.12g}')
        finally:self._derived_updating=False

    def apply_derived(self,profile,key):
        if self._changing or self._derived_updating:return
        group=self.profile_inputs[profile]
        try:
            ramp=self._number(group['rampUs'],'rampUs');requested=self._number(self.derived_inputs[profile][key],key)
            if ramp<=0 or requested<=0:raise ValueError('扫频时间和组合参数须大于0。')
            if key=='chirpUs' and requested<ramp:raise ValueError('Chirp周期不能小于扫频时间。')
            self._derived_updating=True
            target='slopeMHzUs' if key=='bandwidthMHz' else 'idleUs'
            group[target].setText(f'{requested/ramp if key=="bandwidthMHz" else requested-ramp:.12g}')
        except ValueError as exc:QMessageBox.warning(self,'组合参数未更新',str(exc))
        finally:self._derived_updating=False;self.refresh_derived(profile)
        self.mark_dirty()

    def mark_dirty(self,*_):
        if not self._changing and hasattr(self,'state'):
            self.state.setText('输入已改变，按F5计算；图形、记录和导出仍对应上一次成功结果。')

    def load_adc_demo(self):
        if self._busy:return
        c=default_waveform_configuration()
        c['params'].update(samples=64,sampleRateMsps=4,targetVelocity=0,targetRange=60)
        c['profileB'].update(targetVelocity=0,targetRange=60)
        c['abOptions']['enabled']=False;c['activeProfile']='A'
        c['rangeOptions'].update(method='dual_rate',secondSampleRateMsps=5,secondSlopeMHzUs=9.6,searchMaxM=120,knownVelocityMps=0,toleranceBins=.75)
        self.apply_configuration(c);self.state.setText('已载入ADC折叠演示的波形与实验参数；按F5计算，旧结果保留。')

    def load_pri_demo(self):
        if self._busy:return
        c=self.config();c['priOptions'].update(method='dual_period',period1Us=38,period2Us=43,searchMaxM=20000,timingResolutionNs=2,targetRangeM=8000)
        self.apply_configuration(c);self.state.setText('已载入独立时延实验设置；主波形参数保持，按F5计算。')

    def start(self,radar_action='waveform'):
        if self._busy or self.thread is not None:return
        if type(radar_action) is bool:radar_action='waveform'
        try:configuration=self.config()
        except (ValueError,TypeError,ArithmeticError) as exc:
            QMessageBox.warning(self,'波形参数检查',str(exc));return
        self._busy=True;self._cancel_pending=False;self._last_error=None
        self.parameter_widget.setEnabled(False);self.state.setText('正在计算雷达波形与天线设计…')
        if radar_action in ('sparse','evaluate'):
            self.view.setCurrentIndex(self.view.findData('sparse_design'))
        worker=WaveformWorker(configuration,self,radar_action);self.thread=worker
        worker.progress.connect(self.state.setText);worker.finished.connect(self.complete)
        worker.start();self._refresh_host()

    def cancel(self):
        if self.thread is not None:
            self._cancel_pending=True;self.thread.stop.set();self.thread.requestInterruption()
            self.state.setText('正在停止计算；完成回收后保留原波形结果。')

    def complete(self):
        worker=self.thread
        if worker is None:return
        try:
            if worker.cancelled or self._cancel_pending:self.state.setText('计算已取消，原波形结果保留。')
            elif worker.error:self._last_error=worker.error;self.state.setText('计算未完成，原结果保留：'+worker.error)
            else:
                previous=(self.report or {}).get('sparse_design')
                if (worker.radar_action=='waveform' and previous and
                        self.report['configuration']['antennaDesign']['sparseOptions']==worker.result['configuration']['antennaDesign']['sparseOptions']):
                    worker.result['sparse_design']={**deepcopy(previous),'retainedFromPreviousCalculation':True}
                self.accept_result(worker.result)
                if (worker.result.get('sparse_design') or {}).get('retainedFromPreviousCalculation'):
                    self.state.setText('波形、天线和增益已更新；保留此前的独立综合快照，综合图注明原参考频率，未重新搜索。')
        except Exception as exc:self._last_error=f'{type(exc).__name__}: {exc}';self.state.setText('结果显示未完成，原结果保留：'+self._last_error)
        finally:
            self.thread=None;worker.deleteLater();self._busy=False;self.parameter_widget.setEnabled(True);self._refresh_host()
            if getattr(self.host,'_waveform_close_pending',False):
                self.host._waveform_close_pending=False;QTimer.singleShot(0,self.host.close)

    def _refresh_host(self):
        if getattr(self.host,'_workflow_ready',False) and getattr(self.host,'waveform_panel',None) is self:self.host.refresh_actions()

    def _input_state(self):
        controls=list(self.inputs.values())+[c for group in self.profile_inputs.values() for c in group.values()]+list(self.range_inputs.values())+list(self.pri_inputs.values())
        controls+=self.radar_controls.all_controls()
        return [(c,c.isChecked() if isinstance(c,QCheckBox) else c.currentIndex() if isinstance(c,QComboBox) else c.toPlainText() if isinstance(c,QPlainTextEdit) else c.text()) for c in controls]

    def _restore_inputs(self,state):
        self._changing=True
        try:
            for control,value_ in state:
                with QSignalBlocker(control):
                    if isinstance(control,QCheckBox):control.setChecked(value_)
                    elif isinstance(control,QComboBox):control.setCurrentIndex(value_)
                    elif isinstance(control,QPlainTextEdit):control.setPlainText(value_)
                    else:control.setText(value_)
            self._options_changed();self._velocity_changed()
            for profile in ('A','B'):self.refresh_derived(profile)
            self.radar_controls.refresh_enabled()
        finally:self._changing=False

    def _make_figure(self,report,view,profile):
        figure=Figure(figsize=(8,5),dpi=100)
        if report is None:
            axis=figure.add_subplot(111);axis.set_axis_off();axis.text(.5,.5,'设置波形参数后按F5计算',ha='center',va='center',transform=axis.transAxes)
        else:draw_waveform_figure(figure,report,view if view in VIEWS else 'overview',profile,chinese=True)
        return figure

    def _install_figure(self,figure):
        if self.canvas.widgetlock.isowner(self.toolbar):self.canvas.widgetlock.release(self.toolbar)
        self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
        original=self.canvas.figure._original_dpi;figure.set_dpi(self.canvas.figure.dpi);figure._original_dpi=original
        self.figure=figure;self.canvas.figure=figure;figure.set_canvas(self.canvas)
        old=self.toolbar;self.toolbar=WaveformToolbar(self.canvas,self);self.plot_layout.replaceWidget(old,self.toolbar);old.hide();old.deleteLater()
        self.reflow()

    def accept_result(self,report,*,restore_inputs=False,view=None,profile=None,comparisons=None):
        chosen=view or self.view.currentData();profile=profile or report['configuration']['activeProfile']
        if self.view.findData(chosen)<0:raise ValueError('未知波形结果视图。')
        if profile not in ('A','B') or not report['analysis']['profiles'].get(profile):raise ValueError('该记录没有所选A/B组结果。')
        figure=self._make_figure(report,chosen,profile)
        old=(self.report,self.figure,self.view.currentIndex(),self.result_profile.currentIndex(),self._input_state(),self.comparisons,
             self.radar_controls.pages.currentIndex(),self.radar_controls._section)
        try:
            if restore_inputs:self.apply_configuration(report['configuration'])
            self.report=report
            if comparisons is not None:self.comparisons=deepcopy(comparisons)
            with QSignalBlocker(self.view),QSignalBlocker(self.result_profile):
                self.view.setCurrentIndex(self.view.findData(chosen));self.result_profile.setCurrentIndex(self.result_profile.findData(profile))
            self.result_profile.model().item(1).setEnabled(bool(report['analysis']['profiles'].get('B')))
            self._install_figure(figure);self._show_view();self.update_summary()
        except Exception:
            self.report=old[0];self.comparisons=old[5];self._restore_inputs(old[4])
            with QSignalBlocker(self.radar_controls.pages):self.radar_controls.pages.setCurrentIndex(old[6])
            self.radar_controls._section=old[7]
            with QSignalBlocker(self.view),QSignalBlocker(self.result_profile):
                self.view.setCurrentIndex(old[2]);self.result_profile.setCurrentIndex(old[3])
            self._install_figure(old[1]);self._show_view();self.update_summary();raise
        self._last_error=None
        self.state.setText('已恢复完整波形记录；使用已保存结果，未重新计算。' if restore_inputs else '波形计算完成；结果对应本次成功输入。')
        self.host.statusBar().showMessage(self.state.text());self._refresh_host()

    @staticmethod
    def _fill(table,columns,rows):
        table.setColumnCount(len(columns));table.setHorizontalHeaderLabels(columns);table.setRowCount(len(rows))
        for i,row in enumerate(rows):
            for j,cell in enumerate(row):table.setItem(i,j,QTableWidgetItem(str(cell)))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.resizeRowsToContents()

    def update_summary(self):
        if self.report is None:self.summary.setText('尚无波形计算结果。');return
        a=self.report['analysis'];c=self.report['configuration'];m=a['metrics'];mode=c['params']['mode'].upper()
        tag=('A/B整轮交替' if a['options']['layout']=='alternating' else 'A/B分段') if a['options']['enabled'] else '单组A'
        self.summary.setText(f"{mode} · {c['params']['txCount']}Tx / {c['params']['rxCount']}Rx · {tag} · 总采集 {value(m['burstMs'])} / {value(m['frameMs'])} ms · 数据率 {value(m['dataRateMBps'])} MB/s")

    def _metric_rows(self):
        profiles=self.report['analysis']['profiles'];rows=[]
        for key,label,unit in METRICS:
            row=[label]
            for profile in ('A','B'):
                record=profiles.get(profile);metric=record['result']['metrics'].get(key) if record else None
                row.append(value(metric))
            rows.append(row+[unit])
            if unit=='m/s':rows.append([label+'（km/h）']+[value(profiles[p]['result']['metrics'].get(key)*3.6) if profiles.get(p) else '—' for p in ('A','B')]+['km/h'])
        return rows

    def _show_view(self):
        view=self.view.currentData();recovery=view in ('range_recovery','pri_recovery')
        radar=view in ('antenna','sparse_design','gain_budget')
        candidate_table=(recovery or radar) and self.recovery_table_toggle.isChecked()
        plot=view in VIEWS and not candidate_table;metrics=view=='metrics';compare=view=='comparisons';theory=view=='theory'
        self.recovery_table_toggle.setVisible(recovery or radar)
        self.recovery_table_toggle.setText('详细表格' if radar else '候选表')
        self.recovery_table_toggle.setToolTip('切换图形与完整结果表。' if radar else '显示全部候选区间；图中只展示前80项。')
        self.canvas.setVisible(plot);self.toolbar.setVisible(plot)
        self.table.setVisible(metrics or compare or candidate_table);self.check_table.setVisible(metrics or radar)
        self.table_note.setVisible(metrics or compare or recovery or radar)
        self.theory.setVisible(theory);self.comparison_bar.setVisible(compare)
        self.add_comparison_button.setEnabled(self.report is not None and not self._busy)
        self.remove_comparison_button.setEnabled(bool(self.comparisons) and not self._busy)
        if self.report is None:
            self.table.setRowCount(0);self.check_table.setRowCount(0);self.table_note.clear()
            self.theory.setPlainText('先设置参数并按F5计算。');return
        if radar:
            from .antenna_views import radar_table
            columns,rows,checks,note=radar_table(self.report,view,self.result_profile.currentData())
            if candidate_table:self._fill(self.table,columns,rows)
            self._fill(self.check_table,['检查项','状态','说明'],checks)
            self.check_table.horizontalHeader().setSectionResizeMode(2,QHeaderView.ResizeMode.Stretch)
            self.check_table.resizeRowsToContents();self.table_note.setText(note)
        elif metrics:
            self._fill(self.table,['指标','A','B','单位'],self._metric_rows())
            rows=[[{'pass':'通过','warn':'提示','fail':'失败'}.get(c['status'],c['status']),c['label'],c['detail']] for c in self.report['analysis']['checks']]
            self._fill(self.check_table,['状态','检查','说明'],rows)
            self.check_table.horizontalHeader().setSectionResizeMode(2,QHeaderView.ResizeMode.Stretch)
            self.check_table.resizeRowsToContents()
            self.table_note.setText('总帧指标见上方；速度单位另列km/h。DDM的原始慢时间范围不等于已解模糊范围。')
        elif compare:
            rows=[];self._comparison_row_indices=[]
            entries=[('当前结果',self.report,None)]+[(x['name'],x['report'],index) for index,x in enumerate(self.comparisons)]
            for name,snapshot,index in entries:
                for key in ('A','B'):
                    record=snapshot['analysis']['profiles'].get(key)
                    if record is None:continue
                    m=record['result']['metrics'];p=record['params'];combined=snapshot['analysis']['metrics']
                    rows.append([name,key,p['mode'].upper(),f"{p['txCount']}/{p['rxCount']}",value(m['rangeResolutionM']),value(m['maxVelocityMps']),value(m['maxVelocityMps']*3.6),value(m['velocityResolutionMps']),value(m['coherentMs']),value(combined['burstMs']),value(combined['frameMs']),value(combined['dataRateMBps'])])
                    self._comparison_row_indices.append(index)
            self._fill(self.table,['方案','分析组','模式','Tx/Rx','ΔR/m','±v/m/s','±v/km/h','Δv/m/s','组内CPI/ms','总采集/ms','帧/ms','MB/s'],rows)
            self.table_note.setText(f'已保存{len(self.comparisons)}/8个完整对比方案，A/B各列一行。移除任一组所在行会移除该完整方案。比较时同时核对CPI与帧预算；保存完整记录可保留本列表。')
        elif view in ('range_recovery','pri_recovery'):
            result=self.report['range_recovery'][self.result_profile.currentData()] if view=='range_recovery' else self.report['pri_recovery']
            observations=[]
            for i,measurement in enumerate(result.get('measurements',[]),1):
                if view=='range_recovery':
                    observations.append(f"观测{i}：折叠拍频 {value(measurement.get('foldedBeatMHz'))} MHz，表观距离 {value(measurement.get('apparentRangeM'))} m")
                else:
                    observations.append(f"观测{i}：周期 {value(measurement.get('periodUs'))} μs，时延余数 {value(measurement.get('foldedDelayUs',measurement.get('delayUs')))} μs")
            self.table_note.setText(STATUS.get(result['status'],result['status'])+'：'+result.get('reason','')+'\n'+'\n'.join(observations+result.get('warnings',[])))
            if candidate_table:
                if view=='range_recovery':
                    columns=['候选','距离/m','下界/m','上界/m','残差/Hz','折叠阶1','折叠阶2','镜像1','镜像2']
                    rows=[[i,value(r['rangeM']),value(r['minRangeM']),value(r['maxRangeM']),value(r['residualHz']),r['foldOrder1'],r['foldOrder2'],r['sign1'],r['sign2']] for i,r in enumerate(result.get('candidates',[]),1)]
                else:
                    columns=['候选','距离/m','下界/m','上界/m','周期阶1','周期阶2','残差/ns']
                    rows=[[i,value(r['rangeM']),value(r['minM']),value(r['maxM']),r['order1'],r['order2'],value(r['residualNs'])] for i,r in enumerate(result.get('candidates',[]),1)]
                self._fill(self.table,columns,rows)
        elif theory:self._show_theory()

    def _show_theory(self):
        self.theory.clear();lines=['# 公式与适用范围',
            '- ADC时长：N / fs；有效采样带宽：S × Tadc。',
            '- 名义距离分辨率：c / (2 Beff)；FFT补零只细化栅格。',
            '- 同组同Tx间隔决定速度范围；TDM需计轮流发射，DDM保守单槽范围还需除以编码槽数。',
            '- A/B分段各自做慢时间FFT；整轮交替用A、B及切换间隔构成的真实重访周期。',
            '- ADC折叠求解只使用两组观测、采集参数、模拟IF和外部速度先验；跨周期求解使用时延余数区间。',
            '- 频谱演示为理想单目标、无噪声、无距离迁移；未实现DDM盲解码或硬件配置。','']
        lines += ['- 天线虚拟位置为Tx+Rx，重合位置保留通道重数。HPBW与双目标角分辨率分别报告。',
                  '- 高分辨判据：MUSIC、MVDR、LS-ESPRIT或DML，已知K=2、有限含噪快拍；选择候选和最终验证使用独立种子。',
                  '- 增益预算：SNR = Pt·Gt·Gr·λ²·σ·Tadc·Nc·ηfast·ηslow·ΓRx / ((4π)³·R⁴·k·T·F·L)。']
        sources=[]
        for block in [self.report['analysis'],*[(x['result']) for x in self.report['analysis']['profiles'].values() if x],self.report['pri_recovery']]:
            for text in block.get('assumptions',[]):
                if '- '+text not in lines:lines.append('- '+text)
            sources.extend(block.get('sources',[]))
        for recovery in self.report['range_recovery'].values():
            for text in recovery.get('assumptions',[]):
                if '- '+text not in lines:lines.append('- '+text)
            sources.extend(recovery.get('sources',[]))
        lines+=['','## 原始资料'];seen=set()
        for source in sources:
            url=source.get('url','')
            if url.startswith('https://') and url not in seen:
                lines.append(f"- [{source.get('title',url)}]({url})");seen.add(url)
        self.theory.setMarkdown('\n'.join(lines))

    def _candidate_display_changed(self,*_):
        self._show_view();self._refresh_host()

    def redraw(self,*_):
        profile=self.result_profile.currentData()
        if self.report is not None and not self.report['analysis']['profiles'].get(profile):
            with QSignalBlocker(self.result_profile):self.result_profile.setCurrentIndex(0)
            profile='A'
        self._install_figure(self._make_figure(self.report,self.view.currentData(),profile))
        self._show_view();self.update_summary();self._refresh_host()

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if hasattr(self,'resize_timer'):self.resize_timer.start(60)

    def reflow(self):
        if not hasattr(self,'canvas'):return
        if self.canvas.resize_pending:return
        ratio=self.canvas.device_pixel_ratio
        self.figure.set_size_inches(max(1,self.canvas.width())*ratio/self.figure.dpi,max(1,self.canvas.height())*ratio/self.figure.dpi,forward=False)
        if getattr(self.figure,'_arraylab_responsive_plots',None):reflow_plot_layout(self.figure)
        else:self.canvas.setMinimumHeight(360)
        if hasattr(self.figure,'_arraylab_footer_lines'):draw_metric_footer(self.figure,self.figure._arraylab_footer_lines)
        self.canvas.draw_idle()

    def add_comparison(self,name):
        if self.report is None:raise ValueError('请先完成一次波形计算。')
        if self._busy:raise ValueError('计算或导出完成后再保存对比。')
        if not isinstance(name,str) or not 1<=len(name.strip())<=80:raise ValueError('对比名称须为1–80个字符。')
        if len(self.comparisons)>=8:raise ValueError('最多保存8个对比方案，请先移除不需要的条目。')
        self.comparisons.append({'name':name.strip(),'report':deepcopy(self.report)})
        self.view.setCurrentIndex(self.view.findData('comparisons'));self._show_view()

    def add_comparison_dialog(self):
        name,ok=QInputDialog.getText(self,'加入方案对比','方案名称（最多80字）')
        if ok:
            try:self.add_comparison(name)
            except ValueError as exc:QMessageBox.warning(self,'未加入对比',str(exc))

    def remove_selected_comparison(self):
        if self._busy:return
        row=self.table.currentRow();mapping=getattr(self,'_comparison_row_indices',[])
        index=mapping[row] if 0<=row<len(mapping) else None
        if index is None or not 0<=index<len(self.comparisons):
            self.state.setText('请选择一条已保存方案的A或B行；当前结果不能从对比列表移除。');return
        name=self.comparisons[index]['name'];self.comparisons.pop(index);self._show_view()
        self.state.setText('已移除完整对比方案“'+name+'”（包含其A/B组）；当前计算结果保留。')

    def save_path(self,path):
        from arraylab.features.waveform.io import save_waveform_configuration
        save_waveform_configuration(path,self.config())

    def open_path(self,path):
        if self._busy or self.thread is not None:raise ValueError('请先完成或取消波形计算。')
        from arraylab.features.waveform.io import load_waveform_configuration
        self.apply_configuration(load_waveform_configuration(path))
        self.state.setText('已载入波形、解模糊与雷达天线设置；按F5更新分析，综合使用独立按钮，旧结果保留。')

    load_path=open_path

    def save_dialog(self):
        path,_=QFileDialog.getSaveFileName(self,'保存波形参数',str(Path(self.host.last_directory)/'waveform.json'),'JSON (*.json)')
        if path:
            try:self.save_path(path)
            except (ValueError,OSError) as exc:QMessageBox.warning(self,'保存失败',str(exc))

    def open_dialog(self):
        path,_=QFileDialog.getOpenFileName(self,'打开ArrayLab或WaveLab波形参数',self.host.last_directory,'JSON (*.json)')
        if path:
            try:self.open_path(path)
            except (ValueError,OSError) as exc:QMessageBox.warning(self,'打开失败',str(exc))

    load_dialog=open_dialog

    def save_record_path(self,path):
        if self.report is None:raise ValueError('请先完成波形计算。')
        from arraylab.features.waveform.io import save_waveform_record
        save_waveform_record(path,self.report,view=self.view.currentData(),profile=self.result_profile.currentData(),comparisons=self.comparisons)

    def load_record_path(self,path):
        if self._busy or self.thread is not None:raise ValueError('请先完成或取消波形计算。')
        from arraylab.features.waveform.io import load_waveform_record
        record=load_waveform_record(path)
        self.accept_result(record['report'],restore_inputs=True,view=record['view'],profile=record.get('profile'),comparisons=record.get('comparisons',[]))

    def save_record_dialog(self):
        if self.report is None:return
        path,_=QFileDialog.getSaveFileName(self,'保存完整波形与对比记录',str(Path(self.host.last_directory)/'waveform-record.json'),'JSON (*.json)')
        if path:
            try:self.save_record_path(path)
            except (ValueError,OSError) as exc:QMessageBox.warning(self,'记录保存失败',str(exc))

    def load_record_dialog(self):
        path,_=QFileDialog.getOpenFileName(self,'打开完整波形记录',self.host.last_directory,'JSON (*.json)')
        if path:
            try:self.load_record_path(path)
            except (ValueError,OSError) as exc:QMessageBox.warning(self,'记录打开失败',str(exc))

    def export_path(self,path):
        if self.report is None:raise ValueError('请先完成波形计算。')
        from arraylab.features.waveform.io import export_waveform_directory
        export_waveform_directory(path,self.report,plot='all');self.last_export=Path(path)

    def export_dialog(self):
        if self.report is None or self._busy:return
        parent=QFileDialog.getExistingDirectory(self,'选择波形导出父目录',self.host.last_directory)
        if not parent:return
        target=Path(parent)/datetime.now().strftime('waveform_%Y%m%d_%H%M%S_%f')
        self._busy=True;self._refresh_host()
        try:self.export_path(target);self.state.setText('波形结果已导出：'+str(target))
        except (ValueError,OSError) as exc:QMessageBox.warning(self,'导出失败',str(exc))
        finally:self._busy=False;self._refresh_host()
