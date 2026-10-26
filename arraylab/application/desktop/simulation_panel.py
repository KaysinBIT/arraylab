"""One native workbench workflow shared by the three integrated radar tools."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import threading

from PySide6.QtCore import QThread,Signal,QTimer,QSignalBlocker
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QComboBox,QPushButton,
    QCheckBox,QDoubleSpinBox,QTableWidget,QTableWidgetItem,QHeaderView,QFileDialog,QMessageBox,QProgressBar,QGridLayout)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

from arraylab.application.simulation import provider,run_process,record_configuration,record_compatibility
from arraylab.framework.parallel import ComputationCancelled,requested_workers
from arraylab.shared.qt.configuration_form import ConfigurationForm
from arraylab.shared.qt.canvas import InteractiveFigureCanvas
from arraylab.shared.plotting.layout import reflow_plot_layout
from arraylab.shared.plotting.metrics import draw_metric_footer
from arraylab.shared.simulation_io import read_json,write_json,read_npz,write_npz,configuration_digest


def form_metadata(module):
    module=getattr(module,'metadata',module)
    metadata={key:deepcopy(getattr(module,key,{})) for key in ('GROUPS','LABELS','CHOICES','HELP','LIST_DEFAULTS')}
    scales={}
    microseconds={'pulse_width_s','ramp_s','adc_start_s','pri_s','pri_sequence_s','recovery_s','receive_start_s','receive_end_s'}
    for path,label in metadata['LABELS'].items():
        key=path.split('.')[-1]
        if key=='fc_hz':scales[path]=1e9;metadata['LABELS'][path]=label.replace('(Hz)','(GHz)')
        elif key in ('bandwidth_hz','sample_rate_hz','fs_hz','frequency_step_hz','receiver_bandwidth_hz','rx_bandwidth_hz'):
            scales[path]=1e6;metadata['LABELS'][path]=label.replace('(Hz)','(MHz)')
        elif key in microseconds:
            scales[path]=1e-6;metadata['LABELS'][path]=label.replace('(s)','(μs)')
    metadata['DISPLAY_SCALES']=scales
    return metadata


class SimulationWorker(QThread):
    progress=Signal(str)
    def __init__(self,function,parent=None,*,cancellable=True):
        super().__init__(parent);self.function=function;self.cancellable=cancellable
        self.stop=threading.Event();self.result=None;self.error=None;self.cancelled=False
    def run(self):
        try:self.result=self.function(self.progress.emit,self.stop.is_set)
        except (ComputationCancelled,InterruptedError):self.cancelled=True
        except Exception as exc:self.error=f'{type(exc).__name__}: {exc}'


class SimulationPanel(QWidget):
    def __init__(self,host,feature):
        super().__init__();self.host=host;self.feature=feature;self.backend=provider(feature)
        self.report=None;self.arrays={};self.thread=None;self._busy=False;self._operation='';self._changing=False;self._dirty=False
        self.scene_widget=None;self._scene_report_key=None
        self.parameter_widget=ConfigurationForm(self.backend.default_configuration(),form_metadata(self.backend))
        self.parameter_widget.changed.connect(self.mark_dirty)
        body=QVBoxLayout(self);body.setContentsMargins(8,6,8,6)
        self.state=QLabel('选择示例或设置参数，再执行所选操作。');self.state.setWordWrap(True);self.state.setObjectName('state');body.addWidget(self.state)
        self.snapshot_notice=QLabel();self.snapshot_notice.setWordWrap(True);self.snapshot_notice.setObjectName('warning');self.snapshot_notice.hide();body.addWidget(self.snapshot_notice)
        self.snapshot_notice.setStyleSheet('color:#855015;background:#fff5dd;padding:6px;border:1px solid #e7cb8e;')
        row=QHBoxLayout();row.addWidget(QLabel('操作'))
        self.action=QComboBox()
        for key,title in self.backend.ACTIONS:self.action.addItem(title,key)
        self.action.currentIndexChanged.connect(self.action_changed);row.addWidget(self.action,1)
        self.presets=self.backend.presets();self.preset=QComboBox()
        for item in self.presets:self.preset.addItem(item['name'],item['id'])
        row.addWidget(self.preset,1);self.preset_button=QPushButton('载入示例');self.preset_button.clicked.connect(self.load_preset);row.addWidget(self.preset_button)
        body.addLayout(row)
        self.progress=QProgressBar();self.progress.setRange(0,0);self.progress.hide();body.addWidget(self.progress)
        controls=QHBoxLayout();controls.addWidget(QLabel('结果视图'));self.view=QComboBox()
        for key,title in self.backend.VIEWS:self.view.addItem(title,key)
        controls.addWidget(self.view,1)
        self.table_toggle=QCheckBox('详细表格');controls.addWidget(self.table_toggle)
        self.selection=QComboBox();self.selection.setMinimumWidth(95);controls.addWidget(self.selection)
        self.scene_time=QDoubleSpinBox();self.scene_time.setRange(0,5);self.scene_time.setDecimals(3);self.scene_time.setSingleStep(.1);self.scene_time.setSuffix(' s');controls.addWidget(self.scene_time)
        body.addLayout(controls)
        self.figure=Figure(figsize=(9,6),dpi=100);self.canvas=InteractiveFigureCanvas(self.figure);self.canvas.setMinimumSize(480,360)
        body.addWidget(self.canvas,1);self.plot_layout=body
        self.toolbar=NavigationToolbar2QT(self.canvas,self);body.addWidget(self.toolbar)
        self.table_widget=QTableWidget();self.table_widget.setAlternatingRowColors(True)
        self.table_widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers);body.addWidget(self.table_widget,1)
        self.note=QLabel();self.note.setWordWrap(True);body.addWidget(self.note)
        action_bar=QWidget();self.integration_bar=QGridLayout(action_bar);self.integration_bar.setContentsMargins(0,0,0,0);body.addWidget(action_bar)
        self._integration_widgets=[]
        self.apply_button=QPushButton('应用所选候选');self.apply_button.clicked.connect(self.apply_candidate)
        self.calibration_button=QPushButton('应用当前H0校准');self.calibration_button.clicked.connect(self.apply_calibration)
        self.reset_array_button=QPushButton('按通道数重建规则阵列');self.reset_array_button.clicked.connect(self.reset_array)
        self.terrain_button=QPushButton('导入GeoTIFF地形');self.terrain_button.clicked.connect(self.import_terrain)
        for widget,visible in [(self.apply_button,feature in ('pulse','phased')),(self.calibration_button,feature=='equation'),
                               (self.reset_array_button,feature in ('pulse','equation')),(self.terrain_button,feature=='pulse')]:
            self.add_integration_action(widget,visible=visible)
        self.view.currentIndexChanged.connect(self.redraw);self.table_toggle.toggled.connect(self.redraw)
        self.selection.currentIndexChanged.connect(self.redraw);self.scene_time.valueChanged.connect(self.redraw)
        self.canvas.resize_finished.connect(self.reflow)
        self.resize_timer=QTimer(self);self.resize_timer.setSingleShot(True);self.resize_timer.timeout.connect(self.reflow)
        self.redraw()
        self.update_input_availability()

    def add_integration_action(self,widget,index=None,*,visible=True):
        widget.setParent(self.integration_bar.parentWidget())
        widget.setVisible(visible)
        self._integration_widgets.insert(len(self._integration_widgets) if index is None else index,(widget,visible))
        while self.integration_bar.count():self.integration_bar.takeAt(0)
        shown=[w for w,show in self._integration_widgets if show]
        for i,item in enumerate(shown):self.integration_bar.addWidget(item,i//2,i%2)
        self.integration_bar.setColumnStretch(0,1);self.integration_bar.setColumnStretch(1,1)

    def refresh_host(self,*_):
        if getattr(self.host,'_workflow_ready',False) and self.feature in getattr(self.host,'simulation_panels',{}):self.host.refresh_actions()

    def configuration(self):return self.backend.normalize_configuration(self.parameter_widget.value())
    read_parameters=configuration

    def apply_configuration(self,configuration,*,from_record=False):
        configuration=record_configuration(self.feature,configuration) if from_record else self.backend.normalize_configuration(configuration)
        self._changing=True
        try:self.parameter_widget.set_value(configuration)
        finally:self._changing=False
        self.mark_dirty()

    def mark_dirty(self,*_):
        if not self._changing:
            self._dirty=True;self.state.setText('输入已改变；结果与导出仍对应上次成功快照。请选择操作后按F5。')
            self.update_context_actions()
            self.update_input_availability()

    def action_changed(self,*_):
        if not hasattr(self,'state'):return
        if not self._changing:self.state.setText('已选择“'+self.action.currentText()+'”；按F5执行，当前结果保留。')
        self.refresh_host()

    def update_input_availability(self):
        editors=self.parameter_widget.editors
        def value(path,default=None):
            try:return editors[path].value() if path in editors else default
            except (ValueError,TypeError):return default
        def enable(paths,condition):
            for path in paths:
                if path in editors:editors[path].setEnabled(bool(condition))
        if self.feature=='phased':
            geometry=value('array.geometry');custom=geometry=='custom'
            enable(['array.custom'],custom);enable(['array.nx','array.dx','array.window'],not custom)
            enable(['array.ny','array.dy'],geometry=='rectangular')
            enable(['array.sidelobeDb'],not custom and value('array.window')=='taylor')
            continuous=value('planar.mode')=='continuous'
            enable(['planar.gridNx','planar.gridNy','planar.dx','planar.dy'],not continuous)
            enable(['planar.width','planar.height','planar.amplitudeMin','planar.optimizePositions','planar.optimizeAmplitudes','planar.initialization'],continuous)
            enable(['planar.initialPositions'],continuous and value('planar.initialization')=='provided')
        elif self.feature=='equation':
            fmcw=value('waveform.kind')=='fmcw'
            enable(['waveform.ramp_s','waveform.adc_start_s'],fmcw)
            enable(['waveform.pulse_width_s'],not fmcw)
            enable(['array.ddm_slots'],value('array.mode')=='ddm')
            nf=value('radar.noise_mode')=='nf'
            enable(['radar.antenna_temperature_k','radar.noise_figure_db'],nf);enable(['radar.system_temperature_k'],not nf)
            mode=value('propagation.model');enable(['propagation.epsilon_r','propagation.conductivity_s_m'],mode=='two_ray')
            enable(['propagation.custom_paths'],mode=='custom')
            atmosphere=value('propagation.atmosphere.enabled',False)
            enable([path for path in editors if path.startswith('propagation.atmosphere.') and not path.endswith('.enabled')],atmosphere)
            enable([path for path in editors if path.startswith('clutter.') and not path.endswith('.kind')],value('clutter.kind')!='none')
        else:
            kind=value('waveform.kind');stepped=kind=='stepped'
            enable(['waveform.frequency_steps','waveform.frequency_step_hz','timing.groups'],stepped)
            enable(['timing.pulses'],not stepped);enable(['waveform.phase_code'],kind=='phase')
            enable(['waveform.bandwidth_hz'],kind=='lfm')
            enable(['timing.pri_sequence_s'],value('timing.mode')!='uniform')
            custom=value('array.geometry')=='custom'
            enable(['array.tx_positions_m','array.rx_positions_m'],custom);enable(['array.spacing_lambda'],not custom)
            enable([path for path in editors if path.startswith('processing.doa.') and not path.endswith('.enabled')],value('processing.doa.enabled',True))

    def update_context_actions(self):
        if not hasattr(self,'integration_bar'):return
        for index in range(self.integration_bar.count()):
            widget=self.integration_bar.itemAt(index).widget()
            if widget is not None:widget.setEnabled(not self._busy)
        result=self.report['result'] if self.report else {}
        compatibility=record_compatibility(self.report)
        self.snapshot_notice.setText(compatibility.get('message',''))
        self.snapshot_notice.setVisible(not compatibility.get('current',False))
        available=bool(result.get('candidates')) if self.feature=='pulse' else bool(result.get('synthesis'))
        self.apply_button.setEnabled(not self._busy and available)
        self.calibration_button.setEnabled(not self._busy and not self._dirty and compatibility.get('current',False) and result.get('kind')=='calibration' and result.get('ready') is True)
        if getattr(self.host,'_workflow_ready',False):
            getattr(self.host,'refresh_simulation_handoffs',lambda:None)()

    def load_preset(self):
        if self._busy:return
        index=self.preset.currentIndex()
        if index<0:return
        self.apply_configuration(self.presets[index]['configuration'])
        self.action.setCurrentIndex(0)
        self.state.setText('示例已载入，尚未计算。请选择操作后按F5。')

    def _launch(self,function,operation,*,cancellable=True):
        if self._busy or self.thread is not None:return
        if getattr(self.host,'simulation_busy',lambda *_:False)(self):return
        self._busy=True;self._operation=operation;self.parameter_widget.setEnabled(False)
        self.action.setEnabled(False);self.preset_button.setEnabled(False);self.progress.show()
        worker=SimulationWorker(function,self,cancellable=cancellable);self.thread=worker
        worker.progress.connect(self.state.setText);worker.finished.connect(self.complete)
        self.state.setText('正在'+operation+'…');worker.start();self.refresh_host()
        self.update_context_actions()

    def start(self,action=None):
        if self._busy:return
        if type(action) is bool:action=None
        action=action or self.action.currentData()
        try:
            config=self.configuration();validation=self.backend.validate_configuration(config)
            if not validation.get('valid'):raise ValueError('；'.join(map(str,validation.get('errors',[]))))
        except (ValueError,TypeError,KeyError) as exc:QMessageBox.warning(self,'参数检查',str(exc));return
        workers=requested_workers()
        self._launch(lambda progress,cancel:run_process(self.feature,action,config,workers=workers,progress=progress,cancel=cancel),'计算')

    def cancel(self):
        if self.thread is not None:
            if not self.thread.cancellable:self.state.setText('正在完成原子文件写入，请等待。');return
            self.thread.stop.set();self.state.setText('正在取消并回收计算进程；保留原结果。')

    def complete(self):
        worker=self.thread;operation=self._operation
        if worker is None:return
        try:
            if worker.cancelled or operation=='计算' and worker.cancellable and worker.stop.is_set():self.state.setText('操作已取消，原结果保留。')
            elif worker.error:self.state.setText('操作未完成，原结果保留：'+worker.error)
            elif operation in ('计算','打开记录'):
                result,arrays=worker.result;self.accept_result(result,arrays,restore_inputs=operation=='打开记录')
            elif operation=='导入地形':
                config=self.configuration();config['scene']['terrain']=worker.result
                self.apply_configuration(config);self.state.setText('地形已导入工程；运行前请核对高度基准、位置和散射设置。')
            else:self.state.setText(operation+'完成。')
        except Exception as exc:self.state.setText('结果未更新：'+str(exc))
        finally:
            self.thread=None;worker.deleteLater();self._busy=False;self.progress.hide()
            self.parameter_widget.setEnabled(True);self.action.setEnabled(True);self.preset_button.setEnabled(True)
            self.update_context_actions()
            self.refresh_host()
            if getattr(self.host,'_simulation_close_pending',False):self.host._simulation_close_pending=False;QTimer.singleShot(0,self.host.close)

    def accept_result(self,report,arrays,*,restore_inputs=False):
        if report.get('format')!='arraylab.simulation.result' or report.get('schema_version')!=1 or report.get('feature')!=self.feature:
            raise ValueError('记录类型与当前模块不一致。')
        configuration=record_configuration(self.feature,report['configuration'])
        if configuration_digest(configuration)!=report.get('configuration_sha256'):raise ValueError('记录输入摘要不一致。')
        from arraylab.application.simulation import validate_record
        validate_record(report,arrays,expected_feature=self.feature)
        old=(self.report,self.arrays,self.parameter_widget.snapshot(),self.view.currentData(),
             [(self.selection.itemText(i),deepcopy(self.selection.itemData(i))) for i in range(self.selection.count())],
             self.selection.currentIndex(),self._dirty,self.action.currentIndex(),self.table_toggle.isChecked(),self.scene_time.value(),self.scene_time.maximum())
        try:
            self.report=report;self.arrays=arrays
            if restore_inputs:self.apply_configuration(configuration,from_record=True)
            if restore_inputs:
                with QSignalBlocker(self.action):self.action.setCurrentIndex(self.action.findData(report['action']))
            preferred={'analysis':'cuts','budget':'budget' if self.feature=='equation' else 'metrics','simulation':'rd','simulate':'rd','snr':'snr',
                       'inverse':'candidates','statistics':'statistics','calibration':'statistics','distance':'statistics','roc':'statistics',
                       'scan':'scan','optimize1d':'cuts','optimize2d':'uv'}.get(report['action'])
            index=self.view.findData(preferred)
            with QSignalBlocker(self.view),QSignalBlocker(self.selection):
                if index>=0:self.view.setCurrentIndex(index)
                self.selection.clear()
                selections=getattr(self.backend,'selections',lambda r:[])(report['result'])
                for value,label in selections:self.selection.addItem(str(label),value)
                if not selections:
                    candidates=report['result'].get('candidates',report['result'].get('summary',{}).get('candidates',[]))
                    for i in range(len(candidates)):self.selection.addItem(f'候选 {i+1}',i)
                state=report.get('view_state',{}) if restore_inputs else {}
                saved_view=self.view.findData(state.get('view'))
                if saved_view>=0:self.view.setCurrentIndex(saved_view)
                selected=self.selection.findData(state.get('selection'))
                if selected>=0:self.selection.setCurrentIndex(selected)
            with QSignalBlocker(self.table_toggle),QSignalBlocker(self.scene_time):
                if self.feature=='pulse':
                    scene=configuration['scene'];items=[scene['platform'],*scene['targets']]
                    maximum=max([5.,*[float(p['time_s']) for item in items for p in item.get('trajectory',[])]])
                    self.scene_time.setMaximum(maximum)
                if state:
                    self.table_toggle.setChecked(state.get('table',False));self.scene_time.setValue(state.get('scene_time',0))
            self.redraw()
        except Exception:
            self.report,self.arrays=old[:2]
            if restore_inputs:
                self._changing=True
                try:self.parameter_widget.restore_snapshot(old[2])
                finally:self._changing=False
            with QSignalBlocker(self.view),QSignalBlocker(self.selection),QSignalBlocker(self.action),QSignalBlocker(self.table_toggle),QSignalBlocker(self.scene_time):
                self.view.setCurrentIndex(self.view.findData(old[3]));self.selection.clear()
                for title,value in old[4]:self.selection.addItem(title,value)
                self.selection.setCurrentIndex(old[5])
                self.action.setCurrentIndex(old[7]);self.table_toggle.setChecked(old[8]);self.scene_time.setMaximum(old[10]);self.scene_time.setValue(old[9])
            self._dirty=old[6]
            self.redraw();raise
        try:self._dirty=configuration_digest(self.configuration())!=report['configuration_sha256']
        except (ValueError,TypeError):self._dirty=True
        self.state.setText('已恢复完整计算快照，未重新求解。' if restore_inputs else '本次计算完成，结果对应成功提交的输入。')
        self.update_context_actions()
        self.refresh_host()

    def redraw(self,*_):
        self.update_context_actions()
        view=self.view.currentData();scene=self.feature=='pulse' and view=='scene'
        self.scene_time.setVisible(scene);self.selection.setVisible(self.selection.count()>0 and not scene)
        is_table=self.table_toggle.isChecked() or view in ('detections','metrics','elements','candidates','cases','ambiguity')
        scene_graph=scene and not is_table
        self.canvas.setVisible(not is_table and not scene_graph);self.toolbar.setVisible(not is_table and not scene_graph);self.table_widget.setVisible(is_table)
        if self.scene_widget is not None:self.scene_widget.setVisible(scene_graph)
        if scene_graph:
            self.show_scene();self.refresh_host();return
        if self.report is None:
            self.figure.clear();ax=self.figure.add_subplot(111);ax.set_axis_off();ax.text(.5,.5,'设置参数，选择操作后按F5运行',ha='center',va='center',transform=ax.transAxes)
            self.canvas.draw_idle();self.table_widget.setRowCount(0);self.note.setText('尚无计算结果。打开配置与切换页面不会自动求解。');return
        result=self.report['result'];selection=self.scene_time.value() if scene else self.selection.currentData() or 0
        columns,rows,note=self.backend.table(result,self.arrays,view,selection=selection)
        self.note.setText(note)
        if is_table:
            self.table_widget.setColumnCount(len(columns));self.table_widget.setHorizontalHeaderLabels(columns)
            self.table_widget.setRowCount(min(5000,len(rows)))
            for i,row in enumerate(rows[:5000]):
                for j,value in enumerate(row):self.table_widget.setItem(i,j,QTableWidgetItem('—' if value is None else str(value)))
            self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            if len(rows)>5000:self.note.setText(note+'\n表格显示前5000行，完整数据保留在导出文件。')
        else:
            old=self.figure;figure=Figure(figsize=(9,6),dpi=old.dpi);figure._original_dpi=old._original_dpi
            self.canvas.reset_interaction();self.canvas.figure=figure;figure.set_canvas(self.canvas)
            try:self.backend.draw(figure,result,self.arrays,view,selection=selection)
            except Exception:
                self.canvas.figure=old;old.set_canvas(self.canvas);raise
            self.figure=figure;previous=self.toolbar;self.toolbar=NavigationToolbar2QT(self.canvas,self)
            self.plot_layout.replaceWidget(previous,self.toolbar);previous.hide();previous.deleteLater();self.reflow()
        self.refresh_host()

    def show_scene(self):
        if self.scene_widget is None:
            self.scene_widget=self.backend.SceneWidget(self)
            self.plot_layout.insertWidget(self.plot_layout.indexOf(self.canvas),self.scene_widget,1)
            self.scene_widget.timeChanged.connect(self.update_scene_time)
        self.scene_widget.show()
        if self.report is None:
            config=self.configuration();result={'configuration':config}
            key=('input',configuration_digest(config))
        else:result=self.report['result'];key=('result',id(self.report))
        if key!=self._scene_report_key:
            payload=self.backend.scene_payload(result,self.scene_time.value())
            self.scene_time.setMaximum(payload['max_time_s'])
            camera=self.report.get('view_state',{}).get('scene_camera') if self.report else None
            self.scene_widget.set_scene(payload,key=key,camera=camera);self._scene_report_key=key
        else:self.scene_widget.set_time(self.scene_time.value())
        self.note.setText('原PulseLab三维场景：雷达、地形、目标和波束方向示意。透明锥体不是实际增益等值面；播放、旋转与缩放不重新运行回波计算。')

    def update_scene_time(self,value):
        with QSignalBlocker(self.scene_time):self.scene_time.setValue(value)

    def shutdown_scene(self):
        if self.scene_widget is not None:self.scene_widget.shutdown()

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if hasattr(self,'resize_timer'):self.resize_timer.start(60)

    def reflow(self):
        if not self.canvas.isVisible():return
        if self.canvas.resize_pending:return
        ratio=self.canvas.device_pixel_ratio
        self.figure.set_size_inches(max(1,self.canvas.width())*ratio/self.figure.dpi,max(1,self.canvas.height())*ratio/self.figure.dpi,forward=False)
        reflow_plot_layout(self.figure)
        if hasattr(self.figure,'_arraylab_footer_lines'):draw_metric_footer(self.figure,self.figure._arraylab_footer_lines)
        self.canvas.draw_idle()

    def save_path(self,path):
        configuration=self.configuration()
        if self.feature=='pulse' and Path(path).suffix.lower()=='.prlab':self.backend.export_project(path,configuration)
        else:write_json(path,{'format':'arraylab.simulation','schema_version':1,'feature':self.feature,'configuration':configuration,'operation':self.action.currentData()})

    def open_path(self,path):
        if self._busy:raise ValueError('请先结束当前任务。')
        from arraylab.application.simulation import load_configuration
        configuration=load_configuration(self.feature,path)
        self.apply_configuration(configuration)
        index=0
        if Path(path).suffix.lower()=='.json':
            document=read_json(path)
            if document.get('format')=='arraylab.simulation':index=max(0,self.action.findData(document.get('operation')))
        self.action.setCurrentIndex(index)

    def save_dialog(self):
        path,_=QFileDialog.getSaveFileName(self,'保存当前工程',str(Path(self.host.last_directory)/(self.feature+'.json')),
                                          'JSON工程 (*.json);;脉冲便携工程 (*.prlab)' if self.feature=='pulse' else 'JSON工程 (*.json)')
        if path:
            try:self.save_path(path);self.state.setText('工程已保存；未计算的新输入也包含在内。')
            except Exception as exc:QMessageBox.warning(self,'保存失败',str(exc))

    def open_dialog(self):
        path,_=QFileDialog.getOpenFileName(self,'打开当前模块工程',self.host.last_directory,'工程 (*.json *.prlab)')
        if path:
            try:self.open_path(path)
            except Exception as exc:QMessageBox.warning(self,'打开失败',str(exc))

    def save_record_path(self,path):
        if self.report is None:raise ValueError('请先完成一次计算。')
        write_npz(path,self.record_snapshot(),self.arrays)

    def record_snapshot(self):
        camera=None
        if self.scene_widget is not None and self.scene_widget.ready:
            camera=self.scene_widget.get_state(sync=True).get('camera')
        state={'view':self.view.currentData(),'selection':self.selection.currentData(),
               'table':self.table_toggle.isChecked(),'scene_time':self.scene_time.value()}
        if camera:state['scene_camera']=camera
        return {**self.report,'view_state':state}

    def save_record_dialog(self):
        if self.report is None:return
        path,_=QFileDialog.getSaveFileName(self,'保存完整计算记录',str(Path(self.host.last_directory)/(self.feature+'-record.npz')),'完整记录 (*.npz)')
        if path:
            result,arrays=self.record_snapshot(),self.arrays
            self._launch(lambda progress,cancel:write_npz(path,result,arrays),'保存记录',cancellable=False)

    def load_record_path(self,path):
        from arraylab.application.simulation import load_record
        report,arrays=load_record(path);self.accept_result(report,arrays,restore_inputs=True)

    def load_record_async(self,path):
        from arraylab.application.simulation import load_record
        self._launch(lambda progress,cancel:load_record(path),'打开记录',cancellable=False)

    def export_dialog(self):
        if self.report is None:return
        parent=QFileDialog.getExistingDirectory(self,'选择导出父目录',self.host.last_directory)
        if parent:
            target=Path(parent)/(self.feature+'-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
            scene_png=None
            if self.scene_widget is not None and self.scene_widget.isVisible():
                try:
                    self.scene_widget.pause();scene_png=self.scene_widget.png_bytes()
                except ValueError as exc:QMessageBox.warning(self,'场景未导出',str(exc));return
            result,arrays=self.record_snapshot(),self.arrays
            def export(progress,cancel):
                from arraylab.application.simulation import export_process
                return export_process(target,result,arrays,progress=progress,cancel=cancel,scene_png=scene_png)
            self._launch(export,'导出结果',cancellable=True)

    def apply_calibration(self):
        if self._busy or self.report is None:return
        try:
            self.apply_configuration(self.backend.apply_calibration(self.configuration(),self.report['result']))
            self.state.setText('校准门限已应用；请用独立统计操作验证，不把校准当作验收。')
        except Exception as exc:QMessageBox.warning(self,'校准未应用',str(exc))

    def apply_candidate(self):
        if self._busy or self.report is None:return
        try:
            index=self.selection.currentData()
            if index is None:index=max(0,self.table_widget.currentRow())
            method=self.backend.apply_inverse_candidate if self.feature=='pulse' else self.backend.apply_selected_case
            self.apply_configuration(method(self.report['result'],int(index)))
            self.state.setText('已应用候选配置；旧结果保留，请选择分析/仿真并按F5重新核对。')
        except Exception as exc:QMessageBox.warning(self,'候选未应用',str(exc))

    def reset_array(self):
        if self._busy:return
        try:
            config=self.parameter_widget.value()
            if self.feature=='equation':config=self.backend.reset_array(config)
            else:
                config['array']['tx_positions_m']=[];config['array']['rx_positions_m']=[]
                if config['array']['geometry']=='custom':config['array']['geometry']='ula'
            self.apply_configuration(config);self.state.setText('已按当前通道数重建规则阵列初值，尚未计算。')
        except Exception as exc:QMessageBox.warning(self,'阵列未更新',str(exc))

    def import_terrain(self):
        if self._busy:return
        path,_=QFileDialog.getOpenFileName(self,'导入单波段GeoTIFF',self.host.last_directory,'GeoTIFF (*.tif *.tiff)')
        if not path:return
        from PySide6.QtWidgets import QDialog,QDialogButtonBox
        dialog=QDialog(self);dialog.setWindowTitle('地形导入范围与高度基准');dialog.resize(480,520);layout=QVBoxLayout(dialog)
        options={'max_size':65,'height_unit':'auto','vertical_datum':'','crop_bounds':[], 'sigma0_db':-35.,'seed':83}
        metadata={'LABELS':{'max_size':'最大分析边长','height_unit':'原高程单位','vertical_datum':'原高程基准（请核实）',
                  'crop_bounds':'裁剪范围：原CRS的左、下、右、上；留空完整范围','sigma0_db':'地面后向散射系数 (dB)','seed':'散射种子'},
                  'CHOICES':{'height_unit':[('读取文件标注','auto'),('米（显式指定）','m'),('英尺（显式指定）','ft')]}}
        form=ConfigurationForm(options,metadata);layout.addWidget(form)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject)
        if dialog.exec()==QDialog.DialogCode.Accepted:
            try:
                values=form.value()
                if not values['crop_bounds']:values.pop('crop_bounds')
                if values['height_unit']=='auto':values.pop('height_unit')
                if not values['vertical_datum'].strip():values.pop('vertical_datum')
                self._launch(lambda progress,cancel:self.backend.import_dem(path,values),'导入地形',cancellable=False)
            except Exception as exc:QMessageBox.warning(self,'地形未导入',str(exc))
