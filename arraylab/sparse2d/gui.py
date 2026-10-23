"""Sparse array experiment controls, hosted by the ArrayLab integration window."""
from datetime import datetime
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np

from PySide6.QtCore import QObject, QThread, QTimer, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices, QFont, QFontDatabase, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication,QComboBox,QDoubleSpinBox,QFileDialog,
    QFormLayout,QFrame,QHBoxLayout,QHeaderView,QLabel,QLayout,QLineEdit,QMainWindow,QMessageBox,
    QPushButton,QScrollArea,QSpinBox,QSplitter,QTableWidget,QTableWidgetItem,QTabWidget,
    QTextBrowser,QVBoxLayout,QWidget)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from .core import Config,available_mask,run
from .plots import comparison_figure,draw_case_results,draw_element,draw_weight_comparison,draw_phase_comparison,result_name,structure_figure,style
from .detail_view import DetailView
from .weighting import METHODS,Weighting,add_weighting
from .scenarios import LIGHT_MM_GHZ,scenarios
from .elements import MODELS,build_element,write_example_csv
from .runtime_paths import resource_path,document_directory
from .phase_quantization import excitation_phases


class Worker(QObject):
    complete=Signal(object)
    failed=Signal(str)
    progress=Signal(str)

    def __init__(self,config=None,weight_payload=None):
        super().__init__(); self.config=config; self.weight_payload=weight_payload

    @Slot()
    def calculate(self):
        try:
            if self.weight_payload is None: result=run(self.config,self.progress.emit)
            else:
                bundle,index,settings=self.weight_payload
                result=add_weighting(bundle,index,settings,self.progress.emit)
            self.complete.emit(result)
        except Exception as exc: self.failed.emit(f'{type(exc).__name__}: {exc}')


class Window(QMainWindow):
    def __init__(self,config=Config(),autorun=False):
        super().__init__()
        style(); self.bundle=None; self.thread=None; self.last_export=None; self._busy=False; self._last_selected_row=None
        self._initializing=True; self._last_error=None; self._job_kind=None
        self._initial_phase_bits=config.phase_bits
        self.setWindowTitle('二维稀疏阵列实验台 · 独立 Demo')
        self.resize(1460,1000); self.setMinimumSize(1160,820)
        central=QWidget(); self.setCentralWidget(central)
        layout=QVBoxLayout(central); layout.setContentsMargins(20,16,20,12); layout.setSpacing(12)
        title=QLabel('二维稀疏阵列实验台'); title.setObjectName('title')
        subtitle=QLabel('固定阵元数选点 · 方向图综合 · 差分共阵   /   独立研究 Demo')
        subtitle.setStyleSheet('color:#617287;font-size:13px')
        layout.addWidget(title); layout.addWidget(subtitle)
        split=QSplitter(); layout.addWidget(split,1)
        left=QFrame(); left.setObjectName('panel'); left.setMaximumWidth(260)
        leftbox=QVBoxLayout(left); leftbox.setContentsMargins(16,16,16,16); leftbox.setSpacing(12)
        leftbox.addWidget(QLabel('实验参数'))
        self.hole_shape=QComboBox()
        for label,data in [('无中心孔','none'),('中心矩形孔','rectangle'),('中心圆孔','circle')]:
            self.hole_shape.addItem(label,data)
        self.hole_shape.setCurrentIndex(self.hole_shape.findData(config.hole_shape))
        leftbox.addWidget(self.hole_shape)
        form=QFormLayout(); form.setVerticalSpacing(12); self.inputs={}
        specs=[('nx','X 向网格',4,20,2),('ny','Y 向网格',4,20,2),('k','目标阵元 K',4,400,2),
               ('spacing','网格间距 / λ0',.1,2,.1),('theta','扫描 θ / °',0.,60.,5.),
               ('phi','方位 φ / °',-360.,360.,15.),('seed','随机种子',0,2147483647,1),
               ('iterations','退火步数',1,20000,100),('opt_grid','优化网格',25,101,2),
               ('eval_grid','复核网格',81,401,20),('hole_x','孔宽X / 直径 / λ0',.01,100.,.25),
               ('hole_y','矩形孔高Y / λ0',.01,100.,.25),('amplitude_dr_db','调幅范围 / dB',0.,40.,5.)]
        for key,label,lo,hi,step in specs:
            box=QDoubleSpinBox() if type(lo) is float else QSpinBox()
            box.setRange(lo,hi); box.setSingleStep(step); box.setValue(getattr(config,key))
            if isinstance(box,QDoubleSpinBox): box.setDecimals(2)
            box.valueChanged.connect(self.mark_stale); self.inputs[key]=box
            form.addRow(label,box)
        leftbox.addLayout(form)
        self.availability=QLabel(); self.availability.setWordWrap(True)
        self.availability.setStyleSheet('color:#617287;font-size:12px')
        leftbox.addWidget(self.availability)
        self.hollow_preset=QPushButton('中心掏洞示例')
        self.hollow_preset.clicked.connect(self.load_hollow_preset); leftbox.addWidget(self.hollow_preset)
        self.run_button=QPushButton('开始比较'); self.run_button.setObjectName('primary')
        self.run_button.clicked.connect(self.start); leftbox.addWidget(self.run_button)
        self.export_button=QPushButton('导出本次结果'); self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export); leftbox.addWidget(self.export_button)
        note=QLabel('中心孔内的位置禁止激活。\n调幅范围只约束固定位置调幅。\n\n所有选点按中心对称成对保留。\n满阵与 ℓ1 加权解阵元数不同。\n\nθ 从 +z 轴量起；φ 在 xy 平面\n从 +x 朝 +y 量起。\n\nPSL：固定掩膜外采样峰值。\nHPBW：方向余弦宽度。\n\nF1：操作指南')
        note.setWordWrap(True); note.setStyleSheet('color:#617287;font-size:12px;line-height:1.6')
        leftbox.addWidget(note); leftbox.addStretch()
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(left); scroll.setMinimumWidth(265); scroll.setMaximumWidth(285)
        split.addWidget(scroll)
        right=QWidget(); rightbox=QVBoxLayout(right); rightbox.setContentsMargins(12,0,0,0)
        self.snapshot=QLabel('尚无已计算结果。请在左侧完成设置后开始计算。')
        self.snapshot.setWordWrap(True); self.snapshot.setObjectName('snapshot'); rightbox.addWidget(self.snapshot)
        casebar=QHBoxLayout(); casebar.addWidget(QLabel('显示工况'))
        self.case_selector=QComboBox(); self.case_selector.setEnabled(False)
        self.case_selector.currentIndexChanged.connect(self.select_result)
        casebar.addWidget(self.case_selector,1); rightbox.addLayout(casebar)
        self.pattern_view=QComboBox()
        for label,value in [('总方向图：单元 × AF','total'),('仅阵因子 AF','array_factor'),('仅单元方向图','element')]: self.pattern_view.addItem(label,value)
        self.pattern_view.setToolTip('仅切换“选中方案”图形；表格指标和加权对比仍使用已计算单元的总方向图。')
        self.pattern_view.currentIndexChanged.connect(self.select_result); casebar.addWidget(self.pattern_view)
        self.phase_view=QComboBox(); self.phase_view.addItem('理想相位','ideal'); self.phase_view.addItem('量化实际相位','quantized')
        self.phase_view.setToolTip('仅切换方案详情及大图；主表、选点/加权 LP 和加权对比使用理想相位。量化曲线保留扫描损失。')
        self.phase_view.setEnabled(False); self.phase_view.currentIndexChanged.connect(self.select_result); casebar.addWidget(self.phase_view)
        self.table=QTableWidget(0,11)
        self.table.setHorizontalHeaderLabels(['方法','阵元','最差 PSL/dB','网格差/dB','幅度范围/dB','HPBW u','HPBW v','不同 lag','中心无孔 lag','最小距/λ0','间距违规'])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide(); self.table.setMaximumHeight(290)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self.select_result); rightbox.addWidget(self.table)
        self.tabs=QTabWidget(); rightbox.addWidget(self.tabs,1)
        self.detail_view=DetailView(self); self.detail_panel=self.detail_view
        self.tabs.addTab(self.detail_panel,'选中方案')
        self.compare_panel=QWidget(); self.compare_box=QVBoxLayout(self.compare_panel)
        self.tabs.addTab(self.compare_panel,'方法比较')
        self.joint_panel=QWidget(); jointbox=QVBoxLayout(self.joint_panel)
        description=QLabel('共同物理布局、共同幅度；联合各扫描角 × 各频率的最差副瓣。相位逐频理想校正。尺寸与最小间距按参考波长 λ0 定义。')
        description.setWordWrap(True); jointbox.addWidget(description)
        jointform=QFormLayout()
        self.joint_scans=QLineEdit(config.scan_angles); self.joint_scans.setMaxLength(2000)
        self.joint_scans.setPlaceholderText('留空用左侧 θ/φ；示例 0,0;30,0;30,90')
        self.joint_frequencies=QLineEdit(config.frequencies_ghz); self.joint_frequencies.setMaxLength(500)
        self.joint_frequencies.setPlaceholderText('留空用参考频率；示例 9,10,11（GHz）')
        ref=QDoubleSpinBox(); ref.setRange(.1,1000); ref.setDecimals(4); ref.setValue(config.reference_ghz)
        minimum=QDoubleSpinBox(); minimum.setRange(0,100); minimum.setDecimals(3); minimum.setSingleStep(.05); minimum.setValue(config.min_spacing)
        self.inputs['reference_ghz']=ref; self.inputs['min_spacing']=minimum
        jointform.addRow('参考频率 f0 / GHz',ref); jointform.addRow('频率列表 / GHz',self.joint_frequencies)
        jointform.addRow('扫描角列表 θ,φ / °',self.joint_scans); jointform.addRow('最小欧氏中心距 / λ0',minimum)
        jointbox.addLayout(jointform)
        self.joint_preview=QLabel(); self.joint_preview.setWordWrap(True); jointbox.addWidget(self.joint_preview)
        actions=QHBoxLayout(); self.joint_preset=QPushButton('联合综合示例（3角 × 3频）')
        self.joint_preset.clicked.connect(self.load_joint_preset)
        self.joint_run=QPushButton('开始联合综合'); self.joint_run.setObjectName('primary'); self.joint_run.clicked.connect(self.start)
        actions.addWidget(self.joint_preset); actions.addWidget(self.joint_run); jointbox.addLayout(actions)
        self.joint_status=QLabel('计算后选择结果行查看逐工况复核和实际最小距离。'); self.joint_status.setWordWrap(True); jointbox.addWidget(self.joint_status)
        joint_views=QTabWidget(); jointbox.addWidget(joint_views,1)
        self.case_table=QTableWidget(0,11)
        self.case_table.setHorizontalHeaderLabels(['工况','GHz','θ / °','φ / °','总场 PSL/dB','HPBW u','HPBW v','最小距/λf','单元扫描/dB','AF PSL/dB','总峰/dB'])
        self.case_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers); self.case_table.verticalHeader().hide()
        self.case_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        joint_views.addTab(self.case_table,'逐工况复核')
        self.joint_fig=Figure(figsize=(12,4.3),dpi=100); self.joint_canvas=FigureCanvasQTAgg(self.joint_fig)
        joint_views.addTab(self.joint_canvas,'工况副瓣图')
        self.distance_table=QTableWidget(0,4); self.distance_table.setHorizontalHeaderLabels(['阵元 A 编号','阵元 B 编号','距离 / λ0','距离 / mm'])
        self.distance_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers); self.distance_table.verticalHeader().hide()
        self.distance_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        joint_views.addTab(self.distance_table,'最近 20 对阵元')
        self.joint_tab=self.tabs.addTab(self.joint_panel,'联合综合与间距')
        for control in (ref,minimum): control.valueChanged.connect(self.mark_stale)
        self.joint_scans.textChanged.connect(self.mark_stale); self.joint_frequencies.textChanged.connect(self.mark_stale)
        self.element_panel=QWidget(); elementbox=QVBoxLayout(self.element_panel)
        heading=QLabel('单元为共同朝向的标量幅度包络。默认理想点源；总场 = 单元包络 × 阵因子。预置贴片是 cos^q 前向近似，导入不推断极化或绝对增益。')
        heading.setWordWrap(True); elementbox.addWidget(heading)
        element_controls=QHBoxLayout(); leftform=QFormLayout(); rightform=QFormLayout()
        self.element_model=QComboBox()
        for model,label in MODELS.items(): self.element_model.addItem(label,model)
        self.element_model.setCurrentIndex(self.element_model.findData(config.element_model))
        self.element_axis=QComboBox()
        for axis in ('x','y','z'): self.element_axis.addItem(axis.upper()+' 轴',axis)
        self.element_axis.setCurrentIndex(self.element_axis.findData(config.element_dipole_axis))
        length=QDoubleSpinBox(); length.setRange(.05,1); length.setDecimals(3); length.setSingleStep(.05); length.setValue(config.element_dipole_length)
        hpbw=QDoubleSpinBox(); hpbw.setRange(30,150); hpbw.setValue(config.element_patch_hpbw); hpbw.setSuffix(' °')
        self.inputs['element_dipole_length']=length; self.inputs['element_patch_hpbw']=hpbw
        self.element_import_mode=QComboBox(); self.element_import_mode.addItem('完整 θ×φ 角网格','grid'); self.element_import_mode.addItem('绕 +z 轴对称（切面假设）','axisymmetric')
        self.element_import_mode.setCurrentIndex(self.element_import_mode.findData(config.element_import_mode))
        self.element_file=QLineEdit(config.element_file); self.element_file.setReadOnly(True)
        self.element_file.setPlaceholderText('选择 UTF-8 CSV，支持多频率')
        self.element_import_button=QPushButton('导入 CSV…'); self.element_import_button.clicked.connect(self.import_element)
        leftform.addRow('默认/导入单元',self.element_model); leftform.addRow('偶极子轴向',self.element_axis); leftform.addRow('偶极子总长度 / λ0',length)
        rightform.addRow('贴片近似单元 HPBW',hpbw); rightform.addRow('导入角域假设',self.element_import_mode)
        rightform.addRow(self.element_import_button,self.element_file)
        element_controls.addLayout(leftform,1); element_controls.addLayout(rightform,2); elementbox.addLayout(element_controls)
        element_actions=QHBoxLayout()
        self.element_example_button=QPushButton('载入多频示例 CSV'); self.element_example_button.clicked.connect(self.load_example_element)
        self.element_template_button=QPushButton('导出 CSV 模板…'); self.element_template_button.clicked.connect(self.save_element_template)
        self.element_preview_button=QPushButton('预览待计算单元'); self.element_preview_button.clicked.connect(self.preview_element)
        self.element_frequency=QComboBox(); self.element_frequency.currentIndexChanged.connect(self.preview_element)
        for button in (self.element_example_button,self.element_template_button,self.element_preview_button): element_actions.addWidget(button)
        element_actions.addWidget(QLabel('预览频率')); element_actions.addWidget(self.element_frequency)
        elementbox.addLayout(element_actions)
        self.element_pending_info=QLabel('点击预览查看待计算的单元；修改单元后需要重新计算布局。'); self.element_pending_info.setWordWrap(True); elementbox.addWidget(self.element_pending_info)
        self.element_fig=Figure(figsize=(12,4.3),dpi=100); self.element_canvas=FigureCanvasQTAgg(self.element_fig)
        elementbox.addWidget(self.element_canvas,1)
        self.element_result_info=QLabel('尚未形成已计算单元快照。'); self.element_result_info.setWordWrap(True); elementbox.addWidget(self.element_result_info)
        self.element_tab=self.tabs.addTab(self.element_panel,'单元方向图')
        for control in (self.element_model,self.element_axis,self.element_import_mode): control.currentIndexChanged.connect(self.element_changed)
        for control in (length,hpbw): control.valueChanged.connect(self.element_changed)
        self.element_file.textChanged.connect(self.mark_stale)
        self.weight_panel=QWidget(); weightbox=QVBoxLayout(self.weight_panel)
        self.weight_target=QLabel('先计算布局，再从上方表格选中要加权的方案。')
        self.weight_target.setWordWrap(True); weightbox.addWidget(self.weight_target)
        controlrow=QHBoxLayout(); form1=QFormLayout(); form2=QFormLayout()
        self.weight_method=QComboBox()
        for method,label in METHODS.items(): self.weight_method.addItem(label,method)
        self.weight_dr=QDoubleSpinBox(); self.weight_dr.setRange(0,40); self.weight_dr.setValue(20); self.weight_dr.setSuffix(' dB')
        self.weight_sll=QDoubleSpinBox(); self.weight_sll.setRange(20,80); self.weight_sll.setValue(30); self.weight_sll.setSuffix(' dB')
        self.weight_nbar=QSpinBox(); self.weight_nbar.setRange(2,10); self.weight_nbar.setValue(4)
        self.weight_sigma=QDoubleSpinBox(); self.weight_sigma.setRange(.05,1); self.weight_sigma.setDecimals(2); self.weight_sigma.setSingleStep(.05); self.weight_sigma.setValue(.35)
        form1.addRow('加权方式',self.weight_method); form1.addRow('加权幅度范围',self.weight_dr)
        form2.addRow('窗参考抑制参数',self.weight_sll); form2.addRow('Taylor nbar',self.weight_nbar)
        form2.addRow('Gaussian σ/跨度',self.weight_sigma)
        controlrow.addLayout(form1,1); controlrow.addLayout(form2,1)
        action=QVBoxLayout(); self.weight_button=QPushButton('对选中布局加权'); self.weight_button.setObjectName('primary')
        self.weight_button.clicked.connect(self.apply_weighting); action.addWidget(self.weight_button)
        explanation=QLabel('保持全部位置和阵元数，替换幅度。\n原结果保留；使用已计算参数快照。\n窗参考参数不保证稀疏阵实际副瓣。')
        explanation.setWordWrap(True); explanation.setStyleSheet('color:#617287;font-size:12px')
        action.addWidget(explanation); controlrow.addLayout(action,2); weightbox.addLayout(controlrow)
        self.weight_summary=QLabel('加权后显示副瓣、波束宽度及幅度效率的变化。'); self.weight_summary.setWordWrap(True)
        weightbox.addWidget(self.weight_summary)
        self.weight_fig=Figure(figsize=(12,3.4),dpi=100); self.weight_canvas=FigureCanvasQTAgg(self.weight_fig)
        self.weight_views=QTabWidget()
        self.weight_views.addTab(self.weight_canvas,'加权前后方向图')
        weightbox.addWidget(self.weight_views,1)
        self.weight_table=QTableWidget(0,6)
        self.weight_table.setHorizontalHeaderLabels(['候选编号','x / λ','y / λ','原幅度(和=1)','新幅度(和=1)','新相对幅度 / dB'])
        self.weight_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.weight_table.verticalHeader().hide()
        self.weight_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.weight_views.addTab(self.weight_table,'逐阵元权重')
        self.weight_tab=self.tabs.addTab(self.weight_panel,'稀疏后加权')
        self.weight_panel.setEnabled(False)
        self.weight_method.currentIndexChanged.connect(self.update_weight_controls)
        self.structure_canvas=FigureCanvasQTAgg(structure_figure()); self.tabs.addTab(self.structure_canvas,'结构化共阵')
        manual=QTextBrowser(); manual.setOpenExternalLinks(True)
        manual_path=resource_path('MANUAL.md')
        manual.setMarkdown(manual_path.read_text(encoding='utf-8') if manual_path.exists() else '请查看 README.md 的操作方法。')
        self.manual_tab=self.tabs.addTab(manual,'操作指南')
        QShortcut(QKeySequence('F1'),self).activated.connect(lambda:self.tabs.setCurrentIndex(self.manual_tab))
        research=QTextBrowser(); research.setOpenExternalLinks(True)
        research_path=resource_path('RESEARCH.md')
        research.setMarkdown(research_path.read_text(encoding='utf-8') if research_path.exists() else '请查看随附调研报告。')
        self.tabs.addTab(research,'方法与边界')
        split.addWidget(right); split.setStretchFactor(1,1)
        self.detail_note=QLabel('先计算，再从表格选择方案查看布局、方向图和共阵。')
        self.detail_note.setWordWrap(True); self.detail_note.setStyleSheet('color:#617287;font-size:12px')
        rightbox.addWidget(self.detail_note)
        self.arrange_workflow(split,scroll,layout,rightbox,jointbox,joint_views,elementbox)
        self.statusBar().showMessage('就绪 · ArrayLab 二维稀疏模块')
        self.hole_shape.currentIndexChanged.connect(self.mark_stale)
        self._initializing=False
        self.update_hole_fields()
        self.update_joint_preview()
        self.update_element_controls()
        self.update_workflow()
        if autorun: QTimer.singleShot(150,self.start)

    def arrange_workflow(self,split,old_sidebar,layout,rightbox,jointbox,joint_views,elementbox):
        """Place all draft inputs together, followed by one compute action and results."""
        def detached(widget):
            parent=widget.parentWidget()
            if parent:
                for existing in parent.findChildren(QLayout): existing.removeWidget(widget)
            widget.setParent(None); return widget
        def clear(container):
            while container.count():
                item=container.takeAt(0)
                if item.widget(): item.widget().deleteLater()
                elif item.layout(): clear(item.layout()); item.layout().deleteLater()
        def text(value):
            label=QLabel(value); label.setWordWrap(True); return label
        side=QFrame(); side.setObjectName('panel'); side.setMinimumWidth(320); side.setMaximumWidth(375)
        self._sidebar=side
        box=QVBoxLayout(side); box.setContentsMargins(14,14,14,14); box.setSpacing(10)
        box.addWidget(text('① 设置本次输入'))
        presets=QHBoxLayout(); self.default_preset=QPushButton('默认设置')
        self.default_preset.clicked.connect(lambda:self.set_config(Config()))
        self.hollow_preset.setText('掏洞示例'); self.joint_preset.setText('角频示例')
        for button in (self.default_preset,self.hollow_preset,self.joint_preset): presets.addWidget(detached(button))
        box.addLayout(presets)
        self.settings_tabs=QTabWidget(); box.addWidget(self.settings_tabs,1)
        def page(name):
            content=QWidget(); body=QVBoxLayout(content); body.setContentsMargins(10,12,10,12); body.setSpacing(10)
            form=QFormLayout(); form.setVerticalSpacing(10)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            body.addLayout(form)
            scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.Shape.NoFrame); scroll.setWidget(content)
            self.settings_tabs.addTab(scroll,name); return body,form
        def row(form,label,widget): form.addRow(label,detached(widget))
        geometry,form=page('阵列')
        for key,label in [('nx','X 向候选数'),('ny','Y 向候选数'),('k','目标阵元 K'),
                          ('spacing','候选间距 / λ0'),('reference_ghz','参考频率 / GHz')]: row(form,label,self.inputs[key])
        row(form,'中心禁布区',self.hole_shape)
        for key,label in [('hole_x','孔宽 X / 直径 / λ0'),('hole_y','矩形孔高 Y / λ0'),
                          ('min_spacing','最小中心距 / λ0')]: row(form,label,self.inputs[key])
        geometry.addWidget(detached(self.availability)); geometry.addWidget(text('λ0 由参考频率确定。圆孔只使用直径；最小中心距为 0 时不追加限制。')); geometry.addStretch()
        unit,form=page('单元')
        for label,widget in [('单元形式',self.element_model),('偶极子轴向',self.element_axis),
            ('偶极子长度 / λ0',self.inputs['element_dipole_length']),('贴片近似 HPBW',self.inputs['element_patch_hpbw']),
            ('CSV 角域假设',self.element_import_mode),('CSV 文件',self.element_file)]: row(form,label,widget)
        unit.addWidget(detached(self.element_import_button))
        for button in (self.element_example_button,self.element_template_button,self.element_preview_button): unit.addWidget(detached(button))
        unit.addWidget(text('默认理想点源。预览只检查待计算单元；应用到阵列需要点击下方“开始计算”。')); unit.addStretch()
        scans,form=page('工况')
        for key,label in [('theta','单一扫描 θ / °'),('phi','单一方位 φ / °')]: row(form,label,self.inputs[key])
        row(form,'扫描角列表',self.joint_scans); row(form,'频率列表 / GHz',self.joint_frequencies)
        bits=QSpinBox(); bits.setRange(0,12); bits.setSpecialValueText('理想（不量化）')
        self.inputs['phase_bits']=bits
        bits.setValue(self._initial_phase_bits)
        bits.valueChanged.connect(self.mark_stale); form.addRow('移相器位数 / bit',bits)
        scans.addWidget(text('扫描角格式：0,0;30,0;30,90\n频率格式：9,10,11\n列表留空使用单一值；非空时替代单一值，并取扫描角 × 频率的全部组合。'))
        scans.addWidget(detached(self.joint_preview))
        scans.addWidget(text('0 为理想相位；1–12 位按最近码量化。先理想综合，再固定布局/幅度复核量化影响；多频分别编程。'))
        self.phase_result_button=QPushButton('查看已计算量化结果')
        self.phase_result_button.clicked.connect(lambda:self.tabs.setCurrentIndex(self.phase_tab)); scans.addWidget(self.phase_result_button); scans.addStretch()
        advanced,form=page('高级')
        for key,label in [('seed','随机种子'),('iterations','退火步数'),('opt_grid','优化网格'),
                          ('eval_grid','复核网格'),('amplitude_dr_db','综合调幅范围 / dB')]: row(form,label,self.inputs[key])
        advanced.addWidget(text('首次使用保持默认即可。综合调幅范围约束计算阶段的固定位置 LP；后续加权使用“加权”页的独立设置。')); advanced.addStretch()
        self.draft_summary=text(''); box.addWidget(self.draft_summary)
        self.restore_button=QPushButton('恢复已计算设置'); self.restore_button.setEnabled(False)
        self.restore_button.clicked.connect(lambda:self.set_config(Config(**self.bundle['config'])) if self.bundle else None)
        self.restore_button.setToolTip('只恢复输入字段，不重新计算，也不重读单元源文件。')
        box.addWidget(self.restore_button)
        self.run_button.setText('② 开始计算'); box.addWidget(detached(self.run_button))
        # Keep the legacy attribute as an alias; the interface has one compute button.
        self.joint_run=self.run_button
        # Remove draft controls from result tabs after their widgets have been moved.
        detached(joint_views); detached(self.joint_status); clear(jointbox)
        jointbox.addWidget(text('已计算方案的工况与距离。修改角频或约束请使用左侧设置，再重新计算。'))
        jointbox.addWidget(self.joint_status); jointbox.addWidget(joint_views,1)
        keep=(self.element_pending_info,self.element_frequency,self.element_canvas,self.element_result_info)
        for widget in keep: detached(widget)
        clear(elementbox)
        elementbox.addWidget(text('待计算单元预览：不会改变已有方案。预览后点击左侧“开始计算”应用。'))
        frequency=QHBoxLayout(); frequency.addWidget(QLabel('预览频率')); frequency.addWidget(self.element_frequency,1)
        elementbox.addLayout(frequency); elementbox.addWidget(self.element_pending_info)
        elementbox.addWidget(self.element_canvas,1); elementbox.addWidget(self.element_result_info)
        self.tabs.setTabText(self.joint_tab,'工况与间距'); self.tabs.setTabText(self.element_tab,'单元预览')
        self.tabs.setTabText(self.weight_tab,'④ 稀疏后加权'); self.tabs.setTabText(0,'③ 方案详情')
        # Put the optional unit preview after result and weighting pages.
        self.tabs.tabBar().moveTab(self.element_tab,self.weight_tab)
        self.joint_tab=self.tabs.indexOf(self.joint_panel); self.element_tab=self.tabs.indexOf(self.element_panel)
        self.weight_tab=self.tabs.indexOf(self.weight_panel); self.manual_tab=self.tabs.indexOf(self.tabs.widget(self.manual_tab))
        actions=QHBoxLayout(); actions.addWidget(QLabel('③ 选择下方方案 →'))
        self.to_weight_button=QPushButton('④ 对所选方案加权'); self.to_weight_button.setEnabled(False)
        self.to_weight_button.clicked.connect(lambda:self.tabs.setCurrentIndex(self.weight_tab))
        actions.addWidget(self.to_weight_button); actions.addStretch(); actions.addWidget(detached(self.export_button))
        rightbox.insertLayout(1,actions)
        hint=text('方案列表：选点方案以 K 个阵元比较；参考与 ℓ1 加权方案的阵元数可能不同。')
        hint.setStyleSheet('font-size:12px;color:#617287'); rightbox.insertWidget(rightbox.indexOf(self.table),hint)
        flow=text('① 设置阵列 / 单元 / 工况   →   ② 开始计算   →   ③ 选择方案   →   ④ 加权（可选）   →   ⑤ 导出')
        flow.setStyleSheet('font-size:13px;color:#52647a'); layout.insertWidget(2,flow)
        self.workflow_status=text(''); layout.insertWidget(3,self.workflow_status)
        split.replaceWidget(0,side); old_sidebar.deleteLater()
        self.resize(1460,1000); self.setMinimumSize(1160,760)
        # The list/detail split is draggable so plots can use more of the window.
        position=rightbox.indexOf(self.table)
        rightbox.removeWidget(self.table); rightbox.removeWidget(self.tabs)
        self.result_splitter=QSplitter(Qt.Orientation.Vertical)
        self.result_splitter.setChildrenCollapsible(False); self.result_splitter.setHandleWidth(7)
        self.table.setMinimumHeight(90); self.table.setMaximumHeight(16777215)
        self.result_splitter.addWidget(self.table); self.result_splitter.addWidget(self.tabs)
        self.result_splitter.setStretchFactor(0,0); self.result_splitter.setStretchFactor(1,1)
        rightbox.insertWidget(position,self.result_splitter,1)
        self.result_splitter.setSizes([100,560])
        self.create_phase_panel()
        self.settings_tabs.currentChanged.connect(self.settings_changed)

    def create_phase_panel(self):
        manual=self.tabs.widget(self.manual_tab)
        self.phase_panel=QWidget(); box=QVBoxLayout(self.phase_panel)
        note=QLabel('固定位置、固定幅度的硬件复核。理想选点和 LP 加权不含量化约束；曲线共用理想扫描点参考，保留实际损失。')
        note.setWordWrap(True); box.addWidget(note)
        self.phase_info=QLabel('尚无已计算结果。请在左侧“工况”设置移相器位数后计算。'); self.phase_info.setWordWrap(True); box.addWidget(self.phase_info)
        actions=QHBoxLayout()
        self.phase_worst_button=QPushButton('显示量化最差工况'); self.phase_worst_button.setEnabled(False)
        self.phase_worst_button.clicked.connect(self.select_phase_worst)
        self.phase_max_button=QPushButton('放大量化切线'); self.phase_max_button.setEnabled(False)
        self.phase_max_button.clicked.connect(self.open_phase_cuts)
        actions.addWidget(self.phase_worst_button); actions.addWidget(self.phase_max_button); actions.addStretch(); box.addLayout(actions)
        pages=QTabWidget(); box.addWidget(pages,1)
        self.phase_fig=Figure(figsize=(12,4.5),dpi=100); self.phase_canvas=FigureCanvasQTAgg(self.phase_fig)
        pages.addTab(self.phase_canvas,'理想 / 量化对比')
        self.phase_cases=QTableWidget(0,10)
        self.phase_cases.setHorizontalHeaderLabels(['工况','GHz','θ / °','φ / °','理想 PSL/dB','量化 PSL/dB','变化/dB','扫描损失/dB','主瓣区峰偏差/°','RMS误差/°'])
        self.phase_cases.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers); self.phase_cases.verticalHeader().hide()
        self.phase_cases.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        pages.addTab(self.phase_cases,'逐工况量化复核')
        self.phase_commands=QTableWidget(0,7)
        self.phase_commands.setHorizontalHeaderLabels(['候选编号','x / λ0','y / λ0','理想相位/°','实际相位/°','环形误差/°','码字'])
        self.phase_commands.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers); self.phase_commands.verticalHeader().hide()
        self.phase_commands.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        pages.addTab(self.phase_commands,'逐阵元移相指令')
        index=self.tabs.addTab(self.phase_panel,'移相器量化'); self.tabs.tabBar().moveTab(index,3)
        self.phase_tab=self.tabs.indexOf(self.phase_panel); self.weight_tab=self.tabs.indexOf(self.weight_panel)
        self.element_tab=self.tabs.indexOf(self.element_panel); self.manual_tab=self.tabs.indexOf(manual)

    def select_phase_worst(self):
        if not self.bundle or self.table.currentRow()<0: return
        info=self.bundle['results'][self.table.currentRow()].get('phase_quantization',{})
        if info.get('enabled'): self.case_selector.setCurrentIndex(info['metrics']['worst_case_index'])

    def open_phase_cuts(self):
        if not self.bundle or not self.bundle['config'].get('phase_bits'): return
        self.phase_view.setCurrentIndex(self.phase_view.findData('quantized')); self.tabs.setCurrentIndex(0)
        self.detail_view.open_plot('cuts')

    def update_phase_result(self,result,case_index):
        info=result.get('phase_quantization',{})
        self.phase_cases.setRowCount(0); self.phase_commands.setRowCount(0)
        enabled=info.get('enabled',False)
        self.phase_worst_button.setEnabled(enabled and not self._busy); self.phase_max_button.setEnabled(enabled and not self._busy)
        if not enabled:
            self.phase_info.setText('当前已计算快照未启用量化。左侧“工况”设为 1–12 位并重新计算；0 表示理想相位。')
            draw_phase_comparison(self.phase_fig,self.bundle,result,case_index); self.phase_canvas.draw_idle(); return
        row=info['scenarios'][case_index]; metrics=info['metrics']
        number=lambda value:'未定义' if value is None else f'{value:.3f}'
        warning='；扫描点响应过低，该工况的扫描点归一化 PSL 未定义' if not row['steering_response_defined'] else ''
        self.phase_info.setText(f"{info['bits']} bit · 步进 {info['step_deg']:g}° · 当前工况扫描损失 {row['steering_loss_db']:.3f} dB · 理想/量化 PSL {row['ideal_validation_psl_db']:.3f} / {number(row['validation_psl_db'])} dB · 最大/RMS 相位误差 {row['max_phase_error_deg']:.3f}/{row['rms_phase_error_deg']:.3f}°\n主瓣区采样峰偏差 {row['pointing_error_deg']:.3f}°（相对理想变化 {row['beam_shift_from_ideal_deg']:.3f}°）；全局峰偏差 {row['global_pointing_error_deg']:.3f}°；跨工况最大扫描损失 {metrics['max_steering_loss_db']:.3f} dB。指向为采样估计。"+warning)
        self.phase_cases.setRowCount(len(info['scenarios']))
        for i,case in enumerate(info['scenarios']):
            values=[str(i+1),f"{case['frequency_ghz']:g}",f"{case['theta']:g}",f"{case['phi']:g}",
                number(case['ideal_validation_psl_db']),number(case['validation_psl_db']),number(case['delta_psl_db']),
                number(case['steering_loss_db']),number(case['pointing_error_deg']),number(case['rms_phase_error_deg'])]
            for j,value in enumerate(values): self.phase_cases.setItem(i,j,QTableWidgetItem(value))
        phases=excitation_phases(result['xy'],row,info['bits']); self.phase_commands.setRowCount(len(result['xy']))
        for i,(index,xy,ideal,actual,error,code) in enumerate(zip(result['indices'],result['xy'],phases['ideal_phase_deg'],phases['command_phase_deg'],phases['error_deg'],phases['phase_code'])):
            values=[str(int(index)),f'{xy[0]:g}',f'{xy[1]:g}',f'{ideal:.6f}',f'{actual:.6f}',f'{error:+.6f}',str(int(code))]
            for j,value in enumerate(values): self.phase_commands.setItem(i,j,QTableWidgetItem(value))
        draw_phase_comparison(self.phase_fig,self.bundle,result,case_index); self.phase_canvas.draw_idle()

    def settings_changed(self,index):
        if index==1: self.tabs.setCurrentIndex(self.element_tab)

    def set_config(self,config):
        if self._busy: return
        self._initializing=True
        try:
            for key,box in self.inputs.items(): box.setValue(getattr(config,key))
            for widget,value in ((self.hole_shape,config.hole_shape),(self.element_model,config.element_model),
                (self.element_axis,config.element_dipole_axis),(self.element_import_mode,config.element_import_mode)):
                widget.setCurrentIndex(widget.findData(value))
            self.joint_scans.setText(config.scan_angles); self.joint_frequencies.setText(config.frequencies_ghz)
            self.element_file.setText(config.element_file)
        finally: self._initializing=False
        self.mark_stale(); self.settings_tabs.setCurrentIndex(0)

    def update_workflow(self):
        if self._initializing or not hasattr(self,'workflow_status'): return
        config=self.config_from_inputs(); stale=self.bundle is not None and config!=Config(**self.bundle['config'])
        try:
            config.validate(); count=len(scenarios(config))
            draft=f"本次输入：{config.nx}×{config.ny}，K={config.k}；{count} 工况；{MODELS[config.element_model]}"
            if config.phase_bits: draft+=f'；{config.phase_bits} 位移相复核'
            invalid=None
        except ValueError as exc: draft='本次输入需修正：'+str(exc); invalid=str(exc)
        self.draft_summary.setText(draft)
        self.restore_button.setEnabled(not self._busy and self.bundle is not None and stale)
        selected=self.bundle is not None and self.table.currentRow()>=0
        self.to_weight_button.setEnabled(not self._busy and selected)
        self.export_button.setText('⑤ 导出已计算结果' + ('（旧参数）' if stale else ''))
        self.export_button.setEnabled(not self._busy and self.bundle is not None)
        self.phase_result_button.setEnabled(not self._busy and self.bundle is not None)
        if self._busy:
            message={'weight':'正在对所选已计算布局加权，使用原工况与单元快照。',
                     'export':'正在导出已计算结果，请稍候。'}.get(self._job_kind,'正在计算当前输入；完成前保留上次结果。')
            color='#e1eef8'
        elif self._last_error:
            message='上次操作失败，已保留已有结果。'+self._last_error; color='#fbe8e6'
        elif self.bundle is None:
            message='下一步：完成左侧设置，点击“② 开始计算”。' if not invalid else '下一步：修正左侧输入，再开始计算。'
            color='#e1eef8' if not invalid else '#fff1d6'
        elif stale:
            message='输入已修改，尚未重算。当前图表、加权和导出仍使用已计算快照；请重新计算，或恢复已计算设置。'
            color='#fff1d6'
        else:
            message='输入与已计算参数一致。下一步：选择方案查看，按需加权，或导出结果。'; color='#e1f2ed'
        self.workflow_status.setText(message)
        self.workflow_status.setStyleSheet(f'background:{color};color:#243349;padding:9px 12px;border-radius:7px')
        if not self._busy: self.run_button.setText('② 重新计算' if self.bundle is not None else '② 开始计算')

    def mark_stale(self):
        if self._initializing: return
        self._last_error=None
        if hasattr(self,'availability'):
            self.update_hole_fields()
        if hasattr(self,'joint_preview'): self.update_joint_preview()
        if self.bundle:
            stale=self.config_from_inputs()!=Config(**self.bundle['config'])
            self.statusBar().showMessage('参数已修改，尚未重算；图表和导出仍对应已计算快照。' if stale else '输入已与已计算参数一致，可继续查看、加权或导出。')
        self.update_workflow()

    def config_from_inputs(self):
        return Config(**{key:box.value() for key,box in self.inputs.items()},hole_shape=self.hole_shape.currentData(),
            scan_angles=self.joint_scans.text() if hasattr(self,'joint_scans') else '',
            frequencies_ghz=self.joint_frequencies.text() if hasattr(self,'joint_frequencies') else '',
            element_model=self.element_model.currentData() if hasattr(self,'element_model') else 'isotropic',
            element_file=self.element_file.text() if hasattr(self,'element_file') else '',
            element_import_mode=self.element_import_mode.currentData() if hasattr(self,'element_import_mode') else 'grid',
            element_dipole_axis=self.element_axis.currentData() if hasattr(self,'element_axis') else 'x')

    def element_changed(self):
        self.update_element_controls(); self.mark_stale()

    def update_element_controls(self):
        if not hasattr(self,'element_model'): return
        model=self.element_model.currentData(); enabled=not self._busy
        self.element_axis.setEnabled(enabled and model=='dipole')
        self.inputs['element_dipole_length'].setEnabled(enabled and model=='dipole')
        self.inputs['element_patch_hpbw'].setEnabled(enabled and model=='patch')
        self.element_file.setEnabled(enabled and model=='imported')
        self.element_import_mode.setEnabled(enabled and model=='imported')

    def import_element(self):
        path,_=QFileDialog.getOpenFileName(self,'导入单元方向图',str(document_directory()),'方向图 CSV (*.csv)')
        if path: self.load_element_file(path)

    def load_element_file(self,path):
        try:
            config=replace(self.config_from_inputs(),element_model='imported',element_file=str(Path(path).resolve()))
            build_element(config)
        except (OSError,ValueError) as exc:
            QMessageBox.warning(self,'方向图导入失败',str(exc)); return False
        self.element_file.setText(config.element_file)
        self.element_model.setCurrentIndex(self.element_model.findData('imported'))
        self.preview_element(); return True

    def load_example_element(self):
        self.element_import_mode.setCurrentIndex(self.element_import_mode.findData('grid'))
        self.load_element_file(resource_path('examples/element_grid_multifrequency.csv'))

    def save_element_template(self):
        path,_=QFileDialog.getSaveFileName(self,'导出新的单元方向图模板',str(document_directory()/'element_template.csv'),'CSV (*.csv)')
        if not path: return
        try:
            write_example_csv(path,self.element_import_mode.currentData()=='axisymmetric')
            self.statusBar().showMessage('已导出示例模板：'+path)
        except (OSError,ValueError) as exc: QMessageBox.warning(self,'模板导出失败',str(exc))

    def preview_element(self):
        try:
            c=self.config_from_inputs(); element=build_element(c); cases=scenarios(c)
            frequencies=list(dict.fromkeys(case['frequency_ghz'] for case in cases))
            previous=self.element_frequency.currentData()
            self.element_frequency.blockSignals(True); self.element_frequency.clear()
            for frequency in frequencies: self.element_frequency.addItem(f'{frequency:g} GHz',frequency)
            self.element_frequency.setCurrentIndex(frequencies.index(previous) if previous in frequencies else 0)
            self.element_frequency.blockSignals(False)
            frequency=self.element_frequency.currentData()
            draw_element(self.element_fig,c,element,frequency); self.element_canvas.draw_idle()
            snap=element.snapshot
            descriptions={'isotropic':'各方向、各频率的标量响应恒为 1。',
                'dipole':f"沿 {c.element_dipole_axis.upper()} 轴，总长 {c.element_dipole_length:g}λ0；固定物理长度、逐频峰值归一化，不含阻抗和效率。",
                'patch':f"朝 +z 的 cos^q 包络，单元半功率全宽 {c.element_patch_hpbw:g}°；是频率不变的解析近似。",
                'imported':'全部导入频率共用同一幅度参考；按线性幅度插值，禁止频率外推。'}
            info='待计算预览：'+snap['name']+'。 '+descriptions[snap['model']]
            if 'source' in snap:
                mode='完整二维角网格' if snap['parameters']['import_mode']=='grid' else '绕 +z 轴对称假设'
                info+=f"\n{snap['source']['filename']}；{snap['source']['rows']} 行；{mode}；SHA-256 {snap['source']['sha256'][:12]}…"
            self.element_pending_info.setText(info)
            self.tabs.setCurrentIndex(self.element_tab)
        except (OSError,ValueError) as exc: QMessageBox.warning(self,'单元预览失败',str(exc))

    def update_joint_preview(self):
        c=self.config_from_inputs()
        try:
            cases=scenarios(c)
            self.joint_preview.setText(f'当前 {len(cases)} 个离散工况；λ0={LIGHT_MM_GHZ/c.reference_ghz:.4f} mm；最小距要求 {c.min_spacing:g} λ0 = {c.min_spacing*LIGHT_MM_GHZ/c.reference_ghz:.4f} mm。0 表示不追加间距限制。')
        except ValueError as exc: self.joint_preview.setText('参数待修正：'+str(exc))

    def update_hole_fields(self):
        shape=self.hole_shape.currentData()
        busy=self._busy
        self.inputs['hole_x'].setEnabled(not busy and shape!='none')
        self.inputs['hole_y'].setEnabled(not busy and shape=='rectangle')
        joint=bool(self.joint_scans.text().strip() or self.joint_frequencies.text().strip()) if hasattr(self,'joint_scans') else False
        self.inputs['amplitude_dr_db'].setEnabled(not busy and (shape!='none' or joint))
        if hasattr(self,'joint_scans'):
            single=not self.joint_scans.text().strip()
            self.inputs['theta'].setEnabled(not busy and single); self.inputs['phi'].setEnabled(not busy and single)
        c=self.config_from_inputs()
        count=int(available_mask(c).sum())
        self.availability.setText(f'可用位置 {count} / {c.nx*c.ny}；中心禁布 {c.nx*c.ny-count} 个。')

    def load_hollow_preset(self):
        self.set_config(Config(hole_shape='rectangle',hole_x=2.,hole_y=2.))
        self.statusBar().showMessage('已载入中心矩形孔整套示例参数，请点击“开始计算”。')

    def load_joint_preset(self):
        self.set_config(Config(k=24,scan_angles='0,0;30,0;30,90',frequencies_ghz='9,10,11',min_spacing=.6))
        self.settings_tabs.setCurrentIndex(2)
        self.statusBar().showMessage('已载入 9 工况整套示例参数，请点击统一的“开始计算”。')

    def set_busy(self,busy):
        self._busy=busy
        self.settings_tabs.setEnabled(not busy)
        self.default_preset.setEnabled(not busy)
        self.run_button.setEnabled(not busy)
        for box in self.inputs.values(): box.setEnabled(not busy)
        self.hole_shape.setEnabled(not busy); self.hollow_preset.setEnabled(not busy)
        self.update_hole_fields()
        self.export_button.setEnabled(not busy and self.bundle is not None)
        self.table.setEnabled(not busy)
        self.case_selector.setEnabled(not busy and self.bundle is not None)
        self.pattern_view.setEnabled(not busy and self.bundle is not None)
        quantized=bool(self.bundle and self.bundle['config'].get('phase_bits'))
        self.phase_view.setEnabled(not busy and quantized)
        self.phase_panel.setEnabled(not busy)
        self.phase_worst_button.setEnabled(not busy and quantized)
        self.phase_max_button.setEnabled(not busy and quantized)
        self.joint_panel.setEnabled(not busy)
        self.element_panel.setEnabled(not busy)
        self.update_element_controls()
        self.weight_panel.setEnabled(not busy and self.bundle is not None)
        self.update_weight_controls()
        self.update_workflow()

    def start(self):
        if self.thread is not None or self._busy: return
        try: config=self.config_from_inputs().validate()
        except ValueError as exc:
            self._last_error=str(exc); self.update_workflow()
            QMessageBox.warning(self,'参数检查',str(exc)); return
        self.begin_worker(Worker(config),'synthesis')

    def begin_worker(self,worker,kind='synthesis'):
        self._last_error=None; self._job_kind=kind
        self.set_busy(True); self.statusBar().showMessage('正在计算…')
        self.thread=QThread(self); self.worker=worker; self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.calculate)
        self.worker.progress.connect(lambda text:self.statusBar().showMessage('正在计算 · '+text))
        self.worker.complete.connect(self.receive); self.worker.failed.connect(self.failure)
        self.worker.complete.connect(self.thread.quit); self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.cleanup); self.thread.start()

    def update_weight_controls(self):
        if not hasattr(self,'weight_method'): return
        method=self.weight_method.currentData()
        enabled=not self._busy and self.bundle is not None
        self.weight_sll.setEnabled(enabled and method in ('taylor','chebyshev'))
        self.weight_nbar.setEnabled(enabled and method=='taylor')
        self.weight_sigma.setEnabled(enabled and method=='gaussian')
        self.weight_dr.setEnabled(enabled and method!='uniform')
        self.weight_button.setEnabled(enabled and self.table.currentRow()>=0)
        if self.bundle:
            self.weight_nbar.setMaximum(min(10,self.bundle['config']['nx'],self.bundle['config']['ny']))

    def apply_weighting(self):
        if self.thread is not None or self._busy or not self.bundle: return
        index=self.table.currentRow()
        if index<0: return
        settings=Weighting(method=self.weight_method.currentData(),dr_db=self.weight_dr.value(),
            sll_db=self.weight_sll.value(),nbar=self.weight_nbar.value(),sigma=self.weight_sigma.value())
        try: settings.validate(Config(**self.bundle['config']))
        except ValueError as exc:
            QMessageBox.warning(self,'加权参数检查',str(exc)); return
        self.begin_worker(Worker(weight_payload=(self.bundle,index,settings)),'weight')

    def cleanup(self):
        old=self.thread; self.thread=None; old.deleteLater(); self._job_kind=None; self.set_busy(False)

    def failure(self,message):
        self._last_error=message; self.update_workflow()
        self.statusBar().showMessage('计算失败；保留上一份结果。')
        QMessageBox.critical(self,'计算失败',message)

    def receive(self,bundle):
        first_result=self.bundle is None
        self.bundle=bundle; self._last_error=None
        c=bundle['config']
        scan=c['scan_angles'].strip() or f"θ = {c['theta']:g}° / φ = {c['phi']:g}°"
        frequency=c['frequencies_ghz'].strip() or f"{c['reference_ghz']:g}"
        self.snapshot.setText(f"已计算参数   {c['nx']} × {c['ny']} 网格   ·   K = {c['k']}   ·   d = {c['spacing']:g} λ0   ·   扫描 {scan}   ·   频率 {frequency} GHz   ·   seed = {c['seed']}")
        if c['hole_shape']!='none':
            description=(f"圆孔直径 {c['hole_x']:g} λ" if c['hole_shape']=='circle' else f"矩形孔 {c['hole_x']:g} × {c['hole_y']:g} λ")
            self.snapshot.setText(self.snapshot.text()+f"   ·   {description}   ·   可用 {bundle['candidate_counts']['available']}")
        self.snapshot.setText(self.snapshot.text()+f"   ·   {len(bundle['scenarios'])} 工况   ·   f0={c['reference_ghz']:g} GHz   ·   dmin≥{c['min_spacing']:g} λ0")
        if c.get('phase_bits'): self.snapshot.setText(self.snapshot.text()+f"   ·   理想综合 / {c['phase_bits']} 位量化复核")
        self.phase_view.blockSignals(True)
        self.phase_view.setCurrentIndex(self.phase_view.findData('quantized' if c.get('phase_bits') else 'ideal'))
        self.phase_view.blockSignals(False); self.phase_view.setEnabled(not self._busy and bool(c.get('phase_bits')))
        self.table.horizontalHeaderItem(2).setText('理想最差 PSL/dB' if c.get('phase_bits') else '最差 PSL/dB')
        element_info='已计算快照：'+bundle['element_pattern']['name']
        if 'source' in bundle['element_pattern']:
            source_info=bundle['element_pattern']['source']
            element_info+=f"；{source_info['filename']}；SHA-256 {source_info['sha256'][:12]}…"
        self.element_result_info.setText(element_info+'；图表和指标使用此快照，修改单元后请重新计算。')
        self.case_selector.blockSignals(True); self.case_selector.clear()
        for case in bundle['scenarios']: self.case_selector.addItem(case['label'],case['index'])
        self.case_selector.blockSignals(False); self._last_selected_row=None
        self.table.blockSignals(True); self.table.setRowCount(len(bundle['results']))
        for row,r in enumerate(bundle['results']):
            m=r['metrics']; width=lambda x:'—' if x is None else f'{x:.4f}'
            values=[result_name(r),str(m['elements']),f"{m['validation_psl_db']:.2f}",f"{m['grid_gap_db']:+.2f}",f"{m['amplitude_dynamic_range_db']:.2f}",
                    width(m['hpbw_u']),width(m['hpbw_v']),str(m['unique_lags']),str(m['central_rectangle_lags']),width(m['min_distance_lambda_ref']),str(m['spacing_violations'])]
            for col,value in enumerate(values):
                item=QTableWidgetItem(value)
                if col: item.setTextAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
                item.setToolTip(r['notes']+'\n最差工况：'+m['worst_case_label']); self.table.setItem(row,col,item)
        self.table.blockSignals(False)
        if first_result:
            height=max(500,self.result_splitter.height())
            self.result_splitter.setSizes([170,max(330,height-170)])
        while self.compare_box.count():
            widget=self.compare_box.takeAt(0).widget()
            if widget: widget.deleteLater()
        self.compare_canvas=FigureCanvasQTAgg(comparison_figure(bundle)); self.compare_box.addWidget(self.compare_canvas)
        best=min((i for i,r in enumerate(bundle['results']) if r['metrics']['elements']==c['k'] and r['name']!='L1 weighted solution'),
                 key=lambda i:bundle['results'][i]['metrics']['validation_psl_db'])
        best=bundle.get('last_weighted_index',best)
        self.table.selectRow(best); self.select_result()
        if 'last_weighted_index' in bundle:
            self.tabs.setCurrentIndex(self.weight_tab)
            message='加权完成 · 保留原结果和相同阵元位置，已重新复核方向图。'
            if self.config_from_inputs()!=Config(**bundle['config']): message+=' 输入参数尚未重算，本次使用已计算快照。'
            self.statusBar().showMessage(message)
        else: self.statusBar().showMessage('计算完成 · 已用密网格、错位网格和边界采样复核 · 尚无连续角域保证')
        self.update_workflow()

    def select_result(self):
        if not self.bundle: return
        row=self.table.currentRow()
        if row<0: return
        result=self.bundle['results'][row]
        m=result['metrics']
        if row!=self._last_selected_row:
            self.case_selector.blockSignals(True); self.case_selector.setCurrentIndex(m['worst_case_index']); self.case_selector.blockSignals(False)
            self._last_selected_row=row
        case_index=max(0,self.case_selector.currentIndex())
        self.detail_view.set_context(self.bundle,result,case_index,self.pattern_view.currentData(),self.phase_view.currentData())
        selected_case=result['scenario_metrics'][case_index]
        self.detail_note.setText(f"{result_name(result)}  |  指标：已计算单元的总场  |  当前单元扫描响应 {selected_case['element_at_steering_db']:.2f} dB  |  总峰 {selected_case['total_peak_db']:.2f} dB（扫描点参考）  |  权重效率 {m['taper_efficiency']:.3f}  |  禁布违规 {m['forbidden_elements']}")
        if result.get('phase_quantization',{}).get('enabled'):
            q=result['phase_quantization']['scenarios'][case_index]
            self.detail_note.setText(self.detail_note.text()+f"  |  {self.phase_view.currentText()}；量化扫描损失 {q['steering_loss_db']:.3f} dB（曲线使用理想扫描点参考）")
        self.update_phase_result(result,case_index)
        self.weight_target.setText(f'加权来源：第 {row+1} 行 · '+result_name(result)+f"  ·  {m['elements']} 个阵元；使用已计算单元和 {len(self.bundle['scenarios'])} 工况，保持位置与数量。")
        draw_weight_comparison(self.weight_fig,self.bundle,result,case_index); self.weight_canvas.draw_idle()
        self.case_table.setRowCount(len(result['scenario_metrics']))
        for j,case in enumerate(result['scenario_metrics']):
            width=lambda x:'—' if x is None else f'{x:.4f}'
            values=[str(j+1),f"{case['frequency_ghz']:g}",f"{case['theta']:g}",f"{case['phi']:g}",f"{case['validation_psl_db']:.2f}",width(case['hpbw_u']),width(case['hpbw_v']),width(m['min_distance_lambda_ref']*case['frequency_ratio']),f"{case['element_at_steering_db']:.2f}",f"{case['array_factor_psl_db']:.2f}",f"{case['total_peak_db']:.2f}"]
            for col,value in enumerate(values): self.case_table.setItem(j,col,QTableWidgetItem(value))
        self.joint_status.setText(f"{result_name(result)}  |  最差工况：{m['worst_case_label']}  |  实际最小距 {m['min_distance_lambda_ref']:.4f} λ0 = {m['min_distance_mm']:.4f} mm  |  间距违规 {m['spacing_violations']}；下方列最近 20 对，全部阵元对均已检查。")
        draw_case_results(self.joint_fig,self.bundle,result); self.joint_canvas.draw_idle()
        distances=np.linalg.norm(result['xy'][:,None,:]-result['xy'][None,:,:],axis=-1)
        a,b=np.triu_indices(len(result['xy']),1); order=np.argsort(distances[a,b])[:20]
        self.distance_table.setRowCount(len(order))
        mm=LIGHT_MM_GHZ/self.bundle['config']['reference_ghz']
        for j,index in enumerate(order):
            values=[str(int(result['indices'][a[index]])),str(int(result['indices'][b[index]])),f'{distances[a[index],b[index]]:.6f}',f'{distances[a[index],b[index]]*mm:.6f}']
            for col,value in enumerate(values): self.distance_table.setItem(j,col,QTableWidgetItem(value))
        self.weight_table.setRowCount(0)
        if 'weighting' in result:
            source=self.bundle['results'][result['weighting']['source_index']]
            before=source['metrics']; delta=m['validation_psl_db']-before['validation_psl_db']
            now_case=result['scenario_metrics'][case_index]; old_case=source['scenario_metrics'][case_index]
            width=lambda value:'—' if value is None else f'{value:.4f}'
            self.weight_summary.setText(f"理想综合对比 · 相同 {m['elements']} 个位置  |  最差 PSL {before['validation_psl_db']:.2f} → {m['validation_psl_db']:.2f} dB（差 {delta:+.2f} dB）  |  当前工况 HPBW u {width(old_case['hpbw_u'])} → {width(now_case['hpbw_u'])}，v {width(old_case['hpbw_v'])} → {width(now_case['hpbw_v'])}  |  权重效率 {before['taper_efficiency']:.3f} → {m['taper_efficiency']:.3f}")
            self.weight_table.setRowCount(len(result['indices']))
            for row,(index,xy,old,new) in enumerate(zip(result['indices'],result['xy'],source['weights'],result['weights'])):
                values=[str(int(index)),f'{xy[0]:g}',f'{xy[1]:g}',f'{old:.8g}',f'{new:.8g}',f'{20*np.log10(new/result["weights"].max()):.3f}']
                for col,value in enumerate(values): self.weight_table.setItem(row,col,QTableWidgetItem(value))
        else: self.weight_summary.setText('选择加权方式并点击“对选中布局加权”。原结果保留，添加新的结果行。')
        self.update_weight_controls()
        self.update_workflow()

    def export(self):
        if not self.bundle: return
        parent=QFileDialog.getExistingDirectory(self,'选择结果的父目录',str(document_directory()))
        if not parent: return
        from .demo import export_bundle
        target=Path(parent)/datetime.now().strftime('sparse2d_%Y%m%d_%H%M%S_%f')
        self._job_kind='export'; self.set_busy(True)
        self.statusBar().showMessage('正在导出已计算快照…'); QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.last_export=export_bundle(self.bundle,target)
            self.statusBar().showMessage('已导出：'+str(target))
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target/'report.html')))
        except Exception as exc: QMessageBox.critical(self,'导出失败',str(exc))
        finally:
            QApplication.restoreOverrideCursor(); self._job_kind=None; self.set_busy(False)

    def closeEvent(self,event):
        if self.thread is not None or self._busy:
            self.statusBar().showMessage('正在计算或导出，请在完成后关闭窗口。'); event.ignore()
        else:
            self.detail_view.close_dialogs(); event.accept()


def configure_app(app):
    # Qt's offscreen platform may not discover Windows fallback fonts.
    import os
    if os.environ.get('QT_QPA_PLATFORM')=='offscreen':
        font=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/'msyh.ttc'
        if font.exists(): QFontDatabase.addApplicationFont(str(font))
    app.setStyle('Fusion')
    app.setFont(QFont('Microsoft YaHei UI',9))
    app.setStyleSheet('''QMainWindow{background:#f3f6fa} QLabel#title{font-size:25px;font-weight:600;color:#20344d} QFrame#panel{background:white;border:1px solid #dce4ee;border-radius:10px} QLabel#snapshot{background:#e1f2ed;color:#225c50;padding:12px;border-radius:8px} QPushButton{padding:9px;border:1px solid #cdd8e5;border-radius:6px;background:white} QPushButton#primary{background:#087e70;color:white;border:0;font-weight:600} QPushButton:disabled{color:#9aa6b3;background:#e8edf3} QSpinBox,QDoubleSpinBox{padding:4px;border:1px solid #cdd8e5;border-radius:4px;background:white} QTableWidget{border:1px solid #dce4ee;gridline-color:#edf1f5;background:white;selection-background-color:#d9eeea;selection-color:#20344d} QHeaderView::section{background:#edf2f7;padding:7px;border:0;color:#52647a} QTabWidget::pane{border:1px solid #dce4ee;background:white} QTabBar::tab{padding:8px 20px;background:#e7edf4} QTabBar::tab:selected{background:white;color:#087e70}''')
def launch(config=Config()):
    app=QApplication.instance() or QApplication(sys.argv)
    configure_app(app)
    window=Window(config); window.show()
    return app.exec()
