"""Planar electrical workbench for rectangular grids and explicit x-y coordinates."""
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QGridLayout,QLabel,QLineEdit,QComboBox,
                              QPushButton,QTabWidget,QTableWidget,QFileDialog,QMessageBox,QApplication)
from PySide6.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from .interactive_canvas import InteractiveFigureCanvas as FigureCanvasQTAgg

from .electrical_widgets import ElectricalControls
from .feed_widgets import FeedPanel
from .feed import FeedSettings
from .planar import (PlanarParameters,design_planar,read_coordinates_csv,validate_planar_design,
                     save_planar_configuration,load_planar_configuration,export_planar_directory,planar_amplitudes)
from .planar_plot import draw_planar_figure


class PlanarPanel(QWidget):
    fill_table = staticmethod(FeedPanel.fill_table)

    def __init__(self, window):
        super().__init__()
        self.window=window;self.report=None;self.coordinates=();self._changing=False;self.last_export=None
        self._branch_delay_settings = dict(branch_delays_deg=(), branch_delay_elements=0, branch_delay_topology='')
        layout=QVBoxLayout(self);layout.setContentsMargins(8,6,8,6)
        title=QLabel("独立平面阵参数 · 各向同性点源 · z=0 · 最多200阵元、最大孔径40λ")
        self.intro=title
        title.setWordWrap(True);layout.addWidget(title)
        self.settings_tabs=QTabWidget();self.settings_tabs.setMaximumHeight(178)
        geometry=QWidget();grid=QGridLayout(geometry);grid.setContentsMargins(6,3,6,3)
        self.geometry=QComboBox();self.geometry.addItem("规则矩形阵","rectangular");self.geometry.addItem("任意坐标CSV","arbitrary")
        self.unit=QComboBox();self.unit.addItem("λ","lambda");self.unit.addItem("mm","mm")
        self.unit.setToolTip("切换单位按新单位解释现有数值，不自动换算。")
        grid.addWidget(QLabel("几何"),0,0);grid.addWidget(self.geometry,0,1)
        grid.addWidget(QLabel("坐标单位"),0,2);grid.addWidget(self.unit,0,3)
        self.geometry_fields={}
        for index,(key,label,value) in enumerate([("nx","Nx","4"),("ny","Ny","4"),("spacing_x","x间距","0.5"),("spacing_y","y间距","0.5")]):
            edit=QLineEdit(value);edit.setMinimumWidth(45);self.geometry_fields[key]=edit
            row,col=1+index//2,2*(index%2);grid.addWidget(QLabel(label),row,col);grid.addWidget(edit,row,col+1)
            edit.textChanged.connect(self.mark_dirty)
        self.import_coordinates_button=QPushButton("导入x-y坐标");self.import_coordinates_button.clicked.connect(self.import_coordinates_dialog)
        grid.addWidget(self.import_coordinates_button,3,0,1,2)
        self.coordinate_label=QLabel("CSV：element,x,y；按1起始编号")
        grid.addWidget(self.coordinate_label,3,2,1,2);grid.setColumnStretch(1,1);grid.setColumnStretch(3,1)
        self.settings_tabs.addTab(geometry,"阵列几何")
        scan=QWidget();sg=QGridLayout(scan);sg.setContentsMargins(6,3,6,3)
        for index,(key,label,value) in enumerate([("frequency_ghz","频率(GHz)","3"),
                                                ("cut_phi_deg","切面φ(°)","0"),("step_deg","切面步长(°)","0.5")]):
            edit=QLineEdit(value);edit.setMinimumWidth(45);self.geometry_fields[key]=edit
            row,col=index//2,2*(index%2);sg.addWidget(QLabel(label),row,col);sg.addWidget(edit,row,col+1);edit.textChanged.connect(self.mark_dirty)
        sg.addWidget(QLabel("切面方位与波束目标可分别设置"),2,0,1,4);sg.setColumnStretch(1,1);sg.setColumnStretch(3,1)
        self.settings_tabs.addTab(scan,"频率与切面")
        network=QWidget();ng=QGridLayout(network);ng.setContentsMargins(6,3,6,3)
        self.topology=QComboBox()
        for text,value in [("并联树形","corporate"),("串行抽头","series"),("1分2＋对称串馈","symmetric_series")]:self.topology.addItem(text,value)
        ng.addWidget(QLabel("馈电拓扑"),0,0);ng.addWidget(self.topology,0,1)
        self.fields={}
        for index,(key,label,value) in enumerate([("input_ohm","输入Ω","50"),("junction_ohm","节点Ω","50"),("load_ohm","负载Ω","50"),
                                                ("min_line_ohm","提示下限Ω","20"),("max_line_ohm","提示上限Ω","120")],1):
            edit=QLineEdit(value);edit.setMinimumWidth(45);self.fields[key]=edit
            row,col=index//2,2*(index%2);ng.addWidget(QLabel(label),row,col);ng.addWidget(edit,row,col+1);edit.textChanged.connect(self.mark_dirty)
        ng.setColumnStretch(1,1);ng.setColumnStretch(3,1);self.settings_tabs.addTab(network,"馈电阻抗")
        self.electrical=ElectricalControls(self,self.settings_tabs)
        self.beam_row=QWidget();beam=QHBoxLayout(self.beam_row);beam.setContentsMargins(0,0,0,0)
        for key,label,value in [("scan_theta_deg","目标 θ (°)","30"),("scan_phi_deg","φ (°)","0")]:
            edit=QLineEdit(value);edit.setMaximumWidth(75);self.geometry_fields[key]=edit
            edit.setAccessibleName("平面阵目标波束"+label)
            edit.setToolTip("θ为0–90°，从+z轴量起；φ从+x朝+y，范围−180°至180°。")
            edit.textChanged.connect(self.mark_dirty);beam.addWidget(QLabel(label));beam.addWidget(edit)
        self.pointing_button=QPushButton("按目标生成");self.pointing_button.clicked.connect(lambda:self.generate(True))
        self.pointing_button.setToolTip("按实际阵元坐标计算目标相位，启用指向补偿并生成输出方向图。")
        beam.addWidget(self.pointing_button)
        self.beam_mode_hint=QLabel();self.beam_mode_hint.setWordWrap(True);beam.addWidget(self.beam_mode_hint,1)
        self.electrical.phase_mode.currentIndexChanged.connect(self.update_beam_hint)
        self.update_beam_hint();layout.addWidget(self.beam_row)
        layout.addWidget(self.settings_tabs)
        actions=QHBoxLayout()
        self.design_button=QPushButton("生成平面阵设计");self.design_button.clicked.connect(self.generate)
        self.open_button=QPushButton("打开设置");self.open_button.clicked.connect(self.open_dialog)
        self.save_button=QPushButton("保存设置");self.save_button.clicked.connect(self.save_dialog)
        self.export_button=QPushButton("导出结果");self.export_button.clicked.connect(self.export_dialog);self.export_button.setEnabled(False)
        self.toggle_settings=QPushButton("收起设置");self.toggle_settings.clicked.connect(self.toggle_controls)
        for button in [self.design_button,self.open_button,self.save_button,self.export_button,self.toggle_settings]:actions.addWidget(button)
        self.study_button=QPushButton("扫描评估");self.study_button.clicked.connect(self.open_study);actions.addWidget(self.study_button)
        actions.addStretch();layout.addLayout(actions)
        self.state=QLabel("请设置平面阵参数后生成设计。右侧直线阵参数不会覆盖本页。")
        self.state.setObjectName("state");self.state.setWordWrap(True);layout.addWidget(self.state)
        row=QHBoxLayout();row.addWidget(QLabel("结果视图"));self.view=QComboBox()
        for text,value in [("阵元坐标","geometry"),("二维切面比较","cut"),("上半球方向图","map"),("全空间三维方向图","3d"),("端口幅相表","ports"),("网络阻抗表","sections")]:self.view.addItem(text,value)
        self.view.currentIndexChanged.connect(self.change_view);row.addWidget(self.view,1);layout.addLayout(row)
        self.metrics=QLabel();self.metrics.setWordWrap(True);layout.addWidget(self.metrics)
        self.figure=Figure(figsize=(7,4),dpi=100)
        self.figure.subplots_adjust(left=.11,right=.9,bottom=.18,top=.88)
        self.canvas=FigureCanvasQTAgg(self.figure);layout.addWidget(self.canvas,1)
        self.toolbar=NavigationToolbar2QT(self.canvas,self);layout.addWidget(self.toolbar)
        self.table=QTableWidget();self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers);self.table.setAlternatingRowColors(True)
        self.table.hide();layout.addWidget(self.table,1)
        note=QLabel("前后半空间对称，不承诺唯一峰。数值搜索峰值标为估计；3D为阵因子，不是全波或实体模型。")
        note.setWordWrap(True);note.setObjectName("notice");layout.addWidget(note)
        for combo in (self.geometry,self.unit,self.topology):combo.currentIndexChanged.connect(self.mark_dirty)
        self.geometry.currentIndexChanged.connect(self.update_geometry_controls)
        self.update_geometry_controls()

    def update_geometry_controls(self):
        regular=self.geometry.currentData()=="rectangular"
        for key in ("nx","ny","spacing_x","spacing_y"):self.geometry_fields[key].setEnabled(regular)

    def update_beam_hint(self):
        self.beam_mode_hint.setText({"scan":"θ从+z轴；φ从+x朝+y。", "equalize":"当前为正侧向0°；在“幅相”中切换。",
                                     "none":"当前不补偿；在“幅相”中启用指向补偿。"}[self.electrical.phase_mode.currentData()])

    open_study = FeedPanel.open_study

    def read_parameters(self):
        values={key:int(edit.text().strip()) if key in ("nx","ny") else float(edit.text().strip()) for key,edit in self.geometry_fields.items()}
        values.update(geometry=self.geometry.currentData(),coordinate_unit=self.unit.currentData(),
                      coordinates=self.coordinates if self.geometry.currentData()=="arbitrary" else ())
        parameters=PlanarParameters(**values);parameters.validate();return parameters

    def array_parameters(self):return self.read_parameters().array_parameters()

    def preview_amplitudes(self):return planar_amplitudes(self.read_parameters(),self.read_settings())

    def read_settings(self,phase_mode=None):
        electrical=self.electrical.values()
        if phase_mode is not None:electrical["phase_mode"]=phase_mode
        settings=FeedSettings(topology=self.topology.currentData(),**{key:float(edit.text().strip()) for key,edit in self.fields.items()},**electrical,**self._branch_delay_settings)
        settings.validate();return settings

    def mark_dirty(self):
        if not self._changing and hasattr(self,"state"):
            self.state.setText("输入已改变，尚未重新生成。图形、表格与导出仍对应已有平面阵结果。")

    def toggle_controls(self):
        visible=not self.settings_tabs.isVisible();self.settings_tabs.setVisible(visible)
        self.toggle_settings.setText("收起设置" if visible else "展开设置")

    def generate(self,force_pointing=False):
        busy=False
        try:
            p,s=self.read_parameters(),self.read_settings("scan" if force_pointing else None)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            busy=True
            report=design_planar(p,s)
        except (ValueError,ArithmeticError) as exc:
            if busy:QApplication.restoreOverrideCursor();busy=False
            QMessageBox.warning(self,"平面阵设计失败",str(exc));return
        finally:
            if busy:QApplication.restoreOverrideCursor()
        if force_pointing:
            self._changing=True
            try:self.electrical.phase_mode.setCurrentIndex(self.electrical.phase_mode.findData("scan"))
            finally:self._changing=False
        self.report=report;self.export_button.setEnabled(True)
        self.state.setText(f"已生成：{p.geometry} · {p.elements}阵元 · {s.topology} · {self.electrical.phase_mode.currentText()} · {report['execution']['workers']}路计算")
        self.window.statusBar().showMessage(self.state.text())
        if not getattr(self,'parameters_in_sidebar',False) and self.view.currentData() not in ("ports","sections") and self.settings_tabs.isVisible():self.toggle_controls()
        self.redraw()
        self.window.refresh_actions()

    def change_view(self):
        if not getattr(self,'parameters_in_sidebar',False) and self.report and self.view.currentData() in ("cut","map","3d") and self.settings_tabs.isVisible():self.toggle_controls()
        self.redraw()
        self.window.refresh_actions()

    def redraw(self):
        if self.report is None:return
        view=self.view.currentData();table_view=view in ("ports","sections")
        self.canvas.setVisible(not table_view);self.toolbar.setVisible(not table_view and self.width()>=700 and self.window.height()>=820);self.table.setVisible(table_view)
        if table_view:
            if view=="ports":
                columns=[("port","阵元"),("x_lambda","x/λ"),("y_lambda","y/λ"),("power_fraction","功率占比"),
                         ("target_phase_deg","目标相位°"),("applied_delay_deg","补偿电长度°"),("output_amplitude","输出幅度"),("output_phase_deg","输出相位°")]
            else:columns=[("id","线段"),("parent","起点"),("child","终点"),("local_power_fraction","分配比"),("line_ohm","变换线Ω"),
                          ("electrical_length_deg","变换θ°"),("matched_delay_ohm","匹配段Ω"),("matched_delay_deg","支路延迟θ°"),("outside_range","超范围")]
            self.fill_table(self.table,columns,self.report["network"]["ports" if view=="ports" else "sections"])
            self.table.resizeColumnsToContents()
        else:
            draw_planar_figure(self.figure,self.report,view,chinese=True)
            self.update_plot_layout()
        self.metrics.setVisible(table_view)
        metric=self.report["electrical"]["patterns"]["output"]["metrics"]
        def number(value):return "未定义" if value is None else f"{0 if abs(value)<1e-9 else value:.4g}"
        estimate="（数值估计）" if metric["peak_search_method"]=="disk_grid_multistart_estimate" else ""
        pointing=self.report["electrical"]["pointing"]
        self.metrics.setText(f"有效目标 θ={number(pointing['target_theta_deg'])}° / φ={number(pointing['target_phi_deg'])}°  |  "
                             f"输出代表峰 θ={number(metric['representative_peak_theta_deg'])}° / φ={number(metric['representative_peak_phi_deg'])}° {estimate}\n"
                             f"代表峰与目标角距 {number(pointing['representative_peak_separation_deg'])}°（非唯一波束误差）  |  输出方向性 {number(metric['directivity_dbi'])} dBi")

    def resizeEvent(self,event):
        super().resizeEvent(event)
        self.update_plot_layout()

    def update_plot_layout(self):
        if not hasattr(self,"toolbar"):return
        compact=self.width()<700 or self.window.height()<820
        self.toolbar.setVisible(not compact and self.view.currentData() not in ("ports","sections"))
        bottom=max(.24 if compact else .18,getattr(self.figure,'_arraylab_footer_reserved',.22)+.08) if self.view.currentData() in ('cut','map','3d') else (.24 if compact else .18)
        self.figure.subplots_adjust(left=.13 if compact else .11,right=.87 if compact else .9,bottom=bottom,top=.85 if compact else .88)
        for ax in self.figure.axes:
            ax.tick_params(labelsize=7 if compact else 9)
            ax.title.set_fontsize(8 if compact else 10)
            ax.xaxis.label.set_fontsize(8 if compact else 10);ax.yaxis.label.set_fontsize(8 if compact else 10)
            if hasattr(ax,"zaxis"):ax.zaxis.label.set_fontsize(8 if compact else 10)
        self.canvas.draw_idle()

    def import_coordinates_path(self,path):
        coordinates=read_coordinates_csv(path)
        # Parse unchanged visible fields without requiring an existing arbitrary list.
        values={key:int(edit.text().strip()) if key in ("nx","ny") else float(edit.text().strip()) for key,edit in self.geometry_fields.items()}
        p=PlanarParameters(**values,geometry="arbitrary",coordinate_unit=self.unit.currentData(),coordinates=coordinates)
        validate_planar_design(p,self.read_settings())
        self.coordinates=coordinates;self.geometry.setCurrentIndex(1);self.coordinate_label.setText(f"已导入{len(coordinates)}个坐标；单位{self.unit.currentText()}")
        self.mark_dirty()

    def import_coordinates_dialog(self):
        path,_=QFileDialog.getOpenFileName(self,"导入平面阵坐标",self.window.last_directory,"坐标CSV (*.csv)",options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            try:self.import_coordinates_path(path)
            except (OSError,ValueError,OverflowError) as exc:QMessageBox.warning(self,"坐标导入失败",str(exc))

    def save_path(self,path):save_planar_configuration(path,self.read_parameters(),self.read_settings())

    def load_path(self,path):
        p,s=load_planar_configuration(path)
        self.apply_configuration(p,s)

    def apply_configuration(self,p,s):
        from .planar import validate_planar_design
        validate_planar_design(p,s)
        self._changing=True
        try:
            self.geometry.setCurrentIndex(self.geometry.findData(p.geometry));self.unit.setCurrentIndex(self.unit.findData(p.coordinate_unit))
            for key,edit in self.geometry_fields.items():edit.setText(str(getattr(p,key)))
            self.coordinates=p.coordinates;self.coordinate_label.setText(f"任意坐标：{len(p.coordinates)}点")
            self.topology.setCurrentIndex(self.topology.findData(s.topology))
            for key,edit in self.fields.items():edit.setText(str(getattr(s,key)))
            self.electrical.apply(s)
            self._branch_delay_settings = {key:getattr(s,key) for key in ('branch_delays_deg','branch_delay_elements','branch_delay_topology')}
        finally:self._changing=False
        self.mark_dirty()

    def open_dialog(self):
        path,_=QFileDialog.getOpenFileName(self,"打开平面阵设置",self.window.last_directory,"平面阵设置 (*.json)",options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            try:self.load_path(path)
            except (OSError,ValueError,OverflowError) as exc:QMessageBox.warning(self,"读取失败",str(exc))

    def save_dialog(self):
        dialog=QFileDialog(self,"保存平面阵设置",self.window.last_directory,"平面阵设置 (*.json)")
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog);dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setDefaultSuffix("json");dialog.selectFile("planar-config.json")
        if dialog.exec():
            try:self.save_path(dialog.selectedFiles()[0])
            except (OSError,ValueError,OverflowError) as exc:QMessageBox.warning(self,"保存失败",str(exc))

    def export_path(self,path):
        if self.report is None:raise ValueError("请先生成平面阵设计。")
        export_planar_directory(path,self.report,plot="all");self.last_export=Path(path)

    def export_dialog(self):
        parent=QFileDialog.getExistingDirectory(self,"选择导出位置（新建平面阵结果文件夹）",self.window.last_directory,options=QFileDialog.Option.DontUseNativeDialog)
        if parent:
            target=Path(parent)/("planar-"+datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
            try:self.export_path(target);self.window.statusBar().showMessage(f"已导出平面阵设计：{target}")
            except (OSError,ValueError) as exc:QMessageBox.warning(self,"导出失败",str(exc))
