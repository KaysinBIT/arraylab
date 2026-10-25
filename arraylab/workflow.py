"""Task-oriented commands shared by the toolbar, menus and shortcuts."""
from PySide6.QtCore import Qt,QSize
from PySide6.QtGui import QAction,QKeySequence
from PySide6.QtWidgets import QApplication,QFileDialog,QMessageBox
from .icons import workbench_icon

TASKS={
    'home':('开始','选择一项任务，或打开以前保存的计算记录。',7),
    'linear':('直线阵分析','设置参数 → F5计算并自动留存 → 选择历史结果 → 将该阵列导入馈电设计',0),
    'feed':('直线阵馈电','右侧设置阵列、馈电拓扑与幅相 → 运行设计 → 核对端口与方向图 → 保存记录',4),
    'planar':('平面阵馈电','右侧设置二维布局、目标角与馈电 → 运行设计 → 查看结果 → 导出结果',5),
    'sparse':('二维稀疏设计','右侧设置候选阵列与约束 → 运行综合 → 选择方案 → 调幅或继续设计馈电',6),
    'subarray':('子阵中心优化','右侧设置内部模板、孔径与扫描 → 优化中心位置 → 核对实际阵元及方向图 → 导出结果',8),
}


def picture_pixmap(widget):
    """Capture the full 3D surface, even if a gesture has not settled yet."""
    finish = getattr(widget, 'finish_interaction', None)
    if finish is not None:
        finish(redraw=False)
        widget.draw()
    return widget.grab()


class WorkflowActions:
    def active_task(self):
        return {4:'feed',5:'planar',6:'sparse',7:'home',8:'subarray'}.get(self.tabs.currentIndex(),'linear')

    def activate_task(self,kind):
        self.show_view(TASKS[kind][2])

    def show_view(self,index):
        self.tabs.blockSignals(True)
        self.tabs.setTabVisible(index,True);self.tabs.setCurrentIndex(index)
        self.tabs.blockSignals(False)
        self.ribbon.setCurrentIndex(0)
        self.change_workbench_context(index)

    def open_sparse(self):self.activate_task('sparse')

    def change_workbench_context(self,index):
        if not getattr(self,'_workflow_ready',False):return
        kind=self.active_task();previous=getattr(self,'_task_context',None);self._task_context=kind
        self.tabs.blockSignals(True)
        for i in range(self.tabs.count()):self.tabs.setTabVisible(i,i<4 if kind=='linear' else i==index)
        self.tabs.setCurrentIndex(index);self.tabs.blockSignals(False)
        self.tabs.tabBar().setVisible(kind=='linear')
        self.task_header.setVisible(kind!='home');self.result_bar.setVisible(kind=='linear')
        self.state_label.setVisible(kind=='linear')
        self.task_title.setText(TASKS[kind][0]);self.task_hint.setText(TASKS[kind][1])
        self.task_icon.setPixmap(workbench_icon(kind).pixmap(QSize(32,32),self.devicePixelRatioF()))
        self.project_tree_root.setText(0,'工作任务')
        self.tree.blockSignals(True)
        item={'home':self.home_tree_item,'linear':self.linear_tree_item,'feed':self.feed_tree_item,
              'planar':self.planar_tree_item,'sparse':self.sparse_tree_item,'subarray':self.subarray_tree_item}[kind]
        self.tree.setCurrentItem(item);self.linear_tree_item.setExpanded(kind=='linear')
        self.sparse_tree_item.setExpanded(kind=='sparse');self.tree.blockSignals(False)
        self.inspector_stack.setCurrentIndex({'sparse':1,'planar':2,'subarray':3}.get(kind,0))
        self.inspector_dock.setWindowTitle(TASKS[kind][0]+' · 参数')
        self.feed_parameter_host.setVisible(kind=='feed')
        for widget in self.linear_only_widgets:widget.setVisible(kind=='linear')
        self.parameter_form.setRowVisible(self.inputs['scan_deg'],kind=='linear')
        self.parameter_form.setRowVisible(self.phase_mode,kind=='linear')
        self.parameter_form.setRowVisible(self.linear_element_label,kind=='linear')
        self.parameters_box.setTitle('阵列参数' if kind=='linear' else '1. 直线阵参数')
        if previous!=kind:
            self.inspector_dock.setVisible(kind!='home')
            self.messages_dock.setVisible(False)
            self.results_scroll.verticalScrollBar().setValue(0)
            self.results_scroll.horizontalScrollBar().setValue(0)
            self.statusBar().showMessage(TASKS[kind][1])
        self.context_caption.setText(TASKS[kind][0])
        if kind == 'sparse':
            if not getattr(self, '_sparse_nav_updating', False):
                self.sparse_panel.ensure_history_restored()
            self.sync_sparse_navigation()
        self.refresh_actions()

    def refresh_actions(self):
        if not getattr(self,'_workflow_ready',False):return
        kind=self.active_task()
        panel={'sparse':self.sparse_panel,'subarray':self.subarray_panel}.get(kind)
        busy=panel is not None and (panel._busy or panel.thread is not None)
        ready={'home':False,'linear':bool(self.selected_dataset()),'feed':bool(self.feed_panel.report),
               'planar':bool(self.planar_panel.report),'sparse':bool(self.sparse_panel.bundle),'subarray':bool(self.subarray_panel.report)}[kind]
        run_text={'home':'运行计算','linear':'计算直线阵','feed':'生成馈电设计','planar':'生成平面阵设计','sparse':'开始稀疏综合','subarray':'优化子阵中心'}[kind]
        self.compute_button.setText(run_text+'\nF5');self.compute_button.setEnabled(kind!='home' and not busy)
        self.compute_button.ensurePolished()
        self.compute_button.setMinimumHeight(max(66,self.compute_button.sizeHint().height()))
        source=self.selected_dataset()
        self.to_feed_button.setVisible(kind=='linear')
        self.to_feed_button.setEnabled(kind=='linear' and source is not None and source.result is not None)
        self.to_feed_button.setMinimumHeight(max(66,self.to_feed_button.sizeHint().height()))
        self.save_button.setEnabled(ready and not busy);self.save_button.setText('导出所选曲线' if kind=='linear' else '导出当前结果')
        self.load_scheme_button.setEnabled(kind!='home' and not busy);self.save_scheme_button.setEnabled(kind!='home' and not busy)
        self.history_save_button.setVisible(kind in ('linear','feed'))
        self.history_save_button.setEnabled(bool(self.datasets if kind=='linear' else self.feed_panel.history) and not busy)
        self.history_open_button.setEnabled(not any(p._busy or p.thread is not None for p in (self.sparse_panel,self.subarray_panel)))
        self.resume_button.setEnabled(self.history_open_button.isEnabled())
        self.scan_button.setVisible(kind in ('feed','planar'));self.scan_button.setEnabled(ready)
        self.cancel_button.setVisible(kind in ('sparse','subarray'));self.cancel_button.setEnabled(busy)
        self.model_button.setEnabled(kind=='linear');self.result_button.setEnabled(kind=='linear')
        for widget in (self.rect_button,self.polar_button):widget.setEnabled(kind in ('linear','feed') and ready)
        picture=self.current_picture() is not None
        for widget in (self.image_button,self.copy_button):widget.setEnabled(picture)
        self.reset_view_button.setEnabled(ready)
        for name in ('image_action','copy_action'):
            if hasattr(self,name):getattr(self,name).setEnabled(picture)
        for name,button in [('run_action',self.compute_button),('save_action',self.save_button),
                           ('load_parameters_action',self.load_scheme_button),('save_parameters_action',self.save_scheme_button),
                           ('history_save_action',self.history_save_button),('history_open_action',self.history_open_button),
                           ('scan_action',self.scan_button)]:
            if hasattr(self,name):
                action=getattr(self,name);action.setEnabled(button.isEnabled())
                action.setVisible(kind in ('linear','feed') if name=='history_save_action' else kind in ('feed','planar') if name=='scan_action' else True)
        if hasattr(self,'run_action'):self.run_action.setText(run_text)
        if hasattr(self,'save_action'):self.save_action.setText(self.save_button.text()+'…')
        if hasattr(self,'to_feed_action'):
            self.to_feed_action.setVisible(kind=='linear')
            self.to_feed_action.setEnabled(self.to_feed_button.isEnabled())

    def import_selected_linear_to_feed(self):
        if self.active_task() != 'linear':
            return
        try:
            self.feed_panel.import_linear_result(self.selected_dataset())
        except (ValueError, ArithmeticError) as exc:
            QMessageBox.warning(self,'阵列导入失败',str(exc))
            return
        self.activate_task('feed')
        self.statusBar().showMessage('已导入所选直线阵记录；请按F5生成馈电设计。')
        self.focus_parameters()

    def compute(self):
        kind=self.active_task()
        if kind=='linear':self.compute_linear()
        elif kind=='feed':self.feed_panel.generate()
        elif kind=='planar':self.planar_panel.generate()
        elif kind=='sparse':self.sparse_panel.start()
        elif kind=='subarray':self.subarray_panel.start()
        self.refresh_actions()

    def cancel_current(self):
        panel={'sparse':self.sparse_panel,'subarray':self.subarray_panel}.get(self.active_task())
        if panel is not None:panel.cancel()

    def run_scan(self):
        panel={'feed':self.feed_panel,'planar':self.planar_panel}.get(self.active_task())
        if panel is not None:panel.open_study()

    def open_configuration_dialog(self):
        kind=self.active_task()
        if kind=='feed':self.feed_panel.open_dialog()
        elif kind=='planar':self.planar_panel.open_dialog()
        elif kind=='sparse':self.sparse_panel.open_settings_dialog()
        elif kind=='subarray':self.subarray_panel.open_dialog()
        elif kind=='linear':self.open_linear_configuration_dialog()
        self.refresh_actions()

    def save_configuration_dialog(self):
        kind=self.active_task()
        if kind=='feed':self.feed_panel.save_dialog()
        elif kind=='planar':self.planar_panel.save_dialog()
        elif kind=='sparse':self.sparse_panel.save_settings_dialog()
        elif kind=='subarray':self.subarray_panel.save_dialog()
        elif kind=='linear':self.save_linear_configuration_dialog()

    def save_dialog(self):
        kind=self.active_task()
        if kind=='feed':self.feed_panel.export_dialog()
        elif kind=='planar':self.planar_panel.export_dialog()
        elif kind=='subarray':self.subarray_panel.export_dialog()
        elif kind=='sparse':
            if not self.sparse_panel._busy:self.sparse_panel.export()
        elif kind=='linear':self.save_linear_dialog()

    def import_curve_dialog(self):
        self.activate_task('linear');self.open_dialog()

    def focus_parameters(self):
        if self.active_task()=='home':return
        self.inspector_dock.show();self.inspector_dock.raise_()
        field={'planar':self.planar_panel.geometry_fields['nx'],'sparse':self.sparse_panel.inputs['nx'],
               'subarray':self.subarray_panel.inputs['count']}.get(self.active_task(),self.inputs['frequency_ghz'])
        field.setFocus()

    def set_plot_view(self,kind):
        if self.active_task()=='linear':self.plot_type.setCurrentIndex(kind);self.show_view(0)
        elif self.active_task()=='feed':
            self.feed_panel.result_tabs.setCurrentIndex(3)
            self.feed_panel.pattern_view.setCurrentIndex(kind);self.feed_panel.change_pattern_view(kind)

    def current_picture(self):
        kind=self.active_task()
        ready={'home':False,'linear':bool(self.selected_dataset()),'feed':bool(self.feed_panel.report),
               'planar':bool(self.planar_panel.report),'sparse':bool(self.sparse_panel.bundle),'subarray':bool(self.subarray_panel.report)}[kind]
        if not ready:return None
        if kind=='linear':return self.geometry if self.tabs.currentIndex()==3 else self.plot.canvas if self.tabs.currentIndex()==0 else None
        if kind=='feed':
            return {1:self.feed_panel.diagram,3:self.feed_panel.pattern_plot.canvas}.get(self.feed_panel.result_tabs.currentIndex())
        if kind=='planar':return self.planar_panel.canvas if self.planar_panel.view.currentData() not in ('ports','sections') else None
        if kind=='sparse':return self.sparse_panel.canvas if self.sparse_panel.tabs.currentIndex()==0 else None
        if kind=='subarray':return self.subarray_panel.canvas

    def save_plot(self):
        widget=self.current_picture()
        if widget is None:self.statusBar().showMessage('请先切换到本任务的图形视图。');return
        path,_=QFileDialog.getSaveFileName(self,'保存当前图形',self.last_directory+'/arraylab-figure.png','PNG图像 (*.png)',options=QFileDialog.Option.DontUseNativeDialog)
        if path and not picture_pixmap(widget).save(path,'PNG'):QMessageBox.warning(self,'保存失败','无法写入图像文件。')

    def copy_plot(self):
        widget=self.current_picture()
        if widget is not None:QApplication.clipboard().setPixmap(picture_pixmap(widget));self.statusBar().showMessage('当前任务图形已复制。')
        else:self.statusBar().showMessage('请先切换到本任务的图形视图。')

    def reset_plot_view(self):
        kind=self.active_task()
        if kind=='linear':
            if self.tabs.currentIndex()==3:self.geometry.reset_view()
            else:self.redraw()
        elif kind=='feed':
            if self.feed_panel.result_tabs.currentIndex()==1:self.feed_panel.diagram.readable_view()
            else:self.feed_panel.redraw_patterns()
        elif kind=='planar':self.planar_panel.redraw()
        elif kind=='sparse':self.sparse_panel.redraw()
        elif kind=='subarray':self.subarray_panel.redraw()

    def show_help(self):
        kind=self.active_task()
        if kind=='home':return
        if kind=='sparse':self.sparse_panel.tabs.setCurrentIndex(self.sparse_panel.manual_tab);return
        if kind=='subarray':
            QMessageBox.information(self,'子阵中心优化',TASKS[kind][1]+'\n\n固定内部矩形模板、朝向与子阵数量，在孔径内连续优化中心位置。内部阵元随各工况理想调相。检查实际阵元间距与子阵包络不交叠；指标为采样估计。\n\n保存参数可供下次重新计算；导出包含布局、激励、报告及图形。');return
        note={'linear':'直线阵：输入频率、阵元数、间距与扫描角。每次成功F5自动保存历史，重启后恢复。勾选“叠加比较”只影响显示。选择已计算结果后，用顶部“将该阵列导入馈电设计”继续。',
              'feed':'直线阵馈电：先设阵列，再选并联、串行或1分2对称串馈。目标角需配合“按目标指向补偿”模式；端口表和方向图用于核对输出。',
              'planar':'平面阵馈电：选择矩形阵或导入x-y坐标，设置θ/φ和馈电。切面φ与目标φ分别设置。参数独立于直线阵。'}[kind]
        QMessageBox.information(self,TASKS[kind][0]+' · 操作说明',note+'\n\n'+TASKS[kind][1]+
            '\n\n保存参数：只保存当前输入，下次需重新计算。\n保存计算记录：直线阵和直线阵馈电的结果、标签及选中项可恢复，不重新计算。\n导出结果：导出当前所选计算快照，供外部查看或使用；不包含尚未计算的输入。\n\n各向同性理想电气模型；物理尺寸及误差统计暂缓。')

    def build_menus(self):
        def action(menu,title,callback,shortcut=None):
            a=QAction(title,self)
            if shortcut:a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(callback);menu.addAction(a);return a
        files=self.workbench_menu.addMenu('文件(&F)');tasks=self.workbench_menu.addMenu('任务(&T)')
        view=self.workbench_menu.addMenu('视图(&V)');help_menu=self.workbench_menu.addMenu('帮助(&H)')
        self.to_feed_action=action(tasks,'将所选阵列导入馈电设计',self.import_selected_linear_to_feed)
        self.to_feed_action.setIcon(workbench_icon('feed'))
        self.load_parameters_action=action(files,'打开当前任务参数…',self.open_configuration_dialog,'Ctrl+Shift+O')
        self.save_parameters_action=action(files,'保存当前任务参数…',self.save_configuration_dialog,'Ctrl+Shift+S')
        files.addSeparator()
        self.history_open_action=action(files,'打开计算记录…',self.load_history_dialog,'Ctrl+O')
        self.history_save_action=action(files,'保存计算记录…',self.save_history_dialog,'Ctrl+S')
        self.save_action=action(files,'导出当前结果…',self.save_dialog,'Ctrl+E')
        action(files,'读入外部曲线（直线阵）…',self.import_curve_dialog)
        files.addSeparator();action(files,'退出',self.close,'Ctrl+Q')
        self.task_actions={}
        for kind,shortcut in [('home','Ctrl+Home'),('linear',None),('feed','F6'),('planar','F7'),('sparse','F8'),('subarray','F9')]:
            entry=action(tasks,TASKS[kind][0],lambda checked=False,k=kind:self.activate_task(k),shortcut)
            entry.setIcon(workbench_icon(kind));self.task_actions[kind]=entry
        tasks.addSeparator();self.run_action=action(tasks,'运行当前任务',self.compute,'F5')
        self.scan_action=action(tasks,'评估多个扫描角',self.run_scan)
        self.scan_action.setIcon(workbench_icon('scan'))
        action(view,'显示当前参数区',self.focus_parameters)
        for item in (self.nav_dock,self.inspector_dock,self.messages_dock):view.addAction(item.toggleViewAction())
        action(view,'恢复工作布局',self.restore_workbench)
        self.image_action=action(view,'保存当前图形…',self.save_plot);self.copy_action=action(view,'复制当前图形',self.copy_plot)
        action(help_menu,'本任务操作说明',self.show_help,'F1')
        action(help_menu,'查看开始页与保存说明',lambda:self.activate_task('home'))
        action(help_menu,'直线阵说明书例题',self.load_example)
        action(help_menu,'关于 ArrayLab',self.show_about)
