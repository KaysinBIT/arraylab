"""Application-owned task-tree routes to existing views and parameter pages.

Navigation never starts a solver, exports a file, or creates a history store.
Linear and sparse records keep their existing feature-owned restore semantics.
"""
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import QTreeWidgetItem, QScrollArea

from arraylab.shared.qt.icons import workbench_icon
from .task_registry import TASK_REGISTRY


class TaskNavigationActions:
    def build_task_navigation(self):
        self._task_nav_nodes = {}
        self._task_nav_preferred = {}
        self._task_nav_updating = False
        self._task_navigation_ready = False
        self._feed_tree_signature = None
        self._record_nav_selected = {}
        self._task_sparse_signature = (self._sparse_view_key(), self.sparse_panel.table.currentRow(),
                                       self.sparse_panel.current_history_id, id(self.sparse_panel.bundle))
        self._task_nav_parameters = {
            'feed': self.feed_panel.settings_tabs, 'planar': self.planar_panel.settings_tabs,
            'sparse': self.sparse_panel.settings_tabs, 'subarray': self.subarray_panel.parameter_widget,
            'mimo': self.mimo_panel.parameter_widget, 'waveform': self.waveform_panel.parameter_widget}
        self._task_nav_views = {kind: getattr(self, kind+'_panel').view for kind in ('planar','subarray','mimo','waveform')}
        self._task_nav_parameters.update({key:panel.parameter_widget for key,panel in self.simulation_panels.items()})
        self._task_nav_views.update({key:panel.view for key,panel in self.simulation_panels.items()})
        for index in range(4):
            self._task_nav_nodes[('linear','result',index)] = self.linear_tree_item.child(index)
        self._nav_node(self.linear_tree_item, '参数设置', 'parameters', ('linear','parameter',0))
        for index in range(self.feed_panel.result_tabs.count()):
            self._nav_node(self.feed_tree_item, self.feed_panel.result_tabs.tabText(index),
                           'plot' if index in (1,3) else 'table', ('feed','result',index))
        pattern = self._task_nav_nodes[('feed','result',3)]
        for index, title in enumerate(('直角坐标方向图','极坐标方向图','三维方向图')):
            self._nav_node(pattern, title, 'plot', ('feed','feed_pattern',index))
        self.feed_history_tree_item = self._nav_group(self.feed_tree_item, '设计记录（本会话）')
        self.feed_history_tree_item.setToolTip(0, '已有馈电设计记录；通过顶部保存记录可备份，不新增自动历史库。')
        for kind, selector in self._task_nav_views.items():
            parent = getattr(self, kind+'_tree_item')
            groups = {}
            for index in range(selector.count()):
                key = selector.itemData(index)
                destination = parent
                if kind == 'waveform':
                    group = '天线设计结果' if key in ('antenna','sparse_design','gain_budget') else '波形与解模糊结果' if key not in ('comparisons','theory') else None
                    if group:
                        if group not in groups: groups[group] = self._nav_group(parent, group)
                        destination = groups[group]
                icon = 'table' if key in ('ports','sections','channels','unique','metrics','comparisons') else 'help' if key=='theory' else 'plot'
                self._nav_node(destination, selector.itemText(index), icon, (kind,'result',key))
            selector.currentIndexChanged.connect(lambda _, k=kind: self._task_result_changed(k))
        for kind, tabs in self._task_nav_parameters.items():
            parent = self._nav_group(getattr(self,kind+'_tree_item'), '参数设置', 'parameters')
            if kind == 'feed': self._nav_node(parent, '阵列与目标指向', 'parameters', ('feed','parameter',-1))
            for index in range(tabs.count()):
                item = self._nav_node(parent, tabs.tabText(index), 'parameters', (kind,'parameter',index))
                if kind=='waveform' and index==5:
                    radar = self.waveform_panel.radar_controls.pages
                    for subindex in range(radar.count()):
                        self._nav_node(item, radar.tabText(subindex), 'parameters', ('waveform','radar_parameter',subindex))
            tabs.currentChanged.connect(lambda index,k=kind: self._task_selector_changed(k,'parameter',index))
        self.waveform_panel.radar_controls.pages.currentChanged.connect(
            lambda index:self._task_selector_changed('waveform','radar_parameter',index))
        self.feed_panel.result_tabs.currentChanged.connect(lambda index:self._task_selector_changed('feed','result',index))
        self.feed_panel.pattern_view.currentIndexChanged.connect(lambda index:self._task_selector_changed('feed','feed_pattern',index))
        self.feed_panel.history_selection.currentIndexChanged.connect(lambda index:self._task_selector_changed('feed','history',index))
        for signal in (self.feed_panel.history_selection.model().dataChanged,
                       self.feed_panel.history_selection.model().rowsInserted,
                       self.feed_panel.history_selection.model().rowsRemoved):
            signal.connect(lambda *_:self.sync_task_navigation())
        self.selection.currentIndexChanged.connect(lambda index:self._task_selector_changed('linear','history',index))
        self.tabs.currentChanged.connect(lambda index:self._task_selector_changed('linear','result',index) if index in range(4) else None)
        self.sparse_panel.navigation_changed.connect(self._task_sparse_result_changed)
        self.tree.itemActivated.connect(self.activate_task_tree_item)
        # Parent click/Enter opens its branch even when it was already selected.
        self.tree.itemClicked.connect(self.expand_task_tree_root)
        self.tree.itemActivated.connect(self.expand_task_tree_root)
        self._task_navigation_ready = True
        self.sync_task_navigation()

    def _nav_group(self, parent, title, icon='open'):
        return self._nav_node(parent,title,icon)

    def _nav_node(self, parent, title, icon, key=None):
        item=QTreeWidgetItem(parent,[title]);item.setIcon(0,workbench_icon(icon));item.setToolTip(0,title)
        if key is not None:
            item.setData(0,Qt.ItemDataRole.UserRole,('task_node',key));self._task_nav_nodes[key]=item
        return item

    def expand_task_tree_root(self,item,*_):
        if any(item is getattr(self,task.tree_attr,None) for task in TASK_REGISTRY if task.key!='home'):
            item.setExpanded(True)

    def _task_selector_changed(self,task,section,target):
        if not self._task_navigation_ready or self._task_nav_updating:return
        if (task,section,target) not in self._task_nav_nodes and section!='history':return
        self._task_nav_preferred[task]=(task,section,target)
        self.sync_task_navigation()

    def _task_result_changed(self,task):
        self._task_selector_changed(task,'result',self._task_nav_views[task].currentData())

    def _task_sparse_result_changed(self):
        signature=(self._sparse_view_key(),self.sparse_panel.table.currentRow(),
                   self.sparse_panel.current_history_id,id(self.sparse_panel.bundle))
        changed=signature!=self._task_sparse_signature
        self._task_sparse_signature=signature
        if self._task_nav_updating or not changed:return
        self._task_nav_preferred.pop('sparse',None)
        self.sync_sparse_navigation()

    def note_legacy_navigation(self,payload):
        if not getattr(self,'_task_navigation_ready',False):return
        kind,target=payload
        if kind.startswith('sparse_'):
            self._task_nav_preferred.pop('sparse',None)
        elif kind=='result':self._task_nav_preferred['linear']=('linear','history',target)
        elif kind=='view' and target in range(4):self._task_nav_preferred['linear']=('linear','result',target)

    def navigate_task_node(self,task,section,target):
        key=(task,section,target)
        if key not in self._task_nav_nodes:return
        self._task_nav_updating=True
        try:
            self.activate_task(task)
            if section=='parameter':
                if task in self._task_nav_parameters and target>=0:
                    self._task_nav_parameters[task].setCurrentIndex(target)
                self.inspector_dock.show();self.inspector_dock.raise_()
                container=self.inspector_stack.currentWidget()
                if isinstance(container,QScrollArea):
                    if task=='feed' and target>=0:container.ensureWidgetVisible(self.feed_panel.settings_tabs)
                    else:container.verticalScrollBar().setValue(0)
            elif section=='radar_parameter':
                self.waveform_panel.parameter_widget.setCurrentIndex(5)
                self.waveform_panel.radar_controls.pages.setCurrentIndex(target)
                self.inspector_dock.show();self.inspector_dock.raise_()
            elif section=='feed_pattern':
                self.feed_panel.result_tabs.setCurrentIndex(3)
                self.feed_panel.pattern_view.setCurrentIndex(target)
            elif section=='history' and task=='feed':
                self.feed_panel.history_selection.setCurrentIndex(target)
                self.feed_panel.activate_history(target)
                actual=self.feed_panel.history_selection.currentIndex()
                if actual!=target:key=(task,section,actual)
            elif section=='result':
                if task=='linear':self.show_view(target)
                elif task=='feed':self.feed_panel.result_tabs.setCurrentIndex(target)
                else:
                    selector=self._task_nav_views[task];selector.setCurrentIndex(selector.findData(target))
            self._task_nav_preferred[task]=key
        finally:
            self._task_nav_updating=False
            self.sync_task_navigation()

    def activate_task_tree_item(self,item,*_):
        payload=item.data(0,Qt.ItemDataRole.UserRole) if item else None
        if payload and payload[0]=='task_node':
            self.navigate_task_node(*payload[1])

    def _sync_feed_tree_records(self):
        panel=self.feed_panel
        signature=tuple((id(row['report']),row['label']) for row in panel.history)
        if signature!=self._feed_tree_signature:
            self.feed_history_tree_item.takeChildren()
            for key in list(self._task_nav_nodes):
                if key[:2]==('feed','history'):del self._task_nav_nodes[key]
            for index,(_,label) in enumerate(signature):
                item=self._nav_node(self.feed_history_tree_item,label,'open',('feed','history',index))
                item.setToolTip(0,label+'\n选择恢复已保存参数和结果；双击或Enter可重选同一条，不重新设计。')
            if not signature:
                self._nav_node(self.feed_history_tree_item,'尚无设计记录','open').setDisabled(True)
            self.feed_history_tree_item.setText(0,f'设计记录（本会话 {len(signature)}）')
            self._feed_tree_signature=signature
        for index in range(len(signature)):
            item=self._task_nav_nodes[('feed','history',index)]
            font=item.font(0);font.setBold(index==panel.history_selection.currentIndex());item.setFont(0,font)

    def _current_task_navigation_key(self,task):
        preferred=self._task_nav_preferred.get(task)
        if preferred:
            _,section,target=preferred
            if section=='parameter' and (task=='linear' or target==-1 or self._task_nav_parameters[task].currentIndex()==target):return preferred
            if section=='radar_parameter' and self.waveform_panel.parameter_widget.currentIndex()==5 and self.waveform_panel.radar_controls.pages.currentIndex()==target:return preferred
            if section=='history':
                selector=self.selection if task=='linear' else self.feed_panel.history_selection
                if selector.currentIndex()==target:return preferred
            if section=='feed_pattern' and self.feed_panel.result_tabs.currentIndex()==3 and self.feed_panel.pattern_view.currentIndex()==target:return preferred
        if task=='linear':return ('linear','result',self.tabs.currentIndex())
        if task=='feed':return ('feed','result',self.feed_panel.result_tabs.currentIndex())
        if task in self._task_nav_views:return (task,'result',self._task_nav_views[task].currentData())
        return None

    def sync_task_navigation(self):
        if not getattr(self,'_task_navigation_ready',False) or self._task_nav_updating:return
        self._task_nav_updating=True
        try:
            with QSignalBlocker(self.tree):
                self._sync_feed_tree_records()
                for task,records,selector in [('linear',self.datasets,self.selection),('feed',self.feed_panel.history,self.feed_panel.history_selection)]:
                    index=selector.currentIndex();record=records[index] if 0<=index<len(records) else None
                    token=(index,id(record) if record is not None else None)
                    if token!=self._record_nav_selected.get(task) and record is not None:
                        self._task_nav_preferred[task]=(task,'history',index)
                    self._record_nav_selected[task]=token
                task=self.active_task();key=self._current_task_navigation_key(task)
                item=self._task_nav_nodes.get(key)
                if key and key[:2]==('linear','history') and 0<=key[2]<self.history_tree_item.childCount():
                    item=self.history_tree_item.child(key[2])
                if task=='sparse' and item is None:return  # Sparse owns its result/history navigation.
                if item is None:item=getattr(self,task+'_tree_item',None)
                if item is not None and self.tree.currentItem() is not item:
                    parent=item.parent()
                    while parent is not None:
                        parent.setExpanded(True);parent=parent.parent()
                    self.tree.setCurrentItem(item);self.tree.scrollToItem(item)
        finally:self._task_nav_updating=False
