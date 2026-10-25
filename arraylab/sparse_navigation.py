"""Task-tree navigation for sparse result views, schemes and saved runs."""
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import QTreeWidgetItem, QMessageBox

from .icons import workbench_icon
from .sparse2d.plots import result_name


class SparseNavigationActions:
    def build_sparse_navigation(self):
        self._sparse_nav_nodes = {}
        self._sparse_scheme_nodes = {}
        self._sparse_history_nodes = {}
        self._sparse_nav_updating = False
        self._sparse_nav_preferred = None
        self._sparse_nav_signature = None
        self._sparse_scheme_signature = None
        self._sparse_history_signature = None

        def node(parent, text, icon, payload=None):
            item = QTreeWidgetItem(parent, [text])
            item.setIcon(0, workbench_icon(icon))
            if payload is not None:
                item.setData(0, Qt.ItemDataRole.UserRole, payload)
                self._sparse_nav_nodes[tuple(payload)] = item
            item.setToolTip(0, text)
            return item

        for text, key, icon in [('布局与二维方向图', 'overview', 'model'),
                                ('扫描切线', 'cuts', 'plot')]:
            node(self.sparse_tree_item, text, icon, ('sparse_view', key))
        for text, tab in [('阵元幅相', 1), ('工况指标', 2)]:
            node(self.sparse_tree_item, text, 'table', ('sparse_tab', tab))
        node(self.sparse_tree_item, '方法比较', 'plot', ('sparse_view', 'comparison'))
        more = node(self.sparse_tree_item, '更多分析', 'open')
        for text, key in [('阵列布局（单图）', 'layout'), ('二维方向图（单图）', 'pattern'),
                          ('差分共阵', 'coarray'), ('工况比较', 'cases'),
                          ('加权前后', 'weights'), ('量化复核', 'quantization'),
                          ('单元快照', 'element'), ('结构化共阵示例', 'structures')]:
            node(more, text, 'plot', ('sparse_view', key))
        self.sparse_more_tree_item = more
        self.sparse_preview_tree_item = node(more, '单元预览（待计算）', 'parameters')
        self.sparse_preview_tree_item.setToolTip(0, '当前显示待计算输入的单元预览；点击其他结果节点返回已算快照。')
        self.sparse_preview_tree_item.setHidden(True)
        self._sparse_nav_nodes[('sparse_preview', 0)] = self.sparse_preview_tree_item
        node(self.sparse_tree_item, '稀疏后加权', 'parameters', ('sparse_tab', self.sparse_panel.weight_tab))
        self.sparse_schemes_tree_item = node(self.sparse_tree_item, '本次综合方案', 'open')
        self.sparse_history_tree_item = node(self.sparse_tree_item, '历史计算记录', 'open')
        node(self.sparse_tree_item, '操作说明', 'help', ('sparse_tab', self.sparse_panel.manual_tab))
        self.sparse_panel.navigation_changed.connect(self.sync_sparse_navigation)
        self.tree.itemActivated.connect(self.activate_sparse_tree_item)
        self.sync_sparse_navigation()

    def _sparse_view_key(self):
        panel = self.sparse_panel
        tab = panel.tabs.currentIndex()
        if tab != 0:
            return ('sparse_tab', tab)
        if getattr(panel, 'preview_active', False):
            return ('sparse_preview', 0)
        return ('sparse_view', panel.view.currentData())

    def sync_sparse_navigation(self):
        if not hasattr(self, '_sparse_nav_nodes') or self._sparse_nav_updating:
            return
        panel = self.sparse_panel
        busy = panel._busy or panel.thread is not None
        rows = panel.bundle['results'] if panel.bundle is not None else []
        entries = getattr(panel, 'history_entries', [])
        current_id = getattr(panel, 'current_history_id', None)
        row = panel.table.currentRow()
        signature = (self._sparse_view_key(), row, current_id, id(panel.bundle))
        previous = self._sparse_nav_signature
        with QSignalBlocker(self.tree):
            self.sparse_preview_tree_item.setHidden(not getattr(panel, 'preview_active', False))
            schemes = tuple((index, result_name(result)) for index, result in enumerate(rows))
            if schemes != self._sparse_scheme_signature:
                self.sparse_schemes_tree_item.takeChildren()
                self._sparse_scheme_nodes = {}
                for index, label in schemes:
                    item = QTreeWidgetItem(self.sparse_schemes_tree_item, [label])
                    item.setIcon(0, workbench_icon('plot'))
                    item.setData(0, Qt.ItemDataRole.UserRole, ('sparse_scheme', index))
                    item.setToolTip(0, label + '\n并排查看该方案的阵列布局与二维方向图。')
                    self._sparse_scheme_nodes[index] = item
                if not schemes:
                    item = QTreeWidgetItem(self.sparse_schemes_tree_item, ['尚无综合方案'])
                    item.setDisabled(True)
                self._sparse_scheme_signature = schemes
            histories = tuple((entry['id'], entry['label']) for entry in entries)
            if histories != self._sparse_history_signature:
                self.sparse_history_tree_item.takeChildren()
                self._sparse_history_nodes = {}
                for record_id, label in histories:
                    item = QTreeWidgetItem(self.sparse_history_tree_item, [f'#{record_id} {label}'])
                    item.setIcon(0, workbench_icon('open'))
                    item.setData(0, Qt.ItemDataRole.UserRole, ('sparse_history', record_id))
                    item.setToolTip(0, label + '\n单击选择；双击或按Enter可重新载入保存的参数和结果。')
                    self._sparse_history_nodes[record_id] = item
                if not histories:
                    item = QTreeWidgetItem(self.sparse_history_tree_item, ['尚无历史记录'])
                    item.setDisabled(True)
                self._sparse_history_signature = histories
            self.sparse_schemes_tree_item.setText(0, f'本次综合方案（{len(rows)}）')
            self.sparse_history_tree_item.setText(0, f'历史计算记录（{len(entries)}）')
            for index, item in self._sparse_scheme_nodes.items():
                item.setDisabled(busy)
                font = item.font(0); font.setBold(index == row); item.setFont(0, font)
            for record_id, item in self._sparse_history_nodes.items():
                item.setDisabled(busy)
                font = item.font(0); font.setBold(record_id == current_id); item.setFont(0, font)
            if previous is not None and signature != previous:
                if current_id != previous[2] and current_id in self._sparse_history_nodes:
                    self._sparse_nav_preferred = ('sparse_history', current_id)
                elif signature[0] != previous[0] or signature[3] != previous[3]:
                    self._sparse_nav_preferred = signature[0]
                elif row != previous[1] and row in self._sparse_scheme_nodes:
                    self._sparse_nav_preferred = ('sparse_scheme', row)
            preferred = self._sparse_nav_preferred or signature[0]
            selected = self._sparse_nav_nodes.get(preferred)
            if preferred[0] == 'sparse_scheme':
                selected = self._sparse_scheme_nodes.get(preferred[1])
            elif preferred[0] == 'sparse_history':
                selected = self._sparse_history_nodes.get(preferred[1])
            if self.active_task() == 'sparse':
                self.sparse_tree_item.setExpanded(True)
                if selected is not None:
                    parent = selected.parent()
                    if parent is not None: parent.setExpanded(True)
                self.tree.setCurrentItem(selected or self.sparse_tree_item)
            self._sparse_nav_signature = signature
        if getattr(self, '_workflow_ready', False):
            self.refresh_actions()

    def activate_sparse_tree_item(self, item, column):
        payload = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None
        if payload and payload[0] == 'sparse_history':
            panel = self.sparse_panel
            if panel.current_history_id == payload[1] and panel.bundle is not None and not panel.preview_active:
                from .sparse2d.core import Config
                try:
                    clean = panel.config_from_inputs() == Config(**panel.bundle['config'])
                    weighting = (panel.selected_result() or {}).get('weighting')
                    if weighting:
                        clean = clean and panel.weight_method.currentData() == weighting['method'] and all(
                            control.value() == weighting[key] for control, key in (
                                (panel.weight_dr, 'dr_db'), (panel.weight_sll, 'sll_db'),
                                (panel.weight_nbar, 'nbar'), (panel.weight_sigma, 'sigma')))
                    if clean:
                        return  # The first click already loaded this immutable snapshot.
                except (ValueError, OverflowError):
                    pass
            self.navigate_sparse(*payload)

    def navigate_sparse(self, kind, target):
        panel = self.sparse_panel
        if kind in ('sparse_scheme', 'sparse_history') and (panel._busy or panel.thread is not None):
            self.statusBar().showMessage('二维稀疏计算进行中，完成后再切换方案或历史记录。')
            self.sync_sparse_navigation()
            return
        self._sparse_nav_updating = True
        try:
            self.activate_task('sparse')
            if kind != 'sparse_history':
                panel.ensure_history_restored()
            if kind == 'sparse_view':
                index = panel.view.findData(target)
                if index < 0: raise ValueError('未知稀疏图形视图。')
                preview = getattr(panel, 'preview_active', False)
                changed = panel.view.currentIndex() != index
                panel.tabs.setCurrentIndex(0)
                panel.view.setCurrentIndex(index)
                if preview and not changed: panel.redraw()
            elif kind == 'sparse_tab':
                panel.tabs.setCurrentIndex(target)
            elif kind == 'sparse_scheme':
                if not 0 <= target < panel.table.rowCount(): raise ValueError('所选综合方案已不存在。')
                panel.tabs.setCurrentIndex(0)
                with QSignalBlocker(panel.view), QSignalBlocker(panel.table):
                    panel.view.setCurrentIndex(panel.view.findData('overview'))
                    panel.table.selectRow(target)
                panel.select_result()
            elif kind == 'sparse_history':
                if panel.restore_history(target) is False:
                    return
            self._sparse_nav_preferred = (kind, target)
            # A left-tree operation already expresses the desired focus.
            self._sparse_nav_signature = None
        except (ValueError, OSError, RuntimeError) as exc:
            QMessageBox.warning(self, '稀疏导航', str(exc))
        finally:
            self._sparse_nav_updating = False
            self.sync_sparse_navigation()
