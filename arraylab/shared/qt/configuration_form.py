"""Native typed forms for nested scientific configurations; no engine imports."""
from copy import deepcopy
import json
import math
import re

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QGroupBox,QScrollArea,
    QTabWidget,QLineEdit,QComboBox,QCheckBox,QPlainTextEdit,QLabel,QPushButton,QListWidget,
    QDialog,QDialogButtonBox,QMessageBox)


def _lookup(mapping,path,default=None):
    return mapping.get(path,mapping.get(path.replace('[]',''),mapping.get(path.split('.')[-1],default)))


class ScalarEditor(QWidget):
    changed=Signal()
    def __init__(self,value,path,metadata,parent=None):
        super().__init__(parent);self.original=deepcopy(value);self.path=path
        self.scale=_lookup(metadata.get('DISPLAY_SCALES',{}),path,1.)
        layout=QHBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        choices=_lookup(metadata.get('CHOICES',{}),path)
        if choices:
            control=QComboBox()
            for title,item in choices:control.addItem(str(title),item)
            index=control.findData(value)
            if index<0:control.addItem(str(value),value);index=control.count()-1
            control.setCurrentIndex(index);control.currentIndexChanged.connect(self.changed)
        elif type(value) is bool:
            control=QCheckBox('启用');control.setChecked(value);control.toggled.connect(self.changed)
        else:
            shown=(f'{value/self.scale:.12g}' if type(value) in (int,float) and self.scale!=1 else str(value))
            control=QLineEdit('' if value is None else shown);control.setMinimumWidth(70)
            if value is None:control.setPlaceholderText('留空：未设置')
            control.textChanged.connect(self.changed)
            if path.split('.')[-1] in ('schema_version','schemaVersion','app','calibration_hash','calibration_factor'):
                control.setReadOnly(True)
        self.control=control;layout.addWidget(control)
        self.initial_text=control.text() if isinstance(control,QLineEdit) else None
        help_text=_lookup(metadata.get('HELP',{}),path,'')
        control.setToolTip(help_text);control.setAccessibleName(_lookup(metadata.get('LABELS',{}),path,path))

    def value(self):
        c=self.control
        if isinstance(c,QComboBox):return deepcopy(c.currentData())
        if isinstance(c,QCheckBox):return c.isChecked()
        if type(self.original) is str:return c.text()
        text=c.text().strip()
        if text==self.initial_text:return deepcopy(self.original)
        if self.original is None:
            if not text:return None
            try:value=json.loads(text)
            except ValueError:value=text
            if isinstance(value,float) and not math.isfinite(value):raise ValueError(self.path+'须为有限值。')
            return value
        try:number=float(text)
        except ValueError as exc:raise ValueError(c.accessibleName()+'须填写数字。') from exc
        if not math.isfinite(number):raise ValueError(c.accessibleName()+'须为有限数值。')
        number*=self.scale
        if type(self.original) is int and self.scale==1:
            if not number.is_integer():raise ValueError(c.accessibleName()+'须为整数。')
            return int(number)
        return number


class NumericListEditor(QWidget):
    changed=Signal()
    def __init__(self,value,path,metadata,parent=None):
        super().__init__(parent);self.path=path;self.original=deepcopy(value)
        self.scale=_lookup(metadata.get('DISPLAY_SCALES',{}),path,1.)
        template=_lookup(metadata.get('LIST_DEFAULTS',{}),path)
        self.matrix=bool(value and isinstance(value[0],(list,tuple))) or isinstance(template,(list,tuple)) or path.endswith(('_positions_m','_weights'))
        self.strings=bool(value and all(isinstance(item,str) for item in value))
        layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        self.edit=QPlainTextEdit();self.edit.setMaximumHeight(125 if self.matrix else 76)
        self.edit.setAccessibleName(_lookup(metadata.get('LABELS',{}),path,path))
        show=lambda x:f'{x/self.scale:.12g}' if type(x) in (int,float) and self.scale!=1 else str(x)
        if self.matrix:text='\n'.join(', '.join(map(show,row)) for row in value)
        elif self.strings:text='\n'.join(value)
        else:text=', '.join(map(show,value))
        self.initial_text=text
        self.edit.setPlainText(text);layout.addWidget(self.edit)
        note=_lookup(metadata.get('HELP',{}),path,'每行一组数，逗号或空格分隔；留空表示空列表。' if self.matrix else '数值用逗号、空格或换行分隔；留空表示空列表。')
        label=QLabel(note);label.setWordWrap(True);layout.addWidget(label)
        self.edit.textChanged.connect(self.changed)

    def value(self):
        text=self.edit.toPlainText().strip()
        if text==self.initial_text:return deepcopy(self.original)
        if not text:return []
        if self.strings:return [line.strip() for line in text.splitlines() if line.strip()]
        def numbers(line):
            pieces=[x for x in re.split(r'[,，;；\s]+',line.strip()) if x]
            try:values=[float(x) for x in pieces]
            except ValueError as exc:raise ValueError(self.edit.accessibleName()+'中有无效数字。') from exc
            if any(not math.isfinite(x) for x in values):raise ValueError('列表数值必须有限。')
            return [int(x*self.scale) if (x*self.scale).is_integer() else x*self.scale for x in values]
        return [numbers(line) for line in text.splitlines() if line.strip()] if self.matrix else numbers(text)


class ObjectListEditor(QWidget):
    changed=Signal()
    def __init__(self,value,path,metadata,parent=None):
        super().__init__(parent);self.rows=deepcopy(value);self.path=path;self.metadata=metadata
        self.template=deepcopy(_lookup(metadata.get('LIST_DEFAULTS',{}),path,value[0] if value else {}))
        layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        self.list=QListWidget();self.list.setMaximumHeight(115);layout.addWidget(self.list)
        buttons=QHBoxLayout();layout.addLayout(buttons)
        for title,callback in [('新增',self.add),('编辑',self.edit),('删除',self.remove)]:
            button=QPushButton(title);button.clicked.connect(callback);buttons.addWidget(button)
        self.list.itemDoubleClicked.connect(self.edit);self.refresh()

    def refresh(self):
        index=self.list.currentRow();self.list.clear()
        for i,row in enumerate(self.rows):
            title=row.get('name',row.get('id',f'第{i+1}项'))
            self.list.addItem(str(title))
        if self.rows:self.list.setCurrentRow(max(0,min(index,len(self.rows)-1)))

    def _dialog(self,row):
        dialog=QDialog(self);dialog.setWindowTitle(_lookup(self.metadata.get('LABELS',{}),self.path,self.path));dialog.resize(620,650)
        layout=QVBoxLayout(dialog);form=ConfigurationForm(row,self.metadata,prefix=self.path+'[]');layout.addWidget(form)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);layout.addWidget(buttons)
        value=[]
        def accept():
            try:value.append(form.value());dialog.accept()
            except ValueError as exc:QMessageBox.warning(dialog,'输入检查',str(exc))
        buttons.accepted.connect(accept);buttons.rejected.connect(dialog.reject)
        return value[0] if dialog.exec()==QDialog.DialogCode.Accepted else None

    def add(self):
        if not self.template:
            QMessageBox.information(self,'新增项目','该列表没有行模板，请先导入带完整结构的工程。');return
        value=self._dialog(deepcopy(self.template))
        if value is not None:self.rows.append(value);self.refresh();self.changed.emit()

    def edit(self,*_):
        index=self.list.currentRow()
        if index<0:return
        value=self._dialog(self.rows[index])
        if value is not None:self.rows[index]=value;self.refresh();self.changed.emit()

    def remove(self):
        index=self.list.currentRow()
        if index>=0:self.rows.pop(index);self.refresh();self.changed.emit()

    def value(self):return deepcopy(self.rows)


class ResourceValue(QWidget):
    changed=Signal()
    def __init__(self,value,parent=None):
        super().__init__(parent);self.saved=deepcopy(value)
        layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        rows=len(value) if isinstance(value,list) else 0
        text=f'已载入资源：{rows}行。使用地形导入或工程文件更新，不在此手工编辑。'
        label=QLabel(text);label.setWordWrap(True);layout.addWidget(label)
    def value(self):return deepcopy(self.saved)


class ConfigurationForm(QTabWidget):
    changed=Signal()
    def __init__(self,configuration,metadata=None,*,prefix='',parent=None):
        super().__init__(parent);self.metadata=metadata or {};self.prefix=prefix;self.editors={};self.hidden={}
        self.set_uses_scroll_buttons();self.set_value(configuration)

    def set_uses_scroll_buttons(self):
        self.tabBar().setUsesScrollButtons(True)

    def _path(self,key):return self.prefix+'.'+key if self.prefix else key

    def set_value(self,configuration):
        self.blockSignals(True)
        try:
            while self.count():
                widget=self.widget(0);self.removeTab(0);widget.deleteLater()
            self.editors={};self.hidden={};self.original=deepcopy(configuration)
            general={k:v for k,v in configuration.items() if not isinstance(v,dict) and k not in ('app','schema_version','schemaVersion')}
            order=list(dict.fromkeys([*self.metadata.get('GROUPS',{}),*configuration]))
            for key in order:
                if key not in configuration:continue
                value=configuration[key]
                if isinstance(value,dict):
                    if key in ('source','provenance','arraylab_import'):
                        self.hidden[key]=deepcopy(value);continue
                    self._page(self.metadata.get('GROUPS',{}).get(key,key),value,self._path(key))
            if general:self._page('属性' if self.prefix else '工程信息',general,self.prefix)
        finally:self.blockSignals(False)

    def _page(self,title,values,prefix):
        page=QWidget();body=QVBoxLayout(page);self._form(body,values,prefix);body.addStretch()
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(page)
        self.addTab(scroll,title)

    def _form(self,body,values,prefix):
        form=QFormLayout();form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows);body.addLayout(form)
        for key,value in values.items():
            path=prefix+'.'+key if prefix else key
            if key in ('app','schema_version','schemaVersion','calibration_hash'):continue
            label=_lookup(self.metadata.get('LABELS',{}),path,key.replace('_',' '))
            if '.terrain.' in path and key=='metadata':
                editor=ResourceValue(value);self.editors[path]=editor;form.addRow(label,editor);continue
            if isinstance(value,dict):
                group=QGroupBox(label);nested=QVBoxLayout(group);self._form(nested,value,path);form.addRow(group);continue
            if '.terrain.' in path and key in ('heights_m','valid','x_m','y_m','metadata'):
                editor=ResourceValue(value)
            elif isinstance(value,list):
                template=_lookup(self.metadata.get('LIST_DEFAULTS',{}),path)
                editor=ObjectListEditor(value,path,self.metadata) if (value and isinstance(value[0],dict)) or isinstance(template,dict) else NumericListEditor(value,path,self.metadata)
            else:editor=ScalarEditor(value,path,self.metadata)
            editor.changed.connect(self.changed);self.editors[path]=editor
            title=QLabel(label);title.setWordWrap(True);form.addRow(title,editor)

    def value(self):
        result=deepcopy(self.original)
        for path,editor in self.editors.items():
            relative=path[len(self.prefix)+1:] if self.prefix else path
            parts=relative.split('.');target=result
            for key in parts[:-1]:target=target[key]
            target[parts[-1]]=editor.value()
        return result

    def focus_first(self):
        page=self.currentWidget()
        if page:
            controls=page.findChildren(QLineEdit)
            for control in controls:
                if control.isVisible() and control.isEnabled() and not control.isReadOnly():control.setFocus();break

    def snapshot(self):
        values={}
        for path,editor in self.editors.items():
            if isinstance(editor,ScalarEditor):
                control=editor.control
                values[path]=('choice',deepcopy(control.currentData())) if isinstance(control,QComboBox) else ('bool',control.isChecked()) if isinstance(control,QCheckBox) else ('text',control.text())
            elif isinstance(editor,NumericListEditor):values[path]=('list_text',editor.edit.toPlainText())
            elif isinstance(editor,ObjectListEditor):values[path]=('objects',deepcopy(editor.rows))
        return {'configuration':deepcopy(self.original),'tab':self.currentIndex(),'values':values}

    def restore_snapshot(self,snapshot):
        self.set_value(snapshot['configuration'])
        for path,(kind,value) in snapshot['values'].items():
            editor=self.editors[path]
            if kind=='choice':editor.control.setCurrentIndex(editor.control.findData(value))
            elif kind=='bool':editor.control.setChecked(value)
            elif kind=='text':editor.control.setText(value)
            elif kind=='list_text':editor.edit.setPlainText(value)
            elif kind=='objects':editor.rows=deepcopy(value);editor.refresh()
        self.setCurrentIndex(snapshot['tab'])
