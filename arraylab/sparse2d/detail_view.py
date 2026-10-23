"""Four independent detail plots and modeless maximized plot windows."""
from PySide6.QtCore import Qt,QTimer
from PySide6.QtWidgets import (QDialog,QFrame,QGridLayout,QHBoxLayout,QLabel,
    QPushButton,QScrollArea,QStackedWidget,QVBoxLayout,QWidget)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg,NavigationToolbar2QT
from matplotlib.figure import Figure
from .plots import DETAIL_PLOTS,display_case,draw_detail_plot,result_name


class PlotCard(QFrame):
    def __init__(self,kind,title,open_plot,parent=None):
        super().__init__(parent)
        self.kind=kind; self.open_plot=open_plot
        self.setStyleSheet('PlotCard{background:white;border:1px solid #dce4ee;border-radius:8px}')
        layout=QVBoxLayout(self); layout.setContentsMargins(7,5,7,5); layout.setSpacing(2)
        bar=QHBoxLayout(); label=QLabel(title); label.setStyleSheet('font-weight:600;color:#34435a')
        bar.addWidget(label); bar.addStretch()
        self.maximize_button=QPushButton('最大化'); self.maximize_button.setEnabled(False)
        self.maximize_button.setToolTip('独立窗口最大化查看；也可双击图形。')
        self.maximize_button.setStyleSheet('padding:3px 9px;color:#087e70')
        self.maximize_button.clicked.connect(lambda:self.open_plot(self.kind))
        bar.addWidget(self.maximize_button); layout.addLayout(bar)
        self.figure=Figure(figsize=(5,3.2),dpi=100,layout='constrained')
        self.canvas=FigureCanvasQTAgg(self.figure); self.canvas.setMinimumSize(200,155)
        self.canvas.mpl_connect('button_press_event',self.double_click)
        layout.addWidget(self.canvas,1)

    def double_click(self,event):
        if event.dblclick and event.button==1 and self.maximize_button.isEnabled():
            self.open_plot(self.kind)


class PlotWindow(QDialog):
    def __init__(self,kind,title,parent):
        super().__init__(parent,Qt.WindowType.Window|Qt.WindowType.WindowMinMaxButtonsHint|Qt.WindowType.WindowCloseButtonHint)
        self.kind=kind; self.title=title; self.context=None
        self.setModal(False); self.resize(1180,850); self.setMinimumSize(700,500)
        layout=QVBoxLayout(self); layout.setContentsMargins(14,12,14,12)
        self.context_label=QLabel(); self.context_label.setWordWrap(True)
        self.context_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.context_label)
        self.figure=Figure(figsize=(12,8),dpi=100,layout='constrained')
        self.canvas=FigureCanvasQTAgg(self.figure)
        self.toolbar=NavigationToolbar2QT(self.canvas,self)
        layout.addWidget(self.toolbar); layout.addWidget(self.canvas,1)
        footer=QHBoxLayout(); footer.addWidget(QLabel('跟随主窗口的方案、工况和显示方式 · Esc 关闭此窗口'))
        footer.addStretch(); self.return_button=QPushButton('关闭，返回四图总览')
        self.return_button.clicked.connect(self.close); footer.addWidget(self.return_button); layout.addLayout(footer)
        self.finished.connect(lambda _:QTimer.singleShot(0,self.return_main_focus))

    def return_main_focus(self):
        parent=self.parentWidget()
        if parent is not None and parent.isVisible():
            parent.raise_(); parent.activateWindow(); parent.setFocus(Qt.FocusReason.OtherFocusReason)

    def set_context(self,bundle,result,case_index,view_mode,phase_mode='ideal'):
        self.context=(bundle,result,case_index,view_mode,phase_mode)
        _,_,case=display_case(bundle,result,case_index)
        view={'total':'总方向图','array_factor':'仅阵因子 AF','element':'仅单元方向图'}[view_mode]
        phase=f"{bundle['config'].get('phase_bits',0)} 位量化" if phase_mode=='quantized' else '理想相位'
        text=f"{result_name(result)} · {case['label']} · {view} · {phase} · 已计算快照"
        self.setWindowTitle(self.title+' — '+text); self.context_label.setText(text)
        draw_detail_plot(self.figure,bundle,result,self.kind,case_index,view_mode,phase_mode=phase_mode)
        self.toolbar.update(); self.canvas.draw_idle()


class DetailView(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.context=None; self.dialogs={}; self.cards={}
        outer=QVBoxLayout(self); outer.setContentsMargins(0,0,0,0); outer.setSpacing(5)
        note=QLabel('四图独立查看：点击“最大化”或双击图形。放大窗口可缩放、平移和保存单图。')
        note.setWordWrap(True); note.setStyleSheet('font-size:12px;color:#617287;padding:3px 6px')
        outer.addWidget(note)
        self.stack=QStackedWidget(); outer.addWidget(self.stack,1)
        self.empty_hint=QLabel('在左侧完成设置后点击“开始计算”\n计算完成后，在上方表格选择方案')
        self.empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_hint.setStyleSheet('font-size:16px;color:#617287')
        self.stack.addWidget(self.empty_hint)
        body=QWidget(); grid=QGridLayout(body); grid.setContentsMargins(5,3,5,3); grid.setSpacing(8)
        for index,(kind,title) in enumerate(DETAIL_PLOTS):
            card=PlotCard(kind,title,self.open_plot,body); self.cards[kind]=card
            grid.addWidget(card,index//2,index%2)
        for index in (0,1): grid.setRowStretch(index,1); grid.setColumnStretch(index,1)
        scroll=QScrollArea(); scroll.setFrameShape(QFrame.Shape.NoFrame); scroll.setWidgetResizable(True)
        scroll.setWidget(body); self.stack.addWidget(scroll)

    def set_context(self,bundle,result,case_index=None,view_mode='total',phase_mode='ideal'):
        self.context=(bundle,result,case_index,view_mode,phase_mode)
        self.stack.setCurrentIndex(1)
        for kind,card in self.cards.items():
            draw_detail_plot(card.figure,bundle,result,kind,case_index,view_mode,compact=True,phase_mode=phase_mode)
            card.maximize_button.setEnabled(True); card.canvas.draw_idle()
        for dialog in self.dialogs.values():
            if dialog.isVisible(): dialog.set_context(*self.context)

    def open_plot(self,kind):
        if self.context is None: return None
        titles=dict(DETAIL_PLOTS)
        if kind not in titles: raise ValueError('Unknown plot: '+str(kind))
        if kind not in self.dialogs:
            self.dialogs[kind]=PlotWindow(kind,titles[kind],self.window())
        dialog=self.dialogs[kind]; dialog.set_context(*self.context)
        dialog.showMaximized(); dialog.raise_(); dialog.activateWindow()
        return dialog

    def close_dialogs(self):
        for dialog in self.dialogs.values(): dialog.close()
