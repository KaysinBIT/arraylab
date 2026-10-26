"""Shared offline radar scene controls; no numerical or feature dependencies."""
from pathlib import Path
import base64
import json
import math

from PySide6.QtCore import Qt,QUrl,QTimer,QEventLoop,QSignalBlocker,Signal
from PySide6.QtGui import QColor,QPixmap
from PySide6.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QPushButton,QSlider,QLabel
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage,QWebEngineProfile,QWebEngineSettings,QWebEngineUrlRequestInterceptor


RESOURCE=Path(__file__).resolve().parents[1]/'resources/radar_scene'


class _LocalResources(QWebEngineUrlRequestInterceptor):
    def interceptRequest(self,info):
        url=info.requestUrl()
        if url.scheme() in ('data','blob','about'):return
        try:allowed=url.isLocalFile() and Path(url.toLocalFile()).resolve().is_relative_to(RESOURCE.resolve())
        except (ValueError,OSError):allowed=False
        info.block(not allowed)


class _ScenePage(QWebEnginePage):
    message=Signal(str)
    def acceptNavigationRequest(self,url,kind,is_main):
        try:return url.isLocalFile() and Path(url.toLocalFile()).resolve().is_relative_to(RESOURCE.resolve())
        except (ValueError,OSError):return False
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel:self.message.emit(str(message))


class SceneWidget(QWidget):
    timeChanged=Signal(float)
    readyChanged=Signal(bool)
    def __init__(self,parent=None):
        super().__init__(parent)
        self.ready=False;self._closed=False;self._loaded=False;self._polling=False
        self._pending=None;self._key=None;self._state={};self._maximum=5.;self._camera=None
        body=QVBoxLayout(self);body.setContentsMargins(0,0,0,0)
        self.view=QWebEngineView(self);self.view.setMinimumSize(480,360)
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        profile=QWebEngineProfile(self);self.profile=profile
        self._interceptor=_LocalResources(profile);profile.setUrlRequestInterceptor(self._interceptor)
        page=_ScenePage(profile,self.view);profile.setParent(page);self.view.setPage(page)
        page.setBackgroundColor(QColor('#0c1c2c'));page.message.connect(self._error)
        settings=page.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls,False)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls,True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled,True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows,False)
        body.addWidget(self.view,1)
        row=QHBoxLayout();self.play_button=QPushButton('▶ 播放');self.play_button.setCheckable(True)
        row.addWidget(self.play_button);self.slider=QSlider(Qt.Orientation.Horizontal);self.slider.setRange(0,10000);row.addWidget(self.slider,1)
        self.time_label=QLabel('0.000 s');row.addWidget(self.time_label)
        self.reset_button=QPushButton('复位视角');row.addWidget(self.reset_button);body.addLayout(row)
        self.status_label=QLabel('正在载入原三维场景…');self.status_label.setWordWrap(True);body.addWidget(self.status_label)
        self.play_button.clicked.connect(lambda value:self._call('setPlaying',bool(value),callback=self._received))
        self.slider.valueChanged.connect(lambda value:self.set_time(value*self._maximum/10000,emit=True))
        self.reset_button.clicked.connect(self.reset_camera)
        self.timer=QTimer(self);self.timer.setInterval(100);self.timer.timeout.connect(self._poll)
        self.view.loadFinished.connect(self._load_finished)
        self.view.renderProcessTerminated.connect(lambda *_:self._error('三维渲染进程结束，请重新打开场景。'))
        self.view.setUrl(QUrl.fromLocalFile(str(RESOURCE/'index.html')))

    def _error(self,message):
        if not self._closed:self.status_label.setText('三维场景：'+str(message))

    def _load_finished(self,ok):
        if self._closed:return
        self._loaded=bool(ok)
        if not ok:self._error('离线场景资源未成功加载。');return
        self._call('status',callback=self._received)
        if self.isVisible():self.timer.start()

    def _call(self,method,*args,callback=None):
        if self._closed or not self._loaded:
            if callback:callback({})
            return
        encoded=','.join(json.dumps(arg,ensure_ascii=True,allow_nan=False) for arg in args)
        script=f'(()=>{{try{{return JSON.stringify(window.ArrayLabScene.{method}({encoded}));}}catch(e){{return JSON.stringify({{error:String(e)}});}}}})()'
        def done(raw):
            if self._closed:return
            try:value=json.loads(raw) if isinstance(raw,str) else raw
            except (TypeError,ValueError):value={}
            if callback:callback(value)
        self.view.page().runJavaScript(script,done)

    def _sync(self,method,*args):
        if not self._loaded or self._closed:return {}
        loop=QEventLoop();result=[]
        def done(value):result.append(value);loop.quit()
        timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit);timer.start(2000)
        self._call(method,*args,callback=done)
        if not result:loop.exec()
        timer.stop()
        return result[0] if result else {}

    def _poll(self):
        if self._polling or self._closed or not self.isVisible():return
        self._polling=True
        def received(value):self._polling=False;self._received(value)
        self._call('status',callback=received)

    def _received(self,state):
        if self._closed or not isinstance(state,dict):return
        if state.get('error'):
            self._error(state['error']);return
        if not state.get('ready'):return
        became_ready=not self.ready;self.ready=True;self._state=state
        if became_ready:
            self.readyChanged.emit(True)
            if self._pending is not None:self._submit();return
        time=float(state.get('time_s',0))
        with QSignalBlocker(self.slider),QSignalBlocker(self.play_button):
            self.slider.setValue(round(min(self._maximum,max(0,time))/self._maximum*10000))
            self.play_button.setChecked(bool(state.get('playing')))
        self.play_button.setText('Ⅱ 暂停' if state.get('playing') else '▶ 播放')
        self.time_label.setText(f'{time:.3f} s');self.timeChanged.emit(time)
        self.status_label.setText('左键旋转 · 滚轮缩放 · 右键平移；播放仅更新场景预览，不重新计算回波。')

    def set_scene(self,payload,key=None,camera=None):
        self._maximum=max(.001,float(payload.get('max_time_s',5)))
        if key is not None and key==self._key and self._pending is None:
            self.set_time(payload.get('time_s',0))
            if camera:self.restore_camera(camera)
            return
        self._key=key;self._pending=payload;self._camera=camera
        if self.ready:self._submit()

    def _submit(self):
        payload=self._pending
        if payload is None:return
        self._pending=None
        def done(state):
            self._received(state)
            if self._camera:self.restore_camera(self._camera);self._camera=None
            self._call('setVisible',self.isVisible())
        self._call('setScene',payload,callback=done)

    def set_time(self,time_s,*,emit=False):
        time=float(time_s)
        if not math.isfinite(time):raise ValueError('场景时间须为有限数值。')
        time=min(self._maximum,max(0,time))
        if self._pending is not None:self._pending={**self._pending,'time_s':time}
        if self.ready:self._call('setTime',time,callback=self._received)
        if emit:self.timeChanged.emit(time)

    def get_state(self,sync=False):
        if sync and self.ready:
            state=self._sync('getState')
            if isinstance(state,dict) and state.get('ready'):self._received(state)
        return dict(self._state)

    def restore_camera(self,camera):
        if self.ready:self._call('restoreCamera',camera,callback=self._received)
        else:self._camera=camera

    def reset_camera(self):
        self._call('resetCamera',callback=self._received)

    def pause(self):
        self._received(self._sync('setPlaying',False))

    def png_bytes(self):
        if not self.ready:raise ValueError('请等待三维场景载入后再保存图像。')
        value=self._sync('png')
        if not isinstance(value,str) or not value.startswith('data:image/png;base64,'):raise ValueError('未取得三维场景图像。')
        return base64.b64decode(value.split(',',1)[1],validate=True)

    def capture_pixmap(self):
        pixmap=QPixmap();pixmap.loadFromData(self.png_bytes(),'PNG');return pixmap

    def showEvent(self,event):
        super().showEvent(event)
        if self._closed:return
        if self._loaded:self._call('setVisible',True);self.timer.start()

    def hideEvent(self,event):
        self.timer.stop()
        if not self._closed:
            self._call('setPlaying',False,callback=self._received);self._call('setVisible',False)
        super().hideEvent(event)

    def shutdown(self):
        if self._closed:return
        self.timer.stop()
        if self._loaded:self._sync('dispose')
        self._closed=True;self.ready=False;self._pending=None;self.view.stop();self.view.page().deleteLater()

    def closeEvent(self,event):
        self.shutdown();super().closeEvent(event)
