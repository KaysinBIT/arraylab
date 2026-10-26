"""Coalesce desktop resizes while retaining the last rendered plot as a preview.

Only the temporary screen image is scaled. The Figure, its numerical artists,
camera and export resolution are rendered normally after the resize settles.
"""
from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QImage, QPainter, QResizeEvent
from PySide6.QtWidgets import QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


class ResizeAwareFigureCanvas(FigureCanvasQTAgg):
    resize_finished = Signal()
    resize_settle_ms = 120

    def __init__(self, figure=None):
        self._resize_ready = False
        self._completed_renderer = None
        super().__init__(figure)
        self._resize_pending = False
        self._resize_applying = False
        self._resize_image = None
        self.resize_preview_paints = 0
        self.resize_flush_count = 0
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self.finish_resize)
        self._resize_ready = True

    @property
    def resize_pending(self):
        return getattr(self, '_resize_pending', False)

    def resizeEvent(self, event):
        if not getattr(self, '_resize_ready', False) or self._resize_applying:
            return super().resizeEvent(event)
        if not self._resize_pending:
            renderer = self._completed_renderer
            if renderer is not None:
                buffer = renderer.buffer_rgba()
                # Own the memory: subsequent footer/layout work may replace the
                # Agg renderer, and QImage must not retain a dangling view.
                self._resize_image = QImage(buffer, buffer.shape[1], buffer.shape[0],
                    QImage.Format.Format_RGBA8888).copy()
            self._resize_pending = True
        self._draw_pending = False
        QWidget.resizeEvent(self, event)
        self._resize_timer.start(self.resize_settle_ms)
        self.update()

    def paintEvent(self, event):
        if not self.resize_pending:
            return super().paintEvent(event)
        painter = QPainter(self)
        try:
            if self._resize_image is not None:
                painter.drawImage(self.rect(), self._resize_image)
                self.resize_preview_paints += 1
            else:
                painter.fillRect(event.rect(), self.palette().window())
        finally:
            painter.end()

    def draw_idle(self):
        if self.resize_pending:
            return
        return super().draw_idle()

    def _draw_idle(self):
        if self.resize_pending:
            self._draw_pending = False
            return
        return super()._draw_idle()

    def finish_resize(self, *, redraw=True):
        """Apply the latest size once; explicit draw/copy/export may force it."""
        if not getattr(self, '_resize_ready', False) or not self._resize_pending:
            return False
        self._resize_timer.stop()
        self._resize_pending = False
        self._resize_applying = True
        try:
            self.resize_flush_count += 1
            super().resizeEvent(QResizeEvent(self.size(), self.size()))
            self.resize_finished.emit()
        finally:
            self._resize_applying = False
            self._resize_image = None
        if not redraw:
            self._draw_pending = False
        return True

    def draw(self):
        self.finish_resize(redraw=False)
        result = super().draw()
        self._completed_renderer = getattr(self, 'renderer', None)
        return result

    def print_figure(self, *args, **kwargs):
        self.finish_resize(redraw=False)
        return super().print_figure(*args, **kwargs)

    def mousePressEvent(self, event):
        if self.resize_pending:
            self.draw()
        return super().mousePressEvent(event)

    def wheelEvent(self, event):
        if self.resize_pending:
            self.draw()
        return super().wheelEvent(event)

    def hideEvent(self, event):
        self.finish_resize(redraw=False)
        return super().hideEvent(event)

    def closeEvent(self, event):
        if getattr(self, '_resize_ready', False):
            self.finish_resize(redraw=False)
            self._resize_timer.stop()
            self._resize_pending = False
            self._resize_image = None
            self._draw_pending = False
            self._completed_renderer = None
        return super().closeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if getattr(self, '_resize_ready', False):
            self.draw_idle()
