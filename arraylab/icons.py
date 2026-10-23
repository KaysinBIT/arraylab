"""Small original vector-style workbench icons, drawn locally with Qt."""
from functools import lru_cache
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF


@lru_cache(maxsize=32)
def workbench_icon(kind):
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    pixmap.setDevicePixelRatio(2)
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor("#446b89"), 1.6))
    p.setBrush(QColor("#e2edf5"))
    if kind in ("model", "array"):
        p.drawLine(3, 25, 29, 10)
        for x, y in [(7, 23), (16, 18), (25, 13)]:
            p.setBrush(QColor("#d2a444"))
            p.drawPolygon(QPolygonF([QPointF(x-4, y-5), QPointF(x, y-8), QPointF(x+4, y-5), QPointF(x, y-2)]))
            p.setBrush(QColor("#e8bf61"))
            p.drawPolygon(QPolygonF([QPointF(x-4, y-5), QPointF(x, y-2), QPointF(x, y+3), QPointF(x-4, y)]))
            p.setBrush(QColor("#bc8b30"))
            p.drawPolygon(QPolygonF([QPointF(x, y-2), QPointF(x+4, y-5), QPointF(x+4, y), QPointF(x, y+3)]))
    elif kind == "run":
        p.setBrush(QColor("#4d943c")); p.setPen(QPen(QColor("#3c7c30"), 1.3))
        p.drawPolygon(QPolygonF([QPointF(9, 4), QPointF(27, 16), QPointF(9, 28)]))
    elif kind in ("plot", "polar"):
        if kind == "plot":
            p.setPen(QPen(QColor("#7c8b98"), 1.3)); p.drawLine(5, 4, 5, 27); p.drawLine(5, 27, 29, 27)
            p.setPen(QPen(QColor("#2a72b8"), 2))
            p.drawPolyline(QPolygonF([QPointF(6, 23), QPointF(10, 19), QPointF(14, 23), QPointF(19, 7), QPointF(24, 22), QPointF(28, 18)]))
        else:
            p.setBrush(Qt.BrushStyle.NoBrush); p.setPen(QPen(QColor("#a2b3c1"), 1))
            p.drawEllipse(QRectF(4, 4, 24, 24)); p.drawLine(16, 3, 16, 29); p.drawLine(3, 16, 29, 16)
            p.setPen(QPen(QColor("#2a72b8"), 2)); p.drawEllipse(QRectF(13, 5, 11, 20))
    elif kind in ("table", "parameters"):
        p.drawRect(QRectF(4, 5, 24, 23)); p.fillRect(QRectF(4, 5, 24, 6), QColor("#77a6ca"))
        for y in (11, 17, 23): p.drawLine(4, y, 28, y)
        for x in (12, 20): p.drawLine(x, 11, x, 28)
    elif kind == "open":
        p.setBrush(QColor("#edcc75")); p.setPen(QPen(QColor("#b18b3d"), 1.4))
        p.drawPolygon(QPolygonF([QPointF(3, 8), QPointF(12, 8), QPointF(15, 11), QPointF(28, 11), QPointF(28, 26), QPointF(3, 26)]))
        p.setBrush(QColor("#f6dda0")); p.drawPolygon(QPolygonF([QPointF(3, 14), QPointF(30, 14), QPointF(26, 27), QPointF(3, 27)]))
    elif kind == "save":
        p.setBrush(QColor("#598bb5")); p.drawRect(QRectF(5, 4, 22, 24))
        p.setBrush(QColor("#f6f8fa")); p.drawRect(QRectF(9, 5, 14, 8)); p.drawRect(QRectF(9, 18, 14, 10))
        p.setPen(QPen(QColor("#7b8d9b"), 1)); p.drawLine(12, 22, 20, 22); p.drawLine(12, 25, 20, 25)
    elif kind == "copy":
        p.drawRect(QRectF(4, 4, 17, 21)); p.setBrush(QColor("#ffffff")); p.drawRect(QRectF(11, 10, 17, 19))
        p.drawLine(15, 16, 24, 16); p.drawLine(15, 21, 24, 21)
    elif kind == "reset":
        p.setBrush(Qt.BrushStyle.NoBrush); p.setPen(QPen(QColor("#367db4"), 2.5))
        p.drawArc(QRectF(6, 5, 22, 22), 30*16, 275*16)
        p.setBrush(QColor("#367db4")); p.drawPolygon(QPolygonF([QPointF(4, 9), QPointF(13, 9), QPointF(5, 16)]))
    elif kind == "layout":
        p.drawRect(QRectF(3, 4, 26, 24)); p.fillRect(QRectF(4, 5, 24, 5), QColor("#77a6ca"))
        p.drawLine(10, 10, 10, 28); p.drawLine(23, 10, 23, 28); p.drawLine(10, 23, 29, 23)
    elif kind == "help":
        p.setBrush(QColor("#4a88b7")); p.drawEllipse(QRectF(4, 4, 24, 24))
        p.setPen(Qt.GlobalColor.white); font = p.font(); font.setPixelSize(22); font.setBold(True); p.setFont(font)
        p.drawText(QRectF(4, 2, 24, 27), Qt.AlignmentFlag.AlignCenter, "?")
    p.end()
    return QIcon(pixmap)
