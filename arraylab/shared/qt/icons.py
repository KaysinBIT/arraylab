"""Application artwork and small vector-style workbench icons."""
from functools import lru_cache
from pathlib import Path
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF


@lru_cache(maxsize=1)
def application_icon():
    """Load the same application artwork in source and frozen desktop builds."""
    from arraylab.framework.resources import resource_path
    resources = resource_path("resources", "icons")
    icon = QIcon(str(resources / "arraylab.ico"))
    icon.addFile(str(resources / "arraylab.png"))
    return icon


MODULE_ICON_KINDS = frozenset(('home', 'linear', 'feed', 'planar', 'sparse', 'subarray', 'scan', 'mimo', 'waveform', 'phased', 'equation', 'pulse'))


def _draw_module(p, kind):
    """Original module silhouettes on a 32-unit grid, independent of size."""
    blue, teal, gold = '#285f88', '#008f9c', '#d7a13d'

    def pen(color=blue, width=2):
        p.setPen(QPen(QColor(color), width, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))

    def line(x1, y1, x2, y2):
        p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    def dot(x, y, radius, color):
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(color))
        p.drawEllipse(QRectF(x-radius, y-radius, 2*radius, 2*radius))

    def tile(x, y, size=5):
        pen('#956b23', 1); p.setBrush(QColor('#edbf5c'))
        p.drawRoundedRect(QRectF(x, y, size, size), .7, .7)

    if kind == 'home':
        pen(); p.setBrush(QColor('#e3eff7'))
        p.drawPolygon(QPolygonF([QPointF(3, 14), QPointF(16, 3), QPointF(29, 14),
                                QPointF(26, 14), QPointF(26, 29), QPointF(6, 29), QPointF(6, 14)]))
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(teal))
        p.drawRoundedRect(QRectF(13, 20, 6, 9), 1, 1)
        tile(9, 14, 4); tile(19, 14, 4)
    elif kind == 'linear':
        # A broadside lobe above a single physical row of elements.
        pen(teal, 1.8); p.setBrush(QColor('#c7edf0'))
        beam = QPainterPath(QPointF(16, 21))
        beam.cubicTo(5, 16, 8, 3, 16, 3)
        beam.cubicTo(24, 3, 27, 16, 16, 21)
        p.drawPath(beam)
        pen(teal, 1.3); line(16, 5, 16, 20)
        pen(blue, 2); line(3, 29, 29, 29)
        for x in (3, 10, 17, 24):
            line(x+2.5, 25, x+2.5, 29)
        for x in (3, 10, 17, 24): tile(x, 22)
    elif kind == 'feed':
        # One input, two junctions, four output ports.
        pen(blue, 2.3)
        line(16, 27, 16, 21); line(8, 21, 24, 21)
        line(8, 21, 8, 13); line(24, 21, 24, 13)
        pen(teal, 2.3)
        line(4, 13, 12, 13); line(20, 13, 28, 13)
        for x in (4, 12, 20, 28): line(x, 13, x, 8)
        for x in (1.5, 9.5, 17.5, 25.5): tile(x, 3)
        dot(16, 28, 2.5, blue)
    elif kind == 'planar':
        # A fully populated 3-by-3 aperture with visible distribution lines.
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor('#e3eff7'))
        p.drawRoundedRect(QRectF(2, 2, 28, 28), 3, 3)
        pen(teal, 1.8)
        for y in (7, 16, 25): line(7, y, 25, y)
        for x in (7, 16, 25): line(x, 7, x, 25)
        for y in (4.5, 13.5, 22.5):
            for x in (4.5, 13.5, 22.5): tile(x, y)
    elif kind == 'sparse':
        # Empty candidate positions remain visible around the selected support.
        selected = {(0, 0), (0, 2), (1, 1), (1, 3), (2, 0), (2, 2), (3, 1), (3, 3)}
        for row in range(4):
            for column in range(4):
                x, y = 5.5 + column*7, 5.5 + row*7
                active = (row, column) in selected
                dot(x, y, 2.8 if active else 1.35, teal if active else '#b8c8d5')
        pen(blue, 1.5); p.setBrush(Qt.BrushStyle.NoBrush)
        line(2, 9, 2, 2); line(2, 2, 9, 2)
        line(23, 30, 30, 30); line(30, 30, 30, 23)
    elif kind == 'subarray':
        # Three intact blocks are repositioned as groups, not individual dots.
        pen(teal, 1.6); p.setBrush(Qt.BrushStyle.NoBrush)
        line(8, 15, 8, 21); line(8, 21, 13, 21)
        line(24, 14, 24, 24); line(24, 24, 21, 24)
        p.drawPolyline(QPolygonF([QPointF(10, 18), QPointF(13, 21), QPointF(10, 24)]))
        for x, y in ((2.5, 3), (19, 3), (11, 22)):
            pen(blue, 1.6); p.setBrush(QColor('#e3eff7'))
            p.drawRoundedRect(QRectF(x, y, 10, 8), 1.4, 1.4)
            dot(x+3, y+4, 1.55, gold); dot(x+7, y+4, 1.55, gold)
    elif kind == 'waveform':
        pen(blue, 1.3); line(3, 28, 29, 28); line(3, 3, 3, 28)
        pen(teal, 2); p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPolyline(QPolygonF([QPointF(5, 22), QPointF(13, 5), QPointF(13, 22), QPointF(21, 5), QPointF(21, 22), QPointF(29, 5)]))
        pen(gold, 2); line(6, 25, 12, 25); line(16, 25, 22, 25); line(26, 25, 29, 25)
    elif kind == 'mimo':
        # Independent Tx and Rx rows form a larger virtual aperture.
        for x in (4, 23): tile(x, 3, 5)
        for x in (5, 12, 20, 27): dot(x, 27, 2.1, teal)
        pen(blue, 1.5); line(6, 10, 6, 13); line(25, 10, 25, 13)
        line(6, 13, 25, 13); line(16, 13, 16, 16)
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(blue))
        p.drawPolygon(QPolygonF([QPointF(13, 14), QPointF(19, 14), QPointF(16, 18)]))
        for x in (4, 8, 12, 16, 20, 24, 28): dot(x, 20, 1.4, gold)
    elif kind == 'phased':
        for y in (20,26):
            for x in (3,10,17,24):tile(x,y,4)
        pen(teal,1.8);p.setBrush(QColor('#c7edf0'))
        beam=QPainterPath(QPointF(16,20));beam.cubicTo(15,11,21,2,27,3);beam.cubicTo(30,7,24,15,16,20);p.drawPath(beam)
        pen(blue,1.4)
        for x in (5,12,19,26):line(x,19,x+3,13)
    elif kind == 'equation':
        pen(blue,1.8);p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(2,3,28,26));line(7,9,14,9);line(7,22,14,22);line(7,9,12,15);line(12,15,7,22)
        pen(teal,1.8);line(18,13,26,13);line(18,18,26,18);dot(23,24,1.8,gold)
    elif kind == 'pulse':
        pen(teal,2.2);p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPolyline(QPolygonF([QPointF(2,24),QPointF(6,24),QPointF(6,8),QPointF(12,8),QPointF(12,24),QPointF(18,24),QPointF(18,8),QPointF(24,8),QPointF(24,24),QPointF(30,24)]))
        pen(gold,1.6);line(7,5,23,5);line(7,3,7,6);line(23,3,23,6)
    elif kind == 'scan':
        # A tilted main beam and a curved sweep arrow share the source point.
        pen('#a9bfce', 1.5); p.setBrush(Qt.BrushStyle.NoBrush)
        line(16, 28, 5, 10); line(16, 28, 16, 4)
        pen(teal, 1.8); p.setBrush(QColor('#c7edf0'))
        beam = QPainterPath(QPointF(16, 28))
        beam.cubicTo(14, 16, 19, 4, 25, 5)
        beam.cubicTo(32, 7, 24, 23, 16, 28)
        p.drawPath(beam)
        pen(teal, 1.2); line(16, 28, 25, 7)
        pen(blue, 2.1); p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(QRectF(4, 5, 24, 25), 45*16, 88*16)
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(blue))
        p.drawPolygon(QPolygonF([QPointF(27, 7), QPointF(26, 13), QPointF(21, 9)]))
        dot(16, 28, 2.5, gold)


def _module_icon(kind):
    # Rasterize from the vector geometry for each native size. Qt can then
    # select a sharp source at both normal and high display scaling factors.
    icon = QIcon()
    for size in (16, 20, 24, 29, 32, 40, 48, 58, 64, 96, 128):
        pixmap = QPixmap(size, size); pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.scale(size/32, size/32)
        _draw_module(painter, kind)
        painter.end()
        icon.addPixmap(pixmap)
    return icon


@lru_cache(maxsize=32)
def workbench_icon(kind):
    if kind in MODULE_ICON_KINDS:
        return _module_icon(kind)
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
