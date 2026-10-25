from dataclasses import dataclass
import math

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

import matplotlib
matplotlib.rcParams.update({
    "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "agg.path.chunksize": 10000,
})
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from .interactive_canvas import InteractiveFigureCanvas as FigureCanvasQTAgg
from .surface_plot import clear_figure
from matplotlib.figure import Figure

from .core import ArrayParameters, ArrayResult, pattern_db_at


@dataclass
class PatternSet:
    label: str
    angles: np.ndarray
    db: np.ndarray
    phase: np.ndarray | None = None
    result: ArrayResult | None = None
    feed_report: dict | None = None
    pattern_key: str = "output"
    color: str | None = None


class NumericTableModel(QAbstractTableModel):
    def __init__(self, headers, columns, parent=None):
        super().__init__(parent)
        self.headers = headers
        self.columns = columns

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() or not self.columns else len(self.columns[0])

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.headers)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return f"{self.columns[index.column()][index.row()]:.9g}"
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            return self.headers[section] if orientation == Qt.Orientation.Horizontal else str(section + 1)
        return None


class GeometryView(QWidget):
    def __init__(self):
        super().__init__()
        self.parameters = ArrayParameters()
        self.grid_visible = True
        self.axes_visible = True
        self.perspective = True
        self.zoom = 1.0
        self.setMinimumSize(360, 260)
        self.setAccessibleName("均匀直线阵模型示意")
        self.setToolTip("滚轮缩放；模型为各向同性点源位置示意。")

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.update()

    def set_grid(self, enabled):
        self.grid_visible = enabled
        self.update()

    def set_axes(self, enabled):
        self.axes_visible = enabled
        self.update()

    def set_view(self, index):
        self.perspective = index == 0
        self.update()

    def reset_view(self):
        self.zoom = 1.0
        self.update()

    def wheelEvent(self, event):
        self.zoom = max(.35, min(2.5, self.zoom * (1.12 if event.angleDelta().y() > 0 else 1/1.12)))
        self.update()
        event.accept()

    def paintEvent(self, event):
        from PySide6.QtGui import QLinearGradient, QRadialGradient
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        background = QLinearGradient(0, 0, 0, h)
        background.setColorAt(0, QColor("#eef2f6")); background.setColorAt(.6, QColor("#ffffff"))
        p.fillRect(self.rect(), background)
        cx, cy = w * .5, h * .62
        span = min(w * .72, h * 1.15, 700) * self.zoom
        def project(x, y=0, z=0):
            if self.perspective:
                return QPointF(cx + .94*x - .55*y, cy - .25*x - .22*y - .92*z)
            return QPointF(cx + x, cy - z)
        if self.grid_visible:
            p.setPen(QPen(QColor("#e1e7ec"), 1))
            if self.perspective:
                for i in range(-6, 7):
                    x = span*i/10
                    p.drawLine(project(x, -span*.32), project(x, span*.32))
                for i in range(-4, 5):
                    y = span*i/12
                    p.drawLine(project(-span*.6, y), project(span*.6, y))
            else:
                for i in range(-6, 7):
                    p.drawLine(project(span*i/10, 0, -span*.3), project(span*i/10, 0, span*.5))
                for i in range(-3, 6):
                    p.drawLine(project(-span*.6, 0, span*i/10), project(span*.6, 0, span*i/10))
        p.setPen(QPen(QColor("#a2885f"), 1.5))
        p.drawLine(project(-span*.5), project(span*.5))
        n = self.parameters.elements
        step = span/max(n-1, 4)
        radius = max(1.8, min(9.0, step*.2))
        for i in range(n):
            x = (i-(n-1)/2)*step
            point = project(x)
            gradient = QRadialGradient(point.x()-radius*.3, point.y()-radius*.3, radius*1.5)
            gradient.setColorAt(0, QColor("#fff1c2")); gradient.setColorAt(.5, QColor("#edbd61")); gradient.setColorAt(1, QColor("#aa6f20"))
            p.setBrush(gradient); p.setPen(QPen(QColor("#987036"), .8))
            p.drawEllipse(point, radius, radius)
            if n <= 16:
                p.setPen(QColor("#758797"))
                p.drawText(QRectF(point.x()-14, point.y()+12, 28, 20), Qt.AlignmentFlag.AlignCenter, str(i+1))
        angle = math.radians(self.parameters.scan_deg)
        origin = project(0)
        tip = project(span*.36*math.sin(angle), 0, span*.36*math.cos(angle))
        p.setPen(QPen(QColor("#287db3"), 2.4)); p.drawLine(origin, tip)
        direction = math.atan2(tip.y()-origin.y(), tip.x()-origin.x())
        for delta in (-.45, .45):
            p.drawLine(tip, QPointF(tip.x()-12*math.cos(direction+delta), tip.y()-12*math.sin(direction+delta)))
        p.setPen(QColor("#246b9b"))
        p.drawText(QPointF(tip.x()+10, tip.y()-6), f"θ₀ = {self.parameters.scan_deg:g}°")
        if self.axes_visible:
            ox, oy = 52, h-56
            for dx, dy, label, color in [(35, 10, "x", "#be514c"), (-23, 17, "y", "#559d66"), (0, -37, "z", "#397fc2")]:
                p.setPen(QPen(QColor(color), 2)); p.drawLine(QPointF(ox, oy), QPointF(ox+dx, oy+dy))
                p.drawText(QPointF(ox+dx+4, oy+dy-2), label)
        p.setPen(QColor("#34556d")); p.setFont(QFont("Microsoft YaHei", 10))
        p.drawText(QPointF(18, 28), "均匀直线阵  /  所选计算结果")
        p.setPen(QColor("#6c8192")); p.setFont(QFont("Microsoft YaHei", 9))
        p.drawText(QPointF(18, 49), f"N = {n}    f = {self.parameters.frequency_ghz:g} GHz    d = {self.parameters.spacing_mm:.5g} mm")
        p.drawText(QRectF(110, h-31, w-125, 21), Qt.AlignmentFlag.AlignRight,
                   f"参考网格（非地平面）   ·   视图缩放 {self.zoom:.0%}")
        p.end()


class PatternPlot(QWidget):
    cursor_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(7, 4), dpi=100, facecolor="white", layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setAccessibleName("方向图画布")
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setIconSize(self.toolbar.iconSize() * .85)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.toolbar)
        self.datasets = []
        self.selected = -1
        self.polar = False
        self.floor = -40.0
        self.cursor_artist = None
        self.axes = None
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.draw_sets([], -1)

    def draw_sets(self, datasets, selected, polar=False, floor=-40.0, view=None):
        self.datasets, self.selected, self.polar, self.floor = datasets, selected, polar, floor
        self.three_dimensional = view == "3d"
        clear_figure(self.figure)
        self.cursor_artist = None
        if self.three_dimensional and datasets and 0 <= selected < len(datasets):
            from .linear_plot import sphere_pattern, sphere_from_feed, draw_sphere
            data = datasets[selected]
            if data.result is not None:
                sphere = sphere_pattern(data.result.parameters)
            elif data.feed_report is not None:
                sphere = sphere_from_feed(data.feed_report, data.pattern_key)
            else:
                self.axes = self.figure.add_subplot(111); self.axes.set_axis_off()
                self.axes.text(.5, .5, "此记录仅含二维曲线\n缺少完整阵列或激励，无法生成三维方向图。", ha="center", va="center", transform=self.axes.transAxes)
                self.canvas.draw_idle(); return
            self.axes = draw_sphere(self.figure, sphere, chinese=True)
            self.draw_metrics(data)
            self.canvas.draw_idle(); return
        self.axes = self.figure.add_subplot(111, projection="polar" if polar else None)
        ax = self.axes
        if not datasets:
            ax.set_axis_off()
            ax.text(.5, .5, "设置参数后点击“计算”\n或读取方向图文件", transform=ax.transAxes,
                    ha="center", va="center", color="#7b8c9d", fontsize=13, linespacing=2)
            self.canvas.draw_idle()
            return
        colors = ["#087fa8", "#d77a28", "#6c5bc4", "#26916b", "#bc4774"]
        lo = min(-90, min(float(d.angles[0]) for d in datasets))
        hi = max(90, max(float(d.angles[-1]) for d in datasets))
        top = max(0.0, math.ceil(max(float(np.max(d.db)) for d in datasets) / 10) * 10)
        for i, data in enumerate(datasets):
            y = np.maximum(data.db, floor)
            ax.plot(np.deg2rad(data.angles) if polar else data.angles,
                    y - floor if polar else y, color=data.color or colors[i % len(colors)],
                    lw=2.1 if i == selected else 1.25, alpha=1 if i == selected else .62,
                    label=data.label)
        if polar:
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_thetamin(lo)
            ax.set_thetamax(hi)
            ax.set_ylim(0, top - floor)
            levels = np.linspace(floor, top, 5)
            ax.set_yticks(levels - floor)
            ax.set_yticklabels([])
            # Place radial labels inside the visible semicircle. Matplotlib's
            # automatic half-plane labels otherwise crowd the baseline.
            for value in levels[1:]:
                ax.text(math.radians(-45), value - floor - .6, f"{value:g}", fontsize=8,
                        ha="right", va="center", color="#516779",
                        bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": .8})
        else:
            ax.set_xlim(lo, hi)
            ax.set_ylim(floor, top + 2)
            ax.set_xlabel("θ (°)   从 +z 朝 +x 为正")
            ax.set_ylabel("方向图幅度 (dB)")
        ax.grid(True, color="#dce5ed", linewidth=.75)
        if 0 <= selected < len(datasets):
            r = datasets[selected].result
            if r is not None and len(r.peaks_deg) == 1:
                peak = r.peaks_deg[0]
                ax.plot([math.radians(peak) if polar else peak], [-floor if polar else 0],
                        "o", color="#d77a28", markersize=5, zorder=5)
                if r.half_power_deg:
                    a, b = r.half_power_deg
                    ax.plot(np.deg2rad([a, b]) if polar else [a, b],
                            [10 * math.log10(.5) - (floor if polar else 0)] * 2,
                            "--", color="#7a8b9a", lw=1)
        legend = ax.legend(loc="lower left" if not polar else "upper left", fontsize=8, framealpha=.92)
        for text in legend.get_texts():
            text.set_parse_math(False)
        ax.tick_params(labelsize=9)
        if 0 <= selected < len(datasets):
            self.draw_metrics(datasets[selected])
        self.canvas.draw_idle()

    def draw_metrics(self, data):
        from .linear_plot import linear_metrics, metric_lines
        from .pattern_metrics import draw_metric_footer
        self.current_metrics = linear_metrics(data.result, data.feed_report, data.pattern_key, data.angles, data.db,
                                              sphere=self.three_dimensional)
        if not self.three_dimensional and data.result is None:
            peak = self.current_metrics.get("main_lobe_angle_deg")
            level = self.current_metrics.get("peak_level_db")
            if peak is not None and level is not None:
                self.axes.plot([math.radians(peak) if self.polar else peak],
                               [level - self.floor if self.polar else level], "o", color="#d77a28", markersize=5)
                edges = self.current_metrics.get("half_power_edges_deg")
                if edges:
                    threshold = level + 10 * math.log10(.5)
                    self.axes.plot(np.deg2rad(edges) if self.polar else edges,
                                   [threshold - (self.floor if self.polar else 0)] * 2,
                                   "o--", color="#7a8b9a", lw=1, markersize=3)
        draw_metric_footer(self.figure, metric_lines(self.current_metrics, chinese=True,
                                                     sphere=self.three_dimensional, label=data.label))


    def _read(self, event):
        if self.three_dimensional or event.inaxes != self.axes or event.xdata is None or not 0 <= self.selected < len(self.datasets):
            return None
        theta = math.degrees(event.xdata) if self.polar else event.xdata
        if self.polar:
            theta = (theta + 180) % 360 - 180
        data = self.datasets[self.selected]
        if not data.angles[0] <= theta <= data.angles[-1]:
            return None
        value = float(pattern_db_at(theta, data.result.parameters)) if data.result else float(np.interp(theta, data.angles, data.db))
        return theta, value

    def on_motion(self, event):
        point = self._read(event)
        if point:
            theta, db = point
            self.cursor_changed.emit(f"所选曲线  θ = {theta:.3f}°    幅度 = {db:.3f} dB    点击可定位游标")

    def on_click(self, event):
        if event.button != 1 or self.toolbar.mode:
            return
        point = self._read(event)
        if point:
            theta, db = point
            if self.cursor_artist:
                self.cursor_artist.remove()
            self.cursor_artist = self.axes.axvline(math.radians(theta) if self.polar else theta,
                                                 linestyle="--", color="#b5692f", lw=1)
            self.canvas.draw_idle()
            self.cursor_changed.emit(f"游标已定位  θ = {theta:.3f}°    幅度 = {db:.3f} dB")
