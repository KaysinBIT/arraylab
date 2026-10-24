"""Footer layout compatibility and bounded reuse without GUI timing assertions."""
from types import SimpleNamespace
import unicodedata

import pytest

from arraylab.pattern_metrics import _cached_footer_wrap, _footer_font_key, _wrap_footer_lines


def old_wrap(lines, available, text_width):
    """Independent copy of the original character-by-character formatter."""
    wrapped = []
    for original in lines:
        for line in str(original).splitlines() or [""]:
            pending = ""
            for character in line:
                if pending and text_width(pending + character) > available:
                    boundary = pending.rfind(" ")
                    if boundary > len(pending) // 2:
                        wrapped.append(pending[:boundary].rstrip())
                        pending = pending[boundary + 1:].lstrip() + character
                    else:
                        wrapped.append(pending.rstrip())
                        pending = character.lstrip()
                else:
                    pending += character
            wrapped.append(pending.rstrip())
    return tuple(wrapped)


def measured_width(text):
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in text)


@pytest.mark.parametrize("lines", [
    ["代表峰 θ=30°，φ=45°  |  切面 3 dB 宽度 8.34°", "峰值方向性 D 12.345 dBi"],
    ["Local great-circle HPBW (meridional / transverse) 2.34 / 4.56 deg"],
    ["Selected history: " + "abcdefghij" * 8],
    ["自定义长标签" * 30],
    ["", "a\n\nb", " \n  ", "尾部空格  "],
    ["  leading    spaced  words   ", "one         two three", "a b c d e"],
    ["x", "字", 123, "abcdefghijklmno"],
])
@pytest.mark.parametrize("available", [.1, 1, 2, 5, 11, 29, 80, 1000])
def test_binary_wrap_preserves_original_lines_and_space_rules(lines, available):
    assert _wrap_footer_lines(lines, available, measured_width) == old_wrap(lines, available, measured_width)


def test_bisection_uses_substantially_fewer_width_queries():
    lines = ["所选：" + "阵列优化结果 123.456 " * 12,
             "Peak directivity D 14.56 dBi  |  Gain G unavailable (efficiency missing)"]
    counts = {"old": 0, "new": 0}

    def width(kind):
        def measure(text):
            counts[kind] += 1
            return measured_width(text)
        return measure

    expected = old_wrap(lines, 140, width("old"))
    assert _wrap_footer_lines(lines, 140, width("new")) == expected
    assert counts["new"] < counts["old"] / 4


def test_figure_cache_is_bounded_lru_and_separate_between_figures():
    first, second = SimpleNamespace(), SimpleNamespace()
    calls = []

    def measure(text):
        calls.append(text)
        return measured_width(text)

    def wrap(figure, label, width=20, dpi=100, font=("sans",), renderer="Agg"):
        return _cached_footer_wrap(figure, [label], width, dpi, font, renderer, measure)

    result = wrap(first, "same content")
    initial = len(calls)
    assert wrap(first, "same content") is result
    assert len(calls) == initial
    assert wrap(second, "same content") == result
    assert len(calls) > initial
    for index in range(7):
        wrap(first, "entry " + str(index))
    assert len(first._arraylab_footer_wrap_cache) == 8
    wrap(first, "same content")
    wrap(first, "ninth entry")
    assert len(first._arraylab_footer_wrap_cache) == 8
    assert any(key[0] == ("same content",) for key in first._arraylab_footer_wrap_cache)
    assert not any(key[0] == ("entry 0",) for key in first._arraylab_footer_wrap_cache)


@pytest.mark.parametrize("changed", [
    {"lines": ["changed text"]}, {"available": 41}, {"dpi": 144},
    {"font_key": ("serif",)}, {"renderer_key": "fallback"},
])
def test_cache_invalidates_changed_text_width_dpi_font_or_renderer(changed):
    figure = SimpleNamespace()
    calls = []
    options = dict(lines=["normal text"], available=40, dpi=100,
                   font_key=("sans",), renderer_key="Agg")

    def measure(text):
        calls.append(text)
        return len(text)

    _cached_footer_wrap(figure, **options, text_width=measure)
    initial = len(calls)
    options.update(changed)
    _cached_footer_wrap(figure, **options, text_width=measure)
    assert len(calls) > initial


def test_font_signature_covers_font_changes():
    from matplotlib.font_manager import FontProperties

    font = FontProperties(size=8)
    original = _footer_font_key(font)
    for setter, value in [("set_family", "serif"), ("set_size", 9),
                          ("set_style", "italic"), ("set_weight", "bold"),
                          ("set_stretch", "condensed")]:
        changed = font.copy()
        getattr(changed, setter)(value)
        assert _footer_font_key(changed) != original


@pytest.mark.parametrize("dpi,available", [(100, 140), (100, 420), (144, 672)])
def test_actual_font_measurements_preserve_old_wrapping(dpi, available):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties

    figure = Figure(figsize=(7, 5), dpi=dpi)
    renderer = FigureCanvasAgg(figure).get_renderer()
    font = FontProperties(family=["Microsoft YaHei", "DejaVu Sans"], size=8)
    lines = ["所选：线阵扫描实验 " + "123.456 " * 12,
             "Peak directivity D 14.56 dBi  |  Gain G unavailable (efficiency missing)",
             "无空格中文标签" * 10, "", "LongUnspacedHistoryName" * 4]

    def measure(text):
        return renderer.get_text_width_height_descent(text, font, ismath=False)[0]

    assert _wrap_footer_lines(lines, available, measure) == old_wrap(lines, available, measure)


def test_actual_footer_reuses_width_layout_when_only_height_changes(monkeypatch):
    from matplotlib.backends.backend_agg import FigureCanvasAgg, RendererAgg
    from matplotlib.figure import Figure
    from arraylab.pattern_metrics import draw_metric_footer

    figure = Figure(figsize=(7, 4), dpi=100, layout="constrained")
    canvas = FigureCanvasAgg(figure)
    figure.add_subplot(111)
    original_measure = RendererAgg.get_text_width_height_descent
    calls = []

    def measure(renderer, text, *args, **kwargs):
        calls.append(text)
        return original_measure(renderer, text, *args, **kwargs)

    monkeypatch.setattr(RendererAgg, "get_text_width_height_descent", measure)
    lines = ["Selected: " + "abcdefghij" * 8,
             "Local great-circle HPBW (meridional / transverse) 2.34 / 4.56 deg"]
    first = draw_metric_footer(figure, lines)
    first_text, first_margin = first.get_text(), figure._arraylab_footer_reserved
    initial_calls = len(calls)
    assert initial_calls > 0
    resize_count = len(canvas.callbacks.callbacks.get("resize_event", {}))
    figure.set_size_inches(7, 8, forward=False)
    second = draw_metric_footer(figure, lines)
    assert second.get_text() == first_text
    assert len(calls) == initial_calls
    assert figure._arraylab_footer_reserved < first_margin
    assert len(figure.texts) == 1
    assert len(canvas.callbacks.callbacks.get("resize_event", {})) == resize_count
    figure.set_size_inches(4, 8, forward=False)
    narrow = draw_metric_footer(figure, lines)
    assert len(calls) > initial_calls
    assert len(narrow.get_text().splitlines()) > len(first_text.splitlines())
    previous_calls = len(calls)
    figure.set_dpi(144)
    draw_metric_footer(figure, lines)
    assert len(calls) > previous_calls
    assert len(figure.texts) == 1
    assert len(canvas.callbacks.callbacks.get("resize_event", {})) == resize_count


def test_actual_footer_invalidates_changed_default_font(monkeypatch):
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg, RendererAgg
    from matplotlib.figure import Figure
    from arraylab.pattern_metrics import draw_metric_footer

    figure = Figure(figsize=(5, 4), dpi=100)
    FigureCanvasAgg(figure)
    original_measure = RendererAgg.get_text_width_height_descent
    calls = []

    def measure(renderer, text, *args, **kwargs):
        calls.append(text)
        return original_measure(renderer, text, *args, **kwargs)

    monkeypatch.setattr(RendererAgg, "get_text_width_height_descent", measure)
    with matplotlib.rc_context({"font.family": ["DejaVu Sans"]}):
        draw_metric_footer(figure, ["Selected pattern: " + "abcdefghijk" * 5])
    initial_calls = len(calls)
    with matplotlib.rc_context({"font.family": ["DejaVu Sans Mono"]}):
        draw_metric_footer(figure, ["Selected pattern: " + "abcdefghijk" * 5])
    assert len(calls) > initial_calls
