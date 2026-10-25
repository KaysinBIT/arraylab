from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from arraylab.shared.plotting.surface import clear_figure


def test_old_3d_axes_do_not_receive_later_mouse_events_even_when_referenced():
    figure = Figure()
    canvas = FigureCanvasAgg(figure)
    retained_axes = []
    events = ('motion_notify_event', 'button_press_event', 'button_release_event')
    initial = {event: len(canvas.callbacks.callbacks.get(event, {})) for event in events}
    observer = canvas.mpl_connect('motion_notify_event', lambda event: None)
    for _ in range(20):
        retained_axes.append(figure.add_subplot(111, projection='3d'))
        clear_figure(figure)
        for event in events:
            assert len(canvas.callbacks.callbacks.get(event, {})) == initial[event] + (event == 'motion_notify_event')
    assert observer in canvas.callbacks.callbacks['motion_notify_event']
    assert len(retained_axes) == 20  # Deliberately do not rely on garbage collection.
