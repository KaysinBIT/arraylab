"""Desktop/headless composition, with all concrete factories explicit."""
from arraylab import __version__
from arraylab.framework.bootstrap import run,set_windows_application_id
def _headless(argv):
    from .cli import main
    return main(argv)
def _window(**kwargs):
    from .desktop.window import MainWindow
    return MainWindow(**kwargs)
def _icon():
    from arraylab.shared.qt.icons import application_icon
    return application_icon()
def _all(app,window,output):
    from .verification.full import schedule_smoke_test
    return schedule_smoke_test(app,window,output)
def _sparse(app,window,output):
    from .verification.suites import schedule_sparse_view_test
    return schedule_sparse_view_test(app,window,output)
def _mimo(app,window,output):
    from .verification.suites import schedule_mimo_test
    return schedule_mimo_test(app,window,output)
def _waveform(app,window,output):
    from .verification.suites import schedule_waveform_test
    return schedule_waveform_test(app,window,output)
def _navigation(app,window,output):
    from .verification.suites import schedule_navigation_test
    return schedule_navigation_test(app,window,output)
def _resize(app,window,output):
    from .verification.suites import schedule_resize_test
    return schedule_resize_test(app,window,output)
def _radar_integration(app,window,output):
    from .verification.suites import schedule_feature_test
    return schedule_feature_test(app,window,output,'radar-integration')
def main(argv=None):
    return run(argv,headless_main=_headless,window_factory=_window,icon_factory=_icon,version=__version__,
               smoke_schedulers={'all':_all,'sparse-view':_sparse,'mimo':_mimo,'waveform':_waveform,'navigation':_navigation,'resize':_resize,'radar-integration':_radar_integration})
_set_windows_application_id=set_windows_application_id
