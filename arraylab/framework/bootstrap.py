"""Qt application lifecycle with concrete factories supplied by composition."""
import argparse
import os
from pathlib import Path
import sys
import tempfile


def set_windows_application_id():
    """Preserve the established Windows taskbar identity across entrypoints."""
    if sys.platform != 'win32':
        return
    import ctypes
    try:
        setter = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
        setter.argtypes = [ctypes.c_wchar_p]
        setter.restype = ctypes.c_long
        setter('KaysinBIT.ArrayLab')
    except (AttributeError, OSError):
        pass


def run(argv=None, *, headless_main, window_factory, icon_factory, version,
        smoke_schedulers, application_name='ArrayLab'):
    """Start a desktop app, or dispatch headless before importing any Qt code.

    The application owns all concrete business dependencies. Factories and
    scheduler callbacks are injected explicitly; no service container is used.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--headless' in argv:
        return headless_main(argv)

    parser = argparse.ArgumentParser(description='ArrayLab desktop')
    parser.add_argument('--headless', action='store_true',
                        help='Run without windows; use --headless --help for calculation options')
    parser.add_argument('--smoke-test', metavar='OUTPUT_DIR',
                        help='Run the packaged Qt integration check and exit')
    parser.add_argument('--smoke-suite', choices=tuple(smoke_schedulers),
                        help='Desktop check scope (default: all); requires --smoke-test')
    args = parser.parse_args(argv)
    if args.smoke_suite and not args.smoke_test:
        parser.error('--smoke-suite requires --smoke-test OUTPUT_DIR')
    suite = args.smoke_suite or 'all'
    if args.smoke_test and suite not in smoke_schedulers:
        parser.error('No default desktop check suite is registered')

    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'arraylab-matplotlib'))
    set_windows_application_id()
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, True)
    app = QApplication(sys.argv[:1])
    app.setApplicationName(application_name)
    app.setWindowIcon(icon_factory())
    app.setApplicationVersion(version)
    app.setStyle('Fusion')
    window = window_factory(auto_compute=not bool(args.smoke_test))
    screen = app.primaryScreen().availableGeometry()
    window.resize(min(1440, screen.width() - 60), min(900, screen.height() - 80))
    window.show()
    if args.smoke_test:
        smoke_schedulers[suite](app, window, Path(args.smoke_test))
    return app.exec()
