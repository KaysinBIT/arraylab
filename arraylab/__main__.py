import argparse
import os
from pathlib import Path
import sys
import tempfile


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Dispatch before importing Qt, the desktop window, or any GUI plotting.
    if "--headless" in argv:
        from .cli import main as cli_main
        return cli_main(argv)
    parser = argparse.ArgumentParser(description="ArrayLab desktop")
    parser.add_argument("--headless", action="store_true", help="Run without windows; use --headless --help for calculation options")
    parser.add_argument("--smoke-test", metavar="OUTPUT_DIR", help="Run the packaged Qt integration check and exit")
    args = parser.parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "arraylab-matplotlib"))
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from .window import MainWindow
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, True)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("ArrayLab")
    from . import __version__
    app.setApplicationVersion(__version__)
    app.setStyle("Fusion")
    window = MainWindow(auto_compute=not bool(args.smoke_test))
    screen = app.primaryScreen().availableGeometry()
    window.resize(min(1440, screen.width() - 60), min(900, screen.height() - 80))
    window.show()
    if args.smoke_test:
        from .smoke import schedule_smoke_test
        schedule_smoke_test(app, window, Path(args.smoke_test))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
