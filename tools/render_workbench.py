"""Capture the real Qt workbench at two layouts, without desktop capture APIs."""
import argparse
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "arraylab-matplotlib"))

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from arraylab.window import MainWindow


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, True)
    app = QApplication([]); app.setStyle("Fusion")
    window = MainWindow()
    window.show()

    def capture():
        try:
            window.resize(1440, 900)
            QTest.qWait(100)
            window.plot.canvas.draw()
            assert window.grab().save(str(args.output / "workbench.png"))
            window.show_view(3)
            QTest.qWait(100)
            assert window.grab().save(str(args.output / "model.png"))
            window.show_view(0)
            window.resize(1100, 740)
            QTest.qWait(100)
            window.plot.canvas.draw()
            assert window.grab().save(str(args.output / "compact.png"))
            app.exit(0)
        except Exception:
            import traceback
            traceback.print_exc()
            app.exit(1)

    QTimer.singleShot(500, capture)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
