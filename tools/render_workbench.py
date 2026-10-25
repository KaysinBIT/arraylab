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
    window = MainWindow(auto_compute=False)
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
            window.resize(1440, 900)
            window.reset_parameters()
            window.ribbon.setCurrentIndex(1)
            window.show_view(4)
            QTest.qWait(100)
            assert window.grab().save(str(args.output / "feed-corporate.png"))
            feed = window.feed_panel
            feed.topology.setCurrentIndex(1); feed.generate()
            QTest.qWait(100)
            assert window.grab().save(str(args.output / "feed-series.png"))
            window.resize(1100, 740)
            QTest.qWait(100)
            assert window.grab().save(str(args.output / "feed-compact.png"))
            window.resize(1440, 900)
            feed.result_tabs.setCurrentIndex(1)
            QTest.qWait(100)
            assert window.grab().save(str(args.output / "feed-diagram-series.png"))
            feed.topology.setCurrentIndex(0); feed.generate()
            QTest.qWait(100)
            assert window.grab().save(str(args.output / "feed-diagram-corporate.png"))
            feed.topology.setCurrentIndex(2); feed.generate()
            QTest.qWait(100)
            assert window.grab().save(str(args.output / "feed-diagram-symmetric.png"))
            feed.diagram.fit(); QTest.qWait(100)
            assert window.grab().save(str(args.output / "feed-symmetric-overview.png"))
            feed.result_tabs.setCurrentIndex(0)
            QTest.qWait(100)
            assert window.grab().save(str(args.output / "feed-symmetric.png"))
            window.resize(1100, 740); QTest.qWait(100)
            assert window.grab().save(str(args.output / "feed-symmetric-compact.png"))
            window.resize(1440, 900)
            window.phase_mode.setCurrentIndex(1); window.inputs["scan_deg"].setText("30")
            feed.electrical.weighting.setCurrentIndex(feed.electrical.weighting.findData("taylor"))
            feed.electrical.fields["sidelobe_db"].setText("30")
            feed.generate(); feed.settings_tabs.setCurrentIndex(1)
            QTest.qWait(100)
            assert window.grab().save(str(args.output / "electrical-settings.png"))
            feed.result_tabs.setCurrentIndex(3)
            QTest.qWait(100); feed.pattern_plot.canvas.draw()
            assert window.grab().save(str(args.output / "electrical.png"))
            feed.electrical.fields["phase_bits"].setText("3")
            feed.electrical.fields["phase_error_deg"].setText("5")
            feed.electrical.fields["amplitude_error_db"].setText("0.5")
            feed.electrical.fields["seed"].setText("42")
            feed.electrical.failed.setText("3")
            feed.generate(); QTest.qWait(100); feed.pattern_plot.canvas.draw()
            assert window.grab().save(str(args.output / "electrical-errors.png"))
            window.resize(1100, 740); QTest.qWait(100); feed.pattern_plot.canvas.draw()
            assert window.grab().save(str(args.output / "electrical-compact.png"))
            window.resize(1440,900);window.show_view(5)
            planar=window.planar_panel
            planar.geometry_fields["scan_theta_deg"].setText("30")
            planar.geometry_fields["scan_phi_deg"].setText("45")
            planar.geometry_fields["cut_phi_deg"].setText("45")
            planar.generate();QTest.qWait(100)
            assert window.grab().save(str(args.output / "planar-geometry.png"))
            for view in ("cut","map","3d"):
                planar.view.setCurrentIndex(planar.view.findData(view));QTest.qWait(100);planar.canvas.draw()
                assert window.grab().save(str(args.output / f"planar-{view}.png"))
            window.resize(1100,740);QTest.qWait(100);planar.canvas.draw()
            assert window.grab().save(str(args.output / "planar-compact.png"))
            app.exit(0)
        except Exception:
            import traceback
            traceback.print_exc()
            app.exit(1)

    QTimer.singleShot(500, capture)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
