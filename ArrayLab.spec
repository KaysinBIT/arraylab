# Desktop and headless programs share files, not startup code or GUI hooks.
from pathlib import Path
import re

ROOT = Path(SPECPATH)
version = re.search(r'__version__ = "([^"]+)"', (ROOT / "arraylab/__init__.py").read_text()).group(1)
common = ["tkinter", "PyQt5", "PyQt6", "PySide2", "pytest"]
sparse_resources = [(str(ROOT / "arraylab/sparse2d/resources"), "arraylab/sparse2d/resources")]

desktop = Analysis(
    [str(ROOT / "launch.py")], pathex=[str(ROOT)],
    datas=sparse_resources,
    hiddenimports=["arraylab.smoke", "arraylab.study_smoke", "arraylab.sparse_smoke", "arraylab.pointing_smoke", "arraylab.multicore_smoke", "arraylab.history_smoke", "arraylab.workflow_smoke", "PySide6.QtTest"],
    excludes=common, hooksconfig={"matplotlib": {"backends": ["QtAgg"]}},
)
headless = Analysis(
    [str(ROOT / "launch_cli.py")], pathex=[str(ROOT)],
    excludes=common + ["PySide6", "arraylab.window", "arraylab.widgets", "arraylab.feed_widgets", "arraylab.electrical_widgets", "arraylab.planar_widgets", "arraylab.study_widgets", "arraylab.study_smoke", "arraylab.sparse_widgets", "arraylab.sparse_smoke", "arraylab.sparse2d.gui", "arraylab.sparse2d.detail_view", "arraylab.smoke", "arraylab.pointing_smoke", "arraylab.multicore_smoke", "arraylab.history_smoke", "arraylab.workflow_smoke"],
    hooksconfig={"matplotlib": {"backends": ["Agg"]}},
)
gui = EXE(PYZ(desktop.pure), desktop.scripts, [], exclude_binaries=True,
          name="ArrayLab", console=False, upx=False)
cli = EXE(PYZ(headless.pure), headless.scripts, [], exclude_binaries=True,
          name="ArrayLab-cli", console=True, upx=False)
COLLECT(gui, cli, desktop.binaries, desktop.datas, headless.binaries, headless.datas,
        name=f"ArrayLab-{version}", upx=False)
