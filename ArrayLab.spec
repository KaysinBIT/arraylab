# Desktop and headless programs share files, not startup code or GUI hooks.
from pathlib import Path
import re
import ast

ROOT = Path(SPECPATH)
version = re.search(r'__version__ = "([^"]+)"', (ROOT / "arraylab/__init__.py").read_text()).group(1)
common = ["tkinter", "PyQt5", "PyQt6", "PySide2", "pytest"]
sparse_resources = [(str(ROOT / "arraylab/sparse2d/resources"), "arraylab/sparse2d/resources")]
application_icons = ROOT / "arraylab/resources/icons"
legacy_tree = ast.parse((ROOT / 'arraylab/application/legacy_modules.py').read_text(encoding='utf-8'))
legacy_modules = next(ast.literal_eval(node.value) for node in legacy_tree.body
                      if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'LEGACY_MODULES' for target in node.targets))

desktop = Analysis(
    [str(ROOT / "launch.py")], pathex=[str(ROOT)],
    datas=sparse_resources + [(str(application_icons), "arraylab/resources/icons")],
    hiddenimports=[*legacy_modules, "PySide6.QtTest"],
    excludes=common, hooksconfig={"matplotlib": {"backends": ["QtAgg"]}},
)
headless = Analysis(
    [str(ROOT / "launch_cli.py")], pathex=[str(ROOT)],
    hiddenimports=[old for old, target in legacy_modules.items()
                   if not any(token in target for token in ('.verification.', '.desktop', '.presenter', '.shared.qt.', '.integration', '.application.bridges.', '.linear.layout', '.linear.widgets', '.feed.controls', '.feed.branch_widgets'))
                   and old not in ('arraylab.smoke_suites',)],
    excludes=common + ["PySide6", "arraylab.window", "arraylab.widgets", "arraylab.feed_widgets", "arraylab.electrical_widgets", "arraylab.planar_widgets", "arraylab.study_widgets", "arraylab.study_smoke", "arraylab.sparse_widgets", "arraylab.sparse_smoke", "arraylab.sparse2d.gui", "arraylab.sparse2d.detail_view", "arraylab.smoke", "arraylab.pointing_smoke", "arraylab.multicore_smoke", "arraylab.history_smoke", "arraylab.workflow_smoke",
        "arraylab.application.desktop", "arraylab.application.verification", "arraylab.application.bridges",
        "arraylab.shared.qt", "arraylab.features.linear.widgets", "arraylab.features.linear.presenter", "arraylab.features.linear.layout",
        "arraylab.features.feed.desktop", "arraylab.features.feed.controls", "arraylab.features.feed.branch_widgets",
        "arraylab.features.planar.desktop", "arraylab.features.sparse.desktop", "arraylab.features.sparse.navigation",
        "arraylab.features.subarray.desktop", "arraylab.features.mimo.desktop", "arraylab.features.waveform.desktop",
        "arraylab.features.studies.desktop", "arraylab.features.waveform.integration",
        "arraylab.features.linear.verification", "arraylab.features.feed.verification", "arraylab.features.planar.verification",
        "arraylab.features.sparse.verification", "arraylab.features.subarray.verification", "arraylab.features.mimo.verification",
        "arraylab.features.waveform.verification", "arraylab.features.studies.verification"],
    hooksconfig={"matplotlib": {"backends": ["Agg"]}},
)
gui = EXE(PYZ(desktop.pure), desktop.scripts, [], exclude_binaries=True,
          name="ArrayLab", console=False, upx=False, icon=str(application_icons / "arraylab.ico"))
cli = EXE(PYZ(headless.pure), headless.scripts, [], exclude_binaries=True,
          name="ArrayLab-cli", console=True, upx=False, icon=str(application_icons / "arraylab.ico"))
COLLECT(gui, cli, desktop.binaries, desktop.datas, headless.binaries, headless.datas,
        name=f"ArrayLab-{version}", upx=False)
