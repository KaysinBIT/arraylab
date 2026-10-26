"""Build a local, onedir Windows release with dependency license notices."""
from importlib import metadata
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import re

ROOT = Path(__file__).resolve().parents[1]
VERSION = re.search(r'__version__ = "([^"]+)"', (ROOT / "arraylab/__init__.py").read_text()).group(1)
RUNTIME = ["PySide6-Essentials", "shiboken6", "numpy", "scipy", "threadpoolctl", "matplotlib", "contourpy", "cycler", "fonttools", "kiwisolver", "packaging", "pillow", "pyparsing", "python-dateutil", "six", "tzdata", "pyinstaller", "psutil", "rasterio", "pyproj", "affine", "attrs", "certifi", "click"]


def main():
    # Qt uses Windows' system ICU. A developer PATH (e.g. Poppler/Conda)
    # can cause PyInstaller to bundle an ABI-incompatible icuuc.dll instead.
    # Use only this interpreter and Windows for external DLL discovery.
    env = os.environ.copy()
    windows = Path(env.get("SystemRoot", "C:/Windows"))
    env["PATH"] = os.pathsep.join(map(str, [Path(sys.executable).parent, Path(sys.base_prefix),
                                           Path(sys.base_prefix) / "DLLs", windows / "System32", windows]))
    env["PYTHONNOUSERSITE"] = "1"
    env["PYINSTALLER_CONFIG_DIR"] = str(ROOT / "tmp" / "pyinstaller-cache")
    subprocess.run([
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build" / VERSION),
        str(ROOT / "ArrayLab.spec"),
    ], cwd=ROOT, env=env, check=True)
    finish_release()


def finish_release():
    """Copy documentation and dependency notices after a full or incremental build."""
    release = ROOT / "dist" / f"ArrayLab-{VERSION}"
    if (release / "_internal" / "icuuc.dll").exists():
        raise RuntimeError("Unexpected bundled icuuc.dll; check the DLL provenance before shipping.")
    licenses = release / "licenses"
    licenses.mkdir(exist_ok=True)
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if python_license.exists():
        (licenses / "Python").mkdir(exist_ok=True)
        shutil.copy2(python_license, licenses / "Python" / "LICENSE.txt")
    manifest = {"app_version": VERSION, "python": sys.version, "packages": {}}
    for name in RUNTIME:
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        manifest["packages"][name] = dist.version
        for f in dist.files or []:
            # Only license metadata, never application data or user files.
            if ".dist-info" in str(f) and any(s in str(f).lower() for s in ("license", "copying", "notice")):
                source = Path(dist.locate_file(f))
                if source.is_file():
                    relative = Path(*f.parts[1:])
                    destination = licenses / name / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
    (release / "versions.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name in ["README.md", "LICENSE", "THIRD_PARTY_NOTICES.md"]:
        shutil.copy2(ROOT / name, release / name)
    shutil.copytree(ROOT / "examples", release / "examples", dirs_exist_ok=True)
    shutil.copytree(ROOT / "docs", release / "docs", dirs_exist_ok=True)
    shutil.copytree(ROOT / "arraylab/sparse2d/resources", release / "_internal/arraylab/sparse2d/resources", dirs_exist_ok=True)
    shutil.copy2(ROOT / "ROADMAP.md", release / "ROADMAP.md")
    shutil.copy2(ROOT / "CONTRIBUTING.md", release / "CONTRIBUTING.md")
    print(release / "ArrayLab.exe")


if __name__ == "__main__":
    main()
