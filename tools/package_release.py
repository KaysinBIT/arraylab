"""Create source and portable ZIPs from an explicit project-file allowlist."""
from pathlib import Path
import hashlib
import zipfile
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "release"
VERSION = re.search(r'__version__ = "([^"]+)"', (ROOT / "arraylab/__init__.py").read_text()).group(1)


def zip_files(target, paths, base, prefix):
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(paths):
            archive.write(path, str(Path(prefix) / path.relative_to(base)))


def source_files():
    """Return the explicit source allowlist without constructing an archive."""
    sources = [ROOT / p for p in ["README.md", "ROADMAP.md", "CONTRIBUTING.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "pyproject.toml", "requirements-lock.txt", ".gitignore", ".gitattributes", "launch.py", "launch_cli.py", "ArrayLab.spec", "启动 ArrayLab.cmd"]]
    for directory in ["arraylab", "tests", "tools"]:
        sources.extend((ROOT / directory).rglob("*.py"))
    sources.extend(p for p in (ROOT / "arraylab/sparse2d/resources").rglob("*") if p.is_file() and p.suffix in {".md", ".csv"})
    sources.extend(p for p in (ROOT / "arraylab/features/equation/data").rglob("*") if p.is_file() and p.suffix in {".json", ".md"})
    sources.extend(ROOT / "arraylab/resources/icons" / name for name in ("arraylab.png", "arraylab.ico"))
    sources.extend(p for p in (ROOT / "examples").iterdir() if p.is_file() and p.suffix in {".json", ".csv"})
    # Synthetic sample terrain only; do not collect arbitrary local GeoTIFFs.
    sources.append(ROOT / "examples/pulse-demo-hill-utm.tif")
    sources.extend((ROOT / ".github").rglob("*.yml"))
    sources.extend(p for p in (ROOT / "docs").rglob("*") if p.is_file() and p.suffix in {".md", ".png"})
    return sources


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    portable = ROOT / "dist" / f"ArrayLab-{VERSION}"
    assert (portable / "ArrayLab.exe").exists()
    zip_files(OUT / f"ArrayLab-{VERSION}-windows-x64.zip", [p for p in portable.rglob("*") if p.is_file()], portable, "ArrayLab")
    sources = source_files()
    zip_files(OUT / f"ArrayLab-{VERSION}-source.zip", sources, ROOT, "ArrayLab-source")
    hashes = []
    for p in sorted(OUT.glob("*.zip")):
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        hashes.append(f"{digest}  {p.name}")
        with zipfile.ZipFile(p) as archive:
            assert archive.testzip() is None
        print(p.name, p.stat().st_size)
    (OUT / "SHA256SUMS.txt").write_text("\n".join(hashes) + "\n", encoding="ascii")
    current = [line for line in hashes if f"  ArrayLab-{VERSION}-" in line]
    (OUT / f"SHA256SUMS-{VERSION}.txt").write_text("\n".join(current) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
