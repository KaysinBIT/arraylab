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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    portable = ROOT / "dist" / f"ArrayLab-{VERSION}"
    assert (portable / "ArrayLab.exe").exists()
    zip_files(OUT / f"ArrayLab-{VERSION}-windows-x64.zip", [p for p in portable.rglob("*") if p.is_file()], portable, "ArrayLab")
    sources = [ROOT / p for p in ["README.md", "ROADMAP.md", "CONTRIBUTING.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "pyproject.toml", "requirements-lock.txt", ".gitignore", ".gitattributes", "launch.py", "launch_cli.py", "ArrayLab.spec", "启动 ArrayLab.cmd"]]
    for directory in ["arraylab", "tests", "tools"]:
        sources.extend((ROOT / directory).rglob("*.py"))
    sources.extend(p for p in (ROOT / "examples").iterdir() if p.is_file() and p.suffix in {".json", ".csv"})
    sources.extend((ROOT / ".github").rglob("*.yml"))
    sources.extend(p for p in (ROOT / "docs").rglob("*") if p.is_file() and p.suffix in {".md", ".png"})
    zip_files(OUT / f"ArrayLab-{VERSION}-source.zip", sources, ROOT, "ArrayLab-source")
    hashes = []
    for p in sorted(OUT.glob("*.zip")):
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        hashes.append(f"{digest}  {p.name}")
        with zipfile.ZipFile(p) as archive:
            assert archive.testzip() is None
        print(p.name, p.stat().st_size)
    (OUT / "SHA256SUMS.txt").write_text("\n".join(hashes) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
