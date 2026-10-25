"""Stable package paths, independent of concrete business capabilities."""
from pathlib import Path
def package_root():return Path(__file__).resolve().parents[1]
def source_root():return package_root().parent
def resource_path(*parts):return package_root().joinpath(*parts)
