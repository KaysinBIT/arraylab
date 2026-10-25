"""Separate read-only application resources from writable user output."""
import os
from pathlib import Path
import sys


def resource_path(name):
    from arraylab.framework.resources import resource_path as package_resource
    return package_resource('sparse2d', 'resources', name)


def data_directory():
    override = os.environ.get('ARRAYLAB_SPARSE_DATA_DIR')
    if override: return Path(override).expanduser().resolve()
    local = os.environ.get('LOCALAPPDATA')
    return (Path(local) if local else Path.home()) / 'ArrayLab' / 'Sparse2D'


def document_directory():
    return data_directory()
