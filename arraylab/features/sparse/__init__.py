"""Public sparse API; implementation imports are lazy and owned by the feature."""

__all__ = ['Config', 'SparseNavigationActions', 'SparsePanel', 'main', 'restore_bundle', 'run', 'save_bundle']


def __getattr__(name):
    if name == 'Config':
        from .engine.core import Config
        return Config
    if name == 'SparseNavigationActions':
        from .navigation import SparseNavigationActions
        return SparseNavigationActions
    if name == 'SparsePanel':
        from .desktop import SparsePanel
        return SparsePanel
    if name == 'main':
        from .cli import main
        return main
    if name == 'restore_bundle':
        from .engine.headless import restore_bundle
        return restore_bundle
    if name == 'run':
        from .engine.core import run
        return run
    if name == 'save_bundle':
        from .engine.headless import save_bundle
        return save_bundle
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
