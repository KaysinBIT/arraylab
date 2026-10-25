"""Public mimo API; implementation imports are lazy and owned by the feature."""

__all__ = ['MIMOPanel', 'MIMOParameters', 'calculate_mimo', 'is_mimo_record', 'load_mimo_configuration', 'main', 'mimo_field', 'mimo_geometry', 'save_mimo_configuration']


def __getattr__(name):
    if name == 'MIMOPanel':
        from .desktop import MIMOPanel
        return MIMOPanel
    if name == 'MIMOParameters':
        from .model import MIMOParameters
        return MIMOParameters
    if name == 'calculate_mimo':
        from .model import calculate_mimo
        return calculate_mimo
    if name == 'is_mimo_record':
        from .io import is_mimo_record
        return is_mimo_record
    if name == 'load_mimo_configuration':
        from .io import load_mimo_configuration
        return load_mimo_configuration
    if name == 'main':
        from .cli import main
        return main
    if name == 'mimo_field':
        from .model import mimo_field
        return mimo_field
    if name == 'mimo_geometry':
        from .model import mimo_geometry
        return mimo_geometry
    if name == 'save_mimo_configuration':
        from .io import save_mimo_configuration
        return save_mimo_configuration
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
