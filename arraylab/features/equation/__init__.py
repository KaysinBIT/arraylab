"""Public, Qt-free EquationLab 1.0 capability entrypoints."""
from .provenance import SOURCE_VERSION, SOURCE_ARCHIVE_SHA256

__all__ = ['SOURCE_VERSION', 'SOURCE_ARCHIVE_SHA256', 'default_configuration',
           'normalize_configuration', 'validate_configuration', 'calculate_budget', 'execute',
           'presets', 'apply_calibration', 'reset_array', 'save_configuration', 'load_configuration',
           'export_result', 'plot_data', 'draw', 'table', 'GROUPS', 'LABELS', 'CHOICES', 'HELP', 'ACTIONS', 'VIEWS', 'LIST_DEFAULTS']


def __getattr__(name):
    if name == 'calculate_budget':
        from .service import budget
        return budget
    if name in ('default_configuration', 'normalize_configuration', 'validate_configuration',
                'execute', 'presets', 'apply_calibration', 'reset_array'):
        from . import service
        return getattr(service, name)
    if name in ('save_configuration', 'load_configuration', 'export_result', 'plot_data'):
        from . import io
        return getattr(io, name)
    if name in ('draw', 'table'):
        from . import presentation
        return getattr(presentation, name)
    if name in ('GROUPS', 'LABELS', 'CHOICES', 'HELP', 'ACTIONS', 'VIEWS', 'LIST_DEFAULTS'):
        from . import metadata
        return getattr(metadata, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
