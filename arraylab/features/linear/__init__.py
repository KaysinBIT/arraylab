"""Public linear API; implementation imports are lazy and owned by the feature."""

__all__ = ['ArrayParameters', 'ArrayResult', 'FIELDS', 'GeometryView', 'InputError', 'LinearAnalysisActions', 'MAX_BYTES', 'NumericTableModel', 'PatternPlot', 'PatternSet', 'RESULT_FIELDS', 'array_result_from_snapshot', 'array_result_snapshot', 'build_linear_parameters', 'build_linear_views', 'calculate', 'draw_sphere', 'linear_metrics', 'linear_record_bytes', 'linear_record_from_bytes', 'load_configuration', 'metric_lines', 'parameters_from_values', 'pattern_from_record', 'plain_linear_record', 'read_pattern', 'save_configuration', 'sphere_from_feed', 'sphere_pattern', 'validate_array_result', 'validate_linear_record', 'validate_parameters', 'write_pattern', 'write_sphere_csv']


def __getattr__(name):
    if name == 'ArrayParameters':
        from .model import ArrayParameters
        return ArrayParameters
    if name == 'ArrayResult':
        from .model import ArrayResult
        return ArrayResult
    if name == 'FIELDS':
        from .configuration import FIELDS
        return FIELDS
    if name == 'GeometryView':
        from .widgets import GeometryView
        return GeometryView
    if name == 'InputError':
        from .model import InputError
        return InputError
    if name == 'LinearAnalysisActions':
        from .presenter import LinearAnalysisActions
        return LinearAnalysisActions
    if name == 'MAX_BYTES':
        from .configuration import MAX_BYTES
        return MAX_BYTES
    if name == 'NumericTableModel':
        from .widgets import NumericTableModel
        return NumericTableModel
    if name == 'PatternPlot':
        from .widgets import PatternPlot
        return PatternPlot
    if name == 'PatternSet':
        from .widgets import PatternSet
        return PatternSet
    if name == 'RESULT_FIELDS':
        from .records import RESULT_FIELDS
        return RESULT_FIELDS
    if name == 'array_result_from_snapshot':
        from .records import array_result_from_snapshot
        return array_result_from_snapshot
    if name == 'array_result_snapshot':
        from .records import array_result_snapshot
        return array_result_snapshot
    if name == 'build_linear_parameters':
        from .layout import build_linear_parameters
        return build_linear_parameters
    if name == 'build_linear_views':
        from .layout import build_linear_views
        return build_linear_views
    if name == 'calculate':
        from .model import calculate
        return calculate
    if name == 'draw_sphere':
        from .plotting import draw_sphere
        return draw_sphere
    if name == 'linear_metrics':
        from .plotting import linear_metrics
        return linear_metrics
    if name == 'linear_record_bytes':
        from .records import linear_record_bytes
        return linear_record_bytes
    if name == 'linear_record_from_bytes':
        from .records import linear_record_from_bytes
        return linear_record_from_bytes
    if name == 'load_configuration':
        from .configuration import load_configuration
        return load_configuration
    if name == 'metric_lines':
        from .plotting import metric_lines
        return metric_lines
    if name == 'parameters_from_values':
        from .configuration import parameters_from_values
        return parameters_from_values
    if name == 'pattern_from_record':
        from .presenter import pattern_from_record
        return pattern_from_record
    if name == 'plain_linear_record':
        from .records import plain_linear_record
        return plain_linear_record
    if name == 'read_pattern':
        from .dataio import read_pattern
        return read_pattern
    if name == 'save_configuration':
        from .configuration import save_configuration
        return save_configuration
    if name == 'sphere_from_feed':
        from .plotting import sphere_from_feed
        return sphere_from_feed
    if name == 'sphere_pattern':
        from .plotting import sphere_pattern
        return sphere_pattern
    if name == 'validate_array_result':
        from .records import validate_array_result
        return validate_array_result
    if name == 'validate_linear_record':
        from .records import validate_linear_record
        return validate_linear_record
    if name == 'validate_parameters':
        from .records import validate_parameters
        return validate_parameters
    if name == 'write_pattern':
        from .dataio import write_pattern
        return write_pattern
    if name == 'write_sphere_csv':
        from .plotting import write_sphere_csv
        return write_sphere_csv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
