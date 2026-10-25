"""Public subarray API; implementation imports are lazy and owned by the feature."""

__all__ = ['SubarrayConfig', 'attach_subarray', 'export_subarray', 'load_config', 'main', 'optimize_subarrays', 'save_config']


def __getattr__(name):
    if name == 'SubarrayConfig':
        from .model import SubarrayConfig
        return SubarrayConfig
    if name == 'attach_subarray':
        from .desktop import attach_subarray
        return attach_subarray
    if name == 'export_subarray':
        from .service import export_subarray
        return export_subarray
    if name == 'load_config':
        from .service import load_config
        return load_config
    if name == 'main':
        from .cli import main
        return main
    if name == 'optimize_subarrays':
        from .service import optimize_subarrays
        return optimize_subarrays
    if name == 'save_config':
        from .service import save_config
        return save_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
