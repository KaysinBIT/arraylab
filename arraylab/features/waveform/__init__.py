"""Public waveform API; implementation imports are lazy and owned by the feature."""

__all__ = ['attach_waveform', 'calculate_waveform', 'default_waveform_configuration', 'is_waveform_record', 'load_waveform_configuration', 'main', 'normalize_waveform_configuration', 'save_waveform_configuration', 'scene_payload', 'normalize_scene_state']


def __getattr__(name):
    if name in ('scene_payload', 'normalize_scene_state'):
        from . import scene
        return getattr(scene, name)
    if name == 'attach_waveform':
        from .integration import attach_waveform
        return attach_waveform
    if name == 'calculate_waveform':
        from .model import calculate_waveform
        return calculate_waveform
    if name == 'default_waveform_configuration':
        from .model import default_waveform_configuration
        return default_waveform_configuration
    if name == 'is_waveform_record':
        from .io import is_waveform_record
        return is_waveform_record
    if name == 'load_waveform_configuration':
        from .io import load_waveform_configuration
        return load_waveform_configuration
    if name == 'main':
        from .cli import main
        return main
    if name == 'normalize_waveform_configuration':
        from .model import normalize_waveform_configuration
        return normalize_waveform_configuration
    if name == 'save_waveform_configuration':
        from .io import save_waveform_configuration
        return save_waveform_configuration
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
