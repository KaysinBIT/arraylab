"""Public native PulseLab integration; no web server or source directory needed."""
from .provenance import SOURCE_VERSION,SOURCE_ARCHIVE_SHA256
from . import metadata
from .metadata import ACTIONS,VIEWS,GROUPS,LABELS,CHOICES,HELP,LIST_DEFAULTS

__all__ = ['default_configuration','normalize_configuration','validate_configuration','presets',
           'execute','import_project','export_project','export_result','display_result','scene_preview','scene_payload','SceneWidget',
           'import_dem','acquisition_plan','beam_power_gain','element_positions','apply_inverse_candidate','array_configuration','draw','table',
           'metadata','ACTIONS','VIEWS','GROUPS','LABELS','CHOICES','HELP','LIST_DEFAULTS']

def __getattr__(name):
    if name in ('default_configuration','normalize_configuration','validate_configuration','presets','execute','apply_inverse_candidate','array_configuration'):
        from . import service
        return getattr(service,name)
    if name in ('import_project','export_project','export_result'):
        from . import io
        return getattr(io,name)
    if name in ('display_result','scene_preview','scene_payload','draw','table'):
        from . import presentation
        return getattr(presentation,name)
    if name=='SceneWidget':
        from .desktop_scene import SceneWidget
        return SceneWidget
    if name=='import_dem':
        from .scene import import_dem
        return import_dem
    if name=='beam_power_gain':
        from .scene import beam_power_gain
        return beam_power_gain
    if name=='element_positions':
        from .scene import element_positions
        return element_positions
    if name=='acquisition_plan':
        from .config import acquisition_plan
        return acquisition_plan
    raise AttributeError(name)
