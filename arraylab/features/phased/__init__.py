"""Public phased-array synthesis service; no GUI import side effects."""
from . import metadata
ACTIONS=metadata.ACTIONS
VIEWS=metadata.VIEWS
__all__=['default_configuration','normalize_configuration','validate_configuration','presets','execute','array_snapshot','apply_selected_case','record_compatibility','main','metadata','ACTIONS','VIEWS','draw','table','selections','export_result','load_configuration','save_configuration']

def __getattr__(name):
    if name in ('default_configuration','normalize_configuration','validate_configuration','presets','execute','array_snapshot','apply_selected_case','record_compatibility'):
        from . import service
        return getattr(service,name)
    if name=='main':
        from .cli import main
        return main
    if name in ('draw','table','selections'):
        from . import presentation
        return getattr(presentation,name)
    if name in ('export_result','load_configuration','save_configuration'):
        from . import io
        return getattr(io,name)
    raise AttributeError(name)
