"""Public planar API; implementation imports are lazy and owned by the feature."""

__all__ = ['PlanarPanel', 'PlanarParameters', 'design_planar', 'geometry_payload', 'load_planar_configuration', 'main', 'planar_amplitudes', 'planar_summary', 'save_planar_configuration', 'validate_planar_design']


def __getattr__(name):
    if name == 'PlanarPanel':
        from .desktop import PlanarPanel
        return PlanarPanel
    if name == 'PlanarParameters':
        from .model import PlanarParameters
        return PlanarParameters
    if name == 'design_planar':
        from .model import design_planar
        return design_planar
    if name == 'geometry_payload':
        from .model import geometry_payload
        return geometry_payload
    if name == 'load_planar_configuration':
        from .model import load_planar_configuration
        return load_planar_configuration
    if name == 'main':
        from .cli import main
        return main
    if name == 'planar_amplitudes':
        from .model import planar_amplitudes
        return planar_amplitudes
    if name == 'planar_summary':
        from .metrics import planar_summary
        return planar_summary
    if name == 'save_planar_configuration':
        from .model import save_planar_configuration
        return save_planar_configuration
    if name == 'validate_planar_design':
        from .model import validate_planar_design
        return validate_planar_design
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
