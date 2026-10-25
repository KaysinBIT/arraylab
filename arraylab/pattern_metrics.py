"""Compatibility facade for the former mixed metric module."""
import arraylab.shared.plotting.metrics as _implementation
from arraylab.framework.compat import forward_module as _forward_module
def _planar_owner():
    import arraylab.features.planar.metrics as implementation
    return implementation
_forward_module(__name__,_implementation,{'planar_summary':_planar_owner})
