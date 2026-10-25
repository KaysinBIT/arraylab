"""Forwarding for historical modules split across explicit owners."""
import sys
from types import ModuleType
class _ForwardingModule(ModuleType):
    def _owner(self,name):
        extra=self._compat_extras.get(name)
        return extra() if extra is not None else self._compat_primary
    def __getattr__(self,name):return getattr(self._owner(name),name)
    def __setattr__(self,name,value):
        if name.startswith('__') or name.startswith('_compat_'):return super().__setattr__(name,value)
        owner=self._owner(name)
        if name in self._compat_extras or hasattr(owner,name):setattr(owner,name,value)
        else:super().__setattr__(name,value)
    def __delattr__(self,name):
        owner=self._owner(name)
        if name in self._compat_extras or hasattr(owner,name):delattr(owner,name)
        else:super().__delattr__(name)
    def __dir__(self):return sorted(set(super().__dir__())|set(dir(self._compat_primary))|set(self._compat_extras))
def forward_module(name,implementation,extras=None):
    module=sys.modules[name];module.__class__=_ForwardingModule
    ModuleType.__setattr__(module,'_compat_primary',implementation)
    ModuleType.__setattr__(module,'_compat_extras',extras or {})
    ModuleType.__setattr__(module,'__all__',[k for k in vars(implementation) if not k.startswith('_')]+list(extras or {}))
    return module
