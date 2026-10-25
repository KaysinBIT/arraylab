"""Compatibility entry for scoped task-navigation tests."""
from arraylab.application.tests import test_navigation as _implementation
globals().update({name:value for name,value in vars(_implementation).items() if not name.startswith('__')})
