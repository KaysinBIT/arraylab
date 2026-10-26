from arraylab.application.tests import test_simulation_desktop as _implementation
globals().update({name:value for name,value in vars(_implementation).items() if not name.startswith('__')})
