from arraylab.features.pulse.tests import test_integration as _implementation
globals().update({name:value for name,value in vars(_implementation).items() if not name.startswith('__')})
