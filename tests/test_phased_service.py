from arraylab.features.phased.tests import test_phased as _implementation
globals().update({name:value for name,value in vars(_implementation).items() if not name.startswith('__')})
