"""Compatibility collection entry for colocated waveform DOA tests."""
from arraylab.features.waveform.tests import test_doa as _implementation
globals().update({name: value for name, value in vars(_implementation).items() if not name.startswith('__')})
