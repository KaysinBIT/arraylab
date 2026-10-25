"""Compatibility entry for isolated radar-extension Qt tests."""
from arraylab.features.waveform.tests import test_radar_desktop as _implementation
globals().update({name: value for name, value in vars(_implementation).items() if not name.startswith('__')})
