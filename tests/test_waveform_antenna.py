"""Compatibility test entry; implementation tests live beside the feature."""
from arraylab.features.waveform.tests import test_waveform_antenna as _implementation
globals().update({key: value for key, value in vars(_implementation).items() if not key.startswith('__')})
