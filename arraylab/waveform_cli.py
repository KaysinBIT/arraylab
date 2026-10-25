"""Compatibility import; implementation: arraylab.features.waveform.cli."""
import sys as _sys
import arraylab.features.waveform.cli as _implementation
if __name__ == '__main__':
    raise SystemExit(_implementation.main())
else:
    _sys.modules[__name__] = _implementation
