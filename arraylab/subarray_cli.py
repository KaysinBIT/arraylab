"""Compatibility import; implementation: arraylab.features.subarray.cli."""
import sys as _sys
import arraylab.features.subarray.cli as _implementation
if __name__ == '__main__':
    raise SystemExit(_implementation.main())
else:
    _sys.modules[__name__] = _implementation
