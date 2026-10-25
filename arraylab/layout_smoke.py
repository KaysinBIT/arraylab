"""Compatibility import; implementation: arraylab.application.verification.layout."""
import sys as _sys
import arraylab.application.verification.layout as _implementation
if __name__ == '__main__':
    raise SystemExit(_implementation.main())
else:
    _sys.modules[__name__] = _implementation
