"""Compatibility import; implementation: arraylab.application.verification.next_stage."""
import sys as _sys
import arraylab.application.verification.next_stage as _implementation
if __name__ == '__main__':
    raise SystemExit(_implementation.main())
else:
    _sys.modules[__name__] = _implementation
