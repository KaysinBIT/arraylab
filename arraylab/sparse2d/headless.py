"""Compatibility import: arraylab.features.sparse.engine.headless."""
import sys as _sys
import arraylab.features.sparse.engine.headless as _implementation
if __name__ == '__main__':
    raise SystemExit(_implementation.main())
else:
    _sys.modules[__name__] = _implementation
