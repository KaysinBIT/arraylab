"""Stable public CLI import and executable entrypoint."""
import sys as _sys
import arraylab.application.cli as _implementation
if __name__=='__main__':raise SystemExit(_implementation.main())
else:_sys.modules[__name__]=_implementation
