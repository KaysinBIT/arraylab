"""Compatibility pytest entry; implementation is next to application bridges."""
import arraylab.application.tests.test_snapshot_contracts as _test_module
globals().update({name: value for name, value in vars(_test_module).items() if not name.startswith('__')})
