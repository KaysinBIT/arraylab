"""Compatibility pytest entry; tests live beside their implementation."""
import arraylab.features.mimo.tests.test_mimo_io as _test_module
globals().update({name: value for name, value in vars(_test_module).items() if not name.startswith("__")})
