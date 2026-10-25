"""Compatibility pytest entry; tests live beside their implementation."""
import arraylab.features.linear.tests.test_configuration as _test_module
globals().update({name: value for name, value in vars(_test_module).items() if not name.startswith("__")})
