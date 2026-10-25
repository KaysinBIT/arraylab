"""Compatibility pytest entry; architecture contracts live with the framework."""
import arraylab.framework.tests.test_architecture_contracts as _test_module

globals().update({name: value for name, value in vars(_test_module).items() if not name.startswith('__')})
