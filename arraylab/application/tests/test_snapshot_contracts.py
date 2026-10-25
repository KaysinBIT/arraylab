"""Cross-feature handoffs use owned snapshots and destination-side policy."""
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np

from arraylab.features.linear.presenter import LinearAnalysisActions
from arraylab.features.sparse.desktop import SparsePanel
from arraylab.application.bridges.mimo import MIMOWorkflowActions


def test_linear_snapshot_owns_arrays():
    positions = np.array([-5., 5.])
    result = SimpleNamespace(positions_mm=positions,
                             parameters=SimpleNamespace(wavelength_mm=10., elements=2, frequency_ghz=29.9792458))
    host = SimpleNamespace(selected_dataset=lambda: SimpleNamespace(result=result, label='saved'))
    snapshot = LinearAnalysisActions.selected_linear_array_snapshot(host)
    np.testing.assert_array_equal(snapshot['positions_lambda'], [[-.5, 0], [.5, 0]])
    snapshot['positions_lambda'][:] = 100
    np.testing.assert_array_equal(positions, [-5., 5.])
    assert snapshot['element_model'] == 'isotropic'


def test_sparse_snapshot_preserves_source_model_without_destination_policy():
    coordinates = np.array([[0., 0.], [1., .5]])
    amplitudes = np.array([.25, .75])
    source = SimpleNamespace(_busy=False, thread=None,
        selected_result=lambda: {'xy': coordinates, 'weights': amplitudes, 'name': 'Saved layout'},
        bundle={'config': {'element_model': 'patch', 'reference_ghz': 10.},
                'scenarios': [{'frequency_ghz': 12.}]},
        case_selector=SimpleNamespace(currentIndex=lambda: 0))
    snapshot = SparsePanel.selected_array_snapshot(source)
    assert snapshot['element_model'] == 'patch'
    np.testing.assert_allclose(snapshot['positions_lambda'], coordinates * 1.2)
    snapshot['positions_lambda'][:] = 10; snapshot['weights'][:] = 0
    np.testing.assert_array_equal(coordinates, [[0., 0.], [1., .5]])
    np.testing.assert_array_equal(amplitudes, [.25, .75])


def test_application_rejects_unsupported_destination_without_changing_inputs(monkeypatch):
    import arraylab.application.bridges.mimo as bridge
    warning = Mock(); monkeypatch.setattr(bridge.QMessageBox, 'warning', warning)
    receiver = SimpleNamespace(_busy=False, thread=None, set_aperture=Mock())
    host = SimpleNamespace(mimo_panel=receiver, active_task=lambda: 'sparse',
                           sparse_panel=SimpleNamespace(selected_array_snapshot=lambda: {'element_model': 'patch'}),
                           activate_task=Mock())
    MIMOWorkflowActions.import_selected_to_mimo(host, 'tx')
    receiver.set_aperture.assert_not_called(); host.activate_task.assert_not_called()
    assert '只支持理想点源' in warning.call_args.args[2]
