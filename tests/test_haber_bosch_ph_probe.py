"""Workload capture must include RHS requests and avoid nested fallback counts."""

from types import SimpleNamespace
from unittest.mock import patch

from scripts.performance import probe_haber_bosch_ph as probe


def test_capture_includes_temperature_requests_and_restores_methods():
    solver_type = probe._ThermoStateSolver
    solver = solver_type(SimpleNamespace(components=['N2']), 'PFR axial state')

    def state(*args, **kwargs):
        return SimpleNamespace(T=600.0), 0.0

    def temperature(self, *args, **kwargs):
        result, residual = self.state_at_enthalpy(*args, **kwargs)
        return result.T, residual

    def run():
        request = (100.0, 1.0, {'N2': 1.0}, 1000.0, 500.0)
        solver.temperature_at_enthalpy(*request, force_phase='vapor')
        solver.state_at_enthalpy(*request, force_phase='vapor')
        return SimpleNamespace(
            converged=True, errors=[], iterations=1, mass_balance_error=0.0,
        )

    simulator = SimpleNamespace(initialize=lambda: None, run=run)
    with (
        patch.object(solver_type, 'state_at_enthalpy', state),
        patch.object(solver_type, 'temperature_at_enthalpy', temperature),
        patch.object(probe.Simulator, 'from_file', return_value=simulator),
    ):
        summary, records = probe.capture_workload()
        assert solver_type.state_at_enthalpy is state
        assert solver_type.temperature_at_enthalpy is temperature
    assert summary['pfr_ph_requests'] == 2
    assert [row['temperature_only'] for row in records] == [True, False]
    assert records[0]['previous_seed'] is None
    assert records[1]['previous_seed'] == 600.0
