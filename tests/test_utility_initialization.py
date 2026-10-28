"""Internal utility package construction and JIT preparation precede solving."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from chemical_properties import ChemicalDatabase
from simulator import SimulationError, Simulator
from thermodynamics import create_thermodynamics
from unit_operations_basic import Cooler, Heater
from utility_heat_transfer import prepare_thermal_utility

ROOT = Path(__file__).resolve().parents[1]


def test_simulator_prepares_all_fluid_utilities_without_solving_states(monkeypatch):
    import utility_heat_transfer
    original = utility_heat_transfer.create_thermodynamics
    created = []
    def create(*args, **kwargs):
        backend = original(*args, **kwargs)
        for name in ('calculate_state', 'calculate_state_PQ', 'calculate_state_PH', 'flash_TP'):
            monkeypatch.setattr(backend, name, lambda *a, **k: pytest.fail('Initialization solved a state'))
        created.append(backend)
        return backend
    monkeypatch.setattr(utility_heat_transfer, 'create_thermodynamics', create)
    sim = Simulator.from_file(ROOT/'examples/utility_surface_sizing.pfd')
    sim.initialize()
    fluid_units = ('H-STEAM', 'H-OIL', 'H-WATER', 'C-WATER', 'C-CHILLED', 'C-AIR', 'C-REFRIGERANT', 'C-GLYCOL')
    assert len(created) == len(fluid_units)
    assert all(backend._runtime_initialized for backend in created)
    for unit in fluid_units:
        assert len(sim.solver.units[unit]._utility_backends) == 1
    for unit in ('H-ELECTRIC', 'H-FIRED'):
        assert not getattr(sim.solver.units[unit], '_utility_backends', {})
    assert not sim.solver.streams and sim.result is None
    packages = [next(iter(sim.solver.units[u]._utility_backends.values())) for u in fluid_units]
    sim.initialize()
    assert len(created) == len(fluid_units)
    assert packages == [next(iter(sim.solver.units[u]._utility_backends.values())) for u in fluid_units]


def test_first_utility_solve_does_not_compile_or_recreate_packages(monkeypatch):
    pytest.importorskip('numba')
    from numba.core.registry import CPUDispatcher
    import utility_heat_transfer
    sim = Simulator.from_file(ROOT/'examples/utility_surface_sizing.pfd')
    sim.initialize()
    refrigerant = sim.solver.units['C-REFRIGERANT']
    backend = next(iter(refrigerant._utility_backends.values()))
    if backend.cubic._compiled_backend is None:
        pytest.skip('Compiled EOS backend unavailable')
    original = CPUDispatcher._compile_for_args
    compiles = []
    def record_compile(self, *args, **kwargs):
        compiles.append(self.py_func.__module__+'.'+self.py_func.__name__)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(CPUDispatcher, '_compile_for_args', record_compile)
    monkeypatch.setattr(utility_heat_transfer, 'create_thermodynamics',
                        lambda *args, **kwargs: pytest.fail('Utility package recreated during run'))
    result = sim.run()
    assert result.converged, result.errors
    assert not result.errors
    assert not compiles, compiles
    assert not any('fallback' in warning.lower() for warning in backend.warnings)


@pytest.mark.parametrize('unit_class,params,method,components', [
    (Heater, {'utility': 'steam'}, 'STEAM', ['water']),
    (Heater, {'utility': 'hot-oil', 'utility_fluid': 'decane'}, 'IDEAL', ['decane']),
    (Cooler, {'utility': 'air'}, 'IDEAL', ['N2', 'O2']),
    (Cooler, {'utility': 'refrigerant', 'utility_fluid': 'ammonia'}, 'PR', ['ammonia']),
    (Cooler, {'utility': 'refrigerant', 'utility_fluid': 'ammonia', 'utility_thermo_method': 'SRK'},
     'SRK', ['ammonia']),
    (Cooler, {'utility': 'thermal_fluid', 'utility_composition': {'water': .7, 'ethylene glycol': .3}},
     'IDEAL', ['water', 'ethylene glycol']),
])
def test_unit_and_sizing_share_the_same_utility_selection(monkeypatch, unit_class, params, method, components):
    import utility_heat_transfer
    created = []
    backend = SimpleNamespace(initialize=lambda: None, _resolver_known_props={})
    def create(selected_components, selected_method, **kwargs):
        created.append((selected_method, selected_components))
        return backend
    monkeypatch.setattr(utility_heat_transfer, 'create_thermodynamics', create)
    unit = unit_class('U', SimpleNamespace(), params)
    assert unit.initialize() is unit
    selected = prepare_thermal_utility(unit)
    assert selected[3] is backend
    assert created == [(method, components)]


@pytest.mark.parametrize('utility', [None, 'electric', 'fired'])
def test_nonfluid_or_absent_utility_needs_no_internal_package(monkeypatch, utility):
    import utility_heat_transfer
    monkeypatch.setattr(utility_heat_transfer, 'create_thermodynamics',
                        lambda *args, **kwargs: pytest.fail('Unexpected fluid package'))
    Heater('U', SimpleNamespace(), {'utility': utility}).initialize()


def test_process_utility_uses_the_existing_scoped_package(monkeypatch):
    import utility_heat_transfer
    backend = create_thermodynamics(['water'], 'STEAM', ChemicalDatabase(enable_online=False))
    monkeypatch.setattr(utility_heat_transfer, 'create_thermodynamics',
                        lambda *args, **kwargs: pytest.fail('Process package was replaced'))
    unit = Heater('U', backend, {'utility': 'hot_water', 'utility_thermo_method': 'process'})
    unit.initialize()
    assert prepare_thermal_utility(unit)[3] is backend
    assert backend._runtime_initialized


def test_process_utility_initialization_uses_the_units_declared_scope():
    sim = Simulator.from_string('''PROCESS: Scoped utility initialization
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: IDEAL
COMPONENTS:
    water | Water
THERMO_SCOPES:
    water_tables | method=STEAM
STREAM Feed : FEED -> H.in
    T = 300 [K]
    P = 3 [bar]
    F = 1 [kmol/h]
    x = water:1
STREAM Product : H.out -> PRODUCT
UNIT H : Heater
    T_out = 330 [K]
    thermo_scope = water_tables
    utility = hot_water
    utility_thermo_method = process
    utility_T_in = 370 [K]
    utility_T_out = 350 [K]
    U = 700 [W/m2-K]
''')
    sim.initialize()
    unit = sim.solver.units['H']
    backend = prepare_thermal_utility(unit)[3]
    assert backend is sim.thermo_packages['water_tables']
    assert backend is not sim.thermo
    assert backend._runtime_initialized


def test_simulator_reports_utility_preparation_errors_as_initialization_errors():
    source = (ROOT/'examples/utility_surface_sizing.pfd').read_text().replace(
        'utility_fluid = ammonia', 'utility_composition = {"ammonia": 0.5, "water": 0.5}')
    sim = Simulator.from_string(source)
    with pytest.raises(SimulationError, match='Failed to initialize.*pure fluid'):
        sim.initialize()
    assert sim.solver is None and not sim._initialized


@pytest.mark.parametrize('online', [False, True])
def test_all_internal_utilities_inherit_resolution_policy_before_initializing(monkeypatch, online):
    import utility_heat_transfer
    databases, initialized = [], []
    def create(components, method, db, **kwargs):
        databases.append(db.enable_online)
        snapshots = {c: {'name': f'utility identity {c}'} for c in components}
        def initialize():
            for known in snapshots.values():
                assert known['_allow_online_lookup'] is online
                assert known['_allow_computation'] is False
                assert known['_psat_minimum_pressure_bar'] == pytest.approx(1e-5)
                assert known['name'].startswith('utility identity ')
            initialized.append(method)
        return SimpleNamespace(initialize=initialize, _resolver_known_props=snapshots)
    monkeypatch.setattr(utility_heat_transfer, 'create_thermodynamics', create)
    source = (ROOT/'examples/utility_surface_sizing.pfd').read_text()
    source = source.replace('ONLINE_LOOKUP: false', f'ONLINE_LOOKUP: {str(online).lower()}\n'
                            'ALLOW_COMPUTATION: false\nPSAT_MINIMUM_PRESSURE: 1 [Pa]')
    sim = Simulator.from_string(source)
    sim.initialize()
    assert databases == [online]*8
    assert len(initialized) == 8
    # The process package is never replaced by the utility constructor stub.
    assert sim.thermo.components == ['water']
    assert sim.thermo._property_resolution_policy['_allow_online_lookup'] is online


def test_standalone_utility_is_offline_in_database_and_property_resolver():
    thermo = create_thermodynamics(['water'], 'STEAM', ChemicalDatabase(enable_online=False))
    unit = Heater('U', thermo, {'utility': 'hot_oil', 'utility_fluid': 'decane'})
    unit.initialize()
    backend = prepare_thermal_utility(unit)[3]
    assert backend.db.enable_online is False
    assert backend._allow_online_lookup_for_component('decane') is False


def test_changing_parent_resolution_policy_does_not_reuse_wrong_utility_package():
    thermo = create_thermodynamics(['water'], 'STEAM', ChemicalDatabase(enable_online=False))
    unit = Heater('U', thermo, {'utility': 'hot_oil', 'utility_fluid': 'decane'})
    thermo._property_resolution_policy = {'_allow_online_lookup': False}
    offline = prepare_thermal_utility(unit)[3]
    thermo._property_resolution_policy = {'_allow_online_lookup': True}
    online = prepare_thermal_utility(unit)[3]
    assert online is not offline
    assert online.db.enable_online is True
    assert online._allow_online_lookup_for_component('decane') is True
    assert offline._allow_online_lookup_for_component('decane') is False


def test_offline_utility_curves_are_reused_across_fresh_simulations(monkeypatch, tmp_path):
    import property_resolver
    from property_resolution.resolver import PropertyResolver
    resolver = PropertyResolver()
    resolver.CANONICAL_PSAT_CACHE_PATH = tmp_path/'canonical.sqlite'
    resolver.SATURATION_PROPERTIES_CACHE_PATH = tmp_path/'saturation.sqlite'
    monkeypatch.setattr(property_resolver, 'get_property_resolver', lambda: resolver)
    for name in ('_prime_nist_antoine_cache', '_fetch_antoine_nist_rows', '_fetch_antoine_pubchem'):
        monkeypatch.setattr(resolver, name, lambda *args, **kwargs: pytest.fail('Offline utility tried an online provider'))
    builds, flags = [], []
    original_build = resolver._build_canonical_vapor_pressure_runtime
    original_resolve = resolver.resolve_vapor_pressure_coefficients
    def build(symbol, props, **options):
        builds.append(symbol)
        return original_build(symbol, props, **options)
    def resolve(symbol, props=None, allow_online=True, **options):
        flags.append(allow_online)
        return original_resolve(symbol, props, allow_online=allow_online, **options)
    monkeypatch.setattr(resolver, '_build_canonical_vapor_pressure_runtime', build)
    monkeypatch.setattr(resolver, 'resolve_vapor_pressure_coefficients', resolve)
    previous = None
    for repeat in range(2):
        sim = Simulator.from_file(ROOT/'examples/utility_surface_sizing.pfd')
        sim.initialize()
        result = sim.run()
        assert result.converged, result.errors
        assert not result.errors
        assert flags and not any(flags)
        for unit in sim.solver.units.values():
            for backend in getattr(unit, '_utility_backends', {}).values():
                assert backend.db.enable_online is False
                assert all(not backend._allow_online_lookup_for_component(c) for c in backend.components)
        if repeat == 0:
            assert '124-18-5' in builds and '107-21-1' in builds
            previous = list(builds)
        else:
            assert builds == previous, 'Fresh utility snapshots caused canonical curves to rebuild'
    # A new resolver must load the same curves from disk, not merely benefit
    # from the original resolver's process-local dictionary.
    fresh = PropertyResolver()
    fresh.CANONICAL_PSAT_CACHE_PATH = resolver.CANONICAL_PSAT_CACHE_PATH
    fresh.SATURATION_PROPERTIES_CACHE_PATH = resolver.SATURATION_PROPERTIES_CACHE_PATH
    monkeypatch.setattr(fresh, '_build_canonical_vapor_pressure_runtime',
                        lambda *args, **kwargs: pytest.fail('Persisted offline curve was not reused'))
    for name in ('_prime_nist_antoine_cache', '_fetch_antoine_nist_rows', '_fetch_antoine_pubchem'):
        monkeypatch.setattr(fresh, name, lambda *args, **kwargs: pytest.fail('Offline utility tried an online provider'))
    monkeypatch.setattr(property_resolver, 'get_property_resolver', lambda: fresh)
    simulation = Simulator.from_file(ROOT/'examples/utility_surface_sizing.pfd')
    simulation.initialize()
    result = simulation.run()
    assert result.converged, result.errors
    assert not result.errors
