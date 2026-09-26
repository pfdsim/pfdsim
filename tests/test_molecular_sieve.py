import json
from pathlib import Path

import pytest

from adsorption_models import sieve_database
from molecular_sieve import MolecularSieveDryer
from pfd_parser import parse_pfd
from simulator import Simulator
from thermodynamics import create_thermodynamics
from unit_operations_base import UnitOperationError


def feed_for(composition, T=310., P=1.):
    thermo = create_thermodynamics(list(composition),'IDEAL')
    feed = thermo.calculate_state(T,P,1.,composition,phase='vapor',flash=False)
    return thermo,feed


def check_balance(feed,result):
    for c,z in feed.composition.items():
        assert sum(s.F*s.composition.get(c,0.) for s in result.outlet_streams.values()) == pytest.approx(feed.F*z,rel=1e-11,abs=1e-14)
    assert result.performance['equilibrium_balance_relative_residual'] < 2e-9


@pytest.mark.parametrize('sieve', ['4A','5A','13X'])
def test_defaults_automatically_adsorb_all_available_feed_species(sieve):
    thermo,feed = feed_for({'CO2':.2,'N2':.7,'CH4':.1})
    result = MolecularSieveDryer('MS',thermo,{'sieve_type':sieve,'adsorbent_mass_flow':80.}).solve({'feed':feed})
    check_balance(feed,result)
    assert set(result.performance['adsorbed_components']) == set(feed.composition)
    assert all(value>0 for value in result.performance['removed_kmol_h'].values())
    for c,q in result.performance['equilibrium_loadings_mol_per_kg'].items():
        assert result.performance['removed_kmol_h'][c] == pytest.approx(80*q/1000,rel=1e-7)


def test_custom_equation_coadsorbs_and_changes_target_sizing():
    thermo,feed = feed_for({'CO2':.2,'N2':.8})
    settings = {'CO2':{'model':'custom','expression':'q*b*f/(1+b*f)','parameters':{'q':4.,'b':10.}},
                'N2':{'model':'langmuir','qmax':4.,'b':.5}}
    params = {'sieve_type':'test sieve','isotherms':settings,'target_component':'CO2','target_mole_fraction':.08}
    result = MolecularSieveDryer('MS',thermo,params).solve({'feed':feed})
    check_balance(feed,result)
    assert result.outlet_streams['product'].composition['CO2'] == pytest.approx(.08,abs=1e-10)
    assert result.performance['removed_kmol_h']['N2'] > 0
    flow = result.performance['adsorbent_mass_flow_kg_h']
    replay = dict(params)
    replay.pop('target_mole_fraction')
    replay['adsorbent_mass_flow'] = flow
    rated = MolecularSieveDryer('MS',thermo,replay).solve({'feed':feed})
    assert rated.outlet_streams['product'].composition == pytest.approx(result.outlet_streams['product'].composition,rel=1e-8)


def test_removal_fraction_sizes_with_coadsorption():
    thermo,feed = feed_for({'CO2':.2,'N2':.8})
    result = MolecularSieveDryer('MS',thermo,{'sieve_type':'13X','target_component':'CO2','removal_fraction':.6}).solve({'feed':feed})
    check_balance(feed,result)
    assert result.performance['removed_kmol_h']['CO2'] == pytest.approx(.12,abs=1e-9)
    assert result.performance['removed_kmol_h']['N2'] > 0


def test_valid_isotherm_takes_precedence_over_nominal_pore_size():
    thermo,feed = feed_for({'water':.02,'N2':.98})
    result = MolecularSieveDryer('MS',thermo,{'sieve_type':'3A','adsorbent_mass_flow':2.,
        'isotherms':{'N2':{'qmax':3.,'b':1.}}}).solve({'feed':feed})
    check_balance(feed,result)
    assert result.performance['removed_kmol_h']['N2'] > 0
    assert result.performance['removed_kmol_h']['water'] > 0


def test_missing_isotherm_warnings_distinguish_size_exclusion():
    thermo,feed = feed_for({'H2':.1,'N2':.8,'water':.1})
    result = MolecularSieveDryer('MS',thermo,{'sieve_type':'3A','adsorbent_mass_flow':2.}).solve({'feed':feed})
    assert any('H2:' in w and 'isotherm is missing' in w for w in result.warnings)
    assert not any('N2:' in w for w in result.warnings)
    assert result.performance['removed_kmol_h']['H2'] == 0
    assert result.performance['removed_kmol_h']['N2'] == 0


def test_unknown_sizes_warn_and_explicit_diameter_enables_missing_curve_warning():
    thermo,feed = feed_for({'ethanol':.2,'water':.8})
    result = MolecularSieveDryer('MS',thermo,{'sieve_type':'custom','adsorbent_mass_flow':2.,
        'pore_diameter':5.,'kinetic_diameters':{'ethanol':4.3}}).solve({'feed':feed})
    assert any('ethanol:' in w and 'isotherm is missing' in w for w in result.warnings)
    check_balance(feed,result)


@pytest.mark.parametrize('params', [ {'sieve_type':'3A'}, {'adsorbent_mass_flow':float('nan')},
    {'adsorbent_mass_flow':-1.}, {'adsorbent_mass_flow':0.},
    {'adsorbent_mass_flow':2.,'target_mole_fraction':.1},
    {'adsorbent_mass_flow':2.,'isotherms':{'misspelled':{'qmax':1.,'b':1.}}},
    {'adsorbent_mass_flow':2.,'isotherms':{'water':{'model':'invalid'}}},
    {'target_component':'N2','target_mole_fraction':.1},
])
def test_invalid_specs_fail_explicitly(params):
    thermo,feed = feed_for({'N2':.9,'water':.1})
    with pytest.raises(UnitOperationError):
        MolecularSieveDryer('MS',thermo,params).solve({'feed':feed})


def test_nist_sources_and_fits_are_traceable():
    database = sieve_database()
    source = json.loads((Path(__file__).resolve().parents[1]/'data/source/molecular_sieve_nist.json').read_text())
    records = {r['filename']:r for r in source['records']}
    count = 0
    for sieve,curves in database['sieves'].items():
        for c,curve in curves.items():
            count += 1
            assert curve['points']>=4
            assert curve['rmse_mol_kg']>=0
            assert curve['mean_absolute_relative_error']<.25
            for record in curve['source_records']:
                if curve.get('source_provider') != 'NIST ISODB':
                    assert curve['source_file']=='source/molecular_sieve_ammonia.json'
                    continue
                assert records[record]['adsorbent']['name'] == f'Zeolite {sieve}'
                assert len(records[record]['adsorbates'])==1
    assert count >= 30
    assert database['excluded_records']


def test_single_temperature_default_is_not_assumed_valid_at_other_temperatures():
    thermo,feed = feed_for({'O2':.2,'N2':.8},T=350.)
    result = MolecularSieveDryer('MS',thermo,{'sieve_type':'13X','adsorbent_mass_flow':50.}).solve({'feed':feed})
    assert result.performance['removed_kmol_h']['O2'] == 0
    assert any('O2/13X' in w and 'not valid' in w for w in result.warnings)


def test_initial_loadings_close_solid_and_fluid_component_balances():
    thermo,feed = feed_for({'CO2':.2,'N2':.8})
    result = MolecularSieveDryer('MS',thermo,{'sieve_type':'custom','adsorbent_mass_flow':50.,
        'isotherms':{'CO2':{'qmax':4.,'b':5.},'N2':{'qmax':4.,'b':1.}},
        'initial_loadings':{'CO2':.001,'N2':.001}}).solve({'feed':feed})
    check_balance(feed,result)
    for c in feed.composition:
        loading_change = result.performance['final_loadings_kg_per_kg'][c]-.001
        assert 50*loading_change == pytest.approx(result.performance['removed_kmol_h'][c]*thermo.props[c].MW,rel=1e-10)


def test_preloading_above_inlet_loading_can_adsorb_at_outlet_equilibrium():
    thermo, feed = feed_for({'CO2':.2, 'N2':.8})
    initial_n2 = .2 * thermo.props['N2'].MW / 1000
    result = MolecularSieveDryer('MS', thermo, {
        'sieve_type':'custom', 'adsorbent_mass_flow':100.,
        'isotherms':{'CO2':{'qmax':4., 'b':100., 'heat':25000.},
                     'N2':{'qmax':4., 'b':1., 'heat':10000.}},
        'initial_loadings':{'N2':initial_n2},
    }).solve({'feed':feed})
    check_balance(feed, result)
    assert all(v > 0 for v in result.performance['removed_kmol_h'].values())
    assert result.performance['equilibrium_loadings_mol_per_kg']['N2'] > .2
    for c, initial in [('CO2',0.), ('N2',initial_n2)]:
        final = result.performance['final_loadings_kg_per_kg'][c]
        assert (final-initial)*100 == pytest.approx(
            result.performance['removed_kmol_h'][c]*thermo.props[c].MW, rel=1e-9)


@pytest.mark.parametrize('value,unit', [(1/36,'kg/s'), (100000.,'g/h'),
                                     (100/.45359237,'lb/h')])
def test_sieve_mass_flow_uses_shared_unit_conversion(value, unit):
    thermo, feed = feed_for({'CO2':.2, 'N2':.8})
    params = {'sieve_type':'13X', 'adsorbent_mass_flow':value,
              '__unit__adsorbent_mass_flow':unit}
    result = MolecularSieveDryer('MS', thermo, params).solve({'feed':feed})
    assert result.performance['adsorbent_mass_flow_kg_h'] == pytest.approx(100.)
    reference = MolecularSieveDryer('MS', thermo, {
        'sieve_type':'13X', 'adsorbent_mass_flow':100.,
    }).solve({'feed':feed})
    assert result.performance['removed_kmol_h'] == pytest.approx(
        reference.performance['removed_kmol_h'], rel=1e-9)


def test_rejects_second_liquid_phase_formed_during_contact(monkeypatch):
    from dataclasses import replace
    thermo, feed = feed_for({'CO2':.2, 'N2':.8})
    feed = replace(feed, vapor_fraction=.4)
    monkeypatch.setattr(thermo, 'calculate_state', lambda *args, **kwargs:
                        replace(feed, liquid2_fraction=.1))
    with pytest.raises(UnitOperationError, match='second liquid phase'):
        MolecularSieveDryer('MS', thermo, {
            'sieve_type':'13X', 'adsorbent_mass_flow':100.,
        }).solve({'feed':feed})


def test_dof_requires_exactly_one_adsorption_specification():
    from dof_analyzer import DOFAnalyzer, SpecificationStatus
    for extra,status in [('',SpecificationStatus.UNDER_SPECIFIED),
                         ('    sieve_flow = 20\n',SpecificationStatus.OK),
                         ('    sieve_flow = 20\n    removal_fraction = 0.5\n',SpecificationStatus.OVER_SPECIFIED)]:
        pfd = parse_pfd('UNIT MS : MolecularSieveDryer\n    sieve_type = 13X\n'+extra)
        assert DOFAnalyzer(pfd)._analyze_unit(pfd.units[0]).status == status


def test_competitive_example_runs_and_reports_balanced_adsorption():
    path = Path(__file__).resolve().parents[1]/'examples/competitive_13x_adsorption.pfd'
    simulator = Simulator.from_file(str(path))
    result = simulator.run()
    assert result.converged, result.errors
    check_balance(result.streams['Feed'],result.units['MS'])
    assert all(v>0 for v in result.units['MS'].performance['removed_kmol_h'].values())
    assert result.energy_balance_error < 1e-9
    assert result.units['MS'].unrepresented_enthalpy_change < 0
    assert 'unrepresented_enthalpy_change =' in simulator._generate_pfr()
    assert simulator.get_results_dict()['units']['MS']['unrepresented_enthalpy_change'] < 0


def test_physical_duty_includes_mixture_adsorption_and_initial_inventory():
    thermo,feed = feed_for({'CO2':.2,'N2':.8})
    heat = {'CO2':25000.,'N2':10000.}
    settings = {c:{'qmax':4.,'b':b,'heat':heat[c],'T_ref':feed.T}
                for c,b in [('CO2',5.),('N2',1.)]}
    result = MolecularSieveDryer('MS',thermo,{'sieve_type':'custom','adsorbent_mass_flow':50.,
        'isotherms':settings,'initial_loadings':{'CO2':.001,'N2':.001}}).solve({'feed':feed})
    expected_release = sum(result.performance['removed_kmol_h'][c]*heat[c] for c in heat)
    assert result.performance['adsorption_heat_release_kJ_h'] == pytest.approx(expected_release,rel=1e-9)
    assert result.heat_duty == pytest.approx(-expected_release,rel=1e-8)
    fluid_duty = sum(s.F*s.H for s in result.outlet_streams.values())-feed.F*feed.H
    assert result.heat_duty == pytest.approx(fluid_duty+result.unrepresented_enthalpy_change,rel=1e-10)
    assert result.to_dict()['unrepresented_enthalpy_change'] == result.unrepresented_enthalpy_change


def test_single_temperature_adsorption_marks_physical_heat_unknown():
    thermo,feed = feed_for({'CO2':.2,'N2':.8})
    result = MolecularSieveDryer('MS',thermo,{'sieve_type':'custom','adsorbent_mass_flow':50.,
        'isotherms':{'CO2':{'qmax':4.,'b':5.}}}).solve({'feed':feed})
    assert result.performance['bed_heat_duty_kJ_h'] is None
    assert result.performance['adsorption_enthalpy_missing_components'] == ['CO2']
    assert any('Physical bed heat duty is unavailable' in w for w in result.warnings)


@pytest.mark.parametrize('component,T',['Ne 77'.split(),'NH3 303.15'.split()])
def test_additional_3a_adsorbates(component,T):
    thermo,feed = feed_for({component:.2,'He':.8},T=float(T))
    result = MolecularSieveDryer('MS',thermo,{'sieve_type':'3A','adsorbent_mass_flow':100.}).solve({'feed':feed})
    check_balance(feed,result)
    assert result.performance['removed_kmol_h'][component] > 0
    assert result.performance['bed_heat_duty_kJ_h'] is None


def test_catalogue_uses_temperature_dependence_for_every_multitemperature_series():
    from adsorption_models import PureIsotherm
    root = Path(__file__).resolve().parents[1]
    records = {r['filename']:r for r in json.loads((root/'data/source/molecular_sieve_nist.json').read_text())['records']}
    corrections = json.loads((root/'data/source/molecular_sieve_corrections.json').read_text())['record_corrections']
    for curves in sieve_database()['sieves'].values():
        for setting in curves.values():
            if setting['source_provider'] != 'NIST ISODB':
                continue
            temperatures = {corrections.get(r,{}).get('temperature_K', records[r]['temperature'])
                            for r in setting['source_records']}
            assert setting['T_min'] == min(temperatures)
            assert setting['T_max'] == max(temperatures)
            iso = PureIsotherm(setting,(min(temperatures)+max(temperatures))/2)
            assert iso.enthalpy_available == (len(temperatures)>1)
            if len(temperatures)>1:
                assert setting['heat'] > 0


def test_methane_4a_temperature_correction_preserves_raw_records():
    from adsorption_models import PureIsotherm
    root = Path(__file__).resolve().parents[1]
    raw = {r['filename']:r for r in json.loads((root/'data/source/molecular_sieve_nist.json').read_text())['records']}
    settings = sieve_database()['sieves']['4A']['CH4']
    record = '10.1007s1045001698416.Isotherm20'
    assert raw[record]['temperature'] == 298
    assert settings['temperature_corrections'][record]['temperature_K'] == 272.990
    assert settings['T_min'] == 272.990
    assert settings['T_max'] == 322.989
    assert settings['mean_absolute_relative_error'] < .01
    assert PureIsotherm(settings,298.).enthalpy_available


def test_selected_models_have_validated_improvements():
    forms=set()
    for curves in sieve_database()['sieves'].values():
        for setting in curves.values():
            selection=setting['fit_selection']
            candidates=selection['candidates']
            chosen=candidates[setting['model']]
            assert chosen['eligible']
            errors=[v['validation_mare'] for v in candidates.values() if v['eligible']]
            assert chosen['validation_mare'] <= min(errors)*1.01+1e-10
            forms.add(setting['model'])
    assert {'sips','toth'} <= forms


def test_nested_custom_equations_survive_pfd_roundtrip_and_run():
    text = '''PROCESS: Custom adsorption
ONLINE_LOOKUP: false
THERMO_METHOD: IDEAL
COMPONENTS:
    CO2 | Carbon dioxide | MW=44.0095
    N2 | Nitrogen | MW=28.0134
STREAM Feed : FEED -> MS.feed
    T = 320 [K]
    P = 1 [bar]
    F = 1 [kmol/h]
    x = CO2:0.2, N2:0.8
STREAM Product : MS.product -> PRODUCT
STREAM Adsorbate : MS.adsorbate -> PRODUCT
UNIT MS : MolecularSieveDryer
    sieve_type = custom
    adsorbent_mass_flow = 30 [kg/h]
    isotherms = {CO2: {model: custom, expression: "q*b*f/(1+b*f)", parameters: {q: 4, b: 10}}, N2: {model: langmuir, qmax: 4, b: 0.5}}
    kinetic_diameters = {CO2: 3.3, N2: 3.64}
    initial_loadings = {CO2: 0, N2: 0}
'''
    pfd = parse_pfd(text)
    restored = parse_pfd(pfd.to_pfd())
    assert restored.units[0].params[-3].value == pfd.units[0].params[-3].value
    simulator = Simulator.from_string(text)
    result = simulator.run()
    assert result.converged, result.errors
    assert result.units['MS'].performance['removed_kmol_h']['N2'] > 0
