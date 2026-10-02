"""Coupled top-liquid withdrawal, conservation, Jacobian, and parsing coverage."""

import math
import unittest

import numpy as np
import pytest

from equilibrium_stage_vlle import EquationOrientedVLLEColumn, VLLEProfile, VLLEStabilityCache
from pfd_parser import ParseError, parse_pfd
from thermodynamics import create_thermodynamics
from unit_operations_distillation import RigorousDistillation, CMODistillation, McCabeThieleDistillation, UnitOperationError


def butanol_column(**overrides):
    thermo = create_thermodynamics(['butanol', 'water'], 'NRTL')
    feed = thermo.calculate_state(298.15, 1., 100., {'butanol':.4,'water':.6},
                                  phase='liquid', flash=False)
    params = {'N_stages':20, 'feed_stage':10, 'D_to_F':.6118043106570895,
              'P_condenser':1., 'P_drop_per_stage':0., 'condenser_type':'total',
              'stage_phase_model':'VLLE', 'mesh_tolerance':1e-7,
              'max_iterations':80, 'max_jacobian_evaluations':80,
              'distillate_liquid1_component':'butanol',
              'distillate_liquid1_fraction':0., 'distillate_liquid2_fraction':1.}
    params.update(overrides)
    return RigorousDistillation('BUT', thermo, params), feed


def assert_balanced(unit, feed, result):
    products = [result.outlet_streams['distillate'], result.outlet_streams['bottoms']]
    for component in unit.thermo.components:
        assert sum(s.F*s.composition[component] for s in products) == pytest.approx(
            feed.F*feed.composition[component], abs=1e-5)
    assert sum(s.F*s.H for s in products)-feed.F*feed.H == pytest.approx(result.heat_duty, abs=.01)
    assert result.performance['vlle_max_log_fugacity_residual'] < 1e-6


class VLLEOverheadRoutingTests(unittest.TestCase):
    def test_removed_decanter_syntax_fails_in_default_runner(self):
        for value in ('decanter','heterogeneous','heterogeneous_decanter','top_decanter'):
            for compact in (False,True):
                with self.subTest(condenser=value,compact=compact):
                    test_removed_decanter_fails_during_parsing_with_vlle_guidance(value,compact)

    def test_aqueous_withdrawal_recovers_butanol_without_external_recycle(self):
        unit,feed = butanol_column()
        result = unit.solve({'feed':feed})
        bottoms = result.outlet_streams['bottoms']
        assert bottoms.composition['butanol'] > .995
        assert bottoms.F*bottoms.composition['butanol']/40. > .96
        routing = result.performance['top_liquid_routing']
        assert routing['distillate_liquid1_flow'] == 0.
        assert routing['reflux_liquid2_flow'] == 0.
        assert routing['liquid1_composition']['butanol'] > routing['liquid2_composition']['butanol']
        assert routing['distillate_liquid2_flow'] == pytest.approx(result.outlet_streams['distillate'].F)
        assert result.performance['jacobian_method'] == 'vlle_semi_analytic_local_thermo'
        assert result.performance['liquid_phase_routing'] == 'phase_selective_overhead'
        performance = result.performance
        phase1_is_rich = (performance['stage_liquid1_compositions'][0]['butanol']
                         >performance['stage_liquid2_compositions'][0]['butanol'])
        rich_flows = performance['stage_liquid1_flows' if phase1_is_rich else 'stage_liquid2_flows']
        aqueous_flows = performance['stage_liquid2_flows' if phase1_is_rich else 'stage_liquid1_flows']
        assert aqueous_flows[0] == 0.
        assert rich_flows[0] == pytest.approx(performance['liquid_flows'][0])
        assert set(result.outlet_streams) == {'distillate','bottoms'}
        assert_balanced(unit,feed,result)

    def test_equal_withdrawal_fractions_reproduce_co_routed_reflux(self):
        unit,feed = butanol_column(D_to_F=.773, distillate_liquid1_fraction=1/2.2,
                                  distillate_liquid2_fraction=1/2.2, vlle_seed='cheap')
        result = unit.solve({'feed':feed})
        assert result.performance['reflux_ratio'] == pytest.approx(1.2, rel=1e-6)
        assert result.outlet_streams['bottoms'].composition['butanol'] > .995
        distillate = result.outlet_streams['distillate']
        assert distillate.liquid2_fraction > 0
        assert distillate.effective_liquid1_fraction > 0
        assert distillate.x1 != distillate.x2
        for c in unit.thermo.components:
            assert distillate.composition[c] == pytest.approx(
                distillate.effective_liquid1_fraction*distillate.x1[c]
                +distillate.liquid2_fraction*distillate.x2[c])
        assert_balanced(unit,feed,result)

    def test_phase_selective_mass_spec_uses_actual_product_composition(self):
        unit,feed = butanol_column()
        xD = .02119778742844989
        target = 61.18043106570895*(xD*unit.thermo.props['butanol'].MW
                                   +(1-xD)*unit.thermo.props['water'].MW)
        unit.params.pop('D_to_F')
        unit.params['distillate_mass_flow'] = target
        result = unit.solve({'feed':feed})
        assert result.outlet_streams['distillate'].mass_flow() == pytest.approx(target, rel=1e-6)
        assert result.outlet_streams['bottoms'].composition['butanol'] > .995
        assert_balanced(unit,feed,result)

    def test_reversing_component_selector_and_fraction_labels_preserves_physical_routing(self):
        unit,feed = butanol_column(distillate_liquid1_component='water',
                                  distillate_liquid1_fraction=1.,distillate_liquid2_fraction=0.)
        result = unit.solve({'feed':feed})
        assert result.outlet_streams['bottoms'].composition['butanol'] > .995
        assert result.outlet_streams['bottoms'].F*result.outlet_streams['bottoms'].composition['butanol']/40 > .96
        routing = result.performance['top_liquid_routing']
        assert routing['liquid1_composition']['water'] > routing['liquid2_composition']['water']
        assert routing['distillate_liquid2_flow'] == 0.
        assert routing['reflux_liquid1_flow'] == 0.
        assert_balanced(unit,feed,result)

    def test_mixed_condenser_retains_routed_liquid_and_vapor_inventory(self):
        vapor_fraction = .25
        product_butanol = (1-vapor_fraction)*.02119778742844989+vapor_fraction*.2245933737319081
        cut = (.997-.4)/(.997-product_butanol)
        unit,feed = butanol_column(condenser_type='mixed',distillate_vapor_fraction=vapor_fraction,D_to_F=cut)
        result = unit.solve({'feed':feed})
        distillate = result.outlet_streams['distillate']
        assert distillate.vapor_fraction == pytest.approx(.25)
        assert distillate.liquid2_fraction == pytest.approx(.75)
        assert distillate.effective_liquid1_fraction == 0.
        liquid = result.outlet_streams['distillate_liquid']
        vapor = result.outlet_streams['distillate_vapor']
        assert liquid.F+vapor.F == pytest.approx(distillate.F)
        assert liquid.F*liquid.H+vapor.F*vapor.H == pytest.approx(distillate.F*distillate.H,abs=.01)
        assert_balanced(unit,feed,result)

    def test_selective_routing_rejects_an_absent_top_liquid(self):
        thermo = create_thermodynamics(['methanol','water'],'NRTL')
        feed = thermo.calculate_state(298.15,1.,100.,{'methanol':.4,'water':.6},phase='liquid',flash=False)
        unit = RigorousDistillation('NO-SPLIT',thermo,{
            'N_stages':6,'feed_stage':3,'D_to_F':.4,'P_condenser':1.,'P_drop_per_stage':0.,
            'stage_phase_model':'VLLE','vlle_seed':'cheap','mesh_tolerance':1e-6,
            'distillate_liquid1_component':'methanol','distillate_liquid1_fraction':.2,
            'distillate_liquid2_fraction':.8})
        with pytest.raises(UnitOperationError,match='selected phase is absent'):
            unit.solve({'feed':feed})


@pytest.mark.parametrize('condenser,vapor_fraction', [('total',0.),('mixed',.25)])
@pytest.mark.parametrize('mass_spec', [False,True])
def test_routed_local_jacobian_matches_full_residual(condenser,vapor_fraction,mass_spec):
    unit,feed = butanol_column(N_stages=2, feed_stage=2, condenser_type=condenser,
        distillate_vapor_fraction=vapor_fraction, distillate_liquid1_fraction=.2,
        distillate_liquid2_fraction=.8, vlle_topology_policy='residual_gate')
    comps = ['butanol','water']
    candidate = unit._vlle_azeotrope_candidates(comps,1.)[0]
    x1,x2,T = candidate['liquid1'],candidate['liquid2'],candidate['T']
    beta = .4
    aggregate = {c:(1-beta)*x1[c]+beta*x2[c] for c in comps}
    profile = VLLEProfile(T=[T,390.],aggregate_x=[aggregate,{'butanol':.98,'water':.02}],
        L=[50.,40.],V=[30.,60.],Q_cond=-5e5,Q_reb=8e5,
        split_data=[(x1,x2,beta),None])
    inlet,feeds = unit._aggregate_feeds({'feed':feed},2,2)
    model = EquationOrientedVLLEColumn(unit,inlet,feeds,comps,[1.,1.],2.,vapor_fraction,
        {'kind':'mass' if mass_spec else 'molar','value':1000. if mass_spec else 30.},
        280.,420.,100.,5e6,{'butanol':40.,'water':60.},[True,False],profile,
        VLLEStabilityCache(unit.thermo,comps,1e-7),1e-6,1e-3,1e-99,{})
    vector = model.pack_initial()
    f0 = model.residual(vector)
    analytic,_,_ = model.local_jacobian(vector,f0,1e-6)
    groups = unit._color_jacobian_columns(model._sparsity)
    numeric,_ = unit._finite_difference_jacobian(model.residual,vector,f0,model._sparsity,groups,1e-6)
    reference = numeric.toarray()
    scaled = np.abs(analytic.toarray()-reference)/np.maximum(1.,np.abs(reference))
    assert scaled.max() < 1e-5


@pytest.mark.parametrize('value', ['decanter','heterogeneous','heterogeneous_decanter','top_decanter'])
@pytest.mark.parametrize('compact', [False,True])
def test_removed_decanter_fails_during_parsing_with_vlle_guidance(value,compact):
    header = 'UNIT COL : RigorousDistillation\n' if compact else 'UNIT COL\n    TYPE: RigorousDistillation\n    PARAMS:\n'
    with pytest.raises(ParseError,match='Enable stage_phase_model=VLLE'):
        parse_pfd(header+f'    condenser_type = {value}\n')


@pytest.mark.parametrize('parameter', ['decanter_reflux_component','decanter_reflux_purge_fraction',
                                      'decanter_T','decanter_distillate_guess','reflux_phase_component'])
def test_removed_decanter_parameters_are_not_silently_ignored(parameter):
    with pytest.raises(ParseError,match='coupled three-phase'):
        parse_pfd(f'UNIT COL : RigorousDistillation\n    {parameter} = 0.1\n')


def test_removed_condenser_cannot_be_hidden_by_a_later_duplicate():
    with pytest.raises(ParseError,match='coupled three-phase'):
        parse_pfd('UNIT COL : RigorousDistillation\n    condenser_type = decanter\n'
                  '    condenser_type = total\n')


def test_duplicate_withdrawal_fraction_is_rejected():
    with pytest.raises(ParseError,match='Duplicate distillation specification'):
        parse_pfd('UNIT COL : RigorousDistillation\n    stage_phase_model = VLLE\n'
                  '    distillate_liquid1_component = water\n'
                  '    distillate_liquid1_fraction = 0\n'
                  '    distillate_liquid1_fraction = 1\n'
                  '    distillate_liquid2_fraction = 1\n')


@pytest.mark.parametrize('extra,match', [
    ('', 'Specify both'),
    ('    distillate_liquid2_fraction = 1\n', 'requires stage_phase_model=VLLE'),
    ('    stage_phase_model = VLLE\n    distillate_liquid2_fraction = 1\n', 'require distillate_liquid1_component'),
    ('    stage_phase_model = VLLE\n    distillate_liquid2_fraction = 1\n    distillate_liquid1_component = water\n    RR = 2\n', 'omit reflux_ratio'),
])
def test_invalid_routing_fails_during_parsing(extra,match):
    with pytest.raises(ParseError,match=match):
        parse_pfd('UNIT COL : RigorousDistillation\n    distillate_liquid1_fraction = 0\n'+extra)


def test_routing_fractions_reject_units_during_parsing():
    with pytest.raises(ParseError,match='dimensionless'):
        parse_pfd('UNIT COL : RigorousDistillation\n    stage_phase_model = VLLE\n'
                  '    distillate_liquid1_component = water\n'
                  '    distillate_liquid1_fraction = 0 [percent]\n'
                  '    distillate_liquid2_fraction = 1\n')


def test_routing_can_inherit_global_vlle_policy():
    parsed = parse_pfd('FLUID_PHASE_MODEL: VLLE\nUNIT COL : RigorousDistillation\n'
                       '    distillate_liquid1_component = water\n'
                       '    distillate_liquid1_fraction = 0\n'
                       '    distillate_liquid2_fraction = 1\n')
    assert parsed.units[0].unit_type == 'RigorousDistillation'


@pytest.mark.parametrize('unit_type', ['CMODistillation','McCabeThieleDistillation'])
def test_vle_only_columns_reject_new_routing_during_parsing(unit_type):
    with pytest.raises(ParseError,match='requires RigorousDistillation'):
        parse_pfd(f'UNIT COL : {unit_type}\n    stage_phase_model = VLLE\n'
                  '    distillate_liquid1_fraction = 0\n    distillate_liquid2_fraction = 1\n'
                  '    distillate_liquid1_component = water\n')


@pytest.mark.parametrize('unit_type', [CMODistillation,McCabeThieleDistillation])
def test_vle_only_python_columns_do_not_silently_ignore_new_routing(unit_type):
    reference,feed = butanol_column()
    unit = unit_type('VLE-ONLY',reference.thermo,{'distillate_liquid1_fraction':.2,
                                               'distillate_liquid2_fraction':.8})
    with pytest.raises(UnitOperationError,match='requires RigorousDistillation'):
        unit.solve({'feed':feed})


@pytest.mark.parametrize('changes,match', [
    ({'distillate_liquid1_fraction':-1},'between 0 and 1'),
    ({'distillate_liquid2_fraction':math.nan},'between 0 and 1'),
    ({'distillate_liquid1_component':'unknown'},'declared column component'),
    ({'reflux_ratio':1.2},'omit reflux_ratio'),
    ({'stage_phase_model':'VLE'},'requires stage_phase_model=VLLE'),
    ({'condenser_type':'partial'},'total or mixed'),
    ({'distillate_liquid1_fraction':0,'distillate_liquid2_fraction':0},'positive distillate'),
])
def test_routing_specification_rejects_invalid_values(changes,match):
    unit,feed = butanol_column(**changes)
    with pytest.raises(UnitOperationError,match=match):
        unit.solve({'feed':feed})
