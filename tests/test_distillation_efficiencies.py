"""Native VLE/VLLE efficiency contracts, physical audits, and integration."""

import math
import unittest
from unittest.mock import Mock, patch

import numpy as np

from distillation_specifications import stage_efficiency_specification
from equilibrium_stage_vlle import EquationOrientedVLLEColumn, VLLEProfile, VLLEStabilityCache
from equilibrium_stage_vlle import _ActiveSetChange, solve_vlle_active_set, VLLETopologyCycle
from pfd_parser import ParseError, parse_pfd
from simulator import Simulator
from thermodynamics import create_thermodynamics
from unit_operations_distillation import (
    RigorousDistillation, CMODistillation, McCabeThieleDistillation, ShortcutDistillation, UnitOperationError,
)
from tests.test_vlle_overhead_routing import butanol_column


def binary_column(method='NRTL',vapor=False,**overrides):
    t = create_thermodynamics(['methanol','water'],method)
    f = t.calculate_state(400. if vapor else 298.15,1.,100.,{'methanol':.4,'water':.6},
                         phase='vapor' if vapor else 'liquid',flash=False)
    p = {'N_stages':20,'feed_stage':10,'reflux_ratio':2.,'D_to_F':.45 if vapor else .35,
         'P_condenser':1.,'P_drop_per_stage':0.,'mesh_tolerance':1e-7,
         'acceptable_mesh_residual':1e-7,'max_iterations':80,'max_jacobian_evaluations':80}
    p.update(overrides)
    return RigorousDistillation('COL',t,p),f


class DistillationEfficiencyTests(unittest.TestCase):
    def assert_balanced(self,unit,inlets,result):
        products = [s for name,s in result.outlet_streams.items()
                    if name not in ('distillate_vapor','distillate_liquid')]
        for c in unit.thermo.components:
            incoming = sum(f.F*f.composition.get(c,0.) for f in inlets.values())
            outgoing = sum(f.F*f.composition.get(c,0.) for f in products)
            self.assertAlmostEqual(incoming,outgoing,delta=2e-5)
        energy = sum(s.F*s.H for s in products)-sum(f.F*f.H for f in inlets.values())
        p = result.performance
        self.assertAlmostEqual(energy,3600*(p['condenser_duty_kW']+p['reboiler_duty_kW']),delta=.1)
        self.assertLess(p['mesh_residual'],1e-7)
        self.assertLess(p['max_murphree_residual'],1e-7)
        # Check every component independently; normalized dependent component
        # must obey the same equation, not simply disappear from diagnostics.
        for j,E in enumerate(p['stage_efficiencies']):
            if E == 1.:
                continue
            below = p['stage_vapor_compositions'][j+1]
            flow = p['vapor_flows'][j+1]
            incoming = {c:flow*below[c] for c in unit.thermo.components}
            for port,feed in inlets.items():
                stage = next(f['stage'] for f in p['feeds'] if f['port']==port)-1
                if stage == j and feed.vapor_fraction > 0:
                    amount = feed.F*feed.vapor_fraction
                    y = feed.y or feed.composition
                    flow += amount
                    for c in incoming:
                        incoming[c] += amount*y[c]
            for c in incoming:
                yin = incoming[c]/flow
                self.assertAlmostEqual(p['stage_vapor_compositions'][j][c],
                    yin+E*(p['stage_equilibrium_vapor_compositions'][j][c]-yin),delta=1e-7)

    def test_uniform_ordered_and_range_efficiency_inputs(self):
        uniform = stage_efficiency_specification({'stage_efficiency':.7},6)
        self.assertEqual(uniform,(1.,.7,.7,.7,.7,1.))
        self.assertEqual(stage_efficiency_specification({'stage_efficiencies':[.5,.6,.7,.8]},6),
                         (1.,.5,.6,.7,.8,1.))
        self.assertEqual(stage_efficiency_specification({'stage_efficiencies':[1.,.5,.6,.7,.8,1.]},6),
                         (1.,.5,.6,.7,.8,1.))
        self.assertEqual(stage_efficiency_specification({'stage_efficiency':.9,
            'stage_efficiencies':{'2-3':.5,'5':0.}},6),(1.,.5,.5,.9,0.,1.))
        for compact in (False,True):
            header = 'UNIT COL : RigorousDistillation\n' if compact else 'UNIT COL\n    TYPE: RigorousDistillation\n    PARAMS:\n'
            for spec in ('[0.5, 0.6, 0.7, 0.8]','[1,0.5,0.6,0.7,0.8,1]',
                         '{"2-3": 0.5, "4": 0.7, "5": 0.8}'):
                with self.subTest(compact=compact,spec=spec):
                    parsed = parse_pfd(header+f'    N_stages = 6\n    stage_efficiencies = {spec}\n')
                    values = {p.name:p.value for p in parsed.units[0].params}
                    self.assertEqual(stage_efficiency_specification(values)[0],1.)

    def test_invalid_efficiencies_fail_before_solving_or_during_parsing(self):
        bad = [({'stage_efficiency':math.nan},'finite'),({'stage_efficiency':-.1},'between'),
               ({'stage_efficiency':1.1},'between'),({'stage_efficiencies':[.7,.8]},'profile'),
               ({'stage_efficiencies':{'2-4':.7,'4':.8}},'Overlapping'),
               ({'stage_efficiencies':{'1':.7}},'Condenser'),
               ({'stage_efficiencies':{'6':.7}},'Condenser'),
               ({'stage_efficiencies':{'2-7':.7}},'outside'),
               ({'stage_efficiencies':{'4-2':.7}},'outside'),
               ({'stage_efficiencies':{'other':.7}},'Invalid'),
               ({'stage_efficiency':.7,'__unit__stage_efficiency':'%'},'dimensionless')]
        for params,error in bad:
            with self.subTest(params=params),self.assertRaisesRegex(ValueError,error):
                stage_efficiency_specification(params,6)
        for cls in (CMODistillation,McCabeThieleDistillation,ShortcutDistillation):
            unit,feed = binary_column(stage_efficiency=.7)
            with self.subTest(cls=cls),self.assertRaisesRegex(UnitOperationError,'require RigorousDistillation'):
                cls('BAD',unit.thermo,unit.params).solve({'feed':feed})
        for spec in ('stage_efficiency = 1.1','stage_efficiency = 0.7 [%]',
                     'stage_efficiencies = {"2-4": 0.7, "4": 0.8}',
                     'stage_efficiencies = {"1": 0.5}'):
            with self.subTest(spec=spec),self.assertRaises(ParseError):
                parse_pfd(f'UNIT COL : RigorousDistillation\n    N_stages = 6\n    {spec}\n')

    def test_equilibrium_default_and_explicit_unit_efficiency_keep_small_model(self):
        unit,feed = binary_column()
        first = unit.solve({'feed':feed})
        unit.params['stage_efficiency'] = 1.
        other = unit.solve({'feed':feed})
        self.assertEqual(first.performance['stage_vapor_compositions'],other.performance['stage_vapor_compositions'])
        self.assertEqual(first.performance['solver_iterations'],other.performance['solver_iterations'])
        self.assertEqual(other.performance['max_murphree_residual'],0.)
        self.assert_balanced(unit,{'feed':feed},other)

    def test_low_efficiency_decreases_separation_and_matches_probe_pin(self):
        unit,feed = binary_column(stage_efficiency=.5)
        result = unit.solve({'feed':feed})
        self.assertAlmostEqual(result.outlet_streams['distillate'].composition['methanol'],.976169,delta=1e-4)
        self.assert_balanced(unit,{'feed':feed},result)
        # Efficiency zero on a single tray is a valid bypass-composition limit.
        unit.params['stage_efficiencies'] = {'5':0.}
        bypass = unit.solve({'feed':feed})
        self.assertEqual(bypass.performance['stage_vapor_compositions'][4].keys(),
                         bypass.performance['stage_vapor_compositions'][5].keys())
        self.assert_balanced(unit,{'feed':feed},bypass)

    def test_partial_mixed_and_subcooled_condensers_with_vapor_feed(self):
        for condenser in ('total','partial','mixed'):
            unit,feed = binary_column(vapor=True,stage_efficiency=.7,condenser_type=condenser)
            if condenser == 'mixed':
                unit.params['distillate_vapor_fraction'] = .3
            else:
                if condenser == 'total':
                    unit.params['condenser_subcooling'] = 5.
            with self.subTest(condenser=condenser):
                result = unit.solve({'feed':feed})
                self.assert_balanced(unit,{'feed':feed},result)

    def test_two_phase_and_multiple_feeds_preserve_actual_vapor_inventory(self):
        unit,feed = binary_column(stage_efficiency=.7)
        # An equilibrium TP flash provides actual phase compositions; mixture
        # z must not be substituted for vapor y in the efficiency driving force.
        mixed = unit.thermo.calculate_state(360.,1.,40.,{'methanol':.4,'water':.6},flash=True)
        self.assertGreater(mixed.vapor_fraction,0.)
        self.assertLess(mixed.vapor_fraction,1.)
        liquid = unit.thermo.calculate_state(298.15,1.,60.,feed.composition,phase='liquid',flash=False)
        unit.params['feed_stages'] = {'feed':10,'second':14}
        inlets = {'feed':mixed,'second':liquid}
        result = unit.solve(inlets)
        self.assert_balanced(unit,inlets,result)

    def test_eos_actual_vapor_and_mass_spec_side_draw_balances(self):
        unit,feed = binary_column(method='NRTL-RK',stage_efficiency=.7,
                                 side_draws='stage:15,phase:vapor,flow:1,port:vapor_side')
        first = unit.solve({'feed':feed})
        unit.params.pop('D_to_F')
        mass = first.outlet_streams['distillate'].mass_flow()
        unit.params['distillate_mass_flow'] = mass
        result = unit.solve({'feed':feed})
        self.assertAlmostEqual(result.outlet_streams['distillate'].mass_flow(),mass,delta=1e-5)
        self.assertAlmostEqual(result.outlet_streams['vapor_side'].F,1.)
        self.assert_balanced(unit,{'feed':feed},result)
        t = create_thermodynamics(['benzene','toluene'],'PR')
        f = t.calculate_state(425.,1.,100.,{'benzene':.5,'toluene':.5},phase='vapor',flash=False)
        p = dict(N_stages=20,feed_stage=10,reflux_ratio=2.,D_to_F=.45,P_condenser=1.,
                 P_drop_per_stage=0.,stage_efficiency=.7,condenser_type='partial',mesh_tolerance=1e-7,
                 acceptable_mesh_residual=1e-7,max_iterations=80)
        unit = RigorousDistillation('EOS',t,p)
        self.assert_balanced(unit,{'feed':f},unit.solve({'feed':f}))

    def test_vlle_selective_routing_subcooling_and_partial_vapor_eos(self):
        for method,condenser,vapor,subcool in (
            ('NRTL','total',False,0.),('NRTL','total',False,5.),
            ('NRTL-RK','partial',True,0.),
        ):
            with self.subTest(method=method,condenser=condenser):
                unit,feed = butanol_column(stage_efficiency=.7,acceptable_mesh_residual=1e-7)
                if method != 'NRTL':
                    unit.thermo = create_thermodynamics(['butanol','water'],method)
                    feed = unit.thermo.calculate_state(430.,1.,100.,{'butanol':.4,'water':.6},phase='vapor',flash=False)
                if condenser == 'partial':
                    for name in ('distillate_liquid1_fraction','distillate_liquid2_fraction','distillate_liquid1_component'):
                        unit.params.pop(name)
                    unit.params.update(condenser_type='partial',reflux_ratio=2.,D_to_F=.65)
                if subcool:
                    unit.params['condenser_subcooling'] = subcool
                result = unit.solve({'feed':feed})
                self.assert_balanced(unit,{'feed':feed},result)
                self.assertLess(result.performance['vlle_max_log_fugacity_residual'],1e-6)
                self.assertFalse(result.performance['stage_vapor_equilibrium_enforced'][1])

    def test_cycle_fallback_recovers_without_disabling_future_solve_projection(self):
        unit,feed = butanol_column(stage_efficiency=.7,vlle_seed='cheap',D_to_F=.773,
                                  reflux_ratio=1.2,acceptable_mesh_residual=1e-7)
        for name in ('distillate_liquid1_fraction','distillate_liquid2_fraction','distillate_liquid1_component'):
            unit.params.pop(name)
        result = unit.solve({'feed':feed})
        p = result.performance
        self.assertEqual(p['vlle_projection_cycle_recoveries'],1)
        self.assertNotIn('vlle_projection_enabled',unit.params)
        self.assertTrue(any(e['reason']=='disable_projection_on_cycle' for e in p['vlle_topology_events']))
        self.assertEqual(p['vlle_topology'],'L'*12+'.'*8)
        self.assert_balanced(unit,{'feed':feed},result)

    def test_cycle_recovery_retains_checkpoint_work_and_limits_second_cycles(self):
        profile = VLLEProfile([350.,360.],[{'a':.5,'b':.5}]*2,[1.,1.],[1.,1.],0.,0.,[None,None],
                              vapor_compositions=[{'a':.7,'b':.3}]*2)
        def change(active):
            error = _ActiveSetChange(profile,active,reason='test',residual_norm=.05)
            error.add_solver_progress(iterations=2,function_evaluations=7,jacobian_evaluations=2)
            return error
        for second_cycle in (False,True):
            with self.subTest(second_cycle=second_cycle):
                unit = RigorousDistillation('ACCOUNT',None,{'vlle_max_topology_updates':4})
                models = [Mock(projection_direction_assessments=2,projection_checks=1) for _ in range(4)]
                models[0].solve.side_effect = change([True,False])
                models[1].solve.side_effect = change([False,False])
                if second_cycle:
                    models[2].solve.side_effect = change([True,False])
                    models[3].solve.side_effect = change([False,False])
                else:
                    models[2].solve.return_value = ({'stages':[{}]},[{}],{'iterations':1,'function_evaluations':3,
                        'jacobian_evaluations':1,'residual_norm':1e-12})
                    models[2].profile_from_decoded.return_value = profile
                    models[2].topology_for_decoded.return_value = [False,False]
                with patch('equilibrium_stage_vlle.EquationOrientedVLLEColumn',side_effect=models) as build:
                    arguments = (unit,None,[],['a','b'],[1.,1.],1.,0.,{'kind':'molar','value':1.},
                                 250.,450.,1.,1.,{'a':1.,'b':1.},profile,{})
                    if second_cycle:
                        with self.assertRaises(VLLETopologyCycle):
                            solve_vlle_active_set(*arguments,initial_active=[False,False])
                    else:
                        solved = solve_vlle_active_set(*arguments,initial_active=[False,False])
                        self.assertEqual(solved.projection_cycle_recoveries,1)
                        self.assertEqual(solved.work['solver_iterations'],5)
                        self.assertEqual(solved.work['function_evaluations'],17)
                        self.assertEqual(solved.work['vlle_topology_solves'],3)
                        self.assertIs(build.call_args_list[2].args[14],profile)
                self.assertTrue(models[0].projection_enabled)
                self.assertTrue(models[1].projection_enabled)
                self.assertFalse(models[2].projection_enabled)
                self.assertNotIn('vlle_projection_enabled',unit.params)

    def test_efficiency_recycle_profiles_retain_actual_vapor_compositions(self):
        for mode in ('VLE','VLLE'):
            with self.subTest(mode=mode):
                unit,feed = binary_column(stage_efficiency=.7,stage_phase_model=mode)
                unit.solve_context['recycle_evaluation'] = 1
                first = unit.solve({'feed':feed})
                second = unit.solve({'feed':feed})
                self.assertEqual(second.performance['initializer'],'previous_recycle')
                for a,b in zip(first.performance['stage_vapor_compositions'],second.performance['stage_vapor_compositions']):
                    for c in a:
                        self.assertAlmostEqual(a[c],b[c],delta=1e-7)
                self.assertLessEqual(second.performance['solver_iterations'],first.performance['solver_iterations'])
                self.assert_balanced(unit,{'feed':feed},second)

    def test_local_jacobians_match_full_residual_with_mixed_feed_and_nonuniform_efficiencies(self):
        for mode in ('VLE','VLLE'):
            unit,feed = binary_column(N_stages=4,feed_stage=2,stage_efficiency=.7,
                stage_efficiencies={'3':1.},condenser_type='mixed',distillate_vapor_fraction=.25)
            unit.params['stage_phase_model'] = mode
            mixed = unit.thermo.calculate_state(360.,1.,100.,feed.composition,flash=True)
            inlet,feeds = unit._aggregate_feeds({'feed':mixed},4,2)
            comps = ['methanol','water']
            profile = VLLEProfile([338.,345.,365.,370.],
                [{'methanol':.9,'water':.1},{'methanol':.7,'water':.3},
                 {'methanol':.3,'water':.7},{'methanol':.1,'water':.9}],
                [70.,70.,170.,65.],[35.,130.,130.,30.],-4e6,5e6,[None]*4)
            with self.subTest(mode=mode):
                if mode == 'VLE':
                    model = unit._build_mesh_model(inlet,feeds,comps,feed.composition,4,1,2.,[1.]*4,
                        'mixed',.25,[],{'kind':'mass','value':1000.},100.,5e6,{'methanol':40.,'water':60.},280.,420.)
                    ys = [model['stage_properties'](j,profile.T[j],profile.aggregate_x[j])['y'] for j in range(4)]
                    vector = unit._pack_variables(profile.T,profile.aggregate_x,profile.L,profile.V,
                        profile.Q_cond,profile.Q_reb,comps,280.,420.,5e6,vapor_compositions=ys)
                    residual,pattern,jac = model['residual'],model['sparsity'],model['jacobian']
                else:
                    model = EquationOrientedVLLEColumn(unit,inlet,feeds,comps,[1.]*4,2.,.25,
                        {'kind':'mass','value':1000.},280.,420.,100.,5e6,{'methanol':40.,'water':60.},
                        [False]*4,profile,VLLEStabilityCache(unit.thermo,comps,1e-7),1e-6,1e-3,0.,{})
                    vector = model.pack_initial()
                    residual,pattern,jac = model.residual,model._sparsity,model.local_jacobian
                f0 = residual(vector)
                analytic = jac(vector,f0,1e-6)[0].toarray()
                numeric,_ = unit._finite_difference_jacobian(residual,vector,f0,pattern,
                    unit._color_jacobian_columns(pattern),1e-6)
                reference = numeric.toarray()
                self.assertLess(np.max(abs(analytic-reference)/np.maximum(1.,abs(reference))),1e-5)

    def test_active_vlle_jacobian_uses_actual_vapor_and_routed_mass_spec(self):
        for method in ('NRTL','NRTL-RK'):
            with self.subTest(method=method):
                unit,feed = butanol_column(N_stages=4,feed_stage=3,stage_efficiency=.7,
                                          stage_efficiencies={'3':1.})
                unit.thermo = create_thermodynamics(['butanol','water'],method)
                inlet,feeds = unit._aggregate_feeds({'feed':feed},4,3)
                comps = ['butanol','water']
                candidate = unit._vlle_azeotrope_candidates(comps,1.)[0]
                x = candidate['composition']
                T = candidate['T']
                split,x1,x2,beta = unit.thermo.liquid_liquid_equilibrium(x,T,tol=1e-10)
                self.assertTrue(split)
                profile = VLLEProfile([T,T,T,390.],[x,x,x,{'butanol':.98,'water':.02}],
                    [80.,80.,180.,40.],[60.,140.,140.,40.],-5e6,7e6,[(x1,x2,beta)]*3+[None])
                model = EquationOrientedVLLEColumn(unit,inlet,feeds,comps,[1.]*4,2.,0.,
                    {'kind':'mass','value':1200.},280.,420.,100.,5e6,{'butanol':40.,'water':60.},
                    [True,True,True,False],profile,VLLEStabilityCache(unit.thermo,comps,1e-7),
                    1e-6,1e-3,0.,{})
                vector = model.pack_initial()
                f0 = model.residual(vector)
                analytic = model.local_jacobian(vector,f0,1e-6)[0].toarray()
                numeric,_ = unit._finite_difference_jacobian(model.residual,vector,f0,model._sparsity,
                    unit._color_jacobian_columns(model._sparsity),1e-6)
                reference = numeric.toarray()
                self.assertLess(np.max(abs(analytic-reference)/np.maximum(1.,abs(reference))),1e-5)

    def test_efficiency_profiles_support_cmo_and_coarse_initialization(self):
        for initializer in ('cmo','cmo-hvap','coarse_rigorous'):
            with self.subTest(initializer=initializer):
                unit,feed = binary_column(N_stages=16,feed_stage=8,stage_efficiency=.7,
                    stage_efficiencies={'9-15':.8},initializer=initializer,coarse_initial_stages=8)
                result = unit.solve({'feed':feed})
                self.assertEqual(result.performance['stage_efficiencies'],[1.]+[.7]*7+[.8]*7+[1.])
                self.assert_balanced(unit,{'feed':feed},result)


class EfficiencyIntegrationTests(unittest.TestCase):
    def test_compact_column_low_efficiency_materially_decreases_product_purity(self):
        text = '''PROCESS: Efficiency integration
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: NRTL
COMPONENTS:
    methanol | Methanol | MW=32.04
    water | Water | MW=18.02
STREAM Feed : -> COL.feed
    T = 25 [C]
    P = 1 [bar]
    F = 100 [kmol/h]
    x = methanol:0.4, water:0.6
STREAM Distillate : COL.distillate
STREAM Bottoms : COL.bottoms
UNIT COL : RigorousDistillation
    N_stages = 20
    feed_stage = 10
    reflux_ratio = 2
    D_to_F = 0.35
    P_condenser = 1 [bar]
    P_drop_per_stage = 0 [bar]
    stage_efficiencies = {"2-19": EFFICIENCY}
    mesh_tolerance = 1e-7
'''
        results = []
        for E in (1.,.5):
            sim = Simulator(parse_pfd(text.replace('EFFICIENCY',str(E))))
            result = sim.run()
            self.assertTrue(result.converged,result.errors)
            self.assertEqual(result.errors,[])
            self.assertLess(result.mass_balance_error,1e-8)
            self.assertLess(result.energy_balance_error,1e-8)
            p = result.units['COL'].performance
            self.assertLess(p['mesh_residual'],1e-7)
            self.assertLess(p['max_murphree_residual'],1e-7)
            self.assertEqual(p['stage_efficiencies'],[1.]+[E]*18+[1.])
            self.assertIn('stage_efficiencies',sim._generate_pfr())
            results.append(result)
        ideal,inefficient = results
        x1 = ideal.streams['Distillate'].composition['methanol']
        x2 = inefficient.streams['Distillate'].composition['methanol']
        self.assertAlmostEqual(x1,.997664,delta=1e-4)
        self.assertAlmostEqual(x2,.976169,delta=1e-4)
        self.assertGreater(x1-x2,.02)
        self.assertGreater(inefficient.streams['Bottoms'].composition['methanol'],
                           ideal.streams['Bottoms'].composition['methanol'])


if __name__ == '__main__':
    unittest.main()
