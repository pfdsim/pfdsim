"""Physical balances and boundary/Jacobian coverage for cooled total condensers."""

import math
import unittest

import numpy as np

from distillation_specifications import total_condenser_specification
from equilibrium_stage_vlle import EquationOrientedVLLEColumn, VLLEProfile, VLLEStabilityCache
from pfd_parser import ParseError, parse_pfd
from simulator import Simulator
from thermodynamics import create_thermodynamics
from unit_operations_distillation import (
    RigorousDistillation, ShortcutDistillation, McCabeThieleDistillation, CMODistillation,
    UnitOperationError,
)
from tests.test_vlle_overhead_routing import butanol_column


def homogeneous_column(**changes):
    thermo = create_thermodynamics(['methanol','water'], 'NRTL')
    feed = thermo.calculate_state(298.15, 1., 100., {'methanol':.4,'water':.6},
                                  phase='liquid', flash=False)
    params = {'N_stages':8,'feed_stage':4,'reflux_ratio':2.,'D_to_F':.35,
              'P_condenser':1.,'P_drop_per_stage':0.,'mesh_tolerance':1e-8,
              'max_iterations':80,'max_jacobian_evaluations':80}
    params.update(changes)
    return RigorousDistillation('SUB',thermo,params),feed


class SubcooledCondenserTests(unittest.TestCase):
    def assert_balanced(self, unit, feed, result):
        D,B = (result.outlet_streams[name] for name in ('distillate','bottoms'))
        for c in unit.thermo.components:
            self.assertAlmostEqual(D.F*D.composition[c]+B.F*B.composition[c],
                                   feed.F*feed.composition[c], delta=1e-5)
        net = D.F*D.H+B.F*B.H-feed.F*feed.H
        p = result.performance
        self.assertAlmostEqual(net,3600*(p['condenser_duty_kW']+p['reboiler_duty_kW']),delta=.02)
        self.assertLess(p['mesh_residual'],1e-7)
        self.assertEqual(D.vapor_fraction,0.)

    def test_vle_subcooling_and_absolute_temperature_agree_and_cold_reflux_changes_traffic(self):
        saturated,feed = homogeneous_column()
        baseline = saturated.solve({'feed':feed})
        cold,_ = homogeneous_column(condenser_subcooling=10.)
        result = cold.solve({'feed':feed})
        p = result.performance
        self.assertEqual(p['condenser_subcooling_K'],10.)
        self.assertAlmostEqual(p['condenser_saturation_temperature_K']-result.outlet_streams['distillate'].T,10.)
        self.assertLess(p['condenser_vapor_saturation_ratio'],.9)
        self.assertLess(p['condenser_duty_kW'],baseline.performance['condenser_duty_kW'])
        # Reflux cooling condenses rising vapor on tray 2; the MESH balances
        # calculate this change, rather than adjusting the final product only.
        self.assertGreater(abs(p['vapor_flows'][2]-baseline.performance['vapor_flows'][2]),.1)
        self.assertEqual(p['stage_phase_counts'][0],1)
        self.assertFalse(p['stage_vapor_equilibrium_enforced'][0])
        self.assert_balanced(cold,feed,result)
        absolute,_ = homogeneous_column(condenser_temperature=result.outlet_streams['distillate'].T)
        other = absolute.solve({'feed':feed})
        self.assert_balanced(absolute,feed,other)
        for name in ('distillate','bottoms'):
            a,b = result.outlet_streams[name],other.outlet_streams[name]
            self.assertAlmostEqual(a.T,b.T,delta=1e-5)
            self.assertAlmostEqual(a.composition['methanol'],b.composition['methanol'],delta=1e-7)
        self.assertAlmostEqual(other.performance['condenser_subcooling_K'],10.,delta=1e-5)

    def test_zero_subcooling_reproduces_saturated_total_condenser(self):
        for mode in ('VLE','VLLE'):
            with self.subTest(mode=mode):
                unit,feed = homogeneous_column(stage_phase_model=mode)
                baseline = unit.solve({'feed':feed})
                unit.params['condenser_subcooling'] = 0.
                result = unit.solve({'feed':feed})
                self.assertAlmostEqual(result.outlet_streams['distillate'].T,
                                       baseline.outlet_streams['distillate'].T,delta=1e-5)
                self.assertAlmostEqual(result.outlet_streams['bottoms'].composition['methanol'],
                                       baseline.outlet_streams['bottoms'].composition['methanol'],delta=1e-7)
                self.assert_balanced(unit,feed,result)

    def test_vlle_subcooling_re_equilibrates_liquids_and_selective_routing(self):
        saturated,feed = butanol_column()
        baseline = saturated.solve({'feed':feed})
        self.assert_balanced(saturated,feed,baseline)
        unit,_ = butanol_column(condenser_subcooling=5.)
        result = unit.solve({'feed':feed})
        p = result.performance
        baseline_bottoms = baseline.outlet_streams['bottoms']
        cooled_bottoms = result.outlet_streams['bottoms']
        butanol_feed = feed.F*feed.composition['butanol']
        baseline_recovery = baseline_bottoms.F*baseline_bottoms.composition['butanol']/butanol_feed
        cooled_recovery = cooled_bottoms.F*cooled_bottoms.composition['butanol']/butanol_feed
        # At the same cut and withdrawal fractions, cooling this tie line
        # improves purity/recovery and reduces reflux. Require material
        # changes, with margins below half the observed effect, rather than
        # pinning solver-dependent digits or claiming a universal direction.
        self.assertGreater(cooled_bottoms.composition['butanol'],
                           baseline_bottoms.composition['butanol']+.0005)
        self.assertGreater(cooled_recovery,baseline_recovery+.0005)
        self.assertLess(p['reflux_ratio'],baseline.performance['reflux_ratio']-.05)
        self.assertAlmostEqual(p['condenser_saturation_temperature_K']-p['condenser_temperature_K'],5.)
        self.assertEqual(p['stage_phase_counts'][0],2)
        self.assertFalse(p['stage_vapor_equilibrium_enforced'][0])
        self.assertLess(p['vlle_max_log_fugacity_residual'],1e-6)
        top = p['top_liquid_routing']
        self.assertEqual(top['distillate_liquid1_flow'],0.)
        self.assertEqual(top['reflux_liquid2_flow'],0.)
        # Both compositions differ from the saturated tie line (.37768/.02120).
        self.assertGreater(top['liquid1_composition']['butanol'],.39)
        self.assertLess(top['liquid2_composition']['butanol'],.0205)
        T = result.outlet_streams['distillate'].T
        f1 = unit.thermo._phase_log_fugacities(T,1.,top['liquid1_composition'],'liquid')
        f2 = unit.thermo._phase_log_fugacities(T,1.,top['liquid2_composition'],'liquid')
        self.assertLess(max(abs(f1[c]-f2[c]) for c in unit.thermo.components),1e-7)
        self.assert_balanced(unit,feed,result)
        # Repeat with absolute temperature, reversed selector, and a mass spec.
        mass = result.outlet_streams['distillate'].mass_flow()
        other,_ = butanol_column(condenser_temperature=T,distillate_liquid1_component='water',
                                distillate_liquid1_fraction=1.,distillate_liquid2_fraction=0.)
        other.params.pop('D_to_F')
        other.params['distillate_mass_flow'] = mass
        second = other.solve({'feed':feed})
        self.assertAlmostEqual(second.outlet_streams['distillate'].mass_flow(),mass,delta=1e-5)
        self.assertAlmostEqual(second.outlet_streams['bottoms'].composition['butanol'],
                               result.outlet_streams['bottoms'].composition['butanol'],delta=1e-7)
        self.assert_balanced(other,feed,second)

    def test_subcooled_vlle_with_homogeneous_top_and_adaptive_mode(self):
        for mode in ('VLLE','VL(L)E'):
            with self.subTest(mode=mode):
                unit,feed = homogeneous_column(stage_phase_model=mode,condenser_subcooling=5.)
                result = unit.solve({'feed':feed})
                self.assertEqual(result.performance['stage_phase_counts'][0],1)
                self.assert_balanced(unit,feed,result)

    def test_subcooled_condenser_preserves_explicit_cmo_and_coarse_initializers(self):
        for initializer in ('cmo','cmo-hvap','coarse_rigorous'):
            with self.subTest(initializer=initializer):
                stages = 16 if initializer == 'coarse_rigorous' else 8
                unit,feed = homogeneous_column(condenser_subcooling=5.,initializer=initializer,
                    N_stages=stages,feed_stage=stages//2,coarse_initial_stages=stages//2)
                result = unit.solve({'feed':feed})
                self.assertEqual(result.performance['condenser_subcooling_K'],5.)
                self.assert_balanced(unit,feed,result)

    def test_subcooled_vle_with_side_draw_and_mass_spec_closes_all_products(self):
        unit,feed = homogeneous_column(N_stages=12,feed_stage=6,condenser_subcooling=5.,
            side_draws='stage:9,phase:liquid,flow:2,port:side_liq')
        first = unit.solve({'feed':feed})
        mass = first.outlet_streams['distillate'].mass_flow()
        unit.params.pop('D_to_F')
        unit.params['distillate_mass_flow'] = mass
        result = unit.solve({'feed':feed})
        self.assertAlmostEqual(result.outlet_streams['distillate'].mass_flow(),mass,delta=1e-5)
        self.assertAlmostEqual(result.outlet_streams['side_liq'].F,2.)
        for c in unit.thermo.components:
            self.assertAlmostEqual(sum(s.F*s.composition[c] for s in result.outlet_streams.values()),
                                   feed.F*feed.composition[c],delta=1e-5)
        net = sum(s.F*s.H for s in result.outlet_streams.values())-feed.F*feed.H
        p = result.performance
        self.assertAlmostEqual(net,3600*(p['condenser_duty_kW']+p['reboiler_duty_kW']),delta=.02)

    def test_both_boundary_jacobians_match_complete_finite_difference(self):
        for mode in ('VLE','VLLE'):
            for spec in ({'condenser_subcooling':5.},{'condenser_temperature':361.189578695}):
                with self.subTest(mode=mode,spec=spec):
                    unit,feed = homogeneous_column(N_stages=2,feed_stage=2,**spec)
                    comps = ['methanol','water']
                    inlet,feeds = unit._aggregate_feeds({'feed':feed},2,2)
                    initial = {'T':[328.,360.], 'x':[{'methanol':.95,'water':.05},
                               {'methanol':.1,'water':.9}], 'L':[70.,65.], 'V':[35.,105.],
                               'Q_cond':-4e6,'Q_reb':4e6}
                    scales = {'methanol':40.,'water':60.}
                    if mode == 'VLE':
                        model = unit._build_mesh_model(inlet,feeds,comps,feed.composition,2,1,2.,
                            [1.,1.],'total',0.,[],{'kind':'molar','value':35.},100.,5e6,scales,280.,420.)
                        vector = unit._pack_variables(initial['T'],initial['x'],initial['L'],initial['V'],
                            initial['Q_cond'],initial['Q_reb'],comps,280.,420.,5e6)
                        residual,pattern = model['residual'],model['sparsity']
                        analytic = model['jacobian'](vector,residual(vector),1e-6)[0]
                    else:
                        profile = VLLEProfile(initial['T'],initial['x'],initial['L'],initial['V'],
                                              initial['Q_cond'],initial['Q_reb'],[None,None])
                        unit.params['stage_phase_model'] = 'VLLE'
                        model = EquationOrientedVLLEColumn(unit,inlet,feeds,comps,[1.,1.],2.,0.,
                            {'kind':'molar','value':35.},280.,420.,100.,5e6,scales,[False,False],profile,
                            VLLEStabilityCache(unit.thermo,comps,1e-7),1e-6,1e-3,1e-99,{})
                        vector = model.pack_initial()
                        residual,pattern = model.residual,model._sparsity
                        analytic = model.local_jacobian(vector,residual(vector),1e-6)[0]
                    numeric,_ = unit._finite_difference_jacobian(residual,vector,residual(vector),
                        pattern,unit._color_jacobian_columns(pattern),1e-6)
                    reference = numeric.toarray()
                    self.assertLess(np.max(abs(analytic.toarray()-reference)/np.maximum(1.,abs(reference))),1e-5)

    def test_input_validation_and_unsupported_models(self):
        bad = [({'condenser_temperature':300.,'condenser_subcooling':5.},'only one'),
               ({'condenser_subcooling':-1.},'nonnegative'),
               ({'condenser_subcooling':math.nan},'finite'),
               ({'condenser_temperature':math.inf},'finite'),
               ({'condenser_temperature':0.,'__unit__condenser_temperature':'K'},'positive'),
               ({'condenser_subcooling':5.,'condenser_type':'mixed'},'total condenser'),
               ({'condenser_temperature':300.,'__unit__condenser_temperature':'bar'},'Unsupported'),
               ({'condenser_subcooling':1.,'__unit__condenser_subcooling':'bar'},'Unsupported')]
        for params,message in bad:
            with self.subTest(params=params),self.assertRaisesRegex(ValueError,message):
                total_condenser_specification(params)
        for cls in (ShortcutDistillation,McCabeThieleDistillation,CMODistillation):
            unit,feed = homogeneous_column(condenser_subcooling=5.)
            with self.subTest(cls=cls),self.assertRaisesRegex(UnitOperationError,'requires RigorousDistillation'):
                cls('UNSUPPORTED',unit.thermo,unit.params).solve({'feed':feed})
        for value in (100.,2000.):
            unit,feed = homogeneous_column(condenser_temperature=value,__unit__condenser_temperature='K')
            with self.subTest(bounds=value),self.assertRaisesRegex(UnitOperationError,'temperature bounds'):
                unit.solve({'feed':feed})
        unit,feed = homogeneous_column(condenser_temperature=350.)
        with self.assertRaisesRegex(UnitOperationError,'above condensate saturation'):
            unit.solve({'feed':feed})

    def test_active_liquid_split_jacobian_includes_subcooling_reference_and_mass_routing(self):
        for spec in ({'condenser_subcooling':5.},{'condenser_temperature':361.189578695}):
            with self.subTest(spec=spec):
                unit,feed = butanol_column(N_stages=2,feed_stage=2,**spec)
                comps = ['butanol','water']
                inlet,feeds = unit._aggregate_feeds({'feed':feed},2,2)
                candidate = unit._vlle_azeotrope_candidates(comps,1.)[0]
                bulk = candidate['composition']
                T = candidate['T']-5.
                split,x1,x2,beta = unit.thermo.liquid_liquid_equilibrium(bulk,T,tol=1e-10)
                self.assertTrue(split)
                profile = VLLEProfile([T,390.],[bulk,{'butanol':.98,'water':.02}],
                    [80.,40.],[60.,140.],-5e6,7e6,[(x1,x2,beta),None])
                model = EquationOrientedVLLEColumn(unit,inlet,feeds,comps,[1.,1.],2.,0.,
                    {'kind':'mass','value':1200.},280.,420.,100.,5e6,{'butanol':40.,'water':60.},
                    [True,False],profile,VLLEStabilityCache(unit.thermo,comps,1e-7),
                    1e-6,1e-3,1e-99,{})
                vector = model.pack_initial()
                f0 = model.residual(vector)
                analytic = model.local_jacobian(vector,f0,1e-6)[0]
                numeric,_ = unit._finite_difference_jacobian(model.residual,vector,f0,model._sparsity,
                    unit._color_jacobian_columns(model._sparsity),1e-6)
                reference = numeric.toarray()
                self.assertLess(np.max(abs(analytic.toarray()-reference)/np.maximum(1.,abs(reference))),1e-5)

    def test_temperature_units_and_parse_time_rejections(self):
        for value,unit in ((25.,'C'),(298.15,'K'),(77.,'F')):
            self.assertAlmostEqual(total_condenser_specification({'condenser_temperature':value,
                '__unit__condenser_temperature':unit})['temperature_K'],298.15)
        for value,unit in ((10.,'C'),(10.,'K'),(18.,'F')):
            self.assertAlmostEqual(total_condenser_specification({'condenser_subcooling':value,
                '__unit__condenser_subcooling':unit})['subcooling_K'],10.)
        for unit_type,params in (
            ('ShortcutDistillation','condenser_subcooling = 5 [K]'),
            ('McCabeThieleDistillation','condenser_subcooling = 5 [K]'),
            ('CMODistillation','condenser_subcooling = 5 [K]'),
            ('RigorousDistillation','condenser_subcooling = -5 [K]'),
            ('RigorousDistillation','condenser_temperature = 300 [K]\n    condenser_subcooling = 5 [K]'),
            ('RigorousDistillation','condenser_subcooling = 5 [K]\n    condenser_type = partial'),
        ):
            with self.subTest(params=params),self.assertRaises(ParseError):
                parse_pfd(f'UNIT COL : {unit_type}\n    {params}\n')
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            total_condenser_specification([('condenser_subcooling',5.),('Condenser_Subcooling',10.)])

    def test_compact_pfd_keeps_difference_units_and_converts_absolute_temperature(self):
        text = '''PROCESS: Subcooled column
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
    N_stages = 8
    feed_stage = 4
    reflux_ratio = 2
    D_to_F = 0.35
    P_condenser = 1 [bar]
    P_drop_per_stage = 0 [bar]
    mesh_tolerance = 1e-8
    condenser_subcooling = 18 [F]
'''
        sim = Simulator(parse_pfd(text))
        result = sim.run()
        self.assertTrue(result.converged,result.errors)
        p = result.units['COL'].performance
        self.assertEqual(p['condenser_subcooling_K'],10.)
        absolute = text.replace('condenser_subcooling = 18 [F]',
                                f"condenser_temperature = {p['condenser_temperature_K']-273.15:.12f} [C]")
        result2 = Simulator(parse_pfd(absolute)).run()
        self.assertTrue(result2.converged,result2.errors)
        self.assertAlmostEqual(result2.streams['Distillate'].T,result.streams['Distillate'].T,delta=1e-6)
