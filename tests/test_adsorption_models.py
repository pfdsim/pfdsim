import math

import numpy as np
import pytest

from adsorption_models import (GSTA_3A_WATER, PureIsotherm, iast,
                               iast_enthalpy, iast_from_loadings, iast_isosteric_heats, R)


def test_iast_matches_competitive_langmuir_for_common_capacity():
    models = {c:PureIsotherm({'qmax':4.,'b':b},300.) for c,b in [('A',3.),('B',.2)]}
    f = {'A':.4,'B':.6}
    q, pi, f0 = iast(models,f)
    denominator = 1+3*.4+.2*.6
    assert q['A'] == pytest.approx(4*3*.4/denominator,rel=1e-10)
    assert q['B'] == pytest.approx(4*.2*.6/denominator,rel=1e-10)
    assert pi == pytest.approx(4*math.log(denominator))
    assert sum(f[c]/f0[c] for c in f) == pytest.approx(1.)


def test_unequal_capacities_obey_spreading_and_gibbs_adsorption_equation():
    models = {'A':PureIsotherm({'qmax':2.,'b':4.},300.),
              'B':PureIsotherm({'model':'dual_site_langmuir','qmax':6.,'b':.3,'qmax2':1.,'b2':8.},300.)}
    f = {'A':.3,'B':1.2}
    q,pi,f0 = iast(models,f)
    for c in f:
        assert models[c].spreading(f0[c]) == pytest.approx(pi,rel=1e-9)
        h = 1e-4
        plus,minus = dict(f),dict(f)
        plus[c] *= math.exp(h)
        minus[c] *= math.exp(-h)
        derivative = (iast(models,plus)[1]-iast(models,minus)[1])/(2*h)
        assert derivative == pytest.approx(q[c],rel=2e-7)
        assert q[c] < models[c].loading(f[c])


@pytest.mark.parametrize('model', ['langmuir','dual_site_langmuir','sips','toth','henry','freundlich','custom'])
def test_spreading_derivative_is_loading(model):
    setting = {'model':model,'qmax':3.,'b':2.,'qmax2':1.,'b2':.1,'k':2.,'n':.7,
               'expression':'cap * affinity * f / (1 + affinity * f)',
               'parameters':{'cap':3.,'affinity':2.}}
    iso = PureIsotherm(setting,310.)
    h = 1e-4
    for f in [1e-5,.1,3.]:
        pi = iso.spreading(f)
        derivative = (iso.spreading(f*math.exp(h))-iso.spreading(f*math.exp(-h)))/(2*h)
        assert derivative == pytest.approx(iso.loading(f),rel=1e-7)
        assert math.exp(iso.log_fugacity_at_spreading(pi)) == pytest.approx(f,rel=1e-7)


def test_gsta_3a_water_retains_existing_curve():
    for T in [273.15,298.15,350.,450.]:
        iso = PureIsotherm(GSTA_3A_WATER,T)
        for f in np.geomspace(1e-12,2.,30):
            terms = [math.exp(-h/(8.31446261815324*T)+s/8.31446261815324)*f**n
                     for n,(h,s) in enumerate(zip(GSTA_3A_WATER['dH'],GSTA_3A_WATER['dS']),1)]
            old = .21/4*sum(n*x for n,x in enumerate(terms,1))/(1+sum(terms))
            assert iso.loading(f)*18.01528/1000 == pytest.approx(old,rel=1e-12)


@pytest.mark.parametrize('expression', ['__import__("os")','f.__class__','[f for f in [1]]','open("file")','f[0]'])
def test_custom_expression_is_restricted(expression):
    with pytest.raises(ValueError):
        PureIsotherm({'model':'custom','expression':expression},300.)


@pytest.mark.parametrize('expression', ['1+f','-f','f*exp(-f)','sqrt(f)-f'])
def test_invalid_custom_curves_rejected(expression):
    with pytest.raises(ValueError):
        PureIsotherm({'model':'custom','expression':expression},300.).validate_range(10.)


def test_zero_components_and_trace_fugacities():
    models = {'A':PureIsotherm({'qmax':2.,'b':4.},300.),'B':PureIsotherm({'qmax':2.,'b':.2},300.)}
    assert iast(models,{'A':0.,'B':0.})[0] == {'A':0.,'B':0.}
    assert iast(models,{'A':1e-30,'B':1.})[0]['A'] > 0
    assert iast(models,{'A':1e-30,'B':0.})[0]['A'] == pytest.approx(models['A'].loading(1e-30),abs=1e-40)


@pytest.mark.parametrize('model',['langmuir','dual_site_langmuir','sips','toth','henry','freundlich','gsta_3a_water','custom'])
def test_integral_enthalpy_is_temperature_derivative_of_spreading(model):
    setting = {'model':model,'qmax':4.,'b':3.,'qmax2':1.,'b2':.1,
               'heat':25000.,'heat2':12000.,'n':.7,'k':2.,'T_ref':310.,
               'expression':'4*b*f/(1+b*f)',
               'parameters':{'b':3.}}
    if model=='custom':
        setting['expression']='4*3*exp(25000/R*(1/T-1/310))*f/(1+3*exp(25000/R*(1/T-1/310))*f)'
    iso = PureIsotherm(setting,310.)
    f,h = .3,.001
    numerical = R*310.**2*(PureIsotherm(setting,310+h).spreading(f)-PureIsotherm(setting,310-h).spreading(f))/(2*h)
    assert iso.integral_enthalpy(f) == pytest.approx(numerical,rel=1e-7)


def test_mixture_enthalpy_matches_grand_potential_derivative_and_partial_heats():
    settings = {'A':{'qmax':2.,'b':4.,'heat':25000.,'T_ref':310.},
                'B':{'model':'dual_site_langmuir','qmax':6.,'b':.3,'heat':14000.,
                     'qmax2':1.,'b2':8.,'heat2':30000.,'T_ref':310.}}
    models = {c:PureIsotherm(s,310.) for c,s in settings.items()}
    f = {'A':.3,'B':1.2}
    q,pi,f0 = iast(models,f)
    inverse_f,inverse_pi,_ = iast_from_loadings(models,q)
    assert inverse_f == pytest.approx(f,rel=1e-9)
    assert inverse_pi == pytest.approx(pi,rel=1e-9)
    h = .002
    pis = [iast({c:PureIsotherm(s,T) for c,s in settings.items()},f)[1] for T in [310+h,310-h]]
    assert iast_enthalpy(models,q,f0) == pytest.approx(R*310**2*(pis[0]-pis[1])/(2*h),rel=1e-7)
    heats = iast_isosteric_heats(models,q)
    for c in q:
        hq = q[c]*1e-4
        plus,minus = dict(q),dict(q)
        plus[c] += hq
        minus[c] -= hq
        partial = (iast_enthalpy(models,plus)-iast_enthalpy(models,minus))/(2*hq)
        assert -partial == pytest.approx(heats[c],rel=2e-6)


def test_missing_temperature_information_is_not_zero_adsorption_heat():
    for setting in [{'qmax':2.,'b':3.},
                    {'qmax':2.,'b':3.,'heat':0.,'temperature_fit':'single temperature; heats unidentifiable'},
                    {'model':'custom','expression':'f'}]:
        iso = PureIsotherm(setting,300.)
        assert not iso.enthalpy_available
        with pytest.raises(ValueError,match='enthalpy needs'):
            iso.integral_enthalpy(1.)
    assert PureIsotherm({'qmax':2.,'b':3.,'heat':0.},300.).integral_enthalpy(1.) == 0


@pytest.mark.parametrize('small_capacity,partial_fugacity', [(.1,.5), (.001,.067)])
def test_inverse_iast_and_heats_with_large_capacity_contrast(small_capacity, partial_fugacity):
    models = {c:PureIsotherm({'qmax':capacity, 'b':1., 'heat':heat},300.)
              for c,capacity,heat in [('A',small_capacity,10000.), ('B',10.,20000.)]}
    fugacities = {'A':partial_fugacity, 'B':partial_fugacity}
    q, pi, _ = iast(models, fugacities)
    recovered, recovered_pi, _ = iast_from_loadings(models, q)
    assert recovered == pytest.approx(fugacities, rel=1e-8)
    assert recovered_pi == pytest.approx(pi, rel=1e-9)
    assert iast_isosteric_heats(models, q) == pytest.approx(
        {'A':10000., 'B':20000.}, rel=1e-6)


@pytest.mark.parametrize('form',['langmuir','dual_site_langmuir','toth','sips'])
def test_fitter_and_runtime_equations_share_units_and_temperature_dependence(form):
    from scripts.adsorption_fit_tools import prediction, setting_from_fit
    from types import SimpleNamespace
    for thermal in (False,True):
        v=list(np.log([4.,3.]))
        if form=='dual_site_langmuir':
            v+=list(np.log([1.,.1]))
        if form in {'toth','sips'}:
            v+=[math.log(.65)]
        if thermal:
            v += [25.,12.] if form=='dual_site_langmuir' else [25.]
        setting=setting_from_fit(form,SimpleNamespace(x=v),thermal)
        for temperature in (273.,298.15,330.):
            f=np.geomspace(1e-8,10.,30)
            expected=prediction(v,form,f,np.full(len(f),temperature),thermal)
            iso=PureIsotherm(setting,temperature)
            assert [iso.loading(float(p)) for p in f] == pytest.approx(expected,rel=1e-12)


@pytest.mark.parametrize('f',[1e-15,1e-4,1.,1e20,1e100])
def test_logarithmic_quadrature_resolves_large_iast_fugacity_ranges(f):
    reference=PureIsotherm({'qmax':4.,'b':1e5},300.)
    for setting in [{'model':'toth','qmax':4.,'b':1e5,'n':1.},
                    {'model':'custom','expression':'4*1e5*f/(1+1e5*f)'}]:
        iso=PureIsotherm(setting,300.)
        assert iso.spreading(f) == pytest.approx(reference.spreading(f),rel=1e-7,abs=1e-12)
