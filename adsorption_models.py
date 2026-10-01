"""Pure-component isotherms and ideal adsorbed solution equilibrium.

All loadings are mol/kg dry adsorbent, fugacities bar, temperatures K.
IAST equates spreading potentials, not independently clipped capacities.
"""

import math
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .kinetic_models import SafeRateExpression
    from .physical_constants import R_J_MOL_K as R
else:
    from kinetic_models import SafeRateExpression
    from physical_constants import R_J_MOL_K as R


GSTA_3A_WATER = {
    'model': 'gsta_3a_water',
    'qmax': 0.21 * 1000 / 18.01528,
    'dH': [-46.60e3, -125.0e3, -193.6e3, -272.2e3],
    'dS': [-53.70, -221.1, -356.7, -567.5],
}


class IsothermRangeError(ValueError):
    """A requested equilibrium lies outside the numerically representable range."""


def _logsumexp(values):
    """Stable scalar reduction for the small real vectors used by IAST."""
    largest = max(values)
    if not math.isfinite(largest):
        return largest
    return largest + math.log(math.fsum(math.exp(v-largest) for v in values))


def positive(value, name, *, zero=False):
    number = float(value)
    if not math.isfinite(number) or (number < 0 if zero else number <= 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if zero else "positive"}')
    return number


class PureIsotherm:
    """A pure isotherm evaluated at one temperature, including its integral."""

    def __init__(self, setting, temperature):
        if not isinstance(setting, dict):
            raise ValueError('Each isotherm setting must be a mapping')
        self.setting = dict(setting)
        self.T = positive(temperature, 'Adsorption temperature')
        self.model = str(setting.get('model', 'langmuir')).lower()
        self.parameters = {}
        self.expression = None
        if self.model == 'custom':
            parameters = setting.get('parameters', {})
            if not isinstance(parameters, dict):
                raise ValueError('Custom isotherm parameters must be a mapping')
            for key, value in parameters.items():
                if key in {'f', 'T', 'R', 'exp', 'log', 'log10', 'sqrt', 'abs', 'min', 'max'}:
                    raise ValueError(f'Reserved custom isotherm parameter {key}')
                self.parameters[key] = float(value)
                if not math.isfinite(self.parameters[key]):
                    raise ValueError('Custom isotherm parameters must be finite')
            self.expression = SafeRateExpression(
                setting.get('expression'), self.parameters,
                scalar_names={'f', 'T', 'R'}, mapping_names=set(),
                function_names={'exp', 'log', 'log10', 'sqrt', 'abs', 'min', 'max'},
            )
        elif self.model == 'gsta_3a_water':
            self.qmax = GSTA_3A_WATER['qmax']
            self.logk = tuple(s/R-h/(R*self.T)
                             for s,h in zip(GSTA_3A_WATER['dS'],GSTA_3A_WATER['dH']))
        elif self.model in {'langmuir', 'dual_site_langmuir', 'sips', 'toth', 'henry', 'freundlich'}:
            required = {'henry':('k',), 'freundlich':('k','n'),
                        'langmuir':('qmax','b'), 'dual_site_langmuir':('qmax','b','qmax2','b2'),
                        'sips':('qmax','b','n'), 'toth':('qmax','b','n')}[self.model]
            missing = set(required)-setting.keys()
            if missing:
                raise ValueError(f'{self.model} requires isotherm parameters: {", ".join(sorted(missing))}')
            self.qmax = positive(setting.get('qmax', 1.0), 'qmax')
            self.n = positive(setting.get('n', 1.0), 'n')
            self.b = self._affinity(setting, 'b', 'heat')
            if self.model == 'dual_site_langmuir':
                self.qmax2 = positive(setting.get('qmax2'), 'qmax2')
                self.b2 = self._affinity(setting, 'b2', 'heat2')
            if self.model in {'henry', 'freundlich'}:
                self.k = self._affinity(setting, 'k', 'heat')
        else:
            raise ValueError(f'Unknown isotherm model {self.model!r}')
        if self.loading(0.0) != 0.0:
            raise ValueError('Isotherms must have zero loading at zero fugacity')

    def _affinity(self, setting, key, heat_key):
        b = positive(setting.get(key, 1.0), key)
        heat = float(setting.get(heat_key, 0.0))
        if not math.isfinite(heat):
            raise ValueError(f'{heat_key} must be finite')
        tref = positive(setting.get('T_ref', 298.15), 'T_ref')
        return b * math.exp(heat / R * (1/self.T - 1/tref))

    def loading(self, f):
        f = positive(f, 'Isotherm fugacity', zero=True)
        if self.model == 'custom':
            q = self.expression.evaluate(dict(self.parameters, f=f, T=self.T, R=R))
            return positive(q, 'Custom isotherm loading', zero=True)
        if f == 0:
            return 0.0
        if self.model == 'gsta_3a_water':
            weights = self._gsta_weights(f)
            return self.qmax/4 * math.fsum(n*w for n,w in enumerate(weights))
        if self.model == 'henry':
            return self.k * f
        if self.model == 'freundlich':
            return self.k * f**self.n
        logbf = math.log(self.b) + math.log(f)
        if self.model == 'sips':
            return self.qmax * expit(self.n*logbf)
        if self.model == 'toth':
            return self.qmax * math.exp(logbf - np.logaddexp(0, self.n*logbf)/self.n)
        q = self.qmax * expit(logbf)
        if self.model == 'dual_site_langmuir':
            q += self.qmax2 * expit(math.log(self.b2)+math.log(f))
        return float(q)

    def _gsta_logs(self, f):
        logf = math.log(f)
        return (0., *(k+n*logf for n,k in enumerate(self.logk, 1)))

    def _gsta_weights(self, f):
        logs = self._gsta_logs(f)
        normalizer = _logsumexp(logs)
        return tuple(math.exp(v-normalizer) for v in logs)

    def spreading(self, f):
        """Integral from zero to f of q(u)/u du, in mol/kg."""
        if f <= 0:
            return 0.0
        if self.model == 'gsta_3a_water':
            log_terms = _logsumexp(self._gsta_logs(f)[1:])
            # Preserve trace spreading potentials when 1 + sum(terms) rounds
            # to one; also avoid exponent overflow at large fugacities.
            softplus = (math.log1p(math.exp(log_terms)) if log_terms <= 0
                        else log_terms + math.log1p(math.exp(-log_terms)))
            return self.qmax/4 * softplus
        if self.model == 'henry':
            return self.loading(f)
        if self.model == 'freundlich':
            return self.loading(f)/self.n
        if self.model in {'langmuir', 'dual_site_langmuir', 'sips'}:
            n = self.n if self.model == 'sips' else 1.0
            value = self.qmax/n * np.logaddexp(0, n*(math.log(self.b)+math.log(f)))
            if self.model == 'dual_site_langmuir':
                value += self.qmax2*np.logaddexp(0, math.log(self.b2)+math.log(f))
            return float(value)
        # Integrate over log fugacity: u=log(f/p), Pi=int_0^inf q(f*exp(-u)) du.
        # The pressure-space integrand q(f*x)/x develops a very narrow layer
        # at x=0 for strongly adsorbing curves and large IAST fictitious f.
        # Log coordinates resolve that layer without changing the isotherm.
        value, error = quad(lambda u: self.loading(f*math.exp(-u)), 0, np.inf,
                            epsabs=1e-11, epsrel=2e-9, limit=150)
        if not math.isfinite(value) or value <= 0 or error > max(1e-9, abs(value)*1e-6):
            raise ValueError('Isotherm spreading integral is divergent or inaccurate')
        return value

    def log_fugacity_at_spreading(self, spreading):
        if self.model == 'henry':
            return math.log(spreading/self.k)
        if self.model == 'freundlich':
            return math.log(spreading*self.n/self.k)/self.n
        if self.model in {'langmuir', 'sips'}:
            n = self.n if self.model == 'sips' else 1.0
            a = spreading*n/self.qmax
            return (a + math.log(-math.expm1(-a)))/n - math.log(self.b)
        root = self._invert_log_fugacity(
            spreading, self.spreading, getattr(self, '_spreading_inverse_guess', 0.)
        )
        # A guess only: every subsequent query is bracketed and solved anew.
        self._spreading_inverse_guess = root
        return root

    @staticmethod
    def _invert_log_fugacity(value, evaluate, guess):
        """Bracket near the previous solution instead of spanning 740 log units."""
        def residual(logf):
            return evaluate(math.exp(logf)) - value
        center = min(max(guess, -690.), 690.)
        at_center = residual(center)
        if at_center == 0:
            return center
        low = high = center
        step = 1.
        if at_center < 0:
            while high < 690.:
                high = min(center+step, 690.)
                if residual(high) >= 0:
                    break
                low = high
                step *= 2
            else:
                raise IsothermRangeError('Cannot invert isotherm in representable fugacity range')
        else:
            while low > -690.:
                low = max(center-step, -690.)
                if residual(low) <= 0:
                    break
                high = low
                step *= 2
            else:
                raise IsothermRangeError('Cannot invert isotherm in representable fugacity range')
        return brentq(residual, low, high, xtol=1e-11)

    def log_fugacity_at_loading(self, loading):
        """Invert a pure loading directly, without an outer spreading solve."""
        positive(loading, 'Adsorbed loading')
        if self.model in {'langmuir', 'sips', 'toth', 'gsta_3a_water', 'dual_site_langmuir'}:
            capacity = self.qmax + (self.qmax2 if self.model == 'dual_site_langmuir' else 0.)
            if loading >= capacity:
                raise IsothermRangeError('Adsorbed inventory reaches or exceeds the pure isotherm capacity')
        return self._invert_log_fugacity(loading, self.loading, 0.)

    def validate_range(self, upper):
        """Reject negative/decreasing custom curves before equilibrium solving."""
        if self.model != 'custom':
            return
        fs = np.geomspace(max(upper*1e-12, 1e-15), max(upper, 1e-14), 65)
        qs = np.array([self.loading(float(f)) for f in fs])
        if np.any(np.diff(qs) < -1e-10*max(1., float(qs.max()))) or qs[-1] <= 0:
            raise ValueError('Custom isotherm must be nonnegative and increasing with fugacity')
        self.spreading(float(fs[-1]))

    @property
    def enthalpy_available(self):
        """Do not mistake an unidentifiable single-T heat for zero heat."""
        if self.setting.get('temperature_fit', '').startswith('single temperature'):
            return False
        if self.model == 'gsta_3a_water':
            return True
        if self.model == 'custom':
            return 'T' in self.expression.names
        return 'heat' in self.setting and (
            self.model != 'dual_site_langmuir' or 'heat2' in self.setting
        )

    def integral_enthalpy(self, f):
        """Adsorbed enthalpy relative to ideal gas at T [J/kg dry sieve].

        H_ex = R*T²*(d spreading / dT)_f. Negligible adsorbed-phase pV,
        with the same convention as the fugacity-based isotherm model.
        """
        if not self.enthalpy_available:
            raise ValueError('Adsorption enthalpy needs temperature-dependent isotherm data or explicit heats')
        if f <= 0:
            return 0.0
        if self.model == 'gsta_3a_water':
            weights = self._gsta_weights(f)
            return self.qmax/4 * math.fsum(w*h for w,h in zip(weights[1:],GSTA_3A_WATER['dH']))
        if self.model == 'dual_site_langmuir':
            first = self.qmax*expit(math.log(self.b*f))
            second = self.qmax2*expit(math.log(self.b2*f))
            return -first*self.setting['heat']-second*self.setting['heat2']
        if self.model != 'custom':
            factor = self.n if self.model == 'freundlich' else 1.
            return -self.loading(f)*self.setting['heat']/factor
        # Central derivative with Richardson error check. At fixed fugacity
        # the grand-potential derivative avoids noisy inverse isosteres.
        step = self.T*1e-4
        def difference(h):
            plus = PureIsotherm(self.setting,self.T+h).spreading(f)
            minus = PureIsotherm(self.setting,self.T-h).spreading(f)
            return (plus-minus)/(2*h)
        coarse, fine = difference(step), difference(step/2)
        value = R*self.T**2*(4*fine-coarse)/3
        error = R*self.T**2*abs(fine-coarse)/3
        if not math.isfinite(value) or error > max(1e-5,abs(value)*1e-5):
            raise ValueError('Custom isotherm temperature derivative is not numerically resolved')
        return value


def iast(isotherms, fugacities):
    """Return mixture loadings, common spreading potential and fictitious f0.

    sum(f_i/f0_i)=1 and 1/q_total=sum(x_i/q_i(f0_i)). This formulation
    remains thermodynamically consistent for unequal pure capacities.
    """
    active = [c for c in isotherms if fugacities.get(c, 0.) > 0]
    result = dict.fromkeys(isotherms, 0.0)
    if not active:
        return result, 0., {}
    if len(active) == 1:
        c = active[0]
        f = fugacities[c]
        result[c] = isotherms[c].loading(f)
        return result, isotherms[c].spreading(f), {c:f}
    pure_spread = [isotherms[c].spreading(fugacities[c]) for c in active]
    if min(pure_spread) <= 0:
        raise ValueError('Isotherm has no positive spreading potential at positive fugacity')
    logf = np.log([fugacities[c] for c in active])
    def state(logpi):
        pi = math.exp(logpi)
        logf0 = np.array([isotherms[c].log_fugacity_at_spreading(pi) for c in active])
        return logf0, _logsumexp(logf-logf0)
    low = math.log(max(pure_spread))
    high = low + math.log(len(active)+1)
    while state(high)[1] > 0:
        high += 2.
        if high > 690:
            raise ValueError('IAST could not bracket the common spreading potential')
    root = brentq(lambda v: state(v)[1], low, high, xtol=1e-11)
    logf0, _ = state(root)
    x = np.exp(logf-logf0)
    f0 = {c: math.exp(v) for c,v in zip(active,logf0)}
    qpure = np.array([isotherms[c].loading(f0[c]) for c in active])
    qtotal = 1/float(np.sum(x/qpure))
    result.update({c:float(v*qtotal) for c,v in zip(active,x)})
    return result, math.exp(root), f0


def iast_from_loadings(isotherms, loadings):
    """Invert IAST for a specified adsorbed inventory, including preloading."""
    active = [c for c in isotherms if loadings.get(c,0.) > 0]
    if not active:
        return {}, 0., {}
    if len(active) == 1:
        c = active[0]
        f = math.exp(isotherms[c].log_fugacity_at_loading(loadings[c]))
        return {c:f}, isotherms[c].spreading(f), {c:f}
    total = sum(loadings[c] for c in active)
    x = {c:loadings[c]/total for c in active}
    def at(logpi):
        pi = math.exp(logpi)
        f0 = {c:math.exp(isotherms[c].log_fugacity_at_spreading(pi)) for c in active}
        harmonic = sum(x[c]/isotherms[c].loading(f0[c]) for c in active)
        return math.log(total*harmonic), f0
    center = math.log(total)
    low = center-5
    for _ in range(100):
        try:
            if at(low)[0] >= 0:
                break
        except (OverflowError, IsothermRangeError):
            pass
        low -= 2
    else:
        raise IsothermRangeError('Cannot bracket the adsorbed inventory from below')
    # Grow from the lower bracket rather than jumping to 148*q_total.
    # A weak, small-capacity component can require exp(Pi/qmax): that
    # initial jump overflowed even when the inverse solution was moderate.
    high = low + math.log(2.)
    domain_limit = None
    for _ in range(200):
        try:
            residual, _ = at(high)
        except (OverflowError, IsothermRangeError):
            domain_limit = high
        else:
            if residual <= 0:
                break
            low = high
        high = (low+domain_limit)/2 if domain_limit is not None else high+math.log(2.)
        if high <= low:
            raise ValueError('Adsorbed inventory exceeds capacity or the representable isotherm range')
    else:
        raise ValueError('Adsorbed inventory exceeds the IAST mixture capacity')
    root = brentq(lambda v:at(v)[0],low,high,xtol=1e-11)
    f0 = at(root)[1]
    return {c:x[c]*f0[c] for c in active}, math.exp(root), f0


def iast_enthalpy(isotherms, loadings, fictitious_fugacities=None):
    """Integral mixture enthalpy relative to ideal gases [J/kg dry sieve].

    IAST has zero enthalpy of mixing at equal spreading pressure:
    H_ex = sum(q_mix_i * H_pure_i(f0_i) / q_pure_i(f0_i)).
    This is an inventory state function, not final differential heat * uptake.
    """
    active = [c for c in isotherms if loadings.get(c,0.) > 0]
    if not active:
        return 0.
    f0 = fictitious_fugacities
    if f0 is None:
        _,_,f0 = iast_from_loadings(isotherms,loadings)
    return sum(loadings[c]*isotherms[c].integral_enthalpy(f0[c])/
               isotherms[c].loading(f0[c]) for c in active)


def iast_isosteric_heats(isotherms, loadings):
    """Component differential heats [J/mol], relative to ideal-gas enthalpy.

    -R*d ln(f_i)/d(1/T) at fixed entire adsorbed inventory, not fixed bulk
    composition. The integral state function is used for finite heat duties.
    """
    active = {c:m for c,m in isotherms.items() if loadings.get(c,0.) > 0}
    if not active:
        return {}
    if not all(m.enthalpy_available for m in active.values()):
        raise ValueError('Mixture isosteric heats require temperature-dependent isotherms for all adsorbates')
    temperature = next(iter(active.values())).T
    h = temperature*1e-4
    def values(t):
        models = {c:PureIsotherm(m.setting,t) for c,m in active.items()}
        return iast_from_loadings(models,loadings)[0]
    plus,minus = values(temperature+h),values(temperature-h)
    return {c:R*temperature**2*(math.log(plus[c])-math.log(minus[c]))/(2*h) for c in active}


@lru_cache(maxsize=1)
def sieve_database():
    path = Path(__file__).resolve().parent / 'data' / 'molecular_sieve_isotherms.json'
    return json.loads(path.read_text())


def normalize_sieve(value):
    text = str(value).strip().upper().replace('Å', 'A')
    for token in ('MOLECULAR SIEVE', 'ZEOLITE', 'ANGSTROMS', 'ANGSTROM', '-', '_', ' '):
        text = text.replace(token, 'A' if token.startswith('ANGSTROM') else '')
    return text
