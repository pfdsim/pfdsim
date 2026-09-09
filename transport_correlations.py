"""Reusable one-dimensional fluid-flow and pressure-drop correlations."""

from dataclasses import dataclass
import math
from typing import Callable, Union


ScalarProfile = Union[float, Callable[[float], float]]


class TransportCorrelationError(ValueError):
    """Raised when a transport correlation receives an invalid state."""


def _finite_float(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise TransportCorrelationError(f"{label} must be finite")
    return value


def _positive_float(value: float, label: str) -> float:
    value = _finite_float(value, label)
    if value <= 0.0:
        raise TransportCorrelationError(f"{label} must be positive")
    return value


def laminar_flat_plate_transfer(reynolds: float, prandtl: float, schmidt: float) -> dict:
    """Area-average Nu/Sh for parallel laminar flow over an isothermal plate.

    Nu=0.664 Re**0.5 Pr**(1/3), with the heat/mass-transfer analogy for Sh.
    Require Re<5e5, Pr/Sc>=0.6 and thermal/solute Peclet numbers >=100.
    Sources: COMSOL Heat Transfer, External Forced Convection (plate);
    COMSOL Multiphysics Cyclopedia, What Is Mass Transfer? (flat plate).
    """
    re = _positive_float(reynolds, 'Reynolds number')
    pr = _positive_float(prandtl, 'Prandtl number')
    sc = _positive_float(schmidt, 'Schmidt number')
    if re >= 5e5 or min(pr, sc) < 0.6 or min(re * pr, re * sc) < 100:
        raise TransportCorrelationError(
            'Laminar flat-plate transfer requires Re<5e5, Pr and Sc>=0.6, '
            'and thermal and solutal Peclet numbers>=100'
        )
    return {'reynolds': re, 'prandtl': pr, 'schmidt': sc,
            'nusselt': 0.664 * math.sqrt(re) * pr**(1/3),
            'sherwood': 0.664 * math.sqrt(re) * sc**(1/3)}


def _profile_value(profile: ScalarProfile, position_m: float, label: str) -> float:
    value = profile(float(position_m)) if callable(profile) else profile
    return _finite_float(value, label)


@dataclass(frozen=True)
class CircularConduitGeometry:
    """Local geometry profiles along conduit arc length."""

    diameter_m: ScalarProfile
    roughness_m: ScalarProfile = 0.0
    elevation_gradient: ScalarProfile = 0.0

    def diameter_at(self, position_m: float) -> float:
        return _positive_float(
            _profile_value(self.diameter_m, position_m, "diameter"),
            "diameter",
        )

    def area_at(self, position_m: float) -> float:
        diameter = self.diameter_at(position_m)
        return math.pi * diameter * diameter / 4.0

    def roughness_at(self, position_m: float) -> float:
        roughness = _profile_value(self.roughness_m, position_m, "roughness")
        if roughness < 0.0:
            raise TransportCorrelationError("roughness cannot be negative")
        return roughness

    def elevation_gradient_at(self, position_m: float) -> float:
        gradient = _profile_value(
            self.elevation_gradient, position_m, "elevation gradient"
        )
        if abs(gradient) > 1.0 + 1e-12:
            raise TransportCorrelationError(
                "elevation gradient dz/ds must be between -1 and 1"
            )
        return max(-1.0, min(1.0, gradient))


@dataclass(frozen=True)
class PressureGradient:
    """Positive pressure-loss contributions along the flow direction [Pa/m]."""

    friction: float = 0.0
    gravity: float = 0.0
    acceleration: float = 0.0

    @property
    def total(self) -> float:
        return self.friction + self.gravity + self.acceleration


@dataclass(frozen=True)
class SinglePhaseFlow:
    """Local single-phase flow diagnostics and pressure gradient."""

    velocity_m_s: float
    reynolds_number: float
    darcy_friction_factor: float
    pressure_gradient: PressureGradient


@dataclass(frozen=True)
class TwoPhaseFlow:
    """Local Beggs-Brill gas-liquid flow diagnostics and pressure gradient."""

    mixture_velocity_m_s: float
    superficial_liquid_velocity_m_s: float
    superficial_vapor_velocity_m_s: float
    no_slip_liquid_fraction: float
    liquid_holdup: float
    froude_number: float
    liquid_velocity_number: float
    no_slip_reynolds_number: float
    no_slip_friction_factor: float
    two_phase_friction_multiplier: float
    darcy_friction_factor: float
    mass_quality: float
    flow_regime: str
    acceleration_factor: float
    pressure_gradient: PressureGradient


def circular_area(diameter_m: float) -> float:
    """Cross-sectional area [m2] of a circular conduit."""
    diameter = _positive_float(diameter_m, "diameter")
    return math.pi * diameter * diameter / 4.0


def flow_velocity(
    mass_flow_kg_s: float,
    density_kg_m3: float,
    area_m2: float,
) -> float:
    """Mean signed velocity [m/s] from mass continuity."""
    mass_flow = _finite_float(mass_flow_kg_s, "mass flow")
    density = _positive_float(density_kg_m3, "density")
    area = _positive_float(area_m2, "flow area")
    return mass_flow / (density * area)


def reynolds_number(
    density_kg_m3: float,
    velocity_m_s: float,
    diameter_m: float,
    viscosity_pa_s: float,
) -> float:
    """Reynolds number based on the magnitude of mean velocity."""
    density = _positive_float(density_kg_m3, "density")
    velocity = _finite_float(velocity_m_s, "velocity")
    diameter = _positive_float(diameter_m, "diameter")
    viscosity = _positive_float(viscosity_pa_s, "viscosity")
    return density * abs(velocity) * diameter / viscosity


def churchill_friction_factor(reynolds: float, relative_roughness: float = 0.0) -> float:
    """Churchill's all-regime Darcy friction-factor correlation."""
    re = _finite_float(reynolds, "Reynolds number")
    roughness = _finite_float(relative_roughness, "relative roughness")
    if re < 0.0:
        raise TransportCorrelationError("Reynolds number cannot be negative")
    if roughness < 0.0:
        raise TransportCorrelationError("relative roughness cannot be negative")
    if re == 0.0:
        return 0.0
    if re < 10.0:
        return 64.0 / re

    inner = (7.0 / re) ** 0.9 + 0.27 * roughness
    logarithm = 2.457 * math.log(1.0 / inner)
    a = logarithm ** 16
    b = (37530.0 / re) ** 16
    return 8.0 * ((8.0 / re) ** 12 + (a + b) ** -1.5) ** (1.0 / 12.0)


def _haaland_friction_factor(reynolds: float, relative_roughness: float) -> float:
    denominator = -1.8 * math.log10(
        (relative_roughness / 3.7) ** 1.11 + 6.9 / reynolds
    )
    return 1.0 / denominator**2


def _swamee_jain_friction_factor(reynolds: float, relative_roughness: float) -> float:
    denominator = math.log10(
        relative_roughness / 3.7 + 5.74 / reynolds**0.9
    )
    return 0.25 / denominator**2


def _colebrook_friction_factor(reynolds: float, relative_roughness: float) -> float:
    friction = _haaland_friction_factor(reynolds, relative_roughness)
    for _ in range(50):
        inverse_root = -2.0 * math.log10(
            relative_roughness / 3.7
            + 2.51 / (reynolds * math.sqrt(friction))
        )
        updated = 1.0 / inverse_root**2
        if abs(updated - friction) <= 1e-12 * max(updated, 1e-12):
            return updated
        friction = updated
    return friction


def darcy_friction_factor(
    reynolds: float,
    relative_roughness: float = 0.0,
    model: str = "churchill",
) -> float:
    """Return a Darcy friction factor using a selectable correlation."""
    re = _finite_float(reynolds, "Reynolds number")
    roughness = _finite_float(relative_roughness, "relative roughness")
    if re < 0.0:
        raise TransportCorrelationError("Reynolds number cannot be negative")
    if roughness < 0.0:
        raise TransportCorrelationError("relative roughness cannot be negative")
    if re == 0.0:
        return 0.0

    normalized_model = str(model).strip().lower().replace("-", "_")
    if normalized_model == "churchill":
        return churchill_friction_factor(re, roughness)
    if re < 2300.0:
        return 64.0 / re
    if normalized_model == "haaland":
        return _haaland_friction_factor(re, roughness)
    if normalized_model in {"swamee_jain", "swameejain"}:
        return _swamee_jain_friction_factor(re, roughness)
    if normalized_model == "colebrook":
        return _colebrook_friction_factor(re, roughness)
    raise TransportCorrelationError(f"Unknown friction-factor model: {model!r}")


def friction_pressure_gradient(
    darcy_factor: float,
    density_kg_m3: float,
    velocity_m_s: float,
    diameter_m: float,
) -> float:
    """Darcy-Weisbach frictional pressure loss [Pa/m]."""
    factor = _finite_float(darcy_factor, "Darcy friction factor")
    density = _positive_float(density_kg_m3, "density")
    velocity = _finite_float(velocity_m_s, "velocity")
    diameter = _positive_float(diameter_m, "diameter")
    if factor < 0.0:
        raise TransportCorrelationError("Darcy friction factor cannot be negative")
    return factor * density * velocity * velocity / (2.0 * diameter)


def gravity_pressure_gradient(
    density_kg_m3: float,
    elevation_gradient: float,
    gravity_m_s2: float = 9.80665,
) -> float:
    """Hydrostatic pressure-loss gradient rho*g*dz/ds [Pa/m]."""
    density = _positive_float(density_kg_m3, "density")
    slope = _finite_float(elevation_gradient, "elevation gradient")
    gravity = _positive_float(gravity_m_s2, "gravity")
    return density * gravity * slope


def acceleration_pressure_gradient(
    density_kg_m3: float,
    velocity_m_s: float,
    velocity_gradient_s_inv: float,
) -> float:
    """One-dimensional acceleration contribution rho*v*dv/ds [Pa/m]."""
    density = _positive_float(density_kg_m3, "density")
    velocity = _finite_float(velocity_m_s, "velocity")
    velocity_gradient = _finite_float(
        velocity_gradient_s_inv, "velocity gradient"
    )
    return density * velocity * velocity_gradient


def local_loss_pressure_drop(
    loss_coefficient: float,
    density_kg_m3: float,
    velocity_m_s: float,
) -> float:
    """Discrete fitting or area-change pressure loss K*rho*v^2/2 [Pa]."""
    coefficient = _finite_float(loss_coefficient, "loss coefficient")
    density = _positive_float(density_kg_m3, "density")
    velocity = _finite_float(velocity_m_s, "velocity")
    if coefficient < 0.0:
        raise TransportCorrelationError("loss coefficient cannot be negative")
    return coefficient * density * velocity * velocity / 2.0


def single_phase_flow(
    mass_flow_kg_s: float,
    density_kg_m3: float,
    viscosity_pa_s: float,
    diameter_m: float,
    roughness_m: float = 0.0,
    elevation_gradient: float = 0.0,
    velocity_gradient_s_inv: float = 0.0,
    friction_model: str = "churchill",
) -> SinglePhaseFlow:
    """Evaluate local circular-conduit flow and pressure-loss components."""
    mass_flow = _finite_float(mass_flow_kg_s, "mass flow")
    if mass_flow < 0.0:
        raise TransportCorrelationError(
            "single-phase flow must be evaluated along the flow direction"
        )
    diameter = _positive_float(diameter_m, "diameter")
    roughness = _finite_float(roughness_m, "roughness")
    if roughness < 0.0:
        raise TransportCorrelationError("roughness cannot be negative")
    velocity = flow_velocity(
        mass_flow, density_kg_m3, circular_area(diameter)
    )
    re = reynolds_number(
        density_kg_m3, velocity, diameter, viscosity_pa_s
    )
    factor = darcy_friction_factor(re, roughness / diameter, friction_model)
    gradient = PressureGradient(
        friction=friction_pressure_gradient(
            factor, density_kg_m3, velocity, diameter
        ),
        gravity=gravity_pressure_gradient(
            density_kg_m3, elevation_gradient
        ),
        acceleration=acceleration_pressure_gradient(
            density_kg_m3, velocity, velocity_gradient_s_inv
        ),
    )
    return SinglePhaseFlow(
        velocity_m_s=velocity,
        reynolds_number=re,
        darcy_friction_factor=factor,
        pressure_gradient=gradient,
    )


def _beggs_brill_regime(liquid_fraction: float, froude: float) -> str:
    """Classify the horizontal Beggs-Brill flow pattern region."""
    lambda_l = max(min(liquid_fraction, 1.0), 1e-12)
    fr = max(float(froude), 1e-300)
    l1 = 316.0 * lambda_l**0.302
    l2 = 0.0009252 * lambda_l**-2.4684
    l3 = 0.1 * lambda_l**-1.4516
    l4 = 0.5 * lambda_l**-6.738
    if (lambda_l < 0.01 and fr < l1) or (lambda_l >= 0.01 and fr < l2):
        return "segregated"
    if lambda_l >= 0.01 and l2 <= fr <= l3:
        return "transition"
    if (
        (0.01 <= lambda_l < 0.4 and l3 < fr <= l1)
        or (lambda_l >= 0.4 and l3 < fr <= l4)
    ):
        return "intermittent"
    return "distributed"


def _beggs_brill_horizontal_holdup(
    liquid_fraction: float,
    froude: float,
    regime: str,
) -> float:
    lambda_l = max(min(liquid_fraction, 1.0), 1e-12)
    fr = max(float(froude), 1e-300)
    coefficients = {
        "segregated": (0.98, 0.4846, 0.0868),
        "intermittent": (0.845, 0.5351, 0.0173),
        "distributed": (1.065, 0.5824, 0.0609),
    }

    def holdup(name: str) -> float:
        a, b, c = coefficients[name]
        return max(lambda_l, a * lambda_l**b / fr**c)

    if regime == "transition":
        l2 = 0.0009252 * lambda_l**-2.4684
        l3 = 0.1 * lambda_l**-1.4516
        weight = max(0.0, min(1.0, (l3 - fr) / max(l3 - l2, 1e-300)))
        return weight * holdup("segregated") + (1.0 - weight) * holdup("intermittent")
    return holdup(regime)


def _beggs_brill_inclined_holdup(
    horizontal_holdup: float,
    liquid_fraction: float,
    froude: float,
    liquid_velocity_number: float,
    angle_degrees: float,
    regime: str,
) -> float:
    theta = math.radians(_finite_float(angle_degrees, "pipe angle"))
    if abs(theta) <= 1e-15:
        return horizontal_holdup
    lambda_l = max(min(liquid_fraction, 1.0), 1e-12)
    fr = max(float(froude), 1e-300)
    nlv = max(float(liquid_velocity_number), 1e-300)

    uphill = theta > 0.0
    if uphill:
        coefficients = {
            "segregated": (0.011, -3.768, 3.539, -1.614),
            "intermittent": (2.96, 0.305, -0.4473, 0.0978),
            "distributed": (1.0, 0.0, 0.0, 0.0),
        }
    else:
        coefficients = {
            "segregated": (4.70, -0.3692, 0.1244, -0.5056),
            "intermittent": (4.70, -0.3692, 0.1244, -0.5056),
            "distributed": (4.70, -0.3692, 0.1244, -0.5056),
        }

    def corrected(name: str, base_holdup: float) -> float:
        d, e, f, h = coefficients[name]
        argument = d * lambda_l**e * nlv**f * fr**h
        c_factor = max(0.0, (1.0 - lambda_l) * math.log(max(argument, 1e-300)))
        sine = math.sin(1.8 * theta)
        psi = 1.0 + c_factor * (sine - sine**3 / 3.0)
        return max(lambda_l, base_holdup * psi)

    if regime == "transition":
        l2 = 0.0009252 * lambda_l**-2.4684
        l3 = 0.1 * lambda_l**-1.4516
        weight = max(0.0, min(1.0, (l3 - fr) / max(l3 - l2, 1e-300)))
        segregated = _beggs_brill_horizontal_holdup(lambda_l, fr, "segregated")
        intermittent = _beggs_brill_horizontal_holdup(lambda_l, fr, "intermittent")
        return (
            weight * corrected("segregated", segregated)
            + (1.0 - weight) * corrected("intermittent", intermittent)
        )
    return corrected(regime, horizontal_holdup)


def beggs_brill_two_phase_flow(
    mass_flow_kg_s: float,
    mass_quality: float,
    liquid_density_kg_m3: float,
    vapor_density_kg_m3: float,
    liquid_viscosity_pa_s: float,
    vapor_viscosity_pa_s: float,
    surface_tension_n_m: float,
    pressure_pa: float,
    diameter_m: float,
    roughness_m: float = 0.0,
    elevation_gradient: float = 0.0,
    friction_model: str = "churchill",
    gravity_m_s2: float = 9.80665,
    acceleration: bool = True,
) -> TwoPhaseFlow:
    """Evaluate Beggs-Brill gas-liquid pressure-gradient terms.

    The returned pressure-gradient components are positive pressure losses in
    the flow direction.  ``mass_quality`` is the gas mass fraction.
    """
    mass_flow = _finite_float(mass_flow_kg_s, "mass flow")
    if mass_flow < 0.0:
        raise TransportCorrelationError(
            "two-phase flow must be evaluated along the flow direction"
        )
    quality = _finite_float(mass_quality, "mass quality")
    if quality <= 0.0 or quality >= 1.0:
        raise TransportCorrelationError(
            "Beggs-Brill requires a gas mass quality between zero and one"
        )
    rho_l = _positive_float(liquid_density_kg_m3, "liquid density")
    rho_g = _positive_float(vapor_density_kg_m3, "vapor density")
    mu_l = _positive_float(liquid_viscosity_pa_s, "liquid viscosity")
    mu_g = _positive_float(vapor_viscosity_pa_s, "vapor viscosity")
    sigma = _positive_float(surface_tension_n_m, "surface tension")
    pressure = _positive_float(pressure_pa, "pressure")
    diameter = _positive_float(diameter_m, "diameter")
    roughness = _finite_float(roughness_m, "roughness")
    if roughness < 0.0:
        raise TransportCorrelationError("roughness cannot be negative")
    slope = _finite_float(elevation_gradient, "elevation gradient")
    if abs(slope) > 1.0 + 1e-12:
        raise TransportCorrelationError(
            "elevation gradient dz/ds must be between -1 and 1"
        )
    slope = max(-1.0, min(1.0, slope))
    gravity = _positive_float(gravity_m_s2, "gravity")

    area = circular_area(diameter)
    liquid_volume_flow = mass_flow * (1.0 - quality) / rho_l
    vapor_volume_flow = mass_flow * quality / rho_g
    superficial_liquid = liquid_volume_flow / area
    superficial_vapor = vapor_volume_flow / area
    mixture_velocity = superficial_liquid + superficial_vapor
    if mixture_velocity <= 0.0:
        raise TransportCorrelationError("two-phase mixture velocity must be positive")

    lambda_l = superficial_liquid / mixture_velocity
    froude = mixture_velocity * mixture_velocity / (gravity * diameter)
    nlv = 1.938 * superficial_liquid * (rho_l / (gravity * sigma)) ** 0.25
    regime = _beggs_brill_regime(lambda_l, froude)
    horizontal_holdup = _beggs_brill_horizontal_holdup(lambda_l, froude, regime)
    angle_degrees = math.degrees(math.asin(slope))
    holdup = _beggs_brill_inclined_holdup(
        horizontal_holdup,
        lambda_l,
        froude,
        nlv,
        angle_degrees,
        regime,
    )
    holdup = max(min(holdup, 1.0), lambda_l)

    rho_no_slip = rho_l * lambda_l + rho_g * (1.0 - lambda_l)
    mu_no_slip = mu_l * lambda_l + mu_g * (1.0 - lambda_l)
    re_no_slip = reynolds_number(rho_no_slip, mixture_velocity, diameter, mu_no_slip)
    f_no_slip = darcy_friction_factor(
        re_no_slip,
        roughness / diameter,
        friction_model,
    )

    y = max(lambda_l / max(holdup * holdup, 1e-300), 1e-300)
    if 1.0 < y < 1.2:
        multiplier = 2.2 * y - 1.2
    else:
        log_y = math.log(y)
        denominator = (
            -0.0523
            + 3.182 * log_y
            - 0.8725 * log_y * log_y
            + 0.01853 * log_y**4
        )
        multiplier = math.exp(log_y / denominator) if abs(denominator) > 1e-300 else 1.0
    if multiplier <= 0.0 or not math.isfinite(multiplier):
        raise TransportCorrelationError(
            "Beggs-Brill produced an invalid friction multiplier"
        )
    f_two_phase = f_no_slip * multiplier

    friction = friction_pressure_gradient(
        f_two_phase,
        rho_no_slip,
        mixture_velocity,
        diameter,
    )
    rho_holdup = rho_l * holdup + rho_g * (1.0 - holdup)
    gravity_gradient = gravity_pressure_gradient(rho_holdup, slope, gravity)
    acceleration_factor = (
        rho_no_slip * mixture_velocity * superficial_vapor / pressure
        if acceleration
        else 0.0
    )
    if acceleration_factor >= 0.95:
        raise TransportCorrelationError(
            "Beggs-Brill acceleration factor approached singular flow"
        )
    base_gradient = friction + gravity_gradient
    acceleration_gradient = (
        base_gradient * acceleration_factor / (1.0 - acceleration_factor)
        if acceleration_factor > 0.0
        else 0.0
    )

    return TwoPhaseFlow(
        mixture_velocity_m_s=mixture_velocity,
        superficial_liquid_velocity_m_s=superficial_liquid,
        superficial_vapor_velocity_m_s=superficial_vapor,
        no_slip_liquid_fraction=lambda_l,
        liquid_holdup=holdup,
        froude_number=froude,
        liquid_velocity_number=nlv,
        no_slip_reynolds_number=re_no_slip,
        no_slip_friction_factor=f_no_slip,
        two_phase_friction_multiplier=multiplier,
        darcy_friction_factor=f_two_phase,
        mass_quality=quality,
        flow_regime=regime,
        acceleration_factor=acceleration_factor,
        pressure_gradient=PressureGradient(
            friction=friction,
            gravity=gravity_gradient,
            acceleration=acceleration_gradient,
        ),
    )
