"""Shared equilibrium-stage column helpers.

This module holds the MESH model and sparse Newton machinery used by
rigorous column unit operations.  Unit-operation-specific specs, boundary
conditions, initializers, and outlet construction remain with their public
unit classes.
"""

import math
from typing import Optional

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .thermodynamics import StreamState
else:
    from thermodynamics import StreamState
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_operations_base import UnitOperationError
else:
    from unit_operations_base import UnitOperationError
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .sparse_jacobian import FixedPatternCSR, SparsePatternBuilder
    from .distillation_condenser import TotalCondenserBoundary
    from .stage_efficiency import VaporStageEfficiencies
else:
    from sparse_jacobian import FixedPatternCSR, SparsePatternBuilder
    from distillation_condenser import TotalCondenserBoundary
    from stage_efficiency import VaporStageEfficiencies


def _dogleg_step(newton, gradient, jacobian, radius):
    """Minimize along the Cauchy/Newton dogleg inside an unscaled 2-norm ball.

    A diagonally shifted Newton solve need not minimize the original quadratic.
    Discard such an endpoint when it predicts no decrease; retain Cauchy descent.
    """
    import numpy as np

    if newton is not None:
        projected = jacobian @ newton
        predicted = -float(gradient @ newton) - 0.5 * float(projected @ projected)
        if not math.isfinite(predicted) or predicted <= 0.0:
            newton = None
    if newton is not None and np.linalg.norm(newton) <= radius:
        if float(gradient @ newton) < 0.0:
            return newton
    squared_gradient = float(gradient @ gradient)
    projected = jacobian @ gradient
    curvature = float(projected @ projected)
    if squared_gradient <= 0.0 or not math.isfinite(squared_gradient) or curvature <= 0.0:
        return None
    cauchy = -(squared_gradient / curvature) * gradient
    if np.linalg.norm(cauchy) >= radius:
        return -(radius / math.sqrt(squared_gradient)) * gradient
    if newton is None or float(gradient @ newton) >= 0.0:
        return cauchy
    difference = newton - cauchy
    a = float(difference @ difference)
    b = float(cauchy @ difference)
    c = float(cauchy @ cauchy) - radius**2
    tau = (-b + math.sqrt(max(b*b - a*c, 0.0))) / a
    return cauchy + tau * difference


class EquilibriumStageColumnMixin:
    """Reusable equilibrium-stage column solver helpers."""

    def _newton_globalization(self):
        method = str(self.get_param('newton_globalization', 'line_search')).strip().lower()
        if method not in ('line_search', 'dogleg'):
            raise UnitOperationError(
                f"{type(self).__name__} '{self.unit_id}' newton_globalization must be "
                f"line_search or dogleg, got {method!r}"
            )
        return method

    def _component_order(self, inlet: StreamState) -> list[str]:
        threshold = float(self.get_param('component_solve_threshold', 1e-8))
        comps = []
        for comp in getattr(self.thermo, 'components', []):
            if (
                comp in inlet.composition
                and inlet.composition.get(comp, 0.0) > threshold
                and comp not in comps
            ):
                comps.append(comp)
        for comp in inlet.composition:
            if inlet.composition.get(comp, 0.0) > threshold and comp not in comps:
                comps.append(comp)
        if len(comps) < 2:
            comps = [
                comp for comp, _ in sorted(
                    inlet.composition.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:2]
            ]
        reference = None
        reference_selector = getattr(self, '_distillation_cut_reference_component', None)
        if callable(reference_selector):
            reference = reference_selector(inlet, comps)
        if reference in comps:
            comps = [comp for comp in comps if comp != reference] + [reference]
        return comps

    def _aggregate_feeds(
        self,
        inlets: dict[str, StreamState],
        N: int,
        default_feed_stage: int,
    ) -> tuple[StreamState, list[dict]]:
        feed_stage_map = self.get_param('feed_stages', {}) or {}
        if isinstance(feed_stage_map, str):
            parsed = {}
            for item in feed_stage_map.replace(';', ',').split(','):
                item = item.strip()
                if not item:
                    continue
                if ':' in item:
                    key, value = item.split(':', 1)
                elif '=' in item:
                    key, value = item.split('=', 1)
                else:
                    continue
                parsed[key.strip()] = value.strip()
            feed_stage_map = parsed

        feed_specs = []
        component_moles: dict[str, float] = {}
        total_flow = 0.0
        enthalpy_flow = 0.0
        vapor_flow = 0.0
        weighted_T = 0.0
        weighted_P = 0.0
        first_stream = None

        for index, (port, stream) in enumerate(inlets.items()):
            if stream.F <= 0.0:
                continue
            if first_stream is None:
                first_stream = stream
            stage = self._feed_stage_for_port(port, index, N, default_feed_stage, feed_stage_map)
            dense = dict(stream.composition)
            total = sum(max(float(value), 0.0) for value in dense.values())
            if total <= 0.0:
                raise UnitOperationError(
                    f"RigorousDistillation '{self.unit_id}' inlet '{port}' has an empty composition"
                )
            z = {comp: max(float(value), 0.0) / total for comp, value in dense.items()}
            h = stream.H
            if h is None:
                h = self.thermo.mixture_enthalpy(
                    z, stream.T, stream.vapor_fraction, P=stream.P
                )
            feed_specs.append({
                'port': port,
                'stage': stage - 1,
                'stage_number': stage,
                'F': float(stream.F),
                'z': z,
                'H': float(h),
                'T': float(stream.T),
                'P': float(stream.P),
                'vapor_fraction': float(stream.vapor_fraction),
                'vapor_composition': (dict(stream.y) if stream.y is not None
                                      else dict(z) if stream.vapor_fraction >= 1.-1e-12 else None),
            })
            total_flow += stream.F
            enthalpy_flow += stream.F * h
            vapor_flow += stream.F * stream.vapor_fraction
            weighted_T += stream.F * stream.T
            weighted_P += stream.F * stream.P
            for comp, frac in z.items():
                component_moles[comp] = component_moles.get(comp, 0.0) + stream.F * frac

        if total_flow <= 0.0 or first_stream is None:
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' requires at least one positive inlet stream"
            )

        composition = {
            comp: value / total_flow
            for comp, value in component_moles.items()
            if value > 0.0
        }
        aggregate = self.thermo.calculate_state(
            weighted_T / total_flow,
            weighted_P / total_flow,
            total_flow,
            composition,
            phase='liquid' if vapor_flow / total_flow < 0.5 else 'vapor',
            flash=False,
        )
        aggregate.vapor_fraction = vapor_flow / total_flow
        aggregate.H = enthalpy_flow / total_flow
        return aggregate, feed_specs

    def _feed_stage_for_port(
        self,
        port: str,
        index: int,
        N: int,
        default_feed_stage: int,
        feed_stage_map,
    ) -> int:
        value = None
        if isinstance(feed_stage_map, dict):
            value = feed_stage_map.get(port)
            if value is None:
                value = feed_stage_map.get(str(index))
            if value is None:
                value = feed_stage_map.get(str(index + 1))
        if value is None and index == 0:
            value = default_feed_stage
        if value is None:
            for name in (
                f'{port}_feed_stage',
                f'{port}_stage',
                f'feed_stage_{port}',
                f'stage_{port}',
            ):
                value = self.get_param(name)
                if value is not None:
                    break
        if value is None:
            for name in (
                f'{port}_feed_stage_from_bottom',
                f'{port}_stage_from_bottom',
                f'feed_stage_from_bottom_{port}',
                f'stage_from_bottom_{port}',
            ):
                bottom_value = self.get_param(name)
                if bottom_value is not None:
                    value = N - int(bottom_value) + 1
                    break
        if value is None:
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' needs a feed stage for inlet '{port}'"
            )

        stage = int(value)
        if not 1 <= stage <= N:
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' feed stage for inlet '{port}' "
                f"must be between 1 and {N}"
            )
        return stage

    def _normalize(self, composition: dict[str, float]) -> dict[str, float]:
        values = {comp: max(float(value), 0.0) for comp, value in composition.items()}
        total = sum(values.values())
        if total <= 0.0:
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' received an empty composition"
            )
        return {comp: value / total for comp, value in values.items()}

    def _match_component_name(self, comps: list[str], name: str) -> Optional[str]:
        needle = str(name).strip().lower().replace('_', ' ').replace('-', ' ')
        for comp in comps:
            normalized = comp.lower().replace('_', ' ').replace('-', ' ')
            if normalized == needle:
                return comp
        return None

    def _two_phase_stream(
        self,
        T: float,
        P: float,
        F: float,
        z: dict[str, float],
        x: dict[str, float],
        y: dict[str, float],
        vapor_fraction: float,
        H: float,
    ) -> StreamState:
        state = self.thermo.calculate_state(T, P, F, z, phase='liquid', flash=False)
        state.vapor_fraction = vapor_fraction
        state.x = dict(x)
        state.y = dict(y)
        state.H = H
        try:
            state.Cp = self.thermo.phase_weighted_mixture_Cp(
                z, T, vapor_fraction, x, y, P
            )
        except Exception:
            pass
        if state.MW is None:
            state.MW = self.thermo.mixture_MW(z)
        return state

    def _pressure_profile(self, N: int, feed_pressure: float) -> list[float]:
        import numpy as np

        profile = self.get_param('stage_pressures', self.get_param('pressure_profile'))
        if profile is not None:
            if isinstance(profile, str):
                values = [
                    float(token.strip())
                    for token in profile.replace(';', ',').split(',')
                    if token.strip()
                ]
            else:
                values = [float(value) for value in profile]
            if len(values) != N:
                raise UnitOperationError(
                    f"RigorousDistillation '{self.unit_id}' stage_pressures must have {N} values"
                )
            return values

        P_top = float(self.get_param('P_condenser', self.get_param('P_top', feed_pressure)))
        P_bottom = self.get_param('P_bottom')
        if P_bottom is not None:
            return [float(value) for value in np.linspace(P_top, float(P_bottom), N)]

        P_drop = float(self.get_param('P_drop_per_stage', 0.01))
        return [P_top + i * P_drop for i in range(N)]

    def _temperature_bounds(self, comps: list[str]) -> tuple[float, float]:
        boiling_points = [
            self.thermo.props[comp].Tb
            for comp in comps
            if getattr(self.thermo.props.get(comp), 'Tb', None)
        ]
        critical_temperatures = [
            self.thermo.props[comp].Tc
            for comp in comps
            if getattr(self.thermo.props.get(comp), 'Tc', None)
        ]
        if boiling_points:
            default_min = max(5.0, 0.45 * min(boiling_points))
            default_max = max(650.0, 1.5 * max(boiling_points))
        else:
            default_min, default_max = 100.0, 900.0
        if critical_temperatures:
            default_max = max(default_max, 1.1 * max(critical_temperatures))
        T_min = float(self.get_param('T_min', default_min))
        T_max = float(self.get_param('T_max', min(max(default_max, T_min + 100.0), 1200.0)))
        if T_min >= T_max:
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' requires T_min < T_max"
            )
        return T_min, T_max

    def _feed_thermal_condition(
        self,
        inlet: StreamState,
        composition: dict[str, float],
        pressure: float,
        T_min: float,
        T_max: float,
    ) -> float:
        H_feed = inlet.H
        if H_feed is None:
            H_feed = self.thermo.mixture_enthalpy(
                composition, inlet.T, inlet.vapor_fraction, P=inlet.P
            )

        saturation_composition = composition
        try:
            T_bubble = self._bubble_temperature_from_equation(
                saturation_composition, pressure, T_min, T_max
            )
        except Exception:
            condensables = {
                comp: frac
                for comp, frac in composition.items()
                if frac > 0.0 and self._is_likely_condensable_candidate(comp)
            }
            if condensables:
                saturation_composition = self._normalize(condensables)
                retry_T_min, retry_T_max = self._temperature_bounds(
                    list(saturation_composition)
                )
                T_bubble = self._bubble_temperature_from_equation(
                    saturation_composition,
                    pressure,
                    retry_T_min,
                    retry_T_max,
                )
            else:
                return 1.0 - inlet.vapor_fraction

        try:
            T_dew = self.thermo.dew_point_T(saturation_composition, pressure, T_bubble)
        except Exception:
            T_dew = T_bubble

        H_liquid_sat = self.thermo.mixture_enthalpy(
            composition, T_bubble, vapor_fraction=0.0, P=pressure
        )
        H_vapor_sat = self.thermo.mixture_enthalpy(
            composition, T_dew, vapor_fraction=1.0, P=pressure
        )
        latent_span = H_vapor_sat - H_liquid_sat
        if not math.isfinite(latent_span) or abs(latent_span) < 1e-9:
            return 1.0 - inlet.vapor_fraction

        q = (H_vapor_sat - float(H_feed)) / latent_span
        if not math.isfinite(q):
            return 1.0 - inlet.vapor_fraction
        return q

    def _is_likely_condensable_candidate(self, comp: str) -> bool:
        props = self.thermo.props.get(comp)
        if props is None:
            return True
        Tb = getattr(props, 'Tb', None)
        Tc = getattr(props, 'Tc', None)
        if Tb is not None and Tb >= 230.0:
            return True
        return Tc is not None and Tc >= 250.0

    def _parse_side_draws(self, N: int) -> list[dict]:
        raw = self.get_param('side_draws')
        draws = []
        if raw is None and self.get_param('side_draw_stage') is not None:
            raw = [{
                'stage': self.get_param('side_draw_stage'),
                'phase': self.get_param('side_draw_phase', 'liquid'),
                'flow': self.get_param('side_draw_flow'),
                'fraction': self.get_param('side_draw_fraction'),
                'port': self.get_param('side_draw_port', 'side'),
            }]

        if raw is None:
            return []
        if isinstance(raw, str):
            raw_items = []
            for item in raw.split(';'):
                item = item.strip()
                if not item:
                    continue
                entry = {}
                for token in item.split(','):
                    if ':' in token:
                        key, value = token.split(':', 1)
                    elif '=' in token:
                        key, value = token.split('=', 1)
                    else:
                        continue
                    entry[key.strip()] = value.strip()
                raw_items.append(entry)
            raw = raw_items
        elif isinstance(raw, dict):
            raw = [raw]

        for index, item in enumerate(raw):
            stage = int(item.get('stage', item.get('tray', item.get('stage_number', 0))))
            if not 1 <= stage <= N:
                raise UnitOperationError(
                    f"RigorousDistillation '{self.unit_id}' side draw stage must be between 1 and {N}"
                )
            phase = str(item.get('phase', 'liquid')).strip().lower()
            if phase in ('vapour', 'gas'):
                phase = 'vapor'
            if phase not in ('liquid', 'vapor'):
                raise UnitOperationError(
                    f"RigorousDistillation '{self.unit_id}' side draw phase must be liquid or vapor"
                )
            flow_value = item.get('flow', item.get('molar_flow', item.get('F')))
            fraction_value = item.get('fraction', item.get('frac'))
            if flow_value is None and fraction_value is None:
                raise UnitOperationError(
                    f"RigorousDistillation '{self.unit_id}' side draw needs flow or fraction"
                )
            if flow_value is not None and fraction_value is not None:
                raise UnitOperationError(
                    f"RigorousDistillation '{self.unit_id}' side draw cannot specify both flow and fraction"
                )
            draw = {
                'stage': stage - 1,
                'phase': phase,
                'port': str(item.get('port', f"side_{index + 1}")),
                'flow': None,
                'fraction': None,
            }
            if flow_value is not None:
                draw['flow'] = float(flow_value)
                if draw['flow'] < 0.0:
                    raise UnitOperationError(
                        f"RigorousDistillation '{self.unit_id}' side draw flow must be nonnegative"
                    )
            else:
                draw['fraction'] = float(fraction_value)
                if not 0.0 <= draw['fraction'] < 1.0:
                    raise UnitOperationError(
                        f"RigorousDistillation '{self.unit_id}' side draw fraction must be in [0, 1)"
                    )
            draws.append(draw)
        return draws

    def _dense_composition(self, composition: dict[str, float], comps: list[str]) -> dict[str, float]:
        return self._normalize({comp: composition.get(comp, 0.0) for comp in comps})

    def _pack_variables(
        self,
        T: list[float],
        x: list[dict[str, float]],
        L: list[float],
        V: list[float],
        Q_cond: float,
        Q_reb: float,
        comps: list[str],
        T_min: float,
        T_max: float,
        energy_scale: float,
        *,
        vapor_compositions=None,
    ):
        import numpy as np

        span = T_max - T_min
        values = []
        for T_stage in T:
            reduced = min(max((T_stage - T_min) / span, 1e-8), 1.0 - 1e-8)
            values.append(math.log(reduced / (1.0 - reduced)))
        for x_stage in x:
            last = max(x_stage.get(comps[-1], 0.0), 1e-14)
            for comp in comps[:-1]:
                values.append(math.log(max(x_stage.get(comp, 0.0), 1e-14) / last))
        values.extend(math.log(max(value, 1e-14)) for value in L)
        values.extend(math.log(max(value, 1e-14)) for value in V)
        values.append(Q_cond / energy_scale)
        values.append(Q_reb / energy_scale)
        efficiency = VaporStageEfficiencies(self,comps,len(T),[],len(values),len(values))
        vector = np.asarray([*values,*([0.]*efficiency.extra_size)],dtype=float)
        efficiency.pack(vector,vapor_compositions)
        return vector

    def _build_mesh_model(
        self,
        inlet: StreamState,
        feed_specs: list[dict],
        comps: list[str],
        feed_z: dict[str, float],
        N: int,
        feed_index: int,
        RR: float,
        pressures: list[float],
        condenser: str,
        condenser_vapor_fraction: float,
        side_draws: list[dict],
        distillate_spec: dict,
        flow_scale: float,
        energy_scale: float,
        component_scales: dict[str, float],
        T_min: float,
        T_max: float,
    ) -> dict:
        import numpy as np
        nc = len(comps)
        logits_start = N
        L_start = logits_start + N * (nc - 1)
        V_start = L_start + N
        Q_start = V_start + N
        n_vars = Q_start + 2
        core_rows = N*(nc+2)+2
        efficiency = VaporStageEfficiencies(self,comps,N,feed_specs,n_vars,core_rows)
        n_vars += efficiency.extra_size
        span = T_max - T_min
        condenser_boundary = TotalCondenserBoundary(self, comps, pressures[0], T_min, T_max)

        stage_feeds = [[] for _ in range(N)]
        for feed in feed_specs:
            stage = int(feed['stage'])
            if 0 <= stage < N:
                stage_feeds[stage].append(feed)

        def stage_feed_component_flow(stage: int, comp: str) -> float:
            return sum(
                feed['F'] * feed['z'].get(comp, 0.0)
                for feed in stage_feeds[stage]
            )

        def stage_feed_enthalpy_flow(stage: int) -> float:
            return sum(feed['F'] * feed['H'] for feed in stage_feeds[stage])

        def decode(vector):
            theta = np.array(vector[0:N], dtype=float)
            theta = np.clip(theta, -60.0, 60.0)
            T = T_min + span / (1.0 + np.exp(-theta))
            x = []
            idx = logits_start
            for _ in range(N):
                logits = np.array(list(vector[idx:idx + nc - 1]) + [0.0], dtype=float)
                idx += nc - 1
                logits -= np.max(logits)
                exp_values = np.exp(logits)
                fractions = exp_values / np.sum(exp_values)
                x.append({comp: float(fractions[i]) for i, comp in enumerate(comps)})
            L = np.exp(np.clip(vector[L_start:L_start + N], -40.0, 40.0))
            V = np.exp(np.clip(vector[V_start:V_start + N], -40.0, 40.0))
            Q_cond = float(vector[Q_start] * energy_scale)
            Q_reb = float(vector[Q_start + 1] * energy_scale)
            decoded = {'T': T, 'x': x, 'L': L, 'V': V, 'Q_cond': Q_cond, 'Q_reb': Q_reb}
            decoded['actual_vapor_compositions'] = [efficiency.decode(vector,j) for j in range(N)]
            return decoded

        def stage_K_values(T_stage: float, P_stage: float, x_stage: dict[str, float]):
            if hasattr(self.thermo, 'K_values'):
                try:
                    return {
                        comp: min(max(value, 1e-12), 1e12)
                        for comp, value in self.thermo.K_values(T_stage, P_stage, x_stage).items()
                    }
                except Exception:
                    pass
            if hasattr(self.thermo, 'activity_coefficients'):
                gamma = self.thermo.activity_coefficients(T_stage, x_stage)
                return {
                    comp: min(
                        max(gamma.get(comp, 1.0) * self.thermo.Psat(comp, T_stage) / P_stage, 1e-12),
                        1e12,
                    )
                    for comp in comps
                }
            return {
                comp: min(max(self.thermo.K_value(comp, T_stage, P_stage), 1e-12), 1e12)
                for comp in comps
            }

        def stage_properties(stage: int, T_stage: float, x_stage: dict[str, float], actual_y=None,
                             vapor_enthalpy=None):
            K = stage_K_values(float(T_stage), pressures[stage], x_stage)
            kx = {comp: max(K[comp] * x_stage.get(comp, 0.0), 0.0) for comp in comps}
            kx_sum = sum(kx.values())
            if kx_sum <= 0.0:
                y_stage = dict(x_stage)
            else:
                y_stage = {comp: value / kx_sum for comp, value in kx.items()}
            bubble = sum(K[comp] * x_stage.get(comp, 0.0) for comp in comps) - 1.0
            equilibrium_y = y_stage
            if actual_y is not None:
                y_stage = actual_y
            return {
                'K': K,
                'y': y_stage,
                'equilibrium_y': equilibrium_y,
                'bubble': (condenser_boundary.residual(float(T_stage), x_stage, bubble)
                           if stage == 0 else bubble),
                'hL': self.thermo.mixture_enthalpy(
                    x_stage, float(T_stage), vapor_fraction=0.0,
                    P=float(pressures[stage]),
                ),
                'hV': (vapor_enthalpy if vapor_enthalpy is not None else self.thermo.mixture_enthalpy(
                    y_stage, float(T_stage), vapor_fraction=1.0,
                    P=float(pressures[stage]),
                )),
            }

        def side_draw_flow(draw: dict, L, V, x, y) -> float:
            if draw['flow'] is not None:
                return float(draw['flow'])
            stage = draw['stage']
            base = V[stage] if draw['phase'] == 'vapor' else L[stage]
            return float(draw['fraction'] * base)

        residual_labels = []

        def residual(vector):
            decoded = decode(vector)
            T = decoded['T']
            x = decoded['x']
            L = decoded['L']
            V = decoded['V']
            Q_cond = decoded['Q_cond']
            Q_reb = decoded['Q_reb']
            props = [stage_properties(stage,T[stage],x[stage],decoded['actual_vapor_compositions'][stage])
                     for stage in range(N)]
            y = [item['y'] for item in props]
            hL = [item['hL'] for item in props]
            hV = [item['hV'] for item in props]

            residuals = []
            labels = []
            D = V[0]
            D_vapor = condenser_vapor_fraction * D
            D_liquid = (1.0 - condenser_vapor_fraction) * D

            for stage in range(N):
                liquid_in_flow = 0.0 if stage == 0 else L[stage - 1]
                liquid_in_comp = None if stage == 0 else x[stage - 1]
                liquid_in_h = 0.0 if stage == 0 else hL[stage - 1]

                vapor_in_flow = 0.0 if stage == N - 1 else V[stage + 1]
                vapor_in_comp = None if stage == N - 1 else y[stage + 1]
                vapor_in_h = 0.0 if stage == N - 1 else hV[stage + 1]

                heat = Q_cond if stage == 0 else (Q_reb if stage == N - 1 else 0.0)

                stage_side_draws = [draw for draw in side_draws if draw['stage'] == stage]
                for comp in comps:
                    in_comp = stage_feed_component_flow(stage, comp)
                    if liquid_in_comp is not None:
                        in_comp += liquid_in_flow * liquid_in_comp.get(comp, 0.0)
                    if vapor_in_comp is not None:
                        in_comp += vapor_in_flow * vapor_in_comp.get(comp, 0.0)

                    if stage == 0:
                        out_comp = (
                            (L[stage] + D_liquid) * x[stage].get(comp, 0.0)
                            + D_vapor * y[stage].get(comp, 0.0)
                        )
                    elif stage == N - 1:
                        out_comp = L[stage] * x[stage].get(comp, 0.0) + V[stage] * y[stage].get(comp, 0.0)
                    else:
                        out_comp = L[stage] * x[stage].get(comp, 0.0) + V[stage] * y[stage].get(comp, 0.0)

                    for draw in stage_side_draws:
                        draw_flow = side_draw_flow(draw, L, V, x, y)
                        if draw['phase'] == 'vapor':
                            out_comp += draw_flow * y[stage].get(comp, 0.0)
                        else:
                            out_comp += draw_flow * x[stage].get(comp, 0.0)

                    residuals.append((in_comp - out_comp) / component_scales[comp])
                    labels.append(('component', stage + 1, comp, component_scales[comp]))

                in_energy = (
                    liquid_in_flow * liquid_in_h
                    + vapor_in_flow * vapor_in_h
                    + stage_feed_enthalpy_flow(stage)
                    + heat
                )
                if stage == 0:
                    out_energy = (L[stage] + D_liquid) * hL[stage] + D_vapor * hV[stage]
                else:
                    out_energy = L[stage] * hL[stage] + V[stage] * hV[stage]
                for draw in stage_side_draws:
                    draw_flow = side_draw_flow(draw, L, V, x, y)
                    out_energy += draw_flow * (hV[stage] if draw['phase'] == 'vapor' else hL[stage])
                residuals.append((in_energy - out_energy) / energy_scale)
                labels.append(('energy', stage + 1, None, energy_scale))

                residuals.append(props[stage]['bubble'])
                labels.append(('condenser_temperature' if stage == 0 and condenser_boundary.options
                               else 'bubble', stage + 1, None, 1.0))

            if distillate_spec['kind'] == 'molar':
                residuals.append((D - distillate_spec['value']) / flow_scale)
                labels.append(('distillate_spec', None, None, flow_scale))
            else:
                product_mw = sum(
                    (
                        (1.0 - condenser_vapor_fraction) * x[0].get(comp, 0.0)
                        + condenser_vapor_fraction * y[0].get(comp, 0.0)
                    ) * self.thermo.props[comp].MW
                    for comp in comps
                )
                mass_scale = max(abs(distillate_spec['value']), flow_scale * product_mw, 1.0)
                residuals.append((D * product_mw - distillate_spec['value']) / mass_scale)
                labels.append(('distillate_spec', None, None, mass_scale))

            residuals.append((L[0] - RR * D) / flow_scale)
            labels.append(('reflux_spec', None, None, flow_scale))
            residuals.extend(efficiency.residuals(props,V))
            labels.extend(('murphree',j+1,c,1.) for j in efficiency.active for c in comps[:-1])

            if not residual_labels:
                residual_labels.extend(labels)
            return np.array(residuals, dtype=float)

        def sparsity():
            n_rows = core_rows+efficiency.extra_size
            matrix = SparsePatternBuilder((n_rows, n_vars))

            def mark_stage(row: int, stage: int):
                if not 0 <= stage < N:
                    return
                matrix.mark(row, stage)
                start = logits_start + stage * (nc - 1)
                matrix.mark_range(row, start, start + nc - 1)
                matrix.mark(row, L_start + stage)
                matrix.mark(row, V_start + stage)
                for col in efficiency.columns.get(stage,()):
                    matrix.mark(row,col)

            row = 0
            for stage in range(N):
                local_stages = {stage}
                if stage > 0:
                    local_stages.add(stage - 1)
                if stage < N - 1:
                    local_stages.add(stage + 1)
                for _ in comps:
                    for local in local_stages:
                        mark_stage(row, local)
                    if stage == 0:
                        matrix.mark(row, V_start)
                    row += 1
                for local in local_stages:
                    mark_stage(row, local)
                if stage == 0:
                    matrix.mark(row, V_start)
                    matrix.mark(row, Q_start)
                if stage == N - 1:
                    matrix.mark(row, Q_start + 1)
                row += 1
                mark_stage(row, stage)
                row += 1

            mark_stage(row, 0)
            matrix.mark(row, V_start)
            row += 1
            matrix.mark(row, L_start)
            matrix.mark(row, V_start)
            row += 1
            efficiency.mark_sparsity(matrix,lambda j:(j,*range(logits_start+j*(nc-1),
                logits_start+(j+1)*(nc-1)),V_start+j))
            return matrix.tocsr()

        sparsity_matrix = sparsity()
        jacobian_pattern = FixedPatternCSR(sparsity_matrix)

        dense_jacobian_mb = (
            sparsity_matrix.shape[0]
            * sparsity_matrix.shape[1]
            * 8.0
            / (1024.0 * 1024.0)
        )
        dense_limit_mb = float(self.get_param(
            'semi_analytic_dense_limit_mb',
            200.0,
        ))
        local_thermo_available = (
            condenser in ('total', 'partial', 'mixed')
            and not side_draws
            and distillate_spec.get('kind') in ('molar', 'mass')
            and dense_limit_mb > 0.0
            and dense_jacobian_mb <= dense_limit_mb
        )

        def semi_analytic_flow_jacobian(
            vector,
            f0,
            rel_step: float,
        ):
            if (
                not self._truthy_param(self.get_param('semi_analytic_flow_jacobian', True))
                or condenser not in ('total', 'partial', 'mixed')
                or side_draws
                or distillate_spec.get('kind') not in ('molar', 'mass')
            ):
                return None

            decoded = decode(vector)
            T = decoded['T']
            x = decoded['x']
            L = decoded['L']
            V = decoded['V']
            props = [stage_properties(stage,T[stage],x[stage],decoded['actual_vapor_compositions'][stage])
                     for stage in range(N)]
            y = [item['y'] for item in props]
            hL = [item['hL'] for item in props]
            hV = [item['hV'] for item in props]
            beta = float(condenser_vapor_fraction)
            use_local_thermo = (
                local_thermo_available
                and self._truthy_param(
                    self.get_param('semi_analytic_local_thermo_jacobian', True)
                )
            )
            J = jacobian_pattern.empty()
            evaluations = 0

            def add(row: int, col: int, value: float) -> None:
                J.add(row, col, value)

            if use_local_thermo:
                mass_scale = None
                mass_scale_depends_on_product_mw = False
                mass_spec_numerator = None
                if distillate_spec.get('kind') == 'mass':
                    product_mw = sum(
                        (
                            (1.0 - beta) * x[0].get(comp, 0.0)
                            + beta * y[0].get(comp, 0.0)
                        ) * self.thermo.props[comp].MW
                        for comp in comps
                    )
                    mass_scale = max(
                        abs(float(distillate_spec['value'])),
                        flow_scale * product_mw,
                        1.0,
                    )
                    mass_scale_depends_on_product_mw = (
                        flow_scale * product_mw
                        >= max(abs(float(distillate_spec['value'])), 1.0)
                    )
                    mass_spec_numerator = (
                        V[0] * product_mw - float(distillate_spec['value'])
                    )

                for stage in range(N):
                    local_columns = [stage]
                    local_columns.extend(range(
                        logits_start + stage * (nc - 1),
                        logits_start + (stage + 1) * (nc - 1),
                    ))
                    local_columns.extend(efficiency.columns.get(stage,()))
                    for col in local_columns:
                        step = rel_step * max(abs(vector[col]), 1.0)
                        T_perturbed = float(T[stage])
                        x_perturbed = x[stage]
                        actual_y = decoded['actual_vapor_compositions'][stage]
                        if col == stage:
                            theta = min(max(float(vector[col] + step), -60.0), 60.0)
                            T_perturbed = T_min + span / (1.0 + math.exp(-theta))
                        elif col in efficiency.columns.get(stage,()):
                            trial = vector.copy()
                            trial[col] += step
                            actual_y = efficiency.decode(trial,stage)
                        else:
                            start = logits_start + stage * (nc - 1)
                            logits = np.array(
                                list(vector[start:start + nc - 1]) + [0.0],
                                dtype=float,
                            )
                            logits[col - start] += step
                            logits -= np.max(logits)
                            fractions = np.exp(logits)
                            fractions /= np.sum(fractions)
                            x_perturbed = {
                                comp: float(fractions[index])
                                for index, comp in enumerate(comps)
                            }

                        if col in efficiency.columns.get(stage,()):
                            perturbed = efficiency.actual_properties(props[stage],stage,T_perturbed,pressures[stage],actual_y)
                        else:
                            perturbed = stage_properties(stage,T_perturbed,x_perturbed,actual_y,
                                vapor_enthalpy=hV[stage] if actual_y is not None and col != stage else None)
                        dx = {
                            comp: (
                                x_perturbed.get(comp, 0.0)
                                - x[stage].get(comp, 0.0)
                            ) / step
                            for comp in comps
                        }
                        dy = {
                            comp: (
                                perturbed['y'].get(comp, 0.0)
                                - y[stage].get(comp, 0.0)
                            ) / step
                            for comp in comps
                        }
                        dhL = (perturbed['hL'] - hL[stage]) / step
                        dhV = (perturbed['hV'] - hV[stage]) / step
                        dbubble = (
                            perturbed['bubble'] - props[stage]['bubble']
                        ) / step

                        for ci, comp in enumerate(comps):
                            scale = component_scales[comp]
                            liquid_coefficient = L[stage]
                            vapor_coefficient = V[stage]
                            if stage == 0:
                                liquid_coefficient += (1.0 - beta) * V[0]
                                vapor_coefficient = beta * V[0]
                            add(
                                stage * (nc + 2) + ci,
                                col,
                                -(
                                    liquid_coefficient * dx[comp]
                                    + vapor_coefficient * dy[comp]
                                ) / scale,
                            )
                            if stage < N - 1:
                                add(
                                    (stage + 1) * (nc + 2) + ci,
                                    col,
                                    L[stage] * dx[comp] / scale,
                                )
                            if stage > 0:
                                add(
                                    (stage - 1) * (nc + 2) + ci,
                                    col,
                                    V[stage] * dy[comp] / scale,
                                )

                        energy_row = stage * (nc + 2) + nc
                        liquid_coefficient = L[stage]
                        vapor_coefficient = V[stage]
                        if stage == 0:
                            liquid_coefficient += (1.0 - beta) * V[0]
                            vapor_coefficient = beta * V[0]
                        add(
                            energy_row,
                            col,
                            -(
                                liquid_coefficient * dhL
                                + vapor_coefficient * dhV
                            ) / energy_scale,
                        )
                        if stage < N - 1:
                            add(
                                (stage + 1) * (nc + 2) + nc,
                                col,
                                L[stage] * dhL / energy_scale,
                            )
                        if stage > 0:
                            add(
                                (stage - 1) * (nc + 2) + nc,
                                col,
                                V[stage] * dhV / energy_scale,
                            )
                        add(energy_row + 1, col, dbubble)
                        efficiency.add_local_derivatives(add,stage,col,props,perturbed,step,V)

                        if stage == 0 and mass_scale is not None:
                            d_product_mw = sum(
                                (
                                    (1.0 - beta) * dx[comp]
                                    + beta * dy[comp]
                                ) * self.thermo.props[comp].MW
                                for comp in comps
                            )
                            add(
                                N * (nc + 2),
                                col,
                                (
                                    V[0] * d_product_mw * mass_scale
                                    - mass_spec_numerator
                                    * (
                                        flow_scale * d_product_mw
                                        if mass_scale_depends_on_product_mw
                                        else 0.0
                                    )
                                ) / (mass_scale * mass_scale),
                            )
            else:
                column_rows = [
                    sparsity_matrix[:, col].nonzero()[0]
                    for col in range(sparsity_matrix.shape[1])
                ]
                nonlinear_columns = set(range(N))
                nonlinear_columns.update(
                    range(logits_start, logits_start + N * (nc - 1))
                )
                for columns in efficiency.columns.values():
                    nonlinear_columns.update(columns)
                restricted_groups = []
                group_rows = []
                for col in sorted(nonlinear_columns):
                    rows = set(column_rows[col].tolist())
                    for index, used_rows in enumerate(group_rows):
                        if rows.isdisjoint(used_rows):
                            restricted_groups[index].append(col)
                            used_rows.update(rows)
                            break
                    else:
                        restricted_groups.append([col])
                        group_rows.append(set(rows))
                for group in restricted_groups:
                    perturbation = np.zeros_like(vector)
                    for col in group:
                        perturbation[col] = rel_step * max(abs(vector[col]), 1.0)
                    f_step = residual(vector + perturbation)
                    evaluations += 1
                    diff = f_step - f0
                    for col in group:
                        rows = column_rows[col]
                        if rows.size:
                            J.set_column(
                                rows,
                                col,
                                diff[rows] / perturbation[col],
                            )

            for stage in range(N):
                efficiency.add_flow_derivatives(add,stage,V_start+stage,props,V)
                for ci, comp in enumerate(comps):
                    row = stage * (nc + 2) + ci
                    scale = component_scales[comp]
                    if stage > 0:
                        add(
                            row,
                            L_start + stage - 1,
                            L[stage - 1] * x[stage - 1].get(comp, 0.0) / scale,
                        )
                    if stage < N - 1:
                        add(
                            row,
                            V_start + stage + 1,
                            V[stage + 1] * y[stage + 1].get(comp, 0.0) / scale,
                        )
                    if stage == 0:
                        top_product_comp = (
                            (1.0 - beta) * x[0].get(comp, 0.0)
                            + beta * y[0].get(comp, 0.0)
                        )
                        add(row, L_start, -L[0] * x[0].get(comp, 0.0) / scale)
                        add(row, V_start, -V[0] * top_product_comp / scale)
                    else:
                        add(row, L_start + stage, -L[stage] * x[stage].get(comp, 0.0) / scale)
                        add(row, V_start + stage, -V[stage] * y[stage].get(comp, 0.0) / scale)

                energy_row = stage * (nc + 2) + nc
                if stage > 0:
                    add(
                        energy_row,
                        L_start + stage - 1,
                        L[stage - 1] * hL[stage - 1] / energy_scale,
                    )
                if stage < N - 1:
                    add(
                        energy_row,
                        V_start + stage + 1,
                        V[stage + 1] * hV[stage + 1] / energy_scale,
                    )
                if stage == 0:
                    top_product_h = (1.0 - beta) * hL[0] + beta * hV[0]
                    add(energy_row, L_start, -L[0] * hL[0] / energy_scale)
                    add(energy_row, V_start, -V[0] * top_product_h / energy_scale)
                    add(energy_row, Q_start, 1.0)
                else:
                    add(energy_row, L_start + stage, -L[stage] * hL[stage] / energy_scale)
                    add(energy_row, V_start + stage, -V[stage] * hV[stage] / energy_scale)
                    if stage == N - 1:
                        add(energy_row, Q_start + 1, 1.0)

            spec_row = N * (nc + 2)
            if distillate_spec.get('kind') == 'mass':
                product_mw = sum(
                    (
                        (1.0 - beta) * x[0].get(comp, 0.0)
                        + beta * y[0].get(comp, 0.0)
                    ) * self.thermo.props[comp].MW
                    for comp in comps
                )
                mass_scale = max(abs(float(distillate_spec['value'])), flow_scale * product_mw, 1.0)
                add(spec_row, V_start, V[0] * product_mw / mass_scale)
            else:
                add(spec_row, V_start, V[0] / flow_scale)
            add(spec_row + 1, L_start, L[0] / flow_scale)
            add(spec_row + 1, V_start, -RR * V[0] / flow_scale)
            return (
                J.tocsr(),
                evaluations,
                (
                    'semi_analytic_local_thermo'
                    if use_local_thermo else 'semi_analytic_flow'
                ),
            )

        return {
            'decode': decode,
            'residual': residual,
            'sparsity': sparsity_matrix,
            'jacobian': semi_analytic_flow_jacobian,
            'local_jacobian_available': local_thermo_available,
            'jacobian_dense_mb': dense_jacobian_mb,
            'stage_properties': stage_properties,
            'side_draw_flow': side_draw_flow,
            'residual_labels': residual_labels,
            'condenser_boundary': condenser_boundary,
            'efficiency': efficiency,
        }

    def _sparse_newton_solve(self, residual, sparsity, x0, options: dict,
                             jacobian=None, step_event=None) -> dict:
        globalization = self._newton_globalization()
        quality_context = getattr(getattr(self, 'thermo', None), 'quality_context', None)
        if (
            quality_context is not None
            and not getattr(self, '_quality_solver_aux_context_active', False)
        ):
            self._quality_solver_aux_context_active = True
            try:
                with quality_context(phase='solver_iteration', affects_result=False):
                    return self._sparse_newton_solve(
                        residual,
                        sparsity,
                        x0,
                        options,
                        jacobian=jacobian,
                        step_event=step_event,
                    )
            finally:
                self._quality_solver_aux_context_active = False

        import numpy as np
        from scipy.sparse import csc_matrix, eye
        from scipy.sparse.linalg import MatrixRankWarning, spsolve
        import warnings as py_warnings

        x = np.array(x0, dtype=float)
        f = residual(x)
        if not np.all(np.isfinite(f)):
            raise UnitOperationError(
                f"{type(self).__name__} '{self.unit_id}' generated non-finite initial residuals"
            )
        tolerance = options['mesh_tolerance']
        acceptable_tolerance = max(options.get('acceptable_mesh_residual', tolerance), tolerance)
        max_iterations = options['max_iterations']
        max_jacobians = options['max_jacobian_evaluations']
        line_search_steps = options['line_search_steps']
        rel_step = options['finite_difference_rel_step']
        stall_iterations = max(0, int(options.get('stall_iterations', 0)))
        stall_relative_tolerance = max(
            0.0, float(options.get('stall_relative_tolerance', 1e-4))
        )

        groups = None
        function_evaluations = 1
        jacobian_evaluations = 0
        used_model_jacobian = False
        model_jacobian_method = 'semi_analytic_flow'
        message = "maximum iterations reached"
        last_iteration = 0
        stall_best_residual = math.inf
        stall_count = 0
        rejected_steps = 0
        radius = math.sqrt(x.size)
        maximum_radius = 8.0 * radius

        for iteration in range(1, max_iterations + 1):
            last_iteration = iteration
            residual_norm = float(np.linalg.norm(f, ord=np.inf))
            if residual_norm < tolerance:
                return {
                    'success': True,
                    'x': x,
                    'residual_norm': residual_norm,
                    'iterations': iteration - 1,
                    'function_evaluations': function_evaluations,
                    'jacobian_evaluations': jacobian_evaluations,
                    'jacobian_method': (
                        model_jacobian_method
                        if used_model_jacobian else 'colored_finite_difference'
                    ),
                    'message': 'converged',
                    'newton_globalization': globalization,
                    'rejected_steps': rejected_steps,
                }
            if stall_iterations:
                if not math.isfinite(stall_best_residual):
                    stall_best_residual = residual_norm
                else:
                    required_progress = max(
                        stall_relative_tolerance * stall_best_residual,
                        1e-12,
                    )
                    if residual_norm < stall_best_residual - required_progress:
                        stall_best_residual = residual_norm
                        stall_count = 0
                    else:
                        stall_best_residual = min(
                            stall_best_residual, residual_norm
                        )
                        stall_count += 1
                    if stall_count >= stall_iterations:
                        message = (
                            "residual stalled for "
                            f"{stall_iterations} iterations"
                        )
                        break
            if jacobian_evaluations >= max_jacobians:
                message = "maximum Jacobian evaluations reached"
                break

            try:
                jacobian_result = (
                    jacobian(x, f, rel_step) if jacobian is not None else None
                )
            except Exception as exc:
                progress = getattr(exc, "add_solver_progress", None)
                if callable(progress):
                    progress(
                        iterations=iteration,
                        function_evaluations=function_evaluations,
                        jacobian_evaluations=jacobian_evaluations + 1,
                    )
                raise
            if jacobian_result is None:
                if groups is None:
                    groups = self._color_jacobian_columns(sparsity)
                J, evals = self._finite_difference_jacobian(
                    residual, x, f, sparsity, groups, rel_step
                )
            else:
                if len(jacobian_result) == 3:
                    J, evals, model_jacobian_method = jacobian_result
                else:
                    J, evals = jacobian_result
                used_model_jacobian = True
            function_evaluations += evals
            jacobian_evaluations += 1

            dx = None
            for shift in (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
                try:
                    with py_warnings.catch_warnings():
                        py_warnings.simplefilter('error', MatrixRankWarning)
                        matrix = csc_matrix(J)
                        if shift:
                            matrix = matrix + shift * eye(matrix.shape[0], matrix.shape[1], format='csc')
                        trial_dx = spsolve(matrix, -f)
                    if np.all(np.isfinite(trial_dx)):
                        dx = np.array(trial_dx, dtype=float)
                        break
                except Exception:
                    continue
            if dx is None and globalization == 'line_search':
                message = "linear Newton system could not be solved"
                break

            if step_event is not None and dx is not None:
                try:
                    step_event(x, f, dx)
                except Exception as exc:
                    progress = getattr(exc, "add_solver_progress", None)
                    if callable(progress):
                        progress(
                            iterations=iteration,
                            function_evaluations=function_evaluations,
                            jacobian_evaluations=jacobian_evaluations,
                        )
                    raise

            if globalization == 'dogleg':
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from .thermodynamics_models.common import ThermodynamicsError
                else:
                    from thermodynamics_models.common import ThermodynamicsError
                gradient = np.asarray(J.T @ f).ravel()
                merit = 0.5 * float(f @ f)
                accepted = False
                message = 'trust region could not reduce the MESH residual'
                # Independent of line_search_steps: rejected trials reuse J.
                for _ in range(16):
                    step = _dogleg_step(dx, gradient, J, radius)
                    if step is None:
                        message = 'zero or invalid least-squares gradient'
                        break
                    projected = J @ step
                    predicted = -float(gradient @ step) - 0.5 * float(projected @ projected)
                    if predicted <= 0.0 or not math.isfinite(predicted):
                        message = 'nonpositive trust-region model reduction'
                        break
                    trial_x = x + step
                    function_evaluations += 1
                    try:
                        trial_f = residual(trial_x)
                    except ThermodynamicsError:
                        # Initial/Jacobian errors propagate; a trial can leave
                        # the physical property domain and contract the radius.
                        trial_f = np.full_like(f, np.nan)
                    ratio = (
                        (merit - 0.5 * float(trial_f @ trial_f)) / predicted
                        if np.all(np.isfinite(trial_f)) else -math.inf
                    )
                    step_norm = float(np.linalg.norm(step))
                    if ratio < 0.25:
                        radius = 0.25 * step_norm
                    elif ratio > 0.75 and step_norm > 0.95 * radius:
                        radius = min(2.0 * radius, maximum_radius)
                    if ratio > 0.1:
                        x, f = trial_x, trial_f
                        accepted = True
                        break
                    rejected_steps += 1
                    if radius < 1e-12:
                        message = 'trust-region radius underflow'
                        break
                if not accepted:
                    break
                message = 'maximum iterations reached'
                continue

            max_abs_step = float(np.max(np.abs(dx))) if dx.size else 0.0
            step_limit = float(options.get('newton_step_limit', 8.0))
            if max_abs_step > step_limit:
                dx *= step_limit / max_abs_step

            current_merit = 0.5 * float(np.dot(f, f))
            accepted = False
            best_x = x
            best_f = f
            best_merit = current_merit
            for attempt in range(line_search_steps):
                alpha = 0.5 ** attempt
                x_trial = x + alpha * dx
                f_trial = residual(x_trial)
                function_evaluations += 1
                if not np.all(np.isfinite(f_trial)):
                    rejected_steps += 1
                    continue
                trial_merit = 0.5 * float(np.dot(f_trial, f_trial))
                if trial_merit < best_merit:
                    best_merit = trial_merit
                    best_x = x_trial
                    best_f = f_trial
                if trial_merit <= current_merit * (1.0 - 1e-4 * alpha):
                    x = x_trial
                    f = f_trial
                    accepted = True
                    break
                rejected_steps += 1
            if not accepted:
                if best_merit < current_merit:
                    x = best_x
                    f = best_f
                else:
                    message = "line search could not reduce the MESH residual"
                    break

        residual_norm = float(np.linalg.norm(f, ord=np.inf))
        return {
            'success': residual_norm < acceptable_tolerance,
            'x': x,
            'residual_norm': residual_norm,
            'iterations': last_iteration,
            'function_evaluations': function_evaluations,
            'jacobian_evaluations': jacobian_evaluations,
            'jacobian_method': (
                model_jacobian_method
                if used_model_jacobian else 'colored_finite_difference'
            ),
            'message': message,
            'newton_globalization': globalization,
            'rejected_steps': rejected_steps,
        }

    def _color_jacobian_columns(self, sparsity):
        column_rows = [
            set(sparsity[:, col].nonzero()[0].tolist())
            for col in range(sparsity.shape[1])
        ]
        groups: list[list[int]] = []
        group_rows: list[set[int]] = []
        for col, rows in enumerate(column_rows):
            for index, used_rows in enumerate(group_rows):
                if rows.isdisjoint(used_rows):
                    groups[index].append(col)
                    used_rows.update(rows)
                    break
            else:
                groups.append([col])
                group_rows.append(set(rows))
        return groups

    def _finite_difference_jacobian(self, residual, x, f0, sparsity, groups, rel_step: float):
        import numpy as np

        J = FixedPatternCSR(sparsity).empty()
        evaluations = 0
        column_rows = [
            sparsity[:, col].nonzero()[0]
            for col in range(sparsity.shape[1])
        ]
        for group in groups:
            step = np.zeros_like(x)
            for col in group:
                step[col] = rel_step * max(abs(x[col]), 1.0)
            f_step = residual(x + step)
            evaluations += 1
            diff = f_step - f0
            for col in group:
                rows = column_rows[col]
                if rows.size:
                    J.set_column(rows, col, diff[rows] / step[col])
        return J.tocsr(), evaluations

    def _bubble_temperature_from_equation(
        self,
        composition: dict[str, float],
        P: float,
        T_min: Optional[float] = None,
        T_max: Optional[float] = None,
    ) -> float:
        from scipy.optimize import brentq

        comps = list(composition.keys())
        if T_min is None or T_max is None:
            T_min_local, T_max_local = self._temperature_bounds(comps)
            T_min = T_min if T_min is not None else T_min_local
            T_max = T_max if T_max is not None else T_max_local

        def bubble_residual(T):
            if hasattr(self.thermo, 'activity_coefficients'):
                gamma = self.thermo.activity_coefficients(T, composition)
                return sum(
                    composition.get(comp, 0.0) * gamma.get(comp, 1.0) * self.thermo.Psat(comp, T) / P
                    for comp in comps
                ) - 1.0
            K_values = self.thermo.K_values(T, P, composition)
            return sum(
                composition.get(comp, 0.0) * K_values.get(comp, 1.0)
                for comp in comps
            ) - 1.0

        lo = max(1.0, float(T_min))
        hi = float(T_max)
        f_lo = bubble_residual(lo)
        f_hi = bubble_residual(hi)
        if f_lo * f_hi <= 0.0:
            return float(brentq(bubble_residual, lo, hi, xtol=1e-8, rtol=1e-10, maxiter=100))

        try:
            return float(self.thermo.bubble_point_T(composition, P))
        except Exception:
            if abs(f_lo) < abs(f_hi):
                return lo
            return hi

    def _external_component_balance_error_vle(
        self,
        comps: list[str],
        inlet: StreamState,
        feed_z: dict[str, float],
        distillate: StreamState,
        bottoms: StreamState,
        side_streams: list[StreamState],
        flow_scale: float,
    ) -> float:
        max_error = 0.0
        for comp in comps:
            out = (
                distillate.F * distillate.composition.get(comp, 0.0)
                + bottoms.F * bottoms.composition.get(comp, 0.0)
            )
            for stream in side_streams:
                out += stream.F * stream.composition.get(comp, 0.0)
            inc = inlet.F * feed_z.get(comp, 0.0)
            max_error = max(max_error, abs(out - inc) / flow_scale)
        return max_error

    def _rigorous2_residual_diagnostics(self, residual_values, labels: list[tuple]) -> dict:
        diagnostics = {}
        for value, (kind, stage, comp, scale) in zip(residual_values, labels):
            scaled = float(value)
            unscaled = scaled * scale
            bucket = diagnostics.setdefault(kind, {
                'count': 0,
                'max_scaled_abs': 0.0,
                'rms_scaled': 0.0,
                'max_unscaled_abs': 0.0,
                'max_stage': None,
                'max_component': None,
            })
            bucket['count'] += 1
            bucket['rms_scaled'] += scaled * scaled
            if abs(scaled) > bucket['max_scaled_abs']:
                bucket['max_scaled_abs'] = abs(scaled)
                bucket['max_unscaled_abs'] = abs(unscaled)
                bucket['max_stage'] = stage
                bucket['max_component'] = comp
        for bucket in diagnostics.values():
            if bucket['count']:
                bucket['rms_scaled'] = math.sqrt(bucket['rms_scaled'] / bucket['count'])
        return diagnostics
