"""Shared equilibrium diagrams, using each model's authoritative solvers.

Failed points remain explicit gaps. Split phases are checked against material
balance and chemical-potential equality before they enter a diagram.
"""

import math
from scipy.optimize import brentq


class PhaseDiagramMixin:
    @staticmethod
    def _diagram_controls(n_points, T=None, P=None):
        if isinstance(n_points, bool) or not isinstance(n_points, int) or n_points < 2:
            raise ValueError("n_points must be an integer of at least 2")
        if T is not None and (not math.isfinite(T) or T <= 0):
            raise ValueError("Temperature must be positive Kelvin")
        if P is not None and (not math.isfinite(P) or P <= 0):
            raise ValueError("Pressure must be positive bar")

    def _diagram_components(self, components):
        if len(set(components)) != len(components) or any(
            c not in self.components for c in components
        ):
            raise ValueError("Choose distinct components from this thermodynamic model")

    def _check_diagram_vle(self, x, y, T, P):
        # K-values can contain single-phase extrapolations used to initialize
        # flashes. Audit the phases using the shared physical activity path.
        phase_log_fugacities = getattr(self, "_phase_log_fugacities", None)
        if callable(phase_log_fugacities):
            liquid = phase_log_fugacities(T, P, x, "liquid")
            vapor = phase_log_fugacities(T, P, y, "vapor")
        else:
            liquid = {
                c: math.log(max(v, 1e-300))
                for c, v in self.component_activities(T, P, x, "liquid").items()
            }
            vapor = {
                c: math.log(max(v, 1e-300))
                for c, v in self.component_activities(T, P, y, "vapor").items()
            }
        residual = max(
            abs(liquid[c] - vapor[c]) for c in x if x[c] > 1e-12 or y.get(c, 0) > 1e-12
        )
        if not math.isfinite(residual) or residual > 1e-4:
            raise ValueError(
                f"VLE failed fugacity equality (log residual {residual:g})"
            )
        if not callable(getattr(self, "activity_coefficients", None)) and callable(
            getattr(self, "molar_volume", None)
        ):
            liquid_volume = self.molar_volume(T, P, x, "liquid")
            vapor_volume = self.molar_volume(T, P, y, "vapor")
            if abs(liquid_volume - vapor_volume) <= 1e-6 * max(
                liquid_volume, vapor_volume
            ):
                raise ValueError("VLE collapsed to a single homogeneous EOS phase")
        return residual

    def _binary_diagram(
        self,
        comp1,
        comp2,
        *,
        T=None,
        P=None,
        n_points=50,
        include_dew=True,
        lle_aware=False,
        phase_boundary=None,
        progress=None,
    ):
        self._diagram_controls(n_points, T, P)
        self._diagram_components([comp1, comp2])
        if (T is None) == (P is None):
            raise ValueError("Specify exactly one of temperature or pressure")
        result = {
            "x": [],
            "y": [],
            "bubble": [],
            "dew": [],
            "errors": [],
            "components": [comp1, comp2],
        }
        grid = [i / n_points for i in range(n_points + 1)]
        if phase_boundary:
            lean, rich = phase_boundary["x1"][comp1], phase_boundary["x2"][comp1]
            left_intervals = max(1, n_points // 2)
            right_intervals = max(1, n_points - left_intervals)
            # Resolve each stable branch even when one liquid occupies only
            # a few percent of the composition axis. Repeated plateau samples
            # add no curve information.
            grid = [lean * i / left_intervals for i in range(left_intervals + 1)]
            grid += [
                rich + (1 - rich) * i / right_intervals
                for i in range(right_intervals + 1)
            ]
            grid.append((lean + rich) / 2)
        grid = sorted(set(grid))
        for i, coordinate in enumerate(grid):
            x1 = max(1e-8, min(1 - 1e-8, coordinate))
            composition = {comp1: x1, comp2: 1 - x1}
            bubble = dew = y1 = None
            try:
                plateau = (
                    phase_boundary
                    and phase_boundary["x1"][comp1] <= x1 <= phase_boundary["x2"][comp1]
                )
                bubble = (
                    phase_boundary["T"]
                    if plateau
                    else (
                        self.bubble_point_T_vlle(composition, P, max_iter=200)
                        if lle_aware
                        else self.bubble_point_T(composition, P)
                        if P is not None
                        else self.bubble_point_P(composition, T)
                    )
                )
                if not math.isfinite(bubble) or bubble <= 0:
                    raise ValueError("Invalid equilibrium temperature or pressure")
                liquids = [composition]
                if plateau:
                    liquids = [phase_boundary["x1"], phase_boundary["x2"]]
                elif lle_aware:
                    split, a, b, beta = self.liquid_liquid_equilibrium(
                        composition, bubble, max_iter=200, tol=1e-8
                    )
                    if split:
                        self._check_diagram_split(
                            composition, [(1 - beta, a), (beta, b)], bubble
                        )
                        liquids = sorted([a, b], key=lambda x: x[comp1])
                liquid = liquids[0]
                K = self.K_values(
                    bubble if P is not None else T,
                    P if P is not None else bubble,
                    liquid,
                )
                vapor = {
                    component: fraction * K[component]
                    for component, fraction in liquid.items()
                }
                if any(not math.isfinite(v) or v < 0 for v in vapor.values()):
                    raise ValueError("Invalid equilibrium vapor composition")
                total = sum(vapor.values())
                if not math.isfinite(total) or total <= 0:
                    raise ValueError("Invalid equilibrium vapor composition")
                if abs(total - 1.0) > 2e-4:
                    raise ValueError(
                        f"Bubble point did not converge (residual {abs(total - 1.0):g})"
                    )
                vapor = {
                    component: fraction / total for component, fraction in vapor.items()
                }
                for liquid in liquids:
                    self._check_diagram_vle(
                        liquid,
                        vapor,
                        bubble if P is not None else T,
                        P if P is not None else bubble,
                    )
                y1 = vapor[comp1]
                if include_dew:
                    # The stable VLLE bubble solve gives a validated pair of
                    # coexisting liquid/vapor states at one temperature. Plot
                    # that same equilibrium at its vapor coordinate, without
                    # a second solve that could select a metastable liquid.
                    dew = (
                        bubble
                        if lle_aware
                        else (
                            self.dew_point_T(vapor, P)
                            if P is not None
                            else self.dew_point_P(vapor, T)
                        )
                    )
                    if not math.isfinite(dew) or dew <= 0:
                        raise ValueError("Invalid equilibrium temperature or pressure")
            except Exception as error:
                result["errors"].append({"index": i, "x": x1, "error": str(error)})
                if y1 is None:
                    bubble = None
                # Never invent a dew curve by substituting the bubble result.
                dew = None
            result["x"].append(x1)
            result["y"].append(y1)
            result["bubble"].append(bubble)
            result["dew"].append(dew)
            if progress and (i % 10 == 0 or i == len(grid) - 1):
                progress(f"Equilibrium envelope: {i + 1}/{len(grid)} compositions")
        if all(value is None for value in result["bubble"]):
            raise ValueError(
                "No bubble points converged: " + result["errors"][0]["error"]
            )
        return result

    def generate_Txy_data(
        self,
        comp1,
        comp2,
        P,
        n_points=50,
        *,
        lle_aware=False,
        phase_boundary=None,
        progress=None,
    ):
        points = self._binary_diagram(
            comp1,
            comp2,
            P=P,
            n_points=n_points,
            lle_aware=lle_aware,
            phase_boundary=phase_boundary,
            progress=progress,
        )
        bubble = [
            value - 273.15 if value is not None else None
            for value in points.pop("bubble")
        ]
        dew = [
            value - 273.15 if value is not None else None for value in points.pop("dew")
        ]
        azeotrope = None
        for i in range(1, len(points["x"])):
            if (
                points["y"][i - 1] is None
                or points["y"][i] is None
                or bubble[i - 1] is None
                or bubble[i] is None
            ):
                continue
            previous, current = (
                points["x"][i - 1] - points["y"][i - 1],
                points["x"][i] - points["y"][i],
            )
            if previous * current < 0:
                fraction = abs(previous) / (abs(previous) + abs(current))
                temperature = bubble[i - 1] + fraction * (bubble[i] - bubble[i - 1])
                endpoints = [bubble[0], bubble[-1]]
                classification = "unclassified"
                if all(v is not None for v in endpoints):
                    classification = (
                        "minimum"
                        if temperature < min(endpoints)
                        else "maximum"
                        if temperature > max(endpoints)
                        else "unclassified"
                    )
                azeotrope = {
                    "x": points["x"][i - 1]
                    + fraction * (points["x"][i] - points["x"][i - 1]),
                    "T": temperature,
                    "type": classification,
                }
                break
        return {
            **points,
            "T_bubble": bubble,
            "T_dew": dew,
            "azeotrope": azeotrope,
            "pressure_bar": P,
        }

    def generate_binary_vlle_data(
        self, comp1, comp2, P, n_points=50, minimum_temperature=298.15, progress=None
    ):
        """Constant-pressure stable VLE envelope and sampled liquid binodal."""
        self._diagram_controls(n_points, minimum_temperature, P)
        self._diagram_components([comp1, comp2])
        if not callable(getattr(self, "liquid_liquid_equilibrium", None)):
            raise ValueError("Binary VLLE requires an LLE-capable activity model")

        def composition(x):
            return {comp1: x, comp2: 1 - x}

        def split_at(z, temperature):
            split, a, b, beta = self.liquid_liquid_equilibrium(
                z, temperature, max_iter=200, tol=1e-8
            )
            if not split:
                return None
            balance, residual = self._check_diagram_split(
                z, [(1 - beta, a), (beta, b)], temperature
            )
            a, b = sorted([a, b], key=lambda x: x[comp1])
            return a, b, balance, residual

        seed = None
        invariant = None
        errors = []
        ordinary = self.bubble_point_T(composition(0.5), P)
        # An equimolar feed can be outside the miscibility gap (as it is for
        # butanol/water). Search compositions rather than assuming a 50/50 split.
        discovery_grid = sorted(
            set(
                [i / min(n_points, 20) for i in range(1, min(n_points, 20))]
                + [1e-4, 0.001, 0.01, 0.99, 0.999, 1 - 1e-4]
            )
        )
        for temperature in sorted(set([minimum_temperature, ordinary])):
            for coordinate in discovery_grid:
                z = composition(coordinate)
                try:
                    split = split_at(z, temperature)
                    if split:
                        seed = composition((split[0][comp1] + split[1][comp1]) / 2)
                        break
                except Exception as error:
                    errors.append(
                        {
                            "stage": "LLE discovery",
                            "T": temperature - 273.15,
                            "x": z[comp1],
                            "error": str(error),
                        }
                    )
            if seed:
                break
        if seed:
            temperature = self.bubble_point_T_vlle(seed, P, max_iter=200)
            split = split_at(seed, temperature)
            if split:
                a, b, _, _ = split
                K = self.K_values(temperature, P, a)
                vapor = {c: a[c] * K[c] for c in a}
                total = sum(vapor.values())
                if abs(total - 1) > 2e-4:
                    raise ValueError("Heteroazeotrope vapor did not converge")
                vapor = {c: v / total for c, v in vapor.items()}
                self._check_diagram_vle(a, vapor, temperature, P)
                self._check_diagram_vle(b, vapor, temperature, P)
                invariant = {"T": temperature, "x1": a, "x2": b, "y": vapor}
        chart = self.generate_Txy_data(
            comp1,
            comp2,
            P,
            n_points,
            lle_aware=True,
            phase_boundary=invariant,
            progress=progress,
        )
        binodal = {"T": [], "x1": [], "x2": [], "samples": []}
        upper = (
            invariant["T"]
            if invariant
            else min(t for t in chart["T_bubble"] if t is not None) + 273.15
        )
        if seed and minimum_temperature > upper:
            raise ValueError(
                "The binodal lower temperature exceeds the boiling boundary; choose a lower temperature."
            )
        if seed:
            for i in range(n_points + 1):
                temperature = (
                    minimum_temperature + (upper - minimum_temperature) * i / n_points
                )
                point = {
                    "T": temperature - 273.15,
                    "status": "no_split",
                    "z": dict(seed),
                }
                a = b = None
                try:
                    split = split_at(seed, temperature)
                    if split:
                        a, b, balance, residual = split
                        if max(
                            self.bubble_point_P(x, temperature) for x in (a, b)
                        ) > P * (1 + 2e-4):
                            a = b = None
                        else:
                            point.update(
                                status="lle",
                                x1=a,
                                x2=b,
                                liquid2_fraction=(seed[comp1] - a[comp1])
                                / (b[comp1] - a[comp1]),
                                material_balance_residual=balance,
                                activity_log_residual=residual,
                            )
                            seed = composition((a[comp1] + b[comp1]) / 2)
                except Exception as error:
                    a = b = None
                    errors.append(
                        {
                            "stage": "binodal",
                            "T": temperature - 273.15,
                            "error": str(error),
                        }
                    )
                    point.update(status="failed", error=str(error))
                binodal["T"].append(temperature - 273.15)
                binodal["x1"].append(a[comp1] if a else None)
                binodal["x2"].append(b[comp1] if b else None)
                binodal["samples"].append(point)
                if progress and (i % 10 == 0 or i == n_points):
                    progress(f"Liquid binodal: {i + 1}/{n_points + 1} temperatures")
        if invariant:
            chart["azeotrope"] = {
                "x": invariant["y"][comp1],
                "T": invariant["T"] - 273.15,
                "type": "heterogeneous",
                "liquid1": invariant["x1"][comp1],
                "liquid2": invariant["x2"][comp1],
            }
        public_invariant = (
            None
            if invariant is None
            else {
                "temperature_C": invariant["T"] - 273.15,
                **{key: invariant[key] for key in ("x1", "x2", "y")},
            }
        )
        return {
            **chart,
            "binodal": binodal,
            "heteroazeotrope": public_invariant,
            "minimum_temperature_C": minimum_temperature - 273.15,
            "errors": chart["errors"] + errors,
        }

    def generate_Pxy_data(self, comp1, comp2, T, n_points=50):
        points = self._binary_diagram(comp1, comp2, T=T, n_points=n_points)
        return {
            **points,
            "P_bubble": points.pop("bubble"),
            "P_dew": points.pop("dew"),
            "temperature_C": T - 273.15,
        }

    def generate_xy_data(self, comp1, comp2, T=None, P=None, n_points=50):
        points = self._binary_diagram(
            comp1, comp2, T=T, P=P, n_points=n_points, include_dew=False
        )
        points.pop("bubble")
        points.pop("dew")
        return {
            **points,
            "diagonal_x": [0, 1],
            "diagonal_y": [0, 1],
            "pressure_bar": P,
            "temperature_C": T - 273.15 if T else None,
        }

    def _check_diagram_split(self, z, phases, T, P=None):
        active = [
            (fraction, composition)
            for fraction, composition in phases
            if fraction > 1e-8
        ]
        if (
            any(
                not math.isfinite(fraction) or fraction < -1e-9
                for fraction, _ in phases
            )
            or abs(sum(f for f, _ in phases) - 1) > 1e-7
        ):
            raise ValueError("Phase fractions violate material balance")
        balance = max(abs(z[c] - sum(f * x.get(c, 0) for f, x in phases)) for c in z)
        if balance > 1e-6:
            raise ValueError(
                f"Phase split failed material balance (residual {balance:g})"
            )
        for _, composition in active:
            if (
                any(not math.isfinite(v) or v < -1e-9 for v in composition.values())
                or abs(sum(composition.values()) - 1) > 1e-6
            ):
                raise ValueError("Invalid phase composition")
        residual = 0.0
        # phases are liquid1, liquid2 for LLE; vapor, liquid1, liquid2 for VLLE.
        if len(phases) == 2 and len(active) == 2:
            gammas = [self.activity_coefficients(T, x) for _, x in active]
            for c in z:
                if z[c] <= 1e-12:
                    continue
                values = [x[c] * gamma[c] for (_, x), gamma in zip(active, gammas)]
                residual = max(
                    residual,
                    abs(math.log(max(values[0], 1e-300) / max(values[1], 1e-300))),
                )
        elif len(phases) == 3:
            vapor, liquid1, liquid2 = phases
            liquid_phases = [(f, x) for f, x in (liquid1, liquid2) if f > 1e-8]
            if len(liquid_phases) == 2:
                residual = self._check_diagram_split(
                    {
                        c: sum(f * x.get(c, 0) for f, x in liquid_phases)
                        / (liquid1[0] + liquid2[0])
                        for c in z
                    },
                    [(f / (liquid1[0] + liquid2[0]), x) for f, x in liquid_phases],
                    T,
                )[1]
            if vapor[0] > 1e-8:
                for _, x in liquid_phases:
                    residual = max(residual, self._check_diagram_vle(x, vapor[1], T, P))
        if not math.isfinite(residual) or residual > 1e-4:
            raise ValueError(
                f"Phase split failed chemical-potential equality (log residual {residual:g})"
            )
        return balance, residual

    def generate_ternary_lle_data(self, components, T, n_points=12, progress=None):
        self._diagram_controls(n_points, T)
        self._diagram_components(components)
        if len(components) != 3 or not callable(
            getattr(self, "liquid_liquid_equilibrium", None)
        ):
            raise ValueError(
                "Ternary LLE requires three components and an LLE-capable activity model"
            )
        samples = []
        total = (n_points + 1) * (n_points + 2) // 2
        for i in range(n_points + 1):
            for j in range(n_points - i + 1):
                z = dict(
                    zip(
                        components,
                        (i / n_points, j / n_points, (n_points - i - j) / n_points),
                    )
                )
                point = {"z": z, "status": "no_split"}
                try:
                    split, x1, x2, beta = self.liquid_liquid_equilibrium(
                        z, T, max_iter=200, tol=1e-8
                    )
                    if split:
                        balance, residual = self._check_diagram_split(
                            z, [(1 - beta, x1), (beta, x2)], T
                        )
                        point.update(
                            status="lle",
                            x1=x1,
                            x2=x2,
                            liquid2_fraction=beta,
                            material_balance_residual=balance,
                            activity_log_residual=residual,
                        )
                except Exception as error:
                    point.update(status="failed", error=str(error))
                samples.append(point)
            if progress:
                progress(f"LLE grid: {len(samples)}/{total} compositions")
        return {
            "components": components,
            "samples": samples,
            "temperature_C": T - 273.15,
            "grid_intervals": n_points,
            "split_count": sum(p["status"] == "lle" for p in samples),
            "failed_count": sum(p["status"] == "failed" for p in samples),
        }

    def generate_vlle_data(self, components, T, P, n_points=30, progress=None):
        self._diagram_controls(n_points, T, P)
        self._diagram_components(components)
        if len(components) not in {2, 3} or not callable(
            getattr(self, "flash3_TP", None)
        ):
            raise ValueError(
                "VLLE requires two or three components and an LLE-capable activity model"
            )
        if len(components) == 2:
            compositions = [
                {components[0]: i / n_points, components[1]: 1 - i / n_points}
                for i in range(n_points + 1)
            ]
        else:
            compositions = [
                dict(
                    zip(
                        components,
                        (i / n_points, j / n_points, (n_points - i - j) / n_points),
                    )
                )
                for i in range(n_points + 1)
                for j in range(n_points - i + 1)
            ]
        previous_mode = self.fluid_phase_model
        samples = []
        try:
            self.set_fluid_phase_model("VLLE")
            for index, z in enumerate(compositions):
                point = {"z": z}
                try:
                    state = self.calculate_state(T, P, 1.0, z, include=())
                    phases = [
                        (state.vapor_fraction, state.y or z),
                        (state.effective_liquid1_fraction, state.x1 or state.x or z),
                        (state.liquid2_fraction, state.x2 or z),
                    ]
                    balance, residual = self._check_diagram_split(z, phases, T, P)
                    point.update(
                        status=state.phase_status,
                        vapor_fraction=phases[0][0],
                        liquid1_fraction=phases[1][0],
                        liquid2_fraction=phases[2][0],
                        y=phases[0][1],
                        x1=phases[1][1],
                        x2=phases[2][1],
                        material_balance_residual=balance,
                        equilibrium_log_residual=residual,
                    )
                except Exception as error:
                    point.update(status="failed", error=str(error))
                samples.append(point)
                if progress and (index % 10 == 0 or index == len(compositions) - 1):
                    progress(f"VLLE map: {index + 1}/{len(compositions)} compositions")
        finally:
            self.set_fluid_phase_model(previous_mode)
        return {
            "components": components,
            "samples": samples,
            "temperature_C": T - 273.15,
            "pressure_bar": P,
            "failed_count": sum(p["status"] == "failed" for p in samples),
        }

    def bubble_point_P(self, composition: dict[str, float], T: float) -> float:
        self._diagram_controls(2, T)
        activity = getattr(self, "activity_coefficients", None)
        gamma = (
            activity(T, composition)
            if activity
            else dict.fromkeys(self.components, 1.0)
        )
        ideal = sum(
            x * gamma.get(comp, 1.0) * self.Psat(comp, T)
            for comp, x in composition.items()
        )
        correction = getattr(self, "_vapor_phase_correction_active", lambda: True)()
        if not correction:
            return ideal
        return self._bubble_point_P_from_K_values(composition, T, ideal)

    def _bubble_point_P_from_K_values(
        self, composition: dict[str, float], T: float, P_estimate: float
    ) -> float:
        """Solve sum(x_i K_i(T, P)) = 1 so bubble P includes vapor corrections."""

        def residual(P_value: float) -> float:
            K = self.K_values(T, P_value, composition)
            return (
                sum(
                    composition.get(comp, 0.0) * K.get(comp, 1.0)
                    for comp in self.components
                )
                - 1.0
            )

        P_center = max(float(P_estimate), 1e-8)
        values = []
        for factor in (
            1.0,
            0.8,
            1.25,
            0.5,
            2.0,
            0.25,
            4.0,
            0.1,
            10.0,
            0.03,
            30.0,
            0.01,
            100.0,
        ):
            P_try = P_center * factor
            try:
                f_try = residual(P_try)
            except Exception:
                continue
            if not math.isfinite(f_try):
                continue
            if abs(f_try) < 1e-10:
                return P_try
            values.append((P_try, f_try))
        values.sort()
        for (P1, f1), (P2, f2) in zip(values, values[1:]):
            if f1 * f2 < 0.0:
                try:
                    return brentq(residual, P1, P2, xtol=1e-12, rtol=1e-10, maxiter=100)
                except Exception:
                    continue
        raise ValueError("Could not bracket a converged bubble pressure")

    def dew_point_P(self, composition: dict[str, float], T: float) -> float:
        y = composition
        x = dict(y)
        self._diagram_controls(2, T)
        P_dew = 1.01325
        correction = getattr(self, "_vapor_phase_correction_active", lambda: True)()
        history: list[float] = []
        for _ in range(200):
            if correction:
                K = self.K_values(T, max(P_dew, 1e-8), x)
                volatility = {
                    comp: max(K.get(comp, 1.0) * max(P_dew, 1e-8), 1e-30) for comp in y
                }
            else:
                gamma = self.activity_coefficients(T, x)
                volatility = {
                    comp: max(gamma.get(comp, 1.0) * self.Psat(comp, T), 1e-30)
                    for comp in y
                }
            sum_inv = sum(y_i / volatility[comp] for comp, y_i in y.items())
            P_new = 1.0 / max(sum_inv, 1e-30)
            history.append(P_new)
            if len(history) >= 3:
                # Aitken delta-squared acceleration of the linearly
                # convergent pressure substitution, with a step safeguard.
                p1, p2, p3 = history[-3:]
                denominator = (p3 - p2) - (p2 - p1)
                if abs(denominator) > 1e-300:
                    accelerated = p3 - (p3 - p2) ** 2 / denominator
                    if math.isfinite(accelerated) and 0.2 * p3 < accelerated < 5.0 * p3:
                        P_new = accelerated
                        history.clear()
            x_new = {comp: y_i * P_new / volatility[comp] for comp, y_i in y.items()}
            x_sum = sum(x_new.values())
            if x_sum > 0:
                x_new = {comp: value / x_sum for comp, value in x_new.items()}
            x_change = (
                max(abs(x_new.get(comp, 0.0) - x.get(comp, 0.0)) for comp in x_new)
                if x_new
                else 0.0
            )
            P_change = abs(P_new - P_dew)
            P_dew = P_new
            x = x_new
            if P_change <= 1e-10 * max(P_dew, 1e-30) and x_change <= 1e-12:
                break

        if correction:
            # Frozen-composition bracketed polish: the substitution converges
            # slowly when phi depends strongly on P, so finish on the dew
            # residual directly with the incipient liquid held fixed.
            def frozen_residual(P_value: float) -> float:
                K = self.K_values(T, max(P_value, 1e-8), x)
                return (
                    sum(y_i / max(K.get(comp, 1.0), 1e-30) for comp, y_i in y.items())
                    - 1.0
                )

            try:
                f_final = frozen_residual(P_dew)
                if abs(f_final) > 1e-9:
                    low = high = P_dew
                    f_low = f_high = f_final
                    for _ in range(60):
                        if f_low <= 0.0 <= f_high:
                            P_dew = brentq(
                                frozen_residual,
                                low,
                                high,
                                xtol=1e-12,
                                rtol=1e-10,
                                maxiter=100,
                            )
                            break
                        if f_low > 0.0:
                            low = max(1e-10, low * 0.7)
                            f_low = frozen_residual(low)
                        if f_high < 0.0:
                            high = high * 1.4
                            f_high = frozen_residual(high)
            except Exception:
                pass
        K_final = self.K_values(T, P_dew, x)
        residual = abs(
            sum(y_i / max(K_final.get(comp, 1.0), 1e-30) for comp, y_i in y.items())
            - 1.0
        )
        if not math.isfinite(residual) or residual > 1e-7:
            raise ValueError(f"Dew pressure did not converge (residual {residual:g})")
        # A scalar sum can converge even while liquid compositions oscillate.
        # Check each component after the pressure polish, which held x fixed.
        component_residual = max(
            abs(math.log(max(K_final[comp] * x[comp], 1e-300) / y_i))
            for comp, y_i in y.items()
            if y_i > 1e-12
        )
        if not math.isfinite(component_residual) or component_residual > 1e-6:
            raise ValueError(
                "Dew pressure liquid composition did not converge "
                f"(log residual {component_residual:g})"
            )
        self._check_diagram_vle(x, y, T, P_dew)
        return P_dew
