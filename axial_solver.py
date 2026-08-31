"""Shared adaptive integration for one-dimensional process models."""

from dataclasses import dataclass
import math
from typing import Callable, Optional, Sequence, Union

import numpy as np
from scipy.integrate import solve_ivp


AxialRHS = Callable[[float, np.ndarray], Sequence[float]]
AxialEventFunction = Callable[[float, np.ndarray], float]


class AxialIntegrationError(RuntimeError):
    """Raised when an axial balance cannot be integrated."""


@dataclass(frozen=True)
class AxialEvent:
    """Named zero crossing that can terminate or annotate an axial solve."""

    name: str
    function: AxialEventFunction
    terminal: bool = True
    direction: float = 0.0

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("Axial event name cannot be empty")
        if not callable(self.function):
            raise TypeError("Axial event function must be callable")
        if not math.isfinite(float(self.direction)):
            raise ValueError("Axial event direction must be finite")


@dataclass(frozen=True)
class AxialSolverOptions:
    """Numerical controls for adaptive axial integration."""

    method: str = "RK45"
    relative_tolerance: float = 1e-6
    absolute_tolerance: Union[float, Sequence[float]] = 1e-9
    maximum_step_m: Optional[float] = None
    first_step_m: Optional[float] = None


@dataclass(frozen=True)
class AxialSolution:
    """Integrated axial state history."""

    position_m: np.ndarray
    values: np.ndarray
    event_positions_m: dict[str, np.ndarray]
    evaluations: int
    status: int
    message: str

    @property
    def inlet_values(self) -> np.ndarray:
        return self.values[:, 0].copy()

    @property
    def outlet_values(self) -> np.ndarray:
        return self.values[:, -1].copy()

    @property
    def outlet_position_m(self) -> float:
        return float(self.position_m[-1])

    @property
    def terminated_by_event(self) -> bool:
        return self.status == 1


def _positive_finite(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return value


def _evaluation_grid(
    length_m: float,
    evaluation_positions_m: Optional[Sequence[float]],
) -> Optional[np.ndarray]:
    if evaluation_positions_m is None:
        return None
    grid = np.asarray(tuple(evaluation_positions_m), dtype=float)
    if grid.ndim != 1 or not np.all(np.isfinite(grid)):
        raise ValueError("Axial evaluation positions must be a finite one-dimensional sequence")
    if np.any(grid < 0.0) or np.any(grid > length_m):
        raise ValueError("Axial evaluation positions must lie within the integration length")
    return np.unique(np.concatenate(([0.0], grid, [length_m])))


def integrate_axial(
    rhs: AxialRHS,
    length_m: float,
    initial_values: Sequence[float],
    *,
    options: Optional[AxialSolverOptions] = None,
    events: Sequence[AxialEvent] = (),
    evaluation_positions_m: Optional[Sequence[float]] = None,
) -> AxialSolution:
    """Integrate coupled balances from zero to ``length_m``.

    The state vector and its units belong to the caller. A pipe may integrate
    pressure and enthalpy, while a reactor can append component flow rates and
    an exchanger can append a second stream's balances.
    """
    if not callable(rhs):
        raise TypeError("Axial right-hand side must be callable")
    length = _positive_finite(length_m, "Axial length")
    initial = np.asarray(tuple(initial_values), dtype=float)
    if initial.ndim != 1 or initial.size == 0 or not np.all(np.isfinite(initial)):
        raise ValueError("Initial axial state must be a nonempty finite vector")

    controls = options or AxialSolverOptions()
    relative_tolerance = _positive_finite(
        controls.relative_tolerance, "Relative tolerance"
    )
    absolute_tolerance = np.asarray(controls.absolute_tolerance, dtype=float)
    if (
        np.any(~np.isfinite(absolute_tolerance))
        or np.any(absolute_tolerance <= 0.0)
        or (absolute_tolerance.ndim > 1)
        or (absolute_tolerance.ndim == 1 and absolute_tolerance.size not in {1, initial.size})
    ):
        raise ValueError(
            "Absolute tolerance must be positive and scalar or match the axial state"
        )
    if absolute_tolerance.ndim == 0 or absolute_tolerance.size == 1:
        solver_atol: Union[float, np.ndarray] = float(absolute_tolerance.reshape(-1)[0])
    else:
        solver_atol = absolute_tolerance

    maximum_step = (
        math.inf
        if controls.maximum_step_m is None
        else _positive_finite(controls.maximum_step_m, "Maximum axial step")
    )
    first_step = (
        None
        if controls.first_step_m is None
        else _positive_finite(controls.first_step_m, "First axial step")
    )
    if first_step is not None and first_step > length:
        raise ValueError("First axial step cannot exceed the integration length")

    def checked_rhs(position: float, values: np.ndarray) -> np.ndarray:
        try:
            derivative = np.asarray(tuple(rhs(position, values)), dtype=float)
        except Exception as exc:
            raise AxialIntegrationError(
                f"Axial balance evaluation failed at s={position:.6g} m"
            ) from exc
        if derivative.shape != initial.shape:
            raise AxialIntegrationError(
                f"Axial balance returned shape {derivative.shape}; expected {initial.shape}"
            )
        if not np.all(np.isfinite(derivative)):
            raise AxialIntegrationError(
                f"Axial balance returned a non-finite derivative at s={position:.6g} m"
            )
        return derivative

    wrapped_events = []
    event_names = set()
    for event in events:
        if event.name in event_names:
            raise ValueError(f"Duplicate axial event name: {event.name!r}")
        event_names.add(event.name)

        def wrapped_event(
            position: float,
            values: np.ndarray,
            event_function: AxialEventFunction = event.function,
            event_name: str = event.name,
        ) -> float:
            try:
                value = float(event_function(position, values))
            except Exception as exc:
                raise AxialIntegrationError(
                    f"Axial event {event_name!r} failed at s={position:.6g} m"
                ) from exc
            if not math.isfinite(value):
                raise AxialIntegrationError(
                    f"Axial event {event_name!r} returned a non-finite value"
                )
            return value

        wrapped_event.terminal = bool(event.terminal)
        wrapped_event.direction = float(event.direction)
        wrapped_events.append(wrapped_event)

    result = solve_ivp(
        checked_rhs,
        (0.0, length),
        initial,
        method=controls.method,
        rtol=relative_tolerance,
        atol=solver_atol,
        max_step=maximum_step,
        first_step=first_step,
        events=wrapped_events or None,
        t_eval=_evaluation_grid(length, evaluation_positions_m),
    )
    if not result.success:
        raise AxialIntegrationError(f"Axial integration failed: {result.message}")

    positions = np.asarray(result.t, dtype=float)
    values = np.asarray(result.y, dtype=float)
    if result.status == 1:
        terminal_hits = [
            (float(times[-1]), np.asarray(states[-1], dtype=float))
            for times, states in zip(result.t_events, result.y_events)
            if len(times)
        ]
        if terminal_hits:
            terminal_position, terminal_values = max(
                terminal_hits, key=lambda hit: hit[0]
            )
            if not len(positions) or terminal_position > positions[-1] + 1e-12:
                positions = np.append(positions, terminal_position)
                values = np.column_stack((values, terminal_values))

    event_positions = {
        event.name: np.asarray(result.t_events[index], dtype=float)
        for index, event in enumerate(events)
    }
    return AxialSolution(
        position_m=positions,
        values=values,
        event_positions_m=event_positions,
        evaluations=int(result.nfev),
        status=int(result.status),
        message=str(result.message),
    )
