"""Conditioned fitting coordinates with exact physical coefficient bounds."""

import numpy as np
from scipy.optimize import LinearConstraint, OptimizeResult, minimize
from scipy.special import expit, logit


class _EvaluationLimit(Exception):
    pass


def temperature_basis(problem, rows):
    """Representative values of the unchanged law over its training interval."""
    count = len(problem.terms)
    temperatures = np.array([row["T_K"] for row in rows])
    low, high = temperatures.min(), temperatures.max()
    if low == high:
        low, high = low * 0.95, high * 1.05
    temperatures = np.unique(np.r_[temperatures, np.linspace(low, high, max(5, count))])
    u = temperatures / problem.request["T_ref_K"]
    columns = {
        "constant": np.ones_like(u),
        "inverse": 1 / u,
        "anchored": 1 / u - 1 + np.log(u),
        "linear": u,
        "quadratic": u * u,
    }
    return np.column_stack([columns[term] for term in problem.terms])


def temperature_transform(problem, rows):
    """Condition the law without adding observations or altering coefficients."""
    count = len(problem.terms)
    transform = np.eye(len(problem.names))
    if count == 1:
        return transform
    basis = temperature_basis(problem, rows)
    _, singular, right = np.linalg.svd(basis, full_matrices=False)
    # Avoid an enormous numerical map for nearly zero temperature coverage.
    singular = np.maximum(singular, singular[0] * 1e-3)
    block = right.T * (np.sqrt(len(basis)) / singular)
    for first in (0, count):
        transform[first : first + count, first : first + count] = block
    return transform


class ConditionedObjective:
    """Share residual/Jacobian evaluations and exploit observation-local variables."""

    def __init__(self, problem, rows, transform):
        self.problem, self.rows, self.transform = problem, rows, transform
        self.last = None
        self.calls = 0
        self.row_calls = 0
        self.limit = None
        self.local = {}
        for row in rows:
            indices = []
            if row["id"] in problem.lle_names:
                indices.append(problem.lle_names[row["id"]])
            if row["id"] in problem.critical_names:
                indices.append(problem.critical_names[row["id"]])
            indices.extend(problem.vlle_names.get(row["id"], ()))
            for index in indices:
                self.local[index] = row["id"]

    def encode(self, values):
        coordinates = np.linalg.solve(self.transform, values)
        for index in self.local:
            coordinates[index] = logit(values[index])
        return coordinates

    def decode(self, coordinates):
        values = self.transform @ coordinates
        for index in self.local:
            values[index] = expit(coordinates[index])
        return values

    def coordinate_bounds(self, values):
        bounds = values.copy()
        for index in self.local:
            bounds[index] = logit(values[index])
        return bounds

    def decoding_jacobian(self, values):
        matrix = self.transform.copy()
        for index in self.local:
            matrix[index, index] = values[index] * (1 - values[index])
        return matrix

    def _row(self, values, row):
        self.row_calls += 1
        errors = self.problem.row_errors(values, row)[0]
        if not row["pin"]:
            errors = errors * np.sqrt(
                row["weight"] * self.problem.request["weights"][row["kind"]]
            )
        return errors

    def evaluate(self, coordinates):
        if self.last is not None and np.array_equal(coordinates, self.last):
            return self.errors
        if self.limit is not None and self.calls >= self.limit:
            raise _EvaluationLimit
        self.calls += 1
        values = self.decode(coordinates)
        self.parts = [self._row(values, row) for row in self.rows]
        self.slices = []
        offset = 0
        for part in self.parts:
            self.slices.append(slice(offset, offset + len(part)))
            offset += len(part)
        self.errors = np.concatenate(self.parts)
        self.jacobian = None
        self.last = np.array(coordinates, copy=True)
        return self.errors

    def jac(self, coordinates):
        self.evaluate(coordinates)
        if self.jacobian is not None:
            return self.jacobian
        result = np.zeros((len(self.errors), len(coordinates)))
        for index in range(len(coordinates)):
            affected = [
                i
                for i, row in enumerate(self.rows)
                if index not in self.local or row["id"] == self.local[index]
            ]
            step = 1e-5 * max(1, abs(coordinates[index]))
            # Fraction logits scale near-pure phases; bounds stay equivalent.
            if index in self.local:
                forward = max(
                    0, min(step, logit(self.problem.upper[index]) - coordinates[index])
                )
                backward = max(
                    0, min(step, coordinates[index] - logit(self.problem.lower[index]))
                )
            else:
                forward = backward = step
            for attempt in range(8):
                high, low = coordinates.copy(), coordinates.copy()
                high[index] += forward
                low[index] -= backward
                sides, errors = [], []
                for trial, distance in ((high, forward), (low, backward)):
                    try:
                        sides.append(
                            [
                                self._row(self.decode(trial), self.rows[i])
                                if distance
                                else self.parts[i]
                                for i in affected
                            ]
                        )
                        errors.append(None)
                    except self.problem.numerical_errors as error:
                        sides.append(None)
                        errors.append(error)
                if all(side is not None for side in sides):
                    break
                if attempt == 7:
                    if sides[0] is None and sides[1] is None:
                        raise errors[0]
                    if sides[0] is None:
                        forward, sides[0] = 0, [self.parts[i] for i in affected]
                    if sides[1] is None:
                        backward, sides[1] = 0, [self.parts[i] for i in affected]
                    break
                forward, backward = forward / 4, backward / 4
            for position, row_index in enumerate(affected):
                result[self.slices[row_index], index] = (
                    sides[0][position] - sides[1][position]
                ) / (forward + backward)
        self.problem.install(self.decode(coordinates))
        self.jacobian = result
        return result


def solve_start(problem, rows, start):
    """One shared constrained optimizer for initialization and hard-pin refinement."""
    transform = temperature_transform(problem, rows)
    objective = ConditionedObjective(problem, rows, transform)
    coordinates = objective.encode(start)
    bounds = LinearConstraint(transform, objective.coordinate_bounds(problem.lower), objective.coordinate_bounds(problem.upper))
    pinned = np.array([row["pin"] for row in rows])

    def minimize_rows(initial, soft_only, constraints):
        def selected(q):
            objective.evaluate(q)
            return (
                np.concatenate(
                    [
                        np.arange(section.start, section.stop)
                        for row, section in zip(rows, objective.slices)
                        if not soft_only or not row["pin"]
                    ]
                )
                if not soft_only or not pinned.all()
                else np.array([], dtype=int)
            )

        initial_indices = selected(initial)
        initial_errors = objective.evaluate(initial)[initial_indices]
        normalization = max(
            1.0, float(initial_errors @ initial_errors) / max(1, len(initial_errors))
        )
        objective.limit = objective.calls + problem.request["max_nfev"] - 1
        best = [np.inf, initial.copy()]

        def cost(q):
            try:
                residual = objective.evaluate(q)[selected(q)]
            except problem.numerical_errors:
                # An inadmissible trial is outside the physical objective's
                # domain. Let the line search shorten it; never accept it.
                return np.inf
            value = float(residual @ residual) / normalization
            coefficients = objective.decode(q)
            feasible = np.all(coefficients >= problem.lower - 1e-8) and np.all(
                coefficients <= problem.upper + 1e-8
            )
            if feasible and all(
                np.min(constraint["fun"](q)) >= -1e-7 for constraint in constraints
            ):
                if value < best[0]:
                    best[:] = [value, q.copy()]
            return value

        def gradient(q):
            indices = selected(q)
            residual = objective.evaluate(q)[indices]
            return 2 * objective.jac(q)[indices].T @ residual / normalization

        try:
            return minimize(
                cost,
                initial,
                jac=gradient,
                method="SLSQP",
                constraints=[bounds, *constraints],
                options={
                    "maxiter": problem.request["max_nfev"],
                    "ftol": 1e-14 / normalization,
                },
            )
        except _EvaluationLimit:
            return OptimizeResult(
                x=best[1],
                success=False,
                status=0,
                message="The maximum number of residual evaluations was reached.",
            )
        finally:
            objective.limit = None

    fitted = minimize_rows(coordinates, False, [])
    if pinned.any():
        objective.evaluate(fitted.x)
        indices = np.concatenate(
            [
                np.arange(section.start, section.stop)
                for row, section in zip(rows, objective.slices)
                if row["pin"]
            ]
        )
        tolerances = np.concatenate(
            [
                np.full(section.stop - section.start, row["pin_tolerance"])
                for row, section in zip(rows, objective.slices)
                if row["pin"]
            ]
        )

        def constraints(q):
            errors = objective.evaluate(q)[indices]
            return np.r_[tolerances - errors, tolerances + errors]

        def constraint_jac(q):
            jacobian = objective.jac(q)[indices]
            return np.r_[-jacobian, jacobian]

        fitted = minimize_rows(
            fitted.x,
            True,
            [{"type": "ineq", "fun": constraints, "jac": constraint_jac}],
        )
        if np.min(constraints(fitted.x)) < -1e-7:
            raise ValueError(f"Hard pins are infeasible ({fitted.message}).")
    values = objective.decode(fitted.x)
    if np.any(values < problem.lower - 1e-7) or np.any(values > problem.upper + 1e-7):
        raise ValueError("The optimizer did not satisfy the coefficient bounds.")
    # Only remove roundoff outside the exact coefficient box.
    values = np.clip(values, problem.lower, problem.upper)
    optimization_calls = objective.calls
    final = objective.encode(values)
    objective.evaluate(final)
    for row, section in zip(rows, objective.slices):
        if row["pin"] and np.max(np.abs(objective.errors[section])) > row["pin_tolerance"] + 1e-7:
            raise ValueError("Hard pins are infeasible after coefficient-bound verification.")
    indices = (
        np.concatenate(
            [
                np.arange(section.start, section.stop)
                for row, section in zip(rows, objective.slices)
                if not row["pin"]
            ]
        )
        if not pinned.all()
        else np.array([], dtype=int)
    )
    errors = objective.errors[indices]
    physical_jacobian = objective.jac(final) @ np.linalg.inv(objective.decoding_jacobian(values))
    basis = temperature_basis(problem, rows)
    count = len(problem.terms)
    return {
        "values": values,
        "objective": float(errors @ errors),
        "success": bool(fitted.success),
        "message": str(fitted.message),
        "nfev": optimization_calls,
        "row_evaluations": objective.row_calls,
        "verification_evaluations": objective.calls - optimization_calls,
        "rank": int(np.linalg.matrix_rank(physical_jacobian)),
        "temperature_basis_condition": float(np.linalg.cond(basis)),
        "conditioned_temperature_basis_condition": float(
            np.linalg.cond(basis @ transform[:count, :count])
        ),
    }
