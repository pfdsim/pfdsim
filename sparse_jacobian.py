"""Checked fixed-pattern assembly for repeatedly evaluated sparse Jacobians."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


class SparseJacobianPatternError(ValueError):
    """Raised when an assembly attempts to write outside its declared pattern."""


class SparsePatternBuilder:
    """Build a canonical CSR structural pattern without scalar SciPy writes."""

    __slots__ = ("shape", "_rows")

    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape = (int(shape[0]), int(shape[1]))
        self._rows = tuple(set() for _ in range(self.shape[0]))

    def mark(self, row: int, column: int) -> None:
        if not 0 <= row < self.shape[0] or not 0 <= column < self.shape[1]:
            raise SparseJacobianPatternError(
                f"Jacobian entry ({row}, {column}) is outside pattern shape "
                f"{self.shape}"
            )
        self._rows[row].add(column)

    def mark_range(self, row: int, start: int, stop: int) -> None:
        if not 0 <= row < self.shape[0] or not 0 <= start <= stop <= self.shape[1]:
            raise SparseJacobianPatternError(
                f"Jacobian range ({row}, {start}:{stop}) is outside pattern "
                f"shape {self.shape}"
            )
        self._rows[row].update(range(start, stop))

    def tocsr(self):
        indptr = np.empty(self.shape[0] + 1, dtype=np.intc)
        indptr[0] = 0
        ordered_rows = []
        for row, columns in enumerate(self._rows):
            ordered = sorted(columns)
            ordered_rows.extend(ordered)
            indptr[row + 1] = len(ordered_rows)
        indices = np.asarray(ordered_rows, dtype=np.intc)
        data = np.ones(indices.size, dtype=bool)
        return csr_matrix(
            (data, indices, indptr),
            shape=self.shape,
            copy=False,
        )


class FixedPatternCSR:
    """Immutable CSR structure that creates inexpensive numeric accumulators.

    The supplied sparsity matrix is authoritative.  Every numeric write is
    checked against it so a stale or incomplete pattern cannot silently drop a
    Jacobian term.
    """

    __slots__ = ("shape", "_indices", "_indptr", "_positions")

    def __init__(self, sparsity) -> None:
        pattern = csr_matrix(sparsity, dtype=bool, copy=True)
        pattern.sum_duplicates()
        pattern.eliminate_zeros()
        pattern.sort_indices()
        self.shape = pattern.shape
        self._indices = np.asarray(pattern.indices, dtype=np.intc).copy()
        self._indptr = np.asarray(pattern.indptr, dtype=np.intc).copy()
        self._positions = tuple(
            {
                int(column): int(position)
                for position, column in enumerate(
                    self._indices[self._indptr[row]:self._indptr[row + 1]],
                    start=int(self._indptr[row]),
                )
            }
            for row in range(self.shape[0])
        )

    @property
    def nnz(self) -> int:
        return int(self._indices.size)

    def empty(self) -> FixedPatternCSRBuilder:
        return FixedPatternCSRBuilder(self, np.zeros(self.nnz, dtype=float))

    def _position(self, row: int, column: int) -> int:
        try:
            if row < 0:
                raise IndexError(row)
            return self._positions[row][column]
        except (IndexError, KeyError) as exc:
            raise SparseJacobianPatternError(
                f"Jacobian entry ({row}, {column}) is outside the declared "
                f"sparsity pattern {self.shape}"
            ) from exc

    def _matrix(self, data: np.ndarray):
        # Copy values as well as structure: eliminate_zeros compacts data in
        # place, which would invalidate the builder's fixed position mapping.
        # Returned matrices must also remain independent of subsequent writes.
        matrix = csr_matrix(
            (data.copy(), self._indices.copy(), self._indptr.copy()),
            shape=self.shape,
            copy=False,
        )
        # LIL assignment omitted exact zeros.  Preserve that behavior so the
        # linear solver sees the same effective sparsity rather than a dense
        # structural over-approximation full of explicit zeros.
        matrix.eliminate_zeros()
        return matrix


@dataclass(slots=True)
class FixedPatternCSRBuilder:
    """Mutable numeric values for one :class:`FixedPatternCSR` evaluation."""

    pattern: FixedPatternCSR
    data: np.ndarray

    def add(self, row: int, column: int, value: float) -> None:
        if value:
            self.data[self.pattern._position(row, column)] += value

    def set(self, row: int, column: int, value: float) -> None:
        self.data[self.pattern._position(row, column)] = value

    def set_column(self, rows, column: int, values) -> None:
        for row, value in zip(rows, values):
            self.set(int(row), column, float(value))

    def tocsr(self):
        return self.pattern._matrix(self.data)
