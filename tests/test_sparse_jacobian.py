import unittest

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

try:
    from sparse_jacobian import (
        FixedPatternCSR,
        SparseJacobianPatternError,
        SparsePatternBuilder,
    )
except ImportError:
    from pfdsim.sparse_jacobian import (
        FixedPatternCSR,
        SparseJacobianPatternError,
        SparsePatternBuilder,
    )


class FixedPatternCSRTests(unittest.TestCase):
    def test_structural_builder_canonicalizes_marks(self):
        builder = SparsePatternBuilder((3, 4))
        builder.mark(1, 3)
        builder.mark_range(1, 0, 3)
        builder.mark(1, 2)

        pattern = builder.tocsr()

        np.testing.assert_array_equal(
            pattern.toarray(),
            np.array([
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
            ]),
        )
        self.assertTrue(pattern.has_sorted_indices)

    def test_structural_builder_rejects_out_of_bounds_marks(self):
        builder = SparsePatternBuilder((2, 3))
        with self.assertRaisesRegex(SparseJacobianPatternError, "outside"):
            builder.mark(2, 0)
        with self.assertRaisesRegex(SparseJacobianPatternError, "outside"):
            builder.mark_range(0, 1, 4)

    def test_accumulates_repeated_entries_and_preserves_nonfinite_values(self):
        pattern = csr_matrix(np.array([
            [1, 0, 1],
            [0, 1, 0],
        ]))
        builder = FixedPatternCSR(pattern).empty()

        builder.add(0, 0, 2.5)
        builder.add(0, 0, -0.75)
        builder.add(0, 2, 0.0)
        builder.add(1, 1, float("nan"))
        result = builder.tocsr()

        self.assertEqual(result[0, 0], 1.75)
        self.assertEqual(result[0, 2], 0.0)
        self.assertTrue(np.isnan(result[1, 1]))
        self.assertEqual(result.nnz, 2)

    def test_set_column_matches_sparse_assignment(self):
        pattern = csr_matrix(np.ones((3, 2), dtype=int))
        builder = FixedPatternCSR(pattern).empty()
        builder.set_column(
            np.array([0, 2]),
            1,
            np.array([4.0, -3.0]),
        )

        np.testing.assert_array_equal(
            builder.tocsr().toarray(),
            np.array([
                [0.0, 4.0],
                [0.0, 0.0],
                [0.0, -3.0],
            ]),
        )

    def test_duplicate_pattern_entries_are_canonicalized(self):
        pattern = coo_matrix(
            (
                np.ones(3),
                (np.array([0, 0, 0]), np.array([1, 1, 1])),
            ),
            shape=(2, 2),
        )
        fixed = FixedPatternCSR(pattern)
        builder = fixed.empty()
        builder.add(0, 1, 2.0)

        self.assertEqual(fixed.nnz, 1)
        self.assertEqual(builder.tocsr()[0, 1], 2.0)

    def test_zero_assignment_clears_existing_values(self):
        builder = FixedPatternCSR(csr_matrix(np.ones((2, 2)))).empty()
        builder.set(0, 0, 4.0)
        builder.set(0, 0, 0.0)
        builder.set_column([0, 1], 1, [5.0, -3.0])
        builder.set_column([0, 1], 1, [0.0, 0.0])

        self.assertEqual(builder.tocsr().nnz, 0)

    def test_conversion_preserves_builder_and_returns_independent_snapshots(self):
        for values in ([0.0, 4.0, 0.0], [2.0, 4.0, 6.0]):
            with self.subTest(values=values):
                builder = FixedPatternCSR(csr_matrix(np.ones((1, 3)))).empty()
                for column, value in enumerate(values):
                    builder.set(0, column, value)
                first = builder.tocsr()
                second = builder.tocsr()
                np.testing.assert_array_equal(builder.data, values)
                np.testing.assert_array_equal(first.toarray(), [values])
                np.testing.assert_array_equal(second.toarray(), [values])

                builder.add(0, 1, 3.0)
                np.testing.assert_array_equal(first.toarray(), [values])
                first.data[:] = -1.0
                np.testing.assert_array_equal(second.toarray(), [values])
                expected = list(values)
                expected[1] += 3.0
                np.testing.assert_array_equal(builder.tocsr().toarray(), [expected])

    def test_numeric_writes_reject_invalid_indices(self):
        builder = FixedPatternCSR(csr_matrix(np.ones((2, 2)))).empty()
        for row, column in [(-1, 0), (-2, 0), (2, 0), (0, -1), (0, 2)]:
            with self.subTest(row=row, column=column):
                with self.assertRaises(SparseJacobianPatternError):
                    builder.add(row, column, 1.0)
                with self.assertRaises(SparseJacobianPatternError):
                    builder.set(row, column, 1.0)
                with self.assertRaises(SparseJacobianPatternError):
                    builder.set_column([row], column, [1.0])

        self.assertEqual(builder.tocsr().nnz, 0)

    def test_write_outside_pattern_fails_loudly(self):
        builder = FixedPatternCSR(csr_matrix(np.eye(2))).empty()

        with self.assertRaisesRegex(
            SparseJacobianPatternError,
            r"entry \(0, 1\).*pattern \(2, 2\)",
        ):
            builder.add(0, 1, 1.0)

    def test_returned_matrices_do_not_alias_later_builders(self):
        fixed = FixedPatternCSR(csr_matrix(np.eye(2)))
        first = fixed.empty()
        first.add(0, 0, 1.0)
        first_matrix = first.tocsr()

        second = fixed.empty()
        second.add(0, 0, 5.0)
        second_matrix = second.tocsr()
        second_matrix.indices[0] = 1

        self.assertEqual(first_matrix[0, 0], 1.0)
        third = fixed.empty()
        third.add(0, 0, 9.0)
        self.assertEqual(third.tocsr()[0, 0], 9.0)


if __name__ == "__main__":
    unittest.main()
