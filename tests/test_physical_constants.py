"""Check the SI definition and dimensional conversions of the gas constant."""

from decimal import Decimal

from physical_constants import (
    R_BAR_CM3_MOL_K,
    R_BAR_L_MOL_K,
    R_BAR_M3_MOL_K,
    R_CAL_MOL_K,
    R_J_MOL_K,
)


def test_gas_constant_matches_exact_si_definitions():
    exact = Decimal('6.02214076e23') * Decimal('1.380649e-23')
    assert R_J_MOL_K == float(exact)


def test_gas_constant_unit_conversions():
    import math

    for converted, joules_per_unit in (
        (R_BAR_CM3_MOL_K, 0.1),
        (R_BAR_L_MOL_K, 100.0),
        (R_BAR_M3_MOL_K, 100000.0),
        (R_CAL_MOL_K, 4.184),
    ):
        assert math.isclose(converted * joules_per_unit, R_J_MOL_K, rel_tol=5e-16)
