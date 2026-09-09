"""Laminar flat-plate transfer correlation and validity limits."""

import pytest

from transport_correlations import TransportCorrelationError, laminar_flat_plate_transfer


def test_average_transfer_values_and_similarity():
    result = laminar_flat_plate_transfer(1e4, 8, 1000)
    assert result['nusselt'] == pytest.approx(132.8)
    assert result['sherwood'] == pytest.approx(664)
    assert laminar_flat_plate_transfer(1e4, 8, 8)['sherwood'] == result['nusselt']


@pytest.mark.parametrize('re, pr, sc', [(5e5, 8, 1000), (1, 8, 1000),
                                      (1e4, 0.1, 1000), (1e4, 8, 0.1),
                                      (-1, 8, 1000), (float('nan'), 8, 1000)])
def test_invalid_or_out_of_range_flow_is_rejected(re, pr, sc):
    with pytest.raises(TransportCorrelationError):
        laminar_flat_plate_transfer(re, pr, sc)
