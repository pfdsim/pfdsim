"""Compatibility facade for thermodynamic property models."""

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .thermodynamics_models import *
else:
    from thermodynamics_models import *
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .thermodynamics_models.common import (
        _looks_like_molecular_formula,
        _property_lookup_identifier,
        _solve_bubble_point_temperature,
        _solve_dew_point_temperature,
    )
else:
    from thermodynamics_models.common import (
        _looks_like_molecular_formula,
        _property_lookup_identifier,
        _solve_bubble_point_temperature,
        _solve_dew_point_temperature,
    )
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .thermodynamics_models.nrtl_uniquac import (
        _interaction_override_map,
        _oriented_component_override,
    )
else:
    from thermodynamics_models.nrtl_uniquac import (
        _interaction_override_map,
        _oriented_component_override,
    )
