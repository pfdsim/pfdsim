"""Explicit per-component phase-behavior declarations."""

CONVENTIONAL_PHASE_BEHAVIOR = 'conventional'
CONVENTIONAL_WITH_SOLID_PHASE_BEHAVIOR = 'conventional_with_solid'
PERMANENT_SOLID_PHASE_BEHAVIOR = 'permanent_solid'

PHASE_BEHAVIORS = frozenset({
    CONVENTIONAL_PHASE_BEHAVIOR,
    CONVENTIONAL_WITH_SOLID_PHASE_BEHAVIOR,
    PERMANENT_SOLID_PHASE_BEHAVIOR,
})

_ALIASES = {
    'conventional': CONVENTIONAL_PHASE_BEHAVIOR,
    'fluid': CONVENTIONAL_PHASE_BEHAVIOR,
    'normal': CONVENTIONAL_PHASE_BEHAVIOR,
    'conventional_with_solid': CONVENTIONAL_WITH_SOLID_PHASE_BEHAVIOR,
    'conventional-with-solid': CONVENTIONAL_WITH_SOLID_PHASE_BEHAVIOR,
    'conventional with solid': CONVENTIONAL_WITH_SOLID_PHASE_BEHAVIOR,
    'three_phase': CONVENTIONAL_WITH_SOLID_PHASE_BEHAVIOR,
    'three-phase': CONVENTIONAL_WITH_SOLID_PHASE_BEHAVIOR,
    'three phase': CONVENTIONAL_WITH_SOLID_PHASE_BEHAVIOR,
    'permanent_solid': PERMANENT_SOLID_PHASE_BEHAVIOR,
    'permanent-solid': PERMANENT_SOLID_PHASE_BEHAVIOR,
    'permanent solid': PERMANENT_SOLID_PHASE_BEHAVIOR,
}


def normalize_phase_behavior(value) -> str:
    """Return one canonical phase-behavior name or raise ``ValueError``."""
    text = str(value or CONVENTIONAL_PHASE_BEHAVIOR).strip().lower()
    normalized = _ALIASES.get(text)
    if normalized is None:
        choices = ', '.join(sorted(PHASE_BEHAVIORS))
        raise ValueError(
            f"Unknown component phase_behavior {value!r}; expected one of: {choices}"
        )
    return normalized
