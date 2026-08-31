"""Shared declarations for universal fluid-phase equilibrium policy."""

FLUID_PHASE_MODELS = frozenset({'VLE', 'VL(L)E', 'VLLE'})

LLE_CAPABLE_THERMO_METHODS = frozenset({
    'UNIFAC', 'UNIFAC2', 'UNIFDMD', 'UNIFM2', 'UNIFNIST',
    'NRTL', 'UNIQUAC',
    'NRTL-VDM', 'UNIFAC-VDM', 'UNIFDMD-VDM', 'UNIFNIST-VDM', 'UNIQUAC-VDM',
    'UNIFAC-RK', 'UNIFDMD-RK', 'UNIFNIST-RK', 'NRTL-RK', 'UNIQUAC-RK',
    'UNIFAC-PR', 'UNIFDMD-PR', 'UNIFNIST-PR', 'NRTL-PR', 'UNIQUAC-PR',
})


def normalize_fluid_phase_model(value) -> str:
    """Return VLE, VL(L)E, or VLLE from one public/compatibility spelling."""
    text = str(value or 'VLE').strip().upper().replace(' ', '')
    aliases = {
        'VLE': 'VLE',
        'VL(L)E': 'VL(L)E',
        'VLL(E)': 'VL(L)E',
        'ADAPTIVE_VLLE': 'VL(L)E',
        'ADAPTIVE-VLLE': 'VL(L)E',
        'SPINODAL_VLLE': 'VL(L)E',
        'SPINODAL-VLLE': 'VL(L)E',
        'VLLE': 'VLLE',
    }
    normalized = aliases.get(text)
    if normalized is None:
        raise ValueError(
            f"Invalid fluid phase model {value!r}; expected VLE, VL(L)E, or VLLE"
        )
    return normalized
