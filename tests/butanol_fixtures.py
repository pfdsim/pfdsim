"""Fixed pre-2859236 butanol/water inputs for numerical solver regressions.

These are regression fixtures, not assertions about the best experimental fit.
Keeping them explicit isolates solver/column behavior from publication updates.
"""


def legacy_butanol_water_parameters(model="NRTL"):
    """Return a fresh parameter dictionary in butanol/water component order."""
    # Overrides merge with runtime records: clear newer terms explicitly, and
    # retain the old extrapolation used by the near-critical regression states.
    common = {"Tmin_K": 298.15, "Tmax_K": 397.75,
              "extrapolation": "unrestricted", "tau_tref": 298.15}
    parameters = {
        "NRTL": {
            "tau12_c": 3.07626601, "tau12_d": -489.80683594,
            "tau12_e": -60.05225941, "tau12_f": 0., "tau12_g": 0.,
            "tau21_c": 4.36302560, "tau21_d": -241.22842207,
            "tau21_e": -8.71391481, "tau21_f": 0., "tau21_g": 0.,
            "alpha12": 0.45131325,
        },
        "UNIQUAC": {
            "tau12_a": -12.81485434, "tau12_b": 1905.74507135,
            "tau12_c": 0., "tau12_d": 0.02169653866, "tau12_e": 0.,
            "tau21_a": 8.68179405, "tau21_b": -1368.13065752,
            "tau21_c": 0., "tau21_d": -0.01678360690, "tau21_e": 0.,
            "model_variant": "standard_uniquac", "use_q_prime": False,
        },
    }
    return {**common, **parameters[model]}


def legacy_butanol_water_interaction(model="NRTL", butanol="butanol", water="water"):
    return {"component1": butanol, "component2": water, "model": model,
            **legacy_butanol_water_parameters(model)}
