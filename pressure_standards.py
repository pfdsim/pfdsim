"""Named pressure standards used by pfdsim.

Two pressures are intentionally distinct:

* Thermochemical standard-state properties are referenced to 1 bar. This
  applies to standard entropies, formation properties, fugacity-standard
  corrections, and ideal-gas pressure entropy corrections.
* Normal boiling points are referenced to 1 atm, exactly 1.01325 bar. This
  applies to Tb, vapor-pressure rows anchored at Tb, and corresponding-states
  correlations that use the normal boiling point.

Keeping both names explicit avoids silently mixing modern thermochemistry
tables with older normal-boiling conventions.
"""

THERMOCHEMICAL_STANDARD_PRESSURE_BAR = 1.0
NORMAL_BOILING_PRESSURE_BAR = 1.01325
ATM_PRESSURE_BAR = NORMAL_BOILING_PRESSURE_BAR

