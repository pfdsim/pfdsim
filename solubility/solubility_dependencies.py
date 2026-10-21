"""Authoritative repository-relative inputs shared with pfdsim."""

SHARED_CODE_FILES = (
    'unifac.py',
    'compiled_unifac.py',
    'unifac_fragmenter.py',
    'lyngby_parameters.py',
    'compiled_cache.py',
    'physical_constants.py',
)

# Entry tables passed to UNIFACModel, relative to pfdsim/data.
PARAMETERS = {
    'UNIFAC': 'unifac_params.json',
    'UNIFAC2': 'unifac2/UNIFAC2rq.csv',
    'UNIFDMD': 'unifac_dmd.txt',
    'UNIFM2': 'unifac2/UNIFM2rq.xlsx',
    'UNIFNIST': 'nist_modified_unifac_params.json',
    'UNIFLBY': 'lyngby_unifac.json',
}

# Companion interaction matrices opened by the corresponding entry-table loaders.
PARAMETER_FILES = tuple('data/' + name for name in PARAMETERS.values()) + (
    'data/unifac2/UNIFAC2int.csv',
    'data/unifac2/UNIFM2intA.xlsx',
    'data/unifac2/UNIFM2intB.xlsx',
)
SHARED_FILES = SHARED_CODE_FILES + PARAMETER_FILES
