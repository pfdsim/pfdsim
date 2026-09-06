from typing import Optional, Union

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..chemical_properties import ChemicalDatabase
else:
    from chemical_properties import ChemicalDatabase

from .common import ThermodynamicsError
from .base import IdealThermodynamics
from .steam import SteamThermodynamics
from .eos import RKThermodynamics, CubicEOSThermodynamics
from .psrk_thermo import PSRKThermodynamics
from .unifac_models import (UNIFACThermodynamics, UNIFAC2Thermodynamics, UNIFDMDThermodynamics, UNIFM2Thermodynamics, UNIFNISTThermodynamics, UNIFACVDMThermodynamics, UNIFDMDVDMThermodynamics, UNIFNISTVDMThermodynamics)
from .nrtl_uniquac import (
    NRTLThermodynamics, NRTLVDMThermodynamics,
    UNIQUACThermodynamics, UNIQUACVDMThermodynamics,
)
from .gamma_phi import (UNIQUACRKThermodynamics, UNIQUACPRThermodynamics, NRTLRKThermodynamics, NRTLPRThermodynamics, UNIFACRKThermodynamics, UNIFACPRThermodynamics, UNIFDMDRKThermodynamics, UNIFDMDPRThermodynamics, UNIFNISTRKThermodynamics, UNIFNISTPRThermodynamics)

# One canonical name per supported thermo method. Sweep-style tests
# iterate this tuple, so methods added here are covered automatically.
SUPPORTED_METHODS = (
    'IDEAL', 'STEAM',
    'RK', 'SRK', 'PR', 'PSRK', 'RKS-BM', 'PR-BM', 'SRK-MC', 'PR-MC',
    'SRK-TWU', 'PR-TWU', 'PRSV1', 'PRSV2',
    'UNIFAC', 'UNIFAC2', 'UNIFDMD', 'UNIFM2', 'UNIFNIST',
    'UNIFAC-VDM', 'UNIFDMD-VDM', 'UNIFNIST-VDM',
    'UNIFAC-RK', 'UNIFAC-PR', 'UNIFDMD-RK', 'UNIFDMD-PR',
    'UNIFNIST-RK', 'UNIFNIST-PR',
    'NRTL', 'NRTL-VDM', 'NRTL-RK', 'NRTL-PR',
    'UNIQUAC', 'UNIQUAC-VDM', 'UNIQUAC-RK', 'UNIQUAC-PR',
)


def create_thermodynamics(components: list[str], 
                          method: str = 'IDEAL',
                          db: Optional[ChemicalDatabase] = None,
                          unifac_groups: Optional[dict] = None,
                          interaction_overrides: Optional[list[dict]] = None,
                          interaction_estimation: Optional[list[dict]] = None) -> Union[
                              IdealThermodynamics,
                              SteamThermodynamics,
                              RKThermodynamics,
                              CubicEOSThermodynamics,
                              PSRKThermodynamics,
                              UNIFACThermodynamics,
                              UNIFAC2Thermodynamics,
                              UNIFDMDThermodynamics,
                              UNIFM2Thermodynamics,
                              UNIFNISTThermodynamics,
                              UNIFACVDMThermodynamics,
                              UNIFDMDVDMThermodynamics,
                              UNIFNISTVDMThermodynamics,
                              UNIFACRKThermodynamics,
                              UNIFACPRThermodynamics,
                              UNIFDMDRKThermodynamics,
                              UNIFDMDPRThermodynamics,
                              UNIFNISTRKThermodynamics,
                              UNIFNISTPRThermodynamics,
                              NRTLThermodynamics,
                              NRTLVDMThermodynamics,
                              NRTLRKThermodynamics,
                              NRTLPRThermodynamics,
                              UNIQUACThermodynamics,
                              UNIQUACVDMThermodynamics,
                              UNIQUACRKThermodynamics,
                              UNIQUACPRThermodynamics,
                          ]:
    """
    Factory function to create thermodynamics calculator.
    
    Args:
        components: List of component symbols
        method: 'IDEAL', 'STEAM', 'IF97', 'RK', 'SRK', 'PR', 'PSRK', 'RKS-BM', 'PR-BM',
            'SRK-MC', 'PR-MC', 'PRSV1', 'PRSV2', 'SRK-TWU', 'PR-TWU',
            'UNIFAC', 'UNIFAC2', 'UNIFDMD', 'UNIFM2', 'UNIFNIST',
            'UNIFAC-VDM', 'UNIFDMD-VDM',
            'UNIFNIST-VDM', 'UNIFAC-RK', 'UNIFDMD-RK', 'UNIFNIST-RK',
            'UNIFAC-PR', 'UNIFDMD-PR', 'UNIFNIST-PR', 'NRTL', 'NRTL-RK',
            'NRTL-PR', 'NRTL-VDM', 'UNIQUAC', 'UNIQUAC-VDM', 'UNIQUAC-RK', or 'UNIQUAC-PR'
        db: Chemical database (uses default if None)
        unifac_groups: Dict mapping components to UNIFAC groups (for UNIFAC method)
        
    Returns:
        Thermodynamics calculator instance
    """
    method = method.upper()

    if method == 'IDEAL':
        return IdealThermodynamics(components, db, interaction_overrides)
    elif method in ('STEAM', 'IF97', 'IAPWS-IF97', 'IAPWS_IF97'):
        return SteamThermodynamics(components, db)
    elif method in ('RK', 'REDLICH-KWONG', 'REDLICHKWONG'):
        return RKThermodynamics(components, db, interaction_overrides)
    elif method in ('SRK', 'RKS', 'SOAVE-REDLICH-KWONG', 'RK-SOAVE'):
        return CubicEOSThermodynamics(components, 'SRK', db, interaction_overrides)
    elif method in ('PR', 'PENG-ROBINSON', 'PENG_ROBINSON'):
        return CubicEOSThermodynamics(components, 'PR', db, interaction_overrides)
    elif method in ('PSRK', 'PREDICTIVE-SRK', 'PREDICTIVE_SRK'):
        return PSRKThermodynamics(components, db, interaction_overrides)
    elif method in ('RKS-BM', 'RKS_BM', 'SRK-BM', 'SRK_BM'):
        return CubicEOSThermodynamics(components, 'RKS-BM', db, interaction_overrides)
    elif method in ('PR-BM', 'PR_BM', 'PENG-ROBINSON-BM', 'PENG_ROBINSON_BM'):
        return CubicEOSThermodynamics(components, 'PR-BM', db, interaction_overrides)
    elif method in ('SRK-MC', 'SRK_MC', 'RKS-MC', 'RKS_MC', 'RK-SOAVE-MC'):
        return CubicEOSThermodynamics(components, 'SRK-MC', db, interaction_overrides)
    elif method in ('PR-MC', 'PR_MC', 'PENG-ROBINSON-MC', 'PENG_ROBINSON_MC'):
        return CubicEOSThermodynamics(components, 'PR-MC', db, interaction_overrides)
    elif method in ('SRK-TWU', 'SRK_TWU', 'RKS-TWU', 'RKS_TWU', 'RK-SOAVE-TWU'):
        return CubicEOSThermodynamics(components, 'SRK-TWU', db, interaction_overrides)
    elif method in ('PR-TWU', 'PR_TWU', 'PENG-ROBINSON-TWU', 'PENG_ROBINSON_TWU'):
        return CubicEOSThermodynamics(components, 'PR-TWU', db, interaction_overrides)
    elif method in ('PRSV', 'PRSV1', 'PR-SV', 'PR-SV1', 'PENG-ROBINSON-SV',
                    'PENG_ROBINSON_SV', 'PENG-ROBINSON-SV1',
                    'PENG_ROBINSON_SV1', 'PENG-ROBINSON-STRYJEK-VERA',
                    'PENG_ROBINSON_STRYJEK_VERA'):
        return CubicEOSThermodynamics(components, 'PRSV1', db, interaction_overrides)
    elif method in ('PRSV2', 'PR-SV2', 'PENG-ROBINSON-SV2', 'PENG_ROBINSON_SV2'):
        return CubicEOSThermodynamics(components, 'PRSV2', db, interaction_overrides)
    elif method == 'UNIFAC':
        return UNIFACThermodynamics(components, db, unifac_groups, interaction_overrides=interaction_overrides)
    elif method == 'UNIFAC2':
        return UNIFAC2Thermodynamics(components, db, unifac_groups, interaction_overrides=interaction_overrides)
    elif method in ('UNIFDMD', 'UNIFAC-DMD', 'UNIFAC_DMD', 'DORTMUND-UNIFAC',
                    'DORTMUND_UNIFAC', 'MODIFIED-UNIFAC', 'MODIFIED_UNIFAC'):
        return UNIFDMDThermodynamics(components, db, unifac_groups, interaction_overrides=interaction_overrides)
    elif method == 'UNIFM2':
        return UNIFM2Thermodynamics(components, db, unifac_groups, interaction_overrides=interaction_overrides)
    elif method in ('UNIFNIST', 'UNIFAC-NIST', 'UNIFAC_NIST', 'NIST-UNIFAC',
                    'NIST_UNIFAC', 'NIST-MODIFIED-UNIFAC', 'NIST_MODIFIED_UNIFAC'):
        return UNIFNISTThermodynamics(components, db, unifac_groups, interaction_overrides=interaction_overrides)
    elif method in ('UNIFAC-VDM', 'UNIFAC_VDM'):
        return UNIFACVDMThermodynamics(components, db, unifac_groups, interaction_overrides)
    elif method in ('UNIFDMD-VDM', 'UNIFDMD_VDM', 'UNIFAC-DMD-VDM',
                    'UNIFAC_DMD_VDM', 'DORTMUND-UNIFAC-VDM',
                    'DORTMUND_UNIFAC_VDM'):
        return UNIFDMDVDMThermodynamics(components, db, unifac_groups, interaction_overrides)
    elif method in ('UNIFNIST-VDM', 'UNIFNIST_VDM', 'UNIFAC-NIST-VDM',
                    'UNIFAC_NIST_VDM', 'NIST-UNIFAC-VDM',
                    'NIST_UNIFAC_VDM'):
        return UNIFNISTVDMThermodynamics(components, db, unifac_groups, interaction_overrides)
    elif method in ('UNIFAC-RK', 'UNIFAC_RK', 'GAMMA-PHI-RK', 'GAMMA_PHI_RK'):
        return UNIFACRKThermodynamics(components, db, unifac_groups, interaction_overrides)
    elif method in ('UNIFAC-PR', 'UNIFAC_PR', 'UNIFAC-PENG-ROBINSON',
                    'UNIFAC_PENG_ROBINSON', 'GAMMA-PHI-PR', 'GAMMA_PHI_PR'):
        return UNIFACPRThermodynamics(components, db, unifac_groups, interaction_overrides)
    elif method in ('UNIFDMD-RK', 'UNIFDMD_RK', 'UNIFAC-DMD-RK', 'UNIFAC_DMD_RK',
                    'DORTMUND-UNIFAC-RK', 'DORTMUND_UNIFAC_RK'):
        return UNIFDMDRKThermodynamics(components, db, unifac_groups, interaction_overrides)
    elif method in ('UNIFDMD-PR', 'UNIFDMD_PR', 'UNIFAC-DMD-PR', 'UNIFAC_DMD_PR',
                    'DORTMUND-UNIFAC-PR', 'DORTMUND_UNIFAC_PR',
                    'MODIFIED-UNIFAC-PR', 'MODIFIED_UNIFAC_PR'):
        return UNIFDMDPRThermodynamics(components, db, unifac_groups, interaction_overrides)
    elif method in ('UNIFNIST-RK', 'UNIFNIST_RK', 'UNIFAC-NIST-RK',
                    'UNIFAC_NIST_RK', 'NIST-UNIFAC-RK', 'NIST_UNIFAC_RK'):
        return UNIFNISTRKThermodynamics(components, db, unifac_groups, interaction_overrides)
    elif method in ('UNIFNIST-PR', 'UNIFNIST_PR', 'UNIFAC-NIST-PR',
                    'UNIFAC_NIST_PR', 'NIST-UNIFAC-PR', 'NIST_UNIFAC_PR',
                    'NIST-MODIFIED-UNIFAC-PR', 'NIST_MODIFIED_UNIFAC_PR'):
        return UNIFNISTPRThermodynamics(components, db, unifac_groups, interaction_overrides)
    elif method == 'NRTL':
        return NRTLThermodynamics(components, db, interaction_overrides, interaction_estimation, unifac_groups)
    elif method in ('NRTL-VDM', 'NRTL_VDM'):
        return NRTLVDMThermodynamics(components, db, interaction_overrides, interaction_estimation, unifac_groups)
    elif method in ('NRTL-RK', 'NRTL_RK'):
        return NRTLRKThermodynamics(components, db, interaction_overrides, interaction_estimation, unifac_groups)
    elif method in ('NRTL-PR', 'NRTL_PR', 'NRTL-PENG-ROBINSON', 'NRTL_PENG_ROBINSON'):
        return NRTLPRThermodynamics(components, db, interaction_overrides, interaction_estimation, unifac_groups)
    elif method == 'UNIQUAC':
        return UNIQUACThermodynamics(components, db, interaction_overrides, interaction_estimation, unifac_groups)
    elif method in ('UNIQUAC-VDM', 'UNIQUAC_VDM'):
        return UNIQUACVDMThermodynamics(components, db, interaction_overrides, interaction_estimation, unifac_groups)
    elif method in ('UNIQUAC-RK', 'UNIQUAC_RK'):
        return UNIQUACRKThermodynamics(components, db, interaction_overrides, interaction_estimation, unifac_groups)
    elif method in ('UNIQUAC-PR', 'UNIQUAC_PR', 'UNIQUAC-PENG-ROBINSON',
                    'UNIQUAC_PENG_ROBINSON'):
        return UNIQUACPRThermodynamics(components, db, interaction_overrides, interaction_estimation, unifac_groups)
    else:
        raise ThermodynamicsError(f"Unknown thermodynamic method: {method}")


def create_experiment_thermo(components: list[str]) -> IdealThermodynamics:
    """Create thermodynamics instance - alias for compatibility"""
    return IdealThermodynamics(components)
