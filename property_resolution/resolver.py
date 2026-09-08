from .base import PropertyResolverBase
from .identity import ResolverIdentityMixin
from .vapor_pressure import VaporPressureMixin
from .phase_change import PhaseChangeMixin
from .critical import CriticalPropertiesMixin
from .heat_capacity import HeatCapacityMixin
from .liquid_volume import LiquidVolumeMixin
from .solid_volume import SolidVolumeMixin
from .viscosity import ViscosityMixin
from .surface_tension import SurfaceTensionMixin
from .formation import FormationPropertiesMixin
from .online_phase_change import OnlinePhaseChangeMixin
from .dipole_moment import DipoleMomentMixin
from .radius_of_gyration import RadiusOfGyrationMixin


class PropertyResolver(
    OnlinePhaseChangeMixin,
    VaporPressureMixin,
    PhaseChangeMixin,
    CriticalPropertiesMixin,
    HeatCapacityMixin,
    SolidVolumeMixin,
    LiquidVolumeMixin,
    ViscosityMixin,
    SurfaceTensionMixin,
    FormationPropertiesMixin,
    RadiusOfGyrationMixin,
    DipoleMomentMixin,
    ResolverIdentityMixin,
    PropertyResolverBase,
):
    """Composed resolver preserving the original PropertyResolver API."""

    CACHE_DIR = PropertyResolverBase.CACHE_DIR
    RUNTIME_CACHE_PATH = PropertyResolverBase.RUNTIME_CACHE_PATH
    LIQUID_VOLUME_ZRA_CACHE_PATH = PropertyResolverBase.LIQUID_VOLUME_ZRA_CACHE_PATH
