from typing import Optional

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..chemical_properties import ChemicalDatabase
else:
    from chemical_properties import ChemicalDatabase

from .common import ThermodynamicsError
from .activity import ActivityCoefficientThermodynamics, VaporDimerizationActivityMixin


def resolve_component_unifac_groups(
    comp: str,
    props,
    db,
    variant: str,
    get_unifac_groups_func=None,
) -> Optional[dict]:
    """Resolve canonical UNIFAC groups without constructing a thermo model."""
    if get_unifac_groups_func is None:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from ..unifac import get_unifac_groups as get_unifac_groups_func
        else:
            from unifac import get_unifac_groups as get_unifac_groups_func

    identifiers = []
    if props is not None:
        # Process-local symbols are never chemical lookup identifiers.
        # Use only identity resolved from the PFD lookup field.
        identifiers.extend((props.name, props.CAS, props.formula))

    expected_mw = None
    try:
        expected_mw = float(getattr(props, 'MW', None))
    except (TypeError, ValueError):
        pass
    for identifier in identifiers:
        if not identifier:
            continue
        try:
            return get_unifac_groups_func(
                identifier,
                variant=variant,
                expected_mw=expected_mw,
            )
        except ValueError:
            pass

    smiles = getattr(props, 'smiles', None) if props is not None else None
    if not smiles and hasattr(db, 'resolve_smiles_info'):
        for identifier in identifiers:
            if not identifier:
                continue
            result = db.resolve_smiles_info(
                identifier,
                fetch_online=True,
                props=props,
            )
            smiles = result.smiles if result else None
            if smiles:
                break

    if smiles:
        label = getattr(props, 'name', None) or 'resolved component'
        try:
            return get_unifac_groups_func(
                label,
                smiles=smiles,
                variant=variant,
            )
        except ValueError:
            # A resolved structure is not necessarily representable by the
            # selected UNIFAC variant. Let the caller apply its ordinary
            # non-condensable exclusion policy.
            return None
    return None


def can_exclude_component_from_unifac(props) -> bool:
    """Return whether an unfragmentable component may remain an ideal solute."""
    if props is None:
        return False
    if getattr(props, 'phase_at_STP', None) == 'gas':
        return True
    boiling_point = getattr(props, 'Tb', None)
    return boiling_point is not None and boiling_point < 250.0

class UNIFACThermodynamics(ActivityCoefficientThermodynamics):
    """
    UNIFAC (Universal Functional Activity Coefficient) thermodynamic calculator.
    
    Uses UNIFAC group contribution method for liquid phase activity coefficients.
    Vapor phase is treated as ideal gas.
    
    VLE calculation:
        K_i = γ_i * P_sat_i / P
    
    where γ_i is calculated from UNIFAC.
    
    Requires UNIFAC groups to be specified for each component.
    """
    unifac_variant = 'UNIFAC'
    default_unifac_data_filename = 'unifac_params.json'
    
    def __init__(self, components: list[str], 
                 db: Optional[ChemicalDatabase] = None,
                 unifac_groups: Optional[dict[str, dict]] = None,
                 unifac_data_path: Optional[str] = None,
                 interaction_overrides: Optional[list[dict]] = None):
        """
        Initialize UNIFAC thermodynamics.
        
        Args:
            components: List of chemical symbols
            db: Chemical database (uses default if None)
            unifac_groups: Dict mapping component names to their UNIFAC groups
                          e.g., {'C2H5OH': {'CH3': 1, 'CH2': 1, 'OH': 1}}
                          If None, will try to look up from known molecules
            unifac_data_path: Path to UNIFAC parameter file (Excel or JSON)
        """
        super().__init__(components, db, interaction_overrides)
        
        # Import UNIFAC module
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from ..unifac import UNIFACModel, get_unifac_groups
        else:
            from unifac import UNIFACModel, get_unifac_groups
        from pathlib import Path
        
        # Find UNIFAC parameter file
        if unifac_data_path is None:
            # Try default locations
            data_dir = Path(__file__).resolve().parent.parent / 'data'
            default_data_path = data_dir / self.default_unifac_data_filename
            if default_data_path.exists():
                unifac_data_path = str(default_data_path)
            elif self.unifac_variant == 'UNIFAC' and (data_dir / 'UNIFAC.xlsx').exists():
                unifac_data_path = str(data_dir / 'UNIFAC.xlsx')
        
        # Create UNIFAC model
        self.unifac = UNIFACModel(unifac_data_path)
        self._activity_cache: dict[tuple, dict[str, float]] = {}
        self._compiled_unifac = None
        self._compiled_lle = None
        self._compiled_vlle = None
        self._compiled_vlle_initialized = False
        
        # Get UNIFAC groups for each component
        self.component_groups: dict[str, dict[str, int]] = {}
        self.unifac_excluded_components: set[str] = set()
        
        for comp in components:
            if unifac_groups and comp in unifac_groups:
                # Use provided groups
                self.component_groups[comp] = unifac_groups[comp]
            else:
                props = self.props.get(comp)
                groups = self._resolve_component_unifac_groups(
                    comp,
                    props,
                    get_unifac_groups,
                )
                if groups is not None:
                    self.component_groups[comp] = groups
                elif self._can_exclude_from_unifac(comp):
                    # Gas classification is only a fallback for components
                    # that cannot participate in the liquid group model.
                    # Volatile condensables such as acetaldehyde still have
                    # valid UNIFAC groups and must remain available whenever
                    # they are treated as ordinary liquid components.
                    self.unifac_excluded_components.add(comp)
                else:
                    raise ThermodynamicsError(
                        f"UNIFAC groups not found for '{comp}'. "
                        "Please specify groups manually or provide a resolvable SMILES."
                    )
        
        # Calculate r and q for each component
        self.r = {}
        self.q = {}
        for comp, groups in self.component_groups.items():
            r_i, q_i = self.unifac.calculate_r_q(groups)
            self.r[comp] = r_i
            self.q[comp] = q_i

        self._warn_missing_unifac_group_interactions()

        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..compiled_unifac import CompiledUNIFACBackend
            else:
                from compiled_unifac import CompiledUNIFACBackend
            compiled_components = [
                comp for comp in self.components
                if comp in self.component_groups
            ]
            if compiled_components and self.unifac_variant in (
                'UNIFAC', 'UNIFAC2', 'UNIFDMD', 'UNIFM2', 'UNIFNIST'
            ):
                self._compiled_unifac = CompiledUNIFACBackend.from_model(
                    self.unifac,
                    compiled_components,
                    self.component_groups,
                )
        except Exception:
            self._compiled_unifac = None

        if self._compiled_unifac is not None:
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..compiled_lle import CompiledLLEBackend
                else:
                    from compiled_lle import CompiledLLEBackend
                self._compiled_lle = CompiledLLEBackend.from_unifac_backend(
                    self._compiled_unifac
                )
            except Exception:
                self._compiled_lle = None

    def prepare_compiled_backends(
        self,
        *,
        need_lle: bool = False,
        need_vlle: bool = False,
    ) -> None:
        """Compile only numerical paths reachable by this flowsheet."""
        if need_lle and self._compiled_lle is not None:
            self._compiled_lle.compile_kernels()
        if need_vlle:
            backend = self.compiled_vlle_backend()
            if backend is not None:
                backend.compile_kernels()

    def _resolve_component_unifac_groups(self, comp: str, props,
                                         get_unifac_groups_func) -> Optional[dict]:
        """Resolve identity first, then pass only resolved structures to the fragmenter."""
        return resolve_component_unifac_groups(
            comp,
            props,
            self.db,
            self.unifac_variant,
            get_unifac_groups_func,
        )

    def _can_exclude_from_unifac(self, comp: str) -> bool:
        return can_exclude_component_from_unifac(self.props.get(comp))

    def _warn_missing_unifac_group_interactions(self) -> None:
        main_groups: dict[int, str] = {}
        for groups in self.component_groups.values():
            try:
                resolved = self.unifac._resolve_groups(groups)
            except Exception:
                continue
            for subgroup in resolved:
                if subgroup not in self.unifac.subgroups:
                    continue
                main = self.unifac.subgroups[subgroup].main_group
                main_groups[main] = self.unifac.subgroups[subgroup].main_group_name

        missing = []
        for main_i in sorted(main_groups):
            for main_j in sorted(main_groups):
                if main_i == main_j:
                    continue
                has_interaction = (
                    (main_i, main_j) in self.unifac.interaction_coefficients
                    or (main_i, main_j) in self.unifac.interactions
                )
                if not has_interaction:
                    missing.append((main_i, main_j))

        if not missing:
            return

        examples = ', '.join(
            f"{main_groups[i]}->{main_groups[j]}"
            for i, j in missing[:8]
        )
        if len(missing) > 8:
            examples += f", ... (+{len(missing) - 8} more)"
        self.add_warning(
            f"{self.unifac_variant} group interaction parameters missing for "
            f"{len(missing)} ordered main-group pair(s); using zero interaction "
            f"parameters for missing pairs: {examples}."
        )

    def liquid_liquid_equilibrium(self, composition: dict[str, float], T: float,
                                  max_iter: int = 100, tol: float = 1e-6,
                                  *, allow_unconverged_candidate: bool = False) -> tuple[bool, dict, dict, float]:
        self._validate_lle_solver_controls(max_iter, tol)
        normalized = self._normalize_lle_composition(composition)
        # Prefer the compiled splitter whenever it can represent the component
        # set.  The readable implementation remains the fallback only for
        # component sets unsupported by the compiled backend.
        if self._compiled_lle is not None:
            if allow_unconverged_candidate:
                split = self._compiled_lle.split(
                    normalized, T, max_iter=max_iter, tol=tol,
                    allow_unconverged_candidate=True,
                )
            else:
                split = self._compiled_lle.split(
                    normalized, T, max_iter=max_iter, tol=tol
                )
            if split is not None:
                return split
            if len(normalized) > 2:
                return False, dict(normalized), dict(normalized), 0.0
            binary_split = self._binary_liquid_liquid_equilibrium(
                normalized,
                T,
                max_iter,
                tol,
                prefer_adaptive_starts=True,
            )
            if binary_split is not None:
                return binary_split
            return False, dict(normalized), dict(normalized), 0.0
        return super().liquid_liquid_equilibrium(
            normalized, T, max_iter, tol,
            allow_unconverged_candidate=allow_unconverged_candidate,
        )
    
    def activity_coefficients(self, T: float, composition: dict[str, float]) -> dict[str, float]:
        """
        Calculate liquid phase activity coefficients using UNIFAC.
        
        Args:
            T: Temperature [K]
            composition: Liquid mole fractions
            
        Returns:
            Dictionary of activity coefficients
        """
        # Build component list and mole fraction list in consistent order
        cache_key = (
            float(T),
            tuple((comp, float(composition.get(comp, 0.0))) for comp in self.components),
        )
        cached = self._activity_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        x_full = [max(float(composition.get(comp, 0.0)), 0.0) for comp in self.components]
        if len(self.component_groups) < 2:
            return {comp: 1.0 for comp in self.components}

        interaction_T = float(T)

        if self._compiled_unifac is not None:
            active_components = self._compiled_unifac.components
            x_compiled = [
                max(float(composition.get(comp, 0.0)), 0.0)
                for comp in active_components
            ]
            x_sum = sum(x_compiled)
            if x_sum <= 0.0:
                return {comp: 1.0 for comp in self.components}
            x_compiled = [x / x_sum for x in x_compiled]
            gamma_list = self._compiled_unifac.activity_coefficients(
                x_compiled,
                interaction_T,
            )
            gamma = {comp: 1.0 for comp in self.components}
            for i, comp in enumerate(active_components):
                gamma[comp] = gamma_list[i]
            if len(self._activity_cache) > 20000:
                self._activity_cache.clear()
            self._activity_cache[cache_key] = dict(gamma)
            return dict(gamma)

        comp_list = []
        x_list = []
        groups_list = []

        for comp, x_i in zip(self.components, x_full):
            if comp not in self.component_groups:
                continue
            comp_list.append(comp)
            x_list.append(x_i)
            groups_list.append(self.component_groups[comp])

        x_sum = sum(x_list)
        if len(comp_list) < 2 or x_sum <= 0.0:
            gamma = {comp: 1.0 for comp in self.components}
            if len(self._activity_cache) > 20000:
                self._activity_cache.clear()
            self._activity_cache[cache_key] = dict(gamma)
            return gamma

        # Normalize mole fractions
        x_list = [x / x_sum for x in x_list]
        
        gamma_list = self.unifac.activity_coefficients(
            groups_list,
            x_list,
            interaction_T,
        )
        
        # Build result dictionary
        gamma = {}
        for i, comp in enumerate(comp_list):
            gamma[comp] = gamma_list[i]
        
        # Set gamma = 1 for missing components
        for comp in self.components:
            if comp not in gamma:
                gamma[comp] = 1.0
        
        if len(self._activity_cache) > 20000:
            self._activity_cache.clear()
        self._activity_cache[cache_key] = dict(gamma)

        return gamma

    def excess_enthalpy(self, composition: dict[str, float], T: float) -> float:
        backend = self._compiled_unifac
        if backend is None:
            return super().excess_enthalpy(composition, T)
        total = sum(
            max(float(composition.get(comp, 0.0)), 0.0)
            for comp in self.components
        )
        if total <= 0.0:
            return 0.0
        x_active = [
            max(float(composition.get(comp, 0.0)), 0.0) / total
            for comp in backend.components
        ]
        active_fraction = sum(x_active)
        if active_fraction <= 0.0:
            return 0.0
        x_active = [value / active_fraction for value in x_active]
        dT = max(1e-3, 1e-4 * T)
        T_low = max(1.0, T - dT)
        T_high = T + dT
        activity_T_low = T_low
        activity_T_high = T_high
        return active_fraction * backend.excess_enthalpy(
            x_active,
            T,
            activity_T_low,
            activity_T_high,
        )


class UNIFDMDThermodynamics(UNIFACThermodynamics):
    """Dortmund modified UNIFAC activity-coefficient model."""

    unifac_variant = 'UNIFDMD'
    default_unifac_data_filename = 'unifac_dmd.txt'


class UNIFAC2Thermodynamics(UNIFACThermodynamics):
    """UNIFAC 2.0 with the machine-learning-completed interaction matrix."""

    unifac_variant = 'UNIFAC2'
    default_unifac_data_filename = 'unifac2/UNIFAC2rq.csv'


class UNIFM2Thermodynamics(UNIFACThermodynamics):
    """Modified UNIFAC 2.0 with machine-learning-completed interactions."""

    unifac_variant = 'UNIFM2'
    default_unifac_data_filename = 'unifac2/UNIFM2rq.xlsx'


class UNIFNISTThermodynamics(UNIFACThermodynamics):
    """NIST-modified UNIFAC activity-coefficient model."""

    unifac_variant = 'UNIFNIST'
    default_unifac_data_filename = 'nist_modified_unifac_params.json'


class UNIFACVDMThermodynamics(VaporDimerizationActivityMixin, UNIFACThermodynamics):
    """UNIFAC liquid activity model with vapor dimerization correction."""

    def __init__(self, components: list[str],
                 db: Optional[ChemicalDatabase] = None,
                 unifac_groups: Optional[dict[str, dict]] = None,
                 interaction_overrides: Optional[list[dict]] = None):
        super().__init__(
            components,
            db,
            unifac_groups,
            interaction_overrides=interaction_overrides,
        )
        self._initialize_vdm()


class UNIFDMDVDMThermodynamics(VaporDimerizationActivityMixin, UNIFDMDThermodynamics):
    """Dortmund modified UNIFAC liquid activity model with vapor dimerization correction."""

    def __init__(self, components: list[str],
                 db: Optional[ChemicalDatabase] = None,
                 unifac_groups: Optional[dict[str, dict]] = None,
                 interaction_overrides: Optional[list[dict]] = None):
        super().__init__(
            components,
            db,
            unifac_groups,
            interaction_overrides=interaction_overrides,
        )
        self._initialize_vdm()


class UNIFNISTVDMThermodynamics(VaporDimerizationActivityMixin, UNIFNISTThermodynamics):
    """NIST-modified UNIFAC liquid activity model with vapor dimerization correction."""

    def __init__(self, components: list[str],
                 db: Optional[ChemicalDatabase] = None,
                 unifac_groups: Optional[dict[str, dict]] = None,
                 interaction_overrides: Optional[list[dict]] = None):
        super().__init__(
            components,
            db,
            unifac_groups,
            interaction_overrides=interaction_overrides,
        )
        self._initialize_vdm()
