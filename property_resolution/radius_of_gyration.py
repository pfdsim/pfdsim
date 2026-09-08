"""Molecular radii of gyration from the shared optimized gas geometry."""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from .common import PropertyResolutionError, PropertyResolutionResult


GYRATION_GEOMETRY_QUALITY = 0.85


class RadiusOfGyrationMixin:
    """Resolve conventional ``R_g`` and Thompson's modified radius ``R'``."""

    def resolve_radius_of_gyration(
        self,
        identifier: str,
        props: Optional[dict[str, Any]] = None,
        *,
        allow_online: bool = True,
    ) -> PropertyResolutionResult:
        """Resolve the mass-weighted conventional radius of gyration [angstrom]."""
        return self._resolve_gyration_radius(
            identifier,
            props,
            quantity='radius_of_gyration',
            aliases=('R', 'Rg', 'R_g', 'radius_of_gyration_A'),
            allow_online=allow_online,
        )

    def resolve_modified_radius_of_gyration(
        self,
        identifier: str,
        props: Optional[dict[str, Any]] = None,
        *,
        allow_online: bool = True,
    ) -> PropertyResolutionResult:
        """Resolve Thompson's HOC mean radius of gyration ``R'`` [angstrom]."""
        return self._resolve_gyration_radius(
            identifier,
            props,
            quantity='modified_radius_of_gyration',
            aliases=(
                'thompson_radius_of_gyration',
                'radius_of_gyration_prime',
                'R_prime',
                'R_HOC',
            ),
            allow_online=allow_online,
        )

    def resolve_radii_of_gyration(
        self,
        identifier: str,
        props: Optional[dict[str, Any]] = None,
        *,
        allow_online: bool = True,
    ) -> dict[str, PropertyResolutionResult]:
        """Resolve both conventional ``R_g`` and Thompson ``R'`` radii."""
        coerced = self._coerce_props(identifier, props, allow_online=allow_online)
        return {
            'radius_of_gyration': self.resolve_radius_of_gyration(
                identifier,
                coerced,
                allow_online=allow_online,
            ),
            'modified_radius_of_gyration': self.resolve_modified_radius_of_gyration(
                identifier,
                coerced,
                allow_online=allow_online,
            ),
        }

    def _resolve_gyration_radius(
        self,
        identifier: str,
        props: Optional[dict[str, Any]],
        *,
        quantity: str,
        aliases: tuple[str, ...],
        allow_online: bool,
    ) -> PropertyResolutionResult:
        props = self._coerce_props(identifier, props, allow_online=allow_online)
        for key in (quantity, *aliases):
            provided = self._source_result_for_value(props, key, units='angstrom')
            if provided is None:
                continue
            try:
                value = float(provided.value)
            except (TypeError, ValueError) as error:
                raise PropertyResolutionError(
                    f"Invalid provided {quantity} for {identifier!r}"
                ) from error
            if not math.isfinite(value) or value < 0.0:
                raise PropertyResolutionError(
                    f"Invalid provided {quantity} for {identifier!r}: expected "
                    "a finite nonnegative value in angstrom"
                )
            notes = str(provided.notes or '')
            if 'angstrom' not in notes.lower():
                notes = f"{notes}; units angstrom" if notes else 'units angstrom'
            return PropertyResolutionResult(
                value=value,
                source=provided.source,
                method=provided.method,
                quality=provided.quality,
                notes=notes,
            )

        radii, geometry_note = self._gyration_radii_from_shared_geometry(
            identifier,
            props,
            allow_online=allow_online,
        )
        conventional, modified, linear = radii
        value = conventional if quantity == 'radius_of_gyration' else modified
        definition = (
            'mass-weighted root-mean-square distance from the center of mass'
            if quantity == 'radius_of_gyration'
            else (
                "Thompson R' from the two nonzero principal moments (linear molecule)"
                if linear
                else "Thompson R' from all three principal moments (nonlinear molecule)"
            )
        )
        return PropertyResolutionResult(
            value=value,
            source='calculated',
            method=f'gfn2_xtb_geometry_{quantity}',
            quality=GYRATION_GEOMETRY_QUALITY,
            notes=(
                f'{definition}; {geometry_note}; computed-geometry radius '
                'benchmark approximately 4% MAE; units angstrom'
            ),
        )

    def _gyration_radii_from_shared_geometry(
        self,
        identifier: str,
        props: dict[str, Any],
        *,
        allow_online: bool,
    ) -> tuple[tuple[float, float, bool], str]:
        smiles_result = self._resolve_smiles_result(
            identifier,
            props,
            allow_online=allow_online,
        )
        smiles = (
            str(smiles_result.value).strip()
            if smiles_result is not None and smiles_result.value else ''
        )
        if not smiles:
            raise PropertyResolutionError(
                f"Cannot determine radii of gyration for {identifier!r}: "
                "molecular structure unavailable"
            )
        try:
            self._validated_dipole_molecule(smiles)
        except ValueError as error:
            raise PropertyResolutionError(
                f"Cannot determine radii of gyration for {identifier!r}: {error}"
            ) from error

        identity = f'smiles:{smiles}'
        cache_key = (str(self.CACHE_DIR), identity)
        cache = getattr(self, '_gyration_radii_cache', None)
        if cache is None:
            cache = self._gyration_radii_cache = {}
        cached = cache.get(cache_key)
        if cached is not None:
            return cached, 'shared cached GFN2-xTB optimized geometry'

        geometry_record = self._load_dipole_artifact(identity, 'geometry_xtb')
        had_cached_geometry = isinstance(geometry_record, dict)
        dependencies = self._dipole_dependency_state()
        if (
            not had_cached_geometry
            and not self._backend_is_available('xtb', dependencies)
        ):
            raise PropertyResolutionError(
                f"Cannot determine radii of gyration for {identifier!r}: "
                "GFN2-xTB geometry dependencies are not installed"
            )
        try:
            atoms, _charge, _multiplicity = self._resolve_xtb_geometry(
                identity,
                smiles,
                dependencies,
            )
            radii = self._radii_from_atoms(atoms)
        except Exception as error:
            raise PropertyResolutionError(
                f"Cannot determine radii of gyration for {identifier!r}: "
                f"GFN2-xTB geometry unavailable or failed ({type(error).__name__}: "
                f"{error})"
            ) from error
        if len(cache) > 4096:
            cache.clear()
        cache[cache_key] = radii
        geometry_note = (
            'shared cached GFN2-xTB optimized geometry'
            if had_cached_geometry
            else 'new GFN2-xTB optimized geometry cached before evaluation'
        )
        return radii, geometry_note

    @staticmethod
    def _radii_from_atoms(atoms) -> tuple[float, float, bool]:
        """Return ``(R_g, R_prime, linear)`` from atomic masses and positions."""
        masses = np.asarray(atoms.get_masses(), dtype=float)
        positions = np.asarray(atoms.positions, dtype=float)
        if (
            positions.shape != (len(masses), 3)
            or not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(masses))
            or np.any(masses <= 0.0)
        ):
            raise ValueError('optimized geometry has invalid masses or coordinates')
        total_mass = float(np.sum(masses))
        center = np.sum(masses[:, None] * positions, axis=0) / total_mass
        centered = positions - center
        squared_radii = np.sum(centered * centered, axis=1)
        conventional = math.sqrt(float(np.dot(masses, squared_radii)) / total_mass)

        inertia = np.zeros((3, 3), dtype=float)
        for mass, vector, squared_radius in zip(
            masses,
            centered,
            squared_radii,
            strict=True,
        ):
            inertia += mass * (
                squared_radius * np.identity(3) - np.outer(vector, vector)
            )
        moments = np.maximum(np.linalg.eigvalsh(inertia), 0.0)
        largest = float(moments[-1])
        if largest <= 1.0e-20:
            return conventional, 0.0, True
        linear = float(moments[0]) <= 1.0e-6 * largest
        if linear:
            modified = math.sqrt(
                math.sqrt(float(moments[1] * moments[2])) / total_mass
            )
        else:
            geometric_moment = float(np.prod(moments)) ** (1.0 / 3.0)
            modified = math.sqrt(2.0 * math.pi * geometric_moment / total_mass)
        if not all(math.isfinite(value) and value >= 0.0 for value in (conventional, modified)):
            raise ValueError('calculated gyration radius was not finite and nonnegative')
        return conventional, modified, linear


__all__ = ['RadiusOfGyrationMixin']
