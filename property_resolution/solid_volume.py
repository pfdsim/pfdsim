"""Canonical and provided solid-volume resolution.

The bundled CRC source reports molar volume.  Runtime resolution preserves that
native quantity and derives mass or molar density only at the API boundary.
The source has no temperature correlation, so inorganic expansion is exactly
zero and uncertainty is represented by a distance-dependent quality penalty.
"""

from __future__ import annotations

import math
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .common import PropertyResolutionError, PropertyResolutionResult, REFERENCE_TEMPERATURE_K
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..solid_material_forms import normalize_solid_material_form
else:
    from solid_material_forms import normalize_solid_material_form


ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DATABASE_PATH = ROOT / "data" / "solid_volume.sqlite"
CRC_SOLID_VOLUME_QUALITY = 0.94
CRC_QUALITY_PENALTY_PER_25K = 0.01
CRC_MAXIMUM_QUALITY_PENALTY = 0.30


@dataclass(frozen=True)
class SolidVolumeRecord:
    cas: str
    name: str
    molar_volume_m3_per_kmol: float
    material_form: str
    polymorph: str
    quality: float
    source: str
    source_fingerprint: str


_CACHE: dict[tuple[Path, str, str, str], Optional[SolidVolumeRecord]] = {}
_IDENTITY_CACHE: dict[tuple[Path, str], Optional[str]] = {}
_LOCK = threading.Lock()


def _admitted(row: sqlite3.Row, material_form: str, polymorph: str) -> bool:
    requested = normalize_solid_material_form(material_form)
    requested_polymorph = str(polymorph or "").casefold()
    row_form = str(row["material_form"] or "unspecified").casefold()
    row_polymorph = str(row["polymorph"] or "").casefold()
    if requested_polymorph:
        return (
            row_polymorph == requested_polymorph
            and (requested == "unspecified" or row_form in {requested, "unspecified"})
        )
    if requested in {"hydrate", "solvate", "glass", "amorphous"}:
        return row_form == requested
    if requested == "crystalline":
        return row_form in {"crystalline", "unspecified"} and not row_polymorph
    return bool(row["is_default_form"])


def load_bundled_solid_volume(
    cas: str, *, material_form: str = "unspecified", polymorph: str = "",
    path: Path = CANONICAL_DATABASE_PATH,
) -> Optional[SolidVolumeRecord]:
    key = str(cas or "").strip()
    if not key or not path.is_file():
        return None
    normalized_path = path.resolve()
    cache_key = (normalized_path, key, material_form.casefold(), polymorph.casefold())
    with _LOCK:
        if cache_key in _CACHE:
            return _CACHE[cache_key]
    result = None
    try:
        uri = f"file:{normalized_path}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=5.0)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM canonical_solid_volume WHERE cas = ? ORDER BY quality DESC, record_id",
                (key,),
            ).fetchall()
        for row in rows:
            if not _admitted(row, material_form, polymorph):
                continue
            result = SolidVolumeRecord(
                cas=str(row["cas"]), name=str(row["name"] or ""),
                molar_volume_m3_per_kmol=float(row["molar_volume_m3_per_kmol"]),
                material_form=str(row["material_form"]), polymorph=str(row["polymorph"] or ""),
                quality=float(row["quality"]), source=str(row["source_label"]),
                source_fingerprint=str(row["source_fingerprint"]),
            )
            break
    except (OSError, sqlite3.Error, TypeError, ValueError):
        result = None
    with _LOCK:
        _CACHE[cache_key] = result
    return result


def lookup_bundled_solid_volume_cas(identifier: str, *, path: Path = CANONICAL_DATABASE_PATH) -> Optional[str]:
    text = str(identifier or "").strip()
    if not text or not path.is_file():
        return None
    normalized_path = path.resolve()
    cache_key = (normalized_path, text.casefold())
    with _LOCK:
        if cache_key in _IDENTITY_CACHE:
            return _IDENTITY_CACHE[cache_key]
    result = None
    try:
        uri = f"file:{normalized_path}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=5.0)) as connection:
            rows = connection.execute(
                "SELECT DISTINCT cas, name FROM canonical_solid_volume WHERE name = ? COLLATE NOCASE",
                (text,),
            ).fetchall()
        matches = {str(row[0]) for row in rows if str(row[1] or "").casefold() == text.casefold()}
        if len(matches) == 1:
            result = next(iter(matches))
    except (OSError, sqlite3.Error):
        result = None
    with _LOCK:
        _IDENTITY_CACHE[cache_key] = result
    return result


def clear_bundled_solid_volume_cache() -> None:
    with _LOCK:
        _CACHE.clear()
        _IDENTITY_CACHE.clear()


class SolidVolumeMixin:
    """Public solid-volume source order layered over shared density parsing."""

    @staticmethod
    def _solid_material_selection(props: Mapping[str, Any]) -> tuple[str, str]:
        form = normalize_solid_material_form((props or {}).get("solid_material_form"))
        polymorph = str((props or {}).get("solid_polymorph") or "").strip()
        return form or "unspecified", polymorph

    @staticmethod
    def _positive_number(value: Any) -> Optional[float]:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) and result > 0.0 else None

    def _provided_solid_volume(self, props: Mapping[str, Any]) -> Optional[PropertyResolutionResult]:
        value = self._positive_number((props or {}).get("Vm_solid"))
        if value is None:
            return None
        source = self._source_result_for_value(props, "Vm_solid", units="m^3/kmol")
        return PropertyResolutionResult(
            value=value, source=source.source if source else "provided",
            method="provided_solid_molar_volume",
            quality=source.quality if source else 1.0,
            notes="Explicit solid molar volume; units m^3/kmol",
        )

    def _provided_solid_density(self, T: float, props: Mapping[str, Any]) -> Optional[PropertyResolutionResult]:
        correlation = ((props or {}).get("property_correlations") or {}).get("rhos")
        if isinstance(correlation, dict):
            evaluated = self._evaluate_provided_correlation(dict(props), "rhos", T)
            if evaluated is not None:
                value, normalized = evaluated
                if self._positive_number(value) is not None:
                    default_quality = 1.0 if normalized.get("_pfd_override") else 0.96
                    return self._provided_correlation_result(
                        value, normalized, "provided_solid_density_fit",
                        notes_prefix="Solid mass density; units kg/m^3; ",
                        default_quality=default_quality,
                    )
        value = self._positive_number((props or {}).get("rho_solid"))
        if value is None:
            return None
        source = self._source_result_for_value(props, "rho_solid", units="kg/m^3")
        return PropertyResolutionResult(
            value=value, source=source.source if source else "provided",
            method="provided_solid_mass_density",
            quality=source.quality if source else 1.0,
            notes="Explicit constant solid mass density; units kg/m^3",
        )

    def _bundled_solid_volume(self, symbol: str, T: float, props: Mapping[str, Any]) -> Optional[PropertyResolutionResult]:
        form, polymorph = self._solid_material_selection(props)
        candidates = []
        for value in ((props or {}).get("CAS"), (props or {}).get("cas"), symbol):
            text = str(value or "").strip()
            if text and text not in candidates:
                candidates.append(text)
        record = None
        for identity in candidates:
            record = load_bundled_solid_volume(identity, material_form=form, polymorph=polymorph)
            if record is None:
                resolved = lookup_bundled_solid_volume_cas(identity)
                if resolved:
                    record = load_bundled_solid_volume(resolved, material_form=form, polymorph=polymorph)
            if record is not None:
                break
        if record is None:
            return None
        steps = int(math.floor(abs(float(T) - REFERENCE_TEMPERATURE_K) / 25.0 + 1.0e-12))
        penalty = min(CRC_MAXIMUM_QUALITY_PENALTY, CRC_QUALITY_PENALTY_PER_25K * steps)
        return PropertyResolutionResult(
            value=record.molar_volume_m3_per_kmol,
            source="local", method="crc_solid_constant_molar_volume",
            quality=max(0.0, record.quality - penalty),
            notes=(
                f"{record.source}; native constant Vm_s={record.molar_volume_m3_per_kmol:g} "
                f"m^3/kmol; source temperature unspecified/ambient; zero inorganic "
                f"thermal expansion assumed; requested T={float(T):g} K; "
                f"quality penalty={penalty:g} over {steps} complete 25 K interval(s); "
                f"material form={record.material_form}"
            ),
        )

    def _validate_provided_solid_pair(
        self,
        props: Mapping[str, Any],
        volume: Optional[PropertyResolutionResult],
        density: Optional[PropertyResolutionResult],
    ) -> None:
        if volume is None or density is None:
            return
        mw = self._positive_number((props or {}).get("MW"))
        if mw is None:
            return
        implied = mw / float(volume.value)
        relative = abs(implied / float(density.value) - 1.0)
        if relative > 0.02:
            raise PropertyResolutionError(
                "Provided solid molar volume and mass density are inconsistent: "
                f"MW/Vm_s={implied:g} kg/m^3 versus rho_s={float(density.value):g} "
                f"kg/m^3 ({relative * 100.0:.2f}% difference)"
            )

    def resolve_solid_molar_volume(
        self, symbol: str, T: float, props: dict[str, Any] = None, *, allow_online: bool = True,
    ) -> PropertyResolutionResult:
        T = float(T)
        if not math.isfinite(T) or T <= 0.0:
            raise ValueError("Solid-volume temperature must be positive and finite")
        props = self._coerce_props(symbol, props, allow_online=allow_online)
        direct = self._provided_solid_volume(props)
        provided_density = self._provided_solid_density(T, props)
        self._validate_provided_solid_pair(props, direct, provided_density)
        if direct is not None:
            return direct
        density = provided_density
        if density is not None:
            return self._molar_volume_from_solid_density(props, density)
        bundled = self._bundled_solid_volume(symbol, T, props)
        if bundled is not None:
            return bundled
        density = self._resolve_observed_or_estimated_solid_mass_density(
            symbol, T, props, allow_online=allow_online,
        )
        return self._molar_volume_from_solid_density(props, density)

    def _molar_volume_from_solid_density(
        self, props: Mapping[str, Any], density: PropertyResolutionResult,
    ) -> PropertyResolutionResult:
        mw = self._positive_number((props or {}).get("MW"))
        if mw is None:
            raise PropertyResolutionError("Cannot convert solid density without molecular weight")
        value = mw / float(density.value)
        mw_result = self._source_result_for_value(dict(props), "MW", units="g/mol")
        return PropertyResolutionResult(
            value=value,
            source=self._derived_source([density, mw_result], exact_formula=True),
            method="solid_molar_volume_from_mass_density",
            quality=self._combine_quality([density, mw_result], exact_formula=True),
            notes=f"MW/rho_s; units m^3/kmol; {density.notes}",
        )

    def resolve_solid_mass_density(
        self, symbol: str, T: float, props: dict[str, Any] = None, *, allow_online: bool = True,
    ) -> PropertyResolutionResult:
        T = float(T)
        if not math.isfinite(T) or T <= 0.0:
            raise ValueError("Solid-density temperature must be positive and finite")
        props = self._coerce_props(symbol, props, allow_online=allow_online)
        direct = self._provided_solid_density(T, props)
        provided_volume = self._provided_solid_volume(props)
        self._validate_provided_solid_pair(props, provided_volume, direct)
        if direct is not None:
            return direct
        volume = provided_volume or self._bundled_solid_volume(symbol, T, props)
        if volume is not None:
            mw = self._positive_number(props.get("MW"))
            if mw is None:
                raise PropertyResolutionError(
                    f"Cannot convert solid molar volume for '{symbol}' without molecular weight"
                )
            mw_result = self._source_result_for_value(props, "MW", units="g/mol")
            return PropertyResolutionResult(
                value=mw / float(volume.value),
                source=self._derived_source([volume, mw_result], exact_formula=True),
                method="solid_mass_density_from_molar_volume",
                quality=self._combine_quality([volume, mw_result], exact_formula=True),
                notes=f"MW/Vm_s; units kg/m^3; {volume.notes}",
            )
        return self._resolve_observed_or_estimated_solid_mass_density(
            symbol, T, props, allow_online=allow_online,
        )

    def resolve_solid_molar_density(
        self, symbol: str, T: float, props: dict[str, Any] = None, *, allow_online: bool = True,
    ) -> PropertyResolutionResult:
        volume = self.resolve_solid_molar_volume(symbol, T, props, allow_online=allow_online)
        return PropertyResolutionResult(
            value=1.0 / float(volume.value), source=volume.source,
            method="solid_molar_density_from_molar_volume", quality=volume.quality,
            notes=f"Inverse solid molar volume; units mol/dm^3; {volume.notes}",
        )
