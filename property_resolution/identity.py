from .common import *


class ResolverIdentityMixin:
        @classmethod
        def _source_quality_for_key(
            cls,
            props: Dict[str, Any],
            key: str,
            default_quality: float,
        ) -> float:
            source = ((props or {}).get('property_sources') or {}).get(key)
            return cls._meta_quality(source, default_quality)


        @classmethod
        def _source_result_for_antoine(
            cls,
            props: Dict[str, Any],
            antoine: AntoineCoefficients,
            default_source: str,
            default_quality: float,
        ) -> PropertyResolutionResult:
            source = ((props or {}).get('property_sources') or {}).get('Antoine') or {}
            return PropertyResolutionResult(
                value=None,
                source=source.get('source') or default_source,
                method=source.get('method') or 'antoine',
                quality=cls._meta_quality(source, default_quality),
                notes=source.get('notes') or f"Antoine from {antoine.source}",
            )


        def _identifier_candidates(self, identifier: str, props: Optional[Dict[str, Any]] = None) -> list[str]:
            """Return candidate identifiers from the centralized identity resolver."""
            props = props or {}
            has_specific_identity = any(props.get(key) for key in ("name", "CAS", "cas"))
            extra = []
            for key in ("symbol", "name", "CAS", "cas"):
                value = props.get(key)
                if value:
                    if (
                        has_specific_identity
                        and self._looks_like_molecular_formula(str(value))
                        and not self._formula_candidate_matches_identity_cached(str(identifier), str(value), props)
                    ):
                        continue
                    extra.append(str(value))
            formula = props.get("formula")
            if (
                formula
                and self._formula_candidate_matches_identity_cached(str(identifier), str(formula), props)
            ):
                extra.append(str(formula))
            extra.extend(str(value) for value in props.get("identifiers", []) if value)
            cache_key = (str(identifier), tuple(extra))
            cached = self._identifier_candidates_cache.get(cache_key)
            if cached is not None:
                return list(cached)

            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..compound_identity import get_compound_identity_resolver
                else:
                    from compound_identity import get_compound_identity_resolver
                lookup_identifier = str(identifier)
                if (
                    has_specific_identity
                    and self._looks_like_molecular_formula(lookup_identifier)
                    and not self._formula_candidate_matches_identity_cached(lookup_identifier, lookup_identifier, props)
                ):
                    lookup_identifier = str(props.get("CAS") or props.get("cas") or props.get("name") or identifier)
                result = get_compound_identity_resolver().candidate_identifiers(
                    lookup_identifier,
                    extra,
                    allow_formula=False,
                )
            except Exception:
                result = []
                fallback_identifier = str(identifier)
                if (
                    has_specific_identity
                    and self._looks_like_molecular_formula(fallback_identifier)
                    and not self._formula_candidate_matches_identity_cached(
                        fallback_identifier,
                        fallback_identifier,
                        props,
                    )
                ):
                    fallback_identifier = str(props.get("CAS") or props.get("cas") or props.get("name") or identifier)
                for value in [fallback_identifier, *extra]:
                    if value and value not in result:
                        result.append(value)
            self._identifier_candidates_cache[cache_key] = tuple(result)
            return list(result)


        @staticmethod
        def _looks_like_molecular_formula(identifier: str) -> bool:
            text = str(identifier).strip()
            return bool(
                re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", text)
                and any(ch.isdigit() for ch in text)
            )


        def _formula_candidate_matches_identity_cached(
            self,
            identifier: str,
            formula: str,
            props: Dict[str, Any],
        ) -> bool:
            key = (
                str(identifier),
                str(formula),
                str(props.get("symbol") or ""),
                str(props.get("name") or ""),
                str(props.get("CAS") or ""),
                str(props.get("cas") or ""),
            )
            cached = self._formula_identity_match_cache.get(key)
            if cached is not None:
                return cached
            if any(props.get(field) for field in ("name", "CAS", "cas")):
                result = self._formula_candidate_matches_specific_identity(identifier, formula, props)
            else:
                result = self._formula_candidate_matches_identity(identifier, formula, props)
            self._formula_identity_match_cache[key] = result
            return result


        @staticmethod
        def _formula_candidate_matches_specific_identity(
            identifier: str,
            formula: str,
            props: Dict[str, Any],
        ) -> bool:
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..compound_identity import get_compound_identity_resolver
                else:
                    from compound_identity import get_compound_identity_resolver
                identity_resolver = get_compound_identity_resolver()
                if identity_resolver.is_ambiguous_formula(formula):
                    return False
                formula_cas = identity_resolver.resolve_cas(formula, allow_formula=False)
                if not formula_cas:
                    return False
                prop_cas = props.get("CAS") or props.get("cas")
                if not prop_cas:
                    for value in (props.get("name"), identifier):
                        if value:
                            prop_cas = identity_resolver.resolve_cas(str(value), allow_formula=False)
                            if prop_cas:
                                break
                return bool(prop_cas and formula_cas == prop_cas)
            except Exception:
                return False


        @staticmethod
        def _formula_candidate_matches_identity(identifier: str, formula: str, props: Dict[str, Any]) -> bool:
            """Return True when a formula candidate is safe for local table lookup."""
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..compound_identity import get_compound_identity_resolver
                else:
                    from compound_identity import get_compound_identity_resolver
                identity_resolver = get_compound_identity_resolver()
                formula_symbol = identity_resolver.resolve_symbol(formula, allow_formula=True)
                if not formula_symbol:
                    return False

                accepted = {formula_symbol}
                for value in (identifier, props.get("symbol"), props.get("name"), props.get("CAS"), props.get("cas")):
                    if not value:
                        continue
                    resolved = identity_resolver.resolve_symbol(str(value), allow_formula=False)
                    if resolved in accepted:
                        return True
                    if str(value) == formula_symbol:
                        return True
                return False
            except Exception:
                return False
