"""Shared paths for persistent runtime property-resolution caches."""

from pathlib import Path


SATURATION_PROPERTIES_CACHE_PATH = (
    Path(__file__).resolve().parent.parent
    / 'data'
    / 'runtime'
    / 'saturation_properties_cache.sqlite'
)
