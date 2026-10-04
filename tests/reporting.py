"""Portable diagnostic labels for pytest-xdist's subtest reports."""

from enum import Enum


def serializable_subtest_context(value):
    """Convert report labels, preserving the objects used by test assertions."""
    if type(value) in (type(None), bool, int, float, str, bytes):
        return value
    if isinstance(value, dict):
        return {serializable_subtest_context(key): serializable_subtest_context(item)
                for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(serializable_subtest_context(item) for item in value)
    if isinstance(value, list):
        return [serializable_subtest_context(item) for item in value]
    if isinstance(value, Enum):
        return f'{type(value).__name__}.{value.name}'
    if isinstance(value, type):
        return f'{value.__module__}.{value.__qualname__}'
    return repr(value)
