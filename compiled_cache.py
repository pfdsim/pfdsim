"""Shared cache identities for optional Numba kernels.

Numba derives disk-cache filenames from a source basename and function
qualname, but not the function's module.  This project supports importing the
same source tree both as top-level modules and as the ``pfdsim`` package, so a
plain ``njit(cache=True)`` can make those two module identities share an
incompatible serialized cache entry.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Callable, Iterable


def _dependency_fingerprint(dependencies: Iterable[object]) -> str:
    """Return a stable content fingerprint for imported kernel dependencies."""
    digest = hashlib.sha256()
    for dependency in dependencies:
        function = getattr(dependency, "py_func", dependency)
        try:
            filename = inspect.getsourcefile(function) or inspect.getfile(function)
            path = Path(filename)
            payload = path.read_bytes()
        except (OSError, TypeError):
            module = getattr(function, "__module__", "")
            qualname = getattr(function, "__qualname__", repr(function))
            payload = f"{module}:{qualname}".encode()
            path = Path("unavailable")
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def numba_cached(njit, *, dependencies: Iterable[object] = ()) -> Callable:
    """Build a ``njit(cache=True)`` decorator with an unambiguous cache name.

    The owning module is included in the qualname to distinguish package and
    top-level imports. Imported compiled functions may be supplied as
    dependencies; their source contents are then included so callers cannot
    restore a cache compiled against an older dependency implementation.
    Numba's own source stamp continues to invalidate changes in the owning
    module, avoiding redundant dependency on its entire source contents here.
    """
    dependency_tuple = tuple(dependencies)
    fingerprint = (
        _dependency_fingerprint(dependency_tuple) if dependency_tuple else None
    )

    def decorate(function):
        identity = function.__module__
        if fingerprint is not None:
            identity = f"{identity}.{fingerprint}"
        function.__qualname__ = f"{identity}.{function.__qualname__}"
        return njit(cache=True)(function)

    return decorate
