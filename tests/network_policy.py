"""Offline transport by default, with explicit scopes for live provider tests."""

from contextlib import contextmanager
from contextvars import ContextVar
import sys
import urllib.error
import urllib.parse
import urllib.request
from unittest.mock import patch

# Test modules support both repository and installed-package imports. They
# must share one transport policy rather than nest independent allowlists.
for _name in ('tests.network_policy', 'pfdsim.tests.network_policy'):
    sys.modules.setdefault(_name, sys.modules[__name__])


_PROVIDER_HOSTS = {
    'pubchem': {'pubchem.ncbi.nlm.nih.gov'},
    'nist': {'webbook.nist.gov'},
}
_allowed_hosts = ContextVar('pfdsim_live_provider_hosts', default=frozenset())
_urlopen = urllib.request.urlopen


@contextmanager
def allow_live_providers(*providers):
    """Authorize only the named providers for the enclosed integration check."""
    hosts = frozenset().union(*(_PROVIDER_HOSTS[name] for name in providers))
    token = _allowed_hosts.set(_allowed_hosts.get() | hosts)
    try:
        yield
    finally:
        _allowed_hosts.reset(token)


def _offline_urlopen(url, *args, **kwargs):
    address = url.full_url if isinstance(url, urllib.request.Request) else url
    host = urllib.parse.urlsplit(address).hostname
    if host in {'localhost', '127.0.0.1', '::1'} or host in _allowed_hosts.get():
        return _urlopen(url, *args, **kwargs)
    # A disabled provider is unavailable, not a fabricated HTTP 404. Production
    # fallback handling remains exercised without caching false "no data" rows.
    raise urllib.error.URLError(
        f'Network access to {host!r} is disabled in ordinary tests; '
        'mock the response or use allow_live_providers in a live provider check'
    )


@contextmanager
def offline_provider_transport():
    """Keep response mocks usable while preventing accidental real requests."""
    with patch.object(urllib.request, 'urlopen', _offline_urlopen):
        yield
