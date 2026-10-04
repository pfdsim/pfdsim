"""Reusable handling for optional live-provider integration checks."""

from dataclasses import dataclass
import socket
from typing import Any, Callable
from urllib.error import URLError
import warnings

from property_resolution.common import OnlineAttemptState
from .network_policy import allow_live_providers


@dataclass(frozen=True)
class OptionalLiveProviderResult:
    completed: bool
    value: Any
    state: OnlineAttemptState


def _caused_by_dns_failure(error: BaseException) -> bool:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, socket.gaierror):
            return True
        if isinstance(current, URLError) and isinstance(
            current.reason,
            socket.gaierror,
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def run_optional_live_provider(
    resolver: Any,
    operation: Callable[[], Any],
    *,
    label: str,
) -> OptionalLiveProviderResult:
    """Run a live check, warning only for genuine transient connectivity."""
    try:
        with allow_live_providers('pubchem', 'nist'), resolver._online_attempt_scope(True) as attempt:
            value = operation()
    except Exception as error:
        if not _caused_by_dns_failure(error):
            raise
        warnings.warn(
            f'{label} was not exercised because DNS resolution failed: '
            f'{error}',
            RuntimeWarning,
            stacklevel=2,
        )
        return OptionalLiveProviderResult(
            completed=False,
            value=None,
            state=OnlineAttemptState.TRANSIENT_FAILURE,
        )
    if attempt.state is OnlineAttemptState.TRANSIENT_FAILURE:
        warnings.warn(
            f'{label} was not exercised because a live provider had a '
            'transient connectivity failure',
            RuntimeWarning,
            stacklevel=2,
        )
        return OptionalLiveProviderResult(
            completed=False,
            value=value,
            state=attempt.state,
        )
    return OptionalLiveProviderResult(
        completed=True,
        value=value,
        state=attempt.state,
    )
