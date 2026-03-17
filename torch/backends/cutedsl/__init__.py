# mypy: allow-untyped-defs
import sys
from contextlib import contextmanager

from packaging.version import Version

from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule


def is_available() -> bool:
    r"""Return a bool indicating if the CuTeDSL runtime is currently available."""
    from torch._native import cutedsl_utils

    return cutedsl_utils.runtime_available()


def version() -> Version | None:
    r"""Return the installed CuTeDSL runtime version, or None if unavailable."""
    from torch._native import cutedsl_utils

    return cutedsl_utils.runtime_version()


def _set_enabled(_enabled: bool) -> None:
    global enabled
    enabled = _enabled


def _get_enabled() -> bool:
    return enabled


def set_flags(_enabled=None):
    orig_flags = (enabled,)
    if _enabled is not None:
        _set_enabled(_enabled)
    return orig_flags


@contextmanager
def flags(enabled=None):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(enabled)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


class CuTeDSLModule(PropModule):
    global enabled
    enabled = ContextProp(_get_enabled, _set_enabled)


sys.modules[__name__] = CuTeDSLModule(sys.modules[__name__], __name__)

enabled = True
