"""Public domain API."""

from importlib import import_module as _import_module


__all__ = ["Domain", "Patch"]

_LAZY_EXPORTS = {
    "Domain": ("svv.domain.domain", "Domain"),
    "Patch": ("svv.domain.patch", "Patch"),
}


def __getattr__(name):
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(
            "module {!r} has no attribute {!r}".format(__name__, name)
        ) from None

    module = _import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()).union(__all__))
