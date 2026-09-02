"""Public package facade and optional accelerator aliases for svVascularize."""

import sys as _sys
from importlib import import_module as _import_module


__version__ = "0.0.52"

__all__ = [
    "__version__",
    "domain",
    "forest",
    "simulation",
    "tree",
    "utils",
    "visualize",
    "Domain",
    "Patch",
    "Forest",
    "Simulation",
    "Tree",
    "TreeParameters",
    "UnitSystem",
]


# If the optional companion package with compiled accelerators is installed,
# prefer those modules at the existing svv.* import paths.
_ACCELERATOR_ALIASES = {
    "svv.domain.routines.c_allocate": "svv_accel.domain.routines.c_allocate",
    "svv.domain.routines.c_sample": "svv_accel.domain.routines.c_sample",
    "svv.utils.spatial.c_distance": "svv_accel.utils.spatial.c_distance",
    "svv.tree.utils.c_angle": "svv_accel.tree.utils.c_angle",
    "svv.tree.utils.c_basis": "svv_accel.tree.utils.c_basis",
    "svv.tree.utils.c_close": "svv_accel.tree.utils.c_close",
    "svv.tree.utils.c_local_optimize": "svv_accel.tree.utils.c_local_optimize",
    "svv.tree.utils.c_obb": "svv_accel.tree.utils.c_obb",
    "svv.tree.utils.c_update": "svv_accel.tree.utils.c_update",
    "svv.tree.utils.c_extend": "svv_accel.tree.utils.c_extend",
    "svv.simulation.utils.close_segments": "svv_accel.simulation.utils.close_segments",
    "svv.simulation.utils.extract": "svv_accel.simulation.utils.extract",
}


def _install_accelerator_aliases():
    try:
        _import_module("svv_accel")
    except Exception:
        return

    for target, source in _ACCELERATOR_ALIASES.items():
        try:
            _sys.modules[target] = _import_module(source)
        except Exception:
            # A missing accelerator keeps the existing pure-Python fallback.
            pass


_install_accelerator_aliases()
del _install_accelerator_aliases


_LAZY_EXPORTS = {
    "domain": ("svv.domain", None),
    "forest": ("svv.forest", None),
    "simulation": ("svv.simulation", None),
    "tree": ("svv.tree", None),
    "utils": ("svv.utils", None),
    "visualize": ("svv.visualize", None),
    "Domain": ("svv.domain.domain", "Domain"),
    "Patch": ("svv.domain.patch", "Patch"),
    "Forest": ("svv.forest.forest", "Forest"),
    "Simulation": ("svv.simulation.simulation", "Simulation"),
    "Tree": ("svv.tree.tree", "Tree"),
    "TreeParameters": ("svv.tree.data.data", "TreeParameters"),
    "UnitSystem": ("svv.tree.data.units", "UnitSystem"),
}


def __getattr__(name):
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(
            "module {!r} has no attribute {!r}".format(__name__, name)
        ) from None

    module = _import_module(module_name)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()).union(__all__))
