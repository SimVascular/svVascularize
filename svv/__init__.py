__version__ = "0.0.45"


# If the optional companion package with compiled accelerators is installed
# (svv-accelerated), prefer those modules transparently by aliasing them into
# svv.* import paths. This lets users simply `pip install svv[accel]` to get
# faster implementations without changing imports in their code.
try:
    import importlib, sys
    # Try importing the top-level companion namespace first.
    importlib.import_module('svv_accel')
    _ACCEL = {
        'svv.domain.routines.c_allocate': 'svv_accel.domain.routines.c_allocate',
        'svv.domain.routines.c_sample': 'svv_accel.domain.routines.c_sample',
        'svv.utils.spatial.c_distance': 'svv_accel.utils.spatial.c_distance',
        'svv.tree.utils.c_angle': 'svv_accel.tree.utils.c_angle',
        'svv.tree.utils.c_basis': 'svv_accel.tree.utils.c_basis',
        'svv.tree.utils.c_close': 'svv_accel.tree.utils.c_close',
        'svv.tree.utils.c_local_optimize': 'svv_accel.tree.utils.c_local_optimize',
        'svv.tree.utils.c_obb': 'svv_accel.tree.utils.c_obb',
        'svv.tree.utils.c_update': 'svv_accel.tree.utils.c_update',
        'svv.tree.utils.c_extend': 'svv_accel.tree.utils.c_extend',
        'svv.simulation.utils.close_segments': 'svv_accel.simulation.utils.close_segments',
        'svv.simulation.utils.extract': 'svv_accel.simulation.utils.extract',
    }
    for target, source in _ACCEL.items():
        try:
            mod = importlib.import_module(source)
            sys.modules[target] = mod
        except Exception:
            # If a specific accelerator module is missing, leave fallback in place
            pass
except Exception:
    # Companion package not installed; keep pure-Python fallbacks/compiled svv.* if present
    pass
