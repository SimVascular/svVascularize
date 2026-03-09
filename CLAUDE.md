# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

svVascularize (`svv`) is an open-source Python API for automated vascular generation and multi-fidelity hemodynamic simulation. It generates synthetic vasculature for biomanufacturing and CFD analysis. Published on PyPI as `svv`, supports Python 3.9–3.13.

## Build & Install

```bash
pip install -e .                    # Editable install (dev)
pip install svv                     # From PyPI
pip install svv[accel]              # With pre-compiled Cython accelerators
```

Build requires Cython >= 3.0.7 and numpy (see `pyproject.toml`). The setup.py handles:
- Cython extension compilation (12 `.pyx` files)
- MMG binary packaging (platform-specific mesh tools in `svv/bin/`)
- Platform-specific wheel tagging

Environment flags:
- `SVV_ACCEL_COMPANION=1` — build the `svv-accelerated` companion package
- `SVV_BUILD_MMG=1` — compile MMG from source instead of using packaged binaries

## Running Tests

```bash
pytest test/                        # Run all unit tests
pytest test/test_allocate.py        # Run a single test file
pytest test/test_beziercurve.py -k "test_name"  # Run specific test
```

The integration smoke test (used in CI) exercises the full pipeline:
```bash
python .github/scripts/basic_smoke_test.py
```

CI runs the smoke test across Python 3.9–3.13 on Linux, macOS, and Windows (`.github/workflows/basic-smoke-test.yml`).

## Architecture

Four core modules under `svv/`, each with a primary class following a builder pattern:

### Domain (`svv/domain/`)
Geometry and implicit function representation for defining vascular regions.
- `Domain` class: `create()` → `solve()` → `build()` pipeline
- Submodules: `core/` (variational matrices), `io/` (.dmn format), `kernel/`, `routines/` (with Cython: `c_allocate`, `c_sample`), `solver/`

### Tree (`svv/tree/`)
Single branching vascular tree structures.
- `Tree` class: `set_domain()` → `set_root()` → `n_add(count)` pipeline
- Submodules: `branch/` (bifurcation), `collision/`, `data/` (TreeData/TreeParameters/TreeMap), `export/` (centerlines/solids), `utils/` (Cython: `c_angle`, `c_basis`, `c_close`, `c_local_optimize`, `c_obb`, `c_update`, `c_extend`)

### Forest (`svv/forest/`)
Collections of interpenetrating vascular networks.
- `Forest` class: `set_domain()` → `set_roots()` → `add(count)` pipeline
- Submodules: `connect/` (geodesic pathfinding), `export/`

### Simulation (`svv/simulation/`)
Physics simulation setup for CFD/hemodynamic analysis.
- `Simulation` class: initialized with a Tree, then `build_meshes()` for fluid/tissue meshes
- Submodules: `fluid/` (0D/1D/3D solvers, Navier-Stokes), `utils/` (mesh generation, boundary layers, Cython: `close_segments`, `extract`)

### Supporting Modules
- `svv/utils/` — spatial utilities, MMG-based remeshing (`svv/utils/remeshing/`), sampling
- `svv/visualize/` — PyVista-based 3D visualization + PyQt6/PySide6 GUI (`svv/visualize/gui/main_window.py`)
- `svv/telemetry.py` — optional Sentry error reporting (disable with `SVV_TELEMETRY_DISABLED=1`)

## Cython Acceleration

Performance-critical code has Cython implementations (`.pyx` files) alongside pure-Python fallbacks. The optional `svv-accelerated` companion package provides pre-built binaries. The aliasing in `svv/__init__.py` transparently redirects imports when the accelerator package is installed.

## Key External Dependencies

- **MMG** (v5.8.0): Mesh remeshing tool. Pre-built binaries are packaged per-platform in `svv/bin/{os}/{arch}/`. The `svv/utils/remeshing/mmg.py` module handles binary selection.
- **svZeroDSolver**: 0D hemodynamic solver, downloaded during build if needed.
- **PyVista/VTK**: 3D visualization and mesh operations.
- **PySide6**: Qt GUI framework.
- **TetGen**: Tetrahedral mesh generation.

## Performance Optimizations

### Forest Connection Pipeline (`svv/forest/connect/`)
The forest connection pipeline (`forest.connect()`) was the primary bottleneck. Optimizations:
- **Geodesic caching** (`geodesic.py`): Dijkstra result cache per source node + vectorized edge extraction from tetrahedra
- **Assignment deduplication** (`assign.py`): Deduplicated (i,j) terminal pairs before geodesic computation, removed deepcopy
- **Constraint sharing** (`base_connection.py`): All 4 constraint functions share a cached curve evaluation per optimizer iteration (was rebuilding 4×). Reduced sampling: curvature 100→50, boundary 100→40
- **Segment distance vectorization** (`c_distance.py`): Full numpy vectorized pairwise computation, scalar fallback only for degenerate/parallel cases
- **Bezier vectorization** (`bezier.py`): Vectorized De Casteljau evaluates all t values simultaneously
- **CatmullRom vectorization** (`catmullrom.py`): Batched evaluation per segment using `np.outer`
- **deepcopy removal**: Replaced `deepcopy` with `list()`, `.copy()`, or fresh allocation across `bifurcation.py`, `vessel_connection.py`, `tree.py`, `assign.py`

### Import/Meshing Pipeline
- **Skip contour() for imported meshes** (`domain.py:get_boundary`): When `original_boundary` exists (STL/VTP import), skip expensive contour() call
- **In-process TetGen** (`tetrahedralize.py`): Call TetGen directly instead of spawning a subprocess. Falls back to subprocess on failure
- **ConvexHull instead of delaunay_3d** (`domain.py:get_interior`): `scipy.spatial.ConvexHull` replaces expensive `pv.delaunay_3d()`
- **Single cKDTree** (`domain.py:get_interior`, `dmn.py`): Reuse `mesh_tree` cKDTree for radius queries instead of building a separate BallTree
- **Vectorized evaluate_fast** (`domain.py`): Batch index filling by unique lengths instead of Python for loop
- **Cached normalize_scale** (`domain.py`): Pre-computed in `build()`, cached as `_normalize_scale`

## macOS Rendering Fixes

The VTK/Qt visualization had critical issues on macOS Apple Silicon + conda:
- **Offscreen rendering** (`vtk_widget.py`): Uses offscreen VTK rendering with blit-to-QLabel approach, bypassing `QtInteractor` deadlock that caused GUI hangs
- **Ray-cast picking** (`vtk_widget.py`): Implements software ray-cast picking for offscreen mode since VTK hardware picker segfaults without an on-screen OpenGL context
- **Export toolbar** (`main_window.py`): Connected Export button to popup menu with all export options
- **Forest centerline export** (`forest.py`): Added `Forest.export_centerlines()` for centerline export on Forest objects
- **Centerline metadata** (`export_centerlines.py`): Added `BoundaryType` array, `boundary_points`, and inlet/outlet companion file to splines export

## Platform Notes

The codebase has extensive platform detection (Linux/macOS/Windows) particularly around:
- MMG binary selection (`svv/utils/remeshing/mmg.py`)
- GUI initialization (headless/offscreen modes for CI)
- Build configuration in `setup.py` (Visual Studio detection on Windows, CPU feature flags)
