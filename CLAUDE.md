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

## Platform Notes

The codebase has extensive platform detection (Linux/macOS/Windows) particularly around:
- MMG binary selection (`svv/utils/remeshing/mmg.py`)
- GUI initialization (headless/offscreen modes for CI)
- Build configuration in `setup.py` (Visual Studio detection on Windows, CPU feature flags)
