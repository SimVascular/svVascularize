run_0d_script = """import os
import subprocess
import sys

PINNED_SOLVER = {solver_exe!r}
INPUT_FILENAME = {input_filename!r}

def _resolve_solver():
    if PINNED_SOLVER and os.path.isfile(PINNED_SOLVER):
        return PINNED_SOLVER
    try:
        from svv.utils.solvers.solver_0d import get_solver_0d_exe

        solver = str(get_solver_0d_exe())
        if os.path.isfile(solver):
            return solver
    except Exception:
        pass
    return None

def _print_setup_steps():
    print("No svZeroDSolver executable is available.", file=sys.stderr)
    print("Expected a packaged or locally built 0D solver from svVascularize.", file=sys.stderr)
    print("Setup options:", file=sys.stderr)
    print("  1) Build 0D solver binary:", file=sys.stderr)
    print("     python setup.py build_ext --build-solver-0d", file=sys.stderr)
    print("  2) Build all native binaries (MMG + 0D):", file=sys.stderr)
    print("     python setup.py build_ext --build-native-binaries", file=sys.stderr)
    print("  3) Use explicit executable path:", file=sys.stderr)
    print("     export SVV_SOLVER_0D_PATH=/abs/path/to/svzerodsolver", file=sys.stderr)
    print("  4) Re-export with path_to_0d_solver='/abs/path/to/svzerodsolver'", file=sys.stderr)

def main():
    run_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(run_dir, INPUT_FILENAME)
    if not os.path.isfile(input_file):
        print("Input file not found: " + input_file, file=sys.stderr)
        sys.exit(2)

    solver = _resolve_solver()
    if solver is None:
        _print_setup_steps()
        sys.exit(2)

    print("Running svZeroDSolver: " + solver)
    proc = subprocess.run([solver, input_file], cwd=run_dir, check=False)
    sys.exit(proc.returncode)

if __name__ == "__main__":
    main()
"""
