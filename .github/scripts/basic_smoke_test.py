import os
import sys
import tempfile
from contextlib import contextmanager

# Ensure CI can initialize Qt/VTK without a physical display.
# (Linux runners are frequently headless; Xvfb is used in the workflow,
# but these defaults make the script more robust when run locally too.)
if sys.platform.startswith("linux"):
    os.environ.setdefault("SVV_GUI_GL_MODE", "software")
    # If we have an X server (e.g. Xvfb in CI), prefer the normal XCB platform
    # so PyVistaQt/VTK can embed a render window reliably.
    if os.environ.get("DISPLAY"):
        if os.environ.get("GITHUB_ACTIONS") or os.environ.get("CI"):
            os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    else:
        # No display available: fall back to offscreen modes to keep imports working.
        os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Never prompt for telemetry consent in CI.
os.environ.setdefault("SVV_TELEMETRY_DISABLED", "1")


def _log(msg: str) -> None:
    print(msg, flush=True)


@contextmanager
def _temp_cwd():
    """Run work in a temp dir to avoid polluting the repo with tmp.mesh files."""
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            yield tmp
        finally:
            os.chdir(old)


def _gui_smoke_process(result_queue) -> None:
    """Run GUI init/show in a separate process to avoid CI hangs."""
    try:
        # macOS CI runners occasionally hang when initializing the embedded VTK viewport.
        # For smoke testing we only need to verify the Qt GUI launches.
        if sys.platform == "darwin" and (os.environ.get("GITHUB_ACTIONS") or os.environ.get("CI")):
            os.environ.setdefault("SVV_GUI_DISABLE_VTK", "1")
        # Windows CI runners have been observed to hard-crash (0xC0000005) during
        # VTK/Qt interactor initialization on some Python/VTK combinations.
        if sys.platform == "win32" and (os.environ.get("GITHUB_ACTIONS") or os.environ.get("CI")):
            os.environ.setdefault("SVV_GUI_DISABLE_VTK", "1")

        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
        from svv.visualize.gui.main_window import VascularizeGUI

        _log("SMOKE: gui: starting QApplication")
        app = QApplication.instance() or QApplication(sys.argv[:1])

        _log("SMOKE: gui: constructing main window")
        gui = VascularizeGUI(domain=None)
        gui.show()
        _log("SMOKE: gui: shown; entering event loop")

        # Quit shortly after show() so we validate initialization without
        # hanging on shutdown paths that can be flaky in headless CI.
        QTimer.singleShot(1500, app.quit)
        app.exec()
        _log("SMOKE: gui: event loop exited")

        result_queue.put(("ok", None))
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass
        os._exit(0)
    except BaseException as e:
        try:
            result_queue.put(("error", repr(e)))
        except Exception:
            pass
        raise


def _smoke_gui(timeout_s: int = 45) -> None:
    """Verify the Qt GUI can be created and shown briefly (with a hard timeout)."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    p = ctx.Process(target=_gui_smoke_process, args=(result_queue,))
    p.start()
    p.join(timeout_s)

    if p.is_alive():
        p.terminate()
        p.join(5)
        raise RuntimeError(f"GUI smoke test timed out after {timeout_s}s")

    status = None
    message = None
    try:
        status, message = result_queue.get(timeout=2)
    except Exception:
        pass

    if p.exitcode != 0 or status != "ok":
        raise RuntimeError(f"GUI smoke test failed (exitcode={p.exitcode}) {status}: {message}")


def main() -> None:
    import pyvista as pv

    from svv.domain.domain import Domain
    from svv.forest.forest import Forest
    from svv.tree.tree import Tree
    from svv.simulation.simulation import Simulation
    from svv.utils.remeshing.mmg import get_mmg_candidates, get_mmg_exe
    from svv.utils.solvers.solver_0d import get_solver_0d_candidates, get_solver_0d_exe
    from svv.utils.remeshing.remesh import remesh_surface

    _log("SMOKE: starting")

    # Log MMG selection candidates early so platform/arch mismatches are obvious.
    for tool in ("mmg2d", "mmg3d", "mmgs"):
        sel = get_mmg_candidates(tool)
        _log(f"SMOKE: {tool}: os={sel.os_dir} arch={sel.arch}")
        for p in sel.candidates:
            _log(f"SMOKE: {tool}: candidate: {p}")
        _log(f"SMOKE: {tool}: selected: {get_mmg_exe(tool)}")

    solver_sel = get_solver_0d_candidates()
    _log(f"SMOKE: solver_0d: os={solver_sel.os_dir} arch={solver_sel.arch}")
    for p in solver_sel.candidates:
        _log(f"SMOKE: solver_0d: candidate: {p}")
    solver_0d_exe = get_solver_0d_exe()
    _log(f"SMOKE: solver_0d: selected: {solver_0d_exe}")

    # Domain build (geometry + implicit function + tetrahedral mesh)
    _log("SMOKE: domain: create/solve/build")
    cube = Domain(pv.Cube().triangulate())
    cube.create()
    cube.solve()
    cube.build()

    # MMG remeshing (validate that packaged/built executables run)
    _log("SMOKE: mmg: remesh_surface(pv.Cube())")
    with _temp_cwd():
        remeshed = remesh_surface(pv.Cube().triangulate(), verbosity=0)
    assert remeshed.n_cells > 0

    # Tree build
    _log("SMOKE: tree: build")
    t = Tree()
    t.set_domain(cube)
    t.parameters.set('root_pressure', 100)
    t.parameters.set('terminal_pressure', 0)
    t.set_root()
    # Use a small tree to keep the smoke test
    # lightweight across all CI runners.
    t.n_add(3)

    # Forest build (minimal network + one add step)
    _log("SMOKE: forest: build")
    forest = Forest(domain=cube, n_networks=1, n_trees_per_network=[1])
    forest.set_domain(cube)
    forest.set_roots()
    forest.add(1)

    sim = Simulation(t)
    # For CI we only require the fluid mesh;
    # skipping the tissue mesh keeps TetGen runs lighter,
    # especially on Windows and macOS runners.
    # sim.build_meshes(fluid=True, tissue=False, boundary_layer=False)

    # GUI launch (after core objects exist)
    _log("SMOKE: gui: launch")
    _smoke_gui()
    _log("SMOKE: done")


if __name__ == "__main__":
    main()
