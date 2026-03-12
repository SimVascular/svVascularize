import os
from pathlib import Path

from svv.utils.solvers.solver_0d import get_solver_0d_candidates, get_solver_0d_exe


def _exe_name() -> str:
    return "svzerodsolver.exe" if os.name == "nt" else "svzerodsolver"


def test_solver_0d_candidates_include_platform_arch_layout():
    sel = get_solver_0d_candidates(os_dir="Linux", arch="x86_64")
    expected_suffix = Path("svv") / "utils" / "solvers" / "0D" / "Linux" / "x86_64" / _exe_name()
    assert any(str(p).endswith(str(expected_suffix)) for p in sel.candidates)


def test_solver_0d_path_env_override_first(monkeypatch, tmp_path):
    explicit = tmp_path / _exe_name()
    explicit.write_bytes(b"#!/bin/sh\n")

    monkeypatch.setenv("SVV_SOLVER_0D_PATH", str(explicit))
    monkeypatch.delenv("SVV_SOLVER_0D_DIR", raising=False)

    sel = get_solver_0d_candidates(os_dir="Linux", arch="x86_64")
    assert sel.candidates[0] == explicit


def test_get_solver_0d_exe_from_env_dir(monkeypatch, tmp_path):
    exe = tmp_path / _exe_name()
    exe.write_bytes(b"#!/bin/sh\n")

    monkeypatch.delenv("SVV_SOLVER_0D_PATH", raising=False)
    monkeypatch.setenv("SVV_SOLVER_0D_DIR", str(tmp_path))

    resolved = get_solver_0d_exe()
    assert resolved == exe
