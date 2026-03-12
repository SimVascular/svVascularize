"""
Platform/architecture-aware svZeroDSolver executable selection.

This module intentionally avoids importing heavy dependencies so it can be used
in lightweight CI checks and during wheel build validation.
"""

from __future__ import annotations

import errno
import hashlib
import os
import platform
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


SOLVER_0D_EXE_BASENAME = "svzerodsolver"


def _norm_os_dir(system: Optional[str] = None) -> str:
    sysname = (system or platform.system()).strip()
    if sysname == "Linux":
        return "Linux"
    if sysname == "Windows":
        return "Windows"
    if sysname == "Darwin":
        return "Mac"
    raise RuntimeError(f"Unsupported OS: {sysname}")


def _norm_arch(machine: Optional[str] = None) -> str:
    m = (machine or platform.machine()).strip().lower()
    if m in {"x86_64", "amd64"}:
        return "x86_64"
    if m in {"aarch64", "arm64"}:
        return "aarch64"
    return m or "unknown"


def _solver_0d_exe_filename() -> str:
    if os.name == "nt":
        return SOLVER_0D_EXE_BASENAME + ".exe"
    return SOLVER_0D_EXE_BASENAME


def _ensure_executable(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        mode = path.stat().st_mode
        if mode & stat.S_IXUSR:
            return
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        # Best-effort; actual execution will raise if this fails.
        pass


def _is_exec_format_error(e: BaseException) -> bool:
    if isinstance(e, OSError):
        if getattr(e, "errno", None) == errno.ENOEXEC:
            return True
        # Windows: ERROR_BAD_EXE_FORMAT (193)
        if getattr(e, "winerror", None) == 193:
            return True
    return False


def _uid_suffix() -> str:
    try:
        return str(os.getuid())
    except Exception:
        return "unknown"


def _user_cache_dir() -> Optional[Path]:
    try:
        if os.name == "nt":
            base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
            if not base:
                return None
            return Path(base) / "svv" / "solver_0d"
        base = os.environ.get("XDG_CACHE_HOME")
        if base:
            return Path(base) / "svv" / "solver_0d"
        return Path.home() / ".cache" / "svv" / "solver_0d"
    except Exception:
        return None


def _solver_0d_run_roots() -> Tuple[Path, ...]:
    roots: List[Path] = []
    env_root = os.environ.get("SVV_SOLVER_0D_RUN_DIR")
    if env_root:
        roots.append(Path(env_root).expanduser())
    try:
        roots.append(Path(tempfile.gettempdir()))
    except Exception:
        pass
    cache = _user_cache_dir()
    if cache is not None:
        roots.append(cache)
    # De-duplicate while preserving order
    uniq: List[Path] = []
    seen = set()
    for r in roots:
        key = str(r)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    return tuple(uniq)


def _sha256_short(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()[:12]
    except Exception:
        return None


@dataclass(frozen=True)
class Solver0DSelection:
    os_dir: str
    arch: str
    candidates: Tuple[Path, ...]


def get_solver_0d_candidates(
    *,
    os_dir: Optional[str] = None,
    arch: Optional[str] = None,
) -> Solver0DSelection:
    """
    Return an ordered list of candidate executable paths for svZeroDSolver.

    Search order:
    1) SVV_SOLVER_0D_PATH (explicit executable)
    2) SVV_SOLVER_0D_DIR/<exe>
    3) svv/bin/<exe> (locally built override)
    4) bundled: svv/utils/solvers/0D/<OS>/<arch>/<exe>
    5) bundled fallbacks (mac): universal2 -> arch aliases
    6) legacy fallback: svv/solvers/<exe>
    """
    os_dir = os_dir or _norm_os_dir()
    arch_override = os.environ.get("SVV_SOLVER_0D_ARCH")
    arch = (arch_override or arch or "").strip()
    if not arch:
        arch = _norm_arch()
    else:
        a = arch.lower()
        if a in {"amd64", "x86_64"}:
            arch = "x86_64"
        elif a in {"arm64", "aarch64"}:
            arch = "aarch64"
        elif a in {"universal2"}:
            arch = "universal2"

    exe_name = _solver_0d_exe_filename()
    solvers_dir = Path(__file__).resolve().parent / "0D"

    candidates: List[Path] = []

    solver_path = os.environ.get("SVV_SOLVER_0D_PATH")
    if solver_path:
        candidates.append(Path(solver_path).expanduser())

    solver_dir = os.environ.get("SVV_SOLVER_0D_DIR")
    if solver_dir:
        candidates.append(Path(solver_dir).expanduser() / exe_name)

    try:
        import svv

        svv_root = Path(svv.__file__).resolve().parent
        candidates.append(svv_root / "bin" / exe_name)
    except Exception:
        pass

    os_base = solvers_dir / os_dir
    if os_dir == "Mac":
        candidates.append(os_base / "universal2" / exe_name)
        candidates.append(os_base / "x86_64" / exe_name)
        candidates.append(os_base / "arm64" / exe_name)
        candidates.append(os_base / "aarch64" / exe_name)
    else:
        candidates.append(os_base / arch / exe_name)

    # Legacy fallback used by older zero_d export code paths.
    try:
        import svv

        svv_root = Path(svv.__file__).resolve().parent
        candidates.append(svv_root / "solvers" / exe_name)
    except Exception:
        pass

    uniq: List[Path] = []
    seen = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)

    return Solver0DSelection(os_dir=os_dir, arch=arch, candidates=tuple(uniq))


def get_solver_0d_exe() -> Path:
    sel = get_solver_0d_candidates()
    for p in sel.candidates:
        if p.is_file():
            return p
    tried = "\n  - ".join(str(p) for p in sel.candidates)
    raise FileNotFoundError(
        "svZeroDSolver executable not found "
        f"(os={sel.os_dir}, arch={sel.arch}). Tried:\n  - {tried}"
    )


def run_solver_0d(
    args: Sequence[str],
    *,
    stdout=None,
    stderr=None,
    cwd=None,
) -> Path:
    """
    Execute svZeroDSolver using platform/arch candidates as needed.

    Returns the Path of the executable that successfully ran.
    """
    sel = get_solver_0d_candidates()
    last_exec_error: Optional[BaseException] = None

    for exe in sel.candidates:
        if not exe.is_file():
            continue
        try:
            _ensure_executable(exe)
            subprocess.check_call([str(exe), *map(str, args)], stdout=stdout, stderr=stderr, cwd=cwd)
            return exe
        except subprocess.CalledProcessError:
            # Tool ran but failed; this is not an architecture-selection issue.
            raise
        except PermissionError as e:
            # If we can't exec in-place (missing +x, read-only site-packages, or a noexec mount),
            # copy the executable into a writable run directory and execute from there.
            last_exec_error = e
            digest = _sha256_short(exe)
            for root in _solver_0d_run_roots():
                try:
                    base = root / f"svv-solver-0d-{_uid_suffix()}" / sel.os_dir / sel.arch
                    if digest:
                        base = base / digest
                    base.mkdir(parents=True, exist_ok=True)
                    if os.name != "nt":
                        try:
                            base.chmod(0o700)
                        except Exception:
                            pass
                    cached = base / exe.name
                    try:
                        if (not cached.is_file()) or (cached.stat().st_size != exe.stat().st_size):
                            shutil.copy2(exe, cached)
                    except Exception:
                        # If copy/overwrite fails (permissions, races), try running whatever exists.
                        pass
                    _ensure_executable(cached)
                    subprocess.check_call([str(cached), *map(str, args)], stdout=stdout, stderr=stderr, cwd=cwd)
                    return cached
                except subprocess.CalledProcessError:
                    raise
                except BaseException as copy_exc:
                    if isinstance(copy_exc, (FileNotFoundError, PermissionError)) or _is_exec_format_error(copy_exc):
                        last_exec_error = copy_exc
                        continue
                    # Non-exec-related error: surface it.
                    raise
            continue
        except BaseException as e:
            # Exec-format mismatch: try another candidate.
            if _is_exec_format_error(e):
                last_exec_error = e
                continue
            raise

    tried = "\n  - ".join(str(p) for p in sel.candidates)
    if last_exec_error is not None:
        raise RuntimeError(
            "No compatible svZeroDSolver executable could be executed "
            f"(os={sel.os_dir}, arch={sel.arch}). Tried:\n  - {tried}\n"
            f"Last error: {last_exec_error}"
        ) from last_exec_error
    raise FileNotFoundError(
        "svZeroDSolver executable not found "
        f"(os={sel.os_dir}, arch={sel.arch}). Tried:\n  - {tried}"
    )
