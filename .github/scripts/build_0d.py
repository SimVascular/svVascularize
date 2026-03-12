import argparse
import inspect
import os
import platform
import shutil
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve


SOLVER_0D_VERSION_DEFAULT = "3.0"


def _norm_os(os_name: str) -> str:
    if os_name.lower() in {"linux"}:
        return "Linux"
    if os_name.lower() in {"darwin", "mac", "macos"}:
        return "Mac"
    if os_name.lower() in {"windows", "win"}:
        return "Windows"
    raise ValueError(f"Unsupported OS: {os_name}")


def _norm_arch(arch: str) -> str:
    a = arch.strip().lower()
    if a in {"x86_64", "amd64"}:
        return "x86_64"
    if a in {"aarch64", "arm64"}:
        return "aarch64"
    if a in {"universal2"}:
        return "universal2"
    raise ValueError(f"Unsupported arch: {arch}")


def _tar_safe_extract(t: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in t.getmembers():
        member_path = (dest / member.name).resolve()
        if not str(member_path).startswith(str(dest) + os.sep):
            raise RuntimeError(f"Unsafe tar path: {member.name}")
    extractall_sig = inspect.signature(t.extractall)
    if "filter" in extractall_sig.parameters:
        t.extractall(dest, filter="data")
    else:
        t.extractall(dest)


def _chmod_x(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        pass


def _run(cmd: list[str], *, cwd: Optional[Path] = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def _find_built_solver_0d_exes(prefix: Path) -> list[Path]:
    found = []
    for p in prefix.rglob("*"):
        if not p.is_file():
            continue
        if os.name == "nt":
            if p.suffix.lower() != ".exe":
                continue
            if p.stem.lower() == "svzerodsolver":
                found.append(p)
            continue

        if p.name == "svzerodsolver":
            found.append(p)
    return found


def _dest_exe_name(os_name: str) -> str:
    return "svzerodsolver.exe" if os_name == "Windows" else "svzerodsolver"


def build_solver_0d(*, version: str, os_name: str, arch: str, out_dir: Path, jobs: int) -> None:
    os_name = _norm_os(os_name)
    arch = _norm_arch(arch)
    out_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://github.com/SimVascular/svZeroDSolver/archive/refs/tags/v{version}.tar.gz"

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tar_path = tmp / "svZeroDSolver.tar.gz"
        src_root = tmp / "src"
        build_dir = tmp / "build"
        src_root.mkdir(parents=True, exist_ok=True)
        build_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading svZeroDSolver v{version}...", flush=True)
        urlretrieve(url, tar_path)  # nosec - trusted upstream in CI/release flows

        print("Extracting svZeroDSolver...", flush=True)
        with tarfile.open(tar_path, "r:gz") as t:
            _tar_safe_extract(t, src_root)

        subdirs = [p for p in src_root.iterdir() if p.is_dir()]
        if not subdirs:
            raise RuntimeError("svZeroDSolver source extraction produced no subdirectory")
        src = subdirs[0]

        cmake_cmd = [
            "cmake",
            "-S",
            str(src),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-Wno-dev",
        ]

        if os_name == "Mac" and arch == "universal2":
            deployment = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "11.0").strip()
            cmake_cmd += [
                "-DCMAKE_OSX_ARCHITECTURES=x86_64;arm64",
                f"-DCMAKE_OSX_DEPLOYMENT_TARGET={deployment}",
            ]

        if os_name == "Windows":
            cmake_cmd += ["-A", "x64"]

        _run(cmake_cmd)

        build_cmd = ["cmake", "--build", str(build_dir), "--parallel", str(max(1, jobs))]
        if os_name == "Windows":
            build_cmd += ["--config", "Release"]
        _run(build_cmd)

        # Prefer executable directly from the build tree. Some upstream
        # svZeroDSolver CMake configs don't install `svzerodsolver`.
        found = _find_built_solver_0d_exes(build_dir)
        search_roots = [build_dir]

        if not found:
            install_dir = tmp / "install"
            install_dir.mkdir(parents=True, exist_ok=True)
            install_cmd = ["cmake", "--install", str(build_dir), "--prefix", str(install_dir)]
            if os_name == "Windows":
                install_cmd += ["--config", "Release"]
            _run(install_cmd)
            found = _find_built_solver_0d_exes(install_dir)
            search_roots.append(install_dir)

        if not found:
            roots = ", ".join(str(p) for p in search_roots)
            raise RuntimeError(
                "svZeroDSolver build succeeded but executable `svzerodsolver` was not found. "
                f"Searched: {roots}"
            )

        found = sorted(
            found,
            key=lambda p: (
                (p.stem.lower() if p.suffix.lower() == ".exe" else p.name.lower()) != "svzerodsolver",
                len(str(p)),
            ),
        )
        src_exe = found[0]
        dest = out_dir / _dest_exe_name(os_name)
        if dest.exists():
            dest.unlink()
        shutil.copy2(src_exe, dest)
        _chmod_x(dest)

        print("svZeroDSolver installed to:", out_dir, flush=True)
        print(" -", dest, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and stage svZeroDSolver executable for svv packaging.")
    parser.add_argument("--version", default=SOLVER_0D_VERSION_DEFAULT, help="svZeroDSolver version (default: 3.0)")
    parser.add_argument("--os", dest="os_name", default=platform.system(), help="Target OS (Linux/Mac/Windows)")
    parser.add_argument("--arch", default=platform.machine(), help="Target arch (x86_64/aarch64/universal2)")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for executable (default: svv/utils/solvers/0D/<OS>/<arch>)",
    )
    parser.add_argument("--jobs", type=int, default=(os.cpu_count() or 1), help="Parallel build jobs")
    args = parser.parse_args()

    os_name = _norm_os(args.os_name)
    arch = _norm_arch(args.arch)

    repo_root = Path(__file__).resolve().parents[2]
    default_out = repo_root / "svv" / "utils" / "solvers" / "0D" / os_name / arch
    out_dir = Path(args.out_dir).resolve() if args.out_dir else default_out

    build_solver_0d(version=args.version, os_name=os_name, arch=arch, out_dir=out_dir, jobs=args.jobs)


if __name__ == "__main__":
    main()
