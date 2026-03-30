import os
import platform
from pathlib import Path


def tetgen_os_dir():
    sysname = platform.system()
    if sysname == "Linux":
        return "Linux"
    if sysname == "Windows":
        return "Windows"
    if sysname == "Darwin":
        return "Mac"
    raise RuntimeError(f"Unsupported OS for TetGen CLI packaging: {sysname}")


def tetgen_executable_name(os_dir=None):
    if os_dir is None:
        os_dir = tetgen_os_dir()
    return "tetgen.exe" if os_dir == "Windows" else "tetgen"


def tetgen_arch_dir(os_dir=None, base_dir=None):
    if os_dir is None:
        os_dir = tetgen_os_dir()

    override = os.environ.get("SVV_TETGEN_ARCH", "").strip()
    if override:
        ov = override.lower()
        if ov in {"x86_64", "amd64"}:
            return "x86_64"
        if ov in {"aarch64", "arm64"}:
            return "aarch64"
        if ov == "universal2":
            return "universal2"
        return override

    base = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    if os_dir == "Mac":
        exe_name = tetgen_executable_name(os_dir)
        if (base / "Mac" / "universal2" / exe_name).is_file():
            return "universal2"

    machine = platform.machine().strip().lower()
    if machine in {"x86_64", "amd64"}:
        return "x86_64"
    if machine in {"aarch64", "arm64"}:
        return "aarch64"
    return machine or "unknown"


def get_packaged_tetgen_cli_path(base_dir=None):
    base = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    os_dir = tetgen_os_dir()
    arch_dir = tetgen_arch_dir(os_dir=os_dir, base_dir=base)
    exe_name = tetgen_executable_name(os_dir)
    candidate = base / os_dir / arch_dir / exe_name
    return str(candidate) if candidate.is_file() else None
