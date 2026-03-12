import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


def _platform_tags(wheel_name: str) -> str:
    if not wheel_name.endswith(".whl"):
        raise ValueError(f"Not a wheel file: {wheel_name}")

    stem = wheel_name[:-4]
    parts = stem.split("-")
    if len(parts) < 5:
        raise ValueError(f"Unexpected wheel filename: {wheel_name}")
    return parts[-1]


def _arch_from_tags(platform_tags: str) -> str:
    if "_x86_64" in platform_tags:
        return "x86_64"
    if "_aarch64" in platform_tags:
        return "aarch64"
    raise ValueError(f"Unsupported Linux wheel arch in tags: {platform_tags}")


def _manylinux_image(platform_tags: str, arch: str) -> str | None:
    if f"manylinux2014_{arch}" in platform_tags:
        return f"quay.io/pypa/manylinux2014_{arch}"

    matches = [
        (int(major), int(minor))
        for major, minor in re.findall(rf"manylinux_(\d+)_(\d+)_{arch}", platform_tags)
    ]
    if not matches:
        return None

    min_major, min_minor = min(matches)
    return f"quay.io/pypa/manylinux_{min_major}_{min_minor}_{arch}"


def _musllinux_image(platform_tags: str, arch: str) -> str | None:
    match = re.search(rf"musllinux_(\d+)_(\d+)_{arch}", platform_tags)
    if not match:
        return None
    major, minor = match.groups()
    return f"quay.io/pypa/musllinux_{major}_{minor}_{arch}"


def _docker_platform(arch: str) -> str:
    if arch == "x86_64":
        return "linux/amd64"
    if arch == "aarch64":
        return "linux/arm64"
    raise ValueError(f"Unsupported Linux wheel arch: {arch}")


def _container_image(wheel_name: str) -> tuple[str, str]:
    platform_tags = _platform_tags(wheel_name)
    arch = _arch_from_tags(platform_tags)

    image = _manylinux_image(platform_tags, arch)
    if image is None:
        image = _musllinux_image(platform_tags, arch)
    if image is None:
        raise ValueError(f"Unsupported Linux wheel platform tags: {platform_tags}")

    return arch, image


def _python_selector() -> str:
    return (
        "PYTHON_BIN=/opt/python/cp39-cp39/bin/python; "
        'if [ ! -x "$PYTHON_BIN" ]; then '
        'PYTHON_BIN=$(find /opt/python -maxdepth 3 -path "*/bin/python" | sort | head -n 1); '
        "fi; "
        'if [ -z "${PYTHON_BIN:-}" ] || [ ! -x "$PYTHON_BIN" ]; then '
        'echo "Unable to locate a Python interpreter in the container" >&2; exit 1; '
        "fi"
    )


def _validation_command(wheel_basename: str) -> str:
    validation_py = (
        "import os, subprocess, tempfile; "
        "os.chdir(tempfile.mkdtemp(prefix='svv-wheel-validate-')); "
        "from svv.utils.remeshing.mmg import get_mmg_exe; "
        "from svv.utils.solvers.solver_0d import get_solver_0d_exe; "
        "exe = get_mmg_exe('mmgs'); "
        "print('MMG selected:', exe); "
        "proc = subprocess.run([str(exe), '-h'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, check=False); "
        "print('mmgs -h exit:', proc.returncode); "
        "solver_0d = get_solver_0d_exe(); "
        "print('0D solver selected:', solver_0d); "
        "proc = subprocess.run([str(solver_0d), '-h'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, check=False); "
        "print('svzerodsolver -h exit:', proc.returncode)"
    )
    return (
        "set -eu; "
        f"{_python_selector()}; "
        '"$PYTHON_BIN" -m pip install -U pip; '
        f'"$PYTHON_BIN" -m pip install --no-deps /dist/{shlex.quote(wheel_basename)}; '
        f'"$PYTHON_BIN" -c {shlex.quote(validation_py)}'
    )


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: validate_linux_wheel.py <wheel-path>", file=sys.stderr)
        return 2

    wheel_path = Path(sys.argv[1]).resolve()
    if not wheel_path.is_file():
        print(f"Wheel not found: {wheel_path}", file=sys.stderr)
        return 2

    arch, image = _container_image(wheel_path.name)
    docker_platform = _docker_platform(arch)

    print(f"Validating Linux wheel: {wheel_path.name}", flush=True)
    print(f"  arch: {arch}", flush=True)
    print(f"  image: {image}", flush=True)
    print(f"  docker platform: {docker_platform}", flush=True)

    cmd = [
        "docker",
        "run",
        "--rm",
        "--platform",
        docker_platform,
        "-v",
        f"{wheel_path.parent}:/dist",
        image,
        "sh",
        "-lc",
        _validation_command(wheel_path.name),
    ]
    subprocess.check_call(cmd, env=os.environ.copy())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
