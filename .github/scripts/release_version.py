#!/usr/bin/env python3
"""Calculate release versions and validate release artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path


VERSION_FILE_DEFAULT = Path("svv/__init__.py")
SUPPORTED_PROJECTS = {"svv", "svv-accelerated"}
VERSION_ASSIGNMENT = re.compile(
    r"^(?P<prefix>__version__\s*=\s*)(?P<quote>['\"])(?P<version>[^'\"]+)"
    r"(?P=quote)(?P<trailing>[ \t]*)(?:\r)?$",
    re.MULTILINE,
)
RELEASE_VERSION = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")


class ReleaseVersionError(RuntimeError):
    """Raised when release version data is invalid or inconsistent."""


@dataclass(frozen=True)
class Artifact:
    path: Path
    project: str
    version: str
    kind: str


def _version_match(version: str) -> re.Match[str]:
    match = RELEASE_VERSION.fullmatch(version)
    if match is None:
        raise ReleaseVersionError(
            f"Expected a three-component numeric version, found {version!r}"
        )
    return match


def read_version(path: Path = VERSION_FILE_DEFAULT) -> str:
    text = path.read_text(encoding="utf-8")
    matches = list(VERSION_ASSIGNMENT.finditer(text))
    if len(matches) != 1:
        raise ReleaseVersionError(
            f"Expected exactly one __version__ assignment in {path}, found {len(matches)}"
        )
    version = matches[0].group("version")
    _version_match(version)
    return version


def bump_version(version: str, part: str) -> str:
    match = _version_match(version)
    major = int(match.group("major"))
    minor = int(match.group("minor"))
    patch = int(match.group("patch"))

    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ReleaseVersionError(f"Unsupported version increment: {part!r}")


def write_version(
    path: Path,
    version: str,
    *,
    expected_current: str | None = None,
) -> None:
    _version_match(version)
    text = path.read_text(encoding="utf-8")
    matches = list(VERSION_ASSIGNMENT.finditer(text))
    if len(matches) != 1:
        raise ReleaseVersionError(
            f"Expected exactly one __version__ assignment in {path}, found {len(matches)}"
        )

    match = matches[0]
    current = match.group("version")
    _version_match(current)
    if expected_current is not None and current != expected_current:
        raise ReleaseVersionError(
            f"Expected {path} to contain version {expected_current}, found {current}"
        )

    updated = text[: match.start("version")] + version + text[match.end("version") :]
    path.write_text(updated, encoding="utf-8")


def _canonical_project_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_metadata(payload: bytes, source: str) -> tuple[str, str]:
    metadata = BytesParser().parsebytes(payload)
    name = metadata.get("Name")
    version = metadata.get("Version")
    if not name or not version:
        raise ReleaseVersionError(f"Missing Name or Version metadata in {source}")
    return _canonical_project_name(name), version


def _runtime_version(payload: bytes, source: str) -> str:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReleaseVersionError(f"Unable to decode {source} as UTF-8") from exc
    matches = list(VERSION_ASSIGNMENT.finditer(text))
    if len(matches) != 1:
        raise ReleaseVersionError(
            f"Expected exactly one __version__ assignment in {source}, found {len(matches)}"
        )
    return matches[0].group("version")


def _inspect_wheel(path: Path) -> Artifact:
    with zipfile.ZipFile(path) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise ReleaseVersionError(
                f"Expected one METADATA file in {path.name}, found {len(metadata_files)}"
            )
        project, version = _parse_metadata(
            archive.read(metadata_files[0]), f"{path.name}:{metadata_files[0]}"
        )
        if project == "svv":
            init_name = "svv/__init__.py"
            if init_name not in archive.namelist():
                raise ReleaseVersionError(f"Missing {init_name} in {path.name}")
            runtime_version = _runtime_version(
                archive.read(init_name), f"{path.name}:{init_name}"
            )
            if runtime_version != version:
                raise ReleaseVersionError(
                    f"Runtime version {runtime_version} does not match metadata version "
                    f"{version} in {path.name}"
                )
    return Artifact(path=path, project=project, version=version, kind="wheel")


def _inspect_sdist(path: Path) -> Artifact:
    with tarfile.open(path, mode="r:gz") as archive:
        files = [member for member in archive.getmembers() if member.isfile()]
        metadata_files = [
            member for member in files if member.name.endswith("/PKG-INFO")
        ]
        if not metadata_files:
            raise ReleaseVersionError(f"Missing PKG-INFO in {path.name}")
        metadata_member = min(metadata_files, key=lambda member: member.name.count("/"))
        metadata_stream = archive.extractfile(metadata_member)
        if metadata_stream is None:
            raise ReleaseVersionError(
                f"Unable to read {metadata_member.name} from {path.name}"
            )
        project, version = _parse_metadata(
            metadata_stream.read(), f"{path.name}:{metadata_member.name}"
        )
        if project == "svv":
            init_files = [
                member for member in files if member.name.endswith("/svv/__init__.py")
            ]
            if len(init_files) != 1:
                raise ReleaseVersionError(
                    f"Expected one svv/__init__.py in {path.name}, found {len(init_files)}"
                )
            init_stream = archive.extractfile(init_files[0])
            if init_stream is None:
                raise ReleaseVersionError(
                    f"Unable to read {init_files[0].name} from {path.name}"
                )
            runtime_version = _runtime_version(
                init_stream.read(), f"{path.name}:{init_files[0].name}"
            )
            if runtime_version != version:
                raise ReleaseVersionError(
                    f"Runtime version {runtime_version} does not match metadata version "
                    f"{version} in {path.name}"
                )
    return Artifact(path=path, project=project, version=version, kind="sdist")


def inspect_artifacts(dist_dir: Path, expected_version: str) -> list[Artifact]:
    _version_match(expected_version)
    paths = sorted([*dist_dir.glob("*.whl"), *dist_dir.glob("*.tar.gz")])
    if not paths:
        raise ReleaseVersionError(f"No wheel or source artifacts found in {dist_dir}")

    artifacts: list[Artifact] = []
    for path in paths:
        artifact = (
            _inspect_wheel(path) if path.suffix == ".whl" else _inspect_sdist(path)
        )
        if artifact.project not in SUPPORTED_PROJECTS:
            raise ReleaseVersionError(
                f"Unexpected project {artifact.project!r} in {artifact.path.name}"
            )
        if artifact.version != expected_version:
            raise ReleaseVersionError(
                f"Expected version {expected_version}, found {artifact.version} in "
                f"{artifact.path.name}"
            )
        artifacts.append(artifact)

    required = {
        ("svv", "wheel"),
        ("svv", "sdist"),
        ("svv-accelerated", "wheel"),
    }
    present = {(artifact.project, artifact.kind) for artifact in artifacts}
    missing = sorted(required - present)
    if missing:
        formatted = ", ".join(f"{project} {kind}" for project, kind in missing)
        raise ReleaseVersionError(f"Missing required release artifacts: {formatted}")
    return artifacts


def _fetch_pypi_files(project: str, version: str) -> set[str]:
    encoded_project = urllib.parse.quote(project, safe="")
    encoded_version = urllib.parse.quote(version, safe="")
    url = f"https://pypi.org/pypi/{encoded_project}/{encoded_version}/json"
    request = urllib.request.Request(
        url, headers={"User-Agent": "svv-release-workflow"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return set()
        raise ReleaseVersionError(
            f"PyPI returned HTTP {exc.code} for {project} {version}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ReleaseVersionError(
            f"Unable to query PyPI for {project} {version}: {exc.reason}"
        ) from exc
    return {item["filename"] for item in payload.get("urls", []) if "filename" in item}


def verify_pypi(
    dist_dir: Path,
    expected_version: str,
    *,
    attempts: int = 12,
    delay: float = 5.0,
) -> None:
    if attempts < 1:
        raise ReleaseVersionError("PyPI verification attempts must be at least 1")
    artifacts = inspect_artifacts(dist_dir, expected_version)
    expected: dict[str, set[str]] = {project: set() for project in SUPPORTED_PROJECTS}
    for artifact in artifacts:
        expected[artifact.project].add(artifact.path.name)

    last_details = ""
    for attempt in range(1, attempts + 1):
        problems: list[str] = []
        for project in sorted(expected):
            try:
                published = _fetch_pypi_files(project, expected_version)
            except ReleaseVersionError as exc:
                problems.append(str(exc))
                continue
            missing = sorted(expected[project] - published)
            if missing:
                problems.append(f"{project} missing: {', '.join(missing)}")
        if not problems:
            return
        last_details = "; ".join(problems)
        if attempt < attempts:
            print(
                f"PyPI verification attempt {attempt}/{attempts} incomplete: "
                f"{last_details}",
                flush=True,
            )
            time.sleep(delay)

    raise ReleaseVersionError(
        f"PyPI does not contain every expected {expected_version} artifact after "
        f"{attempts} attempts: {last_details}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    current = subparsers.add_parser("current", help="Print the current source version")
    current.add_argument("--path", type=Path, default=VERSION_FILE_DEFAULT)

    next_version = subparsers.add_parser("next", help="Print the next release version")
    next_version.add_argument("--path", type=Path, default=VERSION_FILE_DEFAULT)
    next_version.add_argument(
        "--bump", choices=("patch", "minor", "major"), required=True
    )

    set_version = subparsers.add_parser("set", help="Update the source version")
    set_version.add_argument("--path", type=Path, default=VERSION_FILE_DEFAULT)
    set_version.add_argument("--version", required=True)
    set_version.add_argument("--expected-current")

    verify_dist = subparsers.add_parser(
        "verify-dist", help="Validate built artifact names and embedded versions"
    )
    verify_dist.add_argument("--dist", type=Path, default=Path("dist"))
    verify_dist.add_argument("--version", required=True)

    verify_index = subparsers.add_parser(
        "verify-pypi", help="Confirm that every built artifact is available from PyPI"
    )
    verify_index.add_argument("--dist", type=Path, default=Path("dist"))
    verify_index.add_argument("--version", required=True)
    verify_index.add_argument("--attempts", type=int, default=12)
    verify_index.add_argument("--delay", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "current":
            print(read_version(args.path))
        elif args.command == "next":
            print(bump_version(read_version(args.path), args.bump))
        elif args.command == "set":
            write_version(
                args.path,
                args.version,
                expected_current=args.expected_current,
            )
            print(args.version)
        elif args.command == "verify-dist":
            artifacts = inspect_artifacts(args.dist, args.version)
            print(f"Validated {len(artifacts)} release artifacts for {args.version}")
        elif args.command == "verify-pypi":
            verify_pypi(
                args.dist,
                args.version,
                attempts=args.attempts,
                delay=args.delay,
            )
            print(f"Verified all {args.version} release artifacts on PyPI")
        else:  # pragma: no cover - argparse restricts command values
            raise ReleaseVersionError(f"Unsupported command: {args.command}")
    except (OSError, ReleaseVersionError, tarfile.TarError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
