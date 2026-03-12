from pathlib import Path

from svv.domain.io.dmn import ensure_dmn_path, resolve_dmn_read_path


def test_ensure_dmn_path_appends_extension():
    assert ensure_dmn_path("domain") == "domain.dmn"


def test_ensure_dmn_path_normalizes_case():
    assert ensure_dmn_path("domain.DMN") == "domain.dmn"


def test_ensure_dmn_path_accepts_pathlike():
    assert ensure_dmn_path(Path("domain")) == "domain.dmn"


def test_ensure_dmn_path_collapses_legacy_dmn_npz():
    assert ensure_dmn_path("domain.dmn.npz") == "domain.dmn"


def test_resolve_dmn_read_path_prefers_existing_and_supports_legacy(tmp_path):
    legacy = tmp_path / "legacy.dmn.npz"
    legacy.write_bytes(b"")  # content doesn't matter for path resolution

    assert resolve_dmn_read_path(legacy) == str(legacy)
    assert resolve_dmn_read_path(tmp_path / "legacy.dmn") == str(legacy)

