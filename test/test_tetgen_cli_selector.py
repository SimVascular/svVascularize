from pathlib import Path

from svv.domain.routines import tetgen_constrained as constrained_mod
from svv.utils.meshing.tetgen import (
    get_packaged_tetgen_cli_path,
    tetgen_arch_dir,
    tetgen_executable_name,
    tetgen_os_dir,
)


def _write_fake_tetgen(base_dir):
    os_dir = tetgen_os_dir()
    arch_dir = tetgen_arch_dir(os_dir=os_dir, base_dir=base_dir)
    exe_name = tetgen_executable_name(os_dir)
    exe_path = Path(base_dir) / os_dir / arch_dir / exe_name
    exe_path.parent.mkdir(parents=True, exist_ok=True)
    exe_path.write_text('', encoding='utf-8')
    return exe_path


def test_get_packaged_tetgen_cli_path_finds_staged_binary(tmp_path):
    exe_path = _write_fake_tetgen(tmp_path)
    assert get_packaged_tetgen_cli_path(base_dir=tmp_path) == str(exe_path)


def test_resolve_tetgen_exe_prefers_explicit_path(monkeypatch, tmp_path):
    explicit = tmp_path / 'tetgen-explicit'
    explicit.write_text('', encoding='utf-8')
    monkeypatch.setattr(constrained_mod, 'get_packaged_tetgen_cli_path', lambda: None)
    monkeypatch.setattr(constrained_mod.shutil, 'which', lambda name: '/usr/bin/tetgen')
    monkeypatch.delenv('SVV_TETGEN_PATH', raising=False)

    assert constrained_mod.resolve_tetgen_exe(str(explicit)) == str(explicit)


def test_resolve_tetgen_exe_prefers_env_then_packaged_then_path(monkeypatch, tmp_path):
    packaged = _write_fake_tetgen(tmp_path)
    env_exe = tmp_path / 'tetgen-env'
    env_exe.write_text('', encoding='utf-8')
    monkeypatch.setattr(constrained_mod, 'get_packaged_tetgen_cli_path', lambda: str(packaged))
    monkeypatch.setattr(constrained_mod.shutil, 'which', lambda name: '/usr/bin/tetgen')

    monkeypatch.setenv('SVV_TETGEN_PATH', str(env_exe))
    assert constrained_mod.resolve_tetgen_exe(None) == str(env_exe)

    monkeypatch.delenv('SVV_TETGEN_PATH', raising=False)
    assert constrained_mod.resolve_tetgen_exe(None) == str(packaged)
