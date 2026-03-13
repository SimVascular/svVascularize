import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QTimer

import svv.visualize.gui as gui_mod


def test_launch_gui_restores_and_demotes_activation_policy(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("SVV_GUI_DISABLE_VTK", "1")
    monkeypatch.setenv("SVV_TELEMETRY_DISABLED", "1")

    calls = []
    monkeypatch.setattr(
        gui_mod,
        "_set_macos_activation_policy",
        lambda policy: calls.append(policy) or True,
    )

    app, gui = gui_mod.launch_gui(block=False)
    QTimer.singleShot(0, gui.close)
    app.exec()

    assert calls
    assert calls[0] == "regular"
    assert "accessory" in calls
