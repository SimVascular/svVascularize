from __future__ import annotations

import os
from typing import Optional, Sequence

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QFormLayout,
    QLineEdit,
    QLabel,
    QPushButton,
    QSlider,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QFileDialog,
    QCheckBox,
    QDockWidget,
    QMenuBar,
    QToolBar,
    QToolButton,
    QFrame,
    QScrollArea,
    QSizePolicy,
    QButtonGroup,
    QGridLayout,
    QMessageBox,
    QDialog,
    QColorDialog,
    QDialogButtonBox,
)
from PySide6.QtGui import QAction, QIcon, QColor

from svv.visualize.gui.vtk_widget import VTKWidget
from svv.visualize.gui.theme import CADTheme, CADIcons
from svv.telemetry import capture_exception, capture_message


class CollapsibleGroupBox(QGroupBox):
    """
    A QGroupBox that can be collapsed/expanded by clicking on its title.
    """

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.toggled.connect(self._on_toggled)
        self._content_height = 0

    def _on_toggled(self, checked: bool) -> None:
        """Handle toggle to show/hide contents."""
        for i in range(self.layout().count()) if self.layout() else []:
            widget = self.layout().itemAt(i).widget()
            if widget:
                widget.setVisible(checked)


class CameraPresetsDialog(QDialog):
    """
    Popup dialog for camera preset controls.

    Provides quick access to standard camera views (+/-X, +/-Y, +/-Z, Iso, Fit)
    and parallel projection toggle.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Camera Presets")
        self.setModal(False)
        self.setFixedSize(200, 140)

        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)

        # Grid of preset buttons
        grid = QGridLayout()
        grid.setSpacing(4)

        self._preset_buttons = {}
        preset_info = [
            ("+X", 0, 0),
            ("-X", 0, 1),
            ("+Y", 0, 2),
            ("-Y", 1, 0),
            ("+Z", 1, 1),
            ("-Z", 1, 2),
            ("Iso", 2, 0),
            ("Fit", 2, 1),
        ]

        for label, row, col in preset_info:
            btn = QPushButton(label)
            btn.setFixedHeight(28)
            self._preset_buttons[label] = btn
            grid.addWidget(btn, row, col)

        layout.addLayout(grid)

        # Parallel projection checkbox
        self.parallel_cb = QCheckBox("Parallel projection")
        layout.addWidget(self.parallel_cb)

    def connect_callbacks(
        self,
        view_plus_x,
        view_minus_x,
        view_plus_y,
        view_minus_y,
        view_plus_z,
        view_minus_z,
        view_iso,
        reset_camera,
        toggle_parallel,
    ) -> None:
        """Connect button callbacks to the parent widget's camera methods."""
        self._preset_buttons["+X"].clicked.connect(view_plus_x)
        self._preset_buttons["-X"].clicked.connect(view_minus_x)
        self._preset_buttons["+Y"].clicked.connect(view_plus_y)
        self._preset_buttons["-Y"].clicked.connect(view_minus_y)
        self._preset_buttons["+Z"].clicked.connect(view_plus_z)
        self._preset_buttons["-Z"].clicked.connect(view_minus_z)
        self._preset_buttons["Iso"].clicked.connect(view_iso)
        self._preset_buttons["Fit"].clicked.connect(reset_camera)
        self.parallel_cb.toggled.connect(toggle_parallel)

    def set_parallel_checked(self, checked: bool) -> None:
        """Update the parallel projection checkbox state."""
        self.parallel_cb.blockSignals(True)
        self.parallel_cb.setChecked(checked)
        self.parallel_cb.blockSignals(False)


class ColorbarOptionsDialog(QDialog):
    """
    Popup dialog for colorbar customization options.

    Provides controls for colormap, orientation, size, position,
    font settings, and display formatting.
    """

    # Signal emitted when any option changes
    options_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Colorbar Options")
        self.setModal(False)
        self.setMinimumWidth(280)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        form_layout = QFormLayout()
        form_layout.setSpacing(6)

        # Colormap selection
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems([
            "coolwarm", "viridis", "plasma", "magma", "inferno",
            "cividis", "RdBu", "Spectral", "Greys", "jet", "rainbow",
        ])
        self.cmap_combo.currentTextChanged.connect(self._emit_changed)
        form_layout.addRow("Colormap:", self.cmap_combo)

        # Global normalization over time
        self.global_range_cb = QCheckBox("Normalize over time")
        self.global_range_cb.setToolTip("Use global min/max across all time steps")
        self.global_range_cb.stateChanged.connect(self._emit_changed)
        form_layout.addRow("", self.global_range_cb)

        # Orientation
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(["Vertical", "Horizontal"])
        self.orientation_combo.currentTextChanged.connect(self._emit_changed)
        form_layout.addRow("Orientation:", self.orientation_combo)

        # Width
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.01, 0.5)
        self.width_spin.setValue(0.05)
        self.width_spin.setSingleStep(0.01)
        self.width_spin.setDecimals(2)
        self.width_spin.setToolTip("Colorbar width (fraction of window)")
        self.width_spin.valueChanged.connect(self._emit_changed)
        form_layout.addRow("Width:", self.width_spin)

        # Height
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.1, 1.0)
        self.height_spin.setValue(0.8)
        self.height_spin.setSingleStep(0.05)
        self.height_spin.setDecimals(2)
        self.height_spin.setToolTip("Colorbar height (fraction of window)")
        self.height_spin.valueChanged.connect(self._emit_changed)
        form_layout.addRow("Height:", self.height_spin)

        # Number of tick labels
        self.nlabels_spin = QSpinBox()
        self.nlabels_spin.setRange(2, 20)
        self.nlabels_spin.setValue(5)
        self.nlabels_spin.setToolTip("Number of tick labels on colorbar")
        self.nlabels_spin.valueChanged.connect(self._emit_changed)
        form_layout.addRow("Tick Labels:", self.nlabels_spin)

        # Number format
        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems([
            "Auto (%.3g)",
            "Decimal (%.2f)",
            "Decimal (%.4f)",
            "Scientific (%.2e)",
            "Scientific (%.3e)",
            "Integer (%d)",
            "Percent (%.1f%%)",
        ])
        self.fmt_combo.setToolTip("Number format for tick labels")
        self.fmt_combo.currentIndexChanged.connect(self._emit_changed)
        form_layout.addRow("Format:", self.fmt_combo)

        # Normalize display to [0, 1]
        self.normalize_cb = QCheckBox("Normalize to [0, 1]")
        self.normalize_cb.setToolTip("Display colorbar scale as normalized [0, 1] range")
        self.normalize_cb.stateChanged.connect(self._emit_changed)
        form_layout.addRow("", self.normalize_cb)

        # Position X
        self.pos_x_spin = QDoubleSpinBox()
        self.pos_x_spin.setRange(0.0, 0.95)
        self.pos_x_spin.setValue(0.85)
        self.pos_x_spin.setSingleStep(0.05)
        self.pos_x_spin.setDecimals(2)
        self.pos_x_spin.setToolTip("Colorbar X position (0=left, 1=right)")
        self.pos_x_spin.valueChanged.connect(self._emit_changed)
        form_layout.addRow("Position X:", self.pos_x_spin)

        # Position Y
        self.pos_y_spin = QDoubleSpinBox()
        self.pos_y_spin.setRange(0.0, 0.95)
        self.pos_y_spin.setValue(0.1)
        self.pos_y_spin.setSingleStep(0.05)
        self.pos_y_spin.setDecimals(2)
        self.pos_y_spin.setToolTip("Colorbar Y position (0=bottom, 1=top)")
        self.pos_y_spin.valueChanged.connect(self._emit_changed)
        form_layout.addRow("Position Y:", self.pos_y_spin)

        # Font family
        self.font_combo = QComboBox()
        self.font_combo.addItems(["Arial", "Courier", "Times"])
        self.font_combo.setToolTip("Font family for colorbar text")
        self.font_combo.currentTextChanged.connect(self._emit_changed)
        form_layout.addRow("Font:", self.font_combo)

        # Title font size
        self.title_size_spin = QSpinBox()
        self.title_size_spin.setRange(8, 48)
        self.title_size_spin.setValue(16)
        self.title_size_spin.setToolTip("Font size for colorbar title")
        self.title_size_spin.valueChanged.connect(self._emit_changed)
        form_layout.addRow("Title Size:", self.title_size_spin)

        # Label font size
        self.label_size_spin = QSpinBox()
        self.label_size_spin.setRange(8, 48)
        self.label_size_spin.setValue(14)
        self.label_size_spin.setToolTip("Font size for colorbar tick labels")
        self.label_size_spin.valueChanged.connect(self._emit_changed)
        form_layout.addRow("Label Size:", self.label_size_spin)

        # Bold font option
        self.bold_cb = QCheckBox("Bold text")
        self.bold_cb.setToolTip("Use bold font for colorbar text")
        self.bold_cb.stateChanged.connect(self._emit_changed)
        form_layout.addRow("", self.bold_cb)

        # Shadow for better readability
        self.shadow_cb = QCheckBox("Text shadow")
        self.shadow_cb.setChecked(True)
        self.shadow_cb.setToolTip("Add shadow behind text for better visibility")
        self.shadow_cb.stateChanged.connect(self._emit_changed)
        form_layout.addRow("", self.shadow_cb)

        layout.addLayout(form_layout)

    def _emit_changed(self) -> None:
        """Emit the options_changed signal."""
        self.options_changed.emit()

    def get_options(self) -> dict:
        """Return current options as a dictionary."""
        return {
            "cmap": self.cmap_combo.currentText(),
            "global_range": self.global_range_cb.isChecked(),
            "orientation": self.orientation_combo.currentText(),
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "nlabels": self.nlabels_spin.value(),
            "fmt_index": self.fmt_combo.currentIndex(),
            "normalize": self.normalize_cb.isChecked(),
            "pos_x": self.pos_x_spin.value(),
            "pos_y": self.pos_y_spin.value(),
            "font": self.font_combo.currentText(),
            "title_size": self.title_size_spin.value(),
            "label_size": self.label_size_spin.value(),
            "bold": self.bold_cb.isChecked(),
            "shadow": self.shadow_cb.isChecked(),
        }

    def set_options(self, options: dict) -> None:
        """Set options from a dictionary, blocking signals during update."""
        self.blockSignals(True)
        if "cmap" in options:
            self.cmap_combo.setCurrentText(options["cmap"])
        if "global_range" in options:
            self.global_range_cb.setChecked(options["global_range"])
        if "orientation" in options:
            self.orientation_combo.setCurrentText(options["orientation"])
        if "width" in options:
            self.width_spin.setValue(options["width"])
        if "height" in options:
            self.height_spin.setValue(options["height"])
        if "nlabels" in options:
            self.nlabels_spin.setValue(options["nlabels"])
        if "fmt_index" in options:
            self.fmt_combo.setCurrentIndex(options["fmt_index"])
        if "normalize" in options:
            self.normalize_cb.setChecked(options["normalize"])
        if "pos_x" in options:
            self.pos_x_spin.setValue(options["pos_x"])
        if "pos_y" in options:
            self.pos_y_spin.setValue(options["pos_y"])
        if "font" in options:
            self.font_combo.setCurrentText(options["font"])
        if "title_size" in options:
            self.title_size_spin.setValue(options["title_size"])
        if "label_size" in options:
            self.label_size_spin.setValue(options["label_size"])
        if "bold" in options:
            self.bold_cb.setChecked(options["bold"])
        if "shadow" in options:
            self.shadow_cb.setChecked(options["shadow"])
        self.blockSignals(False)


class BackgroundOptionsDialog(QDialog):
    """
    Popup dialog for background customization options (ParaView-like).

    Supports solid, gradient, and textured backgrounds with controls for
    colors, gradient mode, dithering, and opacity.
    """

    options_changed = Signal()

    _PRESETS = {
        "CAD Default": {
            "mode": "Gradient",
            "color1": CADTheme.get_color("viewport", "background-bottom"),
            "color2": CADTheme.get_color("viewport", "background-top"),
            "gradient_mode": "Vertical",
            "dither": True,
            "opacity": 1.0,
        },
        "Dark": {
            "mode": "Gradient",
            "color1": "#1E1E1E",
            "color2": "#3C3C3C",
            "gradient_mode": "Vertical",
            "dither": True,
            "opacity": 1.0,
        },
        "Light": {
            "mode": "Gradient",
            "color1": "#FFFFFF",
            "color2": "#DADADA",
            "gradient_mode": "Vertical",
            "dither": True,
            "opacity": 1.0,
        },
        "Black": {
            "mode": "Solid",
            "color1": "#000000",
            "color2": "#000000",
            "gradient_mode": "Vertical",
            "dither": True,
            "opacity": 1.0,
        },
        "White": {
            "mode": "Solid",
            "color1": "#FFFFFF",
            "color2": "#FFFFFF",
            "gradient_mode": "Vertical",
            "dither": True,
            "opacity": 1.0,
        },
    }

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Background")
        self.setModal(False)
        self.setMinimumWidth(360)

        self._updating = False
        self._last_valid_color1 = CADTheme.get_color("viewport", "background-bottom")
        self._last_valid_color2 = CADTheme.get_color("viewport", "background-top")

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        row = 0

        grid.addWidget(QLabel("Preset:"), row, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Custom", "CAD Default", "Dark", "Light", "Black", "White"])
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        grid.addWidget(self.preset_combo, row, 1, 1, 2)
        row += 1

        grid.addWidget(QLabel("Mode:"), row, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Solid", "Gradient", "Texture"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        grid.addWidget(self.mode_combo, row, 1, 1, 2)
        row += 1

        self.color1_label = QLabel("Color:")
        grid.addWidget(self.color1_label, row, 0)
        self.color1_swatch, self.color1_edit = self._make_color_picker(self._last_valid_color1)
        grid.addWidget(self._wrap_row(self.color1_swatch, self.color1_edit), row, 1, 1, 2)
        row += 1

        self.color2_label = QLabel("Top Color:")
        grid.addWidget(self.color2_label, row, 0)
        self.color2_swatch, self.color2_edit = self._make_color_picker(self._last_valid_color2)
        grid.addWidget(self._wrap_row(self.color2_swatch, self.color2_edit), row, 1, 1, 2)
        row += 1

        grid.addWidget(QLabel("Gradient:"), row, 0)
        self.gradient_mode_combo = QComboBox()
        self.gradient_mode_combo.addItems([
            "Vertical",
            "Horizontal",
            "Radial (Farthest Side)",
            "Radial (Farthest Corner)",
        ])
        self.gradient_mode_combo.currentTextChanged.connect(self._on_any_changed)
        grid.addWidget(self.gradient_mode_combo, row, 1, 1, 2)
        row += 1

        self.dither_cb = QCheckBox("Dither (reduce banding)")
        self.dither_cb.setChecked(True)
        self.dither_cb.stateChanged.connect(self._on_any_changed)
        grid.addWidget(self.dither_cb, row, 1, 1, 2)
        row += 1

        grid.addWidget(QLabel("Texture:"), row, 0)
        self.texture_path_edit = QLineEdit()
        self.texture_path_edit.setPlaceholderText("Select an image...")
        self.texture_path_edit.editingFinished.connect(self._on_any_changed)
        self.texture_browse_btn = QPushButton("Browse...")
        self.texture_browse_btn.clicked.connect(self._browse_texture)
        texture_row = QHBoxLayout()
        texture_row.setContentsMargins(0, 0, 0, 0)
        texture_row.setSpacing(6)
        texture_row.addWidget(self.texture_path_edit, 1)
        texture_row.addWidget(self.texture_browse_btn, 0)
        texture_widget = QWidget()
        texture_widget.setLayout(texture_row)
        grid.addWidget(texture_widget, row, 1, 1, 2)
        row += 1

        self.texture_interpolate_cb = QCheckBox("Interpolate")
        self.texture_interpolate_cb.setChecked(True)
        self.texture_interpolate_cb.stateChanged.connect(self._on_any_changed)
        grid.addWidget(self.texture_interpolate_cb, row, 1, 1, 2)
        row += 1

        self.texture_repeat_cb = QCheckBox("Repeat")
        self.texture_repeat_cb.setChecked(False)
        self.texture_repeat_cb.stateChanged.connect(self._on_any_changed)
        grid.addWidget(self.texture_repeat_cb, row, 1, 1, 2)
        row += 1

        grid.addWidget(QLabel("Opacity:"), row, 0)
        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0.0, 1.0)
        self.opacity_spin.setSingleStep(0.05)
        self.opacity_spin.setDecimals(2)
        self.opacity_spin.setValue(1.0)
        self.opacity_spin.valueChanged.connect(self._on_any_changed)
        grid.addWidget(self.opacity_spin, row, 1, 1, 2)
        row += 1

        layout.addLayout(grid)

        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(8)

        self.reset_btn = QPushButton("Reset to Theme")
        self.reset_btn.clicked.connect(self._reset_to_theme)
        buttons_layout.addWidget(self.reset_btn, 0)

        self.swap_btn = QPushButton("Swap Colors")
        self.swap_btn.clicked.connect(self._swap_colors)
        buttons_layout.addWidget(self.swap_btn, 0)

        buttons_layout.addStretch(1)

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.close)
        buttons_layout.addWidget(button_box, 0)

        layout.addLayout(buttons_layout)

        self._wire_color_picker(self.color1_swatch, self.color1_edit, key="color1")
        self._wire_color_picker(self.color2_swatch, self.color2_edit, key="color2")
        self._refresh_enabled_state()

        self.set_settings(self._PRESETS["CAD Default"])
        self._updating = True
        try:
            self.preset_combo.setCurrentText("CAD Default")
        finally:
            self._updating = False

    def _wrap_row(self, swatch: QPushButton, edit: QLineEdit) -> QWidget:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(swatch, 0)
        row.addWidget(edit, 1)
        widget = QWidget()
        widget.setLayout(row)
        return widget

    def _make_color_picker(self, initial_color: str) -> tuple[QPushButton, QLineEdit]:
        swatch = QPushButton()
        swatch.setFixedSize(28, 18)
        swatch.setToolTip("Click to choose a color")
        swatch.setFocusPolicy(Qt.NoFocus)

        edit = QLineEdit()
        edit.setText(initial_color)
        edit.setToolTip("Enter a color (e.g. #RRGGBB, 'white', or '0.2,0.3,0.4')")
        edit.setMaximumWidth(140)

        self._set_swatch_color(swatch, initial_color)
        return swatch, edit

    def _parse_color(self, text: str) -> Optional[QColor]:
        text = (text or "").strip()
        if not text:
            return None
        q = QColor(text)
        if q.isValid():
            return q

        if "," in text:
            parts = [p.strip() for p in text.split(",") if p.strip()]
            if len(parts) in (3, 4):
                try:
                    values = [float(p) for p in parts]
                except ValueError:
                    return None

                if max(values) <= 1.0:
                    r, g, b = values[:3]
                    a = values[3] if len(values) == 4 else 1.0
                    q = QColor.fromRgbF(r, g, b, a)
                else:
                    r, g, b = [max(0, min(255, int(round(v)))) for v in values[:3]]
                    a = max(0, min(255, int(round(values[3])))) if len(values) == 4 else 255
                    q = QColor(r, g, b, a)
                return q if q.isValid() else None

        return None

    def _set_swatch_color(self, swatch: QPushButton, color_text: str) -> None:
        color = self._parse_color(color_text)
        if color is None:
            swatch.setStyleSheet("")
            return
        css_color = color.name(QColor.HexRgb)
        border = CADTheme.get_color("border", "strong")
        swatch.setStyleSheet(f"background-color: {css_color}; border: 1px solid {border};")

    def _wire_color_picker(self, swatch: QPushButton, edit: QLineEdit, key: str) -> None:
        def pick():
            initial = self._parse_color(edit.text()) or QColor("#000000")
            chosen = QColorDialog.getColor(initial, self, "Select Color")
            if not chosen.isValid():
                return
            edit.setText(chosen.name(QColor.HexRgb).upper())
            self._on_color_edited(key)

        swatch.clicked.connect(pick)
        edit.editingFinished.connect(lambda: self._on_color_edited(key))

    def _set_custom_preset(self) -> None:
        if self._updating:
            return
        if self.preset_combo.currentText() == "Custom":
            return
        self._updating = True
        try:
            self.preset_combo.setCurrentText("Custom")
        finally:
            self._updating = False

    def _on_color_edited(self, key: str) -> None:
        edit = self.color1_edit if key == "color1" else self.color2_edit
        swatch = self.color1_swatch if key == "color1" else self.color2_swatch
        last = self._last_valid_color1 if key == "color1" else self._last_valid_color2

        color = self._parse_color(edit.text())
        if color is None:
            edit.setText(last)
            return

        normalized = color.name(QColor.HexRgb).upper()
        edit.setText(normalized)
        if key == "color1":
            self._last_valid_color1 = normalized
        else:
            self._last_valid_color2 = normalized

        self._set_swatch_color(swatch, normalized)
        self._set_custom_preset()
        self._emit_changed()

    def _browse_texture(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Background Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return
        self.texture_path_edit.setText(path)
        self._set_custom_preset()
        self._emit_changed()

    def _on_preset_changed(self, preset: str) -> None:
        if self._updating or preset == "Custom":
            return
        settings = self._PRESETS.get(preset)
        if not settings:
            return
        self._updating = True
        try:
            self.set_settings(settings)
        finally:
            self._updating = False
        self._emit_changed()

    def _on_mode_changed(self, mode: str) -> None:
        if not self._updating:
            self._set_custom_preset()
        self._refresh_enabled_state()
        self._emit_changed()

    def _on_any_changed(self) -> None:
        if not self._updating:
            self._set_custom_preset()
        self._emit_changed()

    def _refresh_enabled_state(self) -> None:
        mode = self.mode_combo.currentText()

        is_gradient = mode == "Gradient"
        is_texture = mode == "Texture"

        self.color2_label.setEnabled(is_gradient)
        self.color2_swatch.setEnabled(is_gradient)
        self.color2_edit.setEnabled(is_gradient)
        self.gradient_mode_combo.setEnabled(is_gradient)
        self.dither_cb.setEnabled(is_gradient)
        self.swap_btn.setEnabled(is_gradient)

        self.texture_path_edit.setEnabled(is_texture)
        self.texture_browse_btn.setEnabled(is_texture)
        self.texture_interpolate_cb.setEnabled(is_texture)
        self.texture_repeat_cb.setEnabled(is_texture)

        self.color1_label.setText("Color:" if mode == "Solid" else "Bottom Color:")

    def _swap_colors(self) -> None:
        if self.mode_combo.currentText() != "Gradient":
            return
        c1 = self.color1_edit.text()
        c2 = self.color2_edit.text()
        self.color1_edit.setText(c2)
        self.color2_edit.setText(c1)
        self._on_color_edited("color1")
        self._on_color_edited("color2")

    def _reset_to_theme(self) -> None:
        self._updating = True
        try:
            self.preset_combo.setCurrentText("CAD Default")
        finally:
            self._updating = False
        self._on_preset_changed("CAD Default")

    def _emit_changed(self) -> None:
        self.options_changed.emit()

    def get_settings(self) -> dict:
        return {
            "mode": self.mode_combo.currentText(),
            "color1": self.color1_edit.text().strip() or self._last_valid_color1,
            "color2": self.color2_edit.text().strip() or self._last_valid_color2,
            "gradient_mode": self.gradient_mode_combo.currentText(),
            "dither": self.dither_cb.isChecked(),
            "opacity": float(self.opacity_spin.value()),
            "texture_path": self.texture_path_edit.text().strip(),
            "texture_interpolate": self.texture_interpolate_cb.isChecked(),
            "texture_repeat": self.texture_repeat_cb.isChecked(),
        }

    def set_settings(self, settings: dict) -> None:
        self._updating = True
        try:
            mode = settings.get("mode", "Gradient")
            self.mode_combo.setCurrentText(mode)

            c1 = settings.get("color1", self._last_valid_color1)
            c2 = settings.get("color2", self._last_valid_color2)
            self.color1_edit.setText(c1)
            self.color2_edit.setText(c2)
            self._last_valid_color1 = self._parse_color(c1).name(QColor.HexRgb).upper() if self._parse_color(c1) else self._last_valid_color1
            self._last_valid_color2 = self._parse_color(c2).name(QColor.HexRgb).upper() if self._parse_color(c2) else self._last_valid_color2
            self._set_swatch_color(self.color1_swatch, self._last_valid_color1)
            self._set_swatch_color(self.color2_swatch, self._last_valid_color2)

            gm = settings.get("gradient_mode", "Vertical")
            self.gradient_mode_combo.setCurrentText(gm)

            if "dither" in settings:
                self.dither_cb.setChecked(bool(settings["dither"]))
            if "opacity" in settings:
                self.opacity_spin.setValue(float(settings["opacity"]))

            if "texture_path" in settings:
                self.texture_path_edit.setText(settings["texture_path"] or "")
            if "texture_interpolate" in settings:
                self.texture_interpolate_cb.setChecked(bool(settings["texture_interpolate"]))
            if "texture_repeat" in settings:
                self.texture_repeat_cb.setChecked(bool(settings["texture_repeat"]))
        finally:
            self._updating = False
        self._refresh_enabled_state()


class StatisticsDialog(QDialog):
    """
    Popup dialog for displaying mesh/field statistics.

    Shows statistics for the current scalar field including
    count, min, max, mean, std dev, and median.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Field Statistics")
        self.setModal(False)
        self.setMinimumWidth(280)
        self.setMinimumHeight(200)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        # Field name display
        self.field_label = QLabel("Field: (none)")
        self.field_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(self.field_label)

        # Statistics grid
        stats_grid = QGridLayout()
        stats_grid.setSpacing(6)

        self.stats_labels = {}
        stat_names = [
            ("count", "Count:"),
            ("min", "Min:"),
            ("max", "Max:"),
            ("mean", "Mean:"),
            ("std", "Std Dev:"),
            ("median", "Median:"),
            ("range", "Range:"),
            ("sum", "Sum:"),
        ]

        for i, (key, label_text) in enumerate(stat_names):
            label = QLabel(label_text)
            label.setStyleSheet("font-weight: bold;")
            value_label = QLabel("--")
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            value_label.setStyleSheet("font-family: monospace;")
            stats_grid.addWidget(label, i, 0)
            stats_grid.addWidget(value_label, i, 1)
            self.stats_labels[key] = value_label

        layout.addLayout(stats_grid)

        # Refresh button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setToolTip("Refresh statistics for current field")
        btn_layout.addWidget(self.refresh_btn)

        layout.addLayout(btn_layout)
        layout.addStretch()

    def update_statistics(self, field_name: str, stats: dict) -> None:
        """
        Update the statistics display.

        Parameters
        ----------
        field_name : str
            Name of the field being displayed
        stats : dict
            Dictionary with keys: count, min, max, mean, std, median, range, sum
        """
        self.field_label.setText(f"Field: {field_name}")

        for key, value in stats.items():
            if key not in self.stats_labels:
                continue

            if key == "count":
                self.stats_labels[key].setText(f"{int(value):,}")
            elif abs(value) < 0.001 or abs(value) > 100000:
                self.stats_labels[key].setText(f"{value:.6e}")
            else:
                self.stats_labels[key].setText(f"{value:.6g}")

    def clear_statistics(self) -> None:
        """Clear all statistics display."""
        self.field_label.setText("Field: (none)")
        for key in self.stats_labels:
            self.stats_labels[key].setText("--")


class LinePlotDialog(QDialog):
    """
    Dialog for sampling and plotting values along a line.

    Allows users to define two points and sample scalar values
    along the line between them.
    """

    # Signal emitted when user wants to pick points
    pick_points_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Line Plot")
        self.setModal(False)
        self.setMinimumSize(700, 500)
        self.resize(800, 550)

        self._point1 = None
        self._point2 = None
        self._mpl_available = False

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        # Point selection controls
        points_group = QGroupBox("Line Definition")
        points_layout = QGridLayout()
        points_layout.setSpacing(6)
        points_group.setLayout(points_layout)

        points_layout.addWidget(QLabel("Point 1:"), 0, 0)
        self.point1_label = QLabel("(not set)")
        self.point1_label.setStyleSheet("font-family: monospace;")
        points_layout.addWidget(self.point1_label, 0, 1)

        points_layout.addWidget(QLabel("Point 2:"), 1, 0)
        self.point2_label = QLabel("(not set)")
        self.point2_label.setStyleSheet("font-family: monospace;")
        points_layout.addWidget(self.point2_label, 1, 1)

        self.pick_btn = QPushButton("Pick Points on Mesh")
        self.pick_btn.setToolTip("Click two points on the mesh to define the line")
        self.pick_btn.clicked.connect(self._on_pick_points)
        points_layout.addWidget(self.pick_btn, 0, 2, 2, 1)

        # Number of samples
        points_layout.addWidget(QLabel("Samples:"), 2, 0)
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(10, 1000)
        self.samples_spin.setValue(100)
        self.samples_spin.setToolTip("Number of sample points along the line")
        points_layout.addWidget(self.samples_spin, 2, 1)

        self.plot_btn = QPushButton("Plot")
        self.plot_btn.setToolTip("Sample and plot values along the line")
        self.plot_btn.clicked.connect(self._plot_line)
        self.plot_btn.setEnabled(False)
        points_layout.addWidget(self.plot_btn, 2, 2)

        layout.addWidget(points_group)

        # Matplotlib plot area
        try:
            import matplotlib
            matplotlib.use('QtAgg')
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
            from matplotlib.figure import Figure

            self._mpl_available = True
            self._fig = Figure(figsize=(8, 4), dpi=100)
            self._ax = self._fig.add_subplot(111)
            self._canvas = FigureCanvas(self._fig)
            self._toolbar = NavigationToolbar(self._canvas, self)

            layout.addWidget(self._toolbar)
            layout.addWidget(self._canvas, 1)

        except ImportError:
            error_label = QLabel("Matplotlib is required for line plots.\nInstall with: pip install matplotlib")
            error_label.setStyleSheet("color: #cc0000; padding: 20px;")
            layout.addWidget(error_label)

        # Store references for plotting
        self._mesh = None
        self._scalar_name = None

    def set_mesh(self, mesh, scalar_name: str) -> None:
        """Set the mesh and scalar field to sample from."""
        self._mesh = mesh
        self._scalar_name = scalar_name

    def set_point1(self, point: tuple) -> None:
        """Set the first point."""
        self._point1 = point
        self.point1_label.setText(f"({point[0]:.4g}, {point[1]:.4g}, {point[2]:.4g})")
        self._update_plot_button()

    def set_point2(self, point: tuple) -> None:
        """Set the second point."""
        self._point2 = point
        self.point2_label.setText(f"({point[0]:.4g}, {point[1]:.4g}, {point[2]:.4g})")
        self._update_plot_button()

    def _update_plot_button(self) -> None:
        """Enable plot button if both points are set."""
        self.plot_btn.setEnabled(self._point1 is not None and self._point2 is not None)

    def _on_pick_points(self) -> None:
        """Request point picking from parent."""
        self._point1 = None
        self._point2 = None
        self.point1_label.setText("(click first point...)")
        self.point2_label.setText("(waiting...)")
        self._update_plot_button()
        self.pick_points_requested.emit()

    def _plot_line(self) -> None:
        """Sample and plot values along the line."""
        if not self._mpl_available or self._mesh is None:
            return
        if self._point1 is None or self._point2 is None:
            return

        import numpy as np

        try:
            # Create sample points along the line
            n_samples = self.samples_spin.value()
            p1 = np.array(self._point1)
            p2 = np.array(self._point2)

            # Parameter along line
            t = np.linspace(0, 1, n_samples)
            points = np.outer(1 - t, p1) + np.outer(t, p2)

            # Distance along line
            distances = t * np.linalg.norm(p2 - p1)

            # Sample values using probe
            values = []
            mesh = self._mesh

            # Get scalar values
            if self._scalar_name in mesh.point_data:
                scalar_data = mesh.point_data[self._scalar_name]
            elif self._scalar_name in mesh.cell_data:
                scalar_data = mesh.cell_data[self._scalar_name]
            else:
                QMessageBox.warning(self, "Error", f"Scalar field '{self._scalar_name}' not found.")
                return

            # Sample using interpolation
            from scipy.interpolate import NearestNDInterpolator

            if self._scalar_name in mesh.point_data:
                coords = np.array(mesh.points)
            else:
                coords = np.array(mesh.cell_centers().points)

            # Handle multi-component data
            if scalar_data.ndim > 1:
                scalar_data = np.linalg.norm(scalar_data, axis=1)

            interpolator = NearestNDInterpolator(coords, scalar_data)
            values = interpolator(points)

            # Plot
            self._ax.clear()
            self._ax.plot(distances, values, 'b-', linewidth=1.5)
            self._ax.set_xlabel("Distance along line")
            self._ax.set_ylabel(self._scalar_name)
            self._ax.set_title(f"Line Plot: {self._scalar_name}")
            self._ax.grid(True, alpha=0.3)
            self._fig.tight_layout()
            self._canvas.draw()

        except ImportError:
            QMessageBox.warning(self, "Missing Dependency",
                "scipy is required for line plot interpolation.\nInstall with: pip install scipy")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to create line plot:\n{e}")


class CompareFieldsDialog(QDialog):
    """
    Dialog for comparing two scalar fields side by side.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Compare Fields")
        self.setModal(False)
        self.setMinimumSize(900, 600)
        self.resize(1000, 700)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        # Field selection
        select_layout = QHBoxLayout()
        select_layout.setSpacing(20)

        # Field 1
        field1_layout = QVBoxLayout()
        field1_layout.addWidget(QLabel("Field 1:"))
        self.field1_combo = QComboBox()
        self.field1_combo.currentTextChanged.connect(self._on_field_changed)
        field1_layout.addWidget(self.field1_combo)
        select_layout.addLayout(field1_layout)

        # Field 2
        field2_layout = QVBoxLayout()
        field2_layout.addWidget(QLabel("Field 2:"))
        self.field2_combo = QComboBox()
        self.field2_combo.currentTextChanged.connect(self._on_field_changed)
        field2_layout.addWidget(self.field2_combo)
        select_layout.addLayout(field2_layout)

        select_layout.addStretch()
        layout.addLayout(select_layout)

        # Comparison view with matplotlib
        try:
            import matplotlib
            matplotlib.use('QtAgg')
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
            from matplotlib.figure import Figure

            self._mpl_available = True
            self._fig = Figure(figsize=(12, 5), dpi=100)
            self._ax1 = self._fig.add_subplot(121)
            self._ax2 = self._fig.add_subplot(122)
            self._canvas = FigureCanvas(self._fig)
            self._toolbar = NavigationToolbar(self._canvas, self)

            layout.addWidget(self._toolbar)
            layout.addWidget(self._canvas, 1)

        except ImportError:
            self._mpl_available = False
            error_label = QLabel("Matplotlib is required for field comparison.\nInstall with: pip install matplotlib")
            error_label.setStyleSheet("color: #cc0000; padding: 20px;")
            layout.addWidget(error_label)

        # Statistics comparison
        stats_group = QGroupBox("Statistics Comparison")
        stats_layout = QGridLayout()
        stats_layout.setSpacing(4)
        stats_group.setLayout(stats_layout)

        headers = ["", "Field 1", "Field 2", "Difference"]
        for col, header in enumerate(headers):
            label = QLabel(header)
            label.setStyleSheet("font-weight: bold;")
            stats_layout.addWidget(label, 0, col)

        self._stats_labels = {}
        stat_names = ["Min", "Max", "Mean", "Std Dev"]
        for row, stat in enumerate(stat_names, start=1):
            stats_layout.addWidget(QLabel(f"{stat}:"), row, 0)
            for col in range(1, 4):
                label = QLabel("--")
                label.setStyleSheet("font-family: monospace;")
                stats_layout.addWidget(label, row, col)
                self._stats_labels[(stat, col)] = label

        layout.addWidget(stats_group)

        self._mesh = None

    def set_mesh(self, mesh) -> None:
        """Set the mesh and populate field combos."""
        self._mesh = mesh

        fields = []
        if mesh is not None:
            fields = list(mesh.point_data.keys()) + list(mesh.cell_data.keys())

        self.field1_combo.clear()
        self.field2_combo.clear()
        self.field1_combo.addItems(fields)
        self.field2_combo.addItems(fields)

        if len(fields) >= 2:
            self.field2_combo.setCurrentIndex(1)

    def _on_field_changed(self) -> None:
        """Update comparison when field selection changes."""
        if self._mesh is None:
            return

        field1 = self.field1_combo.currentText()
        field2 = self.field2_combo.currentText()

        if not field1 or not field2:
            return

        self._update_comparison(field1, field2)

    def _update_comparison(self, field1: str, field2: str) -> None:
        """Update the comparison display."""
        import numpy as np

        mesh = self._mesh

        # Get field data
        def get_field(name):
            if name in mesh.point_data:
                data = mesh.point_data[name]
            elif name in mesh.cell_data:
                data = mesh.cell_data[name]
            else:
                return None
            if data.ndim > 1:
                data = np.linalg.norm(data, axis=1)
            return data

        data1 = get_field(field1)
        data2 = get_field(field2)

        if data1 is None or data2 is None:
            return

        # Update statistics
        stats1 = {
            "Min": np.nanmin(data1),
            "Max": np.nanmax(data1),
            "Mean": np.nanmean(data1),
            "Std Dev": np.nanstd(data1),
        }
        stats2 = {
            "Min": np.nanmin(data2),
            "Max": np.nanmax(data2),
            "Mean": np.nanmean(data2),
            "Std Dev": np.nanstd(data2),
        }

        for stat in ["Min", "Max", "Mean", "Std Dev"]:
            v1, v2 = stats1[stat], stats2[stat]
            diff = v2 - v1
            self._stats_labels[(stat, 1)].setText(f"{v1:.4g}")
            self._stats_labels[(stat, 2)].setText(f"{v2:.4g}")
            self._stats_labels[(stat, 3)].setText(f"{diff:+.4g}")

        # Update histograms
        if self._mpl_available:
            self._ax1.clear()
            self._ax2.clear()

            self._ax1.hist(data1.flatten(), bins=50, edgecolor='black', alpha=0.7)
            self._ax1.set_xlabel(field1)
            self._ax1.set_ylabel("Frequency")
            self._ax1.set_title(f"Distribution: {field1}")
            self._ax1.grid(True, alpha=0.3)

            self._ax2.hist(data2.flatten(), bins=50, edgecolor='black', alpha=0.7, color='orange')
            self._ax2.set_xlabel(field2)
            self._ax2.set_ylabel("Frequency")
            self._ax2.set_title(f"Distribution: {field2}")
            self._ax2.grid(True, alpha=0.3)

            self._fig.tight_layout()
            self._canvas.draw()


class FieldDifferenceDialog(QDialog):
    """
    Dialog for computing difference between two fields or timesteps.
    """

    # Signal emitted when a new field is computed
    field_computed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Field Difference")
        self.setModal(False)
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        # Mode selection
        mode_group = QGroupBox("Difference Mode")
        mode_layout = QVBoxLayout()
        mode_group.setLayout(mode_layout)

        self.field_mode_rb = QRadioButton("Between two fields")
        self.field_mode_rb.setChecked(True)
        self.field_mode_rb.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self.field_mode_rb)

        self.time_mode_rb = QRadioButton("Between two timesteps")
        self.time_mode_rb.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self.time_mode_rb)

        layout.addWidget(mode_group)

        # Field selection (for field mode)
        self.field_group = QGroupBox("Field Selection")
        field_layout = QFormLayout()
        field_layout.setSpacing(6)
        self.field_group.setLayout(field_layout)

        self.field_a_combo = QComboBox()
        field_layout.addRow("Field A:", self.field_a_combo)

        self.field_b_combo = QComboBox()
        field_layout.addRow("Field B:", self.field_b_combo)

        layout.addWidget(self.field_group)

        # Timestep selection (for time mode)
        self.time_group = QGroupBox("Timestep Selection")
        time_layout = QFormLayout()
        time_layout.setSpacing(6)
        self.time_group.setLayout(time_layout)

        self.time_field_combo = QComboBox()
        time_layout.addRow("Field:", self.time_field_combo)

        self.time_a_spin = QSpinBox()
        self.time_a_spin.setMinimum(0)
        time_layout.addRow("Timestep A:", self.time_a_spin)

        self.time_b_spin = QSpinBox()
        self.time_b_spin.setMinimum(0)
        time_layout.addRow("Timestep B:", self.time_b_spin)

        self.time_group.setVisible(False)
        layout.addWidget(self.time_group)

        # Output name
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output name:"))
        self.output_edit = QLineEdit("field_difference")
        output_layout.addWidget(self.output_edit)
        layout.addLayout(output_layout)

        # Compute button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.compute_btn = QPushButton("Compute Difference")
        self.compute_btn.clicked.connect(self._compute_difference)
        btn_layout.addWidget(self.compute_btn)

        layout.addLayout(btn_layout)
        layout.addStretch()

        self._mesh = None
        self._time_values = []
        self._load_timestep_callback = None

    def set_mesh(self, mesh) -> None:
        """Set the mesh and populate field combos."""
        self._mesh = mesh

        fields = []
        if mesh is not None:
            fields = list(mesh.point_data.keys()) + list(mesh.cell_data.keys())

        self.field_a_combo.clear()
        self.field_b_combo.clear()
        self.time_field_combo.clear()

        self.field_a_combo.addItems(fields)
        self.field_b_combo.addItems(fields)
        self.time_field_combo.addItems(fields)

        if len(fields) >= 2:
            self.field_b_combo.setCurrentIndex(1)

    def set_time_values(self, time_values: list, load_callback) -> None:
        """Set available timesteps."""
        self._time_values = time_values
        self._load_timestep_callback = load_callback

        n_times = len(time_values) if time_values else 0
        self.time_a_spin.setMaximum(max(0, n_times - 1))
        self.time_b_spin.setMaximum(max(0, n_times - 1))

        if n_times >= 2:
            self.time_b_spin.setValue(n_times - 1)

        # Enable/disable time mode
        self.time_mode_rb.setEnabled(n_times >= 2)

    def _on_mode_changed(self) -> None:
        """Update UI based on selected mode."""
        field_mode = self.field_mode_rb.isChecked()
        self.field_group.setVisible(field_mode)
        self.time_group.setVisible(not field_mode)

    def _compute_difference(self) -> None:
        """Compute the field difference."""
        import numpy as np

        if self._mesh is None:
            QMessageBox.warning(self, "No Data", "No mesh loaded.")
            return

        output_name = self.output_edit.text().strip()
        if not output_name:
            output_name = "field_difference"

        try:
            if self.field_mode_rb.isChecked():
                # Field difference mode
                field_a = self.field_a_combo.currentText()
                field_b = self.field_b_combo.currentText()

                if not field_a or not field_b:
                    QMessageBox.warning(self, "Error", "Please select both fields.")
                    return

                # Get data
                data_a = self._get_field_data(self._mesh, field_a)
                data_b = self._get_field_data(self._mesh, field_b)

                if data_a is None or data_b is None:
                    QMessageBox.warning(self, "Error", "Could not retrieve field data.")
                    return

                if data_a.shape != data_b.shape:
                    QMessageBox.warning(self, "Error", "Fields have different shapes and cannot be compared.")
                    return

                # Compute difference
                diff = data_b - data_a

                # Add to mesh
                if field_a in self._mesh.point_data:
                    self._mesh.point_data[output_name] = diff
                else:
                    self._mesh.cell_data[output_name] = diff

            else:
                # Timestep difference mode
                if self._load_timestep_callback is None:
                    QMessageBox.warning(self, "Error", "Timestep loading not available.")
                    return

                field = self.time_field_combo.currentText()
                idx_a = self.time_a_spin.value()
                idx_b = self.time_b_spin.value()

                if not field:
                    QMessageBox.warning(self, "Error", "Please select a field.")
                    return

                # Load timestep A
                mesh_a = self._load_timestep_callback(idx_a)
                data_a = self._get_field_data(mesh_a, field)

                # Load timestep B
                mesh_b = self._load_timestep_callback(idx_b)
                data_b = self._get_field_data(mesh_b, field)

                if data_a is None or data_b is None:
                    QMessageBox.warning(self, "Error", "Could not retrieve field data.")
                    return

                if data_a.shape != data_b.shape:
                    QMessageBox.warning(self, "Error", "Fields have different shapes at different timesteps.")
                    return

                # Compute difference
                diff = data_b - data_a

                # Add to current mesh
                if field in self._mesh.point_data:
                    self._mesh.point_data[output_name] = diff
                else:
                    self._mesh.cell_data[output_name] = diff

            self.field_computed.emit(output_name)
            QMessageBox.information(self, "Success", f"Field '{output_name}' computed successfully.")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to compute difference:\n{e}")

    def _get_field_data(self, mesh, field_name: str):
        """Get field data from mesh."""
        import numpy as np

        if field_name in mesh.point_data:
            data = mesh.point_data[field_name]
        elif field_name in mesh.cell_data:
            data = mesh.cell_data[field_name]
        else:
            return None

        # Handle multi-component
        if data.ndim > 1:
            data = np.linalg.norm(data, axis=1)

        return data.copy()


class KeyframeAnimationDialog(QDialog):
    """
    Dialog for creating keyframe-based camera animations.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Keyframe Animation")
        self.setModal(False)
        self.setMinimumSize(500, 400)

        self._keyframes = []  # List of (time, camera_position, camera_focal, camera_up)
        self._plotter = None

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        # Keyframe list
        list_group = QGroupBox("Keyframes")
        list_layout = QVBoxLayout()
        list_group.setLayout(list_layout)

        self.keyframe_list = QListWidget()
        self.keyframe_list.currentRowChanged.connect(self._on_keyframe_selected)
        list_layout.addWidget(self.keyframe_list)

        btn_layout = QHBoxLayout()

        self.add_btn = QPushButton("Add Current View")
        self.add_btn.setToolTip("Add current camera position as keyframe")
        self.add_btn.clicked.connect(self._add_keyframe)
        btn_layout.addWidget(self.add_btn)

        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self._remove_keyframe)
        btn_layout.addWidget(self.remove_btn)

        self.preview_btn = QPushButton("Preview")
        self.preview_btn.setToolTip("Jump to selected keyframe")
        self.preview_btn.clicked.connect(self._preview_keyframe)
        btn_layout.addWidget(self.preview_btn)

        list_layout.addLayout(btn_layout)
        layout.addWidget(list_group)

        # Animation settings
        settings_group = QGroupBox("Animation Settings")
        settings_layout = QFormLayout()
        settings_layout.setSpacing(6)
        settings_group.setLayout(settings_layout)

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1.0, 60.0)
        self.duration_spin.setValue(5.0)
        self.duration_spin.setSuffix(" seconds")
        settings_layout.addRow("Duration:", self.duration_spin)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(10, 60)
        self.fps_spin.setValue(30)
        settings_layout.addRow("FPS:", self.fps_spin)

        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(["Linear", "Smooth (Cubic)"])
        settings_layout.addRow("Interpolation:", self.interpolation_combo)

        layout.addWidget(settings_group)

        # Action buttons
        action_layout = QHBoxLayout()
        action_layout.addStretch()

        self.play_btn = QPushButton("Play Preview")
        self.play_btn.clicked.connect(self._play_animation)
        action_layout.addWidget(self.play_btn)

        self.export_btn = QPushButton("Export...")
        self.export_btn.clicked.connect(self._export_animation)
        action_layout.addWidget(self.export_btn)

        layout.addLayout(action_layout)

    def set_plotter(self, plotter) -> None:
        """Set the VTK plotter for camera control."""
        self._plotter = plotter

    def _add_keyframe(self) -> None:
        """Add current camera position as keyframe."""
        if self._plotter is None:
            return

        camera = self._plotter.camera
        position = tuple(camera.position)
        focal = tuple(camera.focal_point)
        up = tuple(camera.up)

        time = len(self._keyframes)
        self._keyframes.append({
            'time': time,
            'position': position,
            'focal': focal,
            'up': up
        })

        self.keyframe_list.addItem(f"Keyframe {time}: pos=({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})")

    def _remove_keyframe(self) -> None:
        """Remove selected keyframe."""
        row = self.keyframe_list.currentRow()
        if row >= 0 and row < len(self._keyframes):
            del self._keyframes[row]
            self.keyframe_list.takeItem(row)
            # Renumber
            for i, kf in enumerate(self._keyframes):
                kf['time'] = i
            self._refresh_list()

    def _refresh_list(self) -> None:
        """Refresh the keyframe list display."""
        self.keyframe_list.clear()
        for kf in self._keyframes:
            pos = kf['position']
            self.keyframe_list.addItem(
                f"Keyframe {kf['time']}: pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
            )

    def _on_keyframe_selected(self, row: int) -> None:
        """Handle keyframe selection."""
        pass

    def _preview_keyframe(self) -> None:
        """Jump to selected keyframe."""
        row = self.keyframe_list.currentRow()
        if row >= 0 and row < len(self._keyframes) and self._plotter is not None:
            kf = self._keyframes[row]
            self._plotter.camera.position = kf['position']
            self._plotter.camera.focal_point = kf['focal']
            self._plotter.camera.up = kf['up']
            self._plotter.render()

    def _play_animation(self) -> None:
        """Play animation preview."""
        if len(self._keyframes) < 2:
            QMessageBox.information(self, "Need Keyframes", "Add at least 2 keyframes to preview animation.")
            return

        if self._plotter is None:
            return

        import numpy as np
        from PySide6.QtCore import QTimer

        duration = self.duration_spin.value()
        fps = self.fps_spin.value()
        n_frames = int(duration * fps)
        smooth = "Smooth" in self.interpolation_combo.currentText()

        # Generate interpolated camera positions
        self._anim_frames = []
        for i in range(n_frames):
            t = i / (n_frames - 1)

            # Find keyframe segment
            n_kf = len(self._keyframes)
            segment_t = t * (n_kf - 1)
            kf_idx = min(int(segment_t), n_kf - 2)
            local_t = segment_t - kf_idx

            if smooth:
                # Cubic smoothstep
                local_t = local_t * local_t * (3 - 2 * local_t)

            kf1 = self._keyframes[kf_idx]
            kf2 = self._keyframes[kf_idx + 1]

            # Interpolate
            pos = tuple(np.array(kf1['position']) * (1 - local_t) + np.array(kf2['position']) * local_t)
            focal = tuple(np.array(kf1['focal']) * (1 - local_t) + np.array(kf2['focal']) * local_t)
            up = tuple(np.array(kf1['up']) * (1 - local_t) + np.array(kf2['up']) * local_t)

            self._anim_frames.append((pos, focal, up))

        self._anim_idx = 0
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._animation_tick)
        self._anim_timer.start(int(1000 / fps))

    def _animation_tick(self) -> None:
        """Update animation frame."""
        if self._anim_idx >= len(self._anim_frames):
            self._anim_timer.stop()
            return

        pos, focal, up = self._anim_frames[self._anim_idx]
        self._plotter.camera.position = pos
        self._plotter.camera.focal_point = focal
        self._plotter.camera.up = up
        self._plotter.render()

        self._anim_idx += 1

    def _export_animation(self) -> None:
        """Export animation to video file."""
        if len(self._keyframes) < 2:
            QMessageBox.information(self, "Need Keyframes", "Add at least 2 keyframes to export animation.")
            return

        from PySide6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Animation", "camera_animation.gif",
            "GIF Files (*.gif);;MP4 Files (*.mp4)"
        )

        if not file_path:
            return

        QMessageBox.information(
            self, "Export",
            f"Animation would be exported to:\n{file_path}\n\n"
            "Full export functionality requires imageio/ffmpeg."
        )


class SplitViewDialog(QDialog):
    """
    Dialog for viewing two timesteps side by side.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Split View - Compare Timesteps")
        self.setModal(False)
        self.setMinimumSize(1000, 600)
        self.resize(1200, 700)

        self._time_values = []
        self._meshes = {}  # Cache loaded meshes
        self._load_callback = None
        self._scalar_name = None

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        # Timestep selection
        select_layout = QHBoxLayout()

        select_layout.addWidget(QLabel("Left timestep:"))
        self.left_spin = QSpinBox()
        self.left_spin.valueChanged.connect(self._update_views)
        select_layout.addWidget(self.left_spin)

        select_layout.addSpacing(20)

        select_layout.addWidget(QLabel("Right timestep:"))
        self.right_spin = QSpinBox()
        self.right_spin.valueChanged.connect(self._update_views)
        select_layout.addWidget(self.right_spin)

        select_layout.addSpacing(20)

        self.sync_cb = QCheckBox("Sync cameras")
        self.sync_cb.setChecked(True)
        select_layout.addWidget(self.sync_cb)

        select_layout.addStretch()
        layout.addLayout(select_layout)

        # Split view with two VTK widgets
        from PySide6.QtWidgets import QSplitter

        splitter = QSplitter(Qt.Horizontal)

        # Left view
        self.left_widget = QWidget()
        left_layout = QVBoxLayout(self.left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.left_label = QLabel("Left View")
        self.left_label.setAlignment(Qt.AlignCenter)
        self.left_label.setStyleSheet("font-weight: bold; padding: 4px;")
        left_layout.addWidget(self.left_label)

        # Placeholder for VTK widget
        self.left_view_placeholder = QLabel("Load timestep to view")
        self.left_view_placeholder.setAlignment(Qt.AlignCenter)
        self.left_view_placeholder.setMinimumSize(400, 400)
        self.left_view_placeholder.setStyleSheet("background-color: #333; color: #888;")
        left_layout.addWidget(self.left_view_placeholder, 1)

        splitter.addWidget(self.left_widget)

        # Right view
        self.right_widget = QWidget()
        right_layout = QVBoxLayout(self.right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_label = QLabel("Right View")
        self.right_label.setAlignment(Qt.AlignCenter)
        self.right_label.setStyleSheet("font-weight: bold; padding: 4px;")
        right_layout.addWidget(self.right_label)

        self.right_view_placeholder = QLabel("Load timestep to view")
        self.right_view_placeholder.setAlignment(Qt.AlignCenter)
        self.right_view_placeholder.setMinimumSize(400, 400)
        self.right_view_placeholder.setStyleSheet("background-color: #333; color: #888;")
        right_layout.addWidget(self.right_view_placeholder, 1)

        splitter.addWidget(self.right_widget)

        layout.addWidget(splitter, 1)

        # Info labels
        info_layout = QHBoxLayout()
        self.left_info = QLabel("")
        self.left_info.setStyleSheet("font-family: monospace;")
        info_layout.addWidget(self.left_info)
        info_layout.addStretch()
        self.right_info = QLabel("")
        self.right_info.setStyleSheet("font-family: monospace;")
        info_layout.addWidget(self.right_info)
        layout.addLayout(info_layout)

    def set_time_values(self, time_values: list, load_callback, scalar_name: str = None) -> None:
        """Set available timesteps and loading callback."""
        self._time_values = time_values
        self._load_callback = load_callback
        self._scalar_name = scalar_name

        n_times = len(time_values) if time_values else 0
        self.left_spin.setMaximum(max(0, n_times - 1))
        self.right_spin.setMaximum(max(0, n_times - 1))

        if n_times >= 2:
            self.left_spin.setValue(0)
            self.right_spin.setValue(n_times - 1)

    def _update_views(self) -> None:
        """Update both views when timestep selection changes."""
        left_idx = self.left_spin.value()
        right_idx = self.right_spin.value()

        if self._time_values:
            left_time = self._time_values[left_idx] if left_idx < len(self._time_values) else 0
            right_time = self._time_values[right_idx] if right_idx < len(self._time_values) else 0

            self.left_label.setText(f"Left View - t = {left_time:.4g}")
            self.right_label.setText(f"Right View - t = {right_time:.4g}")

            # Update info
            self.left_info.setText(f"Timestep {left_idx}: t = {left_time:.6g}")
            self.right_info.setText(f"Timestep {right_idx}: t = {right_time:.6g}")


class TimeSeriesPlotDialog(QDialog):
    """
    Dialog for displaying time series data with matplotlib.

    Provides an interactive matplotlib figure with the standard
    navigation toolbar for pan, zoom, save, and other operations.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Time Series Plot")
        self.setMinimumSize(800, 600)
        self.resize(900, 650)

        # Try to import matplotlib
        self._mpl_available = False
        self._fig = None
        self._ax = None
        self._canvas = None
        self._toolbar = None

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        try:
            import matplotlib
            matplotlib.use('QtAgg')
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
            from matplotlib.figure import Figure
            import matplotlib.backends.backend_qtagg as backend_qtagg

            # Custom toolbar that removes the "Export values" button from figure options
            class CustomNavigationToolbar(NavigationToolbar):
                def __init__(self, canvas, parent):
                    # Disable figureoptions to remove "Export values" button
                    backend_qtagg.figureoptions = None
                    super().__init__(canvas, parent)

            self._mpl_available = True

            # Create matplotlib figure and canvas
            self._fig = Figure(figsize=(10, 6), dpi=100)
            self._ax = self._fig.add_subplot(111)
            self._canvas = FigureCanvas(self._fig)
            self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            # Create navigation toolbar with customized tools (no Export values)
            self._toolbar = CustomNavigationToolbar(self._canvas, self)

            # Add toolbar and canvas to layout
            layout.addWidget(self._toolbar)
            layout.addWidget(self._canvas, 1)

        except ImportError as e:
            # Matplotlib not available
            error_label = QLabel(
                "Matplotlib is required for time series plotting.\n\n"
                "Please install it with:\n"
                "  pip install matplotlib\n\n"
                f"Error: {e}"
            )
            error_label.setWordWrap(True)
            error_label.setStyleSheet("color: #cc0000; padding: 20px;")
            layout.addWidget(error_label)
            return

        # Additional plot controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)

        # Line style
        controls_layout.addWidget(QLabel("Line:"))
        self.line_style_combo = QComboBox()
        self.line_style_combo.addItems(["-", "--", "-.", ":", "None"])
        self.line_style_combo.setToolTip("Line style")
        self.line_style_combo.currentTextChanged.connect(self._update_plot_style)
        controls_layout.addWidget(self.line_style_combo)

        # Marker style
        controls_layout.addWidget(QLabel("Marker:"))
        self.marker_combo = QComboBox()
        self.marker_combo.addItems(["o", "s", "^", "v", "D", "x", "+", "*", "None"])
        self.marker_combo.setToolTip("Marker style")
        self.marker_combo.currentTextChanged.connect(self._update_plot_style)
        controls_layout.addWidget(self.marker_combo)

        # Color
        controls_layout.addWidget(QLabel("Color:"))
        self.color_combo = QComboBox()
        self.color_combo.addItems([
            "blue", "red", "green", "orange", "purple",
            "brown", "pink", "gray", "cyan", "magenta"
        ])
        self.color_combo.setToolTip("Line/marker color")
        self.color_combo.currentTextChanged.connect(self._update_plot_style)
        controls_layout.addWidget(self.color_combo)

        # Grid toggle
        self.grid_cb = QCheckBox("Grid")
        self.grid_cb.setChecked(True)
        self.grid_cb.toggled.connect(self._toggle_grid)
        controls_layout.addWidget(self.grid_cb)

        # Legend toggle
        self.legend_cb = QCheckBox("Legend")
        self.legend_cb.setChecked(True)
        self.legend_cb.toggled.connect(self._toggle_legend)
        controls_layout.addWidget(self.legend_cb)

        controls_layout.addStretch()

        # Refresh button
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setToolTip("Refresh the plot with current data")
        self.refresh_btn.clicked.connect(self._refresh_plot)
        controls_layout.addWidget(self.refresh_btn)

        layout.addLayout(controls_layout)

        # --- Export and Statistics Row ---
        export_stats_layout = QHBoxLayout()
        export_stats_layout.setSpacing(8)

        # Export buttons
        export_label = QLabel("Export:")
        export_stats_layout.addWidget(export_label)

        self.export_csv_btn = QPushButton("CSV")
        self.export_csv_btn.setToolTip("Export data to CSV file")
        self.export_csv_btn.clicked.connect(self._export_to_csv)
        export_stats_layout.addWidget(self.export_csv_btn)

        self.export_clipboard_btn = QPushButton("Clipboard")
        self.export_clipboard_btn.setToolTip("Copy data to clipboard (tab-separated)")
        self.export_clipboard_btn.clicked.connect(self._export_to_clipboard)
        export_stats_layout.addWidget(self.export_clipboard_btn)

        export_stats_layout.addSpacing(20)

        # Statistics toggle
        self.stats_cb = QCheckBox("Statistics")
        self.stats_cb.setChecked(False)
        self.stats_cb.setToolTip("Show/hide statistics panel")
        self.stats_cb.toggled.connect(self._toggle_statistics)
        export_stats_layout.addWidget(self.stats_cb)

        export_stats_layout.addStretch()

        layout.addLayout(export_stats_layout)

        # --- Statistics Panel (initially hidden) ---
        self.stats_group = QGroupBox("Statistics")
        stats_layout = QGridLayout()
        stats_layout.setSpacing(4)
        stats_layout.setContentsMargins(8, 8, 8, 8)
        self.stats_group.setLayout(stats_layout)

        # Create statistics labels
        self.stats_labels = {}
        stat_names = [
            ("min", "Min:"), ("max", "Max:"),
            ("mean", "Mean:"), ("std", "Std Dev:"),
            ("median", "Median:"), ("range", "Range:"),
            ("count", "Points:"), ("sum", "Sum:")
        ]

        for i, (key, label_text) in enumerate(stat_names):
            row = i // 2
            col = (i % 2) * 2
            label = QLabel(label_text)
            label.setStyleSheet("font-weight: bold;")
            value_label = QLabel("--")
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            stats_layout.addWidget(label, row, col)
            stats_layout.addWidget(value_label, row, col + 1)
            self.stats_labels[key] = value_label

        self.stats_group.setVisible(False)
        layout.addWidget(self.stats_group)

        # --- Annotations Panel (initially hidden) ---
        self.annotations_group = QGroupBox("Annotations")
        annotations_layout = QVBoxLayout()
        annotations_layout.setSpacing(4)
        annotations_layout.setContentsMargins(8, 8, 8, 8)
        self.annotations_group.setLayout(annotations_layout)

        # Annotation mode selection
        ann_mode_layout = QHBoxLayout()
        ann_mode_layout.setSpacing(4)
        ann_mode_layout.addWidget(QLabel("Mode:"))
        self.ann_mode_combo = QComboBox()
        self.ann_mode_combo.addItems(["Click to Add", "Mark Min/Max", "Mark Peaks"])
        self.ann_mode_combo.setToolTip("Annotation placement mode")
        ann_mode_layout.addWidget(self.ann_mode_combo)
        annotations_layout.addLayout(ann_mode_layout)

        # Annotation text input
        ann_text_layout = QHBoxLayout()
        ann_text_layout.setSpacing(4)
        ann_text_layout.addWidget(QLabel("Text:"))
        self.ann_text_edit = QLineEdit()
        self.ann_text_edit.setPlaceholderText("Annotation text (or leave blank for value)")
        ann_text_layout.addWidget(self.ann_text_edit)
        annotations_layout.addLayout(ann_text_layout)

        # Annotation style
        ann_style_layout = QHBoxLayout()
        ann_style_layout.setSpacing(4)
        ann_style_layout.addWidget(QLabel("Style:"))
        self.ann_style_combo = QComboBox()
        self.ann_style_combo.addItems(["Text + Arrow", "Text Only", "Marker Only", "Vertical Line"])
        ann_style_layout.addWidget(self.ann_style_combo)
        annotations_layout.addLayout(ann_style_layout)

        # Annotation buttons
        ann_btn_layout = QHBoxLayout()
        ann_btn_layout.setSpacing(4)
        self.ann_add_btn = QPushButton("Add Auto")
        self.ann_add_btn.setToolTip("Add annotations based on selected mode")
        self.ann_add_btn.clicked.connect(self._add_auto_annotations)
        ann_btn_layout.addWidget(self.ann_add_btn)

        self.ann_clear_btn = QPushButton("Clear All")
        self.ann_clear_btn.setToolTip("Remove all annotations")
        self.ann_clear_btn.clicked.connect(self._clear_annotations)
        ann_btn_layout.addWidget(self.ann_clear_btn)
        annotations_layout.addLayout(ann_btn_layout)

        # Enable click-to-annotate
        self.ann_click_cb = QCheckBox("Enable click to annotate")
        self.ann_click_cb.setToolTip("Click on plot to add annotation at that point")
        self.ann_click_cb.toggled.connect(self._toggle_click_annotate)
        annotations_layout.addWidget(self.ann_click_cb)

        self.annotations_group.setVisible(False)
        layout.addWidget(self.annotations_group)

        # --- Derivatives Panel (initially hidden) ---
        self.derivatives_group = QGroupBox("Derivatives")
        derivatives_layout = QVBoxLayout()
        derivatives_layout.setSpacing(4)
        derivatives_layout.setContentsMargins(8, 8, 8, 8)
        self.derivatives_group.setLayout(derivatives_layout)

        # Derivative order selection
        deriv_order_layout = QHBoxLayout()
        deriv_order_layout.setSpacing(4)
        deriv_order_layout.addWidget(QLabel("Order:"))
        self.deriv_order_combo = QComboBox()
        self.deriv_order_combo.addItems(["1st Derivative (dy/dt)", "2nd Derivative (d²y/dt²)"])
        deriv_order_layout.addWidget(self.deriv_order_combo)
        derivatives_layout.addLayout(deriv_order_layout)

        # Smoothing option
        deriv_smooth_layout = QHBoxLayout()
        deriv_smooth_layout.setSpacing(4)
        deriv_smooth_layout.addWidget(QLabel("Smooth:"))
        self.deriv_smooth_spin = QSpinBox()
        self.deriv_smooth_spin.setRange(0, 20)
        self.deriv_smooth_spin.setValue(0)
        self.deriv_smooth_spin.setToolTip("Smoothing window size (0 = no smoothing)")
        deriv_smooth_layout.addWidget(self.deriv_smooth_spin)
        derivatives_layout.addLayout(deriv_smooth_layout)

        # Plot options
        self.deriv_separate_cb = QCheckBox("Plot on separate axis")
        self.deriv_separate_cb.setChecked(True)
        self.deriv_separate_cb.setToolTip("Plot derivative on secondary Y-axis")
        derivatives_layout.addWidget(self.deriv_separate_cb)

        # Derivative buttons
        deriv_btn_layout = QHBoxLayout()
        deriv_btn_layout.setSpacing(4)
        self.deriv_compute_btn = QPushButton("Compute")
        self.deriv_compute_btn.setToolTip("Compute and plot derivative")
        self.deriv_compute_btn.clicked.connect(self._compute_derivative)
        deriv_btn_layout.addWidget(self.deriv_compute_btn)

        self.deriv_clear_btn = QPushButton("Clear")
        self.deriv_clear_btn.setToolTip("Remove derivative plot")
        self.deriv_clear_btn.clicked.connect(self._clear_derivative)
        deriv_btn_layout.addWidget(self.deriv_clear_btn)
        derivatives_layout.addLayout(deriv_btn_layout)

        self.derivatives_group.setVisible(False)
        layout.addWidget(self.derivatives_group)

        # --- Tools Row (Annotations & Derivatives toggles) ---
        tools_layout = QHBoxLayout()
        tools_layout.setSpacing(8)

        tools_layout.addWidget(QLabel("Tools:"))

        self.annotations_cb = QCheckBox("Annotations")
        self.annotations_cb.setToolTip("Show/hide annotations panel")
        self.annotations_cb.toggled.connect(lambda checked: self.annotations_group.setVisible(checked))
        tools_layout.addWidget(self.annotations_cb)

        self.derivatives_cb = QCheckBox("Derivatives")
        self.derivatives_cb.setToolTip("Show/hide derivatives panel")
        self.derivatives_cb.toggled.connect(lambda checked: self.derivatives_group.setVisible(checked))
        tools_layout.addWidget(self.derivatives_cb)

        tools_layout.addStretch()
        layout.addLayout(tools_layout)

        # Store plot data for refreshing
        self._plot_data = []  # List of (times, values, label) tuples
        self._lines = []  # matplotlib line objects
        self._xlabel = "Time"
        self._ylabel = "Value"
        self._title = "Time Series"
        self._annotations = []  # Store annotation objects
        self._derivative_line = None  # Store derivative line
        self._derivative_ax = None  # Secondary axis for derivative
        self._click_cid = None  # Click event connection ID

    def plot_time_series(
        self,
        times,
        values,
        label: str = "Probed Point",
        xlabel: str = "Time",
        ylabel: str = "Value",
        title: str = "Time Series",
        clear: bool = True,
        color: str = None
    ) -> None:
        """
        Plot time series data.

        Parameters
        ----------
        times : array-like
            Time values (x-axis)
        values : array-like
            Data values (y-axis)
        label : str
            Label for the data series
        xlabel : str
            X-axis label
        ylabel : str
            Y-axis label
        title : str
            Plot title
        color : str, optional
            Line/marker color. If None, uses color from combo box or auto-assigns
        clear : bool
            If True, clear previous plots before adding new one
        """
        if not self._mpl_available or self._ax is None:
            return

        import numpy as np
        times = np.asarray(times)
        values = np.asarray(values)

        if clear:
            self._ax.clear()
            self._plot_data = []
            self._lines = []

        # Store data for refresh (include color)
        self._plot_data.append((times.copy(), values.copy(), label, color))
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._title = title

        # Get current style settings
        line_style = self.line_style_combo.currentText()
        marker = self.marker_combo.currentText()

        # Use provided color or fall back to combo box selection
        plot_color = color if color else self.color_combo.currentText()

        # Handle "None" selections - use empty string for no line/marker
        # When line is "None", ensure marker is visible
        if line_style == "None":
            line_style = ""
            if marker == "None":
                marker = "o"  # Default to circles if both are None
        if marker == "None":
            marker = ""

        # Plot the data
        line, = self._ax.plot(
            times, values,
            linestyle=line_style if line_style else "-",
            marker=marker,
            color=plot_color,
            label=label,
            markersize=6 if not line_style else 5,
            linewidth=1.5
        )
        self._lines.append(line)

        # Configure axes
        self._ax.set_xlabel(xlabel)
        self._ax.set_ylabel(ylabel)
        self._ax.set_title(title)

        if self.grid_cb.isChecked():
            self._ax.grid(True, linestyle='--', alpha=0.7)

        if self.legend_cb.isChecked() and label:
            self._ax.legend(loc='best')

        self._fig.tight_layout()
        self._canvas.draw()

        # Update statistics if panel is visible
        if hasattr(self, 'stats_cb') and self.stats_cb.isChecked():
            self._update_statistics()

    def add_series(self, times, values, label: str = "Series") -> None:
        """Add another series to the existing plot."""
        self.plot_time_series(times, values, label=label, clear=False)

    def _update_plot_style(self) -> None:
        """Update the plot style based on current combo selections."""
        if not self._mpl_available or not self._lines:
            return

        line_style = self.line_style_combo.currentText()
        marker = self.marker_combo.currentText()
        color = self.color_combo.currentText()

        # Handle "None" selections
        no_line = (line_style == "None")
        if line_style == "None":
            line_style = ""
        if marker == "None":
            marker = "" if not no_line else "o"  # Show markers if no line

        for line in self._lines:
            line.set_linestyle(line_style if line_style else "-")
            line.set_marker(marker)
            line.set_color(color)
            line.set_markersize(6 if no_line else 5)

        self._canvas.draw()

    def _toggle_grid(self, enabled: bool) -> None:
        """Toggle grid visibility."""
        if not self._mpl_available or self._ax is None:
            return
        self._ax.grid(enabled, linestyle='--', alpha=0.7)
        self._canvas.draw()

    def _toggle_legend(self, enabled: bool) -> None:
        """Toggle legend visibility."""
        if not self._mpl_available or self._ax is None:
            return
        legend = self._ax.get_legend()
        if legend:
            legend.set_visible(enabled)
        elif enabled and self._plot_data:
            self._ax.legend(loc='best')
        self._canvas.draw()

    def _refresh_plot(self) -> None:
        """Refresh the plot with stored data."""
        if not self._mpl_available or not self._plot_data:
            return

        self._ax.clear()
        self._lines = []

        line_style = self.line_style_combo.currentText()
        marker = self.marker_combo.currentText()
        fallback_color = self.color_combo.currentText()

        # Handle "None" selections
        no_line = (line_style == "None")
        if line_style == "None":
            line_style = ""
        if marker == "None":
            marker = "" if not no_line else "o"  # Show markers if no line

        fallback_colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "cyan", "magenta"]

        for i, data_tuple in enumerate(self._plot_data):
            # Handle both old 3-tuple and new 4-tuple formats
            if len(data_tuple) == 4:
                times, values, label, stored_color = data_tuple
            else:
                times, values, label = data_tuple
                stored_color = None

            # Use stored color if available, otherwise use fallback
            if stored_color:
                c = stored_color
            elif len(self._plot_data) == 1:
                c = fallback_color
            else:
                c = fallback_colors[i % len(fallback_colors)]

            line, = self._ax.plot(
                times, values,
                linestyle=line_style if line_style else "-",
                marker=marker,
                color=c,
                label=label,
                markersize=6 if no_line else 5,
                linewidth=1.5
            )
            self._lines.append(line)

        self._ax.set_xlabel(self._xlabel)
        self._ax.set_ylabel(self._ylabel)
        self._ax.set_title(self._title)

        if self.grid_cb.isChecked():
            self._ax.grid(True, linestyle='--', alpha=0.7)

        if self.legend_cb.isChecked():
            self._ax.legend(loc='best')

        self._fig.tight_layout()
        self._canvas.draw()

    def _export_to_csv(self) -> None:
        """Export plot data to a CSV file."""
        if not self._plot_data:
            QMessageBox.warning(self, "No Data", "No data to export.")
            return

        from PySide6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export to CSV",
            "time_series_data.csv",
            "CSV Files (*.csv);;All Files (*)"
        )

        if not file_path:
            return

        try:
            import numpy as np

            with open(file_path, 'w', newline='') as f:
                # Write header
                headers = ["Time"]
                for data_tuple in self._plot_data:
                    label = data_tuple[2]  # label is always at index 2
                    headers.append(label.replace(",", ";"))
                f.write(",".join(headers) + "\n")

                # Get all unique time values and create lookup
                all_times = set()
                for data_tuple in self._plot_data:
                    times = data_tuple[0]
                    all_times.update(times)
                all_times = sorted(all_times)

                # Create data lookup for each series
                series_data = []
                for data_tuple in self._plot_data:
                    times, values = data_tuple[0], data_tuple[1]
                    lookup = dict(zip(times, values))
                    series_data.append(lookup)

                # Write data rows
                for t in all_times:
                    row = [str(t)]
                    for lookup in series_data:
                        if t in lookup:
                            row.append(str(lookup[t]))
                        else:
                            row.append("")
                    f.write(",".join(row) + "\n")

            QMessageBox.information(
                self, "Export Complete",
                f"Data exported to:\n{file_path}"
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Export Error",
                f"Failed to export data:\n{e}"
            )

    def _export_to_clipboard(self) -> None:
        """Copy plot data to clipboard as tab-separated values."""
        if not self._plot_data:
            QMessageBox.warning(self, "No Data", "No data to copy.")
            return

        try:
            from PySide6.QtWidgets import QApplication
            import numpy as np

            lines = []

            # Header
            headers = ["Time"]
            for data_tuple in self._plot_data:
                label = data_tuple[2]  # label is always at index 2
                headers.append(label)
            lines.append("\t".join(headers))

            # Get all unique time values
            all_times = set()
            for data_tuple in self._plot_data:
                times = data_tuple[0]
                all_times.update(times)
            all_times = sorted(all_times)

            # Create data lookup for each series
            series_data = []
            for data_tuple in self._plot_data:
                times, values = data_tuple[0], data_tuple[1]
                lookup = dict(zip(times, values))
                series_data.append(lookup)

            # Data rows
            for t in all_times:
                row = [str(t)]
                for lookup in series_data:
                    if t in lookup:
                        row.append(str(lookup[t]))
                    else:
                        row.append("")
                lines.append("\t".join(row))

            clipboard_text = "\n".join(lines)
            QApplication.clipboard().setText(clipboard_text)

            QMessageBox.information(
                self, "Copied",
                f"Data copied to clipboard ({len(all_times)} rows)."
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Copy Error",
                f"Failed to copy data:\n{e}"
            )

    def _toggle_statistics(self, enabled: bool) -> None:
        """Toggle statistics panel visibility."""
        self.stats_group.setVisible(enabled)
        if enabled:
            self._update_statistics()

    def _update_statistics(self) -> None:
        """Update statistics display based on current plot data."""
        if not self._plot_data:
            for key in self.stats_labels:
                self.stats_labels[key].setText("--")
            return

        try:
            import numpy as np

            # Combine all values from all series for overall statistics
            all_values = []
            for data_tuple in self._plot_data:
                values = data_tuple[1]  # values is always at index 1
                all_values.extend(values)

            if not all_values:
                for key in self.stats_labels:
                    self.stats_labels[key].setText("--")
                return

            values_arr = np.array(all_values)

            # Compute statistics
            stats = {
                "min": np.min(values_arr),
                "max": np.max(values_arr),
                "mean": np.mean(values_arr),
                "std": np.std(values_arr),
                "median": np.median(values_arr),
                "range": np.max(values_arr) - np.min(values_arr),
                "count": len(values_arr),
                "sum": np.sum(values_arr)
            }

            # Format and display
            for key, value in stats.items():
                if key == "count":
                    self.stats_labels[key].setText(f"{int(value)}")
                elif abs(value) < 0.001 or abs(value) > 10000:
                    self.stats_labels[key].setText(f"{value:.4e}")
                else:
                    self.stats_labels[key].setText(f"{value:.4f}")

        except Exception as e:
            for key in self.stats_labels:
                self.stats_labels[key].setText("Error")

    def clear_plot(self) -> None:
        """Clear the plot."""
        if not self._mpl_available or self._ax is None:
            return
        self._ax.clear()
        self._plot_data = []
        self._lines = []
        self._annotations = []
        self._clear_derivative()
        self._canvas.draw()
        # Clear statistics too
        if hasattr(self, 'stats_labels'):
            for key in self.stats_labels:
                self.stats_labels[key].setText("--")

    # --- Annotation Methods ---

    def _toggle_click_annotate(self, enabled: bool) -> None:
        """Enable or disable click-to-annotate mode."""
        if not self._mpl_available or self._canvas is None:
            return

        if enabled:
            self._click_cid = self._canvas.mpl_connect('button_press_event', self._on_click_annotate)
        else:
            if self._click_cid is not None:
                self._canvas.mpl_disconnect(self._click_cid)
                self._click_cid = None

    def _on_click_annotate(self, event) -> None:
        """Handle click event for adding annotations."""
        if event.inaxes != self._ax:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        # Find nearest data point
        nearest_x, nearest_y = self._find_nearest_point(x, y)
        if nearest_x is not None:
            x, y = nearest_x, nearest_y

        # Get annotation text
        custom_text = self.ann_text_edit.text().strip()
        if custom_text:
            text = custom_text
        else:
            text = f"({x:.3g}, {y:.3g})"

        self._add_annotation(x, y, text)

    def _find_nearest_point(self, x: float, y: float):
        """Find the nearest data point to the given coordinates."""
        if not self._plot_data:
            return None, None

        import numpy as np
        min_dist = float('inf')
        nearest_x, nearest_y = None, None

        for data_tuple in self._plot_data:
            times = np.asarray(data_tuple[0])
            values = np.asarray(data_tuple[1])

            # Normalize to plot range for distance calculation
            x_range = times.max() - times.min() if times.max() != times.min() else 1
            y_range = values.max() - values.min() if values.max() != values.min() else 1

            distances = np.sqrt(((times - x) / x_range) ** 2 + ((values - y) / y_range) ** 2)
            idx = np.argmin(distances)

            if distances[idx] < min_dist:
                min_dist = distances[idx]
                nearest_x = times[idx]
                nearest_y = values[idx]

        return nearest_x, nearest_y

    def _add_annotation(self, x: float, y: float, text: str) -> None:
        """Add an annotation at the specified point."""
        if not self._mpl_available or self._ax is None:
            return

        style = self.ann_style_combo.currentText()

        if style == "Text + Arrow":
            ann = self._ax.annotate(
                text, xy=(x, y), xytext=(15, 15),
                textcoords='offset points',
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='black', lw=1),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
            )
        elif style == "Text Only":
            ann = self._ax.annotate(
                text, xy=(x, y), xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7)
            )
        elif style == "Marker Only":
            ann = self._ax.plot(x, y, 'ro', markersize=10, markeredgecolor='black', markeredgewidth=1.5)[0]
        elif style == "Vertical Line":
            ann = self._ax.axvline(x=x, color='red', linestyle='--', alpha=0.7, label=text)

        self._annotations.append(ann)
        self._canvas.draw()

    def _add_auto_annotations(self) -> None:
        """Add annotations automatically based on selected mode."""
        if not self._mpl_available or not self._plot_data:
            return

        import numpy as np
        mode = self.ann_mode_combo.currentText()

        for data_tuple in self._plot_data:
            times = np.asarray(data_tuple[0])
            values = np.asarray(data_tuple[1])
            label = data_tuple[2]

            if mode == "Mark Min/Max":
                # Mark minimum
                min_idx = np.argmin(values)
                self._add_annotation(times[min_idx], values[min_idx], f"Min: {values[min_idx]:.3g}")

                # Mark maximum
                max_idx = np.argmax(values)
                self._add_annotation(times[max_idx], values[max_idx], f"Max: {values[max_idx]:.3g}")

            elif mode == "Mark Peaks":
                # Simple peak detection
                peaks = self._find_peaks(values)
                for idx in peaks:
                    self._add_annotation(times[idx], values[idx], f"{values[idx]:.3g}")

    def _find_peaks(self, values) -> list:
        """Find local maxima in the data."""
        import numpy as np
        values = np.asarray(values)
        peaks = []

        for i in range(1, len(values) - 1):
            if values[i] > values[i-1] and values[i] > values[i+1]:
                peaks.append(i)

        # Limit to top 5 peaks by value
        if len(peaks) > 5:
            peak_values = [(idx, values[idx]) for idx in peaks]
            peak_values.sort(key=lambda x: x[1], reverse=True)
            peaks = [pv[0] for pv in peak_values[:5]]

        return peaks

    def _clear_annotations(self) -> None:
        """Remove all annotations from the plot."""
        if not self._mpl_available or self._ax is None:
            return

        for ann in self._annotations:
            try:
                ann.remove()
            except (ValueError, AttributeError):
                pass

        self._annotations = []
        self._canvas.draw()

    # --- Derivative Methods ---

    def _compute_derivative(self) -> None:
        """Compute and plot the derivative of the data."""
        if not self._mpl_available or not self._plot_data:
            QMessageBox.warning(self, "No Data", "No data to compute derivative.")
            return

        import numpy as np

        order = 1 if "1st" in self.deriv_order_combo.currentText() else 2
        smooth_window = self.deriv_smooth_spin.value()
        use_separate_axis = self.deriv_separate_cb.isChecked()

        # Clear previous derivative
        self._clear_derivative()

        # Use first data series
        times, values, label = self._plot_data[0]
        times = np.asarray(times)
        values = np.asarray(values)

        # Apply smoothing if requested
        if smooth_window > 0:
            values = self._smooth_data(values, smooth_window)

        # Compute derivative
        if order == 1:
            deriv = np.gradient(values, times)
            deriv_label = f"d({label})/dt"
        else:
            deriv1 = np.gradient(values, times)
            deriv = np.gradient(deriv1, times)
            deriv_label = f"d²({label})/dt²"

        # Plot derivative
        if use_separate_axis:
            self._derivative_ax = self._ax.twinx()
            self._derivative_line, = self._derivative_ax.plot(
                times, deriv, 'r--', linewidth=1.5, label=deriv_label
            )
            self._derivative_ax.set_ylabel(deriv_label, color='red')
            self._derivative_ax.tick_params(axis='y', labelcolor='red')
            self._derivative_ax.legend(loc='upper right')
        else:
            self._derivative_line, = self._ax.plot(
                times, deriv, 'r--', linewidth=1.5, label=deriv_label
            )
            self._ax.legend(loc='best')

        self._fig.tight_layout()
        self._canvas.draw()

    def _smooth_data(self, values, window: int):
        """Apply moving average smoothing to data."""
        import numpy as np
        if window < 2:
            return values

        # Simple moving average
        kernel = np.ones(window) / window
        # Pad to maintain length
        padded = np.pad(values, (window // 2, window - 1 - window // 2), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed

    def _clear_derivative(self) -> None:
        """Remove derivative plot."""
        if self._derivative_line is not None:
            try:
                self._derivative_line.remove()
            except (ValueError, AttributeError):
                pass
            self._derivative_line = None

        if self._derivative_ax is not None:
            try:
                self._derivative_ax.remove()
            except (ValueError, AttributeError):
                pass
            self._derivative_ax = None

        if self._mpl_available and self._canvas is not None:
            self._canvas.draw()


class StreamlineOptionsDialog(QDialog):
    """
    Popup dialog for streamline/pathline visualization options.

    Provides controls for seed geometry, integration direction,
    visualization style, and other streamline parameters.
    """

    # Signal emitted when any option changes
    options_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Streamline Options")
        self.setModal(False)
        self.setMinimumWidth(300)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        # --- Velocity Field Selection ---
        field_group = QGroupBox("Velocity Field")
        field_layout = QFormLayout()
        field_layout.setSpacing(6)
        field_group.setLayout(field_layout)

        self.velocity_field_combo = QComboBox()
        self.velocity_field_combo.addItem("(None)")
        self.velocity_field_combo.setToolTip("Select a 3-component vector field for streamlines")
        self.velocity_field_combo.currentTextChanged.connect(self._emit_changed)
        field_layout.addRow("Field:", self.velocity_field_combo)

        layout.addWidget(field_group)

        # --- Seed Geometry ---
        seed_group = QGroupBox("Seed Points")
        seed_layout = QFormLayout()
        seed_layout.setSpacing(6)
        seed_group.setLayout(seed_layout)

        self.seed_type_combo = QComboBox()
        self.seed_type_combo.addItems(["Sphere", "Line", "Plane", "Point Cloud"])
        self.seed_type_combo.setToolTip("Geometry for generating seed points")
        self.seed_type_combo.currentTextChanged.connect(self._on_seed_type_changed)
        seed_layout.addRow("Type:", self.seed_type_combo)

        self.n_points_spin = QSpinBox()
        self.n_points_spin.setRange(1, 1000)
        self.n_points_spin.setValue(100)
        self.n_points_spin.setToolTip("Number of seed points to generate")
        self.n_points_spin.valueChanged.connect(self._emit_changed)
        seed_layout.addRow("Count:", self.n_points_spin)

        # Sphere-specific controls
        self.sphere_radius_spin = QDoubleSpinBox()
        self.sphere_radius_spin.setRange(0.001, 1e6)
        self.sphere_radius_spin.setValue(1.0)
        self.sphere_radius_spin.setDecimals(3)
        self.sphere_radius_spin.setToolTip("Radius of the seed sphere")
        self.sphere_radius_spin.valueChanged.connect(self._emit_changed)
        seed_layout.addRow("Radius:", self.sphere_radius_spin)

        # Center controls (shared by sphere/line/plane)
        self.center_x_spin = QDoubleSpinBox()
        self.center_x_spin.setRange(-1e6, 1e6)
        self.center_x_spin.setDecimals(3)
        self.center_x_spin.valueChanged.connect(self._emit_changed)
        seed_layout.addRow("Center X:", self.center_x_spin)

        self.center_y_spin = QDoubleSpinBox()
        self.center_y_spin.setRange(-1e6, 1e6)
        self.center_y_spin.setDecimals(3)
        self.center_y_spin.valueChanged.connect(self._emit_changed)
        seed_layout.addRow("Center Y:", self.center_y_spin)

        self.center_z_spin = QDoubleSpinBox()
        self.center_z_spin.setRange(-1e6, 1e6)
        self.center_z_spin.setDecimals(3)
        self.center_z_spin.valueChanged.connect(self._emit_changed)
        seed_layout.addRow("Center Z:", self.center_z_spin)

        # Line-specific: end point
        self.end_x_spin = QDoubleSpinBox()
        self.end_x_spin.setRange(-1e6, 1e6)
        self.end_x_spin.setDecimals(3)
        self.end_x_spin.valueChanged.connect(self._emit_changed)
        self.end_x_label = QLabel("End X:")
        seed_layout.addRow(self.end_x_label, self.end_x_spin)

        self.end_y_spin = QDoubleSpinBox()
        self.end_y_spin.setRange(-1e6, 1e6)
        self.end_y_spin.setDecimals(3)
        self.end_y_spin.valueChanged.connect(self._emit_changed)
        self.end_y_label = QLabel("End Y:")
        seed_layout.addRow(self.end_y_label, self.end_y_spin)

        self.end_z_spin = QDoubleSpinBox()
        self.end_z_spin.setRange(-1e6, 1e6)
        self.end_z_spin.setDecimals(3)
        self.end_z_spin.valueChanged.connect(self._emit_changed)
        self.end_z_label = QLabel("End Z:")
        seed_layout.addRow(self.end_z_label, self.end_z_spin)

        # Plane-specific: normal direction
        self.plane_normal_combo = QComboBox()
        self.plane_normal_combo.addItems(["X", "Y", "Z"])
        self.plane_normal_combo.setToolTip("Normal direction for seed plane")
        self.plane_normal_combo.currentTextChanged.connect(self._emit_changed)
        self.plane_normal_label = QLabel("Normal:")
        seed_layout.addRow(self.plane_normal_label, self.plane_normal_combo)

        self.plane_size_spin = QDoubleSpinBox()
        self.plane_size_spin.setRange(0.001, 1e6)
        self.plane_size_spin.setValue(1.0)
        self.plane_size_spin.setDecimals(3)
        self.plane_size_spin.setToolTip("Size of the seed plane")
        self.plane_size_spin.valueChanged.connect(self._emit_changed)
        self.plane_size_label = QLabel("Size:")
        seed_layout.addRow(self.plane_size_label, self.plane_size_spin)

        # Use mesh surface checkbox
        self.use_surface_cb = QCheckBox("Use mesh surface")
        self.use_surface_cb.setToolTip("Generate seed points on the mesh surface (Point Cloud mode)")
        self.use_surface_cb.stateChanged.connect(self._emit_changed)
        seed_layout.addRow("", self.use_surface_cb)

        layout.addWidget(seed_group)

        # --- Integration Settings ---
        integration_group = QGroupBox("Integration")
        integration_layout = QFormLayout()
        integration_layout.setSpacing(6)
        integration_group.setLayout(integration_layout)

        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["Both", "Forward", "Backward"])
        self.direction_combo.setToolTip("Integration direction along velocity field")
        self.direction_combo.currentTextChanged.connect(self._emit_changed)
        integration_layout.addRow("Direction:", self.direction_combo)

        self.max_time_spin = QDoubleSpinBox()
        self.max_time_spin.setRange(0.1, 10000.0)
        self.max_time_spin.setValue(100.0)
        self.max_time_spin.setDecimals(1)
        self.max_time_spin.setToolTip("Maximum integration time/length")
        self.max_time_spin.valueChanged.connect(self._emit_changed)
        integration_layout.addRow("Max Time:", self.max_time_spin)

        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(100, 100000)
        self.max_steps_spin.setValue(2000)
        self.max_steps_spin.setToolTip("Maximum integration steps")
        self.max_steps_spin.valueChanged.connect(self._emit_changed)
        integration_layout.addRow("Max Steps:", self.max_steps_spin)

        self.step_length_spin = QDoubleSpinBox()
        self.step_length_spin.setRange(0.0001, 10.0)
        self.step_length_spin.setValue(0.5)
        self.step_length_spin.setDecimals(4)
        self.step_length_spin.setToolTip("Initial step length for integration")
        self.step_length_spin.valueChanged.connect(self._emit_changed)
        integration_layout.addRow("Step Length:", self.step_length_spin)

        layout.addWidget(integration_group)

        # --- Visualization Style ---
        style_group = QGroupBox("Visualization")
        style_layout = QFormLayout()
        style_layout.setSpacing(6)
        style_group.setLayout(style_layout)

        self.style_combo = QComboBox()
        self.style_combo.addItems(["Lines", "Tubes", "Ribbons"])
        self.style_combo.setToolTip("Streamline representation style")
        self.style_combo.currentTextChanged.connect(self._on_style_changed)
        style_layout.addRow("Style:", self.style_combo)

        self.line_width_spin = QDoubleSpinBox()
        self.line_width_spin.setRange(0.5, 10.0)
        self.line_width_spin.setValue(2.0)
        self.line_width_spin.setDecimals(1)
        self.line_width_spin.setToolTip("Line width for streamlines")
        self.line_width_spin.valueChanged.connect(self._emit_changed)
        self.line_width_label = QLabel("Line Width:")
        style_layout.addRow(self.line_width_label, self.line_width_spin)

        self.tube_radius_spin = QDoubleSpinBox()
        self.tube_radius_spin.setRange(0.001, 100.0)
        self.tube_radius_spin.setValue(0.1)
        self.tube_radius_spin.setDecimals(3)
        self.tube_radius_spin.setToolTip("Tube radius for streamlines")
        self.tube_radius_spin.valueChanged.connect(self._emit_changed)
        self.tube_radius_label = QLabel("Tube Radius:")
        style_layout.addRow(self.tube_radius_label, self.tube_radius_spin)

        self.tube_sides_spin = QSpinBox()
        self.tube_sides_spin.setRange(3, 24)
        self.tube_sides_spin.setValue(8)
        self.tube_sides_spin.setToolTip("Number of sides for tube cross-section")
        self.tube_sides_spin.valueChanged.connect(self._emit_changed)
        self.tube_sides_label = QLabel("Tube Sides:")
        style_layout.addRow(self.tube_sides_label, self.tube_sides_spin)

        # Color options
        self.color_by_combo = QComboBox()
        self.color_by_combo.addItems(["Velocity Magnitude", "Integration Time", "Solid Color"])
        self.color_by_combo.setToolTip("Color streamlines by this quantity")
        self.color_by_combo.currentTextChanged.connect(self._emit_changed)
        style_layout.addRow("Color By:", self.color_by_combo)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems([
            "coolwarm", "viridis", "plasma", "magma", "inferno",
            "jet", "rainbow", "Spectral", "RdBu",
        ])
        self.cmap_combo.setToolTip("Colormap for streamlines")
        self.cmap_combo.currentTextChanged.connect(self._emit_changed)
        style_layout.addRow("Colormap:", self.cmap_combo)

        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0.0, 1.0)
        self.opacity_spin.setValue(1.0)
        self.opacity_spin.setDecimals(2)
        self.opacity_spin.setSingleStep(0.1)
        self.opacity_spin.setToolTip("Streamline opacity")
        self.opacity_spin.valueChanged.connect(self._emit_changed)
        style_layout.addRow("Opacity:", self.opacity_spin)

        layout.addWidget(style_group)

        # --- Action Buttons ---
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)

        self.generate_btn = QPushButton("Generate Streamlines")
        self.generate_btn.setToolTip("Generate streamlines with current settings")
        button_layout.addWidget(self.generate_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setToolTip("Remove streamlines from visualization")
        button_layout.addWidget(self.clear_btn)

        layout.addLayout(button_layout)

        # Initialize visibility
        self._on_seed_type_changed(self.seed_type_combo.currentText())
        self._on_style_changed(self.style_combo.currentText())

    def _emit_changed(self) -> None:
        """Emit the options_changed signal."""
        self.options_changed.emit()

    def _on_seed_type_changed(self, seed_type: str) -> None:
        """Update UI visibility based on seed type."""
        is_sphere = seed_type == "Sphere"
        is_line = seed_type == "Line"
        is_plane = seed_type == "Plane"
        is_point_cloud = seed_type == "Point Cloud"

        # Sphere radius
        self.sphere_radius_spin.setVisible(is_sphere)
        # Find the label in the form layout and hide it too
        for i in range(self.sphere_radius_spin.parent().layout().rowCount()):
            item = self.sphere_radius_spin.parent().layout().itemAt(i, QFormLayout.LabelRole)
            if item and item.widget():
                text = item.widget().text() if hasattr(item.widget(), 'text') else ""
                if "Radius" in text and not "Tube" in text:
                    item.widget().setVisible(is_sphere)
                    break

        # Line end point controls
        self.end_x_spin.setVisible(is_line)
        self.end_x_label.setVisible(is_line)
        self.end_y_spin.setVisible(is_line)
        self.end_y_label.setVisible(is_line)
        self.end_z_spin.setVisible(is_line)
        self.end_z_label.setVisible(is_line)

        # Plane controls
        self.plane_normal_combo.setVisible(is_plane)
        self.plane_normal_label.setVisible(is_plane)
        self.plane_size_spin.setVisible(is_plane)
        self.plane_size_label.setVisible(is_plane)

        # Point cloud uses surface
        self.use_surface_cb.setVisible(is_point_cloud)

        # Center is visible for sphere, line start, plane
        center_visible = is_sphere or is_line or is_plane
        self.center_x_spin.setVisible(center_visible)
        self.center_y_spin.setVisible(center_visible)
        self.center_z_spin.setVisible(center_visible)

        self._emit_changed()

    def _on_style_changed(self, style: str) -> None:
        """Update UI visibility based on visualization style."""
        is_lines = style == "Lines"
        is_tubes = style == "Tubes"
        is_ribbons = style == "Ribbons"

        self.line_width_spin.setVisible(is_lines)
        self.line_width_label.setVisible(is_lines)

        self.tube_radius_spin.setVisible(is_tubes or is_ribbons)
        self.tube_radius_label.setVisible(is_tubes or is_ribbons)
        self.tube_sides_spin.setVisible(is_tubes)
        self.tube_sides_label.setVisible(is_tubes)

        if is_ribbons:
            self.tube_radius_label.setText("Width:")
        else:
            self.tube_radius_label.setText("Tube Radius:")

        self._emit_changed()

    def set_mesh_center(self, center: tuple) -> None:
        """Set the seed center to the mesh center."""
        self.center_x_spin.blockSignals(True)
        self.center_y_spin.blockSignals(True)
        self.center_z_spin.blockSignals(True)
        self.center_x_spin.setValue(center[0])
        self.center_y_spin.setValue(center[1])
        self.center_z_spin.setValue(center[2])
        self.center_x_spin.blockSignals(False)
        self.center_y_spin.blockSignals(False)
        self.center_z_spin.blockSignals(False)

    def set_mesh_bounds(self, bounds: tuple) -> None:
        """Set reasonable defaults based on mesh bounds."""
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        diag = ((xmax - xmin)**2 + (ymax - ymin)**2 + (zmax - zmin)**2)**0.5

        # Set sphere radius to ~10% of diagonal
        self.sphere_radius_spin.blockSignals(True)
        self.sphere_radius_spin.setValue(diag * 0.1)
        self.sphere_radius_spin.blockSignals(False)

        # Set plane size similarly
        self.plane_size_spin.blockSignals(True)
        self.plane_size_spin.setValue(diag * 0.2)
        self.plane_size_spin.blockSignals(False)

        # Set tube radius to small fraction
        self.tube_radius_spin.blockSignals(True)
        self.tube_radius_spin.setValue(diag * 0.005)
        self.tube_radius_spin.blockSignals(False)

        # Set line end point offset from center
        cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
        self.end_x_spin.blockSignals(True)
        self.end_y_spin.blockSignals(True)
        self.end_z_spin.blockSignals(True)
        self.end_x_spin.setValue(cx + diag * 0.2)
        self.end_y_spin.setValue(cy)
        self.end_z_spin.setValue(cz)
        self.end_x_spin.blockSignals(False)
        self.end_y_spin.blockSignals(False)
        self.end_z_spin.blockSignals(False)

    def populate_velocity_fields(self, field_names: list) -> None:
        """Populate the velocity field combo with available vector fields."""
        self.velocity_field_combo.blockSignals(True)
        self.velocity_field_combo.clear()
        self.velocity_field_combo.addItem("(None)")
        for name in field_names:
            self.velocity_field_combo.addItem(name)
        self.velocity_field_combo.blockSignals(False)

    def get_options(self) -> dict:
        """Return current options as a dictionary."""
        return {
            "velocity_field": self.velocity_field_combo.currentText(),
            "seed_type": self.seed_type_combo.currentText(),
            "n_points": self.n_points_spin.value(),
            "center": (
                self.center_x_spin.value(),
                self.center_y_spin.value(),
                self.center_z_spin.value(),
            ),
            "sphere_radius": self.sphere_radius_spin.value(),
            "line_end": (
                self.end_x_spin.value(),
                self.end_y_spin.value(),
                self.end_z_spin.value(),
            ),
            "plane_normal": self.plane_normal_combo.currentText(),
            "plane_size": self.plane_size_spin.value(),
            "use_surface": self.use_surface_cb.isChecked(),
            "direction": self.direction_combo.currentText(),
            "max_time": self.max_time_spin.value(),
            "max_steps": self.max_steps_spin.value(),
            "step_length": self.step_length_spin.value(),
            "style": self.style_combo.currentText(),
            "line_width": self.line_width_spin.value(),
            "tube_radius": self.tube_radius_spin.value(),
            "tube_sides": self.tube_sides_spin.value(),
            "color_by": self.color_by_combo.currentText(),
            "cmap": self.cmap_combo.currentText(),
            "opacity": self.opacity_spin.value(),
        }


class FieldCalculatorPanel(QWidget):
    """
    Panel widget for computing derived fields from existing mesh data.

    Provides preset operations (magnitude, gradient, vorticity) and
    a custom expression evaluator for computing new scalar/vector fields.
    """

    # Signal emitted when a new field is computed
    field_computed = Signal(str)  # field_name

    # Available preset operations
    PRESET_OPERATIONS = {
        "Magnitude": {
            "description": "Compute magnitude of a vector field",
            "input_type": "vector",
            "output_type": "scalar",
        },
        "Gradient": {
            "description": "Compute gradient of a scalar field",
            "input_type": "scalar",
            "output_type": "vector",
        },
        "Gradient Magnitude": {
            "description": "Compute magnitude of gradient",
            "input_type": "scalar",
            "output_type": "scalar",
        },
        "Vorticity": {
            "description": "Compute vorticity (curl) of velocity field",
            "input_type": "vector",
            "output_type": "vector",
        },
        "Vorticity Magnitude": {
            "description": "Compute magnitude of vorticity",
            "input_type": "vector",
            "output_type": "scalar",
        },
        "Divergence": {
            "description": "Compute divergence of a vector field",
            "input_type": "vector",
            "output_type": "scalar",
        },
        "Normalize": {
            "description": "Normalize vector field to unit vectors",
            "input_type": "vector",
            "output_type": "vector",
        },
        "Component X": {
            "description": "Extract X component of vector field",
            "input_type": "vector",
            "output_type": "scalar",
        },
        "Component Y": {
            "description": "Extract Y component of vector field",
            "input_type": "vector",
            "output_type": "scalar",
        },
        "Component Z": {
            "description": "Extract Z component of vector field",
            "input_type": "vector",
            "output_type": "scalar",
        },
    }

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._mesh = None
        self._pv = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # --- Preset Operations ---
        preset_group = QGroupBox("Preset Operations")
        preset_layout = QVBoxLayout()
        preset_layout.setContentsMargins(6, 6, 6, 6)
        preset_layout.setSpacing(4)
        preset_group.setLayout(preset_layout)

        # Operation row
        op_layout = QHBoxLayout()
        op_layout.setSpacing(4)
        op_layout.addWidget(QLabel("Op:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(self.PRESET_OPERATIONS.keys()))
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        self.preset_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        op_layout.addWidget(self.preset_combo, 1)
        preset_layout.addLayout(op_layout)

        # Input field row
        input_layout = QHBoxLayout()
        input_layout.setSpacing(4)
        input_layout.addWidget(QLabel("In:"))
        self.preset_input_combo = QComboBox()
        self.preset_input_combo.setToolTip("Select input field for operation")
        self.preset_input_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        input_layout.addWidget(self.preset_input_combo, 1)
        preset_layout.addLayout(input_layout)

        # Output name row
        out_layout = QHBoxLayout()
        out_layout.setSpacing(4)
        out_layout.addWidget(QLabel("Out:"))
        self.preset_output_edit = QLineEdit()
        self.preset_output_edit.setPlaceholderText("output_name")
        self.preset_output_edit.setToolTip("Name for the computed field")
        self.preset_output_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        out_layout.addWidget(self.preset_output_edit, 1)
        preset_layout.addLayout(out_layout)

        self.preset_desc_label = QLabel()
        self.preset_desc_label.setWordWrap(True)
        self.preset_desc_label.setStyleSheet("font-size: 9px; color: #888;")
        preset_layout.addWidget(self.preset_desc_label)

        self.compute_preset_btn = QPushButton("Compute")
        self.compute_preset_btn.setToolTip("Compute the derived field")
        self.compute_preset_btn.clicked.connect(self._compute_preset)
        preset_layout.addWidget(self.compute_preset_btn)

        layout.addWidget(preset_group)

        # --- Custom Expression ---
        expr_group = QGroupBox("Custom Expression")
        expr_layout = QVBoxLayout()
        expr_layout.setContentsMargins(6, 6, 6, 6)
        expr_layout.setSpacing(4)
        expr_group.setLayout(expr_layout)

        expr_help = QLabel(
            "NumPy expression using field names:\n"
            "  np.sqrt(u**2 + v**2)\n"
            "  pressure / 1000"
        )
        expr_help.setWordWrap(True)
        expr_help.setStyleSheet("font-size: 9px; color: #888;")
        expr_layout.addWidget(expr_help)

        self.expr_edit = QLineEdit()
        self.expr_edit.setPlaceholderText("np.sqrt(vx**2 + vy**2)")
        self.expr_edit.setToolTip("NumPy expression using field names as variables")
        expr_layout.addWidget(self.expr_edit)

        expr_name_layout = QHBoxLayout()
        expr_name_layout.setSpacing(4)
        expr_name_layout.addWidget(QLabel("Out:"))
        self.expr_output_edit = QLineEdit()
        self.expr_output_edit.setPlaceholderText("new_field")
        self.expr_output_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        expr_name_layout.addWidget(self.expr_output_edit, 1)
        expr_layout.addLayout(expr_name_layout)

        self.compute_expr_btn = QPushButton("Evaluate")
        self.compute_expr_btn.clicked.connect(self._compute_expression)
        expr_layout.addWidget(self.compute_expr_btn)

        layout.addWidget(expr_group)

        # --- Computed Fields List ---
        fields_group = QGroupBox("Computed Fields")
        fields_layout = QVBoxLayout()
        fields_layout.setContentsMargins(6, 6, 6, 6)
        fields_layout.setSpacing(4)
        fields_group.setLayout(fields_layout)

        self.fields_list = QComboBox()
        self.fields_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.fields_list.setToolTip("Select a computed field to view or delete")
        fields_layout.addWidget(self.fields_list)

        fields_btn_layout = QHBoxLayout()
        fields_btn_layout.setSpacing(4)

        self.view_field_btn = QPushButton("View")
        self.view_field_btn.setToolTip("Set as active scalar field")
        self.view_field_btn.clicked.connect(self._view_selected_field)
        fields_btn_layout.addWidget(self.view_field_btn)

        self.delete_field_btn = QPushButton("Delete")
        self.delete_field_btn.setToolTip("Remove computed field")
        self.delete_field_btn.clicked.connect(self._delete_selected_field)
        fields_btn_layout.addWidget(self.delete_field_btn)

        fields_layout.addLayout(fields_btn_layout)

        layout.addWidget(fields_group)

        # --- Status ---
        self.status_label = QLabel("Load a solution to compute fields")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 10px; color: #888;")
        layout.addWidget(self.status_label)

        layout.addStretch(1)

        # Initialize preset description
        self._on_preset_changed(self.preset_combo.currentText())

    def set_mesh(self, mesh, pv_module) -> None:
        """Set the current mesh and populate field combos."""
        self._mesh = mesh
        self._pv = pv_module
        self._populate_field_combos()

    def _populate_field_combos(self) -> None:
        """Populate input field combos based on current mesh."""
        self.preset_input_combo.clear()

        if self._mesh is None:
            self.status_label.setText("No mesh loaded")
            return

        # Get scalar and vector fields
        scalar_fields = []
        vector_fields = []

        for name in self._mesh.point_data.keys():
            arr = self._mesh.point_data[name]
            if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
                scalar_fields.append(name)
            elif arr.ndim == 2 and arr.shape[1] == 3:
                vector_fields.append(name)

        for name in self._mesh.cell_data.keys():
            arr = self._mesh.cell_data[name]
            if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
                if name not in scalar_fields:
                    scalar_fields.append(name)
            elif arr.ndim == 2 and arr.shape[1] == 3:
                if name not in vector_fields:
                    vector_fields.append(name)

        # Store for filtering
        self._scalar_fields = sorted(scalar_fields)
        self._vector_fields = sorted(vector_fields)

        # Update combo based on current preset
        self._on_preset_changed(self.preset_combo.currentText())

        self.status_label.setText(
            f"Available: {len(scalar_fields)} scalar, {len(vector_fields)} vector fields"
        )

    def _on_preset_changed(self, preset_name: str) -> None:
        """Update UI when preset operation changes."""
        if preset_name not in self.PRESET_OPERATIONS:
            return

        op_info = self.PRESET_OPERATIONS[preset_name]
        self.preset_desc_label.setText(op_info["description"])

        # Update input combo based on required input type
        self.preset_input_combo.clear()
        input_type = op_info["input_type"]

        if input_type == "scalar":
            fields = getattr(self, "_scalar_fields", [])
        else:  # vector
            fields = getattr(self, "_vector_fields", [])

        self.preset_input_combo.addItems(fields)

        # Suggest output name
        if self.preset_input_combo.count() > 0:
            input_name = self.preset_input_combo.currentText()
            suggested_name = self._suggest_output_name(preset_name, input_name)
            self.preset_output_edit.setText(suggested_name)

    def _suggest_output_name(self, operation: str, input_name: str) -> str:
        """Generate a suggested output field name."""
        op_suffix = {
            "Magnitude": "_magnitude",
            "Gradient": "_gradient",
            "Gradient Magnitude": "_grad_mag",
            "Vorticity": "_vorticity",
            "Vorticity Magnitude": "_vort_mag",
            "Divergence": "_divergence",
            "Normalize": "_normalized",
            "Component X": "_x",
            "Component Y": "_y",
            "Component Z": "_z",
        }
        suffix = op_suffix.get(operation, "_computed")
        return f"{input_name}{suffix}"

    def _compute_preset(self) -> None:
        """Compute a preset derived field."""
        if self._mesh is None or self._pv is None:
            QMessageBox.warning(self, "No Data", "Please load a solution first.")
            return

        preset = self.preset_combo.currentText()
        input_field = self.preset_input_combo.currentText()
        output_name = self.preset_output_edit.text().strip()

        if not input_field:
            QMessageBox.warning(self, "No Input", "Please select an input field.")
            return

        if not output_name:
            QMessageBox.warning(self, "No Name", "Please enter an output field name.")
            return

        # Check if output name already exists
        if output_name in self._mesh.point_data or output_name in self._mesh.cell_data:
            reply = QMessageBox.question(
                self, "Overwrite?",
                f"Field '{output_name}' already exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        try:
            import numpy as np

            # Get input array
            if input_field in self._mesh.point_data:
                arr = self._mesh.point_data[input_field]
                is_point_data = True
            elif input_field in self._mesh.cell_data:
                arr = self._mesh.cell_data[input_field]
                is_point_data = False
            else:
                QMessageBox.warning(self, "Not Found", f"Field '{input_field}' not found.")
                return

            # Compute based on operation
            result = None

            if preset == "Magnitude":
                if arr.ndim == 2 and arr.shape[1] == 3:
                    result = np.linalg.norm(arr, axis=1)
                else:
                    QMessageBox.warning(self, "Invalid Input", "Magnitude requires a 3-component vector field.")
                    return

            elif preset == "Component X":
                if arr.ndim == 2 and arr.shape[1] >= 1:
                    result = arr[:, 0]
                else:
                    QMessageBox.warning(self, "Invalid Input", "Component extraction requires a vector field.")
                    return

            elif preset == "Component Y":
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    result = arr[:, 1]
                else:
                    QMessageBox.warning(self, "Invalid Input", "Component extraction requires a vector field.")
                    return

            elif preset == "Component Z":
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    result = arr[:, 2]
                else:
                    QMessageBox.warning(self, "Invalid Input", "Component extraction requires a vector field.")
                    return

            elif preset == "Normalize":
                if arr.ndim == 2 and arr.shape[1] == 3:
                    mag = np.linalg.norm(arr, axis=1, keepdims=True)
                    mag[mag == 0] = 1  # Avoid division by zero
                    result = arr / mag
                else:
                    QMessageBox.warning(self, "Invalid Input", "Normalize requires a 3-component vector field.")
                    return

            elif preset == "Gradient":
                result = self._compute_gradient(input_field, is_point_data)
                if result is None:
                    return

            elif preset == "Gradient Magnitude":
                grad = self._compute_gradient(input_field, is_point_data)
                if grad is None:
                    return
                result = np.linalg.norm(grad, axis=1)

            elif preset == "Vorticity":
                result = self._compute_vorticity(input_field, is_point_data)
                if result is None:
                    return

            elif preset == "Vorticity Magnitude":
                vort = self._compute_vorticity(input_field, is_point_data)
                if vort is None:
                    return
                result = np.linalg.norm(vort, axis=1)

            elif preset == "Divergence":
                result = self._compute_divergence(input_field, is_point_data)
                if result is None:
                    return

            # Add result to mesh
            if result is not None:
                if is_point_data:
                    self._mesh.point_data[output_name] = result
                else:
                    self._mesh.cell_data[output_name] = result

                self._update_fields_list()
                self._populate_field_combos()  # Refresh available fields
                self.status_label.setText(f"Computed: {output_name}")
                self.field_computed.emit(output_name)

        except Exception as e:
            QMessageBox.critical(self, "Computation Error", f"Failed to compute field:\n{e}")

    def _compute_gradient(self, field_name: str, is_point_data: bool):
        """Compute gradient of a scalar field using PyVista."""
        try:
            # Use PyVista's gradient computation
            grad_mesh = self._mesh.compute_derivative(
                scalars=field_name,
                gradient=True,
                qcriterion=False,
                vorticity=False,
                divergence=False,
            )
            gradient_name = f"gradient"
            if gradient_name in grad_mesh.point_data:
                return grad_mesh.point_data[gradient_name]
            elif gradient_name in grad_mesh.cell_data:
                return grad_mesh.cell_data[gradient_name]
            else:
                # Try alternative name
                for name in grad_mesh.point_data.keys():
                    if "gradient" in name.lower():
                        return grad_mesh.point_data[name]
                QMessageBox.warning(self, "Gradient Error", "Could not find gradient in computed result.")
                return None
        except Exception as e:
            QMessageBox.warning(self, "Gradient Error", f"Failed to compute gradient:\n{e}")
            return None

    def _compute_vorticity(self, field_name: str, is_point_data: bool):
        """Compute vorticity (curl) of a vector field using PyVista."""
        try:
            vort_mesh = self._mesh.compute_derivative(
                scalars=field_name,
                gradient=False,
                qcriterion=False,
                vorticity=True,
                divergence=False,
            )
            vorticity_name = "vorticity"
            if vorticity_name in vort_mesh.point_data:
                return vort_mesh.point_data[vorticity_name]
            elif vorticity_name in vort_mesh.cell_data:
                return vort_mesh.cell_data[vorticity_name]
            else:
                for name in vort_mesh.point_data.keys():
                    if "vorticity" in name.lower():
                        return vort_mesh.point_data[name]
                QMessageBox.warning(self, "Vorticity Error", "Could not find vorticity in computed result.")
                return None
        except Exception as e:
            QMessageBox.warning(self, "Vorticity Error", f"Failed to compute vorticity:\n{e}")
            return None

    def _compute_divergence(self, field_name: str, is_point_data: bool):
        """Compute divergence of a vector field using PyVista."""
        try:
            div_mesh = self._mesh.compute_derivative(
                scalars=field_name,
                gradient=False,
                qcriterion=False,
                vorticity=False,
                divergence=True,
            )
            divergence_name = "divergence"
            if divergence_name in div_mesh.point_data:
                return div_mesh.point_data[divergence_name]
            elif divergence_name in div_mesh.cell_data:
                return div_mesh.cell_data[divergence_name]
            else:
                for name in div_mesh.point_data.keys():
                    if "divergence" in name.lower():
                        return div_mesh.point_data[name]
                QMessageBox.warning(self, "Divergence Error", "Could not find divergence in computed result.")
                return None
        except Exception as e:
            QMessageBox.warning(self, "Divergence Error", f"Failed to compute divergence:\n{e}")
            return None

    def _compute_expression(self) -> None:
        """Evaluate a custom NumPy expression."""
        if self._mesh is None:
            QMessageBox.warning(self, "No Data", "Please load a solution first.")
            return

        expr = self.expr_edit.text().strip()
        output_name = self.expr_output_edit.text().strip()

        if not expr:
            QMessageBox.warning(self, "No Expression", "Please enter an expression.")
            return

        if not output_name:
            QMessageBox.warning(self, "No Name", "Please enter an output field name.")
            return

        try:
            import numpy as np

            # Build namespace with all fields as variables
            namespace = {"np": np, "numpy": np}

            # Add point data fields
            for name in self._mesh.point_data.keys():
                # Make valid Python identifier
                safe_name = name.replace(" ", "_").replace("-", "_")
                namespace[safe_name] = self._mesh.point_data[name]
                # Also add original name if different
                if safe_name != name:
                    namespace[name] = self._mesh.point_data[name]

            # Add cell data fields
            for name in self._mesh.cell_data.keys():
                safe_name = name.replace(" ", "_").replace("-", "_")
                if safe_name not in namespace:
                    namespace[safe_name] = self._mesh.cell_data[name]
                if name not in namespace:
                    namespace[name] = self._mesh.cell_data[name]

            # Evaluate expression
            result = eval(expr, {"__builtins__": {}}, namespace)

            if result is None:
                QMessageBox.warning(self, "Invalid Result", "Expression returned None.")
                return

            # Convert to numpy array
            result = np.asarray(result)

            # Determine if point or cell data based on shape
            n_points = self._mesh.n_points
            n_cells = self._mesh.n_cells

            if result.shape[0] == n_points:
                self._mesh.point_data[output_name] = result
            elif result.shape[0] == n_cells:
                self._mesh.cell_data[output_name] = result
            else:
                QMessageBox.warning(
                    self, "Shape Mismatch",
                    f"Result has {result.shape[0]} values, but mesh has "
                    f"{n_points} points and {n_cells} cells."
                )
                return

            self._update_fields_list()
            self._populate_field_combos()
            self.status_label.setText(f"Computed: {output_name}")
            self.field_computed.emit(output_name)

        except SyntaxError as e:
            QMessageBox.critical(self, "Syntax Error", f"Invalid expression syntax:\n{e}")
        except NameError as e:
            QMessageBox.critical(self, "Name Error", f"Unknown field name in expression:\n{e}")
        except Exception as e:
            QMessageBox.critical(self, "Expression Error", f"Failed to evaluate expression:\n{e}")

    def _update_fields_list(self) -> None:
        """Update the list of computed fields."""
        self.fields_list.clear()
        if self._mesh is None:
            return

        # Find fields that were computed (those with common suffixes)
        computed_suffixes = [
            "_magnitude", "_gradient", "_grad_mag", "_vorticity",
            "_vort_mag", "_divergence", "_normalized", "_x", "_y", "_z", "_computed"
        ]

        computed_fields = []
        for name in list(self._mesh.point_data.keys()) + list(self._mesh.cell_data.keys()):
            for suffix in computed_suffixes:
                if name.endswith(suffix):
                    if name not in computed_fields:
                        computed_fields.append(name)
                    break

        self.fields_list.addItems(sorted(computed_fields))

    def _view_selected_field(self) -> None:
        """Emit signal to view the selected field."""
        field_name = self.fields_list.currentText()
        if field_name:
            self.field_computed.emit(field_name)

    def _delete_selected_field(self) -> None:
        """Delete the selected computed field."""
        field_name = self.fields_list.currentText()
        if not field_name:
            return

        if self._mesh is None:
            return

        reply = QMessageBox.question(
            self, "Delete Field?",
            f"Delete computed field '{field_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # Remove from mesh
        if field_name in self._mesh.point_data:
            del self._mesh.point_data[field_name]
        if field_name in self._mesh.cell_data:
            del self._mesh.cell_data[field_name]

        self._update_fields_list()
        self._populate_field_combos()
        self.status_label.setText(f"Deleted: {field_name}")


class SolutionInspectorWidget(QWidget):
    """
    Dockable widget for inspecting 0D/ROM solution time series.

    Provides an embedded 3D viewport and a side panel for loading a
    ``.pvd`` file, selecting time steps, and adjusting scalar display
    options (field, range, and label).

    Features:
    - Animation playback with speed control
    - Mesh display options (opacity, representation, edges)
    - Data probing and statistics
    - Slicing and threshold filtering
    - Camera presets
    - Vector field visualization (glyphs)
    - Streamline/pathline visualization for velocity fields
    - Field calculator for derived quantities (magnitude, gradient, vorticity, custom expressions)
    - Export capabilities
    """

    # Signals
    probed_value_changed = Signal(float, float, float, float)  # x, y, z, value

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pv = None
        self._reader = None
        self._current_mesh = None
        self._current_scalar_name: Optional[str] = None
        self._time_values: Sequence[float] = []
        self._current_time_index: int = 0
        self._current_cmap: str = "coolwarm"
        self._global_ranges = {}
        self._colorbar_visible: bool = True
        self._background_options_dialog: Optional[BackgroundOptionsDialog] = None
        self._background_settings = {
            "mode": "Gradient",
            "color1": CADTheme.get_color("viewport", "background-bottom"),
            "color2": CADTheme.get_color("viewport", "background-top"),
            "gradient_mode": "Vertical",
            "dither": True,
            "opacity": 1.0,
            "texture_path": "",
            "texture_interpolate": True,
            "texture_repeat": False,
        }
        self._background_texture = None
        self._background_texture_path = ""

        # Animation state
        self._animation_timer: Optional[QTimer] = None
        self._animation_playing: bool = False
        self._animation_loop: bool = True
        self._animation_fps: float = 10.0

        # Mesh display state
        self._mesh_opacity: float = 1.0
        self._mesh_representation: str = "Surface"
        self._mesh_show_edges: bool = False

        # Probing state
        self._probe_enabled: bool = False
        self._probed_point: Optional[tuple] = None  # Current/last probed point
        self._probed_points: list = []  # List of all probed points [(x, y, z), ...]
        self._probe_actor = None
        self._probe_actors: list = []  # List of marker actors for multiple points
        self._probe_colors: list = [
            'yellow', 'cyan', 'magenta', 'lime', 'orange',
            'pink', 'lightblue', 'lightgreen', 'coral', 'gold'
        ]  # Colors for multiple probe markers
        self._time_series_data: dict = {}  # point_key -> [(time, value), ...]

        # Slicing state
        self._slice_enabled: bool = False
        self._slice_actor = None
        self._slice_widget = None

        # Threshold state
        self._threshold_enabled: bool = False
        self._threshold_actor = None

        # Vector field state
        self._vector_field_name: Optional[str] = None
        self._glyph_actor = None
        self._glyph_scale: float = 1.0

        # Streamline state
        self._streamline_enabled: bool = False
        self._streamline_actor = None
        self._streamline_velocity_field: Optional[str] = None
        self._streamline_seed_type: str = "Sphere"  # Sphere, Line, Plane
        self._streamline_direction: str = "Both"  # Forward, Backward, Both
        self._streamline_n_points: int = 100
        self._streamline_max_time: float = 100.0
        self._streamline_tube_radius: float = 0.0  # 0 = lines, >0 = tubes
        self._streamline_line_width: float = 2.0

        # Field calculator state
        self._derived_fields: dict = {}  # name -> array mapping for computed fields
        self._calculator_visible: bool = False

        self._try_import_pyvista()
        self._build_ui()

    def _record_telemetry(self, exc=None, message: Optional[str] = None, level: str = "error",
                          traceback_str: Optional[str] = None, **tags) -> None:
        """
        Send errors or warnings to telemetry without impacting the UI.

        This mirrors the lightweight helpers used elsewhere in the GUI so
        that any error which results in a popup can also be surfaced to
        Sentry when telemetry is enabled.

        Parameters
        ----------
        exc : Exception, optional
            Exception to capture. If provided, captures as an exception event.
        message : str, optional
            Message to capture. Used when exc is None.
        level : str
            Sentry level ("error", "warning", "info").
        traceback_str : str, optional
            Full traceback string to include as extra context.
        **tags
            Additional tags to attach to the event.
        """
        try:
            if exc is not None:
                try:
                    import sentry_sdk  # type: ignore[import]

                    with sentry_sdk.push_scope() as scope:
                        for key, value in tags.items():
                            scope.set_tag(key, value)
                        if traceback_str:
                            scope.set_extra("full_traceback", traceback_str)
                        sentry_sdk.capture_exception(exc)
                        # Flush to ensure the event is sent before the popup blocks
                        sentry_sdk.flush(timeout=2.0)
                except Exception:
                    capture_exception(exc)
                return
            if message:
                capture_message(message, level=level, **tags)
        except Exception:
            # Telemetry must never break the inspector UI
            pass

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _try_import_pyvista(self) -> None:
        try:
            import pyvista as pv  # type: ignore[import]

            self._pv = pv
        except Exception:
            self._pv = None

    def _build_ui(self) -> None:
        # Outer layout: menu-style toolbar + main content
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        self.setLayout(outer_layout)

        # --- Menu toolbar: File / View / Tools + close button ---
        title_bar = QHBoxLayout()
        title_bar.setContentsMargins(4, 4, 4, 4)
        title_bar.setSpacing(6)

        menu_bar = QMenuBar(self)
        # Keep this dock-local menu bar out of the native macOS menu bar.
        menu_bar.setNativeMenuBar(False)
        self.embedded_menu_bar = menu_bar

        # File menu
        file_menu = menu_bar.addMenu("File")
        open_action = QAction(CADIcons.get_icon('open'), "Open...", self)
        open_action.setStatusTip("Open a time-varying solution (.pvd) file")
        open_action.triggered.connect(self._browse_for_file)
        file_menu.addAction(open_action)
        file_menu.addSeparator()

        screenshot_action = QAction(CADIcons.get_icon('save'), "Screenshot...", self)
        screenshot_action.setStatusTip("Save a screenshot of the current 3D view")
        screenshot_action.triggered.connect(self._take_screenshot)
        file_menu.addAction(screenshot_action)

        file_menu.addSeparator()

        # Export submenu
        export_menu = file_menu.addMenu("Export")
        export_mesh_action = QAction(CADIcons.get_icon('mesh'), "Export Mesh (VTP/VTU)...", self)
        export_mesh_action.triggered.connect(self._export_mesh)
        export_menu.addAction(export_mesh_action)

        export_csv_action = QAction(CADIcons.get_icon('download'), "Export Scalars (CSV)...", self)
        export_csv_action.triggered.connect(self._export_csv)
        export_menu.addAction(export_csv_action)

        export_animation_action = QAction(CADIcons.get_icon('movie'), "Export Animation (GIF)...", self)
        export_animation_action.triggered.connect(self._export_animation)
        export_menu.addAction(export_animation_action)

        export_timeseries_action = QAction(CADIcons.get_icon('chart'), "Export Time Series (CSV)...", self)
        export_timeseries_action.triggered.connect(self._export_time_series)
        export_menu.addAction(export_timeseries_action)

        # View menu
        view_menu = menu_bar.addMenu("View")
        reset_view_action = QAction(CADIcons.get_icon('fit'), "Reset View", self)
        reset_view_action.setStatusTip("Reset camera to show entire solution")
        reset_view_action.triggered.connect(self._reset_camera_view)
        view_menu.addAction(reset_view_action)

        iso_view_action = QAction(CADIcons.get_icon('iso'), "Isometric View", self)
        iso_view_action.setStatusTip("Isometric 3D view")
        iso_view_action.triggered.connect(self._view_iso)
        view_menu.addAction(iso_view_action)

        view_menu.addSeparator()

        # Camera presets popup dialog action
        camera_presets_action = QAction("Camera Presets...", self)
        camera_presets_action.setStatusTip("Open camera presets panel")
        camera_presets_action.triggered.connect(self._show_camera_presets_dialog)
        view_menu.addAction(camera_presets_action)

        background_action = QAction("Background", self)
        background_action.setStatusTip("Open background customization options")
        background_action.triggered.connect(self._show_background_options_dialog)
        view_menu.addAction(background_action)

        view_menu.addSeparator()

        self.scale_bar_action = QAction(CADIcons.get_icon('ruler'), "Scale Bar", self)
        self.scale_bar_action.setCheckable(True)
        self.scale_bar_action.setChecked(True)
        self.scale_bar_action.setStatusTip("Toggle visibility of the scale bar in this viewport")
        self.scale_bar_action.triggered.connect(self._toggle_scale_bar)
        view_menu.addAction(self.scale_bar_action)

        self.color_bar_action = QAction("Color Bar", self)
        self.color_bar_action.setCheckable(True)
        self.color_bar_action.setChecked(True)
        self.color_bar_action.setStatusTip("Toggle visibility of the scalar color bar")
        self.color_bar_action.triggered.connect(self._toggle_color_bar)
        view_menu.addAction(self.color_bar_action)

        colorbar_options_action = QAction("Colorbar Options...", self)
        colorbar_options_action.setStatusTip("Open colorbar customization options")
        colorbar_options_action.triggered.connect(self._show_colorbar_options_dialog)
        view_menu.addAction(colorbar_options_action)

        view_menu.addSeparator()

        self.parallel_proj_action = QAction("Parallel Projection", self)
        self.parallel_proj_action.setCheckable(True)
        self.parallel_proj_action.setChecked(False)
        self.parallel_proj_action.setStatusTip("Toggle between perspective and parallel projection")
        self.parallel_proj_action.triggered.connect(self._toggle_parallel_projection)
        view_menu.addAction(self.parallel_proj_action)

        # Tools menu
        tools_menu = menu_bar.addMenu("Tools")

        self.probe_panel_action = QAction(CADIcons.get_icon('probe'), "Probe Data...", self)
        self.probe_panel_action.setCheckable(True)
        self.probe_panel_action.setChecked(False)
        self.probe_panel_action.setStatusTip("Show/hide probe data panel in side panel")
        self.probe_panel_action.triggered.connect(self._toggle_probe_panel)
        tools_menu.addAction(self.probe_panel_action)

        tools_menu.addSeparator()

        self.slice_action = QAction(CADIcons.get_icon('slice'), "Plane Slice...", self)
        self.slice_action.setCheckable(True)
        self.slice_action.setStatusTip("Add a cutting plane to slice the mesh")
        self.slice_action.triggered.connect(self._toggle_slice)
        tools_menu.addAction(self.slice_action)

        self.threshold_action = QAction(CADIcons.get_icon('filter'), "Threshold Filter...", self)
        self.threshold_action.setCheckable(True)
        self.threshold_action.setStatusTip("Filter mesh by scalar value range")
        self.threshold_action.triggered.connect(self._toggle_threshold)
        tools_menu.addAction(self.threshold_action)

        tools_menu.addSeparator()

        self.streamline_action = QAction(CADIcons.get_icon('vector'), "Streamlines...", self)
        self.streamline_action.setStatusTip("Generate streamlines from velocity field")
        self.streamline_action.triggered.connect(self._show_streamline_dialog)
        tools_menu.addAction(self.streamline_action)

        self.calculator_action = QAction(CADIcons.get_icon('adjustments'), "Field Calculator", self)
        self.calculator_action.setCheckable(True)
        self.calculator_action.setStatusTip("Compute derived fields (magnitude, gradient, vorticity, expressions)")
        self.calculator_action.triggered.connect(self._toggle_calculator)
        tools_menu.addAction(self.calculator_action)

        # Analysis menu
        analysis_menu = menu_bar.addMenu("Analysis")

        self.statistics_action = QAction(CADIcons.get_icon('chart'), "Statistics...", self)
        self.statistics_action.setStatusTip("Show statistics for current scalar field")
        self.statistics_action.triggered.connect(self._show_statistics_dialog)
        analysis_menu.addAction(self.statistics_action)

        self.histogram_action = QAction("Histogram...", self)
        self.histogram_action.setStatusTip("Show histogram of scalar field distribution")
        self.histogram_action.triggered.connect(self._show_histogram_dialog)
        analysis_menu.addAction(self.histogram_action)

        analysis_menu.addSeparator()

        self.line_plot_action = QAction("Line Plot...", self)
        self.line_plot_action.setStatusTip("Sample and plot values along a line")
        self.line_plot_action.triggered.connect(self._show_line_plot_dialog)
        analysis_menu.addAction(self.line_plot_action)

        self.time_series_action = QAction("Time Series...", self)
        self.time_series_action.setStatusTip("Plot values over time at a point")
        self.time_series_action.triggered.connect(self._show_time_series_analysis)
        analysis_menu.addAction(self.time_series_action)

        analysis_menu.addSeparator()

        self.compare_fields_action = QAction("Compare Fields...", self)
        self.compare_fields_action.setStatusTip("Compare two scalar fields side by side")
        self.compare_fields_action.triggered.connect(self._show_compare_fields_dialog)
        analysis_menu.addAction(self.compare_fields_action)

        self.field_difference_action = QAction("Field Difference...", self)
        self.field_difference_action.setStatusTip("Compute difference between two fields or timesteps")
        self.field_difference_action.triggered.connect(self._compute_field_difference)
        analysis_menu.addAction(self.field_difference_action)

        # Animation menu
        animation_menu = menu_bar.addMenu("Animation")

        self.anim_play_action = QAction(CADIcons.get_icon('play'), "Play/Pause", self)
        self.anim_play_action.setStatusTip("Play or pause the animation")
        self.anim_play_action.triggered.connect(self._toggle_animation)
        animation_menu.addAction(self.anim_play_action)

        animation_menu.addSeparator()

        self.anim_first_action = QAction(CADIcons.get_icon('skip_prev'), "Go to First Frame", self)
        self.anim_first_action.setStatusTip("Jump to the first time step")
        self.anim_first_action.triggered.connect(self._go_to_first_frame)
        animation_menu.addAction(self.anim_first_action)

        self.anim_prev_action = QAction(CADIcons.get_icon('prev'), "Step Backward", self)
        self.anim_prev_action.setStatusTip("Go to previous time step")
        self.anim_prev_action.triggered.connect(self._step_backward)
        animation_menu.addAction(self.anim_prev_action)

        self.anim_next_action = QAction(CADIcons.get_icon('next'), "Step Forward", self)
        self.anim_next_action.setStatusTip("Go to next time step")
        self.anim_next_action.triggered.connect(self._step_forward)
        animation_menu.addAction(self.anim_next_action)

        self.anim_last_action = QAction(CADIcons.get_icon('skip_next'), "Go to Last Frame", self)
        self.anim_last_action.setStatusTip("Jump to the last time step")
        self.anim_last_action.triggered.connect(self._go_to_last_frame)
        animation_menu.addAction(self.anim_last_action)

        animation_menu.addSeparator()

        self.anim_loop_action = QAction("Loop Animation", self)
        self.anim_loop_action.setCheckable(True)
        self.anim_loop_action.setChecked(True)
        self.anim_loop_action.setStatusTip("Toggle animation looping")
        self.anim_loop_action.triggered.connect(self._toggle_animation_loop)
        animation_menu.addAction(self.anim_loop_action)

        # Playback speed submenu
        speed_menu = animation_menu.addMenu("Playback Speed")
        self._speed_actions = []
        for speed, label in [(0.25, "0.25x"), (0.5, "0.5x"), (1.0, "1x (Normal)"), (2.0, "2x"), (4.0, "4x")]:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(speed == 1.0)
            action.triggered.connect(lambda checked, s=speed: self._set_playback_speed(s))
            speed_menu.addAction(action)
            self._speed_actions.append((speed, action))

        animation_menu.addSeparator()

        self.keyframe_action = QAction("Keyframe Animation...", self)
        self.keyframe_action.setStatusTip("Create custom camera animation with keyframes")
        self.keyframe_action.triggered.connect(self._show_keyframe_dialog)
        animation_menu.addAction(self.keyframe_action)

        # Window menu
        window_menu = menu_bar.addMenu("Window")

        self.detach_action = QAction("Detach Viewer", self)
        self.detach_action.setStatusTip("Pop out 3D view to separate window")
        self.detach_action.triggered.connect(self._detach_viewer)
        window_menu.addAction(self.detach_action)

        window_menu.addSeparator()

        self.side_panel_action = QAction("Side Panel", self)
        self.side_panel_action.setCheckable(True)
        self.side_panel_action.setChecked(True)
        self.side_panel_action.setStatusTip("Toggle visibility of the side panel")
        self.side_panel_action.triggered.connect(self._toggle_side_panel)
        window_menu.addAction(self.side_panel_action)

        self.fullscreen_action = QAction("Fullscreen", self)
        self.fullscreen_action.setCheckable(True)
        self.fullscreen_action.setStatusTip("Toggle fullscreen mode")
        self.fullscreen_action.triggered.connect(self._toggle_fullscreen)
        window_menu.addAction(self.fullscreen_action)

        window_menu.addSeparator()

        self.split_view_action = QAction("Split View...", self)
        self.split_view_action.setStatusTip("Compare two timesteps side by side")
        self.split_view_action.triggered.connect(self._show_split_view)
        window_menu.addAction(self.split_view_action)

        window_menu.addSeparator()

        self.reload_view_action = QAction(CADIcons.get_icon('refresh'), "Reload View", self)
        self.reload_view_action.setStatusTip("Reload and refresh the 3D view")
        self.reload_view_action.triggered.connect(self._reload_view)
        window_menu.addAction(self.reload_view_action)

        title_bar.addWidget(menu_bar, 1)

        close_btn = QPushButton("X")
        close_btn.setFixedSize(20, 20)
        close_btn.setToolTip("Hide Solution Inspector")
        close_btn.clicked.connect(self._request_close)
        title_bar.addWidget(close_btn, 0, alignment=Qt.AlignRight | Qt.AlignVCenter)

        outer_layout.addLayout(title_bar, 0)

        # --- Main content: 3D viewport + properties panel ---
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        outer_layout.addLayout(main_layout, 1)

        # Left: 3D viewport
        self.vtk_widget = VTKWidget(self)
        main_layout.addWidget(self.vtk_widget, stretch=3)

        # Right: scrollable properties panel
        self._side_panel_scroll = QScrollArea(self)
        self._side_panel_scroll.setWidgetResizable(True)
        self._side_panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._side_panel_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._side_panel_scroll.setMinimumWidth(300)
        self._side_panel_scroll.setMaximumWidth(380)

        side_panel = QWidget(self._side_panel_scroll)
        side_layout = QVBoxLayout()
        # Add extra right margin to account for scrollbar (~16px)
        side_layout.setContentsMargins(8, 8, 20, 8)
        side_layout.setSpacing(6)
        side_panel.setLayout(side_layout)
        self._side_panel_scroll.setWidget(side_panel)

        # --- Solution file display ---
        file_group = QGroupBox("Solution File", side_panel)
        file_layout = QVBoxLayout()
        file_layout.setSpacing(4)
        file_group.setLayout(file_layout)

        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        self.file_edit.setPlaceholderText("No solution loaded")
        file_layout.addWidget(self.file_edit)

        side_layout.addWidget(file_group)

        # --- Animation / Playback controls ---
        anim_group = QGroupBox("Playback", side_panel)
        anim_layout = QVBoxLayout()
        anim_layout.setSpacing(4)
        anim_group.setLayout(anim_layout)

        # Time slider
        self.time_label = QLabel("Time: --")
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 0)
        self.time_slider.setEnabled(False)
        self.time_slider.valueChanged.connect(self._on_time_index_changed)
        anim_layout.addWidget(self.time_label)
        anim_layout.addWidget(self.time_slider)

        # Playback buttons
        playback_layout = QHBoxLayout()
        playback_layout.setSpacing(2)

        self.play_btn = QToolButton()
        self.play_btn.setIcon(CADIcons.get_icon('play'))
        self.play_btn.setToolTip("Play animation")
        self.play_btn.clicked.connect(self._toggle_animation)
        playback_layout.addWidget(self.play_btn)

        self.stop_btn = QToolButton()
        self.stop_btn.setIcon(CADIcons.get_icon('stop'))
        self.stop_btn.setToolTip("Stop and reset to start")
        self.stop_btn.clicked.connect(self._stop_animation)
        playback_layout.addWidget(self.stop_btn)

        self.loop_btn = QToolButton()
        self.loop_btn.setIcon(CADIcons.get_icon('loop'))
        self.loop_btn.setCheckable(True)
        self.loop_btn.setChecked(True)
        self.loop_btn.setToolTip("Loop animation")
        self.loop_btn.toggled.connect(self._on_loop_toggled)
        playback_layout.addWidget(self.loop_btn)

        playback_layout.addStretch()

        # Speed control
        playback_layout.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(10)
        self.fps_spin.setToolTip("Animation frames per second")
        self.fps_spin.valueChanged.connect(self._on_fps_changed)
        playback_layout.addWidget(self.fps_spin)

        anim_layout.addLayout(playback_layout)
        side_layout.addWidget(anim_group)

        # --- Mesh Display options ---
        display_group = QGroupBox("Mesh Display", side_panel)
        display_layout = QFormLayout()
        display_layout.setSpacing(4)
        display_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        display_group.setLayout(display_layout)

        # Opacity
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setToolTip("Mesh opacity (0-100%)")
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        display_layout.addRow("Opacity:", self.opacity_slider)

        # Representation
        self.repr_combo = QComboBox()
        self.repr_combo.addItems(["Surface", "Wireframe", "Points", "Surface + Edges"])
        self.repr_combo.currentTextChanged.connect(self._on_representation_changed)
        display_layout.addRow("Style:", self.repr_combo)

        side_layout.addWidget(display_group)

        # --- Scalar Field controls ---
        scalar_group = QGroupBox("Scalar Field", side_panel)
        scalar_layout = QFormLayout()
        scalar_layout.setSpacing(4)
        scalar_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        scalar_group.setLayout(scalar_layout)

        self.scalar_combo = QComboBox()
        self.scalar_combo.currentTextChanged.connect(self._on_scalar_changed)
        scalar_layout.addRow("Field:", self.scalar_combo)

        # Vector component selector (for vector fields)
        self.component_combo = QComboBox()
        self.component_combo.addItems(["Magnitude", "X", "Y", "Z"])
        self.component_combo.setEnabled(False)
        self.component_combo.currentTextChanged.connect(self._on_component_changed)
        scalar_layout.addRow("Component:", self.component_combo)

        self.auto_range_cb = QCheckBox("Auto range")
        self.auto_range_cb.setChecked(True)
        self.auto_range_cb.stateChanged.connect(self._on_auto_range_toggled)
        scalar_layout.addRow("", self.auto_range_cb)

        self.scalar_min_spin = QDoubleSpinBox()
        self.scalar_min_spin.setDecimals(4)
        self.scalar_min_spin.setRange(-1e12, 1e12)
        self.scalar_min_spin.valueChanged.connect(self._on_clim_changed)

        self.scalar_max_spin = QDoubleSpinBox()
        self.scalar_max_spin.setDecimals(4)
        self.scalar_max_spin.setRange(-1e12, 1e12)
        self.scalar_max_spin.valueChanged.connect(self._on_clim_changed)

        scalar_layout.addRow("Min:", self.scalar_min_spin)
        scalar_layout.addRow("Max:", self.scalar_max_spin)

        self.scalar_label_edit = QLineEdit()
        self.scalar_label_edit.setPlaceholderText("Colorbar label (optional)")
        self.scalar_label_edit.editingFinished.connect(self._on_scalar_label_changed)
        scalar_layout.addRow("Label:", self.scalar_label_edit)

        side_layout.addWidget(scalar_group)

        # --- Colorbar Options Dialog ---
        # Create the dialog and store references to its widgets for compatibility
        self._colorbar_options_dialog = ColorbarOptionsDialog(self)
        self._colorbar_options_dialog.cmap_combo.setCurrentText(self._current_cmap)

        # Create attribute aliases pointing to dialog widgets for backward compatibility
        self.cmap_combo = self._colorbar_options_dialog.cmap_combo
        self.global_range_cb = self._colorbar_options_dialog.global_range_cb
        self.orientation_combo = self._colorbar_options_dialog.orientation_combo
        self.colorbar_width_spin = self._colorbar_options_dialog.width_spin
        self.colorbar_height_spin = self._colorbar_options_dialog.height_spin
        self.colorbar_nlabels_spin = self._colorbar_options_dialog.nlabels_spin
        self.colorbar_fmt_combo = self._colorbar_options_dialog.fmt_combo
        self.colorbar_normalize_cb = self._colorbar_options_dialog.normalize_cb
        self.colorbar_pos_x_spin = self._colorbar_options_dialog.pos_x_spin
        self.colorbar_pos_y_spin = self._colorbar_options_dialog.pos_y_spin
        self.colorbar_font_combo = self._colorbar_options_dialog.font_combo
        self.colorbar_title_size_spin = self._colorbar_options_dialog.title_size_spin
        self.colorbar_label_size_spin = self._colorbar_options_dialog.label_size_spin
        self.colorbar_bold_cb = self._colorbar_options_dialog.bold_cb
        self.colorbar_shadow_cb = self._colorbar_options_dialog.shadow_cb

        # Connect dialog signals to existing handlers
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        self.global_range_cb.stateChanged.connect(self._on_global_range_toggled)
        self.orientation_combo.currentTextChanged.connect(self._on_orientation_changed)
        self.colorbar_width_spin.valueChanged.connect(self._on_colorbar_layout_changed)
        self.colorbar_height_spin.valueChanged.connect(self._on_colorbar_layout_changed)
        self.colorbar_nlabels_spin.valueChanged.connect(self._on_colorbar_layout_changed)
        self.colorbar_fmt_combo.currentIndexChanged.connect(self._on_colorbar_layout_changed)
        self.colorbar_normalize_cb.stateChanged.connect(self._on_colorbar_layout_changed)
        self.colorbar_pos_x_spin.valueChanged.connect(self._on_colorbar_layout_changed)
        self.colorbar_pos_y_spin.valueChanged.connect(self._on_colorbar_layout_changed)
        self.colorbar_font_combo.currentTextChanged.connect(self._on_colorbar_layout_changed)
        self.colorbar_title_size_spin.valueChanged.connect(self._on_colorbar_layout_changed)
        self.colorbar_label_size_spin.valueChanged.connect(self._on_colorbar_layout_changed)
        self.colorbar_bold_cb.stateChanged.connect(self._on_colorbar_layout_changed)
        self.colorbar_shadow_cb.stateChanged.connect(self._on_colorbar_layout_changed)

        # --- Statistics Dialog (created lazily) ---
        self._statistics_dialog: Optional[StatisticsDialog] = None

        # --- Probing Panel (initially hidden, toggled via Tools menu) ---
        self.probe_group = QGroupBox("Probe Data", side_panel)
        probe_layout = QVBoxLayout()
        probe_layout.setSpacing(4)
        self.probe_group.setLayout(probe_layout)

        self.probe_label = QLabel("Enable probe mode to inspect values.\nClick on the mesh to probe multiple points.")
        self.probe_label.setWordWrap(True)
        self.probe_label.setStyleSheet("font-family: monospace; font-size: 10px;")
        probe_layout.addWidget(self.probe_label)

        self.probe_mode_btn = QPushButton("Enable Probe Mode")
        self.probe_mode_btn.setIcon(CADIcons.get_icon('probe'))
        self.probe_mode_btn.setCheckable(True)
        self.probe_mode_btn.toggled.connect(self._toggle_probe_mode)
        probe_layout.addWidget(self.probe_mode_btn)

        # Probed points list
        from PySide6.QtWidgets import QListWidget, QAbstractItemView
        probe_layout.addWidget(QLabel("Probed Points:"))
        self.probe_points_list = QListWidget()
        self.probe_points_list.setMaximumHeight(100)
        self.probe_points_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.probe_points_list.setToolTip(
            "List of probed points. Select points to include in time series plot.\n"
            "Use Ctrl+Click to select multiple points."
        )
        self.probe_points_list.itemSelectionChanged.connect(self._on_probe_selection_changed)
        probe_layout.addWidget(self.probe_points_list)

        # Probe point management buttons
        probe_manage_layout = QHBoxLayout()
        probe_manage_layout.setSpacing(4)

        self.remove_probe_btn = QPushButton("Remove")
        self.remove_probe_btn.setToolTip("Remove selected probe points")
        self.remove_probe_btn.clicked.connect(self._remove_selected_probes)
        probe_manage_layout.addWidget(self.remove_probe_btn)

        self.clear_probes_btn = QPushButton("Clear All")
        self.clear_probes_btn.setToolTip("Remove all probe points")
        self.clear_probes_btn.clicked.connect(self._clear_all_probes)
        probe_manage_layout.addWidget(self.clear_probes_btn)

        probe_layout.addLayout(probe_manage_layout)

        # Time series plot button
        probe_btn_layout = QHBoxLayout()
        probe_btn_layout.setSpacing(4)

        self.plot_timeseries_btn = QPushButton("Plot Time Series")
        self.plot_timeseries_btn.setIcon(CADIcons.get_icon('chart'))
        self.plot_timeseries_btn.setToolTip(
            "Plot the scalar value at selected probe locations over all time steps.\n"
            "Select points from the list above, or all points will be plotted.\n"
            "Requires a time-varying solution (.pvd)."
        )
        self.plot_timeseries_btn.clicked.connect(self._plot_probe_time_series)
        probe_btn_layout.addWidget(self.plot_timeseries_btn)

        self.clear_timeseries_btn = QPushButton("Clear Plot")
        self.clear_timeseries_btn.setToolTip("Clear the time series plot")
        self.clear_timeseries_btn.clicked.connect(self._clear_time_series_data)
        probe_btn_layout.addWidget(self.clear_timeseries_btn)

        probe_layout.addLayout(probe_btn_layout)

        # Time series plot dialog (created lazily)
        self._time_series_dialog: Optional[TimeSeriesPlotDialog] = None

        self.probe_group.setVisible(False)  # Initially hidden
        side_layout.addWidget(self.probe_group)

        # --- Vector Field / Glyphs ---
        vector_group = QGroupBox("Vector Field", side_panel)
        vector_layout = QFormLayout()
        vector_layout.setSpacing(4)
        vector_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        vector_group.setLayout(vector_layout)

        self.vector_field_combo = QComboBox()
        self.vector_field_combo.addItem("(None)")
        self.vector_field_combo.currentTextChanged.connect(self._on_vector_field_changed)
        vector_layout.addRow("Field:", self.vector_field_combo)

        self.glyph_scale_spin = QDoubleSpinBox()
        self.glyph_scale_spin.setRange(0.01, 100.0)
        self.glyph_scale_spin.setValue(1.0)
        self.glyph_scale_spin.setSingleStep(0.1)
        self.glyph_scale_spin.setToolTip("Scale factor for vector glyphs")
        self.glyph_scale_spin.valueChanged.connect(self._on_glyph_scale_changed)
        vector_layout.addRow("Scale:", self.glyph_scale_spin)

        self.glyph_density_spin = QSpinBox()
        self.glyph_density_spin.setRange(1, 100)
        self.glyph_density_spin.setValue(20)
        self.glyph_density_spin.setToolTip("Percentage of points to show glyphs on")
        self.glyph_density_spin.valueChanged.connect(self._on_glyph_density_changed)
        vector_layout.addRow("Density %:", self.glyph_density_spin)

        side_layout.addWidget(vector_group)

        # Camera presets dialog (created lazily)
        self._camera_presets_dialog: Optional[CameraPresetsDialog] = None

        # --- Slice Controls (initially hidden) ---
        self.slice_group = QGroupBox("Plane Slice", side_panel)
        slice_layout = QFormLayout()
        slice_layout.setSpacing(4)
        slice_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.slice_group.setLayout(slice_layout)

        self.slice_origin_x = QDoubleSpinBox()
        self.slice_origin_x.setRange(-1e6, 1e6)
        self.slice_origin_x.setDecimals(3)
        self.slice_origin_x.valueChanged.connect(self._update_slice)
        slice_layout.addRow("Origin X:", self.slice_origin_x)

        self.slice_origin_y = QDoubleSpinBox()
        self.slice_origin_y.setRange(-1e6, 1e6)
        self.slice_origin_y.setDecimals(3)
        self.slice_origin_y.valueChanged.connect(self._update_slice)
        slice_layout.addRow("Origin Y:", self.slice_origin_y)

        self.slice_origin_z = QDoubleSpinBox()
        self.slice_origin_z.setRange(-1e6, 1e6)
        self.slice_origin_z.setDecimals(3)
        self.slice_origin_z.valueChanged.connect(self._update_slice)
        slice_layout.addRow("Origin Z:", self.slice_origin_z)

        self.slice_normal_combo = QComboBox()
        self.slice_normal_combo.addItems(["X", "Y", "Z", "Custom"])
        self.slice_normal_combo.currentTextChanged.connect(self._update_slice)
        slice_layout.addRow("Normal:", self.slice_normal_combo)

        self.slice_group.setVisible(False)
        side_layout.addWidget(self.slice_group)

        # --- Threshold Controls (initially hidden) ---
        self.threshold_group = QGroupBox("Threshold Filter", side_panel)
        threshold_layout = QFormLayout()
        threshold_layout.setSpacing(4)
        threshold_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.threshold_group.setLayout(threshold_layout)

        self.threshold_min_spin = QDoubleSpinBox()
        self.threshold_min_spin.setRange(-1e12, 1e12)
        self.threshold_min_spin.setDecimals(4)
        self.threshold_min_spin.valueChanged.connect(self._update_threshold)
        threshold_layout.addRow("Min:", self.threshold_min_spin)

        self.threshold_max_spin = QDoubleSpinBox()
        self.threshold_max_spin.setRange(-1e12, 1e12)
        self.threshold_max_spin.setDecimals(4)
        self.threshold_max_spin.valueChanged.connect(self._update_threshold)
        threshold_layout.addRow("Max:", self.threshold_max_spin)

        self.threshold_invert_cb = QCheckBox("Invert selection")
        self.threshold_invert_cb.toggled.connect(self._update_threshold)
        threshold_layout.addRow("", self.threshold_invert_cb)

        self.threshold_group.setVisible(False)
        side_layout.addWidget(self.threshold_group)

        # --- Streamline Controls (initially hidden) ---
        self.streamline_group = QGroupBox("Streamlines", side_panel)
        streamline_layout = QVBoxLayout()
        streamline_layout.setContentsMargins(6, 6, 6, 6)
        streamline_layout.setSpacing(4)
        self.streamline_group.setLayout(streamline_layout)

        self.streamline_status_label = QLabel("No streamlines generated")
        self.streamline_status_label.setWordWrap(True)
        self.streamline_status_label.setStyleSheet("font-size: 10px; color: #888;")
        streamline_layout.addWidget(self.streamline_status_label)

        streamline_btn_layout = QHBoxLayout()
        streamline_btn_layout.setSpacing(4)

        self.streamline_options_btn = QPushButton("Options...")
        self.streamline_options_btn.setToolTip("Open streamline options dialog")
        self.streamline_options_btn.clicked.connect(self._show_streamline_dialog)
        streamline_btn_layout.addWidget(self.streamline_options_btn)

        self.streamline_clear_btn = QPushButton("Clear")
        self.streamline_clear_btn.setToolTip("Remove streamlines")
        self.streamline_clear_btn.clicked.connect(self._clear_streamlines)
        streamline_btn_layout.addWidget(self.streamline_clear_btn)

        streamline_layout.addLayout(streamline_btn_layout)

        self.streamline_group.setVisible(False)
        side_layout.addWidget(self.streamline_group)

        # Streamline options dialog (created lazily)
        self._streamline_options_dialog: Optional[StreamlineOptionsDialog] = None

        # --- Field Calculator Panel (initially hidden) ---
        self.calculator_group = QGroupBox("Field Calculator", side_panel)
        self.calculator_group.setMinimumWidth(0)
        self.calculator_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        calculator_outer_layout = QVBoxLayout()
        calculator_outer_layout.setContentsMargins(4, 4, 4, 4)
        calculator_outer_layout.setSpacing(0)
        self.calculator_group.setLayout(calculator_outer_layout)

        self._calculator_panel = FieldCalculatorPanel(self.calculator_group)
        self._calculator_panel.field_computed.connect(self._on_calculator_field_computed)
        calculator_outer_layout.addWidget(self._calculator_panel)

        self.calculator_group.setVisible(False)
        side_layout.addWidget(self.calculator_group)

        side_layout.addStretch(1)
        main_layout.addWidget(self._side_panel_scroll, stretch=0)

        self._set_scalar_controls_enabled(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """Release heavy resources when the application closes."""
        # Stop animation timer
        if self._animation_timer is not None:
            self._animation_timer.stop()
            self._animation_timer = None

        if getattr(self, "vtk_widget", None) is not None:
            try:
                self.vtk_widget.shutdown()
            except Exception:
                pass
        self._reader = None
        self._current_mesh = None

    def _request_close(self) -> None:
        """Request closing of the parent dock widget that hosts this inspector."""
        parent = self.parent()
        while parent is not None and not isinstance(parent, QDockWidget):
            parent = parent.parent()
        if isinstance(parent, QDockWidget):
            parent.hide()

    # ------------------------------------------------------------------
    # Animation controls
    # ------------------------------------------------------------------
    def _toggle_animation(self) -> None:
        """Toggle play/pause animation."""
        if self._animation_playing:
            self._pause_animation()
        else:
            self._play_animation()

    def _play_animation(self) -> None:
        """Start animation playback."""
        if not self._time_values or len(self._time_values) < 2:
            return

        self._animation_playing = True
        self.play_btn.setIcon(CADIcons.get_icon('pause'))
        self.play_btn.setToolTip("Pause animation")

        if self._animation_timer is None:
            self._animation_timer = QTimer(self)
            self._animation_timer.timeout.connect(self._advance_frame)

        interval_ms = int(1000.0 / self._animation_fps)
        self._animation_timer.start(interval_ms)

    def _pause_animation(self) -> None:
        """Pause animation playback."""
        self._animation_playing = False
        self.play_btn.setIcon(CADIcons.get_icon('play'))
        self.play_btn.setToolTip("Play animation")

        if self._animation_timer is not None:
            self._animation_timer.stop()

    def _stop_animation(self) -> None:
        """Stop animation and reset to first frame."""
        self._pause_animation()
        self.time_slider.setValue(0)

    def _advance_frame(self) -> None:
        """Advance to next frame in animation."""
        if not self._time_values:
            return

        next_idx = self._current_time_index + 1
        if next_idx >= len(self._time_values):
            if self._animation_loop:
                next_idx = 0
            else:
                self._pause_animation()
                return

        self.time_slider.setValue(next_idx)

    def _on_loop_toggled(self, checked: bool) -> None:
        """Handle loop toggle."""
        self._animation_loop = checked

    def _on_fps_changed(self, value: int) -> None:
        """Handle FPS change."""
        self._animation_fps = float(value)
        if self._animation_playing and self._animation_timer is not None:
            interval_ms = int(1000.0 / self._animation_fps)
            self._animation_timer.setInterval(interval_ms)

    # ------------------------------------------------------------------
    # Mesh display controls
    # ------------------------------------------------------------------
    def _on_opacity_changed(self, value: int) -> None:
        """Handle opacity slider change."""
        self._mesh_opacity = value / 100.0
        self._render_current_mesh()

    def _on_representation_changed(self, text: str) -> None:
        """Handle representation style change."""
        self._mesh_representation = text
        self._mesh_show_edges = (text == "Surface + Edges")
        self._render_current_mesh()

    # ------------------------------------------------------------------
    # Camera controls
    # ------------------------------------------------------------------
    def _reset_camera_view(self) -> None:
        """Reset the 3D view camera."""
        if hasattr(self, "vtk_widget") and self.vtk_widget is not None:
            try:
                self.vtk_widget.reset_camera()
            except Exception:
                pass

    def _view_iso(self) -> None:
        """Switch to isometric view."""
        if hasattr(self, "vtk_widget") and self.vtk_widget is not None:
            try:
                self.vtk_widget.view_iso()
            except Exception:
                pass

    def _view_plus_x(self) -> None:
        """View from +X direction."""
        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter:
            plotter.view_yz()
            plotter.camera.azimuth = 0
            plotter.render()

    def _view_minus_x(self) -> None:
        """View from -X direction."""
        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter:
            plotter.view_yz()
            plotter.camera.azimuth = 180
            plotter.render()

    def _view_plus_y(self) -> None:
        """View from +Y direction."""
        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter:
            plotter.view_xz()
            plotter.camera.azimuth = 90
            plotter.render()

    def _view_minus_y(self) -> None:
        """View from -Y direction."""
        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter:
            plotter.view_xz()
            plotter.camera.azimuth = -90
            plotter.render()

    def _view_plus_z(self) -> None:
        """View from +Z direction (top)."""
        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter:
            plotter.view_xy()
            plotter.render()

    def _view_minus_z(self) -> None:
        """View from -Z direction (bottom)."""
        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter:
            plotter.view_xy()
            plotter.camera.elevation = -90
            plotter.render()

    def _show_camera_presets_dialog(self) -> None:
        """Show the camera presets popup dialog."""
        if self._camera_presets_dialog is None:
            self._camera_presets_dialog = CameraPresetsDialog(self)
            self._camera_presets_dialog.connect_callbacks(
                self._view_plus_x,
                self._view_minus_x,
                self._view_plus_y,
                self._view_minus_y,
                self._view_plus_z,
                self._view_minus_z,
                self._view_iso,
                self._reset_camera_view,
                self._toggle_parallel_projection,
            )
            # Sync initial state
            plotter = getattr(self.vtk_widget, "plotter", None)
            if plotter:
                is_parallel = plotter.camera.GetParallelProjection()
                self._camera_presets_dialog.set_parallel_checked(is_parallel)

        self._camera_presets_dialog.show()
        self._camera_presets_dialog.raise_()
        self._camera_presets_dialog.activateWindow()

    def _show_background_options_dialog(self) -> None:
        """Show the background options popup dialog."""
        if self._background_options_dialog is None:
            self._background_options_dialog = BackgroundOptionsDialog(self)
            self._background_options_dialog.options_changed.connect(self._on_background_options_changed)

        self._background_options_dialog.set_settings(self._background_settings)
        self._background_options_dialog.show()
        self._background_options_dialog.raise_()
        self._background_options_dialog.activateWindow()

    def _on_background_options_changed(self) -> None:
        if self._background_options_dialog is None:
            return
        settings = self._background_options_dialog.get_settings()
        self._background_settings = settings
        self._apply_background_settings(settings)

    def _apply_background_settings(self, settings: dict) -> None:
        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter is None:
            return

        renderer = getattr(plotter, "renderer", None)
        if renderer is None:
            try:
                renderer = plotter.ren_win.GetRenderers().GetFirstRenderer()
            except Exception:
                renderer = None

        mode = (settings.get("mode") or "Gradient").strip()
        color1 = (settings.get("color1") or CADTheme.get_color("viewport", "background-bottom")).strip()
        color2 = (settings.get("color2") or CADTheme.get_color("viewport", "background-top")).strip()

        if renderer is not None:
            try:
                renderer.TexturedBackgroundOff()
            except Exception:
                pass

        if mode == "Texture":
            if renderer is None:
                return
            try:
                renderer.GradientBackgroundOff()
            except Exception:
                pass

            try:
                plotter.set_background(color1)
            except Exception:
                pass

            texture_path = (settings.get("texture_path") or "").strip()
            if not texture_path:
                try:
                    renderer.TexturedBackgroundOff()
                except Exception:
                    pass
            else:
                if texture_path != self._background_texture_path or self._background_texture is None:
                    self._background_texture = self._load_background_texture(texture_path)
                    self._background_texture_path = texture_path if self._background_texture is not None else ""

                texture = self._background_texture
                if texture is None:
                    try:
                        renderer.TexturedBackgroundOff()
                    except Exception:
                        pass
                else:
                    if settings.get("texture_interpolate", True):
                        try:
                            texture.InterpolateOn()
                        except Exception:
                            pass
                    else:
                        try:
                            texture.InterpolateOff()
                        except Exception:
                            pass
                    if settings.get("texture_repeat", False):
                        try:
                            texture.RepeatOn()
                        except Exception:
                            pass
                    else:
                        try:
                            texture.RepeatOff()
                        except Exception:
                            pass

                    try:
                        renderer.SetBackgroundTexture(texture)
                        renderer.TexturedBackgroundOn()
                    except Exception:
                        pass

        elif mode == "Solid":
            try:
                plotter.set_background(color1)
            except Exception:
                if renderer is not None:
                    qc = QColor(color1)
                    if qc.isValid():
                        renderer.SetBackground(qc.redF(), qc.greenF(), qc.blueF())
                    try:
                        renderer.GradientBackgroundOff()
                    except Exception:
                        pass

        else:  # Gradient
            try:
                plotter.set_background(color1, top=color2)
            except Exception:
                if renderer is not None:
                    qc1 = QColor(color1)
                    qc2 = QColor(color2)
                    if qc1.isValid():
                        renderer.SetBackground(qc1.redF(), qc1.greenF(), qc1.blueF())
                    if qc2.isValid():
                        renderer.SetBackground2(qc2.redF(), qc2.greenF(), qc2.blueF())
                    try:
                        renderer.GradientBackgroundOn()
                    except Exception:
                        pass

            if renderer is not None:
                gradient_mode = (settings.get("gradient_mode") or "Vertical").strip()
                try:
                    modes = getattr(renderer, "GradientModes", None)
                    if modes is not None and hasattr(renderer, "SetGradientMode"):
                        mapping = {
                            "Vertical": getattr(modes, "VTK_GRADIENT_VERTICAL", None),
                            "Horizontal": getattr(modes, "VTK_GRADIENT_HORIZONTAL", None),
                            "Radial (Farthest Side)": getattr(modes, "VTK_GRADIENT_RADIAL_VIEWPORT_FARTHEST_SIDE", None),
                            "Radial (Farthest Corner)": getattr(modes, "VTK_GRADIENT_RADIAL_VIEWPORT_FARTHEST_CORNER", None),
                        }
                        mode_value = mapping.get(gradient_mode)
                        if mode_value is None:
                            mode_value = getattr(modes, "VTK_GRADIENT_VERTICAL", None)
                        if mode_value is not None:
                            renderer.SetGradientMode(mode_value)
                except Exception:
                    pass

                try:
                    if hasattr(renderer, "SetDitherGradient"):
                        renderer.SetDitherGradient(bool(settings.get("dither", True)))
                except Exception:
                    pass

        if renderer is not None:
            try:
                if hasattr(renderer, "SetBackgroundAlpha"):
                    renderer.SetBackgroundAlpha(float(settings.get("opacity", 1.0)))
            except Exception:
                pass

        try:
            plotter.render()
        except Exception:
            pass

    def _load_background_texture(self, path: str):
        if not path or not os.path.isfile(path):
            return None
        try:
            import vtk

            factory = vtk.vtkImageReader2Factory()
            reader = factory.CreateImageReader2(path)
            if reader is None:
                return None
            reader.SetFileName(path)
            reader.Update()

            texture = vtk.vtkTexture()
            texture.SetInputData(reader.GetOutput())
            return texture
        except Exception:
            return None

    def _show_colorbar_options_dialog(self) -> None:
        """Show the colorbar options popup dialog."""
        self._colorbar_options_dialog.show()
        self._colorbar_options_dialog.raise_()
        self._colorbar_options_dialog.activateWindow()

    def _toggle_parallel_projection(self, checked: bool = None) -> None:
        """Toggle between perspective and parallel projection."""
        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter is None:
            return

        if checked is None:
            checked = not plotter.camera.GetParallelProjection()

        plotter.camera.SetParallelProjection(checked)
        plotter.render()

        # Sync UI
        self.parallel_proj_action.setChecked(checked)
        if self._camera_presets_dialog is not None:
            self._camera_presets_dialog.set_parallel_checked(checked)

    # ------------------------------------------------------------------
    # Probing
    # ------------------------------------------------------------------
    def _toggle_probe_panel(self, visible: bool = None) -> None:
        """Toggle visibility of the probe data panel in the side panel."""
        if visible is None:
            visible = not self.probe_group.isVisible()

        self.probe_group.setVisible(visible)
        self.probe_panel_action.setChecked(visible)

    def _toggle_probe_mode(self, enabled: bool = None) -> None:
        """Toggle probe mode for inspecting values."""
        if enabled is None:
            enabled = not self._probe_enabled

        self._probe_enabled = enabled
        self.probe_mode_btn.setChecked(enabled)

        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter is None:
            return

        # Always disable any existing picking first to avoid conflicts
        try:
            plotter.disable_picking()
        except Exception:
            pass

        if enabled:
            self.probe_label.setText("Click on mesh to probe values...")
            try:
                # Use surface point picking for accurate depth-aware selection
                # This ensures the picked point aligns with what the user clicks on
                # the visible surface, respecting camera view and occlusion
                try:
                    # Preferred method: enable_surface_point_picking (PyVista >= 0.38)
                    # Disable built-in point display since we add our own marker
                    plotter.enable_surface_point_picking(
                        callback=self._on_probe_point,
                        show_message=False,
                        show_point=False,  # We add our own marker
                        tolerance=0.025,  # Picking tolerance as fraction of viewport
                        pickable_window=False,  # Only pick from meshes
                    )
                except (AttributeError, TypeError):
                    # Fallback: enable_surface_picking (older PyVista)
                    try:
                        plotter.enable_surface_picking(
                            callback=self._on_probe_surface,
                            show_message=False,
                        )
                    except (AttributeError, TypeError):
                        # Final fallback: use cell picking with ray casting
                        plotter.enable_cell_picking(
                            callback=self._on_probe_cell,
                            show_message=False,
                            through=False,  # Only pick visible (front) cells
                        )
            except Exception as e:
                self.probe_label.setText(f"Probe mode error: {e}")
                self._probe_enabled = False
                self.probe_action.setChecked(False)
                self.probe_mode_btn.setChecked(False)
        else:
            self.probe_label.setText("Probe mode disabled")
            # Clear probe marker if exists
            if self._probe_actor is not None:
                try:
                    plotter.remove_actor(self._probe_actor)
                except Exception:
                    pass
                self._probe_actor = None

    def _on_probe_point(self, point) -> None:
        """Handle probed point from surface point picking."""
        if not self._probe_enabled or self._current_mesh is None:
            return

        import numpy as np

        # Handle various input formats
        if point is None:
            return

        # Convert to numpy array and flatten
        point = np.asarray(point)
        if point.ndim == 2:
            # Multiple points or (1, 3) shaped - take first point
            point = point[0]
        point = point.flatten()

        if len(point) != 3:
            return

        self._process_probed_point(point)

    def _on_probe_surface(self, *args) -> None:
        """Handle probed point from surface picking (fallback for older PyVista)."""
        if not self._probe_enabled or self._current_mesh is None:
            return

        import numpy as np
        point = None

        # Different PyVista versions pass different arguments
        if len(args) == 1:
            arg = args[0]
            if hasattr(arg, 'points') and arg.n_points > 0:
                # It's a mesh - get the picked point
                point = arg.points[0]
            elif hasattr(arg, '__len__') and len(arg) == 3:
                point = np.asarray(arg)
        elif len(args) >= 2:
            for arg in args:
                if isinstance(arg, np.ndarray) and arg.shape == (3,):
                    point = arg
                    break
                elif hasattr(arg, '__len__') and len(arg) == 3:
                    point = np.asarray(arg)
                    break
                elif hasattr(arg, 'points') and arg.n_points > 0:
                    point = arg.points[0]
                    break

        if point is not None:
            self._process_probed_point(np.asarray(point).flatten())

    def _on_probe_cell(self, cell) -> None:
        """Handle probed cell from cell picking (final fallback)."""
        if not self._probe_enabled or self._current_mesh is None:
            return

        import numpy as np

        # Cell picking returns a mesh with the picked cell
        if cell is None or not hasattr(cell, 'points') or cell.n_points == 0:
            return

        # Use the center of the picked cell
        point = cell.center
        if point is not None:
            self._process_probed_point(np.asarray(point).flatten())

    def _process_probed_point(self, point) -> None:
        """Process a probed point and update the display."""
        import numpy as np

        mesh = self._current_mesh
        if mesh is None:
            return

        # The picked point is already on the mesh surface (from ray casting),
        # so use it directly for display. Find closest mesh point index for data lookup.
        picked_pt = np.asarray(point)

        # Store the probed point for time series plotting
        self._probed_point = tuple(picked_pt)

        # Check if this point is already in the list (within tolerance)
        point_key = (round(picked_pt[0], 4), round(picked_pt[1], 4), round(picked_pt[2], 4))
        already_exists = False
        for existing_pt in self._probed_points:
            existing_key = (round(existing_pt[0], 4), round(existing_pt[1], 4), round(existing_pt[2], 4))
            if existing_key == point_key:
                already_exists = True
                break

        # Add to list if not already present
        if not already_exists:
            self._probed_points.append(tuple(picked_pt))
            point_idx = len(self._probed_points)
            # Add to list widget
            color = self._probe_colors[(point_idx - 1) % len(self._probe_colors)]
            item_text = f"P{point_idx}: ({picked_pt[0]:.3g}, {picked_pt[1]:.3g}, {picked_pt[2]:.3g})"
            from PySide6.QtWidgets import QListWidgetItem
            from PySide6.QtGui import QColor
            item = QListWidgetItem(item_text)
            # Set background color to match marker
            item.setBackground(QColor(color))
            self.probe_points_list.addItem(item)
            # Add visual marker
            self._add_probe_marker(picked_pt, color, point_idx)

        try:
            closest_idx = mesh.find_closest_point(picked_pt)
        except Exception:
            closest_idx = None

        # Get scalar value at this point
        name = self._current_scalar_name
        value = None

        if closest_idx is not None:
            if name and name in mesh.point_data:
                value = mesh.point_data[name][closest_idx]
            elif name and name in mesh.cell_data:
                # For cell data, find cell containing point
                try:
                    cell_id = mesh.find_containing_cell(picked_pt)
                    if cell_id >= 0:
                        value = mesh.cell_data[name][cell_id]
                except Exception:
                    pass

        # Update probe display using the picked point coordinates
        n_points = len(self._probed_points)
        if value is not None:
            self.probe_label.setText(
                f"Last: ({picked_pt[0]:.4g}, {picked_pt[1]:.4g}, {picked_pt[2]:.4g})\n"
                f"Value: {value:.6g} | Total points: {n_points}"
            )
            self.probed_value_changed.emit(picked_pt[0], picked_pt[1], picked_pt[2], float(value))

            # Store for time series
            if point_key not in self._time_series_data:
                self._time_series_data[point_key] = []

            if self._time_values:
                time = self._time_values[self._current_time_index]
                self._time_series_data[point_key].append((time, float(value)))
        else:
            self.probe_label.setText(
                f"Last: ({picked_pt[0]:.4g}, {picked_pt[1]:.4g}, {picked_pt[2]:.4g})\n"
                f"No scalar value | Total points: {n_points}"
            )

    def _add_probe_marker(self, point, color: str = 'yellow', point_idx: int = 0) -> None:
        """Add a visual marker at the probed point location."""
        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter is None or self._pv is None:
            return

        import numpy as np

        # Calculate marker size based on mesh bounds
        try:
            bounds = self._current_mesh.bounds
            diag = np.sqrt(
                (bounds[1] - bounds[0])**2 +
                (bounds[3] - bounds[2])**2 +
                (bounds[5] - bounds[4])**2
            )
            marker_radius = diag * 0.01  # 1% of diagonal
        except Exception:
            marker_radius = 0.1

        # Create sphere marker at probed point
        try:
            sphere = self._pv.Sphere(radius=marker_radius, center=point)
            actor = plotter.add_mesh(
                sphere,
                color=color,
                opacity=0.9,
                name=f'probe_marker_{point_idx}',
                reset_camera=False,
                pickable=False,
            )
            self._probe_actors.append(actor)
            plotter.render()
        except Exception:
            pass

    def _on_probe_selection_changed(self) -> None:
        """Handle selection change in probe points list."""
        selected_items = self.probe_points_list.selectedItems()
        n_selected = len(selected_items)
        if n_selected > 0:
            self.probe_label.setText(f"{n_selected} point(s) selected for plotting")
        else:
            n_total = len(self._probed_points)
            self.probe_label.setText(f"No selection - all {n_total} points will be plotted")

    def _remove_selected_probes(self) -> None:
        """Remove selected probe points from the list."""
        selected_items = self.probe_points_list.selectedItems()
        if not selected_items:
            return

        plotter = getattr(self.vtk_widget, "plotter", None)

        # Get indices to remove (in reverse order to avoid index shifting)
        indices_to_remove = sorted(
            [self.probe_points_list.row(item) for item in selected_items],
            reverse=True
        )

        for idx in indices_to_remove:
            # Remove from list widget
            self.probe_points_list.takeItem(idx)

            # Remove from probed points list
            if 0 <= idx < len(self._probed_points):
                removed_pt = self._probed_points.pop(idx)
                # Remove corresponding time series data
                pt_key = (round(removed_pt[0], 4), round(removed_pt[1], 4), round(removed_pt[2], 4))
                self._time_series_data.pop(pt_key, None)

            # Remove actor
            if plotter and 0 <= idx < len(self._probe_actors):
                try:
                    plotter.remove_actor(self._probe_actors[idx])
                except Exception:
                    pass
                self._probe_actors.pop(idx)

        # Renumber the remaining items and update markers
        self._renumber_probe_points()

        if plotter:
            plotter.render()

        n_remaining = len(self._probed_points)
        self.probe_label.setText(f"Removed {len(indices_to_remove)} point(s). {n_remaining} remaining.")

    def _renumber_probe_points(self) -> None:
        """Renumber probe points in the list after removal."""
        for i in range(self.probe_points_list.count()):
            item = self.probe_points_list.item(i)
            if i < len(self._probed_points):
                pt = self._probed_points[i]
                color = self._probe_colors[i % len(self._probe_colors)]
                item.setText(f"P{i+1}: ({pt[0]:.3g}, {pt[1]:.3g}, {pt[2]:.3g})")
                from PySide6.QtGui import QColor
                item.setBackground(QColor(color))

    def _clear_all_probes(self) -> None:
        """Clear all probe points."""
        plotter = getattr(self.vtk_widget, "plotter", None)

        # Remove all markers
        if plotter:
            for actor in self._probe_actors:
                try:
                    plotter.remove_actor(actor)
                except Exception:
                    pass

        # Clear lists
        self._probed_points.clear()
        self._probe_actors.clear()
        self._time_series_data.clear()
        self.probe_points_list.clear()

        # Also clear the old single probe actor if present
        if self._probe_actor is not None and plotter:
            try:
                plotter.remove_actor(self._probe_actor)
            except Exception:
                pass
            self._probe_actor = None

        self._probed_point = None

        if plotter:
            plotter.render()

        self.probe_label.setText("All probe points cleared.")

    def _plot_probe_time_series(self) -> None:
        """
        Plot time series data for multiple probe locations.

        Collects scalar values at selected (or all) probed points across all time steps
        and displays them as multiple curves in an interactive matplotlib window.
        """
        # Check prerequisites
        if self._current_mesh is None:
            QMessageBox.warning(
                self, "No Data",
                "Please load a solution file first."
            )
            return

        if not self._probed_points:
            QMessageBox.warning(
                self, "No Probe Points",
                "Please enable probe mode and click on the mesh to select points."
            )
            return

        if not self._time_values or len(self._time_values) < 2:
            QMessageBox.warning(
                self, "No Time Data",
                "Time series plotting requires a time-varying solution (.pvd file)\n"
                "with multiple time steps."
            )
            return

        scalar_name = self._current_scalar_name
        if not scalar_name:
            QMessageBox.warning(
                self, "No Scalar Field",
                "Please select a scalar field to plot."
            )
            return

        # Determine which points to plot - selected or all
        selected_items = self.probe_points_list.selectedItems()
        if selected_items:
            # Get indices of selected items
            selected_indices = [self.probe_points_list.row(item) for item in selected_items]
            points_to_plot = [self._probed_points[i] for i in selected_indices if i < len(self._probed_points)]
        else:
            # Plot all points
            points_to_plot = list(self._probed_points)

        if not points_to_plot:
            QMessageBox.warning(
                self, "No Points to Plot",
                "No valid probe points selected."
            )
            return

        import numpy as np

        # Store current time index to restore later
        original_time_idx = self._current_time_index

        n_points = len(points_to_plot)
        self.probe_label.setText(f"Collecting time series data for {n_points} point(s)...")
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

        # Collect data for all points
        all_series = []  # List of (times, values, label, color) tuples

        try:
            for pt_idx, probed_pt in enumerate(points_to_plot):
                probed_pt = np.asarray(probed_pt)
                times = []
                values = []

                # Get color for this point
                if selected_items:
                    original_idx = selected_indices[pt_idx]
                else:
                    original_idx = pt_idx
                color = self._probe_colors[original_idx % len(self._probe_colors)]

                for time_idx, time_val in enumerate(self._time_values):
                    # Load mesh at this time step
                    if hasattr(self._reader, "set_active_time_index"):
                        self._reader.set_active_time_index(time_idx)

                    try:
                        mesh = self._reader.read()
                        if isinstance(mesh, self._pv.MultiBlock):
                            try:
                                mesh = mesh.combine()
                            except Exception:
                                if len(mesh) > 0:
                                    mesh = mesh[0]

                        # Find closest point and get value
                        closest_idx = mesh.find_closest_point(probed_pt)

                        value = None
                        if scalar_name in mesh.point_data:
                            value = mesh.point_data[scalar_name][closest_idx]
                        elif scalar_name in mesh.cell_data:
                            try:
                                cell_id = mesh.find_containing_cell(probed_pt)
                                if cell_id >= 0:
                                    value = mesh.cell_data[scalar_name][cell_id]
                            except Exception:
                                pass

                        if value is not None:
                            times.append(time_val)
                            values.append(float(value))

                    except Exception:
                        continue

                if len(times) >= 2:
                    pos_str = f"P{original_idx + 1}: ({probed_pt[0]:.2g}, {probed_pt[1]:.2g}, {probed_pt[2]:.2g})"
                    all_series.append((times, values, pos_str, color))

                # Update progress
                self.probe_label.setText(f"Collected data for point {pt_idx + 1}/{n_points}")
                QApplication.processEvents()

            # Restore original time step
            if hasattr(self._reader, "set_active_time_index"):
                self._reader.set_active_time_index(original_time_idx)
            self._update_mesh_from_reader(original_time_idx)

        except Exception as e:
            self.probe_label.setText(f"Error collecting data: {e}")
            return

        if not all_series:
            QMessageBox.warning(
                self, "Insufficient Data",
                "Could not collect enough time series data points.\n"
                "Please ensure the scalar field exists across time steps."
            )
            self.probe_label.setText(f"{n_points} probe point(s)")
            return

        # Create or get the plot dialog
        if self._time_series_dialog is None:
            self._time_series_dialog = TimeSeriesPlotDialog(self)

        # Plot all series
        for i, (times, values, label, color) in enumerate(all_series):
            clear = (i == 0)  # Only clear on first plot
            self._time_series_dialog.plot_time_series(
                times=times,
                values=values,
                label=label,
                xlabel="Time",
                ylabel=scalar_name,
                title=f"Time Series: {scalar_name} ({len(all_series)} points)",
                clear=clear,
                color=color
            )

        # Show the dialog
        self._time_series_dialog.show()
        self._time_series_dialog.raise_()
        self._time_series_dialog.activateWindow()

        self.probe_label.setText(
            f"Plotted {len(all_series)} curve(s) with {len(times)} time points each"
        )

    def _clear_time_series_data(self) -> None:
        """Clear collected time series data and close the plot dialog."""
        self._time_series_data.clear()

        if self._time_series_dialog is not None:
            self._time_series_dialog.clear_plot()

        self.probe_label.setText("Time series data cleared")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def _show_statistics_dialog(self) -> None:
        """Show the statistics dialog."""
        if self._statistics_dialog is None:
            self._statistics_dialog = StatisticsDialog(self)
            self._statistics_dialog.refresh_btn.clicked.connect(self._update_statistics)

        self._update_statistics()
        self._statistics_dialog.show()
        self._statistics_dialog.raise_()
        self._statistics_dialog.activateWindow()

    def _update_statistics(self) -> None:
        """Update statistics display for current scalar field."""
        if self._statistics_dialog is None:
            return

        mesh = self._current_mesh
        name = self._current_scalar_name

        if mesh is None or name is None:
            self._statistics_dialog.clear_statistics()
            QMessageBox.information(self, "No Data", "No data loaded to compute statistics.")
            return

        try:
            import numpy as np

            values = None
            if name in mesh.point_data:
                values = mesh.point_data[name]
            elif name in mesh.cell_data:
                values = mesh.cell_data[name]

            if values is None or len(values) == 0:
                self._statistics_dialog.clear_statistics()
                QMessageBox.information(self, "No Data", "No values for selected field.")
                return

            # Compute statistics
            vmin = float(np.nanmin(values))
            vmax = float(np.nanmax(values))
            vmean = float(np.nanmean(values))
            vstd = float(np.nanstd(values))
            vmedian = float(np.nanmedian(values))
            vsum = float(np.nansum(values))
            count = len(values)

            stats = {
                "count": count,
                "min": vmin,
                "max": vmax,
                "mean": vmean,
                "std": vstd,
                "median": vmedian,
                "range": vmax - vmin,
                "sum": vsum,
            }

            self._statistics_dialog.update_statistics(name, stats)

        except Exception as e:
            self._statistics_dialog.clear_statistics()
            QMessageBox.warning(self, "Error", f"Error computing statistics:\n{e}")

    # ------------------------------------------------------------------
    # Slicing
    # ------------------------------------------------------------------
    def _toggle_slice(self, enabled: bool = None) -> None:
        """Toggle plane slice visualization."""
        if enabled is None:
            enabled = not self._slice_enabled

        self._slice_enabled = enabled
        self.slice_action.setChecked(enabled)
        self.slice_group.setVisible(enabled)

        if enabled and self._current_mesh is not None:
            # Initialize slice origin to mesh center
            center = self._current_mesh.center
            self.slice_origin_x.setValue(center[0])
            self.slice_origin_y.setValue(center[1])
            self.slice_origin_z.setValue(center[2])
            self._update_slice()
        else:
            # Remove slice actor
            plotter = getattr(self.vtk_widget, "plotter", None)
            if plotter and self._slice_actor is not None:
                try:
                    plotter.remove_actor(self._slice_actor)
                except Exception:
                    pass
                self._slice_actor = None
            plotter.render() if plotter else None

    def _update_slice(self) -> None:
        """Update the slice plane."""
        if not self._slice_enabled or self._current_mesh is None:
            return

        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter is None:
            return

        # Remove old slice
        if self._slice_actor is not None:
            try:
                plotter.remove_actor(self._slice_actor)
            except Exception:
                pass

        origin = (
            self.slice_origin_x.value(),
            self.slice_origin_y.value(),
            self.slice_origin_z.value(),
        )

        normal_map = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1), "Custom": (1, 0, 0)}
        normal = normal_map.get(self.slice_normal_combo.currentText(), (1, 0, 0))

        try:
            sliced = self._current_mesh.slice(normal=normal, origin=origin)
            if sliced.n_points > 0:
                name = self._current_scalar_name
                self._slice_actor = plotter.add_mesh(
                    sliced,
                    scalars=name if name else None,
                    cmap=self._current_cmap,
                    line_width=3,
                    name="slice_plane",
                )
        except Exception:
            pass

        plotter.render()

    # ------------------------------------------------------------------
    # Threshold filtering
    # ------------------------------------------------------------------
    def _toggle_threshold(self, enabled: bool = None) -> None:
        """Toggle threshold filter visualization."""
        if enabled is None:
            enabled = not self._threshold_enabled

        self._threshold_enabled = enabled
        self.threshold_action.setChecked(enabled)
        self.threshold_group.setVisible(enabled)

        if enabled and self._current_mesh is not None:
            # Initialize with current scalar range
            rng = self._compute_local_range(self._current_scalar_name)
            if rng:
                self.threshold_min_spin.setValue(rng[0])
                self.threshold_max_spin.setValue(rng[1])
            self._update_threshold()
        else:
            # Restore original mesh visibility
            self._render_current_mesh()

    def _update_threshold(self) -> None:
        """Update threshold filter."""
        if not self._threshold_enabled or self._current_mesh is None:
            return

        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter is None:
            return

        name = self._current_scalar_name
        if name is None:
            return

        vmin = self.threshold_min_spin.value()
        vmax = self.threshold_max_spin.value()
        invert = self.threshold_invert_cb.isChecked()

        try:
            # Remove current mesh and add thresholded version
            plotter.clear()

            thresholded = self._current_mesh.threshold(
                value=[vmin, vmax],
                scalars=name,
                invert=invert,
            )

            if thresholded.n_points > 0:
                self._add_mesh_to_plotter(plotter, thresholded)

            plotter.render()
        except Exception:
            # Fall back to normal rendering
            self._render_current_mesh()

    # ------------------------------------------------------------------
    # Vector field / glyphs
    # ------------------------------------------------------------------
    def _on_vector_field_changed(self, name: str) -> None:
        """Handle vector field selection change."""
        if name == "(None)" or not name:
            self._vector_field_name = None
            self._remove_glyphs()
            return

        self._vector_field_name = name
        self._update_glyphs()

    def _on_glyph_scale_changed(self, value: float) -> None:
        """Handle glyph scale change."""
        self._glyph_scale = value
        self._update_glyphs()

    def _on_glyph_density_changed(self, value: int) -> None:
        """Handle glyph density change."""
        self._update_glyphs()

    def _update_glyphs(self) -> None:
        """Update vector field glyph visualization."""
        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter is None or self._current_mesh is None:
            return

        self._remove_glyphs()

        name = self._vector_field_name
        if name is None:
            return

        try:
            import numpy as np

            # Get vector data
            vectors = None
            if name in self._current_mesh.point_data:
                vectors = self._current_mesh.point_data[name]
            elif name in self._current_mesh.cell_data:
                # Convert to point data for glyphs
                self._current_mesh = self._current_mesh.cell_data_to_point_data()
                vectors = self._current_mesh.point_data.get(name)

            if vectors is None or vectors.ndim != 2 or vectors.shape[1] != 3:
                return

            # Subsample for density
            density = self.glyph_density_spin.value() / 100.0
            n_points = self._current_mesh.n_points
            n_show = max(1, int(n_points * density))

            if n_show < n_points:
                indices = np.random.choice(n_points, n_show, replace=False)
                points = self._current_mesh.points[indices]
                vectors = vectors[indices]
            else:
                points = self._current_mesh.points

            # Create arrow glyphs
            arrow = self._pv.Arrow()
            glyphs = self._pv.PolyData(points)
            glyphs["vectors"] = vectors
            glyphs.set_active_vectors("vectors")

            glyph_mesh = glyphs.glyph(
                orient="vectors",
                scale="vectors",
                factor=self._glyph_scale,
                geom=arrow,
            )

            self._glyph_actor = plotter.add_mesh(
                glyph_mesh,
                scalars=np.linalg.norm(vectors, axis=1),
                cmap=self._current_cmap,
                name="vector_glyphs",
            )

            plotter.render()

        except Exception:
            pass

    def _remove_glyphs(self) -> None:
        """Remove vector glyphs from visualization."""
        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter and self._glyph_actor is not None:
            try:
                plotter.remove_actor(self._glyph_actor)
            except Exception:
                pass
            self._glyph_actor = None

    # ------------------------------------------------------------------
    # Streamline visualization
    # ------------------------------------------------------------------
    def _show_streamline_dialog(self) -> None:
        """Show the streamline options dialog."""
        if self._streamline_options_dialog is None:
            self._streamline_options_dialog = StreamlineOptionsDialog(self)
            self._streamline_options_dialog.generate_btn.clicked.connect(self._generate_streamlines)
            self._streamline_options_dialog.clear_btn.clicked.connect(self._clear_streamlines)

        # Populate velocity fields
        vector_fields = self._get_vector_field_names()
        self._streamline_options_dialog.populate_velocity_fields(vector_fields)

        # Set mesh center and bounds if mesh is loaded
        if self._current_mesh is not None:
            try:
                center = self._current_mesh.center
                self._streamline_options_dialog.set_mesh_center(center)
                bounds = self._current_mesh.bounds
                self._streamline_options_dialog.set_mesh_bounds(bounds)
            except Exception:
                pass

        # Show the streamline group in side panel
        self.streamline_group.setVisible(True)

        self._streamline_options_dialog.show()
        self._streamline_options_dialog.raise_()
        self._streamline_options_dialog.activateWindow()

    def _get_vector_field_names(self) -> list:
        """Get list of 3-component vector fields from current mesh."""
        vector_fields = []
        mesh = self._current_mesh
        if mesh is None:
            return vector_fields

        # Check point data
        for name in mesh.point_data.keys():
            arr = mesh.point_data[name]
            if arr.ndim == 2 and arr.shape[1] == 3:
                vector_fields.append(name)

        # Check cell data
        for name in mesh.cell_data.keys():
            arr = mesh.cell_data[name]
            if arr.ndim == 2 and arr.shape[1] == 3:
                if name not in vector_fields:
                    vector_fields.append(name)

        return vector_fields

    def _generate_streamlines(self) -> None:
        """Generate streamlines from the current velocity field."""
        if self._pv is None or self._current_mesh is None:
            QMessageBox.warning(self, "No Data", "Please load a solution first.")
            return

        if self._streamline_options_dialog is None:
            return

        opts = self._streamline_options_dialog.get_options()
        velocity_field = opts["velocity_field"]

        if velocity_field == "(None)" or not velocity_field:
            QMessageBox.warning(
                self, "No Velocity Field",
                "Please select a velocity field for streamline generation.\n\n"
                "Streamlines require a 3-component vector field (e.g., velocity)."
            )
            return

        # Verify the field exists and is a vector
        mesh = self._current_mesh
        vectors = None
        if velocity_field in mesh.point_data:
            vectors = mesh.point_data[velocity_field]
        elif velocity_field in mesh.cell_data:
            # Convert cell data to point data for streamlines
            try:
                mesh = mesh.cell_data_to_point_data()
                vectors = mesh.point_data.get(velocity_field)
            except Exception:
                pass

        if vectors is None or vectors.ndim != 2 or vectors.shape[1] != 3:
            QMessageBox.warning(
                self, "Invalid Field",
                f"The field '{velocity_field}' is not a valid 3-component vector field."
            )
            return

        # Clear existing streamlines
        self._clear_streamlines()

        try:
            import numpy as np

            # Set the velocity as active vectors
            mesh.set_active_vectors(velocity_field)

            # Generate seed points
            seed_points = self._generate_seed_points(opts, mesh)
            if seed_points is None or len(seed_points) == 0:
                QMessageBox.warning(self, "Seed Error", "Failed to generate seed points.")
                return

            # Create seed source
            seed_source = self._pv.PolyData(seed_points)

            # Determine integration direction
            direction = opts["direction"]
            if direction == "Forward":
                integration_direction = "forward"
            elif direction == "Backward":
                integration_direction = "backward"
            else:
                integration_direction = "both"

            # Generate streamlines
            self.streamline_status_label.setText("Generating streamlines...")
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()

            streamlines = mesh.streamlines_from_source(
                seed_source,
                vectors=velocity_field,
                integration_direction=integration_direction,
                max_time=opts["max_time"],
                max_steps=opts["max_steps"],
                initial_step_length=opts["step_length"],
                terminal_speed=1e-12,
                compute_vorticity=False,
            )

            if streamlines is None or streamlines.n_points == 0:
                self.streamline_status_label.setText("No streamlines generated (check parameters)")
                return

            # Apply visualization style
            plotter = getattr(self.vtk_widget, "plotter", None)
            if plotter is None:
                return

            style = opts["style"]
            cmap = opts["cmap"]
            opacity = opts["opacity"]

            # Determine color scalars
            color_by = opts["color_by"]
            scalars = None
            scalar_name = None

            if color_by == "Velocity Magnitude":
                # Compute velocity magnitude
                if velocity_field in streamlines.point_data:
                    vel = streamlines.point_data[velocity_field]
                    if vel.ndim == 2 and vel.shape[1] == 3:
                        scalars = np.linalg.norm(vel, axis=1)
                        scalar_name = "velocity_magnitude"
                        streamlines.point_data[scalar_name] = scalars
            elif color_by == "Integration Time":
                # Use integration time if available
                if "IntegrationTime" in streamlines.point_data:
                    scalar_name = "IntegrationTime"
                    scalars = streamlines.point_data[scalar_name]

            # Apply tube or ribbon representation
            if style == "Tubes":
                tube_radius = opts["tube_radius"]
                n_sides = opts["tube_sides"]
                try:
                    display_mesh = streamlines.tube(radius=tube_radius, n_sides=n_sides)
                except Exception:
                    display_mesh = streamlines
            elif style == "Ribbons":
                width = opts["tube_radius"]
                try:
                    display_mesh = streamlines.ribbon(width=width)
                except Exception:
                    display_mesh = streamlines
            else:
                display_mesh = streamlines

            # Add to plotter
            kwargs = {
                "cmap": cmap,
                "opacity": opacity,
                "name": "streamlines",
            }

            if scalar_name and scalar_name in display_mesh.point_data:
                kwargs["scalars"] = scalar_name
            elif scalars is not None:
                kwargs["scalars"] = scalars

            if style == "Lines":
                kwargs["line_width"] = opts["line_width"]
                kwargs["render_lines_as_tubes"] = True

            self._streamline_actor = plotter.add_mesh(display_mesh, **kwargs)
            plotter.render()

            # Update status
            n_lines = streamlines.n_lines if hasattr(streamlines, 'n_lines') else "?"
            self.streamline_status_label.setText(
                f"Streamlines: {n_lines} lines, {streamlines.n_points} points"
            )
            self._streamline_enabled = True

        except Exception as e:
            import traceback
            self._record_telemetry(e, action="streamline_generation", traceback_str=traceback.format_exc())
            self.streamline_status_label.setText(f"Error: {str(e)[:50]}")
            QMessageBox.critical(
                self, "Streamline Error",
                f"Failed to generate streamlines:\n{e}"
            )

    def _generate_seed_points(self, opts: dict, mesh) -> "np.ndarray":
        """Generate seed points based on the selected geometry."""
        import numpy as np

        seed_type = opts["seed_type"]
        n_points = opts["n_points"]
        center = np.array(opts["center"])

        if seed_type == "Sphere":
            radius = opts["sphere_radius"]
            # Generate random points on/in sphere
            # Use golden spiral for better distribution
            indices = np.arange(0, n_points, dtype=float) + 0.5
            phi = np.arccos(1 - 2 * indices / n_points)
            theta = np.pi * (1 + 5**0.5) * indices

            # Random radii for volume distribution
            r = radius * np.cbrt(np.random.random(n_points))

            x = r * np.sin(phi) * np.cos(theta) + center[0]
            y = r * np.sin(phi) * np.sin(theta) + center[1]
            z = r * np.cos(phi) + center[2]

            return np.column_stack([x, y, z])

        elif seed_type == "Line":
            end = np.array(opts["line_end"])
            # Generate points along line
            t = np.linspace(0, 1, n_points)
            points = np.outer(1 - t, center) + np.outer(t, end)
            return points

        elif seed_type == "Plane":
            normal_dir = opts["plane_normal"]
            size = opts["plane_size"]

            # Create grid of points on plane
            n_side = int(np.ceil(np.sqrt(n_points)))
            lin = np.linspace(-size / 2, size / 2, n_side)

            if normal_dir == "X":
                yy, zz = np.meshgrid(lin, lin)
                xx = np.zeros_like(yy)
            elif normal_dir == "Y":
                xx, zz = np.meshgrid(lin, lin)
                yy = np.zeros_like(xx)
            else:  # Z
                xx, yy = np.meshgrid(lin, lin)
                zz = np.zeros_like(xx)

            points = np.column_stack([
                xx.ravel() + center[0],
                yy.ravel() + center[1],
                zz.ravel() + center[2],
            ])
            return points[:n_points]

        elif seed_type == "Point Cloud":
            if opts["use_surface"] and mesh is not None:
                # Sample from mesh surface
                try:
                    # Get surface points
                    surface = mesh.extract_surface()
                    if surface.n_points > n_points:
                        indices = np.random.choice(surface.n_points, n_points, replace=False)
                        return surface.points[indices]
                    return surface.points
                except Exception:
                    pass

            # Fallback: random points in bounding box
            if mesh is not None:
                bounds = mesh.bounds
                x = np.random.uniform(bounds[0], bounds[1], n_points)
                y = np.random.uniform(bounds[2], bounds[3], n_points)
                z = np.random.uniform(bounds[4], bounds[5], n_points)
                return np.column_stack([x, y, z])

        return None

    def _clear_streamlines(self) -> None:
        """Remove streamlines from visualization."""
        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter and self._streamline_actor is not None:
            try:
                plotter.remove_actor(self._streamline_actor)
            except Exception:
                pass
            self._streamline_actor = None
            plotter.render()

        self._streamline_enabled = False
        self.streamline_status_label.setText("No streamlines generated")

    # ------------------------------------------------------------------
    # Field Calculator
    # ------------------------------------------------------------------
    def _toggle_calculator(self, checked: bool = None) -> None:
        """Toggle the field calculator panel visibility."""
        if checked is None:
            checked = not self._calculator_visible

        self._calculator_visible = checked
        self.calculator_action.setChecked(checked)
        self.calculator_group.setVisible(checked)

        # Update calculator panel with current mesh data
        if checked and self._current_mesh is not None:
            self._calculator_panel.set_mesh(self._current_mesh, self._pv)

    def _on_calculator_field_computed(self, field_name: str) -> None:
        """Handle when a new field is computed by the calculator."""
        # Refresh scalar field combo to include new field
        self._populate_scalar_fields()
        self._populate_vector_fields()

        # Set the new field as active
        if field_name in [self.scalar_combo.itemText(i) for i in range(self.scalar_combo.count())]:
            self.scalar_combo.setCurrentText(field_name)
            self._on_scalar_changed(field_name)

    def _update_calculator_mesh(self) -> None:
        """Update the calculator panel when mesh changes."""
        if hasattr(self, '_calculator_panel') and self._calculator_visible:
            self._calculator_panel.set_mesh(self._current_mesh, self._pv)

    # ------------------------------------------------------------------
    # Export functions
    # ------------------------------------------------------------------
    def _take_screenshot(self) -> None:
        """Capture a screenshot of the current 3D view."""
        if not hasattr(self, "vtk_widget") or self.vtk_widget is None:
            return
        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Screenshot",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;All Files (*)",
        )
        if not path:
            return

        try:
            plotter.screenshot(path)
        except Exception:
            pass

    def _export_mesh(self) -> None:
        """Export current mesh to VTP/VTU file."""
        if self._current_mesh is None:
            self._record_telemetry(
                message="Mesh export requested but no mesh loaded",
                level="warning",
                action="solution_export_mesh_no_data",
            )
            QMessageBox.warning(self, "No Data", "No mesh loaded to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Mesh",
            "",
            "VTK PolyData (*.vtp);;VTK Unstructured Grid (*.vtu);;All Files (*)",
        )
        if not path:
            return

        try:
            self._current_mesh.save(path)
        except Exception as e:
            self._record_telemetry(e, action="solution_export_mesh")
            QMessageBox.critical(self, "Export Failed", f"Failed to export mesh:\n{e}")

    def _export_csv(self) -> None:
        """Export current scalar data to CSV."""
        if self._current_mesh is None or self._current_scalar_name is None:
            self._record_telemetry(
                message="CSV export requested but no scalar data loaded",
                level="warning",
                action="solution_export_csv_no_data",
            )
            QMessageBox.warning(self, "No Data", "No scalar data loaded to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Scalars to CSV",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        try:
            import numpy as np

            name = self._current_scalar_name
            mesh = self._current_mesh

            if name in mesh.point_data:
                values = mesh.point_data[name]
                points = mesh.points
            elif name in mesh.cell_data:
                values = mesh.cell_data[name]
                points = mesh.cell_centers().points
            else:
                raise ValueError("Scalar field not found")

            # Write CSV
            with open(path, 'w') as f:
                f.write(f"x,y,z,{name}\n")
                for i, (pt, val) in enumerate(zip(points, values)):
                    if np.isscalar(val):
                        f.write(f"{pt[0]},{pt[1]},{pt[2]},{val}\n")
                    else:
                        # Vector data
                        f.write(f"{pt[0]},{pt[1]},{pt[2]},{','.join(map(str, val))}\n")

        except Exception as e:
            self._record_telemetry(e, action="solution_export_csv")
            QMessageBox.critical(self, "Export Failed", f"Failed to export CSV:\n{e}")

    def _export_animation(self) -> None:
        """Export animation as GIF or MP4."""
        if not self._time_values or len(self._time_values) < 2:
            self._record_telemetry(
                message="Animation export requested but insufficient time steps",
                level="warning",
                action="solution_export_animation_no_data",
            )
            QMessageBox.warning(self, "No Animation", "Need multiple time steps for animation.")
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Animation",
            "",
            "GIF Image (*.gif);;MP4 Video (*.mp4);;All Files (*)",
        )
        if not path:
            return

        # Determine format and ensure extension
        is_mp4 = "mp4" in selected_filter.lower() or path.lower().endswith('.mp4')
        if is_mp4:
            if not path.lower().endswith('.mp4'):
                path = path + '.mp4'
        else:
            if not path.lower().endswith('.gif'):
                path = path + '.gif'

        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter is None:
            return

        try:
            from PySide6.QtWidgets import QApplication, QProgressDialog
            from PySide6.QtCore import Qt
            import numpy as np

            # Create progress dialog
            progress = QProgressDialog("Exporting animation...", "Cancel", 0, len(self._time_values), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)

            # Save ALL current plotter state
            saved_state = self._save_plotter_state(plotter)
            original_idx = self._current_time_index

            # Collect frames
            frames = []

            for idx in range(len(self._time_values)):
                if progress.wasCanceled():
                    break

                progress.setValue(idx)
                progress.setLabelText(f"Rendering frame {idx + 1} of {len(self._time_values)}...")
                QApplication.processEvents()

                # Update mesh for this time step with full state restoration
                self._update_mesh_for_animation(idx, saved_state)

                # Force multiple render passes for complete rendering
                plotter.render()
                QApplication.processEvents()
                plotter.render()

                # Get high-quality screenshot as RGB
                frame = plotter.screenshot(return_img=True)

                # Ensure frame is uint8
                if frame is not None:
                    if frame.dtype != np.uint8:
                        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                    # Handle RGBA -> RGB
                    if frame.ndim == 3 and frame.shape[2] == 4:
                        frame = frame[:, :, :3]
                    frames.append(frame.copy())

            progress.setValue(len(self._time_values))
            progress.setLabelText("Encoding video...")
            QApplication.processEvents()

            # Restore original state
            self._on_time_index_changed(original_idx)
            self._restore_plotter_state(plotter, saved_state)
            plotter.render()

            if progress.wasCanceled() or not frames:
                return

            # Save animation
            if is_mp4:
                self._save_as_mp4(path, frames, progress)
            else:
                self._save_as_gif(path, frames, progress)

        except Exception as e:
            import traceback
            # Record the failure (including full traceback) so export issues
            # are visible to developers in telemetry as well as in the UI.
            self._record_telemetry(e, action="solution_export_animation")
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export animation:\n{e}\n\n{traceback.format_exc()}"
            )

    def _save_plotter_state(self, plotter) -> dict:
        """Save all relevant plotter state for animation consistency."""
        from svv.visualize.gui.theme import CADTheme

        state = {}

        # Camera settings - save everything
        try:
            state['camera_position'] = plotter.camera_position
            state['camera_focal_point'] = tuple(plotter.camera.focal_point)
            state['camera_view_up'] = tuple(plotter.camera.up)
            state['camera_parallel_projection'] = plotter.camera.GetParallelProjection()
            state['camera_parallel_scale'] = plotter.camera.GetParallelScale()
            state['camera_clipping_range'] = plotter.camera.clipping_range
            state['camera_view_angle'] = plotter.camera.view_angle
        except Exception:
            state['camera_position'] = None

        # Background colors from theme
        try:
            state['bg_bottom'] = CADTheme.get_color('viewport', 'background-bottom')
            state['bg_top'] = CADTheme.get_color('viewport', 'background-top')
        except Exception:
            state['bg_bottom'] = '#1e1e1e'
            state['bg_top'] = '#2d2d30'

        # Display settings from internal state
        state['cmap'] = self._current_cmap
        state['scalar_name'] = self._current_scalar_name
        state['opacity'] = self._mesh_opacity
        state['representation'] = self._mesh_representation
        state['show_edges'] = self._mesh_show_edges
        state['colorbar_visible'] = self._colorbar_visible

        # Scalar range settings
        state['auto_range'] = self.auto_range_cb.isChecked()
        state['global_range'] = self.global_range_cb.isChecked()
        state['scalar_min'] = self.scalar_min_spin.value()
        state['scalar_max'] = self.scalar_max_spin.value()
        state['scalar_label'] = self.scalar_label_edit.text()
        state['colorbar_orientation'] = self.orientation_combo.currentText()

        # Component selection for vectors
        state['component'] = self.component_combo.currentText()

        return state

    def _restore_plotter_state(self, plotter, state: dict) -> None:
        """Restore plotter state from saved state."""
        # Restore camera
        if state.get('camera_position') is not None:
            try:
                plotter.camera_position = state['camera_position']
                plotter.camera.focal_point = state['camera_focal_point']
                plotter.camera.up = state['camera_view_up']
                plotter.camera.SetParallelProjection(state['camera_parallel_projection'])
                plotter.camera.SetParallelScale(state['camera_parallel_scale'])
                plotter.camera.clipping_range = state['camera_clipping_range']
                plotter.camera.view_angle = state['camera_view_angle']
            except Exception:
                pass

        # Restore background
        try:
            plotter.set_background(state['bg_bottom'], top=state['bg_top'])
        except Exception:
            pass

    def _update_mesh_for_animation(self, time_index: int, saved_state: dict) -> None:
        """Update mesh for animation frame with full state restoration."""
        if self._reader is None or self._pv is None:
            return

        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter is None:
            return

        # Load mesh for this time index
        try:
            if hasattr(self._reader, "set_active_time_index"):
                self._reader.set_active_time_index(int(time_index))
            mesh = self._reader.read()
        except Exception:
            return

        if isinstance(mesh, self._pv.MultiBlock):
            try:
                mesh = mesh.combine()
            except Exception:
                if len(mesh) > 0:
                    mesh = mesh[0]

        self._current_mesh = mesh
        self._current_time_index = time_index

        # Clear plotter
        plotter.clear()

        # Restore background gradient
        try:
            plotter.set_background(saved_state['bg_bottom'], top=saved_state['bg_top'])
        except Exception:
            pass

        # Add mesh using the SAME function that works for normal display
        # This ensures consistent colormap rendering
        if mesh is not None:
            self._add_mesh_to_plotter(plotter, mesh)

        # Restore camera position AFTER adding mesh
        if saved_state.get('camera_position') is not None:
            try:
                plotter.camera_position = saved_state['camera_position']
                plotter.camera.focal_point = saved_state['camera_focal_point']
                plotter.camera.up = saved_state['camera_view_up']
                plotter.camera.SetParallelProjection(saved_state['camera_parallel_projection'])
                plotter.camera.SetParallelScale(saved_state['camera_parallel_scale'])
                plotter.camera.clipping_range = saved_state['camera_clipping_range']
            except Exception:
                pass

    def _add_mesh_to_plotter_with_state(self, plotter, mesh, state: dict) -> None:
        """Add mesh to plotter using saved state for consistent rendering."""
        import numpy as np

        # Get scalar name and handle vector components
        scalar_name = state.get('scalar_name')
        display_scalar_name = scalar_name

        if scalar_name and mesh is not None:
            arr = None
            if scalar_name in mesh.point_data:
                arr = mesh.point_data[scalar_name]
            elif scalar_name in mesh.cell_data:
                arr = mesh.cell_data[scalar_name]

            # Handle vector fields - compute component
            if arr is not None and arr.ndim == 2 and arr.shape[1] == 3:
                component = state.get('component', 'Magnitude')
                if component == "X":
                    computed = arr[:, 0]
                    display_scalar_name = f"{scalar_name}_X"
                elif component == "Y":
                    computed = arr[:, 1]
                    display_scalar_name = f"{scalar_name}_Y"
                elif component == "Z":
                    computed = arr[:, 2]
                    display_scalar_name = f"{scalar_name}_Z"
                else:  # Magnitude
                    computed = np.linalg.norm(arr, axis=1)
                    display_scalar_name = f"{scalar_name}_mag"

                # Add computed array to mesh
                if scalar_name in mesh.point_data:
                    mesh.point_data[display_scalar_name] = computed
                else:
                    mesh.cell_data[display_scalar_name] = computed

        # Determine color limits
        clim = None
        if display_scalar_name:
            if state.get('auto_range', True):
                # Compute range from current mesh
                try:
                    values = mesh[display_scalar_name]
                    if values is not None and len(values) > 0:
                        clim = (float(np.nanmin(values)), float(np.nanmax(values)))
                except Exception:
                    pass

                # Use global range if enabled
                if state.get('global_range', False):
                    global_clim = self._global_ranges.get(f"{scalar_name}_{state.get('component', 'Magnitude')}")
                    if global_clim:
                        clim = global_clim
            else:
                clim = (state.get('scalar_min', 0), state.get('scalar_max', 1))

        # Scalar bar configuration
        scalar_bar_args = {
            "title": state.get('scalar_label', '') or display_scalar_name or '',
            "vertical": state.get('colorbar_orientation', 'Vertical') == 'Vertical',
            "title_font_size": 12,
            "label_font_size": 10,
            "n_labels": 5,
            "fmt": "%.3g",
        }

        # Representation style
        style = "surface"
        show_edges = state.get('show_edges', False)
        repr_mode = state.get('representation', 'Surface')
        if repr_mode == "Wireframe":
            style = "wireframe"
        elif repr_mode == "Points":
            style = "points"
        elif repr_mode == "Surface + Edges":
            show_edges = True

        # Add mesh with full settings
        try:
            cmap = state.get('cmap', 'coolwarm')
            opacity = state.get('opacity', 1.0)
            show_colorbar = bool(display_scalar_name and state.get('colorbar_visible', True))

            kwargs = {
                "scalars": display_scalar_name if display_scalar_name else None,
                "clim": clim,
                "cmap": cmap,
                "opacity": opacity,
                "style": style,
                "show_edges": show_edges,
                "lighting": True,
                "smooth_shading": True,
            }

            if show_colorbar and display_scalar_name:
                kwargs["scalar_bar_args"] = scalar_bar_args
            else:
                kwargs["show_scalar_bar"] = False

            plotter.add_mesh(mesh, **kwargs)
        except Exception:
            # Fallback: add mesh without scalars
            try:
                plotter.add_mesh(mesh, opacity=state.get('opacity', 1.0))
            except Exception:
                pass

    def _save_as_gif(self, path: str, frames: list, progress) -> None:
        """Save frames as GIF using imageio (best compatibility)."""
        from PySide6.QtWidgets import QApplication

        progress.setLabelText("Saving GIF...")
        QApplication.processEvents()

        # Use imageio directly - it handles GIF encoding well
        try:
            import imageio.v3 as iio
            duration_ms = int(1000.0 / self._animation_fps)
            iio.imwrite(path, frames, extension=".gif", duration=duration_ms, loop=0)
            QMessageBox.information(self, "Export Complete", f"Animation saved to:\n{path}")
        except (ImportError, TypeError):
            try:
                import imageio
                duration = 1.0 / self._animation_fps
                imageio.mimsave(path, frames, format='GIF', duration=duration, loop=0)
                QMessageBox.information(self, "Export Complete", f"Animation saved to:\n{path}")
            except ImportError:
                self._record_telemetry(
                    message="GIF export failed: imageio not installed",
                    level="warning",
                    action="solution_export_gif_missing_imageio",
                )
                QMessageBox.warning(
                    self, "Missing Dependency",
                    "imageio is required for GIF export.\n"
                    "Install with: pip install imageio"
                )
            except Exception as e:
                self._record_telemetry(e, action="solution_export_gif")
                QMessageBox.critical(self, "GIF Export Failed", f"Error: {e}")

    def _get_colormap_colors(self, cmap_name: str, n_colors: int = 64) -> list:
        """Get RGB colors from a matplotlib/pyvista colormap."""
        colors = []
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm

            # Get the colormap
            try:
                cmap = plt.get_cmap(cmap_name)
            except ValueError:
                cmap = plt.get_cmap('coolwarm')  # fallback

            # Sample colors from the colormap
            for i in range(n_colors):
                t = i / (n_colors - 1)
                rgba = cmap(t)
                r = int(rgba[0] * 255)
                g = int(rgba[1] * 255)
                b = int(rgba[2] * 255)
                colors.append((r, g, b))

        except ImportError:
            # Fallback: manually define coolwarm-like colors
            # Blue to white to red
            for i in range(n_colors):
                t = i / (n_colors - 1)
                if t < 0.5:
                    # Blue to white
                    t2 = t * 2
                    r = int(59 + t2 * (255 - 59))
                    g = int(76 + t2 * (255 - 76))
                    b = int(192 + t2 * (255 - 192))
                else:
                    # White to red
                    t2 = (t - 0.5) * 2
                    r = int(255)
                    g = int(255 - t2 * (255 - 58))
                    b = int(255 - t2 * (255 - 68))
                colors.append((r, g, b))

        return colors

    def _save_as_mp4(self, path: str, frames: list, progress) -> None:
        """Save frames as MP4 video for better color quality."""
        from PySide6.QtWidgets import QApplication

        try:
            import imageio.v3 as iio

            progress.setLabelText("Encoding MP4 video...")
            QApplication.processEvents()

            # Use ffmpeg plugin for MP4
            fps = int(self._animation_fps)
            iio.imwrite(
                path,
                frames,
                extension=".mp4",
                fps=fps,
                codec='libx264',
                quality=8,  # Higher quality (1-10 scale, higher is better)
            )

            QMessageBox.information(self, "Export Complete", f"Video saved to:\n{path}")

        except ImportError:
            try:
                import imageio

                fps = int(self._animation_fps)
                writer = imageio.get_writer(path, fps=fps, codec='libx264', quality=8)
                for frame in frames:
                    writer.append_data(frame)
                writer.close()

                QMessageBox.information(self, "Export Complete", f"Video saved to:\n{path}")

            except ImportError:
                self._record_telemetry(
                    message="MP4 export failed: imageio/ffmpeg not installed",
                    level="warning",
                    action="solution_export_mp4_missing_ffmpeg",
                )
                QMessageBox.warning(
                    self, "Missing Dependency",
                    "imageio with ffmpeg is required for MP4 export.\n"
                    "Install with: pip install imageio imageio-ffmpeg"
                )
            except Exception as e:
                self._record_telemetry(e, action="solution_export_mp4")
                QMessageBox.critical(self, "Export Failed", f"Failed to save MP4:\n{e}")

    def _export_time_series(self) -> None:
        """Export probed time series data to CSV."""
        if not self._time_series_data:
            self._record_telemetry(
                message="Time series export requested but no data collected",
                level="warning",
                action="solution_export_time_series_no_data",
            )
            QMessageBox.warning(
                self, "No Data",
                "No time series data collected.\nEnable probe mode and click on points to collect data."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Time Series",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        try:
            with open(path, 'w') as f:
                f.write("point_x,point_y,point_z,time,value\n")
                for (px, py, pz), data in self._time_series_data.items():
                    for time, value in data:
                        f.write(f"{px},{py},{pz},{time},{value}\n")
        except Exception as e:
            self._record_telemetry(e, action="solution_export_time_series")
            QMessageBox.critical(self, "Export Failed", f"Failed to export time series:\n{e}")

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------
    def _browse_for_file(self) -> None:
        if self._pv is None:
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Solution (.pvd)",
            "",
            "VTK Collection (*.pvd);;VTK Files (*.vtp *.vtu);;All Files (*)",
        )
        if not path:
            return

        self.file_edit.setText(path)
        self._load_solution(path)

    def _load_solution(self, path: str) -> None:
        if self._pv is None:
            return
        if not os.path.isfile(path):
            return

        try:
            reader = self._pv.get_reader(path)
        except Exception:
            return

        self._reader = reader
        self._time_values = list(getattr(reader, "time_values", []) or [])
        self._global_ranges.clear()
        self._time_series_data.clear()

        if self._time_values:
            self.time_slider.setEnabled(True)
            self.time_slider.setRange(0, len(self._time_values) - 1)
            self.time_slider.setValue(0)
            self._current_time_index = 0
            self._update_time_label(0)
        else:
            self.time_slider.setEnabled(False)
            self.time_slider.setRange(0, 0)
            self.time_label.setText("Time: --")

        self._update_mesh_from_reader(time_index=0)
        self._populate_scalar_fields()
        self._populate_vector_fields()
        self._update_statistics()

    # ------------------------------------------------------------------
    # Time handling
    # ------------------------------------------------------------------
    def _update_time_label(self, index: int) -> None:
        if not self._time_values or index < 0 or index >= len(self._time_values):
            self.time_label.setText("Time: --")
            return
        t = self._time_values[index]
        total = len(self._time_values)
        try:
            self.time_label.setText(f"Time: {t:.4g} ({index + 1}/{total})")
        except Exception:
            self.time_label.setText(f"Time: {t}")

    def _on_time_index_changed(self, index: int) -> None:
        if self._reader is None:
            return
        self._current_time_index = index
        self._update_time_label(index)
        self._update_mesh_from_reader(time_index=index)

    def _update_mesh_from_reader(self, time_index: Optional[int] = None) -> None:
        if self._reader is None or self._pv is None:
            return

        try:
            if time_index is not None and hasattr(self._reader, "set_active_time_index"):
                self._reader.set_active_time_index(int(time_index))
            mesh = self._reader.read()
        except Exception:
            return

        if isinstance(mesh, self._pv.MultiBlock):
            try:
                mesh = mesh.combine()
            except Exception:
                if len(mesh) > 0:
                    mesh = mesh[0]

        self._current_mesh = mesh
        self._update_scalar_range()
        self._render_current_mesh()

        # Update slice if active
        if self._slice_enabled:
            self._update_slice()

        # Update calculator if visible
        self._update_calculator_mesh()

    # ------------------------------------------------------------------
    # Scalar controls
    # ------------------------------------------------------------------
    def _set_scalar_controls_enabled(self, enabled: bool) -> None:
        self.scalar_combo.setEnabled(enabled)
        self.auto_range_cb.setEnabled(enabled)
        self.scalar_min_spin.setEnabled(enabled and not self.auto_range_cb.isChecked())
        self.scalar_max_spin.setEnabled(enabled and not self.auto_range_cb.isChecked())
        self.scalar_label_edit.setEnabled(enabled)

    def _populate_scalar_fields(self) -> None:
        self._current_scalar_name = None
        self.scalar_combo.blockSignals(True)
        self.scalar_combo.clear()
        self.scalar_combo.blockSignals(False)

        mesh = self._current_mesh
        if self._pv is None or mesh is None:
            self._set_scalar_controls_enabled(False)
            return

        names = []
        try:
            names.extend(list(mesh.point_data.keys()))
        except Exception:
            pass
        try:
            names.extend(list(mesh.cell_data.keys()))
        except Exception:
            pass

        unique_names = sorted(set(names))
        if not unique_names:
            self._set_scalar_controls_enabled(False)
            return

        self.scalar_combo.blockSignals(True)
        for name in unique_names:
            self.scalar_combo.addItem(name)
        self.scalar_combo.blockSignals(False)

        self._current_scalar_name = unique_names[0]
        self.scalar_combo.setCurrentText(self._current_scalar_name)
        self.scalar_label_edit.setText(self._current_scalar_name)

        # Check if selected field is a vector (3 components)
        self._check_vector_field()

        self._set_scalar_controls_enabled(True)
        self._update_scalar_range()
        self._render_current_mesh()

    def _populate_vector_fields(self) -> None:
        """Populate vector field combo with 3-component arrays."""
        self.vector_field_combo.blockSignals(True)
        self.vector_field_combo.clear()
        self.vector_field_combo.addItem("(None)")

        mesh = self._current_mesh
        if mesh is None:
            self.vector_field_combo.blockSignals(False)
            return

        # Find 3-component arrays (vectors)
        for name in mesh.point_data.keys():
            arr = mesh.point_data[name]
            if arr.ndim == 2 and arr.shape[1] == 3:
                self.vector_field_combo.addItem(name)

        for name in mesh.cell_data.keys():
            arr = mesh.cell_data[name]
            if arr.ndim == 2 and arr.shape[1] == 3:
                if name not in [self.vector_field_combo.itemText(i)
                               for i in range(self.vector_field_combo.count())]:
                    self.vector_field_combo.addItem(name)

        self.vector_field_combo.blockSignals(False)

    def _check_vector_field(self) -> None:
        """Check if current scalar field is a vector and enable component selector."""
        name = self._current_scalar_name
        mesh = self._current_mesh

        if name is None or mesh is None:
            self.component_combo.setEnabled(False)
            return

        arr = None
        if name in mesh.point_data:
            arr = mesh.point_data[name]
        elif name in mesh.cell_data:
            arr = mesh.cell_data[name]

        if arr is not None and arr.ndim == 2 and arr.shape[1] == 3:
            self.component_combo.setEnabled(True)
        else:
            self.component_combo.setEnabled(False)
            self.component_combo.setCurrentText("Magnitude")

    def _on_component_changed(self, component: str) -> None:
        """Handle vector component selection change."""
        self._update_scalar_range()
        self._render_current_mesh()

    def _compute_local_range(self, name: str) -> Optional[tuple[float, float]]:
        mesh = self._current_mesh
        if mesh is None or name is None:
            return None
        try:
            values = mesh[name]
        except Exception:
            return None
        if values is None or len(values) == 0:
            return None

        import numpy as np

        # Handle vector fields
        if values.ndim == 2 and values.shape[1] == 3:
            component = self.component_combo.currentText()
            if component == "X":
                values = values[:, 0]
            elif component == "Y":
                values = values[:, 1]
            elif component == "Z":
                values = values[:, 2]
            else:  # Magnitude
                values = np.linalg.norm(values, axis=1)

        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        return vmin, vmax

    def _compute_global_range(self, name: str) -> Optional[tuple[float, float]]:
        if self._reader is None or self._pv is None:
            return None
        if not self._time_values:
            return self._compute_local_range(name)

        cache_key = f"{name}_{self.component_combo.currentText()}"
        if cache_key in self._global_ranges:
            return self._global_ranges[cache_key]

        import numpy as np

        mins = []
        maxs = []
        original_index = self._current_time_index

        for idx in range(len(self._time_values)):
            try:
                if hasattr(self._reader, "set_active_time_index"):
                    self._reader.set_active_time_index(idx)
                mesh = self._reader.read()
            except Exception:
                continue

            if isinstance(mesh, self._pv.MultiBlock):
                try:
                    mesh = mesh.combine()
                except Exception:
                    if len(mesh) > 0:
                        mesh = mesh[0]
                    else:
                        mesh = None
            if mesh is None:
                continue
            try:
                values = mesh[name]
            except Exception:
                continue
            if values is None or len(values) == 0:
                continue

            # Handle vector fields
            if values.ndim == 2 and values.shape[1] == 3:
                component = self.component_combo.currentText()
                if component == "X":
                    values = values[:, 0]
                elif component == "Y":
                    values = values[:, 1]
                elif component == "Z":
                    values = values[:, 2]
                else:
                    values = np.linalg.norm(values, axis=1)

            mins.append(float(np.nanmin(values)))
            maxs.append(float(np.nanmax(values)))

        # Restore original time index
        try:
            if hasattr(self._reader, "set_active_time_index"):
                self._reader.set_active_time_index(original_index)
        except Exception:
            pass

        if not mins:
            return None
        vmin = min(mins)
        vmax = max(maxs)
        self._global_ranges[cache_key] = (vmin, vmax)
        return vmin, vmax

    def _update_scalar_range(self) -> None:
        mesh = self._current_mesh
        name = self._current_scalar_name
        if mesh is None or name is None:
            return
        if getattr(self, "global_range_cb", None) and self.global_range_cb.isChecked():
            rng = self._compute_global_range(name)
        else:
            rng = self._compute_local_range(name)
        if rng is None:
            return
        vmin, vmax = rng
        if not self.auto_range_cb.isChecked():
            self.scalar_min_spin.blockSignals(True)
            self.scalar_max_spin.blockSignals(True)
            self.scalar_min_spin.setValue(vmin)
            self.scalar_max_spin.setValue(vmax)
            self.scalar_min_spin.blockSignals(False)
            self.scalar_max_spin.blockSignals(False)

    def _on_scalar_changed(self, name: str) -> None:
        if not name:
            return
        self._current_scalar_name = name
        if not self.scalar_label_edit.text().strip():
            self.scalar_label_edit.setText(name)
        self._check_vector_field()
        self._update_scalar_range()
        self._render_current_mesh()
        self._update_statistics()

    def _on_auto_range_toggled(self, state: int) -> None:
        auto = state == Qt.Checked
        self.scalar_min_spin.setEnabled(not auto)
        self.scalar_max_spin.setEnabled(not auto)
        if auto:
            self._update_scalar_range()
        self._render_current_mesh()

    def _on_clim_changed(self) -> None:
        if self.auto_range_cb.isChecked():
            return
        vmin = self.scalar_min_spin.value()
        vmax = self.scalar_max_spin.value()
        if vmax <= vmin:
            return
        self._render_current_mesh()

    def _on_scalar_label_changed(self) -> None:
        self._render_current_mesh()

    def _on_cmap_changed(self, cmap: str) -> None:
        if not cmap:
            return
        self._current_cmap = cmap
        self._render_current_mesh()

    def _on_global_range_toggled(self, state: int) -> None:
        self._update_scalar_range()
        self._render_current_mesh()

    def _on_colorbar_layout_changed(self, *args) -> None:
        self._render_current_mesh()

    def _on_orientation_changed(self, orientation: str) -> None:
        """Handle colorbar orientation change - adjust defaults appropriately."""
        # Block signals to prevent multiple re-renders
        self.colorbar_pos_x_spin.blockSignals(True)
        self.colorbar_pos_y_spin.blockSignals(True)
        self.colorbar_width_spin.blockSignals(True)
        self.colorbar_height_spin.blockSignals(True)

        if orientation == "Horizontal":
            # Horizontal colorbar: wide and short, at bottom
            self.colorbar_pos_x_spin.setValue(0.25)
            self.colorbar_pos_y_spin.setValue(0.05)
            self.colorbar_width_spin.setValue(0.5)
            self.colorbar_height_spin.setValue(0.08)
        else:
            # Vertical colorbar: narrow and tall, at right
            self.colorbar_pos_x_spin.setValue(0.85)
            self.colorbar_pos_y_spin.setValue(0.1)
            self.colorbar_width_spin.setValue(0.05)
            self.colorbar_height_spin.setValue(0.8)

        self.colorbar_pos_x_spin.blockSignals(False)
        self.colorbar_pos_y_spin.blockSignals(False)
        self.colorbar_width_spin.blockSignals(False)
        self.colorbar_height_spin.blockSignals(False)

        self._render_current_mesh()

    def _toggle_scale_bar(self, checked: bool) -> None:
        """Toggle the scale bar visibility in this solution viewport."""
        if hasattr(self, "vtk_widget") and self.vtk_widget is not None:
            try:
                self.vtk_widget.set_scale_bar_visible(bool(checked))
            except Exception:
                pass

    def _toggle_color_bar(self, checked: bool) -> None:
        """Toggle the scalar color bar visibility."""
        self._colorbar_visible = bool(checked)
        self._render_current_mesh()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _get_display_scalars(self):
        """Get scalar values for display, handling vector components."""
        name = self._current_scalar_name
        mesh = self._current_mesh

        if name is None or mesh is None:
            return None, name

        arr = None
        if name in mesh.point_data:
            arr = mesh.point_data[name]
        elif name in mesh.cell_data:
            arr = mesh.cell_data[name]

        if arr is None:
            return None, name

        # Handle vector fields
        if arr.ndim == 2 and arr.shape[1] == 3:
            import numpy as np
            component = self.component_combo.currentText()

            if component == "X":
                computed = arr[:, 0]
                display_name = f"{name}_X"
            elif component == "Y":
                computed = arr[:, 1]
                display_name = f"{name}_Y"
            elif component == "Z":
                computed = arr[:, 2]
                display_name = f"{name}_Z"
            else:  # Magnitude
                computed = np.linalg.norm(arr, axis=1)
                display_name = f"{name}_mag"

            # Add computed array to mesh
            if name in mesh.point_data:
                mesh.point_data[display_name] = computed
            else:
                mesh.cell_data[display_name] = computed

            return display_name, display_name

        return name, name

    def _add_mesh_to_plotter(self, plotter, mesh) -> None:
        """Add mesh to plotter with current display settings."""
        scalar_name, _ = self._get_display_scalars()

        # Determine color limits
        clim = None
        if scalar_name:
            if self.auto_range_cb.isChecked():
                if getattr(self, "global_range_cb", None) and self.global_range_cb.isChecked():
                    rng = self._compute_global_range(self._current_scalar_name)
                else:
                    rng = self._compute_local_range(self._current_scalar_name)
                if rng is not None and rng[1] > rng[0]:
                    clim = rng
            else:
                vmin = self.scalar_min_spin.value()
                vmax = self.scalar_max_spin.value()
                if vmax > vmin:
                    clim = (vmin, vmax)

        # Scalar bar configuration
        scalar_bar_args = self._build_scalar_bar_args(clim)

        cmap = getattr(self, "_current_cmap", "coolwarm")

        # Representation style
        style = "surface"
        show_edges = self._mesh_show_edges
        if self._mesh_representation == "Wireframe":
            style = "wireframe"
        elif self._mesh_representation == "Points":
            style = "points"

        try:
            kwargs = {
                "scalars": scalar_name if scalar_name else None,
                "clim": clim,
                "cmap": cmap,
                "opacity": self._mesh_opacity,
                "style": style,
                "show_edges": show_edges,
            }
            show_colorbar = bool(scalar_name and getattr(self, "_colorbar_visible", True))
            if show_colorbar:
                kwargs["scalar_bar_args"] = scalar_bar_args
            else:
                kwargs["show_scalar_bar"] = False

            plotter.add_mesh(mesh, **kwargs)
        except Exception:
            pass

    def _build_scalar_bar_args(self, clim=None) -> dict:
        """Build scalar bar arguments dictionary from current UI settings."""
        scalar_bar_args = {}

        # Title
        label = self.scalar_label_edit.text().strip()
        if label:
            scalar_bar_args["title"] = label

        # Orientation
        vertical = True
        if hasattr(self, "orientation_combo"):
            vertical = self.orientation_combo.currentText() == "Vertical"
        scalar_bar_args["vertical"] = vertical

        # Width and height
        if hasattr(self, "colorbar_width_spin"):
            scalar_bar_args["width"] = self.colorbar_width_spin.value()
        if hasattr(self, "colorbar_height_spin"):
            scalar_bar_args["height"] = self.colorbar_height_spin.value()

        # Position
        if hasattr(self, "colorbar_pos_x_spin"):
            scalar_bar_args["position_x"] = self.colorbar_pos_x_spin.value()
        if hasattr(self, "colorbar_pos_y_spin"):
            scalar_bar_args["position_y"] = self.colorbar_pos_y_spin.value()

        # Number of labels
        if hasattr(self, "colorbar_nlabels_spin"):
            scalar_bar_args["n_labels"] = self.colorbar_nlabels_spin.value()

        # Number format
        if hasattr(self, "colorbar_fmt_combo"):
            fmt_text = self.colorbar_fmt_combo.currentText()
            # Extract format string from the combo text
            if "%.3g" in fmt_text:
                fmt = "%.3g"
            elif "%.2f" in fmt_text:
                fmt = "%.2f"
            elif "%.4f" in fmt_text:
                fmt = "%.4f"
            elif "%.2e" in fmt_text:
                fmt = "%.2e"
            elif "%.3e" in fmt_text:
                fmt = "%.3e"
            elif "%d" in fmt_text:
                fmt = "%d"
            elif "%.1f%%" in fmt_text:
                fmt = "%.1f%%"
            else:
                fmt = "%.3g"
            scalar_bar_args["fmt"] = fmt

        # Font family
        if hasattr(self, "colorbar_font_combo"):
            scalar_bar_args["font_family"] = self.colorbar_font_combo.currentText().lower()

        # Font sizes - use larger values for better resolution
        if hasattr(self, "colorbar_title_size_spin"):
            scalar_bar_args["title_font_size"] = self.colorbar_title_size_spin.value()
        if hasattr(self, "colorbar_label_size_spin"):
            scalar_bar_args["label_font_size"] = self.colorbar_label_size_spin.value()

        # Bold text
        if hasattr(self, "colorbar_bold_cb") and self.colorbar_bold_cb.isChecked():
            scalar_bar_args["bold"] = True

        # Shadow for better readability
        if hasattr(self, "colorbar_shadow_cb") and self.colorbar_shadow_cb.isChecked():
            scalar_bar_args["shadow"] = True

        # Italic off by default for cleaner look
        scalar_bar_args["italic"] = False

        # Handle normalized [0, 1] display
        if hasattr(self, "colorbar_normalize_cb") and self.colorbar_normalize_cb.isChecked():
            if clim is not None:
                vmin, vmax = clim
                if vmax > vmin:
                    # Create custom labels for [0, 1] range
                    n_labels = scalar_bar_args.get("n_labels", 5)
                    # PyVista doesn't directly support custom label mapping,
                    # but we can use annotations or adjust the title to indicate normalization
                    original_title = scalar_bar_args.get("title", "")
                    if original_title:
                        scalar_bar_args["title"] = f"{original_title} (normalized)"
                    else:
                        scalar_bar_args["title"] = "Normalized [0, 1]"

        return scalar_bar_args

    def _render_current_mesh(self) -> None:
        plotter = getattr(self.vtk_widget, "plotter", None)
        mesh = self._current_mesh

        if plotter is None:
            return

        plotter.clear()
        if mesh is None:
            plotter.render()
            return

        self._add_mesh_to_plotter(plotter, mesh)

        plotter.reset_camera()
        plotter.render()

        # Re-add glyphs if active
        if self._vector_field_name:
            self._update_glyphs()

    # ------------------------------------------------------------------
    # Analysis Menu Methods
    # ------------------------------------------------------------------
    def _show_histogram_dialog(self) -> None:
        """Show histogram of current scalar field distribution."""
        mesh = self._current_mesh
        name = self._current_scalar_name

        if mesh is None or name is None:
            QMessageBox.information(self, "No Data", "No scalar field loaded to create histogram.")
            return

        try:
            import numpy as np

            values = None
            if name in mesh.point_data:
                values = mesh.point_data[name]
            elif name in mesh.cell_data:
                values = mesh.cell_data[name]

            if values is None or len(values) == 0:
                QMessageBox.information(self, "No Data", "No values for selected field.")
                return

            # Flatten if multi-component
            if values.ndim > 1:
                values = np.linalg.norm(values, axis=1)

            # Create histogram dialog
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Histogram: {name}")
            dialog.setMinimumSize(600, 450)
            layout = QVBoxLayout(dialog)

            try:
                import matplotlib
                matplotlib.use('QtAgg')
                from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
                from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
                from matplotlib.figure import Figure

                fig = Figure(figsize=(8, 5), dpi=100)
                ax = fig.add_subplot(111)
                canvas = FigureCanvas(fig)
                toolbar = NavigationToolbar(canvas, dialog)

                layout.addWidget(toolbar)
                layout.addWidget(canvas)

                # Plot histogram
                ax.hist(values, bins=50, edgecolor='black', alpha=0.7)
                ax.set_xlabel(name)
                ax.set_ylabel("Frequency")
                ax.set_title(f"Distribution of {name}")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                canvas.draw()

            except ImportError:
                label = QLabel("Matplotlib is required for histogram display.\nInstall with: pip install matplotlib")
                layout.addWidget(label)

            dialog.exec()

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to create histogram:\n{e}")

    def _show_line_plot_dialog(self) -> None:
        """Show dialog for sampling values along a line."""
        if self._current_mesh is None:
            QMessageBox.information(self, "No Data", "No mesh loaded for line plot.")
            return

        if not hasattr(self, '_line_plot_dialog') or self._line_plot_dialog is None:
            self._line_plot_dialog = LinePlotDialog(self)
            self._line_plot_dialog.pick_points_requested.connect(self._start_line_plot_picking)

        self._line_plot_dialog.set_mesh(self._current_mesh, self._current_scalar_name)
        self._line_plot_dialog.show()
        self._line_plot_dialog.raise_()
        self._line_plot_dialog.activateWindow()

    def _start_line_plot_picking(self) -> None:
        """Start point picking mode for line plot."""
        self._line_plot_pick_count = 0
        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter is None:
            return

        try:
            plotter.enable_surface_point_picking(
                callback=self._on_line_plot_point_picked,
                show_message=False,
                show_point=True,
                tolerance=0.025,
            )
        except (AttributeError, TypeError):
            plotter.enable_point_picking(
                callback=self._on_line_plot_point_picked,
                show_message=False,
            )

    def _on_line_plot_point_picked(self, point) -> None:
        """Handle point picked for line plot."""
        if not hasattr(self, '_line_plot_dialog') or self._line_plot_dialog is None:
            return

        if not hasattr(self, '_line_plot_pick_count'):
            self._line_plot_pick_count = 0

        if self._line_plot_pick_count == 0:
            self._line_plot_dialog.set_point1(tuple(point))
            self._line_plot_pick_count = 1
        else:
            self._line_plot_dialog.set_point2(tuple(point))
            self._line_plot_pick_count = 0
            # Disable picking
            plotter = getattr(self.vtk_widget, "plotter", None)
            if plotter:
                try:
                    plotter.disable_picking()
                except Exception:
                    pass

    def _show_time_series_analysis(self) -> None:
        """Show time series analysis - redirect to probe time series if available."""
        if not self._time_values or len(self._time_values) < 2:
            QMessageBox.information(
                self, "Time Series",
                "Time series analysis requires a time-varying solution (.pvd) with multiple time steps.\n\n"
                "Please load a .pvd file with multiple time steps, then use the Probe Data panel to select a point."
            )
            return

        # Show the probe panel and inform user
        self._toggle_probe_panel(True)
        QMessageBox.information(
            self, "Time Series",
            "To create a time series plot:\n\n"
            "1. Enable 'Probe Mode' in the Probe Data panel\n"
            "2. Click on a point on the mesh\n"
            "3. Click 'Plot Time Series' to see values over time"
        )

    def _show_compare_fields_dialog(self) -> None:
        """Show dialog to compare two scalar fields."""
        mesh = self._current_mesh
        if mesh is None:
            QMessageBox.information(self, "No Data", "No mesh loaded to compare fields.")
            return

        # Get available fields
        fields = list(mesh.point_data.keys()) + list(mesh.cell_data.keys())
        if len(fields) < 2:
            QMessageBox.information(self, "Insufficient Fields", "Need at least two fields to compare.")
            return

        if not hasattr(self, '_compare_fields_dialog') or self._compare_fields_dialog is None:
            self._compare_fields_dialog = CompareFieldsDialog(self)

        self._compare_fields_dialog.set_mesh(mesh)
        self._compare_fields_dialog.show()
        self._compare_fields_dialog.raise_()
        self._compare_fields_dialog.activateWindow()

    def _compute_field_difference(self) -> None:
        """Compute difference between two fields or timesteps."""
        mesh = self._current_mesh
        if mesh is None:
            QMessageBox.information(self, "No Data", "No mesh loaded to compute difference.")
            return

        if not hasattr(self, '_field_diff_dialog') or self._field_diff_dialog is None:
            self._field_diff_dialog = FieldDifferenceDialog(self)
            self._field_diff_dialog.field_computed.connect(self._on_field_difference_computed)

        self._field_diff_dialog.set_mesh(mesh)
        self._field_diff_dialog.set_time_values(self._time_values, self._load_timestep_mesh)
        self._field_diff_dialog.show()
        self._field_diff_dialog.raise_()
        self._field_diff_dialog.activateWindow()

    def _on_field_difference_computed(self, field_name: str) -> None:
        """Handle newly computed field difference."""
        # Refresh the scalar field combo to include new field
        self._populate_scalar_fields()
        # Select the new field
        idx = self.scalar_combo.findText(field_name)
        if idx >= 0:
            self.scalar_combo.setCurrentIndex(idx)

    def _load_timestep_mesh(self, idx: int):
        """Load mesh at a specific timestep index."""
        if not self._time_files or idx >= len(self._time_files):
            return None

        import pyvista as pv
        try:
            return pv.read(self._time_files[idx])
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Animation Menu Methods
    # ------------------------------------------------------------------
    def _toggle_animation(self) -> None:
        """Toggle animation play/pause."""
        if hasattr(self, '_animation_playing') and self._animation_playing:
            self._stop_animation()
        else:
            self._start_animation()

    def _start_animation(self) -> None:
        """Start animation playback."""
        if not self._time_values or len(self._time_values) < 2:
            QMessageBox.information(self, "No Animation", "No time-varying data loaded.")
            return

        self._animation_playing = True
        self.anim_play_action.setText("Pause")
        self.play_btn.setIcon(CADIcons.get_icon('pause'))

        # Start timer for animation
        if not hasattr(self, '_anim_timer'):
            from PySide6.QtCore import QTimer
            self._anim_timer = QTimer(self)
            self._anim_timer.timeout.connect(self._animation_step)

        speed = getattr(self, '_playback_speed', 1.0)
        interval = int(100 / speed)  # Base interval 100ms
        self._anim_timer.start(interval)

    def _stop_animation(self) -> None:
        """Stop animation playback."""
        self._animation_playing = False
        self.anim_play_action.setText("Play")
        self.play_btn.setIcon(CADIcons.get_icon('play'))

        if hasattr(self, '_anim_timer'):
            self._anim_timer.stop()

    def _animation_step(self) -> None:
        """Advance animation by one frame."""
        if not self._time_values:
            return

        current_idx = self.time_slider.value()
        next_idx = current_idx + 1

        if next_idx >= len(self._time_values):
            if self.anim_loop_action.isChecked():
                next_idx = 0
            else:
                self._stop_animation()
                return

        self.time_slider.setValue(next_idx)

    def _go_to_first_frame(self) -> None:
        """Jump to first frame."""
        if self._time_values:
            self.time_slider.setValue(0)

    def _step_backward(self) -> None:
        """Step backward one frame."""
        if self._time_values:
            current = self.time_slider.value()
            if current > 0:
                self.time_slider.setValue(current - 1)

    def _step_forward(self) -> None:
        """Step forward one frame."""
        if self._time_values:
            current = self.time_slider.value()
            if current < len(self._time_values) - 1:
                self.time_slider.setValue(current + 1)

    def _go_to_last_frame(self) -> None:
        """Jump to last frame."""
        if self._time_values:
            self.time_slider.setValue(len(self._time_values) - 1)

    def _toggle_animation_loop(self, checked: bool) -> None:
        """Toggle animation looping."""
        # State is already tracked by the action's checked state
        pass

    def _set_playback_speed(self, speed: float) -> None:
        """Set animation playback speed."""
        self._playback_speed = speed

        # Update checkmarks in speed menu
        for s, action in self._speed_actions:
            action.setChecked(s == speed)

        # Update timer if animation is playing
        if hasattr(self, '_animation_playing') and self._animation_playing:
            if hasattr(self, '_anim_timer'):
                interval = int(100 / speed)
                self._anim_timer.setInterval(interval)

    def _show_keyframe_dialog(self) -> None:
        """Show keyframe animation dialog."""
        if not hasattr(self, '_keyframe_dialog') or self._keyframe_dialog is None:
            self._keyframe_dialog = KeyframeAnimationDialog(self)

        plotter = getattr(self.vtk_widget, "plotter", None)
        self._keyframe_dialog.set_plotter(plotter)
        self._keyframe_dialog.show()
        self._keyframe_dialog.raise_()
        self._keyframe_dialog.activateWindow()

    # ------------------------------------------------------------------
    # Window Menu Methods
    # ------------------------------------------------------------------
    def _detach_viewer(self) -> None:
        """Detach the 3D viewer to a separate window."""
        if hasattr(self, '_detached_window') and self._detached_window is not None:
            # Re-attach
            self._reattach_viewer()
            return

        # Store reference to the main layout for re-attachment
        self._main_layout_ref = None
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i)
            if item and item.layout():
                # Find the layout containing the vtk_widget
                for j in range(item.layout().count()):
                    sub_item = item.layout().itemAt(j)
                    if sub_item and sub_item.widget() == self.vtk_widget:
                        self._main_layout_ref = item.layout()
                        self._vtk_widget_index = j
                        break

        # Create detached window
        from PySide6.QtWidgets import QMainWindow

        class DetachedViewerWindow(QMainWindow):
            def __init__(self, parent_inspector):
                super().__init__()
                self._parent_inspector = parent_inspector

            def closeEvent(self, event):
                # Return widget to main window before closing
                self._parent_inspector._reattach_viewer()
                event.accept()

        self._detached_window = DetachedViewerWindow(self)
        self._detached_window.setWindowTitle("Solution Inspector - Detached Viewer")
        self._detached_window.resize(800, 600)

        # Move VTK widget to detached window
        self._detached_window.setCentralWidget(self.vtk_widget)
        self._detached_window.show()

        self.detach_action.setText("Re-attach Viewer")

    def _reattach_viewer(self) -> None:
        """Re-attach the VTK viewer to the main window."""
        if not hasattr(self, '_detached_window') or self._detached_window is None:
            return

        # Take widget back from detached window
        self._detached_window.takeCentralWidget()

        # Re-add to main layout
        if hasattr(self, '_main_layout_ref') and self._main_layout_ref is not None:
            # Insert at the original position (stretch=3 for the vtk widget)
            self._main_layout_ref.insertWidget(0, self.vtk_widget, stretch=3)
        else:
            # Fallback: just set parent
            self.vtk_widget.setParent(self)

        # Close and clean up detached window
        self._detached_window.close()
        self._detached_window = None
        self.detach_action.setText("Detach Viewer")

        # Refresh the view
        self._reload_view()

    def _reload_view(self) -> None:
        """Reload and refresh the 3D view."""
        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter is None:
            return

        # Re-render the current mesh
        if self._current_mesh is not None:
            self._render_current_mesh()
        else:
            plotter.reset_camera()
            plotter.render()

    def _toggle_side_panel(self, visible: bool = None) -> None:
        """Toggle visibility of the side panel."""
        if not hasattr(self, '_side_panel_scroll') or self._side_panel_scroll is None:
            return

        if visible is None:
            visible = not self._side_panel_scroll.isVisible()

        self._side_panel_scroll.setVisible(visible)
        self.side_panel_action.setChecked(visible)

    def _toggle_fullscreen(self, fullscreen: bool = None) -> None:
        """Toggle fullscreen mode."""
        if fullscreen is None:
            fullscreen = not self.isFullScreen()

        if fullscreen:
            self.showFullScreen()
        else:
            self.showNormal()

        self.fullscreen_action.setChecked(fullscreen)

    def _show_split_view(self) -> None:
        """Show split view for comparing timesteps."""
        if not self._time_values or len(self._time_values) < 2:
            QMessageBox.information(
                self, "Split View",
                "Split view requires a time-varying solution with multiple time steps."
            )
            return

        if not hasattr(self, '_split_view_dialog') or self._split_view_dialog is None:
            self._split_view_dialog = SplitViewDialog(self)

        self._split_view_dialog.set_time_values(
            self._time_values,
            self._load_timestep_mesh,
            self._current_scalar_name
        )
        self._split_view_dialog.show()
        self._split_view_dialog.raise_()
        self._split_view_dialog.activateWindow()
