"""
Parameter panel for configuring Tree and Forest generation.
"""
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QScrollArea, QFormLayout,
    QMessageBox, QProgressDialog, QSizePolicy,
    QToolButton, QToolTip
)
import numpy as np
import threading
from queue import Queue
from PySide6.QtCore import Signal, Qt, QTimer, QSignalBlocker, QSettings, QSize
from concurrent.futures import ThreadPoolExecutor
from svv.visualize.gui.theme import CADTheme, CADIcons
from svv.tree.data.data import TreeParameters
from svv.tree.data.units import UnitSystem, _LENGTH_UNITS, _MASS_UNITS, _TIME_UNITS
import traceback
from svv.telemetry import capture_exception, capture_message

# Descriptions shown in info-icon popups beside each parameter label.
_PARAM_DESCRIPTIONS = {
    'unit_system': (
        "Base units (Length, Mass, Time) for all dimensional parameters.\n"
        "Changing units will automatically convert existing values\n"
        "to maintain physical consistency."
    ),
    'n_vessels': (
        "Total number of terminal vessel segments to generate.\n"
        "Each vessel is a branch endpoint in the vascular tree.\n"
        "More vessels produce denser, more detailed networks."
    ),
    'physical_clearance': (
        "Minimum distance maintained between vessel walls during\n"
        "generation. Set to 0 to allow vessels to touch or overlap.\n"
        "Larger values create more spaced-out networks."
    ),
    'convexity_tolerance': (
        "Controls how strictly the domain convexity is checked when\n"
        "placing new vessels. Smaller values enforce stricter convexity,\n"
        "while larger values allow more flexibility near concave regions."
    ),
    'kinematic_viscosity': (
        "Dynamic viscosity divided by fluid density (\u03bd = \u03bc/\u03c1).\n"
        "Controls the viscous resistance of blood flow through vessels.\n"
        "Affects vessel radius optimization via Poiseuille's law."
    ),
    'fluid_density': (
        "Mass density of the fluid (blood). Used together with viscosity\n"
        "to compute flow resistance and pressure drops through the\n"
        "vascular network."
    ),
    'murray_exponent': (
        "Exponent in Murray's law governing the relationship between\n"
        "parent and child vessel radii at bifurcations. A value of 3\n"
        "corresponds to optimal laminar flow. Values > 3 favor thinner\n"
        "child branches."
    ),
    'radius_exponent': (
        "Exponent controlling how vessel radius scales with tree level.\n"
        "Affects the tapering profile of the vascular tree."
    ),
    'length_exponent': (
        "Exponent controlling how vessel segment lengths scale relative\n"
        "to their radii. Affects the overall shape and density of the\n"
        "tree structure."
    ),
    'terminal_pressure': (
        "Blood pressure at the terminal (outlet) ends of the vascular\n"
        "tree. Together with root pressure, determines the pressure\n"
        "gradient driving flow through the network."
    ),
    'root_pressure': (
        "Blood pressure at the root (inlet) of the vascular tree.\n"
        "The pressure difference between root and terminal pressures\n"
        "drives flow through the entire network."
    ),
    'terminal_flow': (
        "Volumetric flow rate at each terminal vessel segment.\n"
        "Determines how much blood exits at each branch endpoint.\n"
        "Total network flow equals terminal_flow \u00d7 number of terminals."
    ),
    'root_flow': (
        "Total volumetric flow rate entering at the root vessel.\n"
        "When set to Auto, computed as the sum of all terminal flows.\n"
        "Set manually to override the automatic calculation."
    ),
    'max_nonconvex_count': (
        "Maximum number of candidate positions sampled in non-convex\n"
        "regions before the algorithm moves on. Higher values allow\n"
        "better vessel placement in complex geometries but increase\n"
        "computation time."
    ),
    'n_networks': (
        "Number of independent vascular networks in the forest.\n"
        "Each network is a separate connected vascular system\n"
        "(e.g., arterial and venous networks)."
    ),
    'trees_per_network': (
        "Number of trees in each network. Trees within a network are\n"
        "later connected to form a complete vascular circuit. Each tree\n"
        "grows from its own start point."
    ),
    'vessels_per_network': (
        "Number of vessel-growth steps to run for each network.\n"
        "This controls how many vessels are added to every tree in the\n"
        "network during forest generation. Networks can stop growing\n"
        "earlier than others by setting a smaller value here."
    ),
    'connection_curve': (
        "Curve type used to connect trees within each network.\n"
        "\u2022 Bezier: Smooth polynomial curves\n"
        "\u2022 Catmull-Rom: Interpolating splines through control points\n"
        "\u2022 NURBS: Non-Uniform Rational B-Splines for maximum flexibility"
    ),
    'competition': (
        "When enabled, trees compete for territory during growth.\n"
        "Trees encountering another tree's territory stop expanding\n"
        "in that direction, creating natural boundaries between trees."
    ),
    'decay_probability': (
        "Controls how aggressively trees yield territory during\n"
        "competition. Higher values (closer to 1.0) make trees decay\n"
        "faster at boundaries, creating cleaner separations. Lower\n"
        "values allow more territorial overlap."
    ),
    'random_seed': (
        "Seed for the random number generator. Set to a specific value\n"
        "for reproducible results. 'Random' uses a different seed\n"
        "each run."
    ),
    'preallocation_size': (
        "Number of rows preallocated in memory for each tree's internal\n"
        "data arrays. Higher values support more vessels but consume\n"
        "more memory. Lower values are more memory-efficient but may\n"
        "limit the maximum tree size."
    ),
}


class ParameterPanel(QWidget):
    """
    Widget for configuring Tree and Forest generation parameters.
    """

    # Signals
    parameters_changed = Signal()
    length_unit_changed = Signal(str)  # Emitted when length unit changes (e.g., "cm", "mm", "m")

    def __init__(self, parent=None):
        """
        Initialize the parameter panel.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget (should be VascularizeGUI)
        """
        super().__init__(parent)
        self.parent_gui = parent

        self._connect_future = None
        self._connect_progress = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.tree_param_overrides = {}
        base_tree_params = TreeParameters()
        self.tree_unit_system = base_tree_params.unit_system
        self.tree_param_units = self._build_tree_param_units()
        self.tree_param_unit_labels = {}
        self.tree_base_params = self._default_tree_params(base_tree_params)
        self._generation_future = None
        self._generation_progress = None
        self._generation_progress_queue = None
        self._generation_cancel_event = None

        self._init_ui()

    def _record_telemetry(self, exc=None, message=None, level="error", traceback_str=None, **tags):
        """Send GUI warnings/errors to telemetry without impacting the UI.

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
            pass

    def shutdown(self):
        """
        Cancel any in-flight generation/connect tasks and shut down the worker
        executor so threads and memory are released when the application closes.
        """
        # Signal cancellation to any running tasks
        if getattr(self, "_generation_cancel_event", None) is not None:
            try:
                self._generation_cancel_event.set()
            except Exception:
                pass
        if getattr(self, "_connect_future", None) is not None:
            self._connect_future = None
        # Shut down executor threads
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.setLayout(layout)

        # Create scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(12)
        # Add right margin to account for scrollbar width (~16-20px)
        scroll_layout.setContentsMargins(0, 0, 20, 0)

        # Mode selection
        mode_group = QGroupBox("Generation Mode")
        mode_layout = QVBoxLayout()
        mode_group.setLayout(mode_layout)

        self.mode_combo = QComboBox()
        self.mode_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.mode_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.mode_combo.addItem("Tree (single)")
        self.mode_combo.addItem("Forest (multiple)")
        self.mode_combo.setMaxVisibleItems(100)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.mode_combo.setToolTip("Select whether to generate a single tree or a forest with multiple trees")
        mode_layout.addWidget(self.mode_combo)

        scroll_layout.addWidget(mode_group)

        # Tree parameters
        tree_params_group = QGroupBox("Tree Parameters")
        tree_form = QFormLayout()
        tree_form.setSpacing(10)
        tree_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        tree_params_group.setLayout(tree_form)

        # Unit system selection (dropdowns for base units)
        unit_row = QWidget()
        unit_layout = QHBoxLayout(unit_row)
        unit_layout.setContentsMargins(0, 0, 0, 0)
        unit_layout.setSpacing(6)

        self.length_unit_combo = QComboBox()
        self.length_unit_combo.addItems(sorted(_LENGTH_UNITS.keys()))
        self.length_unit_combo.setCurrentText(self.tree_unit_system.base.length.symbol)
        self.length_unit_combo.setToolTip("Base length unit used for TreeParameters values")
        self.length_unit_combo.currentTextChanged.connect(self._on_unit_system_changed)
        unit_layout.addWidget(QLabel("L:"))
        unit_layout.addWidget(self.length_unit_combo)

        self.mass_unit_combo = QComboBox()
        self.mass_unit_combo.addItems(sorted(_MASS_UNITS.keys()))
        self.mass_unit_combo.setCurrentText(self.tree_unit_system.base.mass.symbol)
        self.mass_unit_combo.setToolTip("Base mass unit used for TreeParameters values")
        self.mass_unit_combo.currentTextChanged.connect(self._on_unit_system_changed)
        unit_layout.addWidget(QLabel("M:"))
        unit_layout.addWidget(self.mass_unit_combo)

        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItems(sorted(_TIME_UNITS.keys()))
        self.time_unit_combo.setCurrentText(self.tree_unit_system.base.time.symbol)
        self.time_unit_combo.setToolTip("Base time unit used for TreeParameters values")
        self.time_unit_combo.currentTextChanged.connect(self._on_unit_system_changed)
        unit_layout.addWidget(QLabel("T:"))
        unit_layout.addWidget(self.time_unit_combo)

        unit_layout.addStretch()
        tree_form.addRow(self._make_label_with_info("Unit System:", _PARAM_DESCRIPTIONS['unit_system']), unit_row)

        # Number of vessels
        self.n_vessels_spin = QSpinBox()
        self.n_vessels_spin.setRange(1, 100000)
        self.n_vessels_spin.setValue(100)
        self.n_vessels_spin.setToolTip("Total number of vessel segments to generate")
        tree_form.addRow(
            self._make_label_with_info("Number of Vessels:", _PARAM_DESCRIPTIONS['n_vessels']),
            self.n_vessels_spin
        )

        # Physical clearance
        self.physical_clearance_spin = QDoubleSpinBox()
        self.physical_clearance_spin.setRange(0.0, 10.0)
        self.physical_clearance_spin.setSingleStep(0.01)
        self.physical_clearance_spin.setDecimals(4)
        self.physical_clearance_spin.setValue(0.0)
        self.physical_clearance_spin.setToolTip("Minimum distance between vessel walls (0 = allow touching)")
        tree_form.addRow(
            self._make_label_with_info("Physical Clearance:", _PARAM_DESCRIPTIONS['physical_clearance']),
            self.physical_clearance_spin
        )

        # Convexity tolerance
        self.convexity_tolerance_spin = QDoubleSpinBox()
        self.convexity_tolerance_spin.setRange(0.0, 1.0)
        self.convexity_tolerance_spin.setSingleStep(0.01)
        self.convexity_tolerance_spin.setDecimals(3)
        self.convexity_tolerance_spin.setValue(0.01)
        self.convexity_tolerance_spin.setToolTip("Tolerance for domain convexity checking (smaller = stricter)")
        tree_form.addRow(
            self._make_label_with_info("Convexity Tolerance:", _PARAM_DESCRIPTIONS['convexity_tolerance']),
            self.convexity_tolerance_spin
        )

        # Per-tree override selector (shown in forest mode)
        self.tree_override_widget = QWidget()
        override_layout = QHBoxLayout()
        override_layout.setContentsMargins(0, 0, 0, 0)
        override_layout.setSpacing(6)
        self.tree_override_widget.setLayout(override_layout)
        override_layout.addWidget(QLabel("Edit Tree:"))
        self.override_network_combo = QComboBox()
        self.override_network_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.override_network_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.override_network_combo.setMaxVisibleItems(100)
        self.override_tree_combo = QComboBox()
        self.override_tree_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.override_tree_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.override_tree_combo.setMaxVisibleItems(100)
        self.override_network_combo.currentIndexChanged.connect(self._on_tree_override_changed)
        self.override_tree_combo.currentIndexChanged.connect(self._on_tree_override_changed)
        override_layout.addWidget(QLabel("Network"))
        override_layout.addWidget(self.override_network_combo)
        override_layout.addWidget(QLabel("Tree"))
        override_layout.addWidget(self.override_tree_combo)
        override_layout.addStretch()
        tree_form.addRow(self.tree_override_widget)
        self.tree_override_widget.hide()

        # TreeParameters fields
        self.tree_param_widgets = {}
        self._add_tree_param_spin(tree_form, 'kinematic_viscosity', "Kinematic Viscosity", 0.0, 10.0, 5, 0.001, self.tree_param_units.get('kinematic_viscosity'), _PARAM_DESCRIPTIONS.get('kinematic_viscosity'))
        self._add_tree_param_spin(tree_form, 'fluid_density', "Fluid Density", 0.0, 20.0, 4, 0.01, self.tree_param_units.get('fluid_density'), _PARAM_DESCRIPTIONS.get('fluid_density'))
        self._add_tree_param_spin(tree_form, 'murray_exponent', "Murray Exponent", 0.5, 10.0, 3, 0.1, self.tree_param_units.get('murray_exponent'), _PARAM_DESCRIPTIONS.get('murray_exponent'))
        self._add_tree_param_spin(tree_form, 'radius_exponent', "Radius Exponent", 0.5, 10.0, 3, 0.1, self.tree_param_units.get('radius_exponent'), _PARAM_DESCRIPTIONS.get('radius_exponent'))
        self._add_tree_param_spin(tree_form, 'length_exponent', "Length Exponent", 0.1, 10.0, 3, 0.1, self.tree_param_units.get('length_exponent'), _PARAM_DESCRIPTIONS.get('length_exponent'))
        self._add_tree_param_spin(tree_form, 'terminal_pressure', "Terminal Pressure", 0.0, 1_000_000.0, 3, 10.0, self.tree_param_units.get('terminal_pressure'), _PARAM_DESCRIPTIONS.get('terminal_pressure'))
        self._add_tree_param_spin(tree_form, 'root_pressure', "Root Pressure", 0.0, 1_000_000.0, 3, 10.0, self.tree_param_units.get('root_pressure'), _PARAM_DESCRIPTIONS.get('root_pressure'))
        self._add_tree_param_spin(tree_form, 'terminal_flow', "Terminal Flow", 0.0, 100.0, 6, 0.0001, self.tree_param_units.get('terminal_flow'), _PARAM_DESCRIPTIONS.get('terminal_flow'))

        root_flow_layout = QHBoxLayout()
        self.root_flow_spin = QDoubleSpinBox()
        self.root_flow_spin.setRange(0.0, 100.0)
        self.root_flow_spin.setDecimals(6)
        self.root_flow_spin.setSingleStep(0.0001)
        self.root_flow_spin.valueChanged.connect(lambda v: self._on_tree_param_changed('root_flow', v))
        self.root_flow_auto_cb = QCheckBox("Auto")
        self.root_flow_auto_cb.setChecked(True)
        self.root_flow_auto_cb.stateChanged.connect(self._on_root_flow_auto_toggled)
        root_flow_layout.addWidget(self.root_flow_spin)
        root_flow_layout.addWidget(self.root_flow_auto_cb)
        root_flow_unit = self._make_unit_label(self.tree_param_units.get('root_flow', ''))
        root_flow_layout.addWidget(root_flow_unit)
        root_flow_layout.addStretch()
        self.tree_param_unit_labels['root_flow'] = root_flow_unit
        tree_form.addRow(self._make_label_with_info("Root Flow:", _PARAM_DESCRIPTIONS['root_flow']), root_flow_layout)
        self.tree_param_widgets['root_flow'] = self.root_flow_spin

        self.max_nonconvex_spin = QSpinBox()
        self.max_nonconvex_spin.setRange(0, 1000000)
        self.max_nonconvex_spin.setSingleStep(10)
        self.max_nonconvex_spin.valueChanged.connect(lambda v: self._on_tree_param_changed('max_nonconvex_count', v))
        max_nonconvex_layout = QHBoxLayout()
        max_nonconvex_layout.setContentsMargins(0, 0, 0, 0)
        max_nonconvex_layout.setSpacing(6)
        max_nonconvex_layout.addWidget(self.max_nonconvex_spin)
        max_nonconvex_unit = self._make_unit_label(self.tree_param_units.get('max_nonconvex_count', ''))
        max_nonconvex_layout.addWidget(max_nonconvex_unit)
        max_nonconvex_layout.addStretch()
        self.tree_param_unit_labels['max_nonconvex_count'] = max_nonconvex_unit
        tree_form.addRow(self._make_label_with_info("Max Nonconvex Count:", _PARAM_DESCRIPTIONS['max_nonconvex_count']), max_nonconvex_layout)
        self.tree_param_widgets['max_nonconvex_count'] = self.max_nonconvex_spin

        scroll_layout.addWidget(tree_params_group)

        # Forest parameters (initially hidden)
        self.forest_params_group = QGroupBox("Forest Parameters")
        forest_form = QFormLayout()
        forest_form.setSpacing(10)
        forest_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.forest_params_group.setLayout(forest_form)

        # Number of networks (Forest only)
        self.n_networks_spin = QSpinBox()
        self.n_networks_spin.setRange(1, 16)
        self.n_networks_spin.setValue(1)
        self.n_networks_spin.setToolTip("Number of independent vascular networks to generate in the forest")
        self.n_networks_spin.valueChanged.connect(self._on_forest_networks_changed)
        forest_form.addRow(self._make_label_with_info("Networks:", _PARAM_DESCRIPTIONS['n_networks']), self.n_networks_spin)

        # Trees per network container (Forest only) – one spinner per network
        self._n_trees_per_network = [2]  # default: 1 network with 2 trees
        self._trees_per_network_widget = QWidget()
        self._trees_per_network_layout = QVBoxLayout()
        self._trees_per_network_layout.setContentsMargins(0, 0, 0, 0)
        self._trees_per_network_layout.setSpacing(4)
        self._trees_per_network_widget.setLayout(self._trees_per_network_layout)
        self._trees_per_network_spins = []
        forest_form.addRow(self._make_label_with_info("Trees per Network:", _PARAM_DESCRIPTIONS['trees_per_network']), self._trees_per_network_widget)
        self._rebuild_trees_per_network_spins(1)

        # Vessels per network container (Forest only) – one spinner per network
        self._n_vessels_per_network = [self.n_vessels_spin.value()]
        self._vessels_per_network_widget = QWidget()
        self._vessels_per_network_layout = QVBoxLayout()
        self._vessels_per_network_layout.setContentsMargins(0, 0, 0, 0)
        self._vessels_per_network_layout.setSpacing(4)
        self._vessels_per_network_widget.setLayout(self._vessels_per_network_layout)
        self._vessels_per_network_spins = []
        forest_form.addRow(
            self._make_label_with_info("Vessels per Network:", _PARAM_DESCRIPTIONS['vessels_per_network']),
            self._vessels_per_network_widget
        )
        self._rebuild_vessels_per_network_spins(1)

        # Curve type used when connecting trees into networks
        self.curve_type_combo = QComboBox()
        # Display labels mapped to internal curve_type strings used by svv.forest.connect.curve.Curve
        self.curve_type_combo.addItem("Bezier", userData="Bezier")
        self.curve_type_combo.addItem("Catmull-Rom", userData="CatmullRom")
        self.curve_type_combo.addItem("NURBS", userData="NURBS")
        self.curve_type_combo.setCurrentIndex(0)
        self.curve_type_combo.setToolTip("Curve family used when connecting forest trees into networks")
        forest_form.addRow(self._make_label_with_info("Connection Curve:", _PARAM_DESCRIPTIONS['connection_curve']), self.curve_type_combo)

        self.compete_cb = QCheckBox("Enable Competition")
        self.compete_cb.setToolTip("Enable competition between trees for territory")
        forest_form.addRow(self._make_label_with_info("Competition:", _PARAM_DESCRIPTIONS['competition']), self.compete_cb)

        self.decay_probability_spin = QDoubleSpinBox()
        self.decay_probability_spin.setRange(0.0, 1.0)
        self.decay_probability_spin.setSingleStep(0.05)
        self.decay_probability_spin.setDecimals(2)
        self.decay_probability_spin.setValue(0.9)
        self.decay_probability_spin.setToolTip("Probability of decay for competitive growth (0-1)")
        forest_form.addRow(
            self._make_label_with_info("Decay Probability:", _PARAM_DESCRIPTIONS['decay_probability']),
            self.decay_probability_spin
        )

        scroll_layout.addWidget(self.forest_params_group)
        self.forest_params_group.hide()

        # Advanced parameters
        advanced_group = QGroupBox("Advanced Parameters")
        advanced_form = QFormLayout()
        advanced_form.setSpacing(10)
        advanced_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        advanced_group.setLayout(advanced_form)

        self.random_seed_spin = QSpinBox()
        self.random_seed_spin.setRange(-1, 999999)
        self.random_seed_spin.setValue(-1)
        self.random_seed_spin.setSpecialValueText("Random")
        self.random_seed_spin.setToolTip("Set random seed for reproducible results (-1 = random)")
        advanced_form.addRow(
            self._make_label_with_info("Random Seed:", _PARAM_DESCRIPTIONS['random_seed']),
            self.random_seed_spin
        )

        # Preallocation size (rows) for tree data
        self.preallocation_spin = QSpinBox()
        self.preallocation_spin.setRange(1000, 20000000)
        self.preallocation_spin.setSingleStep(10000)
        self.preallocation_spin.setValue(int(1e5))
        self.preallocation_spin.setToolTip(
            "Maximum number of rows preallocated for each tree's data array.\n"
            "Lower values reduce memory usage but may trigger resize operations during growth."
        )
        advanced_form.addRow(
            self._make_label_with_info("Preallocation Size (rows):", _PARAM_DESCRIPTIONS['preallocation_size']),
            self.preallocation_spin
        )

        scroll_layout.addWidget(advanced_group)

        # Add stretch to push everything to top
        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # Initialize tree parameter widgets with defaults
        default_trees = 2
        if self.parent_gui and hasattr(self.parent_gui, "point_selector"):
            try:
                default_trees = self.parent_gui.point_selector.tree_combo.count()
            except Exception:
                default_trees = 2
        self._sync_tree_override_options(1, [default_trees])
        self._apply_params_to_widgets(self.tree_base_params)
        self._refresh_unit_labels()

        # Ensure the initial mode (single tree vs forest) is reflected in the
        # point selector's Tree dropdown and parameter visibility.
        try:
            self._on_mode_changed(self.mode_combo.currentIndex())
        except Exception:
            pass

        # Action buttons
        button_layout = QVBoxLayout()
        button_layout.setSpacing(8)

        self.generate_btn = QPushButton("Generate Tree/Forest")
        self.generate_btn.clicked.connect(self._generate)
        self.generate_btn.setObjectName("primaryButton")
        self.generate_btn.setToolTip("Generate vascular tree or forest with current parameters")
        button_layout.addWidget(self.generate_btn)

        self.connect_now_btn = QPushButton("Connect Forest")
        self.connect_now_btn.clicked.connect(self._connect_existing_forest)
        self.connect_now_btn.setObjectName("primaryButton")
        self.connect_now_btn.setToolTip("Connect the currently generated forest into a network")
        self.connect_now_btn.hide()
        button_layout.addWidget(self.connect_now_btn)

        self.export_btn = QPushButton("Export Configuration")
        self.export_btn.clicked.connect(self._export_config)
        self.export_btn.setObjectName("secondaryButton")
        self.export_btn.setToolTip("Export current configuration to JSON file")
        button_layout.addWidget(self.export_btn)

        layout.addLayout(button_layout)

    # ---- Tree parameter helpers ----
    def _build_tree_param_units(self):
        """Return mapping of TreeParameters to their display units."""
        us = self.tree_unit_system
        return {
            'kinematic_viscosity': us.kinematic_viscosity.symbol,
            'fluid_density': us.density.symbol,
            'murray_exponent': 'unitless',
            'radius_exponent': 'unitless',
            'length_exponent': 'unitless',
            'terminal_pressure': us.pressure.symbol,
            'root_pressure': us.pressure.symbol,
            'terminal_flow': us.volumetric_flow.symbol,
            'root_flow': us.volumetric_flow.symbol,
            'max_nonconvex_count': 'count'
        }

    def _unit_system_summary(self):
        base = self.tree_unit_system.base
        return (
            f"Length: {base.length.symbol}, "
            f"Mass: {base.mass.symbol}, "
            f"Time: {base.time.symbol} "
            f"(Pressure: {self.tree_unit_system.pressure.symbol}, "
            f"Flow: {self.tree_unit_system.volumetric_flow.symbol})"
        )

    def _make_unit_label(self, text):
        label = QLabel(text or "")
        label.setStyleSheet("color: #6a6a6a;")
        return label

    def _make_info_button(self, description):
        """Create a small info-circle button that shows *description* on click."""
        btn = QToolButton()
        btn.setIcon(CADIcons.get_icon(
            "info",
            color=CADTheme.get_color('text', 'secondary'),
            size=14,
        ))
        btn.setIconSize(QSize(14, 14))
        btn.setFixedSize(18, 18)
        btn.setCursor(Qt.WhatsThisCursor)
        btn.setStyleSheet(
            "QToolButton { border: none; background: transparent; padding: 0px; }"
            "QToolButton:hover { background: rgba(255,255,255,0.08); border-radius: 9px; }"
        )
        btn.clicked.connect(
            lambda _=None, d=description, b=btn: QToolTip.showText(
                b.mapToGlobal(b.rect().bottomLeft()), d, b
            )
        )
        return btn

    def _make_label_with_info(self, text, description):
        """Return a QWidget containing a QLabel and a small info icon button."""
        widget = QWidget()
        lay = QHBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        widget.setLayout(lay)
        lay.addWidget(QLabel(text))
        lay.addWidget(self._make_info_button(description))
        lay.addStretch()
        return widget

    def _refresh_unit_labels(self):
        self.tree_param_units = self._build_tree_param_units()
        for key, label in self.tree_param_unit_labels.items():
            if label is None:
                continue
            label.setText(self.tree_param_units.get(key, ''))

    def _default_tree_params(self, template=None):
        """Return default tree parameter values as a dict."""
        tp = template or TreeParameters(unit_system=self.tree_unit_system)
        return {
            'kinematic_viscosity': tp.kinematic_viscosity,
            'fluid_density': tp.fluid_density,
            'murray_exponent': tp.murray_exponent,
            'radius_exponent': tp.radius_exponent,
            'length_exponent': tp.length_exponent,
            'terminal_pressure': tp.terminal_pressure,
            'root_pressure': tp.root_pressure,
            'terminal_flow': tp.terminal_flow,
            'root_flow': tp.root_flow,
            'max_nonconvex_count': tp.max_nonconvex_count
        }

    def _on_unit_system_changed(self):
        """
        Handle changes to the base unit system from the dropdowns.

        This updates self.tree_unit_system and converts existing dimensional
        TreeParameters into the new units so values remain consistent.
        """
        length_symbol = self.length_unit_combo.currentText()
        mass_symbol = self.mass_unit_combo.currentText()
        time_symbol = self.time_unit_combo.currentText()

        # Build a new UnitSystem with selected base units
        new_unit_system = UnitSystem(length=length_symbol, mass=mass_symbol, time=time_symbol)

        # Update base params to use the new unit system, converting stored values
        tp = TreeParameters(unit_system=self.tree_unit_system)
        # Rehydrate tp from tree_base_params
        for key, value in self.tree_base_params.items():
            if value is None:
                continue
            tp.set(key, value)
        tp.set_unit_system(new_unit_system, convert_existing=True)

        # Update panel state
        self.tree_unit_system = new_unit_system
        self.tree_base_params = self._default_tree_params(tp)
        self._apply_params_to_widgets(self.tree_base_params)
        self._refresh_unit_labels()

        # Notify listeners that the length unit has changed
        self.length_unit_changed.emit(length_symbol)

    def _add_tree_param_spin(self, form, key, label, min_v, max_v, decimals, step, unit_text=None, info_text=None):
        """Create and register a QDoubleSpinBox for a tree parameter."""
        spin = QDoubleSpinBox()
        spin.setRange(min_v, max_v)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.valueChanged.connect(lambda v, k=key: self._on_tree_param_changed(k, v))
        widget = spin
        if unit_text is not None:
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            row_layout.addWidget(spin)
            unit_label = self._make_unit_label(unit_text)
            row_layout.addWidget(unit_label)
            row_layout.addStretch()
            self.tree_param_unit_labels[key] = unit_label
            widget = row_widget
        row_label = (
            self._make_label_with_info(label + ":", info_text)
            if info_text else label + ":"
        )
        form.addRow(row_label, widget)
        self.tree_param_widgets[key] = spin

    def _apply_params_to_widgets(self, values):
        """Load parameter values into UI controls."""
        for key, widget in self.tree_param_widgets.items():
            if key not in values:
                continue
            with QSignalBlocker(widget):
                widget.setValue(values[key] if values[key] is not None else widget.minimum())
        if 'root_flow' in values:
            rf_val = values['root_flow']
            if rf_val is None:
                with QSignalBlocker(self.root_flow_auto_cb):
                    self.root_flow_auto_cb.setChecked(True)
                with QSignalBlocker(self.root_flow_spin):
                    self.root_flow_spin.setEnabled(False)
                    self.root_flow_spin.setValue(self.root_flow_spin.minimum())
            else:
                with QSignalBlocker(self.root_flow_auto_cb):
                    self.root_flow_auto_cb.setChecked(False)
                with QSignalBlocker(self.root_flow_spin):
                    self.root_flow_spin.setEnabled(True)
                    self.root_flow_spin.setValue(rf_val)
        if 'max_nonconvex_count' in values:
            with QSignalBlocker(self.max_nonconvex_spin):
                self.max_nonconvex_spin.setValue(values['max_nonconvex_count'])

    def _current_override_key(self):
        """Return (network, tree) when in forest mode, else None."""
        if self.mode_combo.currentIndex() != 1:  # forest index
            return None
        net = self.override_network_combo.currentIndex()
        tree = self.override_tree_combo.currentIndex()
        if net < 0 or tree < 0:
            return None
        return (net, tree)

    def _on_tree_override_changed(self, *_):
        """Reload widgets when a different tree override is selected."""
        # Ensure tree combo matches selected network's tree count if available
        if hasattr(self, "_trees_per_network"):
            net = self.override_network_combo.currentIndex()
            desired = 0
            if 0 <= net < len(self._trees_per_network):
                desired = self._trees_per_network[net]
            self._ensure_tree_combo_count(desired)
        self._load_tree_params_for_selected()

    def _load_tree_params_for_selected(self):
        """Apply the appropriate parameter set for the selected tree or base."""
        key = self._current_override_key()
        if key is None:
            self._apply_params_to_widgets(self.tree_base_params)
            return
        values = self.tree_param_overrides.get(key, self.tree_base_params)
        self._apply_params_to_widgets(values)

    def _on_root_flow_auto_toggled(self, state):
        """Handle toggling of root flow auto mode."""
        is_auto = state == Qt.Checked
        self.root_flow_spin.setEnabled(not is_auto)
        if is_auto:
            self._on_tree_param_changed('root_flow', None)
        else:
            self._on_tree_param_changed('root_flow', self.root_flow_spin.value())

    def _on_tree_param_changed(self, key, value):
        """Update stored parameters when a field changes."""
        if key == 'root_flow' and self.root_flow_auto_cb.isChecked():
            value = None
        if self.mode_combo.currentIndex() == 0:
            # Single tree / base params
            self.tree_base_params[key] = value
        else:
            tree_key = self._current_override_key()
            if tree_key is None:
                return
            if tree_key not in self.tree_param_overrides:
                # Start from base values
                self.tree_param_overrides[tree_key] = dict(self.tree_base_params)
            self.tree_param_overrides[tree_key][key] = value

    def _sync_tree_override_options(self, n_networks, n_trees_per_network):
        """Update the override dropdowns based on current forest layout."""
        self._trees_per_network = list(n_trees_per_network)
        with QSignalBlocker(self.override_network_combo):
            self.override_network_combo.clear()
            for i in range(n_networks):
                self.override_network_combo.addItem(f"Network {i}")
        with QSignalBlocker(self.override_tree_combo):
            self.override_tree_combo.clear()
            max_trees = max(n_trees_per_network) if n_trees_per_network else 0
            for j in range(max_trees):
                self.override_tree_combo.addItem(f"Tree {j}")
        if n_trees_per_network:
            self._ensure_tree_combo_count(n_trees_per_network[0])
        # Prune overrides that no longer fit
        valid_keys = set()
        for net_idx, n_t in enumerate(n_trees_per_network):
            for t_idx in range(n_t):
                valid_keys.add((net_idx, t_idx))
        self.tree_param_overrides = {
            k: v for k, v in self.tree_param_overrides.items() if k in valid_keys
        }
        self._load_tree_params_for_selected()

    def _ensure_tree_combo_count(self, count):
        """Ensure the tree combo has the desired number of entries."""
        with QSignalBlocker(self.override_tree_combo):
            current = self.override_tree_combo.count()
            if count < 0:
                count = 0
            if current != count:
                self.override_tree_combo.clear()
                for i in range(count):
                    self.override_tree_combo.addItem(f"Tree {i}")

    def _on_point_selector_network_changed(self, value):
        """Sync override combos when point selector network count changes."""
        # When the user changes the network count in the point selector, keep
        # override combos and per-network spinners in sync.
        if hasattr(self, "_trees_per_network_spins") and self.forest_params_group.isVisible():
            with QSignalBlocker(self.n_networks_spin):
                self.n_networks_spin.setValue(value)
            self._rebuild_trees_per_network_spins(value)
            self._rebuild_vessels_per_network_spins(value)
            trees_per_network = list(self._n_trees_per_network)
        else:
            try:
                n_trees = self.parent_gui.point_selector.tree_combo.count()
            except Exception:
                n_trees = 2
            trees_per_network = [n_trees] * value
        if self.parent_gui and hasattr(self.parent_gui, "point_selector"):
            try:
                self.parent_gui.point_selector.set_trees_per_network(trees_per_network)
            except Exception:
                pass
        self._sync_tree_override_options(value, trees_per_network)

    def _build_tree_parameters(self, values):
        """Create a TreeParameters instance from a value dict."""
        tp = TreeParameters(unit_system=self.tree_unit_system)
        tp.kinematic_viscosity = values.get('kinematic_viscosity', tp.kinematic_viscosity)
        tp.fluid_density = values.get('fluid_density', tp.fluid_density)
        tp.murray_exponent = values.get('murray_exponent', tp.murray_exponent)
        tp.radius_exponent = values.get('radius_exponent', tp.radius_exponent)
        tp.length_exponent = values.get('length_exponent', tp.length_exponent)
        tp.terminal_pressure = values.get('terminal_pressure', tp.terminal_pressure)
        tp.root_pressure = values.get('root_pressure', tp.root_pressure)
        tp.terminal_flow = values.get('terminal_flow', tp.terminal_flow)
        tp.root_flow = values.get('root_flow', tp.root_flow)
        tp.max_nonconvex_count = values.get('max_nonconvex_count', tp.max_nonconvex_count)
        return tp

    # ---- Forest layout helpers ----
    def _on_forest_networks_changed(self, value):
        """
        Handle changes to the number of networks for forest generation.

        This rebuilds the per-network tree count spinners, updates the point
        selector's network dropdown, and syncs the per-tree override dropdowns.
        """
        if self.parent_gui and hasattr(self.parent_gui, "point_selector"):
            ps = self.parent_gui.point_selector
            try:
                if hasattr(ps, "set_network_count"):
                    ps.set_network_count(value)
            except Exception:
                pass
        self._rebuild_trees_per_network_spins(value)
        self._rebuild_vessels_per_network_spins(value)
        if self.parent_gui and hasattr(self.parent_gui, "point_selector"):
            try:
                self.parent_gui.point_selector.set_trees_per_network(list(self._n_trees_per_network))
            except Exception:
                pass
        self._sync_tree_override_options(value, list(self._n_trees_per_network))

    def _rebuild_trees_per_network_spins(self, n_networks):
        """Rebuild the per-network tree count spinners for the given number of networks."""
        old_values = list(self._n_trees_per_network)

        # Clear existing widgets
        while self._trees_per_network_layout.count():
            item = self._trees_per_network_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._trees_per_network_spins = []

        new_values = []
        for i in range(n_networks):
            default_val = old_values[i] if i < len(old_values) else 2
            new_values.append(default_val)

            row = QWidget()
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)
            row.setLayout(row_layout)

            if n_networks > 1:
                row_layout.addWidget(QLabel(f"Network {i}:"))

            spin = QSpinBox()
            spin.setRange(1, 16)
            spin.setValue(default_val)
            spin.setToolTip(f"Number of trees in network {i}")
            spin.valueChanged.connect(lambda v, idx=i: self._on_per_network_trees_changed(idx, v))
            row_layout.addWidget(spin)
            row_layout.addStretch()

            self._trees_per_network_layout.addWidget(row)
            self._trees_per_network_spins.append(spin)

        self._n_trees_per_network = new_values

    def _rebuild_vessels_per_network_spins(self, n_networks):
        """Rebuild the per-network vessel-count spinners for the given number of networks."""
        old_values = list(getattr(self, "_n_vessels_per_network", []))

        # Clear existing widgets
        while self._vessels_per_network_layout.count():
            item = self._vessels_per_network_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._vessels_per_network_spins = []

        default_base = int(self.n_vessels_spin.value()) if hasattr(self, "n_vessels_spin") else 100
        new_values = []
        for i in range(n_networks):
            default_val = int(old_values[i]) if i < len(old_values) else default_base
            default_val = max(1, default_val)
            new_values.append(default_val)

            row = QWidget()
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)
            row.setLayout(row_layout)

            if n_networks > 1:
                row_layout.addWidget(QLabel(f"Network {i}:"))

            spin = QSpinBox()
            spin.setRange(1, 100000)
            spin.setValue(default_val)
            spin.setToolTip(f"Number of vessel-growth steps for network {i}")
            spin.valueChanged.connect(lambda v, idx=i: self._on_per_network_vessels_changed(idx, v))
            row_layout.addWidget(spin)
            row_layout.addStretch()

            self._vessels_per_network_layout.addWidget(row)
            self._vessels_per_network_spins.append(spin)

        self._n_vessels_per_network = new_values

    def _on_per_network_vessels_changed(self, network_idx, value):
        """Handle a change to the vessel count for a specific network."""
        value = max(1, int(value))
        if network_idx < len(self._n_vessels_per_network):
            self._n_vessels_per_network[network_idx] = value

    def _on_per_network_trees_changed(self, network_idx, value):
        """Handle a change to the tree count for a specific network."""
        value = max(1, value)
        if network_idx < len(self._n_trees_per_network):
            self._n_trees_per_network[network_idx] = value

        # Update point selector
        if self.parent_gui and hasattr(self.parent_gui, "point_selector"):
            try:
                self.parent_gui.point_selector.set_trees_per_network(list(self._n_trees_per_network))
            except Exception:
                pass

        # Sync override dropdowns
        n_networks = self.n_networks_spin.value() if hasattr(self, "n_networks_spin") else 1
        self._sync_tree_override_options(n_networks, list(self._n_trees_per_network))

    def _params_for_tree(self, net_idx, tree_idx):
        """Merge base params with per-tree override."""
        params = dict(self.tree_base_params)
        override = self.tree_param_overrides.get((net_idx, tree_idx))
        if override:
            params.update(override)
        return params

    def _on_mode_changed(self, index):
        """Handle mode selection change."""
        if index == 0:  # Single Tree
            self.forest_params_group.hide()
            self.tree_override_widget.hide()
            # When switching back to single tree, show base parameters
            self._apply_params_to_widgets(self.tree_base_params)
            # In single-tree mode, force a single network and a single tree
            if self.parent_gui and hasattr(self.parent_gui, "point_selector"):
                ps = self.parent_gui.point_selector
                try:
                    if hasattr(ps, "set_network_count"):
                        ps.set_network_count(1)
                    ps.set_tree_count(1)
                except Exception:
                    pass
        else:  # Forest
            self.forest_params_group.show()
            self.tree_override_widget.show()
            # Sync override selectors with current point configuration
            if self.parent_gui and hasattr(self.parent_gui, "point_selector"):
                try:
                    n_networks = self.n_networks_spin.value()
                    trees_per_network = list(self._n_trees_per_network)
                    # Ensure list length matches current network count
                    while len(trees_per_network) < n_networks:
                        trees_per_network.append(2)
                    trees_per_network = trees_per_network[:n_networks]
                    # Update point selector's controls to reflect forest settings
                    ps = self.parent_gui.point_selector
                    if hasattr(ps, "set_network_count"):
                        ps.set_network_count(n_networks)
                    ps.set_trees_per_network(trees_per_network)
                    self._sync_tree_override_options(n_networks, trees_per_network)
                except Exception:
                    pass
            self._load_tree_params_for_selected()

        # Hide manual connect button until a forest is generated
        self._update_connect_button_visibility(force_hide=True)

    def _update_connect_button_visibility(self, force_hide=False):
        """
        Show or hide the manual forest connect button based on current state.

        Parameters
        ----------
        force_hide : bool
            If True, always hide the button regardless of current forest state.
        """
        if not hasattr(self, 'connect_now_btn'):
            return

        if force_hide:
            self.connect_now_btn.hide()
            self.connect_now_btn.setEnabled(True)
            return

        forest = self.parent_gui.forest if self.parent_gui else None
        should_show = forest is not None and getattr(forest, 'connections', None) is None
        self.connect_now_btn.setVisible(should_show)
        self.connect_now_btn.setEnabled(should_show)

    def _generate(self):
        """Generate Tree or Forest based on current configuration."""
        # Reset connect button until we know a fresh forest state
        self._update_connect_button_visibility(force_hide=True)

        if not self.parent_gui or not self.parent_gui.domain:
            QMessageBox.warning(
                self,
                "No Domain",
                "Please load a domain file before generating trees.\n\n"
                "Use File > Load Domain to get started."
            )
            self._record_telemetry(
                message="Generation requested without a loaded domain",
                level="warning",
                action="generate_without_domain",
            )
            return

        mode = self.mode_combo.currentIndex()
        n_vessels = self.n_vessels_spin.value()
        physical_clearance = self.physical_clearance_spin.value()
        convexity_tolerance = self.convexity_tolerance_spin.value()
        preallocation_step = self.preallocation_spin.value()
        random_seed = self.random_seed_spin.value() if self.random_seed_spin.value() >= 0 else None
        point_config = self.parent_gui.point_selector.get_configuration()

        if mode == 0:  # Single Tree
            if self.parent_gui:
                self.parent_gui.update_status(f"Generating tree with {n_vessels} vessels...")
            task_fn = lambda cancel, queue: self._generate_tree_task(
                n_vessels,
                physical_clearance,
                convexity_tolerance,
                preallocation_step,
                random_seed,
                point_config,
                cancel,
                queue
            )
            self._start_generation_task("Generating tree...", n_vessels, task_fn, self._on_tree_generated)
        else:  # Forest
            compete = self.compete_cb.isChecked()
            decay_prob = self.decay_probability_spin.value()
            n_networks = point_config.get('n_networks', 1)
            n_trees_per_network = point_config.get('n_trees_per_network', [2] * n_networks)
            self._sync_tree_override_options(n_networks, n_trees_per_network)
            per_network = list(getattr(self, "_n_vessels_per_network", []))
            if len(per_network) != n_networks:
                per_network = [int(n_vessels)] * int(n_networks)
            total_steps = max(per_network) if per_network else int(n_vessels)
            if self.parent_gui:
                if n_networks <= 4:
                    self.parent_gui.update_status(f"Generating forest (vessels/network: {per_network})...")
                else:
                    self.parent_gui.update_status(
                        f"Generating forest (vessels/network: min {min(per_network)} max {max(per_network)})..."
                    )
            task_fn = lambda cancel, queue: self._generate_forest_task(
                per_network,
                physical_clearance,
                convexity_tolerance,
                preallocation_step,
                random_seed,
                compete,
                decay_prob,
                point_config,
                cancel,
                queue
            )
            self._start_generation_task("Generating forest...", total_steps, task_fn, self._on_forest_generated)

    @staticmethod
    def _connect_forest_task(forest, curve_type="Bezier"):
        """
        Run forest.connect() in a worker context and return any exception.

        Returns
        -------
        Exception or None
            None on success, otherwise the raised exception.
        """
        try:
            forest.connect(curve_type=curve_type)
            return None
        except Exception as exc:  # Capture so GUI thread can handle it gracefully
            forest.connections = None
            # Report the exception to telemetry (if enabled) so background
            # connect failures are visible even when handled gracefully.
            try:
                try:
                    import sentry_sdk  # type: ignore[import]

                    with sentry_sdk.push_scope() as scope:
                        scope.set_tag("action", "forest_connect_worker")
                        sentry_sdk.capture_exception(exc)
                except Exception:
                    capture_exception(exc)
            except Exception:
                pass
            return exc

    def _connect_existing_forest(self):
        """Connect the currently generated forest if it is not already connected."""
        if not self.parent_gui or not getattr(self.parent_gui, 'forest', None):
            self._record_telemetry(
                message="Forest connect requested but no forest generated",
                level="warning",
                action="forest_connect_no_forest",
            )
            QMessageBox.information(
                self,
                "No Forest",
                "Generate a forest before attempting to connect it."
            )
            self._update_connect_button_visibility(force_hide=True)
            return

        forest = self.parent_gui.forest
        if getattr(forest, 'connections', None) is not None:
            self._record_telemetry(
                message="Forest connect requested but already connected",
                level="info",
                action="forest_connect_already_connected",
            )
            QMessageBox.information(
                self,
                "Already Connected",
                "The current forest is already connected."
            )
            self._update_connect_button_visibility()
            return

        # Run connection asynchronously to keep the UI responsive
        self._start_forest_connect(forest, show_success_dialog=True)

    def _start_forest_connect(self, forest, show_success_dialog=False):
        """
        Launch forest.connect() on a worker thread and show a non-blocking progress dialog.

        Parameters
        ----------
        forest : svv.forest.forest.Forest
            Forest object to connect
        show_success_dialog : bool
            If True, show an information dialog on success.
        """
        if self._connect_future is not None:
            return  # Connection already in progress

        # Record last action for crash diagnostics so that if the process is
        # killed during forest connection, the next GUI launch can report that
        # this was the last in-flight operation.
        try:
            settings = QSettings("svVascularize", "GUI")
            settings.setValue("session/last_action", "forest_connect")
        except Exception:
            pass

        if self.parent_gui and hasattr(self.parent_gui, "show_progress"):
            self.parent_gui.show_progress("Connecting forest...")
        if hasattr(self, 'connect_now_btn'):
            self.connect_now_btn.setEnabled(False)

        # Determine desired curve family from the GUI selector; default to Bezier
        curve_type = "Bezier"
        try:
            if hasattr(self, "curve_type_combo") and self.curve_type_combo is not None:
                data = self.curve_type_combo.currentData()
                curve_type = data if data is not None else self.curve_type_combo.currentText()
        except Exception:
            curve_type = "Bezier"

        future = self._executor.submit(self._connect_forest_task, forest, curve_type)
        self._connect_future = future
        self._poll_connect_future(future, forest, show_success_dialog)

    def _poll_connect_future(self, future, forest, show_success_dialog):
        """Poll a QtConcurrent future without blocking the UI."""
        if future.done():
            self._on_connect_finished(future, forest, show_success_dialog)
        else:
            QTimer.singleShot(50, lambda: self._poll_connect_future(future, forest, show_success_dialog))

    def _on_connect_finished(self, future, forest, show_success_dialog):
        """Handle completion of the background forest connection."""
        exc = None
        self._connect_future = None
        if self.parent_gui and hasattr(self.parent_gui, "hide_progress"):
            self.parent_gui.hide_progress()

        try:
            exc = future.result()
        except Exception as e:
            exc = e

        if exc:
            # Reset last_action now that connect has completed (even if it
            # failed gracefully inside Python). This ensures that only true
            # process-killing crashes during connect are reported on next run.
            try:
                settings = QSettings("svVascularize", "GUI")
                current = settings.value("session/last_action", "", type=str)
                if current == "forest_connect":
                    settings.setValue("session/last_action", "")
            except Exception:
                pass
            # Ensure connect failures are visible in telemetry even though the
            # GUI handles them with a warning dialog.
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self._record_telemetry(exc, action="forest_connect", traceback_str=tb)
            QMessageBox.warning(
                self,
                "Forest Connect Warning",
                f"Forest connection failed:\n\n{exc}"
            )
            if self.parent_gui:
                self.parent_gui.log_output(f"[Forest] connect failed: {exc}")
                self.parent_gui.update_status("Forest connection failed")
        else:
            if self.parent_gui:
                self.parent_gui.log_output("[Forest] connected")
                self.parent_gui.update_status("Forest connected")
                try:
                    self.parent_gui.vtk_widget.draw_forest(forest)
                except Exception:
                    # Keep UI responsive even if redraw fails
                    pass
            if show_success_dialog:
                QMessageBox.information(
                    self,
                    "Forest Connected",
                    "Forest connections created successfully."
                )

        if hasattr(self, 'connect_now_btn'):
            self.connect_now_btn.setEnabled(True)
        self._update_connect_button_visibility()

    # ---- Long-running task helpers ----
    def _start_generation_task(self, title, total_steps, task_fn, on_success):
        if self._generation_future is not None:
            self._record_telemetry(
                message="Generation requested while another generation is running",
                level="info",
                action="generation_already_running",
            )
            QMessageBox.information(self, "Task Running", "Please wait for the current operation to finish.")
            return
        if self._connect_future is not None:
            self._record_telemetry(
                message="Generation requested while forest connection is running",
                level="info",
                action="generation_connect_running",
            )
            QMessageBox.information(self, "Forest Connection Running", "Please wait for the forest connection to finish.")
            return

        self._generation_cancel_event = threading.Event()
        self._generation_progress_queue = Queue()
        self._generation_progress = QProgressDialog(title, "Cancel", 0, total_steps if total_steps is not None else 0, self)
        self._generation_progress.setWindowModality(Qt.WindowModal)
        self._generation_progress.setMinimumDuration(0)
        self._generation_progress.setAutoClose(True)
        if total_steps is None:
            self._generation_progress.setRange(0, 0)
        self._generation_progress.canceled.connect(self._generation_cancel_event.set)
        self._generation_progress.show()

        future = self._executor.submit(task_fn, self._generation_cancel_event, self._generation_progress_queue)
        self._generation_future = (future, on_success)
        QTimer.singleShot(50, self._poll_generation_future)

    def _poll_generation_future(self):
        if self._generation_future is None:
            return

        # Drain progress updates from the worker thread
        if self._generation_progress_queue is not None and self._generation_progress is not None:
            while not self._generation_progress_queue.empty():
                try:
                    value = self._generation_progress_queue.get_nowait()
                except Exception:
                    break
                try:
                    self._generation_progress.setValue(int(value))
                except Exception:
                    pass

        future, on_success = self._generation_future
        if future.done():
            self._finish_generation_future(on_success)
        else:
            QTimer.singleShot(100, self._poll_generation_future)

    def _finish_generation_future(self, on_success):
        future, _ = self._generation_future
        self._generation_future = None
        cancel_event = self._generation_cancel_event
        # Snapshot cancel state before closing the dialog. Some Qt styles can
        # emit `canceled` when the dialog is closed programmatically, which
        # would otherwise cause a successful run to be reported as canceled.
        was_canceled = bool(cancel_event and cancel_event.is_set())

        if self._generation_progress is not None:
            self._generation_progress.close()
            self._generation_progress = None
        self._generation_progress_queue = None
        self._generation_cancel_event = None

        try:
            result = future.result()
        except Exception as exc:
            # Log full traceback to the main window's output console so that
            # hard-to-reproduce worker errors (e.g., shape issues inside core
            # Tree/Forest routines) can be diagnosed with a complete stack trace.
            tb = traceback.format_exc()
            if self.parent_gui and hasattr(self.parent_gui, "log_output"):
                try:
                    self.parent_gui.log_output(tb)
                except Exception:
                    pass
            self._record_telemetry(exc, action="generation_worker", traceback_str=tb)
            QMessageBox.critical(self, "Generation Error", f"Failed to complete operation:\n\n{exc}")
            if self.parent_gui:
                self.parent_gui.update_status("Generation failed")
            return

        if result is None or was_canceled:
            if self.parent_gui:
                self.parent_gui.update_status("Generation canceled")
            return

        try:
            on_success(result)
        except Exception as exc:
            tb = traceback.format_exc()
            self._record_telemetry(exc, action="generation_finalize", traceback_str=tb)
            QMessageBox.critical(self, "Generation Error", f"Failed to finalize results:\n\n{exc}")
            if self.parent_gui:
                self.parent_gui.update_status("Generation failed")

    def _generate_tree_task(self, n_vessels, physical_clearance, convexity_tolerance,
                            preallocation_step, random_seed, config, cancel_event, progress_queue):
        """Generate a single tree in a worker thread."""
        from svv.tree.tree import Tree

        tree_params = self._build_tree_parameters(self.tree_base_params)
        tree = Tree(parameters=tree_params, preallocation_step=preallocation_step)
        tree.physical_clearance = physical_clearance
        if random_seed is not None:
            tree.random_seed = random_seed

        tree.set_domain(self.parent_gui.domain, convexity_tolerance)

        def _point_array(pt):
            if pt is None:
                return None
            arr = np.asarray(pt, dtype=float)
            d = self.parent_gui.domain.points.shape[1]
            # Accept either (d,) or (1,d) and normalize to a single
            # row of shape (1, d) so Domain.within/evaluate_fast always
            # see a 2D point array, avoiding scalar dists from cKDTree.
            if arr.ndim == 1 and arr.size == d:
                return arr.reshape(1, d)
            if arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] == d:
                return arr
            raise ValueError(f"Invalid start point shape {arr.shape}; expected a single {d}-component coordinate.")

        def _vec_array(vec):
            if vec is None:
                return None
            arr = np.asarray(vec, dtype=float)
            d = self.parent_gui.domain.points.shape[1]
            if arr.ndim == 1 and arr.size == d:
                return arr
            if arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] == d:
                return arr.reshape(-1)
            raise ValueError(f"Invalid direction shape {arr.shape}; expected {d}-component vector.")

        start_point = _point_array(config['start_points'][0][0])
        direction = _vec_array(config['directions'][0][0]) if config['directions'][0][0] is not None else None

        if start_point is not None:
            if direction is not None:
                tree.set_root(start_point, direction)
            else:
                tree.set_root(start_point)
        else:
            tree.set_root()

        update_every = max(1, n_vessels // 100)
        for i in range(n_vessels):
            if cancel_event.is_set():
                break
            tree.add()
            if (i + 1) % update_every == 0 and progress_queue is not None:
                progress_queue.put(i + 1)

        if progress_queue is not None:
            progress_queue.put(min(tree.n_terminals, n_vessels))

        return {
            'tree': tree,
            'physical_clearance': physical_clearance,
            'convexity_tolerance': convexity_tolerance,
            'canceled': cancel_event.is_set(),
        }

    def _generate_forest_task(self, n_vessels, physical_clearance, convexity_tolerance,
                        preallocation_step, random_seed, compete, decay_prob, config, cancel_event, progress_queue):
        """Generate a forest in a worker thread."""
        from svv.forest.forest import Forest

        n_networks = config['n_networks']
        n_trees_per_network = config['n_trees_per_network']
        start_points = config['start_points']
        directions = config['directions']

        def _point_array(pt):
            if pt is None:
                return None
            arr = np.asarray(pt, dtype=float).flatten()
            d = self.parent_gui.domain.points.shape[1]
            # Normalize forest start points to a single (1, d) row so
            # downstream Domain.within/evaluate_fast calls always see a
            # 2D array of points and never a 1D vector.
            # Handle case where we might have gotten multiple points - take first d values
            if arr.size >= d:
                arr = arr[:d]
            if arr.size == d:
                return arr.reshape(1, d)
            raise ValueError(f"Invalid start point: got {arr.size} values, expected {d}-component coordinate.")

        def _vec_array(vec):
            if vec is None:
                return None
            arr = np.asarray(vec, dtype=float).flatten()
            d = self.parent_gui.domain.points.shape[1]
            # Handle case where we might have gotten multiple directions - take first d values
            if arr.size >= d:
                arr = arr[:d]
            if arr.size == d:
                return arr
            raise ValueError(f"Invalid direction: got {arr.size} values, expected {d}-component vector.")

        norm_points = []
        norm_dirs = []
        for nets, dirs in zip(start_points, directions):
            p_net = []
            d_net = []
            for p, d in zip(nets, dirs):
                p_net.append(_point_array(p))
                d_net.append(_vec_array(d))
            norm_points.append(p_net)
            norm_dirs.append(d_net)
        start_points = norm_points
        directions = norm_dirs

        forest = Forest(
            domain=self.parent_gui.domain,
            n_networks=n_networks,
            n_trees_per_network=n_trees_per_network,
            start_points=start_points,
            directions=directions,
            physical_clearance=physical_clearance,
            compete=compete,
            preallocation_step=preallocation_step,
        )

        for net_idx, network in enumerate(forest.networks):
            for tree_idx, tree in enumerate(network):
                params = self._build_tree_parameters(self._params_for_tree(net_idx, tree_idx))
                tree.parameters = params
                if random_seed is not None:
                    tree.random_seed = random_seed

        forest.set_domain(self.parent_gui.domain, convexity_tolerance)
        forest.set_roots(start_points, directions)

        if isinstance(n_vessels, (list, tuple, np.ndarray)):
            per_network = [int(v) for v in np.asarray(n_vessels, dtype=int).reshape(-1).tolist()]
        else:
            per_network = [int(n_vessels)] * int(n_networks)
        if len(per_network) < int(n_networks):
            per_network.extend([per_network[-1] if per_network else 1] * (int(n_networks) - len(per_network)))
        per_network = per_network[:int(n_networks)]
        per_network = [max(1, int(v)) for v in per_network]

        total_steps = max(per_network) if per_network else 0
        update_every = max(1, total_steps // 100) if total_steps > 0 else 1
        for i in range(total_steps):
            if cancel_event.is_set():
                break
            active_networks = [idx for idx in range(int(n_networks)) if i < per_network[idx]]
            if not active_networks:
                break
            forest.add(1, network_id=active_networks, decay_probability=decay_prob)
            if (i + 1) % update_every == 0 and progress_queue is not None:
                progress_queue.put(i + 1)

        total_vessels = sum(tree.n_terminals for network in forest.networks for tree in network)
        if progress_queue is not None:
            progress_queue.put(total_steps)

        return {
            'forest': forest,
            'n_networks': n_networks,
            'n_vessels_per_network': per_network,
            'physical_clearance': physical_clearance,
            'convexity_tolerance': convexity_tolerance,
            'compete': compete,
            'decay_prob': decay_prob,
            'total_vessels': total_vessels,
            'canceled': cancel_event.is_set(),
        }

    def _on_tree_generated(self, result):
        tree = result.get('tree')
        if tree is None:
            if self.parent_gui:
                self.parent_gui.update_status("Tree generation canceled")
            return

        # Visualize and store
        self.parent_gui.vtk_widget.clear_trees()
        self.parent_gui.vtk_widget.clear_connections()
        # Group ID ('single', 0) used for visibility toggles
        self.parent_gui.vtk_widget.add_tree(tree, label="tree_single", group_id=("single", 0))
        self.parent_gui.trees = [tree]
        self.parent_gui.forest = None

        # Update Model Tree view
        if self.parent_gui and hasattr(self.parent_gui, "object_browser"):
            try:
                self.parent_gui.object_browser.set_single_tree(tree)
            except Exception:
                pass

        if self.parent_gui:
            self.parent_gui.update_status(f"Tree generated successfully with {tree.n_terminals} vessels")
            self.parent_gui.log_output(f"[Tree] done: {tree.n_terminals} vessels")

        QMessageBox.information(
            self,
            "Success",
            f"Tree generated successfully!\n\n"
            f"Total vessels: {tree.n_terminals}\n"
            f"Physical clearance: {result.get('physical_clearance')}\n"
            f"Convexity tolerance: {result.get('convexity_tolerance')}"
        )

    def _on_forest_generated(self, result):
        forest = result.get('forest')
        if forest is None:
            if self.parent_gui:
                self.parent_gui.update_status("Forest generation canceled")
            return

        self.parent_gui.vtk_widget.draw_forest(forest)
        self.parent_gui.forest = forest
        self.parent_gui.trees = []
        self._update_connect_button_visibility()

        # Update Model Tree with networks/trees layout
        if self.parent_gui and hasattr(self.parent_gui, "object_browser"):
            try:
                self.parent_gui.object_browser.set_forest(forest)
            except Exception:
                pass

        total_vessels = result.get('total_vessels', 0)
        n_networks = result.get('n_networks', 0)
        compete = result.get('compete', False)
        physical_clearance = result.get('physical_clearance')

        if self.parent_gui:
            self.parent_gui.update_status(
                f"Forest generated successfully with {total_vessels} vessels across {n_networks} networks"
            )
            self.parent_gui.log_output(f"[Forest] done: {total_vessels} vessels across {n_networks} networks")
            if forest.connections is None:
                self.parent_gui.log_output("[Forest] connect pending - click Connect Forest to link trees")

        QMessageBox.information(
            self,
            "Success",
            f"Forest generated successfully!\n\n"
            f"Total vessels: {total_vessels}\n"
            f"Networks: {n_networks}\n"
            f"Competition: {'Enabled' if compete else 'Disabled'}\n"
            f"Physical clearance: {physical_clearance}"
        )

    def _export_config(self):
        """Export the current configuration."""
        if self.parent_gui:
            self.parent_gui.save_configuration()

    def get_parameters(self):
        """
        Get the current parameter configuration.

        Returns
        -------
        dict
            Dictionary of parameters
        """
        params = {
            'mode': 'tree' if self.mode_combo.currentIndex() == 0 else 'forest',
            'n_vessels': self.n_vessels_spin.value(),
            'physical_clearance': self.physical_clearance_spin.value(),
            'convexity_tolerance': self.convexity_tolerance_spin.value(),
            'preallocation_step': self.preallocation_spin.value(),
            'random_seed': self.random_seed_spin.value() if self.random_seed_spin.value() >= 0 else None,
            'compete': self.compete_cb.isChecked(),
            'decay_probability': self.decay_probability_spin.value(),
            'tree_parameters': dict(self.tree_base_params),
            'tree_parameter_overrides': {str(k): v for k, v in self.tree_param_overrides.items()}
        }
        if self.mode_combo.currentIndex() != 0:
            params['n_vessels_per_network'] = list(getattr(self, "_n_vessels_per_network", []))
        return params
