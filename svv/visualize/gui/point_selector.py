"""
Widget for selecting and managing start points and directions.
"""
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QListWidget, QListWidgetItem,
    QLabel, QLineEdit, QCheckBox,
    QDoubleSpinBox, QComboBox, QSizePolicy, QScrollArea, QGridLayout
)
from PySide6.QtCore import Signal, Qt, QSignalBlocker
from svv.visualize.gui.theme import CADTheme, CADIcons


class PointSelectorWidget(QWidget):
    """
    Widget for managing start points and directions for Tree/Forest generation.
    """

    # Signals
    points_changed = Signal()

    def __init__(self, parent=None):
        """
        Initialize the point selector widget.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget (should be VascularizeGUI)
        """
        super().__init__(parent)
        self.parent_gui = parent
        self.domain = None
        self.points = []  # List of dicts with 'point', 'direction', 'network', 'tree_index'
        self._filtered_indices = []  # map visible list rows -> self.points indices

        self._init_ui()
        # Default: 1 network. Use the initial tree combo count as the default
        # per-network tree count until the ParameterPanel sets explicit values.
        self._n_trees_per_network = [max(1, self.tree_combo.count())]
        self.set_network_count(1)

    @staticmethod
    def _configure_combo(combo: QComboBox, *, min_chars: int = 10, max_width: int = 180):
        """Keep combo boxes useful in narrow docked layouts."""
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(min_chars)
        combo.setMinimumWidth(min(120, max_width))
        combo.setMaximumWidth(max_width)

    # ---- Public helpers ----
    def set_network_count(self, count: int):
        """
        Set number of networks and update the network selection dropdown.

        The network count is owned by the general ParameterPanel; this widget
        only reflects the current count for selecting start points.
        """
        try:
            n = int(count)
        except Exception:
            return
        n = max(1, n)

        current = self.network_combo.currentIndex()
        if current < 0:
            current = 0
        new_index = min(current, n - 1)

        with QSignalBlocker(self.network_combo):
            self.network_combo.clear()
            for i in range(n):
                self.network_combo.addItem(f"Network {i}")
            self.network_combo.setCurrentIndex(new_index)

        # Keep per-network tree counts aligned with the new network count.
        trees = getattr(self, "_n_trees_per_network", None)
        if not isinstance(trees, list):
            trees = []
        default_trees = max(1, self.tree_combo.count())
        if len(trees) < n:
            trees = list(trees) + [default_trees] * (n - len(trees))
        else:
            trees = list(trees)[:n]
        self._n_trees_per_network = [max(1, int(v)) for v in trees]

        # Refresh tree dropdown and point list for the current network.
        self._on_network_selection_changed(new_index)

    def set_tree_count(self, count: int):
        """
        Set a uniform tree count for all networks (legacy helper).

        Parameters
        ----------
        count : int
            Desired number of trees (minimum 1).
        """
        n = max(1, self.network_combo.count())
        self.set_trees_per_network([max(1, count)] * n)

    def set_trees_per_network(self, trees_per_network):
        """
        Set per-network tree counts and update the tree combo for the
        currently selected network.

        Parameters
        ----------
        trees_per_network : list of int
            Number of trees for each network.
        """
        self._n_trees_per_network = [max(1, c) for c in trees_per_network]
        current_net = self.network_combo.currentIndex()
        count = (
            self._n_trees_per_network[current_net]
            if 0 <= current_net < len(self._n_trees_per_network)
            else 1
        )
        with QSignalBlocker(self.tree_combo):
            self.tree_combo.clear()
            for i in range(count):
                self.tree_combo.addItem(f"Tree {i}")

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(8, 8, 20, 8)  # Extra right margin for scrollbar
        scroll_layout.setSpacing(8)

        # Group box for point management
        group_box = QGroupBox("Start Points & Directions")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(10)
        group_box.setLayout(group_layout)


        # Current network/tree selection
        selection_layout = QGridLayout()
        selection_layout.setContentsMargins(0, 0, 0, 0)
        selection_layout.setHorizontalSpacing(6)
        selection_layout.setVerticalSpacing(4)
        selection_layout.addWidget(QLabel("Network:"), 0, 0)
        self.network_combo = QComboBox()
        self._configure_combo(self.network_combo)
        self.network_combo.setMaxVisibleItems(100)
        self.network_combo.addItem("Network 0")
        self.network_combo.currentIndexChanged.connect(self._on_network_selection_changed)
        selection_layout.addWidget(self.network_combo, 0, 1)

        selection_layout.addWidget(QLabel("Tree:"), 1, 0)
        self.tree_combo = QComboBox()
        self._configure_combo(self.tree_combo)
        self.tree_combo.setMaxVisibleItems(100)
        self.tree_combo.addItem("Tree 0")
        self.tree_combo.addItem("Tree 1")
        self.tree_combo.currentIndexChanged.connect(self._on_tree_changed)
        selection_layout.addWidget(self.tree_combo, 1, 1)
        selection_layout.setColumnStretch(1, 1)
        group_layout.addLayout(selection_layout)

        # Point list
        self.point_list = QListWidget()
        self.point_list.itemClicked.connect(self._on_point_selected)
        self.point_list.itemChanged.connect(self._on_point_checked)
        group_layout.addWidget(QLabel("Points:"))
        group_layout.addWidget(self.point_list)

        # Add point buttons
        button_layout = QVBoxLayout()
        button_layout.setSpacing(6)

        self.pick_mode_btn = QPushButton("Pick Point (Click on Domain)")
        self.pick_mode_btn.setCheckable(True)
        self.pick_mode_btn.clicked.connect(self._toggle_pick_mode)
        self.pick_mode_btn.setToolTip("Click on the 3D domain to select a start point")
        button_layout.addWidget(self.pick_mode_btn)

        self.manual_btn = QPushButton("Manual Input...")
        self.manual_btn.clicked.connect(self._manual_input)
        self.manual_btn.setObjectName("secondaryButton")
        self.manual_btn.setToolTip("Enter point coordinates manually")
        button_layout.addWidget(self.manual_btn)

        group_layout.addLayout(button_layout)

        # Point details
        details_group = QGroupBox("Point Details")
        details_layout = QVBoxLayout()
        details_group.setLayout(details_layout)

        # Coordinates
        coord_layout = QHBoxLayout()
        coord_layout.addWidget(QLabel("Position:"))
        self.coord_label = QLabel("(0.0, 0.0, 0.0)")
        self.coord_label.setWordWrap(True)
        coord_layout.addWidget(self.coord_label)
        coord_layout.addStretch()
        details_layout.addLayout(coord_layout)

        # Direction checkbox
        self.use_direction_cb = QCheckBox("Use Custom Direction")
        self.use_direction_cb.stateChanged.connect(self._on_direction_toggled)
        self.use_direction_cb.setToolTip("Specify a custom growth direction for this start point")
        details_layout.addWidget(self.use_direction_cb)

        # Direction inputs
        direction_layout = QGridLayout()
        direction_layout.setContentsMargins(0, 0, 0, 0)
        direction_layout.setHorizontalSpacing(6)
        direction_layout.setVerticalSpacing(4)
        self.dir_x = QDoubleSpinBox()
        self.dir_x.setRange(-1.0, 1.0)
        self.dir_x.setSingleStep(0.1)
        self.dir_x.setDecimals(3)
        self.dir_x.valueChanged.connect(self._on_direction_changed)
        direction_layout.addWidget(QLabel("X:"), 0, 0)
        direction_layout.addWidget(self.dir_x, 0, 1)

        self.dir_y = QDoubleSpinBox()
        self.dir_y.setRange(-1.0, 1.0)
        self.dir_y.setSingleStep(0.1)
        self.dir_y.setDecimals(3)
        self.dir_y.valueChanged.connect(self._on_direction_changed)
        direction_layout.addWidget(QLabel("Y:"), 1, 0)
        direction_layout.addWidget(self.dir_y, 1, 1)

        self.dir_z = QDoubleSpinBox()
        self.dir_z.setRange(-1.0, 1.0)
        self.dir_z.setSingleStep(0.1)
        self.dir_z.setDecimals(3)
        self.dir_z.valueChanged.connect(self._on_direction_changed)
        direction_layout.addWidget(QLabel("Z:"), 2, 0)
        direction_layout.addWidget(self.dir_z, 2, 1)
        direction_layout.setColumnStretch(1, 1)

        details_layout.addLayout(direction_layout)

        # Normalize button
        self.normalize_btn = QPushButton("Normalize Direction")
        self.normalize_btn.clicked.connect(self._normalize_direction)
        details_layout.addWidget(self.normalize_btn)

        # Initially disable direction inputs
        self._enable_direction_inputs(False)

        group_layout.addWidget(details_group)

        # Delete and Clear buttons
        action_layout = QVBoxLayout()
        action_layout.setSpacing(6)
        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self._delete_selected)
        self.delete_btn.setObjectName("dangerButton")
        self.delete_btn.setProperty("danger", True)
        self.delete_btn.setToolTip("Delete the currently selected point")
        action_layout.addWidget(self.delete_btn)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self._clear_all)
        self.clear_btn.setObjectName("dangerButton")
        self.clear_btn.setProperty("danger", True)
        self.clear_btn.setToolTip("Remove all start points")
        action_layout.addWidget(self.clear_btn)

        group_layout.addLayout(action_layout)

        scroll_layout.addWidget(group_box)
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

    def set_domain(self, domain):
        """
        Set the domain for point selection.

        Parameters
        ----------
        domain : svv.domain.Domain
            Domain object
        """
        self.domain = domain

    def _toggle_pick_mode(self, checked):
        """Toggle point picking mode."""
        if checked and self.parent_gui:
            self.pick_mode_btn.setText("Picking... (Click Domain)")
            self.pick_mode_btn.setStyleSheet(f"background-color: {CADTheme.get_color('action', 'success')};")
            # Connect to VTK widget's point picking signal
            self.parent_gui.vtk_widget.point_picked.connect(self._on_point_picked)
        else:
            self.pick_mode_btn.setText("Pick Point (Click on Domain)")
            self.pick_mode_btn.setStyleSheet("")
            if self.parent_gui:
                try:
                    self.parent_gui.vtk_widget.point_picked.disconnect(self._on_point_picked)
                except:
                    pass

    def _on_point_picked(self, point):
        """
        Handle picked point from VTK widget.

        Parameters
        ----------
        point : np.ndarray
            Picked point coordinates
        """
        # Ensure point is a 1D array with exactly 3 coordinates
        point = np.asarray(point).flatten()
        if point.size != 3:
            # If we got something unexpected, try to extract first 3 values
            if point.size >= 3:
                point = point[:3]
            else:
                # Invalid point, ignore
                return

        network = self.network_combo.currentIndex()
        tree_idx = self.tree_combo.currentIndex()

        # Ensure only one selected per network/tree slot
        self._clear_selected_for_tree(network, tree_idx)

        point_data = {
            'point': point,
            'direction': None,
            'network': network,
            'tree_index': tree_idx,
            'selected': True
        }

        self.points.append(point_data)
        self._update_point_list()
        self._visualize_point(point_data, len(self.points) - 1)

        # Update status
        if self.parent_gui:
            self.parent_gui.update_status(
                f"Added point at ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})"
            )

        # Disable pick mode after picking
        self.pick_mode_btn.setChecked(False)
        self._toggle_pick_mode(False)

        self.points_changed.emit()

    def _manual_input(self):
        """Open dialog for manual point input."""
        from PySide6.QtWidgets import QDialog, QFormLayout, QDialogButtonBox

        dialog = QDialog(self)
        dialog.setWindowTitle("Manual Point Input")
        layout = QFormLayout()

        x_input = QDoubleSpinBox()
        x_input.setRange(-1000, 1000)
        x_input.setDecimals(3)
        layout.addRow("X:", x_input)

        y_input = QDoubleSpinBox()
        y_input.setRange(-1000, 1000)
        y_input.setDecimals(3)
        layout.addRow("Y:", y_input)

        z_input = QDoubleSpinBox()
        z_input.setRange(-1000, 1000)
        z_input.setDecimals(3)
        layout.addRow("Z:", z_input)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        dialog.setLayout(layout)

        if dialog.exec_():
            point = np.array([x_input.value(), y_input.value(), z_input.value()])
            self._on_point_picked(point)

    def _on_point_selected(self, item):
        """Handle point selection from list."""
        idx = self.point_list.row(item)
        if idx < len(self._filtered_indices):
            point_data = self.points[self._filtered_indices[idx]]
            point = point_data['point']
            direction = point_data['direction']

            self.coord_label.setText(f"({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})")

            if direction is not None:
                self.use_direction_cb.setChecked(True)
                self.dir_x.setValue(direction[0])
                self.dir_y.setValue(direction[1])
                self.dir_z.setValue(direction[2])
            else:
                self.use_direction_cb.setChecked(False)

    def _on_direction_toggled(self, state):
        """Handle direction checkbox toggle."""
        enabled = state == 2  # Qt.Checked
        self._enable_direction_inputs(enabled)

        if enabled:
            # Set default direction if needed
            if self.dir_x.value() == 0 and self.dir_y.value() == 0 and self.dir_z.value() == 0:
                self.dir_z.setValue(1.0)

        self._update_selected_point_direction()

    def _on_direction_changed(self):
        """Handle direction value changes."""
        self._update_selected_point_direction()

    def _normalize_direction(self):
        """Normalize the current direction vector."""
        direction = np.array([self.dir_x.value(), self.dir_y.value(), self.dir_z.value()])
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
            self.dir_x.setValue(direction[0])
            self.dir_y.setValue(direction[1])
            self.dir_z.setValue(direction[2])

    def _clear_selected_for_tree(self, network, tree_idx):
        """Clear selection flag for a given network/tree to enforce single selection."""
        for pd in self.points:
            if pd['network'] == network and pd['tree_index'] == tree_idx:
                pd['selected'] = False

    def _update_selected_point_direction(self):
        """Update the direction of the currently selected point."""
        current_item = self.point_list.currentItem()
        if current_item is not None:
            idx = self.point_list.row(current_item)
            if idx < len(self._filtered_indices):
                if self.use_direction_cb.isChecked():
                    direction = np.array([
                        self.dir_x.value(),
                        self.dir_y.value(),
                        self.dir_z.value()
                    ])
                    self.points[self._filtered_indices[idx]]['direction'] = direction
                    self._update_direction_visualization(self._filtered_indices[idx])
                else:
                    self.points[self._filtered_indices[idx]]['direction'] = None
                    self._clear_direction_visualization(self._filtered_indices[idx])

                self.points_changed.emit()

    def _enable_direction_inputs(self, enabled):
        """Enable or disable direction input fields."""
        self.dir_x.setEnabled(enabled)
        self.dir_y.setEnabled(enabled)
        self.dir_z.setEnabled(enabled)
        self.normalize_btn.setEnabled(enabled)

    def _delete_selected(self):
        """Delete the selected point."""
        current_item = self.point_list.currentItem()
        if current_item is not None:
            idx = self.point_list.row(current_item)
            if idx < len(self._filtered_indices):
                self.points.pop(self._filtered_indices[idx])
                self._update_point_list()
                self._refresh_visualization()
                self.points_changed.emit()

    def _clear_all(self):
        """Clear all points."""
        self.points.clear()
        self._update_point_list()
        if self.parent_gui:
            self.parent_gui.vtk_widget.clear()
        self.points_changed.emit()

    def _on_network_changed(self, value):
        """Handle network count change."""
        # Update network combo box
        self.network_combo.clear()
        for i in range(value):
            self.network_combo.addItem(f"Network {i}")
        self._update_point_list()

    def _on_network_selection_changed(self, value):
        """Handle network selection change in the dropdown."""
        # Update tree combo to reflect the selected network's tree count
        if hasattr(self, '_n_trees_per_network') and 0 <= value < len(self._n_trees_per_network):
            count = self._n_trees_per_network[value]
            with QSignalBlocker(self.tree_combo):
                self.tree_combo.clear()
                for i in range(count):
                    self.tree_combo.addItem(f"Tree {i}")
        # Refresh list to reflect points for the selected network
        self._update_point_list()

    def _on_tree_changed(self, value):
        """Handle tree selection change."""
        # Refresh list to reflect points for the selected tree
        self._update_point_list()

    def _update_point_list(self):
        """Update the point list widget."""
        # Enforce invariant: at most one active starting point per (network, tree)
        active_indices = {}
        for i, pd in enumerate(self.points):
            key = (pd.get('network', 0), pd.get('tree_index', 0))
            if pd.get('selected', False):
                if key in active_indices:
                    pd['selected'] = False
                else:
                    active_indices[key] = i

        self.point_list.blockSignals(True)
        self.point_list.clear()
        self._filtered_indices = []

        current_network = self.network_combo.currentIndex()
        current_tree = self.tree_combo.currentIndex()

        for i, point_data in enumerate(self.points):
            if point_data['network'] != current_network or point_data['tree_index'] != current_tree:
                continue

            point = point_data['point']
            has_dir = point_data['direction'] is not None

            label = f"P{i}: N{current_network}T{current_tree} ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"
            if has_dir:
                label += " [Dir]"

            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            item.setCheckState(Qt.Checked if point_data.get('selected', False) else Qt.Unchecked)
            self.point_list.addItem(item)
            self._filtered_indices.append(i)
        self.point_list.blockSignals(False)

    def _on_point_checked(self, item):
        """Handle checkbox toggles to mark selected points per tree."""
        idx = self.point_list.row(item)
        if idx < 0 or idx >= len(self._filtered_indices):
            return
        checked = item.checkState() == Qt.Checked
        point_data = self.points[self._filtered_indices[idx]]
        network = point_data['network']
        tree_idx = point_data['tree_index']

        # Only one selected per network/tree; uncheck others in same slot
        if checked:
            self.point_list.blockSignals(True)
            for i, pd in enumerate(self.points):
                if i == self._filtered_indices[idx]:
                    continue
                if pd['network'] == network and pd['tree_index'] == tree_idx and pd.get('selected', False):
                    pd['selected'] = False
                    # Update visible list item if it belongs to the current filter
                    if i in self._filtered_indices:
                        visible_row = self._filtered_indices.index(i)
                        it = self.point_list.item(visible_row)
                        if it:
                            it.setCheckState(Qt.Unchecked)
            self.point_list.blockSignals(False)

        point_data['selected'] = checked
        self.points_changed.emit()

    def _visualize_point(self, point_data, index):
        """Visualize a single point."""
        if self.parent_gui:
            vtk_widget = self.parent_gui.vtk_widget
            point = point_data['point']
            direction = point_data['direction']

            vtk_widget.add_start_point(point, index=index)

            if direction is not None:
                vtk_widget.add_direction(point, direction)

    def _update_direction_visualization(self, index):
        """Update direction visualization for a point."""
        if self.parent_gui and index < len(self.points):
            # For simplicity, refresh all visualizations
            self._refresh_visualization()

    def _clear_direction_visualization(self, index):
        """Clear direction visualization for a point."""
        if self.parent_gui:
            # For simplicity, refresh all visualizations
            self._refresh_visualization()

    def _refresh_visualization(self):
        """Refresh all point and direction visualizations."""
        if self.parent_gui:
            vtk_widget = self.parent_gui.vtk_widget
            with vtk_widget.batch_render():
                vtk_widget.clear()
                for i, point_data in enumerate(self.points):
                    self._visualize_point(point_data, i)

    def get_configuration(self):
        """
        Get the current configuration as a dictionary.

        Returns
        -------
        dict
            Configuration with start_points and directions for Forest
        """
        n_networks = max(1, self.network_combo.count())

        # Use per-network tree counts when available, else fall back to
        # the current tree combo count applied uniformly.
        if hasattr(self, '_n_trees_per_network') and len(self._n_trees_per_network) == n_networks:
            n_trees_per_network = list(self._n_trees_per_network)
        else:
            n_trees = self.tree_combo.count()
            n_trees_per_network = [n_trees] * n_networks

        # Organize points by network and tree
        config = {
            'n_networks': n_networks,
            'n_trees_per_network': n_trees_per_network,
            'start_points': [[None for _ in range(n_trees_per_network[i])] for i in range(n_networks)],
            'directions': [[None for _ in range(n_trees_per_network[i])] for i in range(n_networks)]
        }

        for point_data in self.points:
            if not point_data.get('selected', True):
                continue
            network = point_data['network']
            tree_idx = point_data['tree_index']
            point = point_data['point']
            direction = point_data['direction']

            if network < n_networks and tree_idx < n_trees_per_network[network]:
                config['start_points'][network][tree_idx] = point.tolist()
                if direction is not None:
                    config['directions'][network][tree_idx] = direction.tolist()

        return config
