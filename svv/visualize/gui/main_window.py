"""
Unified CAD-style main GUI window for svVascularize.
"""
from __future__ import annotations

from pathlib import Path
import os
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDockWidget, QStatusBar,
    QFileDialog, QMessageBox, QTreeWidget, QTreeWidgetItem, QTabWidget, QToolBar,
    QPlainTextEdit, QProgressBar, QProgressDialog, QFrame, QDialog, QFormLayout,
    QDialogButtonBox, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit, QComboBox,
    QColorDialog, QMenu
)
from PySide6.QtCore import Qt, QSize, QSettings, QTimer, QUrl
from PySide6.QtGui import QAction, QKeySequence, QDesktopServices
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading
from typing import Optional
from svv.visualize.gui.vtk_widget import VTKWidget
from svv.visualize.gui.point_selector import PointSelectorWidget
from svv.visualize.gui.parameter_panel import ParameterPanel
from svv.visualize.gui.theme import CADTheme, CADIcons
import svv.tree.tree as _svv_tree_mod
import svv.forest.forest as _svv_forest_mod
from svv.telemetry import capture_exception, capture_message


class SystemMonitorWidget(QFrame):
    """
    Widget that displays running CPU and memory usage for the application.
    Updates periodically with a timer.
    """

    def __init__(self, parent=None, update_interval_ms=2000):
        """
        Initialize the system monitor widget.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        update_interval_ms : int
            Update interval in milliseconds (default: 2000ms)
        """
        super().__init__(parent)
        self._process = None
        self._psutil_available = False

        # Try to import psutil
        try:
            import psutil
            self._psutil = psutil
            self._process = psutil.Process(os.getpid())
            self._psutil_available = True
        except ImportError:
            self._psutil = None

        self._init_ui()

        # Setup update timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_stats)
        self._timer.start(update_interval_ms)

        # Initial update
        self._update_stats()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QHBoxLayout()
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(8)
        self.setLayout(layout)

        # Memory label with icon-like prefix
        self.mem_label = QLabel("MEM: --")
        self.mem_label.setToolTip("Application memory usage")
        self.mem_label.setStyleSheet(f"color: {CADTheme.get_color('text', 'secondary')}; font-size: 10px;")
        layout.addWidget(self.mem_label)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setStyleSheet(f"color: {CADTheme.get_color('border', 'divider')};")
        layout.addWidget(separator)

        # CPU label
        self.cpu_label = QLabel("CPU: --")
        self.cpu_label.setToolTip("Application CPU usage")
        self.cpu_label.setStyleSheet(f"color: {CADTheme.get_color('text', 'secondary')}; font-size: 10px;")
        layout.addWidget(self.cpu_label)

    def _update_stats(self):
        """Update CPU and memory statistics."""
        if not self._psutil_available or self._process is None:
            self.mem_label.setText("MEM: N/A")
            self.cpu_label.setText("CPU: N/A")
            return

        try:
            # Get memory info
            mem_info = self._process.memory_info()
            mem_mb = mem_info.rss / (1024 * 1024)  # Convert to MB

            # Format memory display
            if mem_mb >= 1024:
                mem_str = f"MEM: {mem_mb / 1024:.1f} GB"
            else:
                mem_str = f"MEM: {mem_mb:.0f} MB"

            # Color code based on memory usage
            if mem_mb > 4096:  # > 4GB - warning
                self.mem_label.setStyleSheet(f"color: {CADTheme.get_color('status', 'warning')}; font-size: 10px;")
            elif mem_mb > 8192:  # > 8GB - critical
                self.mem_label.setStyleSheet(f"color: {CADTheme.get_color('status', 'error')}; font-size: 10px;")
            else:
                self.mem_label.setStyleSheet(f"color: {CADTheme.get_color('text', 'secondary')}; font-size: 10px;")

            self.mem_label.setText(mem_str)

            # Get CPU info (percent since last call)
            cpu_percent = self._process.cpu_percent()
            cpu_str = f"CPU: {cpu_percent:.0f}%"

            # Color code based on CPU usage
            if cpu_percent > 80:
                self.cpu_label.setStyleSheet(f"color: {CADTheme.get_color('status', 'warning')}; font-size: 10px;")
            elif cpu_percent > 95:
                self.cpu_label.setStyleSheet(f"color: {CADTheme.get_color('status', 'error')}; font-size: 10px;")
            else:
                self.cpu_label.setStyleSheet(f"color: {CADTheme.get_color('text', 'secondary')}; font-size: 10px;")

            self.cpu_label.setText(cpu_str)

        except Exception:
            # Process may have ended or other error
            self.mem_label.setText("MEM: --")
            self.cpu_label.setText("CPU: --")

    def set_update_interval(self, interval_ms):
        """
        Set the update interval.

        Parameters
        ----------
        interval_ms : int
            Update interval in milliseconds
        """
        self._timer.setInterval(interval_ms)

    def stop(self):
        """Stop the update timer."""
        self._timer.stop()

    def start(self):
        """Start the update timer."""
        self._timer.start()


class ObjectBrowserWidget(QTreeWidget):
    """
    Object/model tree browser similar to FreeCAD's tree view.
    Shows domain, trees, forests, and their properties.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent

        self.setHeaderLabels(["Model Tree"])
        self.setAlternatingRowColors(True)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        # Root items
        self.domain_item = None
        self.trees_item = None
        self.forests_item = None
        self.points_item = None

        self._init_tree()

    def _init_tree(self):
        """Initialize the tree structure."""
        # Scene root
        self.scene_root = QTreeWidgetItem(self, ["Scene"])
        self.scene_root.setExpanded(True)

        # Domain
        self.domain_item = QTreeWidgetItem(self.scene_root, ["Domain"])
        self.domain_item.setIcon(0, CADIcons.get_icon('mesh'))
        self.domain_item.setCheckState(0, Qt.Checked)

        # Start Points
        self.points_item = QTreeWidgetItem(self.scene_root, ["Start Points"])
        self.points_item.setIcon(0, CADIcons.get_icon('point'))
        self.points_item.setExpanded(True)

        # Trees
        self.trees_item = QTreeWidgetItem(self.scene_root, ["Trees"])
        self.trees_item.setIcon(0, CADIcons.get_icon('tree'))
        self.trees_item.setExpanded(True)

        # Networks (was Forests)
        self.forests_item = QTreeWidgetItem(self.scene_root, ["Networks"])
        self.forests_item.setIcon(0, CADIcons.get_icon('forest'))
        self.forests_item.setExpanded(True)

        # Connect item changes
        self.itemChanged.connect(self._on_item_changed)

    def _on_item_changed(self, item, column):
        """Handle item visibility changes."""
        if not self.main_window:
            return
        vtk_widget = getattr(self.main_window, 'vtk_widget', None)
        if not vtk_widget:
            return

        checked = item.checkState(0) == Qt.Checked

        # Domain visibility toggle
        if item == self.domain_item:
            # Only attempt to change visibility when a domain actor exists and
            # a plotter is available. This avoids errors when no domain has
            # been loaded yet or when 3D visualization is disabled.
            actor = getattr(vtk_widget, 'domain_actor', None)
            plotter = getattr(vtk_widget, 'plotter', None)
            if actor is None or plotter is None:
                return
            actor.SetVisibility(bool(checked))
            plotter.render()
            return

        # Network / tree visibility toggles
        data = item.data(0, Qt.UserRole)
        if not data or not isinstance(data, tuple):
            return
        kind = data[0]

        if kind == "network":
            net_idx = data[1]
            # Propagate checkbox state to children without re-entering handler
            self.blockSignals(True)
            for i in range(item.childCount()):
                child = item.child(i)
                child.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
            self.blockSignals(False)
            vtk_widget.set_network_visibility(net_idx, checked)
        elif kind == "tree":
            mode, net_idx, tree_idx = data[1], data[2], data[3]
            vtk_widget.set_tree_visibility(mode, net_idx, tree_idx, checked)

    def _show_context_menu(self, pos):
        """Show a context menu for the clicked item."""
        item = self.itemAt(pos)
        if item is None:
            return
        data = item.data(0, Qt.UserRole)
        if not data or not isinstance(data, tuple):
            return
        kind = data[0]
        if kind not in ("tree", "network"):
            return

        menu = QMenu(self)
        color_action = menu.addAction("Change Color...")
        action = menu.exec_(self.viewport().mapToGlobal(pos))
        if action == color_action:
            self._change_item_color(item, data)

    @staticmethod
    def _stored_color_to_qcolor(stored):
        """Convert a stored color (name string or RGB tuple) to a QColor."""
        from PySide6.QtGui import QColor
        if isinstance(stored, tuple):
            return QColor.fromRgbF(*stored[:3])
        if isinstance(stored, str):
            c = QColor(stored)
            if c.isValid():
                return c
        return QColor(Qt.red)

    def _change_item_color(self, item, data):
        """Open a color dialog and apply the chosen color to the tree/network."""
        vtk_widget = getattr(self.main_window, 'vtk_widget', None)
        if not vtk_widget:
            return

        kind = data[0]
        initial_color = self._stored_color_to_qcolor(None)

        if kind == "tree":
            mode, net_idx, tree_idx = data[1], data[2], data[3]
            group_key = ("single", tree_idx) if mode == "single" else ("forest", net_idx, tree_idx)
            stored = vtk_widget.tree_group_colors.get(group_key)
            if stored is not None:
                initial_color = self._stored_color_to_qcolor(stored)
        elif kind == "network":
            net_idx = data[1]
            for key, col in vtk_widget.tree_group_colors.items():
                if isinstance(key, tuple) and len(key) >= 3 and key[0] == "forest" and key[1] == net_idx:
                    initial_color = self._stored_color_to_qcolor(col)
                    break

        chosen = QColorDialog.getColor(initial_color, self, "Choose Tree Color")
        if not chosen.isValid():
            return

        rgb = (chosen.redF(), chosen.greenF(), chosen.blueF())

        if kind == "tree":
            vtk_widget.set_tree_color(group_key, rgb)
        elif kind == "network":
            vtk_widget.set_network_color(net_idx, rgb)

    def add_point(self, point_id, point_data):
        """Add a point to the tree."""
        point_item = QTreeWidgetItem(self.points_item, [f"Point {point_id}"])
        point_item.setIcon(0, CADIcons.get_icon('point'))
        point_item.setCheckState(0, Qt.Checked)
        return point_item

    def add_tree(self, tree_id, n_vessels):
        """Add a tree to the tree view."""
        tree_item = QTreeWidgetItem(self.trees_item, [f"Tree {tree_id} ({n_vessels} vessels)"])
        tree_item.setIcon(0, CADIcons.get_icon('tree'))
        tree_item.setCheckState(0, Qt.Checked)
        return tree_item

    def add_forest(self, forest_id, n_networks):
        """Add a forest summary entry to the Networks view."""
        forest_item = QTreeWidgetItem(self.forests_item, [f"Networks ({n_networks})"])
        forest_item.setIcon(0, CADIcons.get_icon('forest'))
        forest_item.setCheckState(0, Qt.Checked)
        return forest_item

    def clear_trees(self):
        """Clear all trees from view."""
        self.trees_item.takeChildren()

    def clear_forests(self):
        """Clear all networks/forests from view."""
        self.forests_item.takeChildren()

    def clear_points(self):
        """Clear all points from view."""
        self.points_item.takeChildren()

    # ---- Helpers to reflect generated trees/forests ----
    def set_single_tree(self, tree):
        """Populate the Trees section for a single generated tree."""
        # Clear any previous tree/forest entries to avoid stale ghosts
        self.clear_trees()
        self.clear_forests()
        if tree is None:
            return
        item = QTreeWidgetItem(self.trees_item, [f"Tree 0 ({tree.n_terminals} vessels)"])
        item.setIcon(0, CADIcons.get_icon('tree'))
        item.setCheckState(0, Qt.Checked)
        # Mark this as a single-tree entry for visibility toggles
        item.setData(0, Qt.UserRole, ("tree", "single", 0, 0))
        self.trees_item.setExpanded(True)

    def set_forest(self, forest):
        """Populate the Networks section for a generated forest."""
        # Clear both trees and networks to avoid mixing single-tree and forest state
        self.clear_trees()
        self.clear_forests()
        if forest is None:
            return

        for net_idx, network in enumerate(forest.networks):
            net_item = QTreeWidgetItem(self.forests_item, [f"Network {net_idx}"])
            net_item.setIcon(0, CADIcons.get_icon('forest'))
            net_item.setCheckState(0, Qt.Checked)
            net_item.setData(0, Qt.UserRole, ("network", net_idx))
            net_item.setExpanded(True)

            for tree_idx, tree in enumerate(network):
                label = f"Tree {tree_idx} ({tree.n_terminals} vessels)"
                t_item = QTreeWidgetItem(net_item, [label])
                t_item.setIcon(0, CADIcons.get_icon('tree'))
                t_item.setCheckState(0, Qt.Checked)
                # Mark as forest tree for per-tree visibility
                t_item.setData(0, Qt.UserRole, ("tree", "forest", net_idx, tree_idx))

        self.forests_item.setExpanded(True)


class VascularizeGUI(QMainWindow):
    """
    Unified CAD-style main GUI window for visualizing and manipulating Domain objects.
    """

    def __init__(self, domain=None):
        """
        Initialize the GUI window.

        Parameters
        ----------
        domain : svv.domain.Domain, optional
            Initial domain object to visualize
        """
        super().__init__()
        self.domain = domain
        self.trees = []
        self.forest = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._load_future = None
        self._load_progress = None
        self._load_cancel_event = None
        self._load_progress_queue = None

        # Persistent layout
        self.settings = QSettings("SimVascular", "svVascularize")

        # Apply CAD theme
        self.setStyleSheet(CADTheme.get_stylesheet())

        self.setWindowTitle("svVascularize - Vascular Design")
        self.setGeometry(100, 100, 1600, 1000)

        # Create central 3D viewport first
        self._init_viewport()

        # Create toolbars
        self._create_toolbars()

        # Create dockable panels
        self._create_dock_widgets()

        # Create menu bar
        self._create_menu_bar()

        # Create status bar
        self._create_status_bar()

        # Restore window geometry and dock layout from settings
        self._restore_layout()

        # Load domain if provided
        if domain is not None:
            self.load_domain(domain)

    def _init_viewport(self):
        """Initialize the central 3D viewport."""
        self.vtk_widget = VTKWidget(self)
        self.setCentralWidget(self.vtk_widget)

    def _create_toolbars(self):
        """Create CAD-standard toolbars."""
        # File Toolbar
        self.file_toolbar = QToolBar("File")
        self.file_toolbar.setObjectName("FileToolbar")
        self.file_toolbar.setIconSize(QSize(24, 24))
        self.file_toolbar.setMovable(True)
        self.file_toolbar.setFloatable(False)
        self.addToolBar(Qt.TopToolBarArea, self.file_toolbar)

        # Open action
        self.action_open = QAction(CADIcons.get_icon('open'), "Open", self)
        self.action_open.setShortcut(QKeySequence.Open)
        self.action_open.setStatusTip("Load domain mesh file")
        self.action_open.setToolTip("Load Domain (Ctrl+O)")
        self.action_open.triggered.connect(self.load_domain_dialog)
        self.file_toolbar.addAction(self.action_open)

        # Save action (configuration)
        self.action_save = QAction(CADIcons.get_icon('save'), "Save", self)
        self.action_save.setShortcut(QKeySequence.Save)
        self.action_save.setStatusTip("Save configuration")
        self.action_save.setToolTip("Save Configuration (Ctrl+S)")
        self.action_save.triggered.connect(self.save_configuration)
        self.file_toolbar.addAction(self.action_save)

        # Save Domain action
        self.action_save_domain = QAction(CADIcons.get_icon('mesh'), "Save Domain...", self)
        self.action_save_domain.setShortcut("Ctrl+Shift+S")
        self.action_save_domain.setStatusTip("Save current domain to .dmn file (preserves create/solve results)")
        self.action_save_domain.setToolTip("Save Domain to .dmn (Ctrl+Shift+S)")
        self.action_save_domain.triggered.connect(self.save_domain_dialog)
        self.file_toolbar.addAction(self.action_save_domain)

        # Save Vascular Object action (Tree or Forest)
        self.action_save_vascular = QAction(CADIcons.get_icon('tree'), "Save Vascular Object...", self)
        self.action_save_vascular.setShortcut("Ctrl+Alt+S")
        self.action_save_vascular.setStatusTip("Save generated Tree (.tree) or Forest (.forest) to file")
        self.action_save_vascular.setToolTip("Save Tree/Forest (Ctrl+Alt+S)")
        self.action_save_vascular.triggered.connect(self.save_vascular_object_dialog)
        self.file_toolbar.addAction(self.action_save_vascular)

        # Export action – opens a menu with all available export options.
        self.action_export = QAction(CADIcons.get_icon('export'), "Export", self)
        self.action_export.setStatusTip("Export generated vasculature")
        self.action_export.setToolTip("Export Results")
        self.action_export.triggered.connect(self._show_export_menu)
        self.file_toolbar.addAction(self.action_export)

        self.file_toolbar.addSeparator()

        # View Toolbar
        self.view_toolbar = QToolBar("View")
        self.view_toolbar.setObjectName("ViewToolbar")
        self.view_toolbar.setIconSize(QSize(24, 24))
        self.view_toolbar.setMovable(True)
        self.view_toolbar.setFloatable(False)
        self.addToolBar(Qt.TopToolBarArea, self.view_toolbar)

        # Fit view
        self.action_fit = QAction(CADIcons.get_icon('fit'), "Fit", self)
        self.action_fit.setShortcut("V, F")
        self.action_fit.setStatusTip("Fit all objects in view")
        self.action_fit.setToolTip("Fit View (V, F)")
        self.action_fit.triggered.connect(self.vtk_widget.reset_camera)
        self.view_toolbar.addAction(self.action_fit)

        # Isometric view
        self.action_iso = QAction(CADIcons.get_icon('iso'), "Iso", self)
        self.action_iso.setShortcut("V, I")
        self.action_iso.setStatusTip("Isometric view")
        self.action_iso.setToolTip("Isometric View (V, I)")
        self.action_iso.triggered.connect(self.vtk_widget.view_iso)
        self.view_toolbar.addAction(self.action_iso)

        # Top view
        self.action_top = QAction(CADIcons.get_icon('top'), "Top", self)
        self.action_top.setShortcut("V, T")
        self.action_top.setStatusTip("Top view")
        self.action_top.setToolTip("Top View (V, T)")
        self.action_top.triggered.connect(self.vtk_widget.view_top)
        self.view_toolbar.addAction(self.action_top)

        # Front view
        self.action_front = QAction(CADIcons.get_icon('front'), "Front", self)
        self.action_front.setShortcut("V, 1")
        self.action_front.setStatusTip("Front view")
        self.action_front.setToolTip("Front View (V, 1)")
        self.action_front.triggered.connect(self.vtk_widget.view_front)
        self.view_toolbar.addAction(self.action_front)

        # Right view
        self.action_right = QAction(CADIcons.get_icon('right'), "Right", self)
        self.action_right.setShortcut("V, 3")
        self.action_right.setStatusTip("Right view")
        self.action_right.setToolTip("Right View (V, 3)")
        self.action_right.triggered.connect(self.vtk_widget.view_right)
        self.view_toolbar.addAction(self.action_right)

        self.view_toolbar.addSeparator()

        # Toggle domain visibility
        self.action_toggle_domain = QAction(CADIcons.get_icon('visible'), "Domain", self)
        self.action_toggle_domain.setCheckable(True)
        self.action_toggle_domain.setChecked(True)
        self.action_toggle_domain.setShortcut("D")
        self.action_toggle_domain.setStatusTip("Toggle domain visibility")
        self.action_toggle_domain.setToolTip("Toggle Domain (D)")
        self.action_toggle_domain.triggered.connect(self.vtk_widget.toggle_domain_visibility)
        self.view_toolbar.addAction(self.action_toggle_domain)

        # Toggle grid visibility
        self.action_toggle_grid = QAction(CADIcons.get_icon('grid'), "Grid", self)
        self.action_toggle_grid.setCheckable(True)
        self.action_toggle_grid.setChecked(False)
        self.action_toggle_grid.setShortcut("G")
        self.action_toggle_grid.setStatusTip("Toggle 3D grid visibility")
        self.action_toggle_grid.setToolTip("Toggle Grid (G)")
        self.action_toggle_grid.triggered.connect(self._toggle_grid)
        self.view_toolbar.addAction(self.action_toggle_grid)

        # Generation Toolbar
        self.gen_toolbar = QToolBar("Generation")
        self.gen_toolbar.setObjectName("GenerationToolbar")
        self.gen_toolbar.setIconSize(QSize(24, 24))
        self.gen_toolbar.setMovable(True)
        self.gen_toolbar.setFloatable(False)
        self.addToolBar(Qt.TopToolBarArea, self.gen_toolbar)

        # Add point tool
        self.action_add_point = QAction(CADIcons.get_icon('point'), "Start Pt", self)
        self.action_add_point.setCheckable(True)
        self.action_add_point.setShortcut("P")
        self.action_add_point.setStatusTip("Add start point")
        self.action_add_point.setToolTip("Add Start Point (P)")
        self.gen_toolbar.addAction(self.action_add_point)

        # Add direction tool
        self.action_add_vector = QAction(CADIcons.get_icon('vector'), "Direction", self)
        self.action_add_vector.setStatusTip("Add direction vector")
        self.action_add_vector.setToolTip("Add Direction Vector")
        self.gen_toolbar.addAction(self.action_add_vector)

        self.gen_toolbar.addSeparator()

        # Generate action
        self.action_generate = QAction(CADIcons.get_icon('generate'), "Generate", self)
        self.action_generate.setShortcut("G")
        self.action_generate.setStatusTip("Generate vascular tree/forest")
        self.action_generate.setToolTip("Generate (G)")
        self.gen_toolbar.addAction(self.action_generate)

    def _create_dock_widgets(self):
        """Create dockable panels (CAD-style)."""
        # Model Tree / Object Browser (Left)
        # Ensure docks have stable object names so Qt state restore never duplicates them
        self.tree_dock = QDockWidget("Model Tree", self)
        self.tree_dock.setObjectName("TreeDock")
        self.tree_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.object_browser = ObjectBrowserWidget(self)
        self.tree_dock.setWidget(self.object_browser)

        self.addDockWidget(Qt.LeftDockWidgetArea, self.tree_dock)

        # Properties / Parameters Panel (Right)
        self.properties_dock = QDockWidget("Properties", self)
        self.properties_dock.setObjectName("PropertiesDock")
        self.properties_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # Create tabbed interface for properties
        self.properties_tabs = QTabWidget()

        # Generation parameters tab
        self.parameter_panel = ParameterPanel(self)
        self.parameter_panel.length_unit_changed.connect(self._on_length_unit_changed)
        self.properties_tabs.addTab(self.parameter_panel, "Generation")

        # Point selector tab
        self.point_selector = PointSelectorWidget(self)
        self.properties_tabs.addTab(self.point_selector, "Start Points")

        # Ensure point selector's tree combo reflects the current generation
        # mode after both panels are constructed. This guarantees that in
        # single-tree mode only a single 'Tree 0' entry is shown.
        try:
            self.parameter_panel._on_mode_changed(self.parameter_panel.mode_combo.currentIndex())
        except Exception:
            pass

        self.properties_dock.setWidget(self.properties_tabs)
        self.addDockWidget(Qt.RightDockWidgetArea, self.properties_dock)

        # Info Panel (Bottom - initially hidden)
        self.info_dock = QDockWidget("Information", self)
        self.info_dock.setObjectName("InfoDock")
        self.info_dock.setAllowedAreas(Qt.BottomDockWidgetArea)

        # Tabs inside info dock: Info text + Output console
        self.info_tabs = QTabWidget()

        self.info_widget = QWidget()
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(8, 8, 8, 8)
        info_layout.setSpacing(6)
        self.info_widget.setLayout(info_layout)

        self.info_label = QLabel("Domain statistics and generation info will appear here.")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("padding: 8px;")
        info_layout.addWidget(self.info_label)

        self.info_progress = QProgressBar()
        self.info_progress.setVisible(False)
        self.info_progress.setTextVisible(True)
        self.info_progress.setRange(0, 1)
        self.info_progress.setValue(0)
        info_layout.addWidget(self.info_progress)

        self.info_tabs.addTab(self.info_widget, "Info")

        self.output_console = QPlainTextEdit()
        self.output_console.setReadOnly(True)
        self.output_console.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.output_console.setPlaceholderText("Session output will appear here.")
        self.info_tabs.addTab(self.output_console, "Output")

        self.info_dock.setWidget(self.info_tabs)

        self.addDockWidget(Qt.BottomDockWidgetArea, self.info_dock)
        self.info_dock.hide()  # Hidden by default

        # Solution inspector panel (dockable; created up front but hidden)
        from svv.visualize.gui.solution_inspector import SolutionInspectorWidget

        self.solution_dock = QDockWidget("Solution Inspector", self)
        self.solution_dock.setObjectName("SolutionInspectorDock")
        self.solution_dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea
        )
        # Ensure the dock is closable via the standard 'X' button
        self.solution_dock.setFeatures(
            QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        self.solution_inspector = SolutionInspectorWidget(self.solution_dock)
        self.solution_dock.setWidget(self.solution_inspector)
        self.addDockWidget(Qt.RightDockWidgetArea, self.solution_dock)
        self.solution_dock.hide()

    def _create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        file_menu.addAction(self.action_open)
        file_menu.addAction(self.action_save)
        file_menu.addAction(self.action_save_domain)
        file_menu.addAction(self.action_save_vascular)
        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        view_menu.addAction(self.action_fit)
        view_menu.addAction(self.action_iso)
        view_menu.addAction(self.action_top)
        view_menu.addAction(self.action_front)
        view_menu.addAction(self.action_right)
        view_menu.addSeparator()
        view_menu.addAction(self.action_toggle_domain)

        # Domain edge toggle (View menu only)
        self.action_toggle_domain_edges = QAction("Domain Edges", self)
        self.action_toggle_domain_edges.setCheckable(True)
        self.action_toggle_domain_edges.setChecked(True)
        self.action_toggle_domain_edges.setStatusTip("Toggle domain edge display")
        self.action_toggle_domain_edges.setToolTip("Toggle Domain Edges")
        self.action_toggle_domain_edges.triggered.connect(
            lambda checked: self.vtk_widget.set_domain_edges_visible(checked)
        )
        view_menu.addAction(self.action_toggle_domain_edges)

        # Grid toggle (also in toolbar)
        view_menu.addAction(self.action_toggle_grid)

        # Scale bar toggle
        self.action_toggle_scale_bar = QAction(CADIcons.get_icon('ruler'), "Scale Bar", self)
        self.action_toggle_scale_bar.setCheckable(True)
        self.action_toggle_scale_bar.setChecked(True)
        self.action_toggle_scale_bar.setShortcut("S")
        self.action_toggle_scale_bar.setStatusTip("Toggle scale bar visibility")
        self.action_toggle_scale_bar.setToolTip("Toggle Scale Bar (S)")
        self.action_toggle_scale_bar.triggered.connect(self._toggle_scale_bar)
        view_menu.addAction(self.action_toggle_scale_bar)
        view_menu.addSeparator()

        # Panel visibility submenu
        panels_menu = view_menu.addMenu("&Panels")
        panels_menu.addAction(self.tree_dock.toggleViewAction())
        panels_menu.addAction(self.properties_dock.toggleViewAction())
        panels_menu.addAction(self.info_dock.toggleViewAction())
        if hasattr(self, "solution_dock") and self.solution_dock is not None:
            panels_menu.addAction(self.solution_dock.toggleViewAction())

        view_menu.addSeparator()

        # Toolbar visibility submenu
        toolbars_menu = view_menu.addMenu("&Toolbars")
        toolbars_menu.addAction(self.file_toolbar.toggleViewAction())
        toolbars_menu.addAction(self.view_toolbar.toggleViewAction())
        toolbars_menu.addAction(self.gen_toolbar.toggleViewAction())

        view_menu.addSeparator()

        # Layout management
        save_layout_action = QAction("Save Layout", self)
        save_layout_action.setStatusTip("Save current window layout")
        save_layout_action.triggered.connect(self._save_layout)
        view_menu.addAction(save_layout_action)

        reset_layout_action = QAction("Reset Layout", self)
        reset_layout_action.setStatusTip("Reset window layout to defaults")
        reset_layout_action.triggered.connect(self._reset_layout)
        view_menu.addAction(reset_layout_action)

        # Generation menu
        gen_menu = menubar.addMenu("&Generate")
        gen_menu.addAction(self.action_add_point)
        gen_menu.addAction(self.action_add_vector)
        gen_menu.addSeparator()
        gen_menu.addAction(self.action_generate)

        # Fabricate menu - export CAD/centerline data for Tree/Forest
        fabricate_menu = menubar.addMenu("&Fabricate")

        export_centerlines_action = QAction("Export Centerlines...", self)
        export_centerlines_action.setStatusTip("Export centerline spline data for the current tree or forest")
        export_centerlines_action.triggered.connect(self.export_centerlines_dialog)
        fabricate_menu.addAction(export_centerlines_action)

        export_solid_action = QAction("Export Solids...", self)
        export_solid_action.setStatusTip("Export solid vessel geometry (STL/VTU) for the current tree or forest")
        export_solid_action.triggered.connect(self.export_solids_dialog)
        fabricate_menu.addAction(export_solid_action)

        export_splines_action = QAction("Export Splines...", self)
        export_splines_action.setStatusTip("Export B-spline centerline samples for the current tree or forest")
        export_splines_action.triggered.connect(self.export_splines_dialog)
        fabricate_menu.addAction(export_splines_action)

        # Simulate menu - export simulation setup files
        simulate_menu = menubar.addMenu("&Simulate")

        export_0d_action = QAction("Export 0D Simulation...", self)
        export_0d_action.setStatusTip("Export 0D lumped parameter simulation files for the current tree or forest")
        export_0d_action.triggered.connect(self.export_0d_simulation_dialog)
        simulate_menu.addAction(export_0d_action)

        export_3d_action = QAction("Export 3D Simulation...", self)
        export_3d_action.setStatusTip("Build meshes and export 3D simulation setup for the current tree or forest")
        export_3d_action.triggered.connect(self.export_3d_simulation_dialog)
        simulate_menu.addAction(export_3d_action)

        inspect_solution_action = QAction("Inspect Solutions...", self)
        inspect_solution_action.setStatusTip("Inspect time-varying simulation solutions in a dedicated viewport")
        inspect_solution_action.triggered.connect(self._show_solution_inspector)
        simulate_menu.addAction(inspect_solution_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction(CADIcons.get_icon('info'), "About", self)
        about_action.setStatusTip("About svVascularize")
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        report_action = QAction("Report Issue...", self)
        report_action.setStatusTip("Open GitHub to report a bug or request")
        report_action.triggered.connect(self._open_github_issues)
        help_menu.addAction(report_action)

        # Local debug action to verify telemetry is wired correctly
        debug_telemetry_action = QAction("Trigger Telemetry Test (Debug)...", self)
        debug_telemetry_action.setStatusTip("Send a test error to the telemetry backend (if enabled)")
        debug_telemetry_action.triggered.connect(self._trigger_telemetry_test)
        help_menu.addAction(debug_telemetry_action)

        # Quick toggle between info/output tabs
        view_menu.addSeparator()
        show_info_action = QAction("Show Info Tab", self)
        show_info_action.triggered.connect(lambda: self.info_tabs.setCurrentWidget(self.info_widget))
        view_menu.addAction(show_info_action)

        show_output_action = QAction("Show Output Tab", self)
        show_output_action.triggered.connect(lambda: self.info_tabs.setCurrentWidget(self.output_console))
        view_menu.addAction(show_output_action)

    def _create_status_bar(self):
        """Create CAD-style status bar with multiple sections."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Main status message
        self.status_message = QLabel("Ready")
        self.status_bar.addWidget(self.status_message, 1)

        # System monitor (CPU/Memory) - leftmost permanent widget
        self.system_monitor = SystemMonitorWidget(self)
        self.status_bar.addPermanentWidget(self.system_monitor)

        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet(f"color: {CADTheme.get_color('border', 'divider')};")
        self.status_bar.addPermanentWidget(sep1)

        # Object count
        self.status_objects = QLabel("Objects: 0")
        self.status_bar.addPermanentWidget(self.status_objects)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet(f"color: {CADTheme.get_color('border', 'divider')};")
        self.status_bar.addPermanentWidget(sep2)

        # Vertex/element count
        self.status_elements = QLabel("Vessels: 0")
        self.status_bar.addPermanentWidget(self.status_elements)

        # Separator
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.VLine)
        sep3.setStyleSheet(f"color: {CADTheme.get_color('border', 'divider')};")
        self.status_bar.addPermanentWidget(sep3)

        # Viewport info (rightmost)
        self.status_viewport = QLabel("View: Perspective")
        self.status_bar.addPermanentWidget(self.status_viewport)

    def load_domain(self, domain):
        """
        Load a domain object into the visualization.

        Parameters
        ----------
        domain : svv.domain.Domain
            Domain object to visualize
        """
        self.domain = domain
        self.vtk_widget.set_domain(domain)
        self.point_selector.set_domain(domain)

        # Update object browser
        if self.object_browser.domain_item:
            self.object_browser.domain_item.setText(0, "Domain (loaded)")

        # Update info panel
        self.info_label.setText(
            f"<b>Domain Information</b><br>"
            f"File: {getattr(domain, 'filename', 'Unknown')}<br>"
            f"Characteristic Length: {getattr(domain, 'characteristic_length', 'N/A')}<br>"
            f"Status: Ready for vascular generation"
        )
        self.info_dock.show()
        self.log_output(f"Loaded domain from {getattr(domain, 'filename', 'Unknown')}")

        self.update_status("Domain loaded successfully")
        self.update_object_count()

    def load_domain_dialog(self):
        """Open file dialog to load domain."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Domain",
            "",
            "Domain/Mesh Files (*.dmn *.vtu *.vtp *.stl);;All Files (*)"
        )

        if file_path:
            self._start_domain_load(file_path)

    def save_configuration(self):
        """Save current configuration."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            "",
            "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            try:
                import json
                config = self.point_selector.get_configuration()
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                self.update_status("Configuration saved")
            except Exception as e:
                self._record_telemetry(e, action="save_configuration")
                self.update_status("Save failed")
                QMessageBox.critical(
                    self,
                    "Error Saving",
                    f"Failed to save:\n\n{str(e)}"
                )

    def save_domain_dialog(self):
        """Save the current domain to a .dmn file."""
        if self.domain is None:
            self._record_telemetry(
                message="Save domain requested but no domain loaded",
                level="warning",
                action="save_domain_no_domain",
            )
            QMessageBox.warning(
                self,
                "No Domain",
                "No domain is currently loaded.\n\n"
                "Load a domain first using File > Open."
            )
            return

        # Options dialog
        dlg = QDialog(self)
        dlg.setWindowTitle("Save Domain Options")
        form = QFormLayout()

        # Include boundary checkbox
        include_boundary_cb = QCheckBox()
        include_boundary_cb.setChecked(True)
        include_boundary_cb.setToolTip(
            "Include boundary mesh data for faster visualization.\n"
            "Increases file size but avoids recomputation on load."
        )
        form.addRow("Include boundary mesh:", include_boundary_cb)

        # Include full mesh checkbox
        include_mesh_cb = QCheckBox()
        include_mesh_cb.setChecked(False)
        include_mesh_cb.setToolTip(
            "Include full tetrahedral mesh data.\n"
            "Significantly increases file size. Only needed for advanced use cases."
        )
        form.addRow("Include full mesh:", include_mesh_cb)

        # Info label
        from PySide6.QtWidgets import QLabel
        info_label = QLabel(
            "<i>Saving to .dmn preserves create() and solve() results,<br>"
            "allowing faster loading in future sessions.</i>"
        )
        info_label.setWordWrap(True)
        form.addRow(info_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        dlg.setLayout(layout)

        if dlg.exec() != QDialog.Accepted:
            return

        include_boundary = include_boundary_cb.isChecked()
        include_mesh = include_mesh_cb.isChecked()

        # File dialog
        default_name = ""
        if hasattr(self.domain, 'filename') and self.domain.filename:
            # Suggest same name with .dmn extension
            from pathlib import Path
            default_name = str(Path(self.domain.filename).with_suffix('.dmn'))

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Domain",
            default_name,
            "Domain Files (*.dmn);;All Files (*)"
        )

        if not file_path:
            return

        from svv.domain.io.dmn import ensure_dmn_path
        file_path = ensure_dmn_path(file_path)

        # Save with progress
        self.update_status("Saving domain...")
        try:
            saved_path = self.domain.save(
                file_path,
                include_boundary=include_boundary,
                include_mesh=include_mesh
            )
            try:
                self.domain.filename = saved_path
            except Exception:
                pass
            self.update_status(f"Domain saved to {saved_path}")
            self.log_output(f"Domain saved to {saved_path}")
            QMessageBox.information(
                self,
                "Domain Saved",
                f"Domain successfully saved to:\n{saved_path}\n\n"
                f"Options:\n"
                f"• Include boundary: {'Yes' if include_boundary else 'No'}\n"
                f"• Include full mesh: {'Yes' if include_mesh else 'No'}"
            )
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self._record_telemetry(e, action="save_domain", traceback_str=tb)
            self.update_status("Save failed")
            QMessageBox.critical(
                self,
                "Error Saving Domain",
                f"Failed to save domain:\n\n{str(e)}"
            )

    def save_vascular_object_dialog(self):
        """Save the current Tree or Forest to a file."""
        # Determine what object we have
        obj = None
        obj_type = None
        if self.forest is not None:
            obj = self.forest
            obj_type = "forest"
        elif self.trees:
            obj = self.trees[0]
            obj_type = "tree"

        if obj is None:
            self._record_telemetry(
                message="Save vascular object requested but no tree/forest generated",
                level="warning",
                action="save_vascular_no_object",
            )
            QMessageBox.warning(
                self,
                "No Vascular Object",
                "No Tree or Forest has been generated.\n\n"
                "Generate a Tree or Forest first using the Parameter Panel."
            )
            return

        # Options dialog
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Save {obj_type.capitalize()} Options")
        form = QFormLayout()

        # Include timing checkbox
        include_timing_cb = QCheckBox()
        include_timing_cb.setChecked(False)
        include_timing_cb.setToolTip(
            "Include generation timing data.\n"
            "Useful for profiling/debugging but increases file size."
        )
        form.addRow("Include timing data:", include_timing_cb)

        # Info label
        if obj_type == "tree":
            info_text = (
                f"<i>Saving Tree with {obj.n_terminals} vessels.<br>"
                "The domain is NOT saved - you must call set_domain() after loading.</i>"
            )
            extension = ".tree"
            filter_str = "Tree Files (*.tree);;All Files (*)"
        else:
            total_vessels = sum(
                sum(tree.n_terminals for tree in network)
                for network in obj.networks
            )
            info_text = (
                f"<i>Saving Forest with {obj.n_networks} network(s), {total_vessels} total vessels.<br>"
                "The domain is NOT saved - you must call set_domain() after loading.</i>"
            )
            extension = ".forest"
            filter_str = "Forest Files (*.forest);;All Files (*)"

        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        form.addRow(info_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        dlg.setLayout(layout)

        if dlg.exec() != QDialog.Accepted:
            return

        include_timing = include_timing_cb.isChecked()

        # File dialog
        default_name = f"vascular_{obj_type}{extension}"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Save {obj_type.capitalize()}",
            default_name,
            filter_str
        )

        if not file_path:
            return

        # Ensure correct extension
        if not file_path.lower().endswith(extension):
            file_path += extension

        # Save with progress
        self.update_status(f"Saving {obj_type}...")
        try:
            saved_path = obj.save(file_path, include_timing=include_timing)
            self.update_status(f"{obj_type.capitalize()} saved to {saved_path}")
            self.log_output(f"{obj_type.capitalize()} saved to {saved_path}")

            # Build summary message
            if obj_type == "tree":
                summary = f"Vessels: {obj.n_terminals}"
            else:
                summary = f"Networks: {obj.n_networks}\n"
                for i, n_trees in enumerate(obj.n_trees_per_network):
                    network_vessels = sum(tree.n_terminals for tree in obj.networks[i])
                    summary += f"• Network {i}: {n_trees} tree(s), {network_vessels} vessels\n"

            QMessageBox.information(
                self,
                f"{obj_type.capitalize()} Saved",
                f"{obj_type.capitalize()} successfully saved to:\n{saved_path}\n\n"
                f"{summary}\n"
                f"Include timing: {'Yes' if include_timing else 'No'}\n\n"
                f"Note: Load with {obj_type.capitalize()}.load('{Path(saved_path).name}')\n"
                f"then call set_domain() to restore domain operations."
            )
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self._record_telemetry(e, action=f"save_{obj_type}", traceback_str=tb)
            self.update_status("Save failed")
            QMessageBox.critical(
                self,
                f"Error Saving {obj_type.capitalize()}",
                f"Failed to save {obj_type}:\n\n{str(e)}"
            )

    # ---- Fabricate / Simulation exports ----

    def _show_export_menu(self):
        """Show a popup menu with all export options (toolbar Export button)."""
        menu = QMenu(self)

        act_centerlines = menu.addAction("Export Centerlines...")
        act_solids = menu.addAction("Export Solids...")
        act_splines = menu.addAction("Export Splines...")
        menu.addSeparator()
        act_0d = menu.addAction("Export 0D Simulation...")
        act_3d = menu.addAction("Export 3D Simulation...")

        # Show the menu below the Export toolbar button.
        btn = self.file_toolbar.widgetForAction(self.action_export)
        if btn is not None:
            pos = btn.mapToGlobal(btn.rect().bottomLeft())
        else:
            pos = self.cursor().pos()

        selected = menu.exec(pos)
        if selected is act_centerlines:
            self.export_centerlines_dialog()
        elif selected is act_solids:
            self.export_solids_dialog()
        elif selected is act_splines:
            self.export_splines_dialog()
        elif selected is act_0d:
            self.export_0d_simulation_dialog()
        elif selected is act_3d:
            self.export_3d_simulation_dialog()

    def _require_synthetic_object(self):
        """
        Return the current synthetic object (Tree or Forest) or show an error.

        Returns
        -------
        object or None
            The Tree or Forest instance currently held by the GUI.
        """
        obj = None
        if self.forest is not None:
            obj = self.forest
        elif self.trees:
            obj = self.trees[0]
        if obj is None:
            QMessageBox.warning(
                self,
                "No Vascular Network",
                "No generated Tree or Forest was found.\n\n"
                "Generate a network first using the Generate menu or panel."
            )
            self._record_telemetry(
                message="Operation requested without a generated Tree/Forest",
                level="warning",
                action="require_synthetic_object",
            )
        return obj

    def export_centerlines_dialog(self):
        """Export centerline spline data for the current Tree/Forest."""
        obj = self._require_synthetic_object()
        if obj is None:
            return
        # Options dialog for centerline export
        dlg = QDialog(self)
        dlg.setWindowTitle("Export Centerlines Options")
        form = QFormLayout()

        points_spin = QSpinBox()
        points_spin.setRange(1, 100000)
        points_spin.setValue(100)  # matches Tree.export_centerlines default
        points_spin.setToolTip("Sampling density along centerlines in points per unit length.")
        form.addRow("Points per unit length:", points_spin)

        boundary_cb = QCheckBox("Export boundary points (inlets/outlets)")
        boundary_cb.setChecked(True)
        boundary_cb.setToolTip(
            "Write a companion file with labeled inlet and outlet coordinates.\n"
            "The inlet is the root (start) point of the tree.\n"
            "Outlets are the terminal (leaf) vessel endpoints."
        )
        form.addRow("", boundary_cb)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        dlg.setLayout(layout)

        if dlg.exec() != QDialog.Accepted:
            return

        points_per_unit_length = points_spin.value()
        export_boundary = boundary_cb.isChecked()

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Centerlines",
            "",
            "VTK PolyData (*.vtp);;All Files (*)"
        )
        if not file_path:
            return

        try:
            import pyvista as pv
            from pathlib import Path as _Path

            boundary_points = []

            if hasattr(obj, "export_centerlines"):
                result = obj.export_centerlines(points_per_unit_length=points_per_unit_length)
                # Unpack: (centerlines, polys, boundary_points) or legacy (centerlines, polys)
                if len(result) >= 3:
                    centerlines, _, boundary_points = result[0], result[1], result[2]
                else:
                    centerlines = result[0]
            else:
                raise ValueError("Selected object does not support centerline export.")
            if isinstance(centerlines, pv.PolyData):
                centerlines.save(file_path)
            else:
                raise ValueError("Centerline export did not return a PyVista PolyData object.")

            # Write companion boundary points file.
            if export_boundary:
                import numpy as np
                bp_path = _Path(file_path).with_name(
                    _Path(file_path).stem + "_inlet_outlet.txt"
                )
                inlets = [bp for bp in boundary_points if bp['type'] == 'inlet']
                outlets = [bp for bp in boundary_points if bp['type'] == 'outlet']
                with bp_path.open("w", encoding="utf-8") as f:
                    f.write("inlet\n")
                    for bp in inlets:
                        p = np.asarray(bp['point'])
                        f.write(f"{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}\n")
                    f.write("outlet\n")
                    for bp in outlets:
                        p = np.asarray(bp['point'])
                        f.write(f"{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}\n")
                self.update_status(
                    f"Centerlines exported to {file_path}  |  boundary points → {bp_path.name}"
                )
            else:
                self.update_status(f"Centerlines exported to {file_path}")
        except Exception as e:
            self._record_telemetry(e, action="export_centerlines")
            QMessageBox.critical(
                self,
                "Export Centerlines Failed",
                f"Failed to export centerlines:\n\n{e}"
            )

    def export_solids_dialog(self):
        """Export solid vessel surfaces for the current Tree/Forest."""
        obj = self._require_synthetic_object()
        if obj is None:
            return
        # Options dialog for solid export
        dlg = QDialog(self)
        dlg.setWindowTitle("Export Solids Options")
        form = QFormLayout()

        thickness_spin = QDoubleSpinBox()
        thickness_spin.setRange(0.0, 10.0)
        thickness_spin.setDecimals(4)
        thickness_spin.setSingleStep(0.01)
        thickness_spin.setValue(0.0)
        thickness_spin.setToolTip("Shell thickness to apply when exporting solids (0 = no shell).")
        form.addRow("Shell thickness:", thickness_spin)

        watertight_cb = QCheckBox("Watertight (Tree only)")
        watertight_cb.setChecked(False)
        watertight_cb.setToolTip("Export a watertight solid model (applies to single Tree exports).")
        # Disable watertight option for Forest objects where it is not supported directly
        from svv.forest.forest import Forest as _ForestType
        if isinstance(obj, _ForestType):
            watertight_cb.setEnabled(False)
        form.addRow("", watertight_cb)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        dlg.setLayout(layout)

        if dlg.exec() != QDialog.Accepted:
            return

        shell_thickness = thickness_spin.value()
        watertight = watertight_cb.isChecked()

        out_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory for Solids",
            ""
        )
        if not out_dir:
            return

        try:
            if hasattr(obj, "export_solid"):
                # For Tree this returns a single mesh; for Forest it writes
                # per-tree solids into the directory.
                from svv.tree.tree import Tree as _TreeType
                if isinstance(obj, _TreeType):
                    model = obj.export_solid(outdir=out_dir, shell_thickness=shell_thickness, watertight=watertight)
                    if model is not None:
                        try:
                            model.save(os.path.join(out_dir, "tree_solid.vtp"))
                        except Exception:
                            pass
                else:
                    # Forest or other object types: ignore watertight flag
                    from svv.forest.forest import Forest as _ForestType
                    if isinstance(obj, _ForestType):
                        import numpy as np
                        import pyvista as pv

                        def _build_independent_network_solid(net_idx: int, tree_idxs: Optional[list[int]]):
                            solids = []

                            def add_segment(p0, p1, radius):
                                p0 = np.asarray(p0, dtype=float).reshape(-1)[:3]
                                p1 = np.asarray(p1, dtype=float).reshape(-1)[:3]
                                direction = p1 - p0
                                length = float(np.linalg.norm(direction))
                                if not np.isfinite(length) or length <= 0.0:
                                    return
                                radius = float(radius)
                                if not np.isfinite(radius) or radius <= 0.0:
                                    return
                                direction = direction / length
                                center = (p0 + p1) / 2.0
                                solids.append(pv.Cylinder(center=center, direction=direction, radius=radius, height=length))

                            if tree_idxs:
                                include_trees = [t for t in tree_idxs if 0 <= t < len(obj.networks[net_idx])]
                            else:
                                include_trees = list(range(len(obj.networks[net_idx])))

                            for tree_idx in include_trees:
                                data = np.asarray(obj.networks[net_idx][tree_idx].data)
                                if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] < 22:
                                    continue
                                for row in data:
                                    add_segment(row[0:3], row[3:6], row[21])

                            if not solids:
                                return None
                            return pv.merge(solids)

                        # Respect GUI visibility toggles: export only networks that are currently checked.
                        selection: dict[int, list[int]] = {}
                        saw_network_items = False
                        browser = getattr(self, "object_browser", None)
                        forests_item = getattr(browser, "forests_item", None) if browser is not None else None
                        if forests_item is not None:
                            try:
                                for i in range(forests_item.childCount()):
                                    net_item = forests_item.child(i)
                                    data = net_item.data(0, Qt.UserRole)
                                    if not data or not isinstance(data, tuple) or data[0] != "network":
                                        continue
                                    saw_network_items = True
                                    net_idx = int(data[1])
                                    if net_item.checkState(0) != Qt.Checked:
                                        continue

                                    # Tree selection is optional: when no trees are checked,
                                    # we export the whole network.
                                    tree_idxs: list[int] = []
                                    for j in range(net_item.childCount()):
                                        t_item = net_item.child(j)
                                        t_data = t_item.data(0, Qt.UserRole)
                                        if (
                                            not t_data
                                            or not isinstance(t_data, tuple)
                                            or t_data[0] != "tree"
                                            or t_data[1] != "forest"
                                        ):
                                            continue
                                        tree_idx = int(t_data[3])
                                        if t_item.checkState(0) == Qt.Checked:
                                            tree_idxs.append(tree_idx)
                                    selection[net_idx] = sorted(set(tree_idxs))
                            except Exception:
                                selection = {}
                                saw_network_items = False

                        # If we couldn't read GUI state, export all networks.
                        if not saw_network_items:
                            selection = {i: [] for i in range(int(getattr(obj, "n_networks", 0) or 0))}

                        if not selection:
                            QMessageBox.information(
                                self,
                                "No Visible Networks",
                                "No visible networks are selected for export.\n\n"
                                "Enable one or more Network checkboxes in the Model Tree and try again."
                            )
                            return

                        connections = getattr(obj, "connections", None)
                        tree_connections = getattr(connections, "tree_connections", None) if connections is not None else None
                        has_connected = (
                            tree_connections is not None
                            and isinstance(tree_connections, list)
                            and len(tree_connections) == obj.n_networks
                        )

                        exported_any = False
                        for net_idx, tree_idxs in sorted(selection.items()):
                            if net_idx < 0 or net_idx >= obj.n_networks:
                                continue

                            net_dir = os.path.join(out_dir, f"network_{net_idx}")
                            os.makedirs(net_dir, exist_ok=True)

                            model = None
                            name = f"network_{net_idx}_mesh.vtp"
                            if has_connected:
                                tc = tree_connections[net_idx]
                                if tc is not None:
                                    model = tc.build_merged_solid(
                                        tree_indices=(tree_idxs if tree_idxs else None),
                                        include_connections=True,
                                    )
                                    name = f"network_{net_idx}_connected_mesh.vtp"
                            if model is None:
                                model = _build_independent_network_solid(net_idx, tree_idxs)

                            if model is None:
                                continue
                            try:
                                model.save(os.path.join(net_dir, name))
                                exported_any = True
                            except Exception:
                                pass

                        if not exported_any:
                            QMessageBox.information(
                                self,
                                "Export Solids",
                                "No solids were exported.\n\n"
                                "Ensure one or more networks are visible and try again."
                            )
                            return
                    else:
                        obj.export_solid(outdir=out_dir, shell_thickness=shell_thickness)
            else:
                raise ValueError("Selected object does not support solid export.")
            self.update_status(f"Solids exported to {out_dir}")
        except Exception as e:
            self._record_telemetry(e, action="export_solids")
            QMessageBox.critical(
                self,
                "Export Solids Failed",
                f"Failed to export solids:\n\n{e}"
            )

    def export_0d_simulation_dialog(self):
        """Export 0D simulation files for the current Tree/Forest."""
        obj = self._require_synthetic_object()
        if obj is None:
            return
        from svv.simulation.fluid.rom.zero_d.zerod_tree import export_0d_simulation as export_tree_0d
        from svv.simulation.fluid.rom.zero_d.zerod_forest import export_0d_simulation as export_forest_0d

        is_tree = isinstance(obj, _svv_tree_mod.Tree)
        is_forest = isinstance(obj, _svv_forest_mod.Forest)

        # ---- Build options dialog ----
        dlg = QDialog(self)
        dlg.setWindowTitle("0D Simulation Options")
        form = QFormLayout()

        # Common options
        steady_cb = QCheckBox("Steady inflow")
        steady_cb.setChecked(True)
        form.addRow("Inflow type:", steady_cb)

        cycles_spin = QSpinBox()
        cycles_spin.setRange(1, 1000)
        cycles_spin.setValue(1)
        cycles_spin.setToolTip("Number of cardiac cycles to simulate.")
        form.addRow("Cardiac cycles:", cycles_spin)

        pts_spin = QSpinBox()
        pts_spin.setRange(1, 1000)
        pts_spin.setValue(5)
        pts_spin.setToolTip("Time points per cardiac cycle.")
        form.addRow("Time points / cycle:", pts_spin)

        folder_edit = QLineEdit("0d_tmp")
        folder_edit.setToolTip("Folder name inside the output directory to hold 0D files.")
        form.addRow("Folder name:", folder_edit)

        # Tree-specific fluid properties
        density_spin = None
        viscosity_spin = None
        if is_tree:
            density_spin = QDoubleSpinBox()
            density_spin.setRange(0.1, 10.0)
            density_spin.setDecimals(3)
            density_spin.setSingleStep(0.01)
            density_spin.setValue(1.06)
            density_spin.setToolTip("Fluid density (typically blood) in CGS units.")
            form.addRow("Density:", density_spin)

            viscosity_spin = QDoubleSpinBox()
            viscosity_spin.setRange(0.0, 10.0)
            viscosity_spin.setDecimals(4)
            viscosity_spin.setSingleStep(0.001)
            viscosity_spin.setValue(0.04)
            viscosity_spin.setToolTip("Fluid viscosity in CGS units.")
            form.addRow("Viscosity:", viscosity_spin)

        # Material model
        material_combo = QComboBox()
        material_combo.addItems(["olufsen", "linear"])
        material_combo.setCurrentText("olufsen")
        material_combo.setToolTip("Wall material model for vessel compliance.")
        form.addRow("Material:", material_combo)

        # Viscosity model
        viscosity_model_combo = QComboBox()
        viscosity_model_combo.addItems(["constant", "modified viscosity law"])
        viscosity_model_combo.setCurrentText("constant")
        viscosity_model_combo.setToolTip("Viscosity model for blood rheology.")
        form.addRow("Viscosity model:", viscosity_model_combo)

        # Solver and BC-related options
        vivo_cb = QCheckBox("In vivo conditions")
        vivo_cb.setChecked(True)
        form.addRow("Mode:", vivo_cb)

        distal_pressure_spin = QDoubleSpinBox()
        distal_pressure_spin.setRange(0.0, 500.0)
        distal_pressure_spin.setDecimals(2)
        distal_pressure_spin.setSingleStep(1.0)
        distal_pressure_spin.setValue(0.0)
        distal_pressure_spin.setToolTip("Distal pressure (e.g., outlet pressure) in CGS-compatible units.")
        form.addRow("Distal pressure:", distal_pressure_spin)

        capacitance_cb = QCheckBox("Include capacitance")
        capacitance_cb.setChecked(True)
        form.addRow("Capacitance:", capacitance_cb)

        inductance_cb = QCheckBox("Include inductance")
        inductance_cb.setChecked(True)
        form.addRow("Inductance:", inductance_cb)

        get_solver_cb = QCheckBox("Search for 0D solver")
        get_solver_cb.setChecked(False)
        form.addRow("Search solver:", get_solver_cb)

        solver_path_edit = QLineEdit()
        solver_path_edit.setPlaceholderText("Optional explicit path to 0D solver")
        form.addRow("Solver path:", solver_path_edit)

        # Tree-specific filenames
        filename_edit = None
        geom_filename_edit = None
        if is_tree:
            filename_edit = QLineEdit("solver_0d.in")
            filename_edit.setToolTip("Name of the main 0D solver input file.")
            form.addRow("Solver input filename:", filename_edit)

            geom_filename_edit = QLineEdit("geom.csv")
            geom_filename_edit.setToolTip("Filename for exported vessel geometry CSV.")
            form.addRow("Geometry filename:", geom_filename_edit)

        # Forest-specific network/inlet options
        network_spin = None
        inlets_edit = None
        if is_forest:
            network_spin = QSpinBox()
            network_spin.setRange(0, max(0, len(obj.networks) - 1))
            network_spin.setValue(0)
            network_spin.setToolTip("Network index to export (0-based).")
            form.addRow("Network ID:", network_spin)

            inlets_edit = QLineEdit("0")
            inlets_edit.setToolTip("Comma-separated list of tree indices to treat as inlets (e.g., '0,2').")
            form.addRow("Inlet trees:", inlets_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        dlg.setLayout(layout)

        if dlg.exec() != QDialog.Accepted:
            return

        # Collect options
        steady = steady_cb.isChecked()
        number_cardiac_cycles = cycles_spin.value()
        number_time_pts_per_cycle = pts_spin.value()
        folder = folder_edit.text().strip() or "0d_tmp"
        material = material_combo.currentText()
        viscosity_model = viscosity_model_combo.currentText()
        vivo = vivo_cb.isChecked()
        distal_pressure = float(distal_pressure_spin.value())
        capacitance = capacitance_cb.isChecked()
        inductance = inductance_cb.isChecked()
        get_0d_solver = get_solver_cb.isChecked()
        path_to_0d_solver = solver_path_edit.text().strip() or None

        density = float(density_spin.value()) if density_spin is not None else None
        viscosity = float(viscosity_spin.value()) if viscosity_spin is not None else None
        filename = filename_edit.text().strip() if filename_edit is not None else "solver_0d.in"
        geom_filename = geom_filename_edit.text().strip() if geom_filename_edit is not None else "geom.csv"

        network_id = network_spin.value() if network_spin is not None else None
        inlets = None
        if inlets_edit is not None:
            txt = inlets_edit.text().strip()
            if txt:
                try:
                    inlets = [int(part.strip()) for part in txt.split(",") if part.strip() != ""]
                except ValueError:
                    QMessageBox.warning(
                        self,
                        "Invalid Inlets",
                        "Inlet list must be a comma-separated list of integers (tree indices)."
                    )
                    self._record_telemetry(
                        message="Invalid inlet list format for 0D export dialog",
                        level="warning",
                        action="export_0d_invalid_inlets",
                    )
                    return

        out_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory for 0D Simulation",
            ""
        )
        if not out_dir:
            return

        # Find a non-conflicting subfolder name (e.g., "0d_tmp (1)") inside the
        # selected output directory and create it up front.
        base_folder_name = folder
        candidate = base_folder_name
        suffix = 1
        while os.path.exists(os.path.join(out_dir, candidate)):
            candidate = f"{base_folder_name} ({suffix})"
            suffix += 1
        target_path = os.path.join(out_dir, candidate)
        try:
            os.makedirs(target_path, exist_ok=True)
        except Exception as exc:
            self._record_telemetry(exc, action="export_0d_output_folder")
            QMessageBox.critical(
                self,
                "Output Folder Error",
                f"Could not create output folder for 0D export:\n\n{exc}"
            )
            return

        if candidate != base_folder_name:
            QMessageBox.information(
                self,
                "Output Folder Renamed",
                f"The folder '{base_folder_name}' already exists in the selected directory.\n"
                f"Using '{candidate}' instead.\n\nFiles will be written to:\n{target_path}"
            )

        folder = candidate

        export_path = target_path

        try:
            if is_tree:
                export_tree_0d(
                    obj,
                    steady=steady,
                    outdir=out_dir,
                    folder=folder,
                    number_cardiac_cycles=number_cardiac_cycles,
                    flow=None,
                    number_time_pts_per_cycle=number_time_pts_per_cycle,
                    density=density if density is not None else 1.06,
                    viscosity=viscosity if viscosity is not None else 0.04,
                    material=material,
                    get_0d_solver=get_0d_solver,
                    path_to_0d_solver=path_to_0d_solver,
                    viscosity_model=viscosity_model,
                    vivo=vivo,
                    distal_pressure=distal_pressure,
                    capacitance=capacitance,
                    inductance=inductance,
                    filename=filename,
                    geom_filename=geom_filename,
                )
            elif is_forest:
                if network_id is None or inlets is None:
                    raise ValueError("Network ID and inlets are required for forest 0D export.")
                export_forest_0d(
                    obj,
                    network_id,
                    inlets,
                    steady=steady,
                    outdir=out_dir,
                    folder=folder,
                    number_cardiac_cycles=number_cardiac_cycles,
                    flow=None,
                    number_time_pts_per_cycle=number_time_pts_per_cycle,
                    material=material,
                    get_0d_solver=get_0d_solver,
                    path_to_0d_solver=path_to_0d_solver,
                    viscosity_model=viscosity_model,
                    vivo=vivo,
                    distal_pressure=distal_pressure,
                    capacitance=capacitance,
                    inductance=inductance,
                )
            else:
                raise ValueError("Unsupported object type for 0D export.")
            self.update_status(f"0D simulation files exported to {export_path}")
        except Exception as e:
            self._record_telemetry(e, action="export_0d")
            QMessageBox.critical(
                self,
                "Export 0D Simulation Failed",
                f"Failed to export 0D simulation files:\n\n{e}"
            )

    def export_3d_simulation_dialog(self):
        """Build meshes and export 3D simulation setup for the current Tree/Forest."""
        obj = self._require_synthetic_object()
        if obj is None:
            return

        out_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory for 3D Simulation",
            ""
        )
        if not out_dir:
            return

        try:
            from svv.simulation.simulation import Simulation

            sim = Simulation(obj, directory=out_dir)
            # Basic workflow: build fluid meshes only, then extract faces.
            sim.build_meshes(fluid=True, tissue=False)
            sim.extract_faces()
            # 3D solver input export is typically handled by downstream tools;
            # here we just ensure meshes/faces exist in the chosen directory.
            self.update_status(f"3D simulation meshes prepared in {out_dir}")
        except Exception as e:
            self._record_telemetry(e, action="export_3d")
            QMessageBox.critical(
                self,
                "Export 3D Simulation Failed",
                f"Failed to prepare 3D simulation setup:\n\n{e}"
            )

    def export_splines_dialog(self):
        """
        Export spline samples to a text file.

        - Tree: exports per-branch spline samples (tree format)
        - Forest:
          - connected: exports connected-network spline samples (network format)
          - unconnected: exports per-tree spline samples (tree format)
        """
        obj = self._require_synthetic_object()
        if obj is None:
            return

        from svv.tree.tree import Tree as _TreeType
        from svv.forest.forest import Forest as _ForestType
        is_tree = isinstance(obj, _TreeType)
        is_forest = isinstance(obj, _ForestType)

        # Options dialog
        dlg = QDialog(self)
        dlg.setWindowTitle("Export Splines Options")
        form = QFormLayout()

        sample_spin = QSpinBox()
        sample_spin.setRange(10, 10000)
        sample_spin.setValue(100)
        sample_spin.setToolTip("Number of sample points per vessel spline.")
        form.addRow("Sample points per vessel:", sample_spin)

        seperate_cb = QCheckBox("Separate inlet/outlet labeling")
        seperate_cb.setChecked(False)
        seperate_cb.setToolTip("If checked, write an extra label column for proximal vs distal points.")
        form.addRow("", seperate_cb)

        boundary_cb = QCheckBox("Export boundary points (inlets/outlets)")
        boundary_cb.setChecked(True)
        boundary_cb.setToolTip(
            "Write a companion file listing inlet and outlet coordinates.\n"
            "Inlets are the root (start) points of each tree.\n"
            "Outlets are the terminal (leaf) vessel endpoints."
        )
        form.addRow("", boundary_cb)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        dlg.setLayout(layout)

        if dlg.exec() != QDialog.Accepted:
            return

        spline_sample_points = sample_spin.value()
        seperate = seperate_cb.isChecked()
        export_boundary = boundary_cb.isChecked()

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Splines",
            "",
            "Text Files (*.txt);;All Files (*)"
        )
        if not file_path:
            return

        try:
            from pathlib import Path as _Path

            import numpy as np
            from scipy.interpolate import splev
            from svv.tree.export.write_splines import get_interpolated_sv_data

            out_path = _Path(file_path)
            if not out_path.suffix:
                out_path = out_path.with_suffix(".txt")

            def _write_tree_splines(tree: _TreeType, dst: _Path) -> None:
                if getattr(tree, "data", None) is None:
                    raise ValueError("Tree has no data to export.")
                *_, interp_xyzr = get_interpolated_sv_data(tree.data)
                if not interp_xyzr:
                    raise ValueError("Tree has no spline branches to export.")
                t = np.linspace(0, 1, num=spline_sample_points)
                with dst.open("w", encoding="utf-8") as spline_file:
                    for vessel in range(len(interp_xyzr)):
                        spline_file.write(f"Vessel: {vessel}, Number of Points: {spline_sample_points}\n\n")
                        data = splev(t, interp_xyzr[vessel][0])
                        for k in range(spline_sample_points):
                            if seperate:
                                label = 1 if k > spline_sample_points // 2 else 0
                                spline_file.write(
                                    f"{data[0][k]}, {data[1][k]}, {data[2][k]}, {data[3][k]}, {label}\n"
                                )
                            else:
                                spline_file.write(
                                    f"{data[0][k]}, {data[1][k]}, {data[2][k]}, {data[3][k]}\n"
                                )
                        spline_file.write("\n")

            def _get_start_point(tree: _TreeType):
                """Return the root segment's proximal point (col 0:3)."""
                data = np.asarray(tree.data)
                if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] < 6:
                    return None
                root_idx = np.where(np.isnan(data[:, 17]))[0]
                if len(root_idx) == 0:
                    return None
                return data[root_idx[0], 0:3].copy()

            def _pair_forest_boundary(forest):
                """Pair tree start points as inlet/outlet per network.

                Within each network trees are paired: (tree0→inlet,
                tree1→outlet), (tree2→inlet, tree3→outlet), etc.
                An unpaired last tree is placed as inlet.
                """
                inlets, outlets = [], []
                for network in (getattr(forest, "networks", []) or []):
                    for i in range(0, len(network), 2):
                        pt = _get_start_point(network[i])
                        if pt is not None:
                            inlets.append(pt)
                        if i + 1 < len(network):
                            pt = _get_start_point(network[i + 1])
                            if pt is not None:
                                outlets.append(pt)
                return inlets, outlets

            def _write_boundary_file(inlets, outlets, dst: _Path):
                with dst.open("w", encoding="utf-8") as f:
                    f.write("inlet\n")
                    for p in inlets:
                        f.write(f"{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}\n")
                    f.write("outlet\n")
                    for p in outlets:
                        f.write(f"{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}\n")

            if is_tree:
                _write_tree_splines(obj, out_path)
                if export_boundary:
                    bp_path = out_path.with_name(out_path.stem + "_inlet_outlet.txt")
                    pt = _get_start_point(obj)
                    _write_boundary_file([pt] if pt is not None else [], [], bp_path)
                    self.update_status(f"Tree splines exported to {out_path}  |  boundary points → {bp_path.name}")
                else:
                    self.update_status(f"Tree splines exported to {out_path}")
                return

            if is_forest and getattr(obj, "connections", None) is not None:
                from svv.forest.export.export_spline import write_splines, export_spline

                # export_spline operates on TreeConnection objects from Forest.connections.tree_connections
                # It returns: (interp_xyz, interp_radii, interp_normals, all_points, all_radii, all_normals)
                all_points = []
                all_radii = []
                for tc in obj.connections.tree_connections:
                    _, _, _, net_points, net_radii, _ = export_spline(tc)
                    all_points.extend(net_points)
                    all_radii.extend(net_radii)

                # Temporarily change working directory to target location for write_splines
                cwd = os.getcwd()
                target_dir = str(out_path.parent) if str(out_path.parent) else cwd
                try:
                    os.chdir(target_dir)
                    # write_splines writes 'network_{i}_b_splines.txt' in cwd; we adapt
                    write_splines(
                        all_points,
                        all_radii,
                        spline_sample_points=spline_sample_points,
                        seperate=seperate,
                        write_splines=True,
                    )
                    default_path = os.path.join(target_dir, "network_0_b_splines.txt")
                    if os.path.exists(default_path):
                        os.replace(default_path, str(out_path))
                finally:
                    os.chdir(cwd)

                if export_boundary:
                    inlets, outlets = _pair_forest_boundary(obj)
                    bp_path = out_path.with_name(out_path.stem + "_inlet_outlet.txt")
                    _write_boundary_file(inlets, outlets, bp_path)
                    self.update_status(f"Connected forest splines exported to {out_path}  |  boundary points → {bp_path.name}")
                else:
                    self.update_status(f"Connected forest splines exported to {out_path}")
                return

            if is_forest:
                # Forest without connections: export each Tree independently using the tree spline format.
                trees = []
                for net_idx, network in enumerate(getattr(obj, "networks", []) or []):
                    for tree_idx, tree in enumerate(network):
                        trees.append((net_idx, tree_idx, tree))
                if not trees:
                    raise ValueError("Forest contains no trees to export.")

                if len(trees) == 1:
                    _, _, tree = trees[0]
                    _write_tree_splines(tree, out_path)
                    if export_boundary:
                        bp_path = out_path.with_name(out_path.stem + "_inlet_outlet.txt")
                        pt = _get_start_point(tree)
                        _write_boundary_file([pt] if pt is not None else [], [], bp_path)
                    self.update_status(f"Tree splines exported to {out_path}")
                    return

                stem = out_path.stem or "splines"
                suffix = out_path.suffix or ".txt"
                written = []
                for net_idx, tree_idx, tree in trees:
                    dst = out_path.with_name(f"{stem}_network{net_idx}_tree{tree_idx}{suffix}")
                    _write_tree_splines(tree, dst)
                    written.append(dst)

                if export_boundary:
                    inlets, outlets = _pair_forest_boundary(obj)
                    bp_path = out_path.with_name(f"{stem}_inlet_outlet.txt")
                    _write_boundary_file(inlets, outlets, bp_path)

                QMessageBox.information(
                    self,
                    "Splines Exported",
                    f"Exported {len(written)} tree spline file(s) to:\n{out_path.parent}"
                )
                self.update_status(f"Exported {len(written)} tree spline file(s)")
                return

            raise ValueError("Unsupported object type for spline export.")
        except Exception as e:
            self._record_telemetry(e, action="export_splines")
            QMessageBox.critical(
                self,
                "Export Splines Failed",
                f"Failed to export splines:\n\n{e}"
            )

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About svVascularize",
            "<b>svVascularize - Vascular Design</b><br><br>"
            "Professional vascular tree and forest generation tool<br>"
            "CAD-style interface for engineering and fabrication<br><br>"
            "Built with PySide6 and PyVista<br>"
            "© SimVascular"
        )

    def _record_telemetry(self, exc=None, message: Optional[str] = None, level: str = "error",
                          traceback_str: Optional[str] = None, **tags):
        """
        Send errors or warnings to telemetry without interrupting the GUI.

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
            # Telemetry should never break the UI
            pass

    def update_status(self, message):
        """Update status bar message."""
        self.status_message.setText(message)

    def log_output(self, message: str):
        """Append a message to the output console."""
        if hasattr(self, "output_console") and self.output_console:
            self.output_console.appendPlainText(message)

    def show_progress(self, message: str = "Working..."):
        """Display a non-blocking progress indicator in the Info panel."""
        if not hasattr(self, "info_progress") or self.info_progress is None:
            return
        safe_message = message.replace("%", "%%")
        self.info_progress.setFormat(f"{safe_message} %p%")
        self.info_progress.setRange(0, 0)  # Indeterminate
        self.info_progress.setValue(0)
        self.info_progress.setVisible(True)
        self.info_tabs.setCurrentWidget(self.info_widget)
        self.info_dock.show()

    def hide_progress(self):
        """Hide the Info panel progress indicator."""
        if not hasattr(self, "info_progress") or self.info_progress is None:
            return
        self.info_progress.setVisible(False)
        self.info_progress.setRange(0, 1)
        self.info_progress.setValue(0)

    def update_object_count(self):
        """Update object count in status bar."""
        count = 0
        if self.domain:
            count += 1
        count += len(self.trees)
        if self.forest:
            count += 1

        self.status_objects.setText(f"Objects: {count}")

    def update_vessel_count(self, count):
        """Update vessel count in status bar."""
        self.status_elements.setText(f"Vessels: {count}")

    def _toggle_scale_bar(self):
        """Toggle the scale bar visibility in the 3D viewport."""
        if hasattr(self, 'vtk_widget') and self.vtk_widget:
            self.vtk_widget.toggle_scale_bar()
            # Update the action checked state to match actual visibility
            if hasattr(self, 'action_toggle_scale_bar'):
                self.action_toggle_scale_bar.setChecked(self.vtk_widget.is_scale_bar_visible())

    def _toggle_grid(self):
        """Toggle the 3D grid visibility in the viewport."""
        if hasattr(self, 'vtk_widget') and self.vtk_widget:
            self.vtk_widget.toggle_grid()
            # Update the action checked state to match actual visibility
            if hasattr(self, 'action_toggle_grid'):
                self.action_toggle_grid.setChecked(self.vtk_widget.is_grid_visible())

    def _show_solution_inspector(self):
        """Show the Solution Inspector dockable panel."""
        if hasattr(self, "solution_dock") and self.solution_dock is not None:
            self.solution_dock.show()
            self.solution_dock.raise_()

    def _on_length_unit_changed(self, unit_symbol: str):
        """Update the scale bar when the length unit changes in the parameter panel."""
        if hasattr(self, 'vtk_widget') and self.vtk_widget:
            self.vtk_widget.set_scale_bar_unit(unit_symbol)

    def _trigger_telemetry_test(self):
        """
        Send a synthetic error to the telemetry backend without crashing the GUI.

        This is a local debug helper to verify that Sentry/telemetry is wired
        correctly. It will only send if telemetry has been initialized.
        """
        # Provide immediate feedback if telemetry is not active so users do not
        # expect an event to appear in Sentry when it cannot be sent.
        try:
            import svv.telemetry as _telemetry_mod

            if not getattr(_telemetry_mod, "telemetry_enabled", lambda: False)():
                # Diagnose why telemetry is not enabled
                reasons = []

                # Check if sentry_sdk is installed
                try:
                    import sentry_sdk
                except ImportError:
                    reasons.append("• sentry-sdk is not installed (run: pip install sentry-sdk)")

                # Check user consent setting
                try:
                    settings = QSettings("svVascularize", "GUI")
                    if settings.contains("telemetry/enabled"):
                        if not settings.value("telemetry/enabled", False, type=bool):
                            reasons.append("• Crash reporting was declined at startup")
                    else:
                        reasons.append("• Crash reporting consent has not been set yet")
                except Exception:
                    pass

                # Check if disabled via environment
                if os.environ.get("SVV_TELEMETRY_DISABLED", "").strip() == "1":
                    reasons.append("• SVV_TELEMETRY_DISABLED=1 is set in environment")

                if not reasons:
                    reasons.append("• Unknown reason - telemetry initialization may have failed")

                reason_text = "\n".join(reasons)
                QMessageBox.warning(
                    self,
                    "Telemetry Not Enabled",
                    f"Telemetry is currently disabled or not initialized.\n\n"
                    f"Possible reasons:\n{reason_text}\n\n"
                    f"To enable telemetry:\n"
                    f"1. Install sentry-sdk: pip install sentry-sdk\n"
                    f"2. Delete ~/.config/svVascularize/GUI.conf to reset consent\n"
                    f"3. Restart the GUI and click 'Yes' on the crash reporting dialog"
                )
                return
        except Exception:
            # If anything goes wrong during the check, fall back to attempting
            # a capture so we do not crash the GUI.
            pass

        try:
            raise RuntimeError("svVascularize GUI telemetry test error")
        except Exception as exc:
            try:
                capture_exception(exc)
            except Exception:
                pass
        QMessageBox.information(
            self,
            "Telemetry Test",
            "Sent a test error to the telemetry backend.\n\n"
            "Check your Sentry dashboard to verify the event was received."
        )

    def _open_github_issues(self):
        """Open the GitHub issues page for svVascularize in the default browser."""
        url = QUrl("https://github.com/SimVascular/svVascularize/issues/new/choose")
        if not url.isValid():
            return
        try:
            QDesktopServices.openUrl(url)
        except Exception:
            pass

    def _restore_layout(self):
        """Restore window geometry and dock widget layout from QSettings."""
        # Restore window geometry
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        # Restore window state (dock positions, toolbar positions, etc.)
        window_state = self.settings.value("windowState")
        if window_state:
            self.restoreState(window_state)

        # Always start with Solution Inspector hidden; user can open it explicitly
        if hasattr(self, "solution_dock") and self.solution_dock is not None:
            self.solution_dock.hide()

    def _save_layout(self):
        """Save current window geometry and dock widget layout to QSettings."""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.update_status("Layout saved")

    def _reset_layout(self):
        """Reset window layout to default configuration."""
        # Clear saved settings
        self.settings.remove("geometry")
        self.settings.remove("windowState")

        # Reset to default geometry
        self.setGeometry(100, 100, 1600, 1000)

        # Reset dock positions
        self.addDockWidget(Qt.LeftDockWidgetArea, self.tree_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.properties_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.info_dock)
        if hasattr(self, "solution_dock") and self.solution_dock is not None:
            self.addDockWidget(Qt.RightDockWidgetArea, self.solution_dock)

        # Show/hide default panels
        self.tree_dock.show()
        self.properties_dock.show()
        self.info_dock.hide()
        if hasattr(self, "solution_dock") and self.solution_dock is not None:
            self.solution_dock.hide()

        # Reset toolbar positions
        self.addToolBar(Qt.TopToolBarArea, self.file_toolbar)
        self.addToolBar(Qt.TopToolBarArea, self.view_toolbar)
        self.addToolBar(Qt.TopToolBarArea, self.gen_toolbar)

        self.update_status("Layout reset to defaults")

    def closeEvent(self, event):
        """
        Handle window close event - save layout and release resources.

        Parameters
        ----------
        event : QCloseEvent
            The close event
        """
        # Stop system monitor timer
        if hasattr(self, 'system_monitor') and self.system_monitor:
            self.system_monitor.stop()

        # Cancel any in-flight domain load
        if getattr(self, "_load_cancel_event", None) is not None:
            try:
                self._load_cancel_event.set()
            except Exception:
                pass

        # Shut down background executors
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

        # Shut down parameter panel worker threads
        if hasattr(self, "parameter_panel") and self.parameter_panel is not None:
            try:
                self.parameter_panel.shutdown()
            except Exception:
                pass

        # Release VTK/PyVista resources
        if hasattr(self, "vtk_widget") and self.vtk_widget is not None:
            try:
                self.vtk_widget.shutdown()
            except Exception:
                pass
        if hasattr(self, "solution_inspector") and self.solution_inspector is not None:
            try:
                self.solution_inspector.shutdown()
            except Exception:
                pass

        # Drop heavy references so Python can reclaim memory promptly
        self.domain = None
        self.trees = []
        self.forest = None
        if hasattr(self, "point_selector") and self.point_selector is not None:
            try:
                self.point_selector.domain = None
                self.point_selector.points.clear()
            except Exception:
                pass

        # Save current layout
        self._save_layout()

        # Mark session as cleanly closed so that next launch does not treat it
        # as a crash. This must happen late in shutdown to avoid false
        # positives when the window is closed programmatically.
        try:
            settings = QSettings("svVascularize", "GUI")
            settings.setValue("session/running", False)
        except Exception:
            pass

        # Accept the close event
        event.accept()

    # ---- Background domain loading ----
    def _load_domain_file(self, file_path: str, cancel_event: threading.Event, progress_queue: Optional[Queue] = None,
                          build_resolution: Optional[int] = None):
        def report_progress(value, label=None):
            """Report progress value and optionally update the label text."""
            if progress_queue is not None:
                try:
                    # Send tuple of (value, label) if label provided, else just value
                    if label is not None:
                        progress_queue.put((value, label))
                    else:
                        progress_queue.put(value)
                except Exception:
                    pass

        try:
            from svv.domain.domain import Domain
            suffix = Path(file_path).suffix.lower()
            if suffix == ".dmn":
                # Step 1: Load the .dmn file
                report_progress(5, "Loading domain file...")
                domain = Domain.load(file_path)
                if cancel_event.is_set():
                    return None
                report_progress(30, "Domain file loaded")

                # Check if mesh data was already loaded from the .dmn file
                # If so, we can skip the expensive tetrahedralization step
                mesh_already_loaded = (
                    getattr(domain, 'mesh', None) is not None and
                    getattr(domain, 'mesh_tree', None) is not None and
                    getattr(domain, 'boundary', None) is not None
                )

                build_failed = False
                build_error_msg = None

                if mesh_already_loaded:
                    # Mesh was included in .dmn file - skip rebuild
                    report_progress(90, "Domain loaded with cached mesh data")
                else:
                    # Step 2: Build (tetrahedralize and extract boundary)
                    report_progress(35, "Building domain (tetrahedralization + boundary extraction)...")
                    try:
                        if build_resolution is not None:
                            domain.build(resolution=build_resolution)
                        else:
                            domain.build()
                        report_progress(90, "Domain built successfully")
                    except Exception as exc:
                        # If build fails (e.g., TetGen errors) continue with loaded
                        # fast-eval structures only so tree/forest generation still
                        # works, but record the failure in telemetry for diagnosis.
                        build_failed = True
                        build_error_msg = str(exc)
                        try:
                            import traceback
                            tb = traceback.format_exc()
                            self._record_telemetry(exc, action="load_domain_build", traceback_str=tb)
                        except Exception:
                            pass
                        report_progress(90, "Domain build failed (continuing without mesh)")

                # Store build status on domain for later reference
                domain._build_failed = build_failed
                domain._build_error = build_error_msg
                if cancel_event.is_set():
                    return None
            elif suffix in {".vtp", ".vtu", ".stl"}:
                import pyvista as pv

                # Step 1: Read mesh file
                report_progress(5, "Reading mesh file...")
                mesh = pv.read(file_path)
                if cancel_event.is_set():
                    return None

                # Step 2: Create domain
                report_progress(10, "Creating domain (initializing implicit function)...")
                domain = Domain(mesh)
                domain.create()
                if cancel_event.is_set():
                    return None
                report_progress(30, "Domain created")

                # Step 3: Solve (compute fast evaluation structures)
                report_progress(35, "Solving domain (computing evaluation structures)...")
                domain.solve()
                if cancel_event.is_set():
                    return None
                report_progress(60, "Domain solved")

                # Step 4: Build (tetrahedralize and extract boundary)
                report_progress(65, "Building domain (tetrahedralization + boundary extraction)...")
                build_failed = False
                build_error_msg = None
                try:
                    if build_resolution is not None:
                        domain.build(resolution=build_resolution)
                    else:
                        domain.build()
                    report_progress(90, "Domain built successfully")
                except Exception as exc:
                    build_failed = True
                    build_error_msg = str(exc)
                    try:
                        import traceback
                        tb = traceback.format_exc()
                        self._record_telemetry(exc, action="load_mesh_build", traceback_str=tb)
                    except Exception:
                        pass
                    report_progress(90, "Domain build failed (continuing without mesh)")

                # Store build status on domain for later reference
                domain._build_failed = build_failed
                domain._build_error = build_error_msg
            else:
                raise ValueError(f"Unsupported file type: {suffix}")

            if cancel_event.is_set():
                return None

            if domain.boundary is None and hasattr(domain, 'patches') and len(domain.patches) > 0:
                report_progress(92, "Building boundary from patches...")
                domain.build()
                report_progress(98, "Boundary built")

            domain.filename = file_path
            report_progress(100, "Domain ready")
            return domain
        except Exception as exc:
            return exc

    def _start_domain_load(self, file_path: str):
        if self._load_future is not None:
            self._record_telemetry(
                message="Domain load requested while another load is in progress",
                level="info",
                action="domain_load_already_running",
            )
            QMessageBox.information(self, "Load In Progress", "Please wait for the current domain load to finish.")
            return

        self.update_status("Loading domain...")
        self._load_cancel_event = threading.Event()
        self._load_progress_queue = Queue()
        self._load_progress = QProgressDialog("Loading domain...", "Cancel", 0, 100, self)
        self._load_progress.setWindowModality(Qt.WindowModal)
        self._load_progress.setMinimumDuration(0)
        self._load_progress.setValue(0)
        self._load_progress.canceled.connect(self._load_cancel_event.set)
        self._load_progress.show()

        # Prompt for build resolution to control surface quality
        suffix = Path(file_path).suffix.lower()
        build_resolution = None
        if suffix in {".dmn", ".vtp", ".vtu", ".stl"}:
            dlg = QDialog(self)
            dlg.setWindowTitle("Domain Build Options")
            form = QFormLayout()

            res_spin = QSpinBox()
            res_spin.setRange(5, 200)
            res_spin.setValue(25)
            if suffix == ".dmn":
                res_spin.setToolTip("Resolution used when extracting the domain surface from the implicit field.\n"
                                    "Higher values yield smoother boundaries at increased compute cost.")
            else:
                res_spin.setToolTip("Resolution used for domain tetrahedralization and boundary extraction.\n"
                                    "Higher values yield finer meshes at increased compute cost.")
            form.addRow("Build resolution:", res_spin)

            # Add info label for mesh files
            if suffix != ".dmn":
                from PySide6.QtWidgets import QLabel
                info_label = QLabel(
                    f"<i>Loading {suffix} file will run:<br>"
                    f"• create() - Initialize implicit function<br>"
                    f"• solve() - Compute evaluation structures<br>"
                    f"• build() - Tetrahedralize and extract boundary</i>"
                )
                info_label.setWordWrap(True)
                form.addRow(info_label)

            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.accepted.connect(dlg.accept)
            buttons.rejected.connect(dlg.reject)

            layout = QVBoxLayout()
            layout.addLayout(form)
            layout.addWidget(buttons)
            dlg.setLayout(layout)

            if dlg.exec() != QDialog.Accepted:
                # User canceled; abort domain load
                self._load_progress.close()
                self._load_progress = None
                self._load_cancel_event = None
                self._load_progress_queue = None
                self.update_status("Domain load canceled")
                return
            build_resolution = res_spin.value()

        future = self._executor.submit(self._load_domain_file, file_path, self._load_cancel_event, self._load_progress_queue, build_resolution)
        self._load_future = future
        self._poll_load_future()

    def _poll_load_future(self):
        if self._load_future is None:
            return
        if self._load_progress_queue is not None and self._load_progress is not None:
            while not self._load_progress_queue.empty():
                try:
                    item = self._load_progress_queue.get_nowait()
                except Exception:
                    break
                try:
                    # Handle both plain values and (value, label) tuples
                    if isinstance(item, tuple):
                        value, label = item
                        self._load_progress.setValue(int(value))
                        if label:
                            self._load_progress.setLabelText(label)
                    else:
                        self._load_progress.setValue(int(item))
                except Exception:
                    pass
        if self._load_future.done():
            self._finish_load_future()
        else:
            QTimer.singleShot(100, self._poll_load_future)

    def _finish_load_future(self):
        # Capture cancel state *before* closing the dialog. Some Qt styles
        # emit `canceled` when the dialog is programmatically closed, which
        # would otherwise make a successful load look canceled.
        was_canceled = bool(self._load_cancel_event and self._load_cancel_event.is_set())

        if self._load_progress:
            self._load_progress.close()
            self._load_progress = None
            self._load_progress_queue = None

        future = self._load_future
        self._load_future = None

        try:
            result = future.result()
        except Exception as exc:
            result = exc

        if isinstance(result, Exception):
            self.update_status("Failed to load domain")
            self._record_telemetry(result, action="load_domain_async")
            QMessageBox.critical(
                self,
                "Error Loading Domain",
                f"Failed to load domain:\n\n{result}"
            )
            self._load_cancel_event = None
            return

        if result is None or was_canceled:
            self.update_status("Domain load canceled")
            self._load_cancel_event = None
            return

        self.load_domain(result)
        self._load_cancel_event = None

        # Warn user if domain build failed (mesh not available)
        if getattr(result, '_build_failed', False):
            error_msg = getattr(result, '_build_error', 'Unknown error')
            self._record_telemetry(
                message=f"Domain build failed: {error_msg}",
                level="warning",
                action="domain_build_failed_warning",
            )
            QMessageBox.warning(
                self,
                "Domain Build Warning",
                f"The domain was loaded but mesh building failed:\n\n{error_msg}\n\n"
                f"Some features (like auto-selecting boundary start points) may not work.\n"
                f"You can still use the domain by manually selecting start points."
            )


# Backwards compatibility alias
VascularizeCADGUI = VascularizeGUI
