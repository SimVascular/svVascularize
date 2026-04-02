import numpy
from textwrap import fill
from typing import Optional

from svv.tree.data.units import UnitSystem, UnitError


def _format_indices(index):
    if isinstance(index, slice):
        start = '' if index.start is None else index.start
        stop = '' if index.stop is None else index.stop
        step = '' if index.step in (None, 1) else f":{index.step}"
        return f"{start}:{stop}{step}"
    if isinstance(index, tuple):
        return ', '.join(str(i) for i in index)
    return str(index)


class TrackedIndexList(list):
    """List that notifies a callback whenever its contents change."""

    def __init__(self, values=(), *, on_change=None):
        super().__init__(values)
        self._on_change = on_change

    def bind(self, on_change):
        self._on_change = on_change

    def _mark_changed(self):
        if self._on_change is not None:
            self._on_change()

    def append(self, value):
        super().append(value)
        self._mark_changed()

    def extend(self, values):
        super().extend(values)
        self._mark_changed()

    def insert(self, index, value):
        super().insert(index, value)
        self._mark_changed()

    def pop(self, index=-1):
        value = super().pop(index)
        self._mark_changed()
        return value

    def remove(self, value):
        super().remove(value)
        self._mark_changed()

    def clear(self):
        super().clear()
        self._mark_changed()

    def reverse(self):
        super().reverse()
        self._mark_changed()

    def sort(self, *args, **kwargs):
        super().sort(*args, **kwargs)
        self._mark_changed()

    def __delitem__(self, index):
        super().__delitem__(index)
        self._mark_changed()

    def __setitem__(self, index, value):
        super().__setitem__(index, value)
        self._mark_changed()

    def __iadd__(self, values):
        result = super().__iadd__(values)
        self._mark_changed()
        return result

    def __imul__(self, value):
        result = super().__imul__(value)
        self._mark_changed()
        return result


class TreeMapEntry(dict):
    """Tracked adjacency entry for a single vessel."""

    def __init__(self, key, value=None, *, on_downstream_change=None):
        super().__init__()
        self._key = int(key)
        self._on_downstream_change = on_downstream_change
        payload = {} if value is None else dict(value)
        self["upstream"] = payload.get("upstream", [])
        self["downstream"] = payload.get("downstream", [])

    def bind(self, key, on_downstream_change):
        self._key = int(key)
        self._on_downstream_change = on_downstream_change
        for field in ("upstream", "downstream"):
            if field in self:
                self[field] = self[field]

    def _notify_downstream_change(self):
        if self._on_downstream_change is not None:
            self._on_downstream_change(self._key)

    def _wrap_list(self, field, value):
        callback = self._notify_downstream_change if field == "downstream" else None
        if isinstance(value, TrackedIndexList):
            value.bind(callback)
            return value
        return TrackedIndexList(value, on_change=callback)

    def __setitem__(self, field, value):
        if field in ("upstream", "downstream"):
            value = self._wrap_list(field, value)
        super().__setitem__(field, value)
        if field == "downstream":
            self._notify_downstream_change()

    def __delitem__(self, field):
        super().__delitem__(field)
        if field == "downstream":
            self._notify_downstream_change()

    def update(self, *args, **kwargs):
        payload = dict(*args, **kwargs)
        for key, value in payload.items():
            self[key] = value

    def pop(self, key, *args):
        value = super().pop(key, *args)
        if key == "downstream":
            self._notify_downstream_change()
        return value

    def clear(self):
        had_downstream = "downstream" in self
        super().clear()
        if had_downstream:
            self._notify_downstream_change()


class TreeMap(dict):
    """Adjacency map used by :class:`~svv.tree.tree.Tree`.

    Keys are integer vessel indices.  Each value is a tracked adjacency entry
    with two well-known fields:

    - ``"upstream"``: list of immediate parent vessel indices (empty for the
      root segment).
    - ``"downstream"``: list of child vessel indices created during
      bifurcation.

    Downstream mutations can notify an owning :class:`~svv.tree.tree.Tree` so
    cached NumPy adjacency views stay coherent with the authoritative Python
    lists.
    """

    def __init__(self, initial=None, *, on_downstream_change=None):
        super().__init__()
        self._on_downstream_change = on_downstream_change
        if initial is not None:
            self.update(initial)

    def bind(self, on_downstream_change):
        self._on_downstream_change = on_downstream_change
        for key, value in super().items():
            value.bind(key, self._on_downstream_change)

    def __setitem__(self, key, value):
        key = int(key)
        if isinstance(value, TreeMapEntry):
            value.bind(key, self._on_downstream_change)
            wrapped = value
        else:
            wrapped = TreeMapEntry(
                key,
                value,
                on_downstream_change=self._on_downstream_change,
            )
        super().__setitem__(key, wrapped)
        if self._on_downstream_change is not None:
            self._on_downstream_change(key)

    def __delitem__(self, key):
        key = int(key)
        super().__delitem__(key)
        if self._on_downstream_change is not None:
            self._on_downstream_change(key)

    def update(self, *args, **kwargs):
        payload = dict(*args, **kwargs)
        for key, value in payload.items():
            self[key] = value

    def setdefault(self, key, default=None):
        key = int(key)
        if key not in self:
            self[key] = {} if default is None else default
        return self[key]

    def pop(self, key, *args):
        key = int(key)
        value = super().pop(key, *args)
        if self._on_downstream_change is not None:
            self._on_downstream_change(key)
        return value

    def clear(self):
        super().clear()
        if self._on_downstream_change is not None:
            self._on_downstream_change(None)


def serialize_tree_map(vessel_map):
    """Return a plain dict-of-lists representation of a vessel map."""

    plain = {}
    for key, value in vessel_map.items():
        plain[int(key)] = {
            "upstream": list(value.get("upstream", [])),
            "downstream": list(value.get("downstream", [])),
        }
    return plain



class TreeParameters(object):
    """Physical and algorithmic settings that steer tree generation.

    Parameters are stored in the unit system referenced by
    :attr:`unit_system`.  By default the centimetre–gram–second convention is
    used; callers can supply an alternate :class:`~svv.tree.data.units.UnitSystem`
    to work in SI or any custom system supported by the unit registry.

    Attributes
    ----------
    unit_system : :class:`~svv.tree.data.units.UnitSystem`
        Active unit system governing the numerical values stored on the class.
    kinematic_viscosity : float
        Blood kinematic viscosity expressed in ``unit_system`` units.
    fluid_density : float
        Mass density expressed in ``unit_system`` units.
    murray_exponent, radius_exponent, length_exponent : float
        Dimensionless growth parameters.
    terminal_pressure, root_pressure : float
        Pressure boundary conditions in ``unit_system`` units.
    terminal_flow, root_flow : float or None
        Volumetric flows in ``unit_system`` units. ``root_flow`` may be ``None``
        and recomputed during growth.
    max_nonconvex_count : int
        Guard limit for attempts to place vessels in non-convex subregions.
    """

    _CGS_SYSTEM = UnitSystem()  # centimetre–gram–second defaults
    _MMHG_SYSTEM = UnitSystem(pressure='mmHg')

    def __init__(self, *, unit_system: Optional[UnitSystem] = None):
        self.unit_system = unit_system or UnitSystem()

        # Dimensional parameters converted into the target unit system
        self.kinematic_viscosity = self.unit_system.convert_value(
            3.6e-2, 'kinematic_viscosity', from_system=self._CGS_SYSTEM
        )
        self.fluid_density = self.unit_system.convert_value(
            1.06, 'density', from_system=self._CGS_SYSTEM
        )
        self.terminal_flow = self.unit_system.convert_value(
            0.125 / 2000.0, 'volumetric_flow', from_system=self._CGS_SYSTEM
        )
        self.root_flow = None
        self.terminal_pressure = self.unit_system.convert_value(
            60.0, 'pressure', from_system=self._MMHG_SYSTEM
        )
        self.root_pressure = self.unit_system.convert_value(
            100.0, 'pressure', from_system=self._MMHG_SYSTEM
        )

        # Dimensionless defaults
        self.murray_exponent = 3.0
        self.radius_exponent = 2.0
        self.length_exponent = 1.0
        self.max_nonconvex_count = 100


    def __str__(self):
        root_flow_line = f"Root Flow: {self.root_flow}"
        if self.root_flow is not None:
            root_flow_line += f" {self.unit_system.volumetric_flow.symbol}"
        return (
            "Tree Parameters:\n"
            "----------------\n"
            f"Kinematic Viscosity: {self.kinematic_viscosity} {self.unit_system.kinematic_viscosity.symbol}\n"
            f"Fluid Density: {self.fluid_density} {self.unit_system.density.symbol}\n"
            f"Murray Exponent: {self.murray_exponent}\n"
            f"Radius Exponent: {self.radius_exponent}\n"
            f"Length Exponent: {self.length_exponent}\n"
            f"Terminal Pressure: {self.terminal_pressure} {self.unit_system.pressure.symbol}\n"
            f"Root Pressure: {self.root_pressure} {self.unit_system.pressure.symbol}\n"
            f"Terminal Flow: {self.terminal_flow} {self.unit_system.volumetric_flow.symbol}\n"
            f"{root_flow_line}"
        )

    def __repr__(self):
        return self.__str__()

    def _coerce_value(self, value, quantity: str, unit=None):
        if unit is None:
            return value
        if isinstance(unit, UnitSystem):
            return self.unit_system.convert_value(value, quantity, from_system=unit)
        if isinstance(unit, tuple):
            symbol, factor = unit
            source = UnitSystem(
                length=self.unit_system.base.length.symbol,
                mass=self.unit_system.base.mass.symbol,
                time=self.unit_system.base.time.symbol,
                **{quantity: (symbol, factor)},
            )
            return self.unit_system.convert_value(value, quantity, from_system=source)
        if isinstance(unit, str):
            source = UnitSystem(
                length=self.unit_system.base.length.symbol,
                mass=self.unit_system.base.mass.symbol,
                time=self.unit_system.base.time.symbol,
                **{quantity: unit},
            )
            return self.unit_system.convert_value(value, quantity, from_system=source)
        raise UnitError(f"Unsupported unit specification '{unit}'.")

    def set(self, parameter, value, *, unit=None):
        """Update a named parameter.

        Parameters
        ----------
        parameter : str
            One of ``{'kinematic_viscosity', 'fluid_density', 'murray_exponent',
            'radius_exponent', 'length_exponent', 'terminal_pressure',
            'root_pressure', 'terminal_flow', 'root_flow', 'max_nonconvex_count'}``.
        value : Any
            New value assigned to the corresponding attribute.  Supply ``unit``
            to convert from alternative units.
        unit : str or tuple or :class:`UnitSystem`, optional
            Source unit definition for dimensional parameters.
        """
        if parameter == 'kinematic_viscosity':
            self.kinematic_viscosity = self._coerce_value(value, 'kinematic_viscosity', unit)
        elif parameter == 'fluid_density':
            self.fluid_density = self._coerce_value(value, 'density', unit)
        elif parameter == 'murray_exponent':
            self.murray_exponent = value
        elif parameter == 'radius_exponent':
            self.radius_exponent = value
        elif parameter == 'length_exponent':
            self.length_exponent = value
        elif parameter == 'terminal_pressure':
            self.terminal_pressure = self._coerce_value(value, 'pressure', unit)
        elif parameter == 'root_pressure':
            self.root_pressure = self._coerce_value(value, 'pressure', unit)
        elif parameter == 'terminal_flow':
            self.terminal_flow = self._coerce_value(value, 'volumetric_flow', unit)
        elif parameter == 'root_flow':
            self.root_flow = self._coerce_value(value, 'volumetric_flow', unit)
        elif parameter == 'max_nonconvex_count':
            self.max_nonconvex_count = value
        else:
            raise ValueError("Invalid parameter: {}.".format(parameter))
        return None

    def set_unit_system(self, unit_system: UnitSystem, *, convert_existing: bool = True):
        """Switch to a new unit system.

        Parameters
        ----------
        unit_system : :class:`UnitSystem`
            Target unit system.
        convert_existing : bool, default True
            Convert stored dimensional values into the new system.  When False,
            values are left untouched and interpreted in the caller-provided
            units.
        """

        if convert_existing:
            mapping = [
                ('kinematic_viscosity', 'kinematic_viscosity'),
                ('fluid_density', 'density'),
                ('terminal_pressure', 'pressure'),
                ('root_pressure', 'pressure'),
                ('terminal_flow', 'volumetric_flow'),
                ('root_flow', 'volumetric_flow'),
            ]
            for attr, quantity in mapping:
                value = getattr(self, attr)
                if value is None:
                    continue
                converted = unit_system.convert_value(
                    value, quantity, from_system=self.unit_system
                )
                setattr(self, attr, converted)
        self.unit_system = unit_system


class TreeData(numpy.ndarray):
    """Vessel attribute table used by the tree growth pipeline.

    The array is laid out per segment with one row per vessel and 31
    floating-point columns.  Use :meth:`TreeData.describe` at runtime to get a
    human-readable summary of the column layout.
    """

    COLUMN_METADATA = (
        {
            'label': 'proximal point',
            'index': slice(0, 3),
            'shape': (3,),
            'units': 'length',
            'description': 'Proximal endpoint coordinates [x, y, z] in domain units.',
            'usage': 'Seed geometry, collision checks, and surface export.'
        },
        {
            'label': 'distal point',
            'index': slice(3, 6),
            'shape': (3,),
            'units': 'length',
            'description': 'Distal endpoint coordinates [x, y, z].',
            'usage': 'Defines vessel extent with the proximal point.'
        },
        {
            'label': 'u_basis',
            'index': slice(6, 9),
            'shape': (3,),
            'units': 'unit vector',
            'description': 'First orthonormal basis vector orthogonal to the vessel axis.',
            'usage': 'Used to orient cross-sectional sampling and bifurcation placement.'
        },
        {
            'label': 'v_basis',
            'index': slice(9, 12),
            'shape': (3,),
            'units': 'unit vector',
            'description': 'Second orthonormal basis vector orthogonal to the vessel axis.',
            'usage': 'Paired with u_basis to span the vessel cross-section.'
        },
        {
            'label': 'w_basis',
            'index': slice(12, 15),
            'shape': (3,),
            'units': 'unit vector',
            'description': 'Unit tangent along the vessel from proximal to distal.',
            'usage': 'Drives growth direction, remeshing, and centerline export.'
        },
        {
            'label': 'children',
            'index': slice(15, 17),
            'shape': (2,),
            'units': 'index',
            'description': 'Integer indices of left and right child vessels (NaN for leaves).',
            'usage': 'Traversal, resistance updates, and serialization of the branching structure.'
        },
        {
            'label': 'parent',
            'index': 17,
            'shape': (),
            'units': 'index',
            'description': 'Index of the upstream parent vessel (NaN for the root).',
            'usage': 'Allows upward traversal during optimization and when exporting connectivity.'
        },
        {
            'label': 'proximal_node',
            'index': 18,
            'shape': (),
            'units': 'node id',
            'description': 'Global node identifier for the proximal endpoint.',
            'usage': 'Shared by adjacent segments so centerline graphs remain watertight.'
        },
        {
            'label': 'distal_node',
            'index': 19,
            'shape': (),
            'units': 'node id',
            'description': 'Global node identifier for the distal endpoint.',
            'usage': 'Supports junction detection and solver mesh connectivity.'
        },
        {
            'label': 'length',
            'index': 20,
            'shape': (),
            'units': 'length',
            'description': 'Segment length cached for resistance and scaling calculations.',
            'usage': 'Computed from endpoints; reused by hydrodynamic updates and export utilities.'
        },
        {
            'label': 'radius',
            'index': 21,
            'shape': (),
            'units': 'length',
            'description': 'Current vessel radius derived from pressure and flow balance.',
            'usage': 'Consumed by surface generation, resistance updates, and simulation writers.'
        },
        {
            'label': 'flow',
            'index': 22,
            'shape': (),
            'units': 'volume/time',
            'description': 'Volumetric flow assigned to the vessel.',
            'usage': 'Aggregated during growth; used by reduced-order models and FSI coupling.'
        },
        {
            'label': 'left_bifurcation',
            'index': 23,
            'shape': (),
            'units': 'dimensionless',
            'description': 'Left daughter radius/flow scaling coefficient at a bifurcation.',
            'usage': 'Propagates Murray’s law ratios through :mod:`svv.tree.utils.c_update`.'
        },
        {
            'label': 'right_bifurcation',
            'index': 24,
            'shape': (),
            'units': 'dimensionless',
            'description': 'Right daughter scaling coefficient complementing left_bifurcation.',
            'usage': 'Ensures branch radii stay consistent during updates and exports.'
        },
        {
            'label': 'reduced_resistance',
            'index': 25,
            'shape': (),
            'units': 'pressure/flow',
            'description': 'Hydraulic resistance from this vessel to its downstream terminals.',
            'usage': 'Updated in C extensions; used by radius updates and ROM exports.'
        },
        {
            'label': 'depth',
            'index': 26,
            'shape': (),
            'units': 'generation',
            'description': 'Integer depth (generation) measured from the root (root = 0).',
            'usage': 'Controls breadth-first updates in :mod:`svv.tree.utils.c_update`.'
        },
        {
            'label': 'reduced_length',
            'index': 27,
            'shape': (),
            'units': 'length',
            'description': 'Weighted downstream path length used for hemodynamic scaling.',
            'usage': 'Maintained with reduced_resistance to avoid repeated recursion.'
        },
        {
            'label': 'radius_scaling',
            'index': 28,
            'shape': (),
            'units': 'dimensionless',
            'description': 'Product of bifurcation ratios from the root to this vessel.',
            'usage': 'Used when reconnecting forests and validating growth heuristics.'
        },
        {
            'label': 'reserved_0',
            'index': 29,
            'shape': (),
            'units': 'n/a',
            'description': 'Reserved padding column kept for binary compatibility.',
            'usage': 'Currently unused; may be repurposed in future releases.'
        },
        {
            'label': 'reserved_1',
            'index': 30,
            'shape': (),
            'units': 'n/a',
            'description': 'Reserved padding column kept for binary compatibility.',
            'usage': 'Currently unused; may be repurposed in future releases.'
        },
    )

    _COLUMN_LOOKUP = {meta['label']: meta for meta in COLUMN_METADATA}

    def __new__(cls, *args, **kwargs):
        if len(args) == 0:
            shape = (1, 31)
            data = super().__new__(cls, shape, **kwargs)
            data[:] = numpy.nan
            return data
        elif isinstance(args[0], numpy.ndarray):
            return args[0].view(cls)
        else:
            data = super().__new__(cls, *args, **kwargs)
            data[:] = numpy.nan
            return data

    @classmethod
    def from_array(cls, arr):
        if isinstance(arr, numpy.ndarray):
            return arr.view(cls)
        else:
            return None

    @classmethod
    def _resolve_metadata(cls, key):
        if key is None:
            return list(cls.COLUMN_METADATA)
        if isinstance(key, str):
            meta = cls._COLUMN_LOOKUP.get(key)
            if meta is None:
                raise KeyError(f"Unknown TreeData label: {key}")
            return [meta]
        if isinstance(key, int):
            for meta in cls.COLUMN_METADATA:
                idx = meta['index']
                if isinstance(idx, int) and idx == key:
                    return [meta]
                if isinstance(idx, slice):
                    start = 0 if idx.start is None else idx.start
                    stop = idx.stop
                    if stop is None:
                        continue
                    step = 1 if idx.step in (None, 0) else idx.step
                    if start <= key < stop and (key - start) % step == 0:
                        return [meta]
            raise KeyError(f"No TreeData column covering index {key}")
        if isinstance(key, slice):
            for meta in cls.COLUMN_METADATA:
                if meta['index'] == key:
                    return [meta]
            raise KeyError(f"Slice {key} does not match a TreeData column definition")
        raise TypeError("label must be a column label, integer index, or slice")

    @classmethod
    def describe(cls, label=None, *, as_dict=False, width=88, return_text=False):
        """Summarise TreeData columns.

        Parameters
        ----------
        label : str or int or slice, optional
            Column label (``'radius'``), integer column index (``21``), or exact
            slice specification.  Omit to list the entire table.
        as_dict : bool, default False
            Return dictionaries suitable for programmatic inspection instead of
            formatted text.  When ``label`` is provided a single dictionary is
            returned, otherwise a list.
        width : int, default 88
            Target line width for wrapped text output.
        return_text : bool, default False
            When True, return the formatted string instead of printing it.
        """

        def normalize(meta):
            return {
                'label': meta['label'],
                'index': meta['index'],
                'index_repr': _format_indices(meta['index']),
                'shape': meta['shape'],
                'units': meta['units'],
                'description': meta['description'],
                'usage': meta['usage'],
            }

        meta_list = cls._resolve_metadata(label)

        if as_dict:
            entries = [normalize(meta) for meta in meta_list]
            return entries[0] if label is not None else entries

        blocks = []
        for meta in meta_list:
            info = normalize(meta)
            block = [
                f"{info['label']} (columns {info['index_repr']}, shape {info['shape']}, units {info['units']})",
                fill(info['description'], width=width, initial_indent='    ', subsequent_indent='    '),
            ]
            if info['usage']:
                usage_text = f"Usage: {info['usage']}"
                block.append(
                    fill(usage_text, width=width, initial_indent='    ', subsequent_indent='        ')
                )
            blocks.append('\n'.join(block))

        text = '\n\n'.join(blocks)
        if return_text:
            return text
        print(text)
        return None

    def __array_finalize__(self, obj):
        if obj is None:
            return None

    def __proximal__(self, *args):
        if len(args) == 0:
            return self[:, 0:3].copy()
        else:
            return self[args[0], 0:3].copy()

    def __distal__(self, *args):
        if len(args) == 0:
            return self[:, 3:6].copy()
        else:
            return self[args[0], 3:6].copy()

    def __u_basis__(self, *args):
        if len(args) == 0:
            return self[:, 6:9].copy()
        else:
            return self[args[0], 6:9].copy()

    def __v_basis__(self, *args):
        if len(args) == 0:
            return self[:, 9:12].copy()
        else:
            return self[args[0], 9:12].copy()

    def __w_basis__(self, *args):
        if len(args) == 0:
            return self[:, 12:15].copy()
        else:
            return self[args[0], 12:15].copy()

    def __children__(self, *args):
        if len(args) == 0:
            return self[:, 15:17].copy()
        else:
            return self[args[0], 15:17].copy()

    def __parent__(self, *args):
        if len(args) == 0:
            return self[:, 17].copy()
        else:
            return self[args[0], 17].copy()

    def __proximal_node__(self, *args):
        if len(args) == 0:
            return self[:, 18].copy()
        else:
            return self[args[0], 18].copy()

    def __distal_node__(self, *args):
        if len(args) == 0:
            return self[:, 19].copy()
        else:
            return self[args[0], 19].copy()

    def __length__(self, *args):
        if len(args) == 0:
            return self[:, 20].copy()
        else:
            return self[args[0], 20].copy()

    def __radius__(self, *args):
        if len(args) == 0:
            return self[:, 21].copy()
        else:
            return self[args[0], 21].copy()

    def __flow__(self, *args):
        if len(args) == 0:
            return self[:, 22].copy()
        else:
            return self[args[0], 22].copy()

    def __left_bifurcation__(self, *args):
        if len(args) == 0:
            return self[:, 23].copy()
        else:
            return self[args[0], 23].copy()

    def __right_bifurcation__(self, *args):
        if len(args) == 0:
            return self[:, 24].copy()
        else:
            return self[args[0], 24].copy()

    def __reduced_resistance__(self, *args):
        if len(args) == 0:
            return self[:, 25].copy()
        else:
            return self[args[0], 25].copy()

    def __depth__(self, *args):
        if len(args) == 0:
            return self[:, 26].copy()
        else:
            return self[args[0], 26].copy()

    def __reduced_length__(self, *args):
        if len(args) == 0:
            return self[:, 27].copy()
        else:
            return self[args[0], 27].copy()

    def __radius_scaling__(self, *args):
        if len(args) == 0:
            return self[:, 28].copy()
        else:
            return self[args[0], 28].copy()

    def get(self, label, *args):
        if label == 'proximal':
            return self.__proximal__(*args)
        elif label == 'distal':
            return self.__distal__(*args)
        elif label == 'u_basis':
            return self.__u_basis__(*args)
        elif label == 'v_basis':
            return self.__v_basis__(*args)
        elif label == 'w_basis':
            return self.__w_basis__(*args)
        elif label == 'children':
            return self.__children__(*args)
        elif label == 'parent':
            return self.__parent__(*args)
        elif label == 'proximal_node':
            return self.__proximal_node__(*args)
        elif label == 'distal_node':
            return self.__distal_node__(*args)
        elif label == 'length':
            return self.__length__(*args)
        elif label == 'radius':
            return self.__radius__(*args)
        elif label == 'flow':
            return self.__flow__(*args)
        elif label == 'left_bifurcation':
            return self.__left_bifurcation__(*args)
        elif label == 'right_bifurcation':
            return self.__right_bifurcation__(*args)
        elif label == 'reduced_resistance':
            return self.__reduced_resistance__(*args)
        elif label == 'depth':
            return self.__depth__(*args)
        elif label == 'reduced_length':
            return self.__reduced_length__(*args)
        elif label == 'radius_scaling':
            return self.__radius_scaling__(*args)
        else:
            raise ValueError("Invalid label.")
