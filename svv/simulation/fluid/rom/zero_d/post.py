make_results = """import pyvista as pv
import numpy as np
import os
from tqdm import tqdm
import csv
from collections import defaultdict


def read_csv_into_nested_dict(file_path):
    # Initialize a nested dictionary structure
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    time_steps = defaultdict(lambda: defaultdict(dict))
    time_counter = defaultdict(int)
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = row['name']
            time = float(row['time'])
            flow_in = float(row['flow_in'])
            flow_out = float(row['flow_out'])
            pressure_in = float(row['pressure_in'])
            pressure_out = float(row['pressure_out'])
            parts = name.split('_')
            branch = int(parts[0].replace('branch', ''))
            segment = int(parts[1].replace('seg', ''))
            # Check if the time has already been encountered for this branch and segment
            if time not in time_steps[branch][segment]:
                time_steps[branch][segment][time] = time_counter[(branch, segment)]
                time_counter[(branch, segment)] += 1
            time_index = time_steps[branch][segment][time]
            data['time'][branch][segment][time_index] = time
            data['flow_in'][branch][segment][time_index] = flow_in
            data['flow_out'][branch][segment][time_index] = flow_out
            data['pressure_in'][branch][segment][time_index] = pressure_in
            data['pressure_out'][branch][segment][time_index] = pressure_out
    return data


geom_data = np.genfromtxt("geom.csv",delimiter=",")
data = read_csv_into_nested_dict("output.csv")

render_pngs = os.environ.get("SVV_0D_RENDER_SCREENSHOTS", "1") == "1"
write_total = os.environ.get("SVV_0D_WRITE_TOTAL_VTP", "1") == "1"
cylinder_resolution = max(3, int(os.environ.get("SVV_0D_CYLINDER_RESOLUTION", "24")))
cylinder_capping = os.environ.get("SVV_0D_CYLINDER_CAPPING", "0") == "1"
max_units = int(os.environ.get("SVV_0D_MAX_VESSELS", "0"))
max_timesteps = int(os.environ.get("SVV_0D_MAX_TIMESTEPS", "0"))
timestep_stride = max(1, int(os.environ.get("SVV_0D_TIMESTEP_STRIDE", "1")))

for folder in ("timeseries", "timeseries_for_pressure_gif", "timeseries_for_flow_gif", "timeseries_for_wss_gif"):
    os.makedirs(folder, exist_ok=True)

all_units = []
for vessel in sorted(data['flow_in'].keys()):
    start = geom_data[vessel, 0:3]
    end = geom_data[vessel, 3:6]
    direction = geom_data[vessel, 3:6] - geom_data[vessel, 0:3]
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0.0:
        continue
    direction = direction / direction_norm
    length = float(geom_data[vessel, 6])
    radius = float(geom_data[vessel, 7])
    center = 0.5 * direction * length + start
    for segment in sorted(data['flow_in'][vessel].keys()):
        base_mesh = pv.Cylinder(
            center=center,
            direction=direction,
            height=length,
            radius=radius,
            resolution=cylinder_resolution,
            capping=cylinder_capping,
        )
        base_mesh = base_mesh.elevation(low_point=end, high_point=start, scalar_range=[0.0, 1.0])
        axis_unit = np.asarray(base_mesh.point_data['Elevation'])
        all_units.append({
            'vessel': vessel,
            'segment': segment,
            'radius': radius,
            'base_mesh': base_mesh,
            'axis_unit': axis_unit,
        })
        if max_units > 0 and len(all_units) >= max_units:
            break
    if max_units > 0 and len(all_units) >= max_units:
        break

if len(all_units) == 0:
    raise RuntimeError("No vessel geometry could be generated from geom.csv.")

first_vessel = next(iter(data['time']))
first_segment = next(iter(data['time'][first_vessel]))
num_timesteps = len(data['time'][first_vessel][first_segment])
if max_timesteps > 0:
    num_timesteps = min(num_timesteps, max_timesteps)
timesteps = list(range(0, num_timesteps, timestep_stride))

print("Found {} vessel segments across {} timesteps".format(len(all_units), len(timesteps)))
print(
    "Options: render_pngs={}, write_total={}, cylinder_resolution={}, cylinder_capping={}, max_vessels={}, max_timesteps={}, timestep_stride={}".format(
        render_pngs,
        write_total,
        cylinder_resolution,
        cylinder_capping,
        max_units,
        max_timesteps,
        timestep_stride,
    )
)

min_pressure = np.inf
max_pressure = -np.inf
min_flow = np.inf
max_flow = -np.inf
min_wss = np.inf
max_wss = -np.inf
for idx in timesteps:
    for unit in all_units:
        vessel = unit['vessel']
        segment = unit['segment']
        radius = unit['radius']
        p_in = float(data['pressure_in'][vessel][segment][idx]) / 1333.33
        p_out = float(data['pressure_out'][vessel][segment][idx]) / 1333.33
        q_in = float(data['flow_in'][vessel][segment][idx])
        vel = (q_in / (np.pi * radius**2)) / 2.0
        re = (1.06 * 2.0 * radius * vel) / 0.04
        fd = 0.0 if re == 0.0 else 64.0 / re
        wss = vel * fd * 1.06
        min_pressure = min(min_pressure, p_in, p_out)
        max_pressure = max(max_pressure, p_in, p_out)
        min_flow = min(min_flow, q_in)
        max_flow = max(max_flow, q_in)
        min_wss = min(min_wss, wss)
        max_wss = max(max_wss, wss)

total = None
for out_idx, idx in enumerate(tqdm(timesteps, desc="Building/Saving Timeseries", position=0)):
    time_value = float(data['time'][first_vessel][first_segment][idx])
    meshes = []
    for unit in all_units:
        vessel = unit['vessel']
        segment = unit['segment']
        radius = unit['radius']
        axis_unit = unit['axis_unit']
        mesh = unit['base_mesh'].copy(deep=True)
        p_in = float(data['pressure_in'][vessel][segment][idx]) / 1333.33
        p_out = float(data['pressure_out'][vessel][segment][idx]) / 1333.33
        q_in = float(data['flow_in'][vessel][segment][idx])
        mesh.point_data['Pressure [mmHg]'] = p_out + axis_unit * (p_in - p_out)
        mesh.cell_data['Flow [mL/s]'] = q_in
        vel = (q_in / (np.pi * radius**2)) / 2.0
        re = (1.06 * 2.0 * radius * vel) / 0.04
        fd = 0.0 if re == 0.0 else 64.0 / re
        mesh.cell_data['WSS [dyne/cm^2]'] = vel * fd * 1.06
        meshes.append(mesh)

    time_mesh = meshes[0].merge(meshes[1:]) if len(meshes) > 1 else meshes[0]
    time_mesh.field_data['time'] = np.array([time_value], dtype=float)
    time_mesh.save(os.path.join(os.getcwd(), "timeseries", "time_point_{}.vtp".format(out_idx)))

    if render_pngs:
        p = pv.Plotter(off_screen=True)
        p.add_mesh(time_mesh, scalars='Pressure [mmHg]', clim=[round(min_pressure, 4), round(max_pressure, 4)], cmap="coolwarm")
        p.show(auto_close=True, screenshot=os.path.join(os.getcwd(), "timeseries_for_pressure_gif", "time_point_{}.png".format(out_idx)))
        p = pv.Plotter(off_screen=True)
        p.add_mesh(time_mesh, scalars='Flow [mL/s]', clim=[round(min_flow, 4), round(max_flow, 4)], cmap="GnBu")
        p.show(auto_close=True, screenshot=os.path.join(os.getcwd(), "timeseries_for_flow_gif", "time_point_{}.png".format(out_idx)))
        p = pv.Plotter(off_screen=True)
        p.add_mesh(time_mesh, scalars='WSS [dyne/cm^2]', clim=[round(min_wss, 2), round(max_wss, 2)], cmap="coolwarm")
        p.show(auto_close=True, screenshot=os.path.join(os.getcwd(), "timeseries_for_wss_gif", "time_point_{}.png".format(out_idx)))

    if write_total:
        total = time_mesh if total is None else total.merge(time_mesh)

if write_total and total is not None:
    total.save(os.path.join(os.getcwd(), "timeseries", "total.vtp"))

"""

view_plots="""import numpy as np
import matplotlib.pyplot as plt
data = np.load("solver_0d_branch_results.npy",allow_pickle=True).item()
vessels = []
fig_flow = plt.figure()
ax_flow = fig_flow.add_subplot()
fig_pressure = plt.figure()
ax_pressure = fig_pressure.add_subplot()
fig_flow_outlets = plt.figure()
ax_flow_outlets = fig_flow_outlets.add_subplot()
fig_pressure_outlets = plt.figure()
ax_pressure_outlets = fig_pressure_outlets.add_subplot()
for vessel in data["flow"]:
    vessels.append(vessel)
for vessel in vessels:
    ax_flow.plot(data["time"],data["flow"][vessel][0],label="vessel_"+str(vessel))
    ax_pressure.plot(data["time"],data["pressure"][vessel][0]/1333.22,label="vessel_"+str(vessel))
import json
info = json.load(open("solver_0d.in"))
all_inlets = []
all_outlets = []
for i in info["junctions"]:
    for j in i["inlet_vessels"]:
        all_inlets.append(j)
    for j in i["outlet_vessels"]:
        all_outlets.append(j)
all_inlets = set(all_inlets)
all_outlets = set(all_outlets)
true_outlets = list(all_outlets.difference(all_inlets))
true_inlets = list(all_inlets.difference(all_outlets))
for vessel in true_outlets:
    ax_flow_outlets.plot(data["time"],data["flow"][vessel][-1],label="vessel_"+str(vessel))
    ax_pressure_outlets.plot(data["time"],data["pressure"][vessel][-1]/1333.22,label="vessel_"+str(vessel))
ax_flow.set_xlabel("Time (sec)")
ax_flow.set_ylabel("Flow (mL/s)")
ax_pressure.set_xlabel("Time (sec)")
ax_pressure.set_ylabel("Pressure (mmHg)")
ax_flow_outlets.set_xlabel("Time (sec)")
ax_pressure_outlets.set_xlabel("Time (sec)")
ax_flow_outlets.set_ylabel("Flow (mL/s)")
ax_flow_outlets.set_title("Outlets Only")
ax_pressure_outlets.set_ylabel("Pressure (mmHg)")
ax_pressure_outlets.set_title("Outlets Only")
#ax_pressure_outlets.set_ylim([-10*np.finfo(float).eps,10*np.finfo(float).eps])
plt.show()

"""

post_data = """import pyvista as pv
import numpy as np
import os
from tqdm import tqdm
import csv
from collections import defaultdict, OrderedDict


def read_csv_into_nested_dict(file_path):
    # Initialize a nested dictionary structure
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    time_steps = defaultdict(lambda: defaultdict(dict))
    time_counter = defaultdict(int)
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = row['name']
            time = float(row['time'])
            flow_in = float(row['flow_in'])
            flow_out = float(row['flow_out'])
            pressure_in = float(row['pressure_in'])
            pressure_out = float(row['pressure_out'])
            parts = name.split('_')
            branch = int(parts[0].replace('branch', ''))
            segment = int(parts[1].replace('seg', ''))
            # Check if the time has already been encountered for this branch and segment
            if time not in time_steps[branch][segment]:
                time_steps[branch][segment][time] = time_counter[(branch, segment)]
                time_counter[(branch, segment)] += 1
            time_index = time_steps[branch][segment][time]
            data['time'][branch][segment][time_index] = time
            data['flow_in'][branch][segment][time_index] = flow_in
            data['flow_out'][branch][segment][time_index] = flow_out
            data['pressure_in'][branch][segment][time_index] = pressure_in
            data['pressure_out'][branch][segment][time_index] = pressure_out
    return data


geom_data = np.genfromtxt("geom.csv",delimiter=",")
data = read_csv_into_nested_dict("output.csv")

# Storing 5 data results: time, pressure_in, pressure_out, flow_in, flow_out
results = np.zeros((geom_data.shape[0], geom_data.shape[1] + 5, len(data['time'][0][0])))
for idx in tqdm(range(len(data['time'][0][0])),desc="Building Timeseries ",position=0):
    time = data['time'][0][0][idx]
    # Assign data from the csv file to the vessels in the results array for a given time point
    for jdx in range(results.shape[0]):
        results[jdx, 0:3, idx] = geom_data[jdx, 0:3]
        results[jdx, 3:6, idx] = geom_data[jdx, 3:6]
        results[jdx, 6, idx] = geom_data[jdx, 6]
        results[jdx, 7, idx] = geom_data[jdx, 7]
        results[jdx, 8, idx] = data['time'][jdx][0][idx]
        results[jdx, 9, idx] = data['pressure_in'][jdx][0][idx]
        results[jdx, 10, idx] = data['pressure_out'][jdx][0][idx]
        results[jdx, 11, idx] = data['flow_in'][jdx][0][idx]
        results[jdx, 12, idx] = data['flow_out'][jdx][0][idx]

# Save the results array to a numpy file
np.save("results.npy", results)
"""

collate_timeseries_to_pvd = """import os
import glob
import re
import xml.etree.ElementTree as ET

import pyvista as pv


def _extract_time_from_mesh(path):
    \"""
    Return the time value stored in the mesh field data if available.
    \"""
    try:
        mesh = pv.read(path)
    except Exception:
        return None
    try:
        if "time" in mesh.field_data:
            values = mesh.field_data["time"]
            # Handle scalar or 1-element arrays
            if hasattr(values, "shape") and values.size > 0:
                return float(values.flat[0])
            return float(values)
    except Exception:
        return None
    return None


def _extract_index_from_name(path):
    \"""
    Extract integer index from a file name like ``time_point_12.vtp``.
    \"""
    name = os.path.basename(path)
    match = re.search(r"time_point_(\\d+)\\.vtp$", name)
    if not match:
        return 0
    return int(match.group(1))


def main():
    base_dir = os.getcwd()
    timeseries_dir = os.path.join(base_dir, "timeseries")

    if not os.path.isdir(timeseries_dir):
        raise FileNotFoundError(
            "No 'timeseries' directory found. Run 'plot_0d_results_to_3d.py' "
            "to generate time_point_*.vtp files first."
        )

    pattern = os.path.join(timeseries_dir, "time_point_*.vtp")
    files = sorted(glob.glob(pattern), key=_extract_index_from_name)

    if not files:
        raise FileNotFoundError(
            "No 'time_point_*.vtp' files found in the 'timeseries' directory. "
            "Run 'plot_0d_results_to_3d.py' first."
        )

    # Build a VTK collection file (.pvd) that references each time point
    root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    collection = ET.SubElement(root, "Collection")

    for path in files:
        time_value = _extract_time_from_mesh(path)
        if time_value is None:
            # Fall back to using the index when no explicit time is stored
            time_value = _extract_index_from_name(path)
        ET.SubElement(
            collection,
            "DataSet",
            timestep=str(time_value),
            part="0",
            file=os.path.basename(path),
        )

    tree = ET.ElementTree(root)
    out_path = os.path.join(timeseries_dir, "timeseries.pvd")
    tree.write(out_path, xml_declaration=True, encoding="UTF-8")
    print(f"Created '{out_path}' with {len(files)} time steps.")


if __name__ == "__main__":
    main()
"""
