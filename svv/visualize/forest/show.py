import numpy as np
import pyvista

from svv.visualize.batch_cylinders import tree_to_merged_mesh, segments_to_merged_mesh


def show(forest, plot_domain=False, return_plotter=False, **kwargs):
    """
    Visualize a forest, optionally including forest connection networks when present.
    """
    colors = kwargs.get('colors', ['red', 'blue', 'green', 'yellow', 'purple',
                                   'orange', 'cyan', 'magenta', 'white', 'black'])
    plotter = pyvista.Plotter(**kwargs)
    count = 0

    has_connections = getattr(forest, "connections", None) is not None and \
        getattr(forest.connections, "tree_connections", None)

    if has_connections:
        # Draw connected trees and connection vessels
        for net_idx, tree_conn in enumerate(forest.connections.tree_connections):
            for tree in tree_conn.connected_network:
                color = colors[count % len(colors)]
                merged = tree_to_merged_mesh(tree)
                if merged is not None:
                    plotter.add_mesh(merged, color=color)
                count += 1

            # Connection vessels (between trees in this network)
            for tree_idx, vessel_list in enumerate(tree_conn.vessels):
                color = colors[tree_idx % len(colors)]
                # Flatten all segments from all vessels into one array
                all_segs = []
                for vessel in vessel_list:
                    for seg in vessel:
                        all_segs.append(seg)
                if all_segs:
                    merged = segments_to_merged_mesh(np.array(all_segs))
                    if merged is not None:
                        plotter.add_mesh(merged, color=color)
    else:
        # Fall back to original visualization without connections
        for network in forest.networks:
            for tree in network:
                color = colors[count % len(colors)]
                merged = tree_to_merged_mesh(tree)
                if merged is not None:
                    plotter.add_mesh(merged, color=color)
                count += 1
    if plot_domain:
        plotter.add_mesh(forest.domain.boundary, color='grey', opacity=0.25)
    if return_plotter:
        return plotter
    else:
        plotter.show()
