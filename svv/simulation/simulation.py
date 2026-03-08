import os
import uuid
from xml.dom import minidom
from copy import deepcopy

import pyvista
import tetgen
import svv
import tqdm
import numpy
import pymeshfix
# Avoid importing the top-level vtk package here to prevent optional IO
# backends (e.g., NetCDF) from being loaded at import time. PyVista will
# import the specific VTK modules it needs internally.

import svv.tree.tree
from svv.simulation.utils.extract_faces import extract_faces
from svv.domain.routines.boolean import boolean
from svv.utils.remeshing.remesh import remesh_surface, remesh_volume
from svv.simulation.general_parameters import GeneralSimulationParameters
from svv.simulation.mesh import GeneralMesh
from svv.simulation.fluid.fluid_equation import FluidEquation
from svv.simulation.utils.boundary_layer import BoundaryLayer
from svv.domain.routines.tetrahedralize import tetrahedralize

# Defer 1D/0D ROM imports to their respective methods to avoid importing
# vtk-heavy modules during Simulation class import.
from svv.simulation.fluid.rom.zero_d.zerod_tree import export_0d_simulation as export_tree_0d_simulation
from svv.simulation.fluid.rom.zero_d.zerod_forest import export_0d_simulation as export_forest_0d_simulation

class Simulation(object):
    def __init__(self, synthetic_object, name=None, directory=None):
        """
        The Simulation class defines a simulation object that
        is used to generate the files to run a physics simulation
        using a synthetic vascular network.
        """
        self.synthetic_object = synthetic_object
        if name is None:
            name = "simulations_" + uuid.uuid4().hex
        if directory is None:
            directory = os.getcwd()
        self.file_path = os.path.join(directory, name)
        self.tissue_domain_faces = []
        self.fluid_domain_faces = []
        self.tissue_domain_surface_meshes = []
        self.fluid_domain_surface_meshes = []
        self.tissue_domain_volume_meshes = []
        self.fluid_domain_volume_meshes = []
        self.tissue_domain_meshes = []
        self.fluid_domain_meshes = []
        self.fluid_domain_boundary_layers = []
        self.fluid_domain_interiors = []
        self.fluid_domain_wall_layers = []
        if isinstance(self.synthetic_object, svv.tree.tree.Tree):
            self.fluid_3d_simulations = [None]
            self.fluid_1d_simulations = [None]
            self.fluid_0d_simulations = [None]
            self.tissue_simulations = [None]
        elif isinstance(self.synthetic_object, svv.forest.forest.Forest):
            self.fluid_3d_simulations = [[None]*len(network) for network in self.synthetic_object.networks]
            self.fluid_1d_simulations = [[None]*len(network) for network in self.synthetic_object.networks]
            self.fluid_0d_simulations = [[None] * len(network) for network in self.synthetic_object.networks]
            self.tissue_simulations = [None]

    def build_meshes(self, fluid=True, tissue=False, hausd=0.0001, hsize=None, minratio=1.1, mindihedral=10.0,
                     order=1, remesh_vol=False, boundary_layer=True, layer_thickness_ratio=0.25,
                     layer_thickness_ratio_adjustment=0.5, boundary_layer_attempts=5, wall_layers=False,
                     wall_thickness=None, upper_num_triangles=1000, lower_num_triangles=100):
        """
        Build the mesh objects for 3D simulations.
        :return:
        [NOTE] Boolean operations and remeshing with of the interface for
        the fluid and tissue domains may need to be redone to ensure mesh
        conformation at the interface.
        """
        self.tissue_domain_surface_meshes = []
        self.fluid_domain_surface_meshes = []
        self.tissue_domain_volume_meshes = []
        self.fluid_domain_volume_meshes = []
        self.fluid_domain_boundary_layers = []
        self.fluid_domain_interiors = []
        self.fluid_domain_wall_layers = []
        if isinstance(self.synthetic_object, svv.tree.tree.Tree):
            if fluid:
                if tissue:
                    extension_scale = 4.0
                    for i in range(5):
                        new_root = self.synthetic_object.data[0, 0:3] + extension_scale * self.synthetic_object.data[0, 21]*self.synthetic_object.data.get('w_basis', 0)
                        if self.synthetic_object.domain(new_root.reshape(1, 3)) > 0:
                            break
                        else:
                            extension_scale += 1.0
                    root_extension = self.synthetic_object.data[0, 21] * extension_scale
                    self.synthetic_object.data[0, 0:3] += root_extension * self.synthetic_object.data.get('w_basis', 0)
                print("Unioning water-tight model")
                fluid_surface_mesh = self.synthetic_object.export_solid(watertight=True)
                print("Finished Unioning water-tight model")
                print("Tetrahedralizing")
                #tet_fluid = tetgen.TetGen(fluid_surface_mesh)
                try:
                    #tet_fluid.make_manifold(verbose=True)
                    #tet_fluid.tetrahedralize(minratio=minratio, mindihedral=10.0, steinerleft=-1, order=order, nobisect=True, verbose=2, switches='M')
                    #tet_fluid.tetrahedralize(switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                    grid, node, elems = tetrahedralize(fluid_surface_mesh, switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                    fluid_volume_mesh = grid
                except:
                    fix = pymeshfix.MeshFix(fluid_surface_mesh)
                    fix.repair()
                    #tet_fluid.make_manifold(verbose=True)
                    #tet_fluid.tetrahedralize(minratio=minratio, mindihedral=10.0, steinerleft=-1, order=order, nobisect=True, verbose=2, switches='M')
                    grid, node, elems = tetrahedralize(fix.mesh, switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                    fluid_volume_mesh = grid
                if isinstance(fluid_volume_mesh, type(None)):
                    print("Failed to generate fluid volume mesh.")
                else:
                    hsize = fluid_surface_mesh.cell_data["hsize"][0]
                    fluid_surface_mesh = fluid_volume_mesh.extract_surface()
                    # faces, wall_surfaces, cap_surfaces, lumen_surfaces, _
                    # fluid_surface_faces = extract_faces(fluid_surface_mesh, fluid_volume_mesh)
                    if boundary_layer:
                        # Prefer lumen (vessel wall) surfaces; fallback to walls if needed
                        fluid_surface_faces = extract_faces(fluid_surface_mesh, fluid_volume_mesh)
                        lumens = fluid_surface_faces[3]
                        walls = fluid_surface_faces[1]
                        wall = lumens[0] if len(lumens) > 0 else (walls[0] if len(walls) > 0 else None)
                        if wall is None:
                            print("No suitable wall surface found for boundary layer generation.")
                            self.fluid_domain_boundary_layers.append(None)
                            self.fluid_domain_interiors.append(None)
                        else:
                            for i in range(boundary_layer_attempts):
                                try:
                                    fluid_boundary_layers = BoundaryLayer(wall, layer_thickness=layer_thickness_ratio*hsize,
                                                                          remesh_vol=remesh_vol)
                                    fluid_volume_mesh, fluid_interior, fluid_boundary = fluid_boundary_layers.generate()
                                    success = True
                                    print("Generated boundary layers on attempt {}/{}.".format(i+1, boundary_layer_attempts))
                                except:
                                    print("Failed to generate boundary layers {}/{}.\n".format(i+1, boundary_layer_attempts))
                                    fluid_boundary = None
                                    fluid_interior = None
                                    success = False
                                    layer_thickness_ratio *= layer_thickness_ratio_adjustment
                                if success:
                                    break
                            self.fluid_domain_boundary_layers.append(fluid_boundary)
                            self.fluid_domain_interiors.append(fluid_interior)
                    else:
                        if remesh_vol:
                            fluid_volume_mesh = remesh_volume(fluid_volume_mesh, hsiz=fluid_surface_mesh.hsize)
                        self.fluid_domain_boundary_layers.append(None)
                        self.fluid_domain_interiors.append(None)
                    if wall_layers:
                        if isinstance(wall_thickness, type(None)):
                            wall_thickness = 2*layer_thickness_ratio*hsize
                        lumens = fluid_surface_faces[3]
                        walls = fluid_surface_faces[1]
                        wall = lumens[0] if len(lumens) > 0 else (walls[0] if len(walls) > 0 else None)
                        if wall is None:
                            print("No suitable wall surface found for wall layer generation.")
                        else:
                            fluid_boundary_layers = BoundaryLayer(wall, negate_warp_vectors=False,
                                                                  layer_thickness=wall_thickness,
                                                                  remesh_vol=False, combine=False)
                            _, _, fluid_wall = fluid_boundary_layers.generate()
                            # Perform tetrahedron re-orientation to ensure positive Jacobian
                            fluid_wall = remesh_volume(fluid_wall, nomove=True, noinsert=True, nosurf=True, verbosity=4)
                            if remesh_vol:
                                fluid_wall = remesh_volume(fluid_wall, hausd=hausd, nosurf=True, verbosity=4)
                            self.fluid_domain_wall_layers.append(fluid_wall)
                    fluid_surface_mesh = fluid_volume_mesh.extract_surface()
                    fluid_surface_mesh.cell_data["hsize"] = hsize
                    fluid_surface_mesh.cell_data["hsize"][0] = hsize
                    self.fluid_domain_surface_meshes.append(fluid_surface_mesh)
                    self.fluid_domain_volume_meshes.append(fluid_volume_mesh)
                    if tissue:
                        self.synthetic_object.data[0, 0:3] += root_extension * self.synthetic_object.data.get('w_basis',0)
            if tissue and not isinstance(self.synthetic_object.domain, type(None)):
                # Extrude the root of the tree to ensure proper intersection with the tissue domain.
                if not fluid:
                    root_extension = max(self.synthetic_object.data[0, 21] * 4, self.synthetic_object.data[0, 20] * 0.5)
                    self.synthetic_object.data[0, 0:3] += root_extension * self.synthetic_object.data.get('w_basis', 0)
                    # Should check to see that the extended point does not intersect with another fluid or tissue domain.
                    fluid_surface_boolean_mesh = self.synthetic_object.export_solid(watertight=True)
                else:
                    if not wall_layers:
                        fluid_surface_boolean_mesh = deepcopy(self.fluid_domain_surface_meshes[-1])
                    else:
                        fluid_surface_boolean_mesh = deepcopy(self.fluid_domain_wall_layers[-1])
                hsize = fluid_surface_boolean_mesh.cell_data["hsize"][0]
                try:
                    tissue_domain = remesh_surface(self.synthetic_object.domain.boundary, hausd=hausd, verbosity=0) # Check if this should be remeshed
                except:
                    print("REMESHING FAILS: CHECKING FOR TRIANGLE INTERSECTIONS")
                    tmp_boundary = pymeshfix.MeshFix(self.synthetic_object.domain.boundary)
                area = tissue_domain.area
                tissue_domain = boolean(tissue_domain, fluid_surface_boolean_mesh, operation='difference')
                if fluid:
                    fluid_faces = extract_faces(tissue_domain, None)
                    face_sizes = [len(face) for face in fluid_faces[0]]
                    wall = numpy.argmax(face_sizes)
                    low_tri_area = area / upper_num_triangles
                    hmin = ((4.0*low_tri_area)/3.0**0.5) ** (0.5)
                    upper_tri_area = area / lower_num_triangles
                    hmax = ((4.0*upper_tri_area)/3.0**0.5) ** (0.5)
                    tissue_domain = remesh_surface(tissue_domain, hausd=hausd, verbosity=0)
                else:
                    tissue_domain = remesh_surface(tissue_domain, hausd=hausd, verbosity=0)
                #tet_tissue = tetgen.TetGen(tissue_domain)
                if not fluid:
                    self.synthetic_object.data[0, 0:3] += root_extension * self.synthetic_object.data.get('w_basis', 0)
                try:
                    grid, nodes, elems = tetrahedralize(switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                    #tet_tissue.tetrahedralize(minratio=minratio, order=order)
                    tissue_volume_mesh = grid
                except:
                    if fluid:
                        print('Mesh interface may be corrupted after mesh fixing for tetrahedralization.')
                    fix = pymeshfix.MeshFix(tissue_domain)
                    fix.repair()
                    #tet_tissue.make_manifold(verbose=True)
                    #tet_tissue.tetrahedralize(minratio=minratio, order=order)
                    grid, nodes, elems = tetrahedralize(fix.mesh, switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                    tissue_volume_mesh = grid
                if isinstance(tissue_volume_mesh, type(None)):
                    print("Failed to generate tissue volume mesh.")
                else:
                    if remesh_vol:
                        tissue_volume_mesh = remesh_volume(tissue_volume_mesh, hausd=hausd, nosurf=True)
                    tissue_domain = tissue_volume_mesh.extract_surface()
                    self.tissue_domain_surface_meshes.append(tissue_domain)
                    self.tissue_domain_volume_meshes.append(tissue_volume_mesh)
        elif isinstance(self.synthetic_object, svv.forest.forest.Forest) and isinstance(self.synthetic_object.connections, type(None)):
            for network in self.synthetic_object.networks:
                network_fluid_surface_meshes = []
                network_fluid_volume_meshes = []
                network_tissue_surface_meshes = []
                network_tissue_volume_meshes = []
                for tree in network:
                    if fluid:
                        fluid_surface_mesh = tree.export_solid(watertight=True)
                        #tet_fluid = tetgen.TetGen(fluid_surface_mesh)
                        try:
                            # tet_fluid.make_manifold(verbose=False)
                            # Tetrahedralize the fluid domain (correct target object)
                            grid, nodes, elems = tetrahedralize(fluid_surface_mesh, switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                            fluid_volume_mesh = grid
                        except:
                            try:
                                #tet_fluid.make_manifold(verbose=True)
                                fix = pymeshfix.MeshFix(fluid_surface_mesh)
                                fix.repair()
                                grid, nodes, elems = tetrahedralize(fix.mesh, switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                                fluid_volume_mesh = grid
                            except:
                                fluid_volume_mesh = None
                        if isinstance(fluid_volume_mesh, type(None)):
                            print("Failed to generate fluid volume mesh.")
                            network_fluid_surface_meshes.append(None)
                            network_fluid_volume_meshes.append(None)
                        else:
                            fluid_volume_mesh = remesh_volume(fluid_volume_mesh, hausd=hausd)
                            fluid_surface_mesh = fluid_volume_mesh.extract_surface()
                            network_fluid_surface_meshes.append(fluid_surface_mesh)
                            network_fluid_volume_meshes.append(fluid_volume_mesh)
                    if tissue:
                        # Extrude the root of the tree to ensure proper intersection with the tissue domain.
                        root_extension = max(tree.data[0, 21] * 4, tree.data[0, 20] * 0.5)
                        tree.data[0, 0:3] += root_extension * tree.data.get('w_basis', 0)
                        # Should check to see that the extended point does not intersect with another fluid or tissue domain.
                        fluid_surface_boolean_mesh = tree.export_solid(watertight=True)
                        if len(self.tissue_domain_surface_meshes) > 0:
                            tissue_domain = self.tissue_domain_surface_meshes[-1]
                        else:
                            tissue_domain = tree.domain.boundary
                        tissue_domain = boolean(tissue_domain, fluid_surface_boolean_mesh, operation='difference')
                        tissue_domain = remesh_surface(tissue_domain, hausd=hausd, verbosity=0)
                        #tet_tissue = tetgen.TetGen(tissue_domain)
                        #tree.data[0, 0:3] += root_extension * tree.data.get('w_basis', 0)
                        try:
                            # tet_tissue.make_manifold(verbose=False)
                            # fix = pymeshfix.MeshFix(tissue_domain)
                            # fix.repair(verbose=True)
                            grid, nodes, elems = tetrahedralize(tissue_domain, switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                            tissue_volume_mesh = grid
                        except:
                            try:
                                # tet_tissue.make_manifold(verbose=True)
                                fix = pymeshfix.MeshFix(fix.mesh)
                                fix.repair()
                                grid, nodes, elems = tetrahedralize(fix.mesh, switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                                tissue_volume_mesh = grid
                            except:
                                tissue_volume_mesh = None
                        if isinstance(tissue_volume_mesh, type(None)):
                            print("Failed to generate tissue volume mesh.")
                            network_tissue_surface_meshes.append(None)
                            network_tissue_volume_meshes.append(None)
                        else:
                            if remesh_vol:
                                tissue_volume_mesh = remesh_volume(tissue_volume_mesh, hausd=hausd)
                            tissue_domain = tissue_volume_mesh.extract_surface()
                            network_tissue_surface_meshes.append(tissue_domain)
                            network_tissue_volume_meshes.append(tissue_volume_mesh)
                self.fluid_domain_surface_meshes.append(network_fluid_surface_meshes)
                self.fluid_domain_volume_meshes.append(network_fluid_volume_meshes)
                self.tissue_domain_surface_meshes.append(network_tissue_surface_meshes)
                self.tissue_domain_volume_meshes.append(network_tissue_volume_meshes)
        elif isinstance(self.synthetic_object, svv.forest.forest.Forest) and not isinstance(self.synthetic_object.connections, type(None)):
            if fluid or tissue:
                if tissue:
                    network_solids, _, _ = self.synthetic_object.connections.export_solid(extrude_roots=True)
                else:
                    network_solids, _, _ = self.synthetic_object.connections.export_solid(extrude_roots=False)
                for i, fluid_surface in enumerate(network_solids):
                    if fluid:
                        #tet_fluid = tetgen.TetGen(fluid_surface)
                        try:
                            grid, nodes, elems = tetrahedralize(fluid_surface, switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                            fluid_volume = grid
                        except:
                            #tet_fluid.make_manifold(verbose=True)
                            fix = pymeshfix.MeshFix(fluid_surface)
                            fix.repair()
                            grid, nodes, elems = tetrahedralize(fix.mesh, switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                            fluid_volume = grid
                        if isinstance(fluid_volume, type(None)):
                            print("Failed to generate fluid volume mesh.")
                            self.fluid_domain_surface_meshes.append(fluid_surface)
                            self.fluid_domain_volume_meshes.append(None)
                        else:
                            hsize = fluid_surface.hsize
                            if (boundary_layer or wall_layers) and fluid:
                                fluid_surface = fluid_volume.extract_surface()
                                # faces, wall_surfaces, cap_surfaces, lumen_surfaces, _
                            if boundary_layer and fluid:
                                fluid_surface_faces = extract_faces(fluid_surface, fluid_volume)
                                # Use lumen (vessel wall) surfaces for boundary-layer generation
                                lumens = fluid_surface_faces[3]
                                if len(lumens) > 1:
                                    print("Boundary layer generation with more than one wall mesh is ambiguous.")
                                    print("Only the first wall mesh will be used.")
                                elif len(lumens) == 0:
                                    print("No wall mesh found for boundary layer generation.")
                                wall = lumens[0]
                                for j in range(boundary_layer_attempts):
                                    try:
                                        fluid_boundary_layers = BoundaryLayer(wall,
                                                                              layer_thickness=layer_thickness_ratio * hsize,
                                                                              remesh_vol=remesh_vol)
                                        fluid_volume, fluid_interior, fluid_boundary = fluid_boundary_layers.generate()
                                        fluid_surface = fluid_volume.extract_surface()
                                        success = True
                                        print("Generated boundary layers on attempt {}/{}.".format(i + 1,
                                                                                                   boundary_layer_attempts))
                                    except:
                                        print("Failed to generate boundary layers {}/{}.\n".format(i + 1,
                                                                                                   boundary_layer_attempts))
                                        fluid_boundary = None
                                        fluid_interior = None
                                        success = False
                                        layer_thickness_ratio *= layer_thickness_ratio_adjustment
                                    if success:
                                        break
                                self.fluid_domain_boundary_layers.append(fluid_boundary)
                                self.fluid_domain_interiors.append(fluid_interior)
                            else:
                                if remesh_vol:
                                    fluid_volume = remesh_volume(fluid_volume, hsiz=hsize)
                                self.fluid_domain_boundary_layers.append(None)
                                self.fluid_domain_interiors.append(None)
                            if wall_layers and fluid:
                                if isinstance(wall_thickness, type(None)):
                                    wall_thickness = 2 * layer_thickness_ratio * hsize
                                wall = fluid_surface_faces[1][0]
                                fluid_boundary_layers = BoundaryLayer(wall, negate_warp_vectors=False,
                                                                      layer_thickness=wall_thickness,
                                                                      remesh_vol=False, combine=False)
                                _, _, fluid_wall = fluid_boundary_layers.generate()
                                # Perform tetrahedron re-orientation to ensure positive Jacobian
                                fluid_wall = remesh_volume(fluid_wall, nomove=True, noinsert=True, nosurf=True, verbosity=4)
                                if remesh_vol:
                                    fluid_wall = remesh_volume(fluid_wall, hausd=hausd, nosurf=True, verbosity=4)
                                self.fluid_domain_wall_layers.append(fluid_wall)
                            else:
                                self.fluid_domain_wall_layers.append(None)
                            fluid_surface.hsize = hsize
                        self.fluid_domain_surface_meshes.append(fluid_surface)
                        self.fluid_domain_volume_meshes.append(fluid_volume)
                    else:
                        self.fluid_domain_surface_meshes.append(fluid_surface)
                        self.fluid_domain_volume_meshes.append(None)
            if tissue:
                tissue_domain = deepcopy(self.synthetic_object.domain.boundary)
                tissue_domain = tissue_domain.compute_normals(auto_orient_normals=True)
                fluid_hsize = min([mesh.cell_data["hsize"][0] for mesh in self.fluid_domain_surface_meshes])
                radii = []
                for net in range(len(self.synthetic_object.networks)):
                    for tr in range(len(self.synthetic_object.networks[net])):
                        radii.append(self.synthetic_object.networks[net][tr].data[0, 21])
                hsize = min(radii) * 2.0
                print("Remeshing tissue domain with edge size {}.".format(hsize))
                tissue_domain = remesh_surface(tissue_domain, hsiz=hsize, verbosity=0)
                for i, fluid_surface in enumerate(self.fluid_domain_surface_meshes):
                    fluid_surface_normals = fluid_surface.compute_normals(auto_orient_normals=True)
                    print("Performing boolean operation with fluid surface mesh {}.".format(i))
                    tissue_domain = boolean(tissue_domain, fluid_surface_normals, operation='difference', engine='blender')
                    tissue_domain = tissue_domain.compute_normals(auto_orient_normals=True)
                    print("Remeshing tissue domain with edge size {}.".format(fluid_hsize))
                    #tissue_domain = remesh_surface(tissue_domain, hmin=fluid_hsize, hmax=hsize)
                    tissue_domain = remesh_surface(tissue_domain, optim=True)
                self.tissue_domain_surface_meshes.append(tissue_domain)
                #tissue_domain = remesh_surface(tissue_domain, hausd=hausd)
                print("Tetrahedralizing tissue domain.")
                #tet_tissue = tetgen.TetGen(tissue_domain)
                try:
                    grid, nodes, elems = tetrahedralize(tissue_domain, switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                    tissue_volume_mesh = grid
                except:
                    #tet_tissue.make_manifold(verbose=True)
                    fix = pymeshfix.MeshFix(tissue_domain)
                    fix.repair()
                    grid, nodes, elems = tetrahedralize(fix.mesh, switches='pq{}/{}MVYSJ'.format(minratio, mindihedral))
                    tissue_volume_mesh = grid
                if isinstance(tissue_volume_mesh, type(None)):
                    print("Failed to generate tissue volume mesh.")
                else:
                    if remesh_vol:
                        tissue_volume_mesh = remesh_volume(tissue_volume_mesh, hausd=hausd, nosurf=True)
                    tissue_surface = tissue_volume_mesh.extract_surface()
                    self.tissue_domain_surface_meshes[-1] = tissue_surface
                    self.tissue_domain_volume_meshes.append(tissue_volume_mesh)
        else:
            raise ValueError("Unsupported synthetic object type.")

    def extract_faces(self, crease_angle=60.0, verbose=False):
        """
        Extract the faces from the mesh objects.
        :return:
        """
        self.tissue_domain_faces = []
        self.fluid_domain_faces = []
        self.fluid_domain_meshes = []
        self.tissue_domain_meshes = []
        if isinstance(self.synthetic_object, svv.tree.tree.Tree):
            if len(self.fluid_domain_surface_meshes) > 0 and len(self.fluid_domain_volume_meshes) > 0:
                faces, walls, caps, lumens, shared_boundaries = extract_faces(
                    self.fluid_domain_surface_meshes[0], self.fluid_domain_volume_meshes[0],
                    crease_angle=crease_angle, verbose=verbose)
                # For fluid, use lumen surfaces as primary vessel walls, but include any remaining walls
                self.fluid_domain_faces.append({'walls': walls, 'lumens': lumens, 'caps': caps, 'shared_boundaries': shared_boundaries})
                fluid_mesh = GeneralMesh()
                fluid_mesh.add_mesh(self.fluid_domain_volume_meshes[0], name='fluid_msh_0')
                for i, wall in enumerate(walls):
                    fluid_mesh.add_face(wall, name='wall_{}'.format(i))
                for i, cap in enumerate(caps):
                    fluid_mesh.add_face(cap, name='cap_{}'.format(i))
                for i, lumen in enumerate(lumens):
                    fluid_mesh.add_face(lumen, name='lumen_{}'.format(i))
                fluid_mesh.check_mesh()
                self.fluid_domain_meshes.append(fluid_mesh)
            if len(self.tissue_domain_surface_meshes) > 0 and len(self.tissue_domain_volume_meshes) > 0:
                faces, walls, caps, lumens, shared_boundaries = extract_faces(
                    self.tissue_domain_surface_meshes[0], self.tissue_domain_volume_meshes[0],
                    crease_angle=crease_angle, verbose=verbose)
                self.tissue_domain_faces.append({'walls': walls, 'lumens': lumens, 'caps': caps, 'shared_boundaries': shared_boundaries})
                tissue_mesh = GeneralMesh()
                tissue_mesh.add_mesh(self.tissue_domain_volume_meshes[0], name='tissue_msh_0')
                for i, wall in enumerate(walls):
                    tissue_mesh.add_face(wall, name='wall_{}'.format(i))
                for i, cap in enumerate(caps):
                    tissue_mesh.add_face(cap, name='cap_{}'.format(i))
                for i, lumen in enumerate(lumens):
                    tissue_mesh.add_face(lumen, name='lumen_{}'.format(i))
                tissue_mesh.check_mesh()
                self.tissue_domain_meshes.append(tissue_mesh)
        elif isinstance(self.synthetic_object, svv.forest.forest.Forest):
            network_tissue_faces = []
            network_tissue_domains = []
            network_fluid_faces = []
            network_fluid_domains = []
            for i in range(len(self.fluid_domain_surface_meshes)):
                for j in range(len(self.fluid_domain_surface_meshes[i])):
                    surface = self.fluid_domain_surface_meshes[i][j]
                    mesh = self.fluid_domain_volume_meshes[i][j]
                    if isinstance(surface, type(None)) or isinstance(mesh, type(None)):
                        network_fluid_faces.append(None)
                        network_fluid_domains.append(None)
                        continue
                    faces, walls, caps, lumens, shared_boundaries = extract_faces(
                        surface, mesh, crease_angle=crease_angle, verbose=verbose)
                    # For fluid, use lumen surfaces as primary vessel walls, but include any remaining walls
                    all_walls = lumens + walls
                    network_fluid_faces.append({'walls': all_walls, 'caps': caps, 'shared_boundaries': shared_boundaries})
                    fluid_mesh = GeneralMesh()
                    fluid_mesh.add_mesh(mesh, name='fluid_msh_{}'.format(len(self.fluid_domain_meshes)))
                    for k, wall in enumerate(walls):
                        fluid_mesh.add_face(wall, name='wall_{}'.format(k))
                    for k, cap in enumerate(caps):
                        fluid_mesh.add_face(cap, name='cap_{}'.format(k))
                    for k, lumen in enumerate(lumens):
                        fluid_mesh.add_face(lumen, name='lumen_{}'.format(k))
                    fluid_mesh.check_mesh()
                    network_fluid_domains.append(fluid_mesh)
                self.fluid_domain_faces.append(network_fluid_faces)
                self.fluid_domain_meshes.append(network_fluid_domains)
            for i in range(len(self.tissue_domain_surface_meshes)):
                for j in range(len(self.tissue_domain_surface_meshes[i])):
                    surface = self.tissue_domain_surface_meshes[i][j]
                    mesh = self.tissue_domain_volume_meshes[i][j]
                    faces, walls, caps, lumens, shared_boundaries = extract_faces(
                        surface, mesh, crease_angle=crease_angle, verbose=False)
                    network_tissue_faces.append({'walls': walls, 'caps': caps, 'shared_boundaries': shared_boundaries})
                    tissue_mesh = GeneralMesh()
                    tissue_mesh.add_mesh(mesh, name='tissue_msh_{}'.format(len(self.tissue_domain_meshes)))
                    for k, wall in enumerate(walls):
                        tissue_mesh.add_face(wall, name='lumen_{}'.format(k))
                    for k, cap in enumerate(caps):
                        tissue_mesh.add_face(cap, name='wall_{}'.format(k))
                    for k, lumen in enumerate(lumens):
                        tissue_mesh.add_face(lumen, name='lumen_{}'.format(i))
                    tissue_mesh.check_mesh()
                    network_tissue_domains.append(tissue_mesh)
                self.tissue_domain_faces.append(network_tissue_faces)
                self.tissue_domain_meshes.append(network_tissue_domains)

    def construct_3d_fluid_equation(self, *args):
        """
        Build the 3D fluid equation object and its boundary conditions.

        This configures a FluidEquation for svMultiPhysics using the
        already extracted GeneralMesh, identifies an inlet cap by
        proximity to the tree root, assigns outlets and walls, and sets
        linear-solver data so the generated XML matches svMultiPhysics
        expectations.

        Behavior
        - Inlet selection: the cap face closest (by centroid distance)
          to the root location `synthetic_object.data[0, 0:3]`.
        - Boundary conditions: inlet is Dirichlet (velocity) with flux
          imposition and parabolic profile; other caps are zero-pressure
          Neumann; wall faces are no-slip Dirichlet.
        - Sign + units: inlet magnitude uses the stored inflow value and
          is negated to follow the solver inflow direction convention.
          Viscosity uses `parameters.kinematic_viscosity`; ensure units
          are consistent with solver configuration.
        - Linear solver: sets GMRES with FSILS linear algebra backend
          and preconditioner to emit the required <Linear_algebra> block.

        Returns
        - FluidEquation
        """
        if isinstance(self.synthetic_object, svv.tree.tree.Tree):
            fluid_mesh = self.fluid_domain_meshes[0]
            fluid_equation = FluidEquation()
            fluid_equation.add_mesh(fluid_mesh)
            inlet = None
            best = numpy.inf
            inlet_center = self.synthetic_object.data[0, 0:3]
            for name, face in fluid_mesh.faces.items():
                if 'cap' in name:
                    dist = numpy.linalg.norm(face.center - inlet_center)
                    if dist < best:
                        best = dist
                        inlet = name
            if isinstance(inlet, type(None)):
                raise ValueError("Inlet not found.")
            fluid_equation.add_inlet(inlet, -1*self.synthetic_object.data[0, 22])
            fluid_equation.set_viscosity('Constant', self.synthetic_object.parameters.kinematic_viscosity)
        elif isinstance(self.synthetic_object, svv.forest.forest.Forest):
            if len(args) == 0:
                network_id = 0
                tree_id = 0
            elif len(args) == 1:
                network_id = args[0]
                tree_id = 0
            elif len(args) == 2:
                network_id = args[0]
                tree_id = args[1]
            else:
                raise ValueError("Too many arguments.")
            fluid_mesh = self.fluid_domain_meshes[network_id][tree_id]
            fluid_equation = FluidEquation()
            fluid_equation.add_mesh(fluid_mesh)
            # Verify cap inlet: choose cap closest to this tree's root
            inlet = None
            best = numpy.inf
            for name, face in fluid_mesh.faces.items():
                inlet_center = self.synthetic_object.networks[network_id][tree_id].data[0, 0:3]
                if 'cap' in name:
                    dist = numpy.linalg.norm(face.center - inlet_center)
                    if dist < best:
                        best = dist
                        inlet = name
            if isinstance(inlet, type(None)):
                raise ValueError("Inlet not found.")
            fluid_equation.add_inlet(inlet, -1*self.synthetic_object.networks[network_id][tree_id].data[0, 22])
            fluid_equation.set_viscosity('Constant', self.synthetic_object.networks[network_id][tree_id].parameters.kinematic_viscosity)
        else:
            raise ValueError("Unsupported synthetic object type.")
        # Assign outlets and walls for remaining faces
        for face in fluid_mesh.faces:
            if face == inlet:
                continue
            if 'cap' in face:
                fluid_equation.add_outlet(face, value=0.0)
            if 'wall' in face:
                fluid_equation.add_wall(face)
        # Sanity check that all faces received a BC
        fluid_equation.check_bcs()

        # Configure linear solver and linear algebra backend so the
        # <LS> section includes the required <Linear_algebra> block
        # for svMultiPhysics. Default to GMRES + FSILS.
        try:
            fluid_equation.linear_solver.set_type("GMRES")
        except Exception:
            fluid_equation.linear_solver.set_type("NS")
        # Ensure Linear_algebra is emitted even if defaults are used.
        if hasattr(fluid_equation.linear_solver, "set_linear_algebra"):
            fluid_equation.linear_solver.set_linear_algebra(linalg_type="fsils", preconditioner="fsils")
        elif hasattr(fluid_equation.linear_solver, "set_linear_algebra_type"):
            fluid_equation.linear_solver.set_linear_algebra_type("fsils")
            if hasattr(fluid_equation.linear_solver, "set_preconditioner"):
                fluid_equation.linear_solver.set_preconditioner("fsils")
        return fluid_equation

    def construct_3d_fluid_simulation(self, *args):
        """
        Construct the 3D simulations.
        :return:
        """
        if len(args) == 0:
            network_id = 0
            tree_id = 0
        elif len(args) == 1:
            network_id = args[0]
            tree_id = 0
        elif len(args) == 2:
            network_id = args[0]
            tree_id = args[1]
        else:
            raise ValueError("Too many arguments.")
        if isinstance(self.synthetic_object, svv.tree.tree.Tree):
            simulation_file = minidom.Document()
            svfsi_file = simulation_file.createElement("svMultiPhysicsFile")
            svfsi_file.setAttribute("version", "0.1")
            general_simulation_parameters = GeneralSimulationParameters()
            fluid_mesh = self.fluid_domain_meshes[0]
            fluid_equation = self.construct_3d_fluid_equation()
            svfsi_file.appendChild(general_simulation_parameters.toxml())
            svfsi_file.appendChild(fluid_mesh.toxml())
            svfsi_file.appendChild(fluid_equation.toxml())
            simulation_file.appendChild(svfsi_file)
            self.fluid_3d_simulations[0] = tuple([simulation_file, fluid_mesh])
        elif isinstance(self.synthetic_object, svv.forest.forest.Forest):
            simulation_file = minidom.Document()
            svfsi_file = simulation_file.createElement("svMultiPhysicsFile")
            svfsi_file.setAttribute("version", "0.1")
            general_simulation_parameters = GeneralSimulationParameters()
            fluid_mesh = self.fluid_domain_meshes[network_id][tree_id]
            # Forward variadic args to equation builder
            fluid_equation = self.construct_3d_fluid_equation(*args)
            svfsi_file.appendChild(general_simulation_parameters.toxml())
            svfsi_file.appendChild(fluid_mesh.toxml())
            svfsi_file.appendChild(fluid_equation.toxml())
            simulation_file.appendChild(svfsi_file)
            self.fluid_3d_simulations[network_id][tree_id] = tuple([simulation_file, fluid_mesh])
        else:
            raise ValueError("Index out of range.")
        return

    def write_3d_fluid_simulation(self, *args):
        """
        Write the fluid simulation to disk.
        :return:
        """
        if len(args) == 0:
            network_id = 0
            tree_id = 0
        elif len(args) == 1:
            network_id = args[0]
            tree_id = 0
        elif len(args) == 2:
            network_id = args[0]
            tree_id = args[1]
        else:
            raise ValueError("Too many arguments.")
        if isinstance(self.synthetic_object, svv.tree.tree.Tree):
            simulation_file, fluid_mesh = self.fluid_3d_simulations[0]
        elif isinstance(self.synthetic_object, svv.forest.forest.Forest):
            simulation_file, fluid_mesh = self.fluid_3d_simulations[network_id][tree_id]
        else:
            raise ValueError("Index out of range.")
        if not isinstance(simulation_file, type(None)) and not isinstance(fluid_mesh, type(None)):
            if not os.path.exists(self.file_path):
                os.mkdir(self.file_path)
                if not os.path.exists(self.file_path + os.sep + "mesh"):
                    os.mkdir(self.file_path + os.sep + "mesh")
                if not os.path.exists(self.file_path + os.sep + "mesh" + os.sep + fluid_mesh.name):
                    os.mkdir(self.file_path + os.sep + "mesh" + os.sep + fluid_mesh.name)
                if not os.path.exists(self.file_path + os.sep + "mesh" + os.sep + fluid_mesh.name + os.sep + "mesh-surfaces"):
                    os.mkdir(self.file_path + os.sep + "mesh" + os.sep + fluid_mesh.name + os.sep + "mesh-surfaces")
            if isinstance(fluid_mesh.mesh, pyvista.UnstructuredGrid):
                fluid_mesh.mesh.save(self.file_path + os.sep + "mesh" + os.sep + fluid_mesh.name + os.sep + "{}.vtu".format(fluid_mesh.name))
            elif isinstance(fluid_mesh.mesh, pyvista.PolyData):
                fluid_mesh.mesh.save(self.file_path + os.sep + "mesh" + os.sep + fluid_mesh.name + os.sep + "{}.vtp".format(fluid_mesh.name))
            else:
                raise ValueError("Mesh must be a pyvista mesh object.")
            for name, face in fluid_mesh.faces.items():
                #face.cell_data["ModelFaceID"] = face.cell_data["ModelFaceID"].astype(numpy.int32)
                #face.cell_data["GlobalElementID"] = face.cell_data["GlobalElementID"].astype(numpy.int32)
                #face.cell_data.remove("ModelFaceID")
                #face.cell_data.remove("GlobalElementID")
                if isinstance(face, pyvista.PolyData):
                    face.save(self.file_path + os.sep + "mesh" + os.sep + fluid_mesh.name + os.sep + "mesh-surfaces" + os.sep + "{}.vtp".format(name))
                else:
                    raise ValueError("Face must be a pyvista mesh object.")
            file_name = self.file_path + os.sep + "fluid_simulation_{}-{}.xml".format(network_id, tree_id)
            with open(file_name, 'w') as f:
                f.write(simulation_file.toprettyxml())
        else:
            raise ValueError("Simulation file or mesh object not found.")

    def construct_1d_fluid_simulation(self, *args, viscosity=None, density=None, time_step_size=0.01,
                                      number_time_steps=100, olufsen_material_exponent=2):
        # Lazy import 1D ROM modules to avoid importing vtk unless needed.
        from svv.simulation.fluid.rom.one_d import parameters as one_d_parameters
        from svv.simulation.fluid.rom.one_d import mesh as one_d_mesh
        if len(args) == 0:
            network_id = 0
            tree_id = 0
        elif len(args) == 1:
            network_id = args[0]
            tree_id = 0
        elif len(args) == 2:
            network_id = args[0]
            tree_id = args[1]
        else:
            raise ValueError("Too many positional input arguments")
        if isinstance(self.synthetic_object, svv.tree.tree.Tree):
            centerlines, *_ = self.synthetic_object.export_centerlines()
            material = one_d_parameters.MaterialModel()
            params = one_d_parameters.Parameters()
            params.output_directory = self.file_path + os.sep + "fluid" + os.sep + "1d"
            params.solver_output_file = self.file_path + os.sep + "fluid" + os.sep + "1d" + os.sep + "1d_simulation_input.json"
            params.centerlines_input_file = self.file_path + os.sep + "fluid" + os.sep + "1d" + os.sep + "centerlines.vtp"
            params.outlet_face_names_file = self.file_path + os.sep + "fluid" + os.sep + "1d" + "outlets"
            params.seg_size_adaptive = True
            params.model_order = 1  # Since this is strictly a 1d ROM simulation it is not an exposed parameter
            params.uniform_bc = False
            params.inflow_input_file = self.file_path + os.sep + "fluid" + os.sep + "1d" + "inflow_1d.flow"
            params.outflow_bc_type = ["rcrt.dat"]
            params.outflow_bc_file = self.file_path + os.sep + "fluid" + os.sep + "1d"
            params.model_name = "1d_model_{}-{}".format(network_id, tree_id)
            params.compute_mesh = True
            params.time_step = time_step_size
            params.num_time_steps = number_time_steps
            params.olufsen_material_exponent = olufsen_material_exponent
            params.material_model = material.OLUFSEN
            params.viscosity = viscosity
            params.density = density
            mesh = one_d_mesh.Mesh()
            self.fluid_1d_simulations[0] = tuple([centerlines, mesh, params])
        else:
            raise ValueError("Index out of range.")

    def write_1d_fluid_simulation(self, *args):
        pass

    def construct_0d_fluid_simulation(self, *args):
        pass

    def write_0d_fluid_simulation(self, *args, **kwargs):
        if isinstance(self.synthetic_object, svv.tree.tree.Tree):
            export_tree_0d_simulation(self.synthetic_object, kwargs)
        elif isinstance(self.synthetic_object, svv.forest.forest.Forest) and not isinstance(self.synthetic_object.connections, type(None)):
            for idx in range(len(self.synthetic_object.networks)):
                # args are the network_id and the inlet selection
                export_forest_0d_simulation(self.synthetic_object, args, kwargs)

    def construct_3d_tissue_perfusion_equation(self, *args):
        pass

    def construct_3d_tissue_perfusion_simulation(self, *args):
        pass

    def write_3d_tissue_perfusion_simulation(self, *args):
        pass

    def pulsatile_waveform(self, *args):
        """
        Generate a pulsatile waveform for fluid simulations. This function should accept either an average flow rate
        or an array of flow rate values over the nominal cardiac cycle t -> [0, 1].

        This function should check that the waveform is properly formatted and that the values are within a reasonable
        range; otherwise it should warn the user that values of the waveform might result in unrealistic/erroneous
        results. (e.g. high reynolds number flows, reversed flow, etc.)
        :param args:
        :return:
        """
        pass

    def generate_inflow(self, *args, filename=None):
        pass

    def write(self):
        """
        Save the simulation file to disk.
        :return:
        """
        os.mkdir(self.file_path)
        mesh_folder = self.file_path + os.sep + "mesh"
        os.mkdir(mesh_folder)
        surface_folder = mesh_folder + os.sep + "mesh-surfaces"
        os.mkdir(surface_folder)

    def write_input_file(self):
        """
        Write the simulation input file.
        :return:
        """
        pass
