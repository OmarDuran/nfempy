import sys

import gmsh
import numpy as np

gmsh.initialize()

from geometry.fracture_network import FractureNetwork
from geometry.geometry_builder import GeometryBuilder


class Mesher:
    def __init__(self, dimension):
        self.dimension = dimension
        self.geometry_builder = None
        self.fracture_network = None
        self.points = None
        self.lc = 1.0
        self.tags_0d = None
        self.tags_1d = None
        self.tags_2d = None

    def set_geometry_builder(self, geometry_builder):
        self.geometry_builder = geometry_builder

    def set_fracture_network(self, fracture_network):
        self.fracture_network = fracture_network

    def set_points(self):
        self.points = self.geometry_builder.points
        if self.fracture_network is None:
            return
        self.points = np.append(self.points, self.fracture_network.points, axis=0)

        max_point_id = len(self.geometry_builder.points)
        max_cell_id = len(self.geometry_builder.cells)
        self.fracture_network.shift_point_ids(max_point_id)
        self.fracture_network.shift_cell_ids(max_cell_id)

    def add_domain_descritpion(self):

        assert self.dimension == 2

        # add domain cells
        graph_nodes = list(self.geometry_builder.graph.nodes())
        geo_cells = self.geometry_builder.cells[graph_nodes]
        geo_1_cells = [cell for cell in geo_cells if cell.dimension == 1]
        geo_2_cells = [cell for cell in geo_cells if cell.dimension == 2]

        self.tags_2d = []
        for geo_2_cell in geo_2_cells:
            for cell_i in geo_2_cell.boundary_cells:
                b = cell_i.boundary_cells[0].point_id + 1
                e = cell_i.boundary_cells[1].point_id + 1
                gmsh.model.geo.addLine(b, e, cell_i.id)
            tags = [cell.id for cell in geo_2_cell.boundary_cells]
            gmsh.model.geo.addCurveLoop(tags, geo_2_cell.id)
            gmsh.model.geo.addPlaneSurface([geo_2_cell.id], geo_2_cell.id)
            self.tags_2d.append(geo_2_cell.id)

        gmsh.model.geo.synchronize()

        # add physical tags
        for geo_1_cell in geo_1_cells:
            gmsh.model.addPhysicalGroup(1, [geo_1_cell.id], geo_1_cell.id)

        gmsh.model.addPhysicalGroup(2, self.tags_2d, self.tags_2d[0])

    def add_fracture_network_description(self):

        # add fn cells
        graph_nodes = list(self.fracture_network.graph.nodes())
        geo_cells = self.fracture_network.cells[graph_nodes]
        geo_0_cells = [cell for cell in geo_cells if cell.dimension == 0]
        i_cells = [
            cell
            for cell in geo_cells
            if cell.dimension == 1 and len(cell.immersed_cells) > 0
        ]
        ni_cells = [cell.id for m_cell in i_cells for cell in m_cell.immersed_cells]
        geo_1_cells = [
            cell
            for cell in geo_cells
            if cell.dimension == 1 and cell.id not in ni_cells
        ]

        self.tags_1d = []
        self.tags_0d = []
        for geo_1_cell in geo_1_cells:
            n_immersed_cells = len(geo_1_cell.immersed_cells)
            if n_immersed_cells == 0:
                b = geo_1_cell.boundary_cells[0].point_id + 1
                e = geo_1_cell.boundary_cells[1].point_id + 1
                gmsh.model.geo.addLine(b, e, geo_1_cell.id)
                self.tags_1d.append(geo_1_cell.id)
                self.tags_0d.append(b)
                self.tags_0d.append(e)
            else:
                for cell_i in geo_1_cell.immersed_cells:
                    b = cell_i.boundary_cells[0].point_id + 1
                    e = cell_i.boundary_cells[1].point_id + 1
                    gmsh.model.geo.addLine(b, e, cell_i.id)
                    self.tags_1d.append(cell_i.id)
                    self.tags_0d.append(b)
                    self.tags_0d.append(e)

        gmsh.model.geo.synchronize()

        for geo_1_cell in geo_1_cells:
            tags = [cell.point_id + 1 for cell in geo_0_cells if cell.physical_tag == geo_1_cell.physical_tag]
            gmsh.model.addPhysicalGroup(0, tags, geo_1_cell.physical_tag)

        frac_tags = self.fracture_network.fracture_tags
        for geo_0_cell in geo_0_cells:
            check_q = geo_0_cell.physical_tag not in frac_tags
            if check_q:
                gmsh.model.addPhysicalGroup(0, [geo_0_cell.point_id + 1], geo_0_cell.physical_tag)

        for geo_1_cell in geo_1_cells:
            tags = []
            n_immersed_cells = len(geo_1_cell.immersed_cells)
            if n_immersed_cells == 0:
                tags = [geo_1_cell.id]
            else:
                tags = [cell.id for cell in geo_1_cell.immersed_cells]
            gmsh.model.addPhysicalGroup(1, tags, geo_1_cell.physical_tag)

    def write_mesh(self, file_name):
        gmsh.write(file_name)
        gmsh.finalize()
        max_point_id = -len(self.geometry_builder.points)
        # max_cell_id = -len(self.geometry_builder.cells)
        self.fracture_network.shift_point_ids(max_point_id)
        # self.fracture_network.shift_cell_ids(max_cell_id)

    def generate(self, lc):
        self.lc = lc
        n_points = len(self.points)
        for tag, point in enumerate(self.points):
            gmsh.model.geo.addPoint(point[0], point[1], 0, self.lc, tag + 1)

        self.add_domain_descritpion()

        if self.fracture_network is not None:
            self.add_fracture_network_description()

            # embed entities
            gmsh.model.mesh.embed(0, self.tags_0d, 2, self.tags_2d[0])
            gmsh.model.mesh.embed(1, self.tags_1d, 2, self.tags_2d[0])

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)

        # if "-nopopup" not in sys.argv:
        #     gmsh.fltk.run()
