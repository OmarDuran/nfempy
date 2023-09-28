import gmsh
import numpy as np

from geometry.domain import Domain


class ConformalMesher:
    def __init__(self, dimension):
        self.dimension = dimension
        self.geometry_builder = None
        self.fracture_network = None
        self.points = None
        self.lc = 1.0
        self.tags_0d = None
        self.tags_1d = None
        self.tags_2d = None
        self.tags_3d = None

        self.domain: Domain = None

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

    def transfer_domain_descritpion(self):
        dimension = 0
        for dimension in range(len(self.domain.shapes)):
            if len(self.domain.shapes[dimension]) == 0:
                break

        vertex_stride = 0
        curve_stride = vertex_stride + len(self.domain.shapes[0])

        # transfer vertices
        for vertex in self.domain.shapes[0]:
            gmsh.model.geo.addPoint(
                vertex.point[0],
                vertex.point[1],
                vertex.point[2],
                self.lc,
                vertex.tag + 1,
            )
        gmsh.model.geo.synchronize()

        # transfer curves
        for curve in self.domain.shapes[1]:
            tag = curve_stride + curve.tag + 1
            if not curve.composite:
                print("curve tag: ", curve.tag)
                b = curve.boundary_shapes[0].tag + 1
                e = curve.boundary_shapes[1].tag + 1
                gmsh.model.geo.addLine(b, e, tag)

        gmsh.model.geo.synchronize()

        # transfer surfaces
        if dimension > 1:
            surface_stride = curve_stride + len(self.domain.shapes[1])
            for surface in self.domain.shapes[2]:
                tag = surface_stride + surface.tag + 1
                if not surface.composite:
                    print("surface tag: ", surface.tag)
                    wire = surface.boundary_shapes[0]
                    loop_tags = [
                        curve_stride + shape.tag + 1 for shape in wire.immersed_shapes
                    ]
                    loop_tags = [
                        loop_tags[i] * sign for i, sign in enumerate(wire.orientation)
                    ]
                    gmsh.model.geo.addCurveLoop(loop_tags, tag)
                    gmsh.model.geo.addPlaneSurface([tag], tag)

        # transfer volumes
        if dimension > 2:
            volume_stride = surface_stride + len(self.domain.shapes[2])
            for volume in self.domain.shapes[3]:
                tag = volume_stride + volume.tag + 1
                if not volume.composite:
                    shell = volume.boundary_shapes[0]
                    loop_tags = [
                        surface_stride + shape.tag + 1
                        for shape in shell.immersed_shapes
                    ]
                    gmsh.model.geo.addSurfaceLoop(loop_tags, tag)
                    gmsh.model.geo.addVolume([tag], tag)

        gmsh.model.geo.synchronize()

        # add physical tags
        for vertex in self.domain.shapes[0]:
            if vertex.physical_tag is not None:
                gmsh.model.addPhysicalGroup(0, [vertex.tag + 1], vertex.physical_tag)
        for curve in self.domain.shapes[1]:
            if curve.physical_tag is not None:
                gmsh.model.addPhysicalGroup(
                    1, [curve_stride + curve.tag + 1], curve.physical_tag
                )
        if dimension > 1:
            for surface in self.domain.shapes[2]:
                if surface.physical_tag is not None:
                    gmsh.model.addPhysicalGroup(
                        2, [surface_stride + surface.tag + 1], surface.physical_tag
                    )

        if dimension > 2:
            for volume in self.domain.shapes[3]:
                if volume.physical_tag is not None:
                    gmsh.model.addPhysicalGroup(
                        3, [volume_stride + volume.tag + 1], volume.physical_tag
                    )

    def transfer_domain_occ_descritpion(self):
        dimension = 0
        for dimension in range(len(self.domain.shapes)):
            if len(self.domain.shapes[dimension]) == 0:
                break

        vertex_stride = 0
        curve_stride = vertex_stride + len(self.domain.shapes[0])

        # transfer vertices
        for vertex in self.domain.shapes[0]:
            if len(vertex.immersed_shapes) > 0:
                continue
            # print("Vertex tag: ", vertex.tag + 1)
            gmsh.model.occ.addPoint(
                vertex.point[0],
                vertex.point[1],
                vertex.point[2],
                self.lc,
                vertex.tag + 1,
            )
        gmsh.model.occ.synchronize()

        # transfer curves
        for curve in self.domain.shapes[1]:
            if not curve.composite:
                # print("curve tag: ", curve_stride + curve.tag + 1)
                if len(curve.immersed_shapes) > 0:
                    continue
                tag = curve_stride + curve.tag + 1
                b = curve.boundary_shapes[0].tag + 1
                e = curve.boundary_shapes[1].tag + 1
                gmsh.model.occ.addLine(b, e, tag)

        gmsh.model.occ.synchronize()

        # transfer surfaces
        if dimension > 1:
            surface_stride = curve_stride + len(self.domain.shapes[1])
            for surface in self.domain.shapes[2]:
                tag = surface_stride + surface.tag + 1
                if not surface.composite:
                    # print("creating surface tag: ", surface.tag)
                    wire = surface.boundary_shapes[0]
                    wire.orient_immersed_edges()
                    loop_tags = [
                        curve_stride + shape.tag + 1 for shape in wire.immersed_shapes
                    ]
                    loop_tags = [
                        loop_tags[i] * sign for i, sign in enumerate(wire.orientation)
                    ]
                    gmsh.model.occ.addCurveLoop(loop_tags, tag)
                    gmsh.model.occ.addPlaneSurface([tag], tag)

        gmsh.model.occ.synchronize()

        # transfer volumes
        if dimension > 2:
            volume_stride = surface_stride + len(self.domain.shapes[2])
            for volume in self.domain.shapes[3]:
                tag = volume_stride + volume.tag + 1
                if not volume.composite:
                    # print("creating surface tag: ", surface.tag)
                    shell = volume.boundary_shapes[0]
                    loop_tags = [
                        surface_stride + shape.tag + 1
                        for shape in shell.immersed_shapes
                    ]
                    gmsh.model.occ.addSurfaceLoop(loop_tags, tag)
                    gmsh.model.occ.addVolume([tag], tag)

        gmsh.model.occ.synchronize()

        # add physical tags
        for vertex in self.domain.shapes[0]:
            if vertex.physical_tag is not None:
                if len(vertex.immersed_shapes) > 0:
                    tags = [ivertex.tag + 1 for ivertex in vertex.immersed_shapes]
                    tags = list(np.unique(tags))
                    gmsh.model.addPhysicalGroup(0, tags, vertex.physical_tag)
                else:
                    gmsh.model.addPhysicalGroup(
                        0, [vertex.tag + 1], vertex.physical_tag
                    )

        for curve in self.domain.shapes[1]:
            if curve.physical_tag is not None:
                if len(curve.immersed_shapes) > 0:
                    tags = [
                        curve_stride + icurve.tag + 1
                        for icurve in curve.immersed_shapes
                    ]
                    tags = list(np.unique(tags))
                    gmsh.model.addPhysicalGroup(1, tags, curve.physical_tag)
                    if self.domain.dimension > 1:
                        tags_0d = [
                            vertex.tag + 1
                            for vertex in curve.boundary_shapes
                            if vertex.physical_tag is None
                        ]
                        gmsh.model.addPhysicalGroup(0, tags_0d, curve.physical_tag)
                else:
                    if self.domain.dimension == 1:
                        no_predecessors = (
                            len(list(self.domain.graph.predecessors((1, curve.tag))))
                            == 0
                        )
                        if no_predecessors:
                            gmsh.model.addPhysicalGroup(
                                1, [curve_stride + curve.tag + 1], curve.physical_tag
                            )
                    elif self.domain.dimension == 2:

                        if self.domain.dimension > 1:
                            tags_0d = [
                                vertex.tag + 1
                                for vertex in curve.boundary_shapes
                                if vertex.physical_tag is None
                            ]
                            gmsh.model.addPhysicalGroup(0, tags_0d, curve.physical_tag)

                        gmsh.model.addPhysicalGroup(
                            1, [curve_stride + curve.tag + 1], curve.physical_tag
                        )
                    else:
                        aka = 0

        if dimension > 1:
            for surface in self.domain.shapes[2]:
                if surface.physical_tag is not None:
                    gmsh.model.addPhysicalGroup(
                        2, [surface_stride + surface.tag + 1], surface.physical_tag
                    )

        if dimension > 2:
            for volume in self.domain.shapes[3]:
                if volume.physical_tag is not None:
                    gmsh.model.addPhysicalGroup(
                        3, [volume_stride + volume.tag + 1], volume.physical_tag
                    )

        gmsh.model.occ.synchronize()

        # embed entities
        if dimension > 1:
            tags_2d = [shape.tag for shape in self.domain.shapes[2]]
            for surface in self.domain.shapes[2]:
                if surface.composite:
                    continue
                print("surface tag: ", surface.tag)
                tags_0d = []
                tags_1d = []
                shapes_c1 = [shape for shape in surface.immersed_shapes]
                for shape_c1 in shapes_c1:
                    if len(shape_c1.immersed_shapes) > 0:
                        tags_1d = tags_1d + [
                            curve_stride + curve.tag + 1
                            for curve in shape_c1.immersed_shapes
                        ]
                    else:
                        tags_1d = tags_1d + [curve_stride + shape_c1.tag + 1]
                    tags_0d = tags_0d + [
                        vertex.tag + 1 for vertex in shape_c1.boundary_shapes
                    ]

                tags_0d = list(np.unique(tags_0d))
                tags_1d = list(np.unique(tags_1d))
                gmsh.model.mesh.embed(0, tags_0d, 2, surface_stride + surface.tag + 1)
                gmsh.model.mesh.embed(1, tags_1d, 2, surface_stride + surface.tag + 1)

                numNodes = 10
                for tag_1d in tags_1d:
                    gmsh.model.mesh.setTransfiniteCurve(
                        tag_1d, numNodes, "Bump", coef=0.125 / 2
                    )
        # embed entities
        if dimension > 2:
            tags_3d = [shape.tag for shape in self.domain.shapes[3]]
            for volume in self.domain.shapes[3]:
                if volume.composite:
                    continue
                print("volume tag: ", volume.tag)
                tags_1d = []
                tags_2d = []
                shapes_c1 = [shape for shape in volume.immersed_shapes]
                for shape_c1 in shapes_c1:
                    if len(shape_c1.immersed_shapes) > 0:
                        tags_1d = tags_1d + [
                            curve_stride + curve.tag + 1
                            for curve in shape_c1.immersed_shapes
                        ]
                    else:
                        tags_2d = tags_2d + [surface_stride + shape_c1.tag + 1]
                    for shape_c2 in shape_c1.boundary_shapes:
                        tags_1d = tags_1d + [
                            curve_stride + curve.tag + 1
                            for curve in shape_c2.immersed_shapes
                        ]

                tags_1d = list(np.unique(tags_1d))
                tags_2d = list(np.unique(tags_2d))
                gmsh.model.mesh.embed(1, tags_1d, 3, volume_stride + volume.tag + 1)
                gmsh.model.mesh.embed(2, tags_2d, 3, volume_stride + volume.tag + 1)

                numNodes = 2
                for tag_1d in tags_1d:
                    gmsh.model.mesh.setTransfiniteCurve(
                        tag_1d, numNodes, "Bump", coef=0.25
                    )

    def add_domain_descritpion(self):
        if self.dimension == 1:
            self.add_domain_1d_descritpion()
        elif self.dimension == 2:
            self.add_domain_2d_descritpion()
        elif self.dimension == 3:
            self.add_domain_3d_descritpion()
        else:
            raise ValueError("Dimension not implemented yet, ", self.dimension)

    def add_domain_1d_descritpion(self):
        # add domain cells
        graph_nodes = list(self.geometry_builder.graph.nodes())
        geo_cells = self.geometry_builder.cells[graph_nodes]
        geo_0_cells = [cell for cell in geo_cells if cell.dimension == 0]
        geo_1_cells = [cell for cell in geo_cells if cell.dimension == 1]

        self.tags_1d = []
        for geo_1_cell in geo_1_cells:
            b = geo_1_cell.boundary_cells[0].point_id + 1
            e = geo_1_cell.boundary_cells[1].point_id + 1
            gmsh.model.geo.addLine(b, e, geo_1_cell.id)
            self.tags_1d.append(geo_1_cell.id)

        gmsh.model.geo.synchronize()

        # add physical tags
        for geo_0_cell in geo_0_cells:
            gmsh.model.addPhysicalGroup(0, [geo_0_cell.id + 1], geo_0_cell.physical_tag)

        for geo_1_cell in geo_1_cells:
            gmsh.model.addPhysicalGroup(1, self.tags_1d, geo_1_cell.physical_tag)

    def add_domain_2d_descritpion(self):
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
            gmsh.model.addPhysicalGroup(1, [geo_1_cell.id], geo_1_cell.physical_tag)

        for geo_2_cell in geo_2_cells:
            gmsh.model.addPhysicalGroup(2, self.tags_2d, geo_2_cell.physical_tag)

    def add_domain_3d_descritpion(self):
        # add domain cells
        graph_nodes = list(self.geometry_builder.graph.nodes())
        geo_cells = self.geometry_builder.cells[graph_nodes]
        geo_1_cells = [cell for cell in geo_cells if cell.dimension == 1]
        geo_2_cells = [cell for cell in geo_cells if cell.dimension == 2]
        geo_3_cells = [cell for cell in geo_cells if cell.dimension == 3]

        for geo_1_cell in geo_1_cells:
            b = geo_1_cell.boundary_cells[0].point_id + 1
            e = geo_1_cell.boundary_cells[1].point_id + 1
            print("cell id: ", geo_1_cell.id)
            gmsh.model.geo.addLine(b, e, geo_1_cell.id)

        gmsh.model.geo.synchronize()

        self.tags_3d = []
        self.tags_2d = []
        for geo_3_cell in geo_3_cells:
            for geo_2_cell in geo_3_cell.boundary_cells:
                tags = list(geo_2_cell.boundary_loop)
                gmsh.model.geo.addCurveLoop(tags, geo_2_cell.id)
                gmsh.model.geo.addPlaneSurface([geo_2_cell.id], geo_2_cell.id)
                self.tags_2d.append(geo_2_cell.id)
            gmsh.model.geo.addSurfaceLoop(self.tags_2d, geo_3_cell.id)
            gmsh.model.geo.addVolume([geo_3_cell.id], geo_3_cell.id)
            self.tags_3d.append(geo_3_cell.id)

        gmsh.model.geo.synchronize()

        # add physical tags
        for geo_2_cell in geo_2_cells:
            gmsh.model.addPhysicalGroup(2, [geo_2_cell.id], geo_2_cell.physical_tag)

        for geo_3_cell in geo_3_cells:
            gmsh.model.addPhysicalGroup(3, self.tags_3d, geo_3_cell.physical_tag)

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
            tags = [
                cell.point_id + 1
                for cell in geo_0_cells
                if cell.physical_tag == geo_1_cell.physical_tag
            ]
            gmsh.model.addPhysicalGroup(0, tags, geo_1_cell.physical_tag)

        frac_tags = self.fracture_network.fracture_tags
        for geo_0_cell in geo_0_cells:
            check_q = geo_0_cell.physical_tag not in frac_tags
            if check_q:
                gmsh.model.addPhysicalGroup(
                    0, [geo_0_cell.point_id + 1], geo_0_cell.physical_tag
                )

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
        if self.fracture_network is not None:
            max_point_id = -len(self.geometry_builder.points)
            self.fracture_network.shift_point_ids(max_point_id)

    def generate_from_domain(self, lc, n_refinements=0):
        gmsh.initialize()
        self.lc = lc

        # self.transfer_domain_descritpion()
        self.transfer_domain_occ_descritpion()

        for d in range(self.dimension + 1):
            gmsh.model.mesh.generate(d)
        for _ in range(n_refinements):
            gmsh.model.mesh.refine()

        # if "-nopopup" not in sys.argv:
        #     gmsh.fltk.run()
        # aka = 0

    def generate(self, lc, n_refinments=0):
        gmsh.initialize()
        self.lc = lc
        for tag, point in enumerate(self.points):
            gmsh.model.geo.addPoint(point[0], point[1], point[2], self.lc, tag + 1)

        self.add_domain_descritpion()

        if self.fracture_network is not None:
            assert self.dimension == 2

            gmsh.model.geo.synchronize()
            self.add_fracture_network_description()
            gmsh.model.geo.synchronize()
            # embed entities
            gmsh.model.mesh.embed(0, self.tags_0d, 2, self.tags_2d[0])
            gmsh.model.mesh.embed(1, self.tags_1d, 2, self.tags_2d[0])

            numNodes = 10
            for tag_1d in self.tags_1d:
                gmsh.model.geo.mesh.setTransfiniteCurve(
                    tag_1d, numNodes, "Bump", coef=0.25
                )

        gmsh.model.geo.synchronize()
        for d in range(self.dimension + 1):
            gmsh.model.mesh.generate(d)
        for _ in range(n_refinments):
            gmsh.model.mesh.refine()
        # if "-nopopup" not in sys.argv:
        #     gmsh.fltk.run()
        # aka = 0
