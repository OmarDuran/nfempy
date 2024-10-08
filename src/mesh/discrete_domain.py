import gmsh
import numpy as np

from topology.domain import Domain


class DiscreteDomain:
    def __init__(self, dimension):
        self.dimension = dimension
        self.domain: Domain = None
        self.lc = None
        self.mesh_arguments: dict = {}
        self.d0_stride = 0
        self.d1_stride = 0
        self.d2_stride = 0
        self.d3_stride = 0

    def __compute_entities_strides(self):

        strides = [0, 0, 0, 0]
        for d in range(len(self.domain.shapes)):
            strides[d] = len(self.domain.shapes[d])
            if strides[d] == 0:
                break
        strides = np.add.accumulate(strides)
        self.d0_stride = strides[0]
        self.d1_stride = strides[1]
        self.d2_stride = strides[2]
        self.d3_stride = strides[3]

    def __transfer_vertices(self):
        for vertex in self.domain.shapes[0]:
            if len(vertex.immersed_shapes) > 0:
                continue
            gmsh.model.occ.addPoint(
                vertex.point[0],
                vertex.point[1],
                vertex.point[2],
                self.lc,
                self.stride_tag(0, vertex.tag),
            )
        gmsh.model.occ.synchronize()

    def __transfer_curves(self):
        for curve in self.domain.shapes[1]:
            if curve.composite:
                continue
            shapes_with_same_dimension = [
                shape
                for shape in curve.immersed_shapes
                if shape.dimension == curve.dimension
            ]
            if len(shapes_with_same_dimension) > 0:
                continue
            tag = self.stride_tag(1, curve.tag)
            b = self.stride_tag(0, curve.boundary_shapes[0].tag)
            e = self.stride_tag(0, curve.boundary_shapes[1].tag)
            gmsh.model.occ.addLine(b, e, tag)
        gmsh.model.occ.synchronize()

    def __transfer_surfaces(self):
        # for curve in self.domain.shapes[1]:
        #     if curve.composite:
        #         continue
        #     shapes_with_same_dimension = [
        #         shape
        #         for shape in curve.immersed_shapes
        #         if shape.dimension == curve.dimension
        #     ]
        #     if len(shapes_with_same_dimension) > 0:
        #         continue
        #     tag = self.stride_tag(1, curve.tag)
        #     b = self.stride_tag(0, curve.boundary_shapes[0].tag)
        #     e = self.stride_tag(0, curve.boundary_shapes[1].tag)
        #     gmsh.model.occ.addLine(b, e, tag)
        # gmsh.model.occ.synchronize()

        for surface in self.domain.shapes[2]:
            tag = self.stride_tag(2, surface.tag)
            if not surface.composite:
                # print("creating surface tag: ", surface.tag)
                wire = surface.boundary_shapes[0]
                wire.orient_immersed_edges()
                loop_tags = [
                    self.stride_tag(1, shape.tag) for shape in wire.immersed_shapes
                ]
                loop_tags = [
                    loop_tags[i] * sign for i, sign in enumerate(wire.orientation)
                ]
                gmsh.model.occ.addCurveLoop(loop_tags, tag)
                gmsh.model.occ.addPlaneSurface([tag], tag)
        gmsh.model.occ.synchronize()

    def stride_tag(self, dim, tag):
        tag_stride = [
            self.d0_stride + tag + 1,
            self.d1_stride + tag + 1,
            self.d2_stride + tag + 1,
            self.d3_stride + tag + 1,
        ]
        return tag_stride[dim]

    def __physical_group_vertices(self):
        physical_tags_0d = np.unique(
            [
                shape.physical_tag
                for shape in self.domain.shapes[0]
                if shape.physical_tag is not None
            ]
        )
        for physical_tag in physical_tags_0d:
            filtered_shapes = [
                shape
                for shape in self.domain.shapes[0]
                if shape.physical_tag == physical_tag
            ]
            if len(filtered_shapes) > 0:
                v_tags = [self.stride_tag(0, shape.tag) for shape in filtered_shapes]
                gmsh.model.addPhysicalGroup(0, v_tags, physical_tag)
        gmsh.model.occ.synchronize()

    def __physical_group_curves(self):
        physical_tags_1d = np.unique(
            [
                shape.physical_tag
                for shape in self.domain.shapes[1]
                if shape.physical_tag is not None
            ]
        )
        for physical_tag in physical_tags_1d:
            filtered_shapes = [
                shape
                for shape in self.domain.shapes[1]
                if shape.physical_tag == physical_tag
            ]
            curve_tags = []
            for curve in filtered_shapes:
                if curve.composite:
                    continue
                shapes_with_same_dimension = [
                    shape
                    for shape in curve.immersed_shapes
                    if shape.dimension == curve.dimension
                ]
                if len(shapes_with_same_dimension) > 0:
                    continue
                curve_tags.append(self.stride_tag(1, curve.tag))
            gmsh.model.addPhysicalGroup(1, curve_tags, physical_tag)

            target_physical_tags, curve_controls = self.mesh_arguments.get(
                "curves_refinement", ([], None)
            )
            if physical_tag in target_physical_tags and curve_controls is not None:
                numNodes = curve_controls.get("n_points", 5)
                mesh_type = curve_controls.get("meshType", "Bump")
                coef = curve_controls.get("coef", 0.1)
                for tag_1d in curve_tags:
                    gmsh.model.mesh.setTransfiniteCurve(
                        tag_1d, numNodes, mesh_type, coef=coef
                    )
        gmsh.model.occ.synchronize()

    def __physical_group_surfaces(self):
        physical_tags_2d = np.unique(
            [
                shape.physical_tag
                for shape in self.domain.shapes[2]
                if shape.physical_tag is not None
            ]
        )
        for physical_tag in physical_tags_2d:
            filtered_shapes = [
                shape
                for shape in self.domain.shapes[2]
                if shape.physical_tag == physical_tag
            ]
            surface_tags = []
            for surface in filtered_shapes:
                surface_tags.append(self.stride_tag(2, surface.tag))
            gmsh.model.addPhysicalGroup(2, surface_tags, physical_tag)
        gmsh.model.occ.synchronize()

    def __embed_vertices_in_curves(self):
        for curve in self.domain.shapes[1]:
            if curve.composite:
                continue
            shapes_with_same_dimension = [
                shape
                for shape in curve.immersed_shapes
                if shape.dimension == curve.dimension
            ]
            if len(shapes_with_same_dimension) > 0:
                continue

            tags_0d = []
            for shape in curve.immersed_shapes:  # only vertices are embedded in curves
                tags_0d = tags_0d + [self.stride_tag(0, shape.tag)]
            tags_0d = list(np.unique(tags_0d))
            gmsh.model.mesh.embed(0, tags_0d, 1, self.stride_tag(1, curve.tag))

    def __embed_curves_in_surfaces(self):
        # tags_2d = [shape.tag for shape in self.domain.shapes[2]]
        for surface in self.domain.shapes[2]:
            if surface.composite:
                continue
            # print("surface tag: ", surface.tag)
            tags_0d = []
            tags_1d = []
            shapes_c1 = [shape for shape in surface.immersed_shapes]
            for shape_c1 in shapes_c1:
                if len(shape_c1.immersed_shapes) > 0:
                    tags_1d = tags_1d + [
                        self.stride_tag(1, curve.tag)
                        for curve in shape_c1.immersed_shapes
                    ]
                else:
                    tags_1d = tags_1d + [self.stride_tag(1, shape_c1.tag)]
                tags_0d = tags_0d + [
                    self.stride_tag(0, vertex.tag)
                    for vertex in shape_c1.boundary_shapes
                ]

            tags_0d = list(np.unique(tags_0d))
            tags_1d = list(np.unique(tags_1d))
            gmsh.model.mesh.embed(0, tags_0d, 2, self.stride_tag(2, surface.tag))
            gmsh.model.mesh.embed(1, tags_1d, 2, self.stride_tag(2, surface.tag))

    def convert_domain_to_occ_description(self):
        self.__compute_entities_strides()
        dimension = 0
        for dimension in range(len(self.domain.shapes)):
            if len(self.domain.shapes[dimension]) == 0:
                break

        # transfer vertices
        self.__transfer_vertices()

        # transfer curves
        if dimension > 0:
            self.__transfer_curves()

        # transfer surfaces
        if dimension > 1:
            self.__transfer_surfaces()

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
        self.__physical_group_vertices()
        if dimension > 0:
            self.__physical_group_curves()

        if dimension > 1:
            self.__physical_group_surfaces()

        if dimension > 2:
            for volume in self.domain.shapes[3]:
                if volume.physical_tag is not None:
                    gmsh.model.addPhysicalGroup(
                        3, [volume_stride + volume.tag + 1], volume.physical_tag
                    )

        gmsh.model.occ.synchronize()

        # embed entities
        if dimension > 0:
            self.__embed_vertices_in_curves()

        # embed entities
        if dimension > 1:
            self.__embed_curves_in_surfaces()

        # embed entities
        if dimension > 2:
            tags_3d = [shape.tag for shape in self.domain.shapes[3]]
            for volume in self.domain.shapes[3]:
                if volume.composite:
                    continue
                # print("volume tag: ", volume.tag)
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

    def generate_mesh(self, mesh_arguments: dict = {}):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        self.mesh_arguments = mesh_arguments
        self.lc = self.mesh_arguments.get("lc", np.inf)
        n_refinements = self.mesh_arguments.get("n_refinements", 0)

        self.convert_domain_to_occ_description()
        for d in range(self.dimension + 1):
            gmsh.model.mesh.generate(d)
        for _ in range(n_refinements):
            gmsh.model.mesh.refine()

        # if "-nopopup" not in sys.argv:
        #     gmsh.fltk.run()
        # aka = 0

    def write_mesh(self, file_name):
        gmsh.write(file_name)
        gmsh.finalize()
