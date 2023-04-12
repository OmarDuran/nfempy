from mesh.mesh import Mesh


class MeshTopology:
    # Entities by dimension or codimension
    # https://defelement.com/ciarlet.html
    def __init__(self, mesh, dimension):

        if mesh.dimension < dimension:
            raise ValueError("MeshTopology:: max dimension available in the mesh is %r" % mesh.dimension)

        self.mesh = mesh
        self.dimension = dimension
        self.entity_maps = []
        self.entity_ids = {}
        self._build_entity_maps()
        self._build_entity_ids()

    def _build_entity_maps(self):

        dim = self.dimension
        for d in range(dim + 1):
            self.entity_maps.append(self.mesh.build_graph(dim, dim - d))

    def _build_entity_maps_on_valid_physical_tags(self):

        dim = self.dimension
        for d in range(dim + 1):
            self.entity_maps.append(self.mesh.build_graph_on_materials(dim, dim - d))

    def _build_entity_ids(self):
        dim = self.dimension
        for d in range(dim + 1):
            self.entity_ids[d] = [
                id
                for id in list(self.entity_maps[d].nodes())
                if self.mesh.cells[id].dimension == d
            ]

    def build_data_on_physical_tags(self):
        self.entity_maps.clear()
        self.entity_ids.clear()
        self._build_entity_maps_on_valid_physical_tags()
        self._build_entity_ids()

    def entity_map_by_codimension(self, codimension):
        dim = self.dimension
        dim_gap = dim - codimension
        return self.entity_maps[dim_gap]

    def entity_map_by_dimension(self, dimension):
        return self.entity_maps[dimension]

    def entities_by_codimension(self, codimension):
        dim = self.dimension
        dim_gap = dim - codimension
        return self.entity_ids[dim_gap]

    def entities_by_dimension(self, dimension):
        if self.dimension < dimension:
            raise ValueError("MeshTopology:: max dimension available is %r" % self.dimension)
        return self.entity_ids[dimension]
