from mesh.mesh import Mesh

class MeshTopology:

    def __init__(self, mesh):
        self.mesh = mesh
        self.entity_maps = []
        self.entity_ids = {}
        self._build_entity_maps()
        self._build_entity_ids()

    def _build_entity_maps(self):
        dim = self.mesh.dimension
        for d in range(dim+1):
            self.entity_maps.append(self.mesh.build_graph(dim, dim - d))

    def _build_entity_ids(self):
        dim = self.mesh.dimension
        for d in range(dim+1):
            self.entity_ids[d] = [
                id
                for id in list(self.entity_maps[d].nodes())
                if self.mesh.cells[id].dimension == d
            ]

    def entity_map_by_codimension(self, codimension):
        dim = self.mesh.dimension
        dim_gap = dim - codimension
        return self.entity_maps[dim_gap]

    def entity_map_by_dimension(self, dimension):
        return self.entity_maps[dimension]

    def entities_by_codimension(self, codimension):
        dim = self.mesh.dimension
        dim_gap = dim - codimension
        return self.entity_ids[dim_gap]

    def entities_dimension(self, dimension):
        return self.entity_ids[dimension]