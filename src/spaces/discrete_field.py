import time
from functools import partial

from basis.finite_element import FiniteElement
from spaces.dof_map import DoFMap
from topology.mesh_topology import MeshTopology


class DiscreteField:
    # The discrete variable representation
    def __init__(
        self, dimension, n_components, family, k_order, mesh, integration_oder=0
    ):
        self.dimension = dimension
        self.n_comp = n_components
        self.family = family
        self.k_order = k_order
        self.integration_oder = integration_oder
        self.name = "unnamed"
        self.mesh_topology = MeshTopology(mesh, dimension)
        self.discontinuous = False
        self.physical_tag_filter = True
        self.element_type = None
        self.elements = []
        self.element_ids = []

    def set_name(self, name):
        self.name = name

    def make_discontinuous(self):
        self.discontinuous = True

    def build_dof_map(self, only_on_physical_tags=True):
        st = time.time()
        if only_on_physical_tags:
            self.mesh_topology.build_data_on_physical_tags()
        else:
            self.physical_tag_filter = False

        self.element_type = FiniteElement.type_by_dimension(self.dimension)
        basis_family = FiniteElement.basis_family(self.family)
        basis_variant = FiniteElement.basis_variant()
        if self.dimension == 0:
            basis_family = FiniteElement.basis_family("Lagrange")
            self.k_order = 0
        if self.dimension == 1 and self.family in ["RT","BDM"]:
            basis_family = FiniteElement.basis_family("Lagrange")
        self.dof_map = DoFMap(
            self.mesh_topology,
            basis_family,
            self.element_type,
            self.k_order,
            basis_variant,
            discontinuous=self.discontinuous,
        )
        self.dof_map.set_topological_dimension(self.dimension)
        self.dof_map.build_entity_maps(n_components=self.n_comp)
        self.n_dof = self.dof_map.dof_number()
        et = time.time()
        elapsed_time = et - st
        print("Field:: DoFMap construction time:", elapsed_time, "seconds")

    def build_elements(self):
        st = time.time()

        self.element_ids = self.mesh_topology.entities_by_dimension(self.dimension)
        if self.physical_tag_filter:
            mesh = self.mesh_topology.mesh
            self.element_ids = [
                id for id in self.element_ids if mesh.cells[id].material_id != None
            ]

        self.elements = list(
            map(
                partial(
                    FiniteElement,
                    mesh=self.mesh_topology.mesh,
                    k_order=self.k_order,
                    family=self.family,
                    discontinuous=self.discontinuous,
                    integration_oder=self.integration_oder,
                ),
                self.element_ids,
            )
        )
        et = time.time()
        elapsed_time = et - st
        n_d_cells = len(self.elements)
        print("Field:: Number of processed elements:", n_d_cells)
        print("Field:: Element construction time:", elapsed_time, "seconds")
