import multiprocessing
import time
from functools import partial

from joblib import Parallel, delayed, wrap_non_picklable_objects

from basis.element_family import basis_variant, family_by_name
from basis.element_type import type_by_dimension
from basis.finite_element import FiniteElement
from spaces.dof_map import DoFMap
from mesh.mesh_topology import MeshTopology


class DiscreteSpace:
    # The discrete space representation
    def __init__(
        self, dimension, n_components, family, k_order, mesh, integration_order=0
    ):
        self.dimension = dimension
        self.n_comp = n_components
        self.generator_family = family_by_name(family)
        self.element_type = None
        self.bc_element_type = None
        self.k_order = k_order
        self.integration_order = integration_order

        self.name = "Unnamed"
        self.mesh_topology = MeshTopology(mesh, dimension)
        self.discontinuous = False
        self.elements = []
        self.element_ids = []
        self.id_to_element = {}

        self.bc_elements = []
        self.bc_element_ids = []
        self.id_to_bc_element = {}

    @property
    def family(self):
        vector_basis_q = self.generator_family in [
            family_by_name("RT"),
            family_by_name("BDM"),
            family_by_name("N1E"),
            family_by_name("N2E"),
        ]
        if (self.dimension == 1 and vector_basis_q) or (self.dimension == 0):
            return family_by_name("Lagrange")
        else:
            return self.generator_family

    def set_name(self, name):
        self.name = name

    def make_discontinuous(self):
        self.discontinuous = True

    def build_structures(self, physical_tags=[None]):
        self._build_dof_map(physical_tags)
        self._build_elements(physical_tags)

    def build_boundary_structures(self, bc_physical_tags=[None]):
        if len(bc_physical_tags) == 0:
            return
        self._build_bc_elements(bc_physical_tags)

    def _build_dof_map(self, physical_tags=[], timing_q=False):
        if timing_q:
            st = time.time()
        self.mesh_topology.build_data_on_pysical_tags(physical_tags)
        self.element_type = type_by_dimension(self.dimension)
        # basis_family = self.family
        # if self.dimension == 0:
        #     self.family = basis_family = family_by_name("Lagrange")
        #     self.k_order = 0
        # if self.dimension == 1 and self.family in [
        #     family_by_name("RT"),
        #     family_by_name("BDM"),
        #     family_by_name("N1E"),
        #     family_by_name("N2E"),
        # ]:
        #     basis_family = family_by_name("Lagrange")
        self.dof_map = DoFMap(
            self.mesh_topology,
            self.family,
            self.element_type,
            self.k_order,
            basis_variant(),
            discontinuous=self.discontinuous,
        )
        self.dof_map.set_topological_dimension(self.dimension)
        self.dof_map.build_entity_maps(n_components=self.n_comp)
        self.n_dof = self.dof_map.dof_number()
        if timing_q:
            et = time.time()
            elapsed_time = et - st
            print(
                self.name + "_fe_space:: DoFMap construction time:",
                elapsed_time,
                "seconds",
            )

    def _build_elements(self, timing_q=False, parallel_run_q=False):
        if timing_q:
            st = time.time()

        self.element_ids = self.mesh_topology.entities_by_dimension(self.dimension)
        if parallel_run_q:
            num_cores = multiprocessing.cpu_count()
            # batch_size = round(len(self.element_ids) / num_cores)

            @wrap_non_picklable_objects
            def task_create_element(
                cell_id, family, k_order, mesh, discontinuous, integration_order
            ):
                return FiniteElement(
                    cell_id=cell_id,
                    family=self.generator_family,
                    k_order=self.k_order,
                    mesh=self.mesh_topology.mesh,
                    discontinuous=self.discontinuous,
                    integration_order=self.integration_order,
                )

            self.elements = Parallel(
                n_jobs=num_cores, backend="threading", batch_size="auto", verbose=10
            )(
                delayed(task_create_element)(
                    cell_id=id,
                    family=self.generator_family,
                    k_order=self.k_order,
                    mesh=self.mesh_topology.mesh,
                    discontinuous=self.discontinuous,
                    integration_order=self.integration_order,
                )
                for id in self.element_ids
            )
        else:
            self.elements = list(
                map(
                    partial(
                        FiniteElement,
                        family=self.generator_family,
                        k_order=self.k_order,
                        mesh=self.mesh_topology.mesh,
                        discontinuous=self.discontinuous,
                        integration_order=self.integration_order,
                    ),
                    self.element_ids,
                )
            )

        self.id_to_element = dict(zip(self.element_ids, range(len(self.element_ids))))
        if timing_q:
            et = time.time()
            elapsed_time = et - st
            print(
                self.name + "_fe_space:: Elements construction time:",
                elapsed_time,
                "seconds",
            )
        print(self.name + "_fe_space:: Number elements:", len(self.elements))

    def _build_bc_elements(self, b_physical_tags, timing_q=False):
        if timing_q:
            st = time.time()

        self.bc_element_ids = self.mesh_topology.entities_by_dimension(
            self.dimension - 1
        )
        mesh = self.mesh_topology.mesh
        self.bc_element_ids = [
            id
            for id in self.bc_element_ids
            if mesh.cells[id].material_id in b_physical_tags
        ]

        bc_discontinuous = self.discontinuous
        bc_k_order = self.k_order
        bc_familiy = self.family

        if self.dimension - 1 < 2:  # it implies traces of H(div) and H(curl) elements
            bc_familiy = family_by_name("Lagrange")
            if self.dimension - 1 < 1:
                bc_k_order = 0

        if family_by_name("BDM") == self.family:
            bc_k_order = self.k_order
            bc_familiy = family_by_name("Lagrange")
            bc_discontinuous = True

        if family_by_name("RT") == self.family:
            bc_k_order = self.k_order - 1
            bc_familiy = family_by_name("Lagrange")
            bc_discontinuous = True

        self.bc_elements = list(
            map(
                partial(
                    FiniteElement,
                    family=bc_familiy,
                    k_order=bc_k_order,
                    mesh=self.mesh_topology.mesh,
                    discontinuous=bc_discontinuous,
                    integration_order=self.integration_order,
                ),
                self.bc_element_ids,
            )
        )

        self.id_to_bc_element = dict(
            zip(self.bc_element_ids, range(len(self.bc_element_ids)))
        )
        if timing_q:
            et = time.time()
            elapsed_time = et - st
            print(
                self.name + "_fe_space:: Boundary Elements construction time:",
                elapsed_time,
                "seconds",
            )
        print(self.name + "_fe_space:: Number bc elements:", len(self.bc_elements))
