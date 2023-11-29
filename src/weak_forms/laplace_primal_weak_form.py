import auto_diff as ad
import numpy as np
from auto_diff.vecvalder import VecValDer

from basis.element_data import ElementData
from weak_forms.weak_from import WeakForm

import basix
from basix import CellType

class LaplacePrimalWeakForm(WeakForm):
    def evaluate_form(self, element_index, alpha):
        i = element_index
        if self.space is None or self.functions is None:
            raise ValueError

        p_space = self.space.discrete_spaces["p"]

        f_rhs = self.functions["rhs"]
        f_kappa = self.functions["kappa"]

        p_components = p_space.n_comp
        p_data: ElementData = p_space.elements[i].data

        cell = p_data.cell
        dim = p_data.dimension
        points = p_data.quadrature.points
        weights = p_data.quadrature.weights
        x = p_data.mapping.x
        det_jac = p_data.mapping.det_jac
        inv_jac = p_data.mapping.inv_jac
        p_phi_tab = p_data.basis.phi

        n_p_phi = p_phi_tab.shape[2]
        n_p_dof = n_p_phi * p_components

        n_dof = n_p_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # Partial local vectorization
        f_val_star = f_rhs(x[:, 0], x[:, 1], x[:, 2])
        phi_s_star = det_jac * weights * p_phi_tab[0, :, :, 0].T

        # constant directors
        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])
        with ad.AutoDiff(alpha) as alpha:
            el_form = np.zeros(n_dof)
            for c in range(p_components):
                b = c
                e = b + n_dof
                el_form[b:e:p_components] -= phi_s_star @ f_val_star[c]

            for i, omega in enumerate(weights):
                xv = x[i]
                if dim == 2:
                    inv_jac_m = np.vstack((inv_jac[i] @ e1, inv_jac[i] @ e2))
                else:
                    inv_jac_m = np.vstack(
                        (inv_jac[i] @ e1, inv_jac[i] @ e2, inv_jac[i] @ e3)
                    )
                grad_phi = (
                    inv_jac_m @ p_phi_tab[1 : p_phi_tab.shape[0] + 1, i, :, 0]
                ).T
                grad_uh = alpha @ grad_phi
                grad_uh *= f_kappa(xv[0], xv[1], xv[2])
                energy_h = (grad_phi @ grad_uh.T).reshape((n_dof,))
                el_form += det_jac[i] * omega * energy_h

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        return r_el, j_el


class LaplacePrimalWeakFormBCDirichlet(WeakForm):
    def evaluate_form(self, element_index, alpha):
        p_D = self.functions["p"]

        i = element_index
        p_space = self.space.discrete_spaces["p"]
        p_components = p_space.n_comp
        p_data: ElementData = p_space.bc_elements[i].data

        cell = p_data.cell
        points = p_data.quadrature.points
        weights = p_data.quadrature.weights
        x = p_data.mapping.x
        det_jac = p_data.mapping.det_jac
        inv_jac = p_data.mapping.inv_jac

        p_phi_tab = p_data.basis.phi
        n_p_phi = p_phi_tab.shape[2]
        n_p_dof = n_p_phi * p_components

        n_dof = n_p_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # compute trace space and compare matrices

        # destination indexes
        # find high-dimension neigh
        entity_map = p_space.dof_map.mesh_topology.entity_map_by_dimension(
            cell.dimension
        )
        neigh_list = list(entity_map.predecessors(cell.id))
        neigh_check_q = len(neigh_list) > 0
        assert neigh_check_q

        neigh_cell_id = neigh_list[0]
        neigh_cell_index = p_space.id_to_element[neigh_cell_id]
        neigh_element = p_space.elements[neigh_cell_index]
        neigh_cell = neigh_element.data.cell

        # destination indexes
        dest_neigh = p_space.dof_map.destination_indices(neigh_cell_id)
        dest_p = p_space.dof_map.bc_destination_indices(neigh_cell_id, cell.id)

        # compute trace space
        facet_index = neigh_cell.sub_cells_ids[cell.dimension].tolist().index(cell.id)
        vertices = basix.geometry(CellType.triangle)
        facet_sub_entities = [
            basix.cell.sub_entity_connectivity(CellType.triangle)[cell.dimension][
                facet_index
            ][d]
            for d in range(cell.dimension + 1)
        ]
        facet_nodes = facet_sub_entities[0]
        # mapped_points = np.array(
        #     [
        #         vertices[facet_nodes[0]] * (1 - x - y)
        #         + vertices[facet_nodes[1]] * x
        #         + vertices[facet_nodes[2]] * y
        #         for x, y in points
        #     ]
        # )
        mapped_points = np.array(
            [
                vertices[facet_nodes[0]] * (1 - x)
                + vertices[facet_nodes[1]] * x
                for x in points
            ]
        )
        el_dofs = neigh_element.data.dof.entity_dofs
        facet_dofs = [
            el_dofs[d][i]
            for d in range(cell.dimension + 1)
            for i in facet_sub_entities[d]
        ]
        dof_p_index = [sub_dof for dof in facet_dofs if len(dof) != 0 for sub_dof in dof]
        p_tr_phi_tab = neigh_element.evaluate_basis(mapped_points, False)
        tr_phi_tab = p_tr_phi_tab[0, :, dof_p_index, 0]

        dest_c = self.space.bc_destination_indexes(i)

        # local blocks
        beta = 1.0e12
        jac_block_p = np.zeros((n_p_phi, n_p_phi))
        for i, omega in enumerate(weights):
            phi = p_phi_tab[0, i, :, 0]
            jac_block_p += beta * det_jac[i] * omega * np.outer(phi, phi)

        for c in range(p_components):
            b = c
            e = b + n_p_dof

            res_block_p = np.zeros(n_p_phi)
            for i, omega in enumerate(weights):
                phi = p_phi_tab[0, i, :, 0]
                p_D_v = p_D(x[i, 0], x[i, 1], x[i, 2])
                res_block_p -= beta * det_jac[i] * omega * p_D_v[c] * phi

            r_el[b:e:p_components] += res_block_p
            j_el[b:e:p_components, b:e:p_components] += jac_block_p

        return r_el, j_el
