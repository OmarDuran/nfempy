import auto_diff as ad
import numpy as np

from basis.element_data import ElementData
from basis.parametric_transformation import transform_lower_to_higher
from geometry.compute_normal import normal
from mesh.topological_queries import find_higher_dimension_neighs
from weak_forms.weak_from import WeakForm


class ContactWeakForm(WeakForm):

    def evaluate_form(self, index_p, index_n, alpha):
        iel_p, iel_n = index_p, index_n

        f_cu_n = self.functions["cu_normal"]
        f_cu_t = self.functions["cu_tangential"]
        f_cr_n = self.functions["cr_normal"]
        f_cr_t = self.functions["cr_tangential"]

        q_c0_space = self.space[0].discrete_spaces["q"]
        p_c1_space = self.space[1].discrete_spaces["p"]

        q_c0_components = q_c0_space.n_comp
        q_c0_data: ElementData = q_c0_space.bc_elements[iel_c0].data

        p_c1_components = p_c1_space.n_comp

        cell = q_c0_data.cell
        dim = cell.dimension
        points, weights = self.space[0].bc_quadrature[dim]
        dim = cell.dimension
        x, jac, det_jac, inv_jac = q_c0_space.bc_elements[iel_c0].evaluate_mapping(
            points
        )

        # Diffusive flux
        # find high-dimension neigh q space
        neigh_list = find_higher_dimension_neighs(
            cell, q_c0_space.dof_map.mesh_topology
        )
        neigh_check_mp = len(neigh_list) > 0
        assert neigh_check_mp
        neigh_cell_id = neigh_list[0][1]
        neigh_cell_index = q_c0_space.id_to_element[neigh_cell_id]
        neigh_element = q_c0_space.elements[neigh_cell_index]
        neigh_cell = neigh_element.data.cell

        # compute trace q space
        mapped_points = transform_lower_to_higher(points, q_c0_data, neigh_element.data)
        _, jac_c0, det_jac_c0, inv_jac_c0 = neigh_element.evaluate_mapping(
            mapped_points
        )
        dq_tr_phi_tab = neigh_element.evaluate_basis(
            mapped_points, jac_c0, det_jac_c0, inv_jac_c0
        )
        dq_facet_index = (
            neigh_cell.sub_cells_ids[cell.dimension].tolist().index(cell.id)
        )
        dq_dof_n_index = np.array(
            neigh_element.data.dof.entity_dofs[cell.dimension][dq_facet_index]
        )

        p_phi_tab = p_c1_space.elements[iel_c1].evaluate_basis(
            points, jac, det_jac, inv_jac
        )
        n_dq_phi = dq_tr_phi_tab[0, :, dq_dof_n_index, 0:dim].shape[0]

        n_q_phi = n_dq_phi
        n_p_phi = p_phi_tab.shape[2]
        n_q_dof = n_q_phi * q_c0_components
        n_p_dof = n_p_phi * p_c1_components

        idx_dof = {
            "q": slice(0, n_q_dof),
            "p": slice(n_q_dof, n_q_dof + n_p_dof),
        }
        n_dof = n_q_dof + n_p_dof

        dim = neigh_cell.dimension
        # compute normal
        n = normal(q_c0_data.mesh, neigh_cell, cell)

        with ad.AutoDiff(alpha) as alpha:
            el_form = np.zeros(n_dof)
            alpha_q = alpha[:, idx_dof["q"]]
            alpha_p = alpha[:, idx_dof["p"]]

            for i, omega in enumerate(weights):
                kappa_n_v = f_kappa(x[i, 0], x[i, 1], x[i, 2])
                delta_v = f_delta(x[i, 0], x[i, 1], x[i, 2])

                dq_h = (dq_tr_phi_tab[0, i, dq_dof_n_index, 0:dim] @ n[0:dim]).reshape(
                    n_dq_phi, 1
                )
                dp_h = p_phi_tab[0, i, :, 0:dim]
                q_h_n = alpha_q @ dq_h
                p_h_c1 = alpha_p @ dp_h

                beta_v = kappa_n_v * (2.0 / delta_v)

                equ_1_integrand = ((1.0 / beta_v) * q_h_n + 0.5 * p_h_c1) @ dq_h.T
                # Robin coupling is a self-adjoint operator of c1 mass conservation
                equ_2_integrand = q_h_n @ dp_h.T

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, idx_dof["q"]] = equ_1_integrand
                multiphysic_integrand[:, idx_dof["p"]] = equ_2_integrand
                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

            r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        return r_el, j_el
