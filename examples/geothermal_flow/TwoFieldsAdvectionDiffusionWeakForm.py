import auto_diff as ad
import numpy as np

from basis.element_data import ElementData
from basis.parametric_transformation import transform_lower_to_higher
from geometry.compute_normal import normal
from topology.topological_queries import find_higher_dimension_neighs
from topology.topological_queries import sub_entity_by_co_dimension
from weak_forms.weak_from import WeakForm


class TwoFieldsAdvectionDiffusionWeakForm(WeakForm):
    def evaluate_form(self, element_index, alpha_n_p_1, alpha_n, t):
        iel = element_index
        if self.space is None or self.functions is None:
            raise ValueError

        q_space = self.space.discrete_spaces["q"]
        u_space = self.space.discrete_spaces["u"]

        f_rhs = self.functions["rhs"]
        f_kappa = self.functions["kappa"]
        f_velocity = self.functions["velocity"]

        q_components = q_space.n_comp
        u_components = u_space.n_comp

        q_data: ElementData = q_space.elements[iel].data
        u_data: ElementData = u_space.elements[iel].data

        cell = q_data.cell
        dim = cell.dimension
        points, weights = self.space.quadrature
        x, jac, det_jac, inv_jac = q_space.elements[iel].evaluate_mapping(points)

        # basis
        dq_h_tab = q_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        du_h_tab = u_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)

        n_dq_h = dq_h_tab.shape[2]
        n_du_h = du_h_tab.shape[2]

        n_q_dof = n_dq_h * q_components
        n_u_dof = n_du_h * u_components

        idx_dof = {
            "q": slice(0, n_q_dof),
            "u": slice(n_q_dof, n_q_dof + n_u_dof),
        }

        n_dof = n_q_dof + n_u_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # Partial local vectorization
        f_val_star = f_rhs(x[:, 0], x[:, 1], x[:, 2])
        du_h_star = det_jac * weights * du_h_tab[0, :, :, 0].T

        # constant directors
        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])

        # # trace of qh
        # codimension = 1
        # sub_entities = sub_entity_by_co_dimension(cell, codimension)
        # sub_entities_q = len(sub_entities) > 0
        # assert sub_entities_q
        #
        # ElementData

        with ad.AutoDiff(alpha_n_p_1) as alpha_n_p_1:
            el_form = np.zeros(n_dof)
            for c in range(u_components):
                b = c + n_q_dof
                e = b + n_u_dof
                el_form[b:e:u_components] -= du_h_star @ f_val_star[c].T

            for i, omega in enumerate(weights):
                xv = x[i]

                # Functions and derivatives at integration point i
                dq_h = dq_h_tab[0, i, :, 0:dim]
                du_h = du_h_tab[0, i, :, 0:dim]
                grad_dq_h = dq_h_tab[1 : dq_h.shape[0] + 1, i, :, 0:dim]
                div_dq_h = np.array(
                    [
                        [
                            np.trace(grad_dq_h[:, j, :]) / det_jac[i]
                            for j in range(n_q_dof)
                        ]
                    ]
                )

                # Dof per field
                alpha_q_n_p_1 = alpha_n_p_1[:, idx_dof["q"]]
                alpha_u_n_p_1 = alpha_n_p_1[:, idx_dof["u"]]
                alpha_u_n = alpha_n[idx_dof["u"]]

                # FEM approximation
                q_h_n_p_1 = alpha_q_n_p_1 @ dq_h
                u_h_n_p_1 = alpha_u_n_p_1 @ du_h
                u_h_n = alpha_u_n @ du_h

                div_q_h = alpha_q_n_p_1 @ div_dq_h.T
                q_h_n_p_1 *= 1.0 / f_kappa(xv[0], xv[1], xv[2])

                # Example of reaction term
                # duh_dt = (u_h_n_p_1 - u_h_n) / delta_t

                equ_1_integrand = (q_h_n_p_1 @ dq_h.T) - (u_h_n_p_1 @ div_dq_h)
                equ_2_integrand = div_q_h @ du_h.T

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, idx_dof["q"]] = equ_1_integrand
                multiphysic_integrand[:, idx_dof["u"]] = equ_2_integrand
                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

            for i, omega in enumerate(weights):
                xv = x[i]

                # Functions and derivatives at integration point i
                dq_h = dq_h_tab[0, i, :, 0:dim]
                du_h = du_h_tab[0, i, :, 0:dim]
                grad_dq_h = dq_h_tab[1 : dq_h.shape[0] + 1, i, :, 0:dim]
                div_dq_h = np.array(
                    [
                        [
                            np.trace(grad_dq_h[:, j, :]) / det_jac[i]
                            for j in range(n_q_dof)
                        ]
                    ]
                )

                # Dof per field
                alpha_q_n_p_1 = alpha_n_p_1[:, idx_dof["q"]]
                alpha_u_n_p_1 = alpha_n_p_1[:, idx_dof["u"]]
                alpha_u_n = alpha_n[idx_dof["u"]]

                # FEM approximation
                q_h_n_p_1 = alpha_q_n_p_1 @ dq_h
                u_h_n_p_1 = alpha_u_n_p_1 @ du_h
                u_h_n = alpha_u_n @ du_h

                div_q_h = alpha_q_n_p_1 @ div_dq_h.T
                q_h_n_p_1 *= 1.0 / f_kappa(xv[0], xv[1], xv[2])

                # Example of reaction term
                # duh_dt = (u_h_n_p_1 - u_h_n) / delta_t

                equ_1_integrand = (q_h_n_p_1 @ dq_h.T) - (u_h_n_p_1 @ div_dq_h)
                equ_2_integrand = div_q_h @ du_h.T

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, idx_dof["q"]] = equ_1_integrand
                multiphysic_integrand[:, idx_dof["u"]] = equ_2_integrand
                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        return r_el, j_el


class TwoFieldsAdvectionDiffusionWeakFormBCRobin(WeakForm):
    def evaluate_form(self, element_index, alpha, t):
        iel = element_index

        f_beta = self.functions["beta"]
        f_gamma = self.functions["gamma"]
        f_c = self.functions["c"]
        f_velocity = self.functions["velocity"]

        q_space = self.space.discrete_spaces["q"]
        q_components = q_space.n_comp
        q_data: ElementData = q_space.bc_elements[iel].data

        cell = q_data.cell
        points, weights = self.space.bc_quadrature
        dim = cell.dimension
        x, jac, det_jac, inv_jac = q_space.bc_elements[iel].evaluate_mapping(points)

        # find high-dimension neigh q space
        neigh_list = find_higher_dimension_neighs(cell, q_space.dof_map.mesh_topology)
        neigh_check_mp = len(neigh_list) > 0
        assert neigh_check_mp
        neigh_cell_id = neigh_list[0]
        neigh_cell_index = q_space.id_to_element[neigh_cell_id]
        neigh_element = q_space.elements[neigh_cell_index]
        neigh_cell = neigh_element.data.cell

        # compute trace q space
        mapped_points = transform_lower_to_higher(points, q_data, neigh_element.data)
        _, jac_c0, det_jac_c0, inv_jac_c0 = neigh_element.evaluate_mapping(
            mapped_points
        )
        dq_tr_phi_tab = neigh_element.evaluate_basis(
            mapped_points, jac_c0, det_jac_c0, inv_jac_c0
        )
        dq_facet_index = (
            neigh_cell.sub_cells_ids[cell.dimension].tolist().index(cell.id)
        )
        dq_dof_n_index = neigh_element.data.dof.entity_dofs[cell.dimension][
            dq_facet_index
        ]

        n_dq_phi = dq_tr_phi_tab[0, :, dq_dof_n_index, 0:dim].shape[0]
        n_q_dof = n_dq_phi * q_components

        idx_dof = {
            "q": slice(0, n_q_dof),
        }

        n_dof = n_q_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        alpha_q = alpha

        # compute normal
        n = normal(q_data.mesh, neigh_cell, cell)
        for c in range(q_components):
            b = c
            e = b + n_q_dof

            res_block_q = np.zeros(n_dq_phi)
            jac_block_q = np.zeros((n_dq_phi, n_dq_phi))
            dim = neigh_cell.dimension
            for i, omega in enumerate(weights):
                beta_v = f_beta(x[i, 0], x[i, 1], x[i, 2])
                gamma_v = f_gamma(x[i, 0], x[i, 1], x[i, 2])
                c_v = f_c(x[i, 0], x[i, 1], x[i, 2])

                dq_h = dq_tr_phi_tab[0, i, dq_dof_n_index, 0:dim] @ n[0:dim]
                # This sign may be needed in 1d computations because BC orientation
                # However in pure Hdiv functions in 2d and 3d it is not needed
                bc_sign = np.sign(dq_h)[0]
                q_h_n = alpha_q @ dq_h
                bc_residual_equ = (1.0 / beta_v) * (
                    bc_sign * q_h_n + beta_v * c_v - gamma_v * c_v
                )
                res_block_q += det_jac[i] * omega * bc_residual_equ * dq_h
                jac_block_q += (
                    det_jac[i] * omega * (1.0 / beta_v) * np.outer(bc_sign * dq_h, dq_h)
                )

            r_el[b:e:q_components] += res_block_q
            j_el[b:e:q_components, b:e:q_components] += jac_block_q

        return r_el, j_el
