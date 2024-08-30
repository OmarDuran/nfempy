import auto_diff as ad
import numpy as np

from basis.element_data import ElementData
from basis.parametric_transformation import transform_lower_to_higher
from geometry.compute_normal import normal
from mesh.topological_queries import find_higher_dimension_neighs
from weak_forms.weak_from import WeakForm

class LaplaceDualWeakForm(WeakForm):
    def evaluate_form(self, element_index, alpha):
        iel = element_index
        if self.space is None or self.functions is None:
            raise ValueError

        u_space = self.space.discrete_spaces["u"]
        p_space = self.space.discrete_spaces["p"]

        f_rhs = self.functions["rhs"]
        f_kappa = self.functions["kappa"]

        u_components = u_space.n_comp
        p_components = p_space.n_comp
        q_data: ElementData = u_space.elements[iel].data

        cell = q_data.cell
        dim = cell.dimension
        points, weights = self.space.quadrature[dim]
        x, jac, det_jac, inv_jac = u_space.elements[iel].evaluate_mapping(points)

        # basis
        u_phi_tab = u_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        p_phi_tab = p_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)

        n_u_phi = u_phi_tab.shape[2]
        n_p_phi = p_phi_tab.shape[2]
        n_q_dof = n_u_phi * u_components
        n_p_dof = n_p_phi * p_components
        n_dof = n_q_dof + n_p_dof

        idx_dof = {
            "q": slice(0, n_q_dof),
            "p": slice(n_q_dof, n_q_dof + n_p_dof),
        }

        alpha = np.zeros(n_dof)
        # Partial local vectorization
        f_val_star = f_rhs(x[:, 0], x[:, 1], x[:, 2])
        phi_s_star = det_jac * weights * p_phi_tab[0, :, :, 0].T

        with ad.AutoDiff(alpha) as alpha:
            el_form = np.zeros(n_dof)
            for c in range(p_components):
                b = c + n_q_dof
                e = b + n_p_dof
                el_form[b:e:p_components] -= phi_s_star @ f_val_star[c].T

            for i, omega in enumerate(weights):
                xv = x[i]

                # Functions and derivatives at integration point i
                psi_h = u_phi_tab[0, i, :, 0:dim]
                w_h = p_phi_tab[0, i, :, 0:dim]

                grad_psi_h = u_phi_tab[1 : u_phi_tab.shape[0] + 1, i, :, 0:dim]
                div_psi_h = np.array(
                    [np.trace(grad_psi_h, axis1=0, axis2=2) / det_jac[i]]
                )

                u_h = alpha[:, idx_dof["q"]] @ psi_h
                u_h *= 1.0 / f_kappa(xv[0], xv[1], xv[2])

                p_h = alpha[:, idx_dof["p"]] @ w_h
                div_uh = alpha[:, idx_dof["q"]] @ div_psi_h.T

                equ_1_integrand = (u_h @ psi_h.T) - (p_h @ div_psi_h)
                equ_2_integrand = div_uh @ w_h.T

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, 0:n_q_dof:1] = equ_1_integrand
                multiphysic_integrand[:, n_q_dof:n_dof:1] = equ_2_integrand
                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        return r_el, j_el


class LaplaceDualWeakFormBCDirichlet(WeakForm):
    def evaluate_form(self, element_index, alpha):
        iel = element_index
        p_D = self.functions["p"]

        u_space = self.space.discrete_spaces["u"]
        u_components = u_space.n_comp
        u_data: ElementData = u_space.bc_elements[iel].data

        cell = u_data.cell
        dim = cell.dimension
        points, weights = self.space.bc_quadrature[dim]
        x, jac, det_jac, inv_jac = u_space.bc_elements[iel].evaluate_mapping(points)

        q_phi_tab = u_space.bc_elements[iel].evaluate_basis(
            points, jac, det_jac, inv_jac
        )

        n_q_phi = q_phi_tab.shape[2]
        n_q_dof = n_q_phi * u_components

        n_dof = n_q_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # find high-dimension neigh
        neigh_list = find_higher_dimension_neighs(cell, u_space.dof_map.mesh_topology)
        neigh_check_q = len(neigh_list) > 0
        assert neigh_check_q
        neigh_cell_id = neigh_list[0][1]
        neigh_cell_index = u_space.id_to_element[neigh_cell_id]
        neigh_element = u_space.elements[neigh_cell_index]
        neigh_cell = neigh_element.data.cell

        # compute trace space
        mapped_points = transform_lower_to_higher(points, u_data, neigh_element.data)
        _, jac_c0, det_jac_c0, inv_jac_c0 = neigh_element.evaluate_mapping(
            mapped_points
        )
        q_tr_phi_tab = neigh_element.evaluate_basis(
            mapped_points, jac_c0, det_jac_c0, inv_jac_c0
        )
        facet_index = neigh_cell.sub_cells_ids[cell.dimension].tolist().index(cell.id)
        dof_n_index = neigh_element.data.dof.entity_dofs[cell.dimension][facet_index]

        # compute normal
        n = normal(u_data.mesh, neigh_cell, cell)
        for c in range(u_components):
            b = c
            e = b + n_q_dof

            res_block_q = np.zeros(n_q_phi)
            dim = neigh_cell.dimension
            for i, omega in enumerate(weights):
                p_D_v = p_D(x[i, 0], x[i, 1], x[i, 2])
                if cell.dimension == 0:
                    phi = q_tr_phi_tab[0, i, dof_n_index, 0:dim] @ np.array([1.0])
                else:
                    phi = q_tr_phi_tab[0, i, dof_n_index, 0:dim] @ n[0:dim]
                res_block_q += det_jac[i] * omega * p_D_v[c] * phi

            r_el[b:e:u_components] += res_block_q

        return r_el, j_el
