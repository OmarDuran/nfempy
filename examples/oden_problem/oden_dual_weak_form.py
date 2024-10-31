import auto_diff as ad
import numpy as np

from basis.element_data import ElementData
from weak_forms.weak_from import WeakForm
from basis.parametric_transformation import transform_lower_to_higher
from geometry.compute_normal import normal
from mesh.topological_queries import find_higher_dimension_neighs


class OdenDualWeakForm(WeakForm):
    def evaluate_form(self, element_index, alpha):
        iel = element_index
        if self.space is None or self.functions is None:
            raise ValueError

        q_space = self.space.discrete_spaces["q"]
        u_space = self.space.discrete_spaces["u"]

        f_rhs = self.functions["rhs"]

        q_components = q_space.n_comp
        u_components = u_space.n_comp

        q_data: ElementData = q_space.elements[iel].data
        u_data: ElementData = u_space.elements[iel].data

        cell = q_data.cell
        dim = cell.dimension
        points, weights = self.space.quadrature[dim]
        x, jac, det_jac, inv_jac = q_space.elements[iel].evaluate_mapping(points)

        # basis
        q_phi_tab = q_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        u_phi_tab = u_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        #
        n_q_phi = q_phi_tab.shape[2]
        n_u_phi = u_phi_tab.shape[2]
        n_q_dof = n_q_phi * q_components
        n_u_dof = n_u_phi * u_components
        n_dof = n_q_dof + n_u_dof

        # R.H.S
        f_val_star = f_rhs(x[:, 0], x[:, 1], x[:, 2])
        phi_s_star = det_jac * weights * u_phi_tab[0, :, :, 0].T

        # constant directors
        with ad.AutoDiff(alpha) as alpha:
            el_form = np.zeros(n_dof)
            for c in range(u_components):
                b = c + n_q_dof
                e = b + n_u_dof
                el_form[b:e:u_components] -= np.ravel(phi_s_star @ f_val_star[c].T)

                for i, omega in enumerate(weights):
                    xv = x[i]
                    qh = alpha[:, 0:n_q_dof:1] @ q_phi_tab[0, i, :, 0:dim]
                    # qh *= 1.0 / f_kappa(xv[0], xv[1], xv[2])
                    uh = alpha[:, n_q_dof:n_dof:1] @ u_phi_tab[0, i, :, 0:dim]
                    grad_qh = q_phi_tab[1 : q_phi_tab.shape[0] + 1, i, :, 0:dim]
                    div_vh = np.array(
                        [
                            [
                                np.trace(grad_qh[:, j, :]) / det_jac[i]
                                for j in range(n_q_dof)
                            ]
                        ]
                    )
                    div_qh = alpha[:, 0:n_q_dof:1] @ div_vh.T

                    equation_1 = (qh @ q_phi_tab[0, i, :, 0:dim].T) - (uh @ div_vh)
                    equation_2 = (
                        div_qh @ u_phi_tab[0, i, :, 0:dim].T
                        + uh @ u_phi_tab[0, i, :, 0:dim].T
                    )
                    multiphysic_integrand = np.zeros((1, n_dof))
                    multiphysic_integrand[:, 0:n_q_dof:1] = equation_1
                    multiphysic_integrand[:, n_q_dof:n_dof:1] = equation_2
                    discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                    el_form += det_jac[i] * omega * discrete_integrand

            r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

            return r_el, j_el


class OdenDualWeakFormBCDirichlet(WeakForm):
    def evaluate_form(self, element_index, alpha):
        iel = element_index
        u_D = self.functions["u"]

        q_space = self.space.discrete_spaces["q"]
        q_components = q_space.n_comp
        q_data: ElementData = q_space.bc_elements[iel].data

        cell = q_data.cell
        dim = cell.dimension
        points, weights = self.space.bc_quadrature[dim]
        x, jac, det_jac, inv_jac = q_space.bc_elements[iel].evaluate_mapping(points)

        q_phi_tab = q_space.bc_elements[iel].evaluate_basis(
            points, jac, det_jac, inv_jac
        )

        n_q_phi = q_phi_tab.shape[2]
        n_q_dof = n_q_phi * q_components

        n_dof = n_q_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # find high-dimension neigh
        neigh_list = find_higher_dimension_neighs(cell, q_space.dof_map.mesh_topology)
        neigh_check_q = len(neigh_list) > 0
        assert neigh_check_q
        neigh_cell_id = neigh_list[0][1]
        neigh_cell_index = q_space.id_to_element[neigh_cell_id]
        neigh_element = q_space.elements[neigh_cell_index]
        neigh_cell = neigh_element.data.cell

        # compute trace space
        mapped_points = transform_lower_to_higher(points, q_data, neigh_element.data)
        _, jac_c0, det_jac_c0, inv_jac_c0 = neigh_element.evaluate_mapping(
            mapped_points
        )
        q_tr_phi_tab = neigh_element.evaluate_basis(
            mapped_points, jac_c0, det_jac_c0, inv_jac_c0
        )
        facet_index = neigh_cell.sub_cells_ids[cell.dimension].tolist().index(cell.id)
        dof_n_index = neigh_element.data.dof.entity_dofs[cell.dimension][facet_index]

        # compute normal
        n = normal(q_data.mesh, neigh_cell, cell)
        for c in range(q_components):
            b = c
            e = b + n_q_dof

            res_block_q = np.zeros(n_q_phi)
            dim = neigh_cell.dimension
            for i, omega in enumerate(weights):
                u_D_v = u_D(x[i, 0], x[i, 1], x[i, 2])
                phi = q_tr_phi_tab[0, i, dof_n_index, 0:dim] @ n[0:dim]
                res_block_q += det_jac[i] * omega * u_D_v[c] * phi

            r_el[b:e:q_components] += res_block_q

        return r_el, j_el
