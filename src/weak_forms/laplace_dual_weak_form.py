import auto_diff as ad
import basix
import numpy as np
from auto_diff.vecvalder import VecValDer
from basix import CellType

from basis.element_data import ElementData
from basis.parametric_transformation import transform_lower_to_higher
from basis.permute_and_transform import (
    _validate_edge_orientation_2d,
    permute_and_transform,
)
from geometry.compute_normal import normal
from topology.topological_queries import find_higher_dimension_neighs
from weak_forms.weak_from import WeakForm


class LaplaceDualWeakForm(WeakForm):
    def evaluate_form(self, element_index, alpha):
        iel = element_index
        if self.space is None or self.functions is None:
            raise ValueError

        q_space = self.space.discrete_spaces["q"]
        p_space = self.space.discrete_spaces["p"]

        f_rhs = self.functions["rhs"]
        f_kappa = self.functions["kappa"]

        q_components = q_space.n_comp
        p_components = p_space.n_comp
        q_data: ElementData = q_space.elements[iel].data
        p_data: ElementData = p_space.elements[iel].data

        cell = q_data.cell
        dim = q_data.dimension
        points = q_data.quadrature.points
        weights = q_data.quadrature.weights
        x = q_data.mapping.x
        det_jac = q_data.mapping.det_jac
        inv_jac = q_data.mapping.inv_jac

        q_phi_tab = q_data.basis.phi
        p_phi_tab = p_data.basis.phi

        n_q_phi = q_phi_tab.shape[2]
        n_p_phi = p_phi_tab.shape[2]
        n_q_dof = n_q_phi * q_components
        n_p_dof = n_p_phi * p_components
        n_dof = n_q_dof + n_p_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)
        alpha = np.zeros(n_dof)
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
                b = c + n_q_dof
                e = b + n_p_dof
                el_form[b:e:p_components] -= phi_s_star @ f_val_star[c]

            for i, omega in enumerate(weights):
                xv = x[i]
                if dim == 2:
                    inv_jac_m = np.vstack((inv_jac[i] @ e1, inv_jac[i] @ e2))
                else:
                    inv_jac_m = np.vstack(
                        (inv_jac[i] @ e1, inv_jac[i] @ e2, inv_jac[i] @ e3)
                    )
                qh = alpha[:, 0:n_q_dof:1] @ q_phi_tab[0, i, :, 0:dim]
                qh *= 1.0 / f_kappa(xv[0], xv[1], xv[2])
                ph = alpha[:, n_q_dof:n_dof:1] @ p_phi_tab[0, i, :, 0:dim]
                grad_qh = q_phi_tab[1 : q_phi_tab.shape[0] + 1, i, :, 0:dim]
                div_vh = np.array(
                    [[np.trace(grad_qh[:, j, :]) / det_jac[i] for j in range(n_q_dof)]]
                )
                div_qh = alpha[:, 0:n_q_dof:1] @ div_vh.T
                equ_1_integrand = (qh @ q_phi_tab[0, i, :, 0:dim].T) - (ph @ div_vh)
                equ_2_integrand = div_qh @ p_phi_tab[0, i, :, 0:dim].T
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

        q_space = self.space.discrete_spaces["q"]
        q_components = q_space.n_comp
        q_data: ElementData = q_space.bc_elements[iel].data

        cell = q_data.cell
        points = q_data.quadrature.points
        weights = q_data.quadrature.weights
        x = q_data.mapping.x
        det_jac = q_data.mapping.det_jac
        inv_jac = q_data.mapping.inv_jac

        q_phi_tab = q_data.basis.phi
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
        neigh_cell_id = neigh_list[0]
        neigh_cell_index = q_space.id_to_element[neigh_cell_id]
        neigh_element = q_space.elements[neigh_cell_index]
        neigh_cell = neigh_element.data.cell

        # compute trace space
        mapped_points = transform_lower_to_higher(points, q_data, neigh_element.data)
        facet_index = neigh_cell.sub_cells_ids[cell.dimension].tolist().index(cell.id)
        dof_n_index = neigh_element.data.dof.entity_dofs[cell.dimension][facet_index]
        q_tr_phi_tab = neigh_element.evaluate_basis(mapped_points, False)

        # compute normal
        n = normal(q_data.mesh, neigh_cell, cell)
        for c in range(q_components):
            b = c
            e = b + n_q_dof

            res_block_q = np.zeros(n_q_phi)
            dim = neigh_cell.dimension
            for i, omega in enumerate(weights):
                p_D_v = p_D(x[i, 0], x[i, 1], x[i, 2])
                phi = q_tr_phi_tab[0, i, dof_n_index, 0:dim] @ n[0:dim]
                res_block_q += det_jac[i] * omega * p_D_v[c] * phi

            r_el[b:e:q_components] += res_block_q

        return r_el, j_el
