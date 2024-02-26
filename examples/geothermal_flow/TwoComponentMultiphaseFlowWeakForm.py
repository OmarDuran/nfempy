import auto_diff as ad
import basix
import numpy as np
from auto_diff.vecvalder import VecValDer
from basix import CellType

from basis.element_data import ElementData
from basis.parametric_transformation import transform_lower_to_higher
from geometry.compute_normal import normal
from topology.topological_queries import find_higher_dimension_neighs
from weak_forms.weak_from import WeakForm


class TwoCompMultiPhaseFlowWeakForm(WeakForm):
    def evaluate_form(self, element_index, alpha):
        iel = element_index
        if self.space is None or self.functions is None:
            raise ValueError

        qm_space = self.space.discrete_spaces["q_mass"]
        qe_space = self.space.discrete_spaces["q_energy"]
        p_space = self.space.discrete_spaces["p"]
        h_space = self.space.discrete_spaces["h"]
        z_space = self.space.discrete_spaces["z"]

        f_rhs_p = self.functions["rhs_p"]
        f_rhs_h = self.functions["rhs_h"]
        f_rhs_z = self.functions["rhs_z"]

        f_kappa = self.functions["kappa"]
        f_delta = self.functions["delta"]

        qm_components = qm_space.n_comp
        qe_components = qe_space.n_comp
        p_components = p_space.n_comp
        h_components = z_space.n_comp
        z_components = z_space.n_comp

        qm_data: ElementData = qm_space.elements[iel].data
        qe_data: ElementData = qe_space.elements[iel].data
        p_data: ElementData = p_space.elements[iel].data
        h_data: ElementData = h_space.elements[iel].data
        z_data: ElementData = z_space.elements[iel].data

        cell = qm_data.cell
        dim = qm_data.dimension
        points, weights = self.space.quadrature
        x, jac, det_jac, inv_jac = qm_space.elements[iel].evaluate_mapping(points)

        # basis
        qm_phi_tab = qm_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        qe_phi_tab = qe_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        p_phi_tab = p_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        h_phi_tab = h_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        z_phi_tab = z_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)

        n_qm_phi = qm_phi_tab.shape[2]
        n_qe_phi = qe_phi_tab.shape[2]
        n_p_phi = p_phi_tab.shape[2]
        n_h_phi = h_phi_tab.shape[2]
        n_z_phi = z_phi_tab.shape[2]

        n_qm_dof = n_qm_phi * qm_components
        n_qe_dof = n_qe_phi * qe_components
        n_p_dof = n_p_phi * p_components
        n_h_dof = n_h_phi * h_components
        n_z_dof = n_z_phi * z_components

        dof_v = np.array([0 , n_qm_dof, n_qe_dof, n_p_dof, n_h_dof, n_z_dof])
        dof_s = np.cumsum(dof_v)
        n_dof = dof_s[-1]
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)
        alpha = np.zeros(n_dof)

        # Partial local vectorization
        f_p_val_star = f_rhs_p(x[:, 0], x[:, 1], x[:, 2])
        f_h_val_star = f_rhs_h(x[:, 0], x[:, 1], x[:, 2])
        f_z_val_star = f_rhs_z(x[:, 0], x[:, 1], x[:, 2])
        phi_p_star = det_jac * weights * p_phi_tab[0, :, :, 0].T
        phi_h_star = det_jac * weights * h_phi_tab[0, :, :, 0].T
        phi_z_star = det_jac * weights * z_phi_tab[0, :, :, 0].T

        # constant directors
        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])
        with ad.AutoDiff(alpha) as alpha:
            el_form = np.zeros(n_dof)
            for c in range(p_components):
                b = c + dof_s[2]
                e = c + dof_s[3]
                el_form[b:e:p_components] -= (phi_p_star @ f_p_val_star[c].T).ravel()

            for c in range(h_components):
                b = c + dof_s[3]
                e = c + dof_s[4]
                el_form[b:e:h_components] -= (phi_h_star @ f_h_val_star[c].T).ravel()

            for c in range(z_components):
                b = c + dof_s[4]
                e = c + dof_s[5]
                el_form[b:e:z_components] -= (phi_z_star @ f_z_val_star[c].T).ravel()

            for i, omega in enumerate(weights):
                xv = x[i]
                if dim == 2:
                    inv_jac_m = np.vstack((inv_jac[i] @ e1, inv_jac[i] @ e2))
                else:
                    inv_jac_m = np.vstack(
                        (inv_jac[i] @ e1, inv_jac[i] @ e2, inv_jac[i] @ e3)
                    )
                a_qm = alpha[:, dof_s[0]:dof_s[1]:1]
                a_qe = alpha[:, dof_s[1]:dof_s[2]:1]
                a_p  = alpha[:, dof_s[2]:dof_s[3]:1]
                a_h  = alpha[:, dof_s[3]:dof_s[4]:1]
                a_z  = alpha[:, dof_s[4]:dof_s[5]:1]

                qmh = a_qm @ qm_phi_tab[0, i, :, 0:dim]
                qeh = a_qe @ qe_phi_tab[0, i, :, 0:dim]
                ph = a_p @ p_phi_tab[0, i, :, 0:dim]
                hh = a_h @ h_phi_tab[0, i, :, 0:dim]
                zh = a_z @ z_phi_tab[0, i, :, 0:dim]

                qmh *= 1.0 / f_kappa(xv[0], xv[1], xv[2])
                qeh *= 1.0 / f_delta(xv[0], xv[1], xv[2])


                grad_vmh = qm_phi_tab[1 : qm_phi_tab.shape[0] + 1, i, :, 0:dim]
                div_vmh = np.array(
                    [[np.trace(grad_vmh[:, j, :]) / det_jac[i] for j in range(n_qm_dof)]]
                )
                grad_veh = qe_phi_tab[1 : qe_phi_tab.shape[0] + 1, i, :, 0:dim]
                div_veh = np.array(
                    [[np.trace(grad_veh[:, j, :]) / det_jac[i] for j in range(n_qe_dof)]]
                )

                div_qmh = a_qm @ div_vmh.T
                div_qeh = a_qe @ div_veh.T

                equ_1_integrand = qmh @ qm_phi_tab[0, i, :, 0:dim].T - ph @ div_vmh
                equ_2_integrand = qeh @ qe_phi_tab[0, i, :, 0:dim].T - hh @ div_veh
                equ_3_integrand = div_qmh @ p_phi_tab[0, i, :, 0:dim].T
                equ_4_integrand = div_qeh @ h_phi_tab[0, i, :, 0:dim].T
                equ_5_integrand = zh @ z_phi_tab[0, i, :, 0:dim].T

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, dof_s[0]:dof_s[1]:1] = equ_1_integrand
                multiphysic_integrand[:, dof_s[1]:dof_s[2]:1] = equ_2_integrand
                multiphysic_integrand[:, dof_s[2]:dof_s[3]:1] = equ_3_integrand
                multiphysic_integrand[:, dof_s[3]:dof_s[4]:1] = equ_4_integrand
                multiphysic_integrand[:, dof_s[4]:dof_s[5]:1] = equ_5_integrand
                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        return r_el, j_el


class TwoCompMultiPhaseFlowWeakFormBCDirichlet(WeakForm):
    def evaluate_form(self, element_index, alpha):
        iel = element_index
        p_D = self.functions["bc_p"]
        h_D = self.functions["bc_h"]

        qm_space = self.space.discrete_spaces["q_mass"]
        qe_space = self.space.discrete_spaces["q_energy"]
        qm_components = qm_space.n_comp
        qm_data: ElementData = qm_space.bc_elements[iel].data
        qe_components = qe_space.n_comp
        qe_data: ElementData = qe_space.bc_elements[iel].data

        cell = qm_data.cell
        points, weights = self.space.bc_quadrature
        dim = qm_data.dimension
        x, jac, det_jac, inv_jac = qm_space.bc_elements[iel].evaluate_mapping(points)

        # find high-dimension neigh q_mass space
        neigh_list = find_higher_dimension_neighs(cell, qm_space.dof_map.mesh_topology)
        neigh_check_q = len(neigh_list) > 0
        assert neigh_check_q
        neigh_cell_id = neigh_list[0]
        neigh_cell_index = qm_space.id_to_element[neigh_cell_id]
        neigh_element = qm_space.elements[neigh_cell_index]
        neigh_cell = neigh_element.data.cell

        # compute trace q space
        mapped_points = transform_lower_to_higher(points, qm_data, neigh_element.data)
        _, jac_c0, det_jac_c0, inv_jac_c0 = neigh_element.evaluate_mapping(
            mapped_points
        )
        qm_tr_phi_tab = neigh_element.evaluate_basis(
            mapped_points, jac_c0, det_jac_c0, inv_jac_c0
        )
        qm_facet_index = neigh_cell.sub_cells_ids[cell.dimension].tolist().index(cell.id)
        qm_dof_n_index = neigh_element.data.dof.entity_dofs[cell.dimension][
            qm_facet_index
        ]

        # find high-dimension neigh q_energy space
        neigh_list = find_higher_dimension_neighs(cell, qe_space.dof_map.mesh_topology)
        neigh_check_q = len(neigh_list) > 0
        assert neigh_check_q
        neigh_cell_id = neigh_list[0]
        neigh_cell_index = qe_space.id_to_element[neigh_cell_id]
        neigh_element = qe_space.elements[neigh_cell_index]
        neigh_cell = neigh_element.data.cell

        # compute trace u space
        mapped_points = transform_lower_to_higher(points, qe_data, neigh_element.data)
        _, jac_c0, det_jac_c0, inv_jac_c0 = neigh_element.evaluate_mapping(
            mapped_points
        )
        qe_tr_phi_tab = neigh_element.evaluate_basis(
            mapped_points, jac_c0, det_jac_c0, inv_jac_c0
        )
        qe_facet_index = neigh_cell.sub_cells_ids[cell.dimension].tolist().index(cell.id)
        qe_dof_n_index = neigh_element.data.dof.entity_dofs[cell.dimension][
            qe_facet_index
        ]

        n_qm_phi = qm_tr_phi_tab[0, :, qm_dof_n_index, 0:dim].shape[0]
        n_qm_dof = n_qm_phi * qm_components
        n_qe_phi = qe_tr_phi_tab[0, :, qe_dof_n_index, 0:dim].shape[0]
        n_qe_dof = n_qe_phi * qe_components

        n_dof = n_qm_dof + n_qe_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # compute normal
        n = normal(qm_data.mesh, neigh_cell, cell)
        for c in range(qm_components):
            b = c
            e = b + n_qm_dof

            res_block_q = np.zeros(n_qm_phi)
            dim = neigh_cell.dimension
            for i, omega in enumerate(weights):
                p_D_v = p_D(x[i, 0], x[i, 1], x[i, 2])
                phi = qm_tr_phi_tab[0, i, qm_dof_n_index, 0:dim] @ n[0:dim]
                res_block_q += det_jac[i] * omega * p_D_v[c] * phi

            r_el[b:e:qm_components] += res_block_q

        for c in range(qe_components):
            b = c + n_qm_dof
            e = b + n_qe_dof

            res_block_u = np.zeros(n_qe_phi)
            dim = neigh_cell.dimension
            for i, omega in enumerate(weights):
                h_D_v = h_D(x[i, 0], x[i, 1], x[i, 2])
                phi = qe_tr_phi_tab[0, i, qe_dof_n_index, 0:dim] @ n[0:dim]
                res_block_u += det_jac[i] * omega * h_D_v[c] * phi

            r_el[b:e:qe_components] += res_block_u

        return r_el, j_el
