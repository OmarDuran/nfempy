import auto_diff as ad
import numpy as np

from basis.element_data import ElementData
from basis.parametric_transformation import transform_lower_to_higher
from geometry.compute_normal import normal
from topology.topological_queries import find_higher_dimension_neighs
from topology.topological_queries import sub_entity_by_co_dimension
from weak_forms.weak_from import WeakForm


class DiffusionWeakForm(WeakForm):
    def evaluate_form(self, element_index, alpha_n, alpha, t):
        iel = element_index
        if self.space is None or self.functions is None:
            raise ValueError

        md_space = self.space.discrete_spaces["md"]
        ca_space = self.space.discrete_spaces["ca"]
        qd_space = self.space.discrete_spaces["qd"]
        qa_space = self.space.discrete_spaces["qa"]
        p_space = self.space.discrete_spaces["p"]
        z_space = self.space.discrete_spaces["z"]
        h_space = self.space.discrete_spaces["h"]
        t_space = self.space.discrete_spaces["t"]
        sv_space = self.space.discrete_spaces["sv"]
        x_H2O_l_space = self.space.discrete_spaces["x_H2O_l"]
        x_H2O_v_space = self.space.discrete_spaces["x_H2O_v"]
        x_NaCl_l_space = self.space.discrete_spaces["x_NaCl_l"]
        x_NaCl_v_space = self.space.discrete_spaces["x_NaCl_v"]

        md_data: ElementData = md_space.elements[iel].data
        ca_data: ElementData = ca_space.elements[iel].data
        qd_data: ElementData = qd_space.elements[iel].data
        qa_data: ElementData = qa_space.elements[iel].data
        p_data: ElementData = p_space.elements[iel].data
        z_data: ElementData = z_space.elements[iel].data
        h_data: ElementData = h_space.elements[iel].data
        t_data: ElementData = t_space.elements[iel].data
        sv_data: ElementData = sv_space.elements[iel].data
        x_H2O_l_data: ElementData = x_H2O_l_space.elements[iel].data
        x_H2O_v_data: ElementData = x_H2O_v_space.elements[iel].data
        x_NaCl_l_data: ElementData = x_NaCl_l_space.elements[iel].data
        x_NaCl_v_data: ElementData = x_NaCl_v_space.elements[iel].data

        n_components = 1
        f_K_thermal = self.functions["K_thermal"]
        f_kappa = self.functions["kappa"]

        cell = md_data.cell
        dim = cell.dimension
        points, weights = self.space.quadrature
        x, jac, det_jac, inv_jac = md_space.elements[iel].evaluate_mapping(points)

        # Hdiv basis
        dv_h_tab = md_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        # Constant basis
        du_h_tab = p_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)

        n_v_dof = md_data.dof.n_dof
        n_u_dof = p_data.dof.n_dof

        md_dof, ca_dof, qd_dof, qa_dof = 4 * [n_v_dof]
        (
            p_dof,
            z_dof,
            h_dof,
            t_dof,
            sv_dof,
            x_H2O_l_dof,
            x_H2O_v_dof,
            x_NaCl_l_dof,
            x_NaCl_v_dof,
        ) = 9 * [n_u_dof]

        v_dofs = md_dof + ca_dof + qd_dof + qa_dof
        idx_dof = {
            "md": slice(0, md_dof),
            "ca": slice(md_dof, md_dof + ca_dof),
            "qd": slice(md_dof + ca_dof, md_dof + ca_dof + qd_dof),
            "qa": slice(md_dof + ca_dof + qd_dof, v_dofs),
            "p": slice(v_dofs, v_dofs + p_dof),
            "z": slice(v_dofs + p_dof, v_dofs + p_dof + z_dof),
            "h": slice(v_dofs + p_dof + z_dof, v_dofs + p_dof + z_dof + h_dof),
            "t": slice(
                v_dofs + p_dof + z_dof + h_dof, v_dofs + p_dof + z_dof + h_dof + t_dof
            ),
            "sv": slice(
                v_dofs + p_dof + z_dof + h_dof + t_dof, v_dofs + p_dof + z_dof + h_dof + t_dof + sv_dof
            ),
            "x_H2O_l": slice(
                v_dofs + p_dof + z_dof + h_dof + t_dof + sv_dof,
                v_dofs + p_dof + z_dof + h_dof + t_dof + sv_dof + x_H2O_l_dof
            ),
            "x_H2O_v": slice(
                v_dofs + p_dof + z_dof + h_dof + t_dof + sv_dof + x_H2O_l_dof,
                v_dofs + p_dof + z_dof + h_dof + t_dof + sv_dof + x_H2O_l_dof + x_H2O_v_dof
            ),
            "x_NaCl_l": slice(
                v_dofs + p_dof + z_dof + h_dof + t_dof + sv_dof + x_H2O_l_dof + x_H2O_v_dof,
                v_dofs + p_dof + z_dof + h_dof + t_dof + sv_dof + x_H2O_l_dof + x_H2O_v_dof + x_NaCl_l_dof
            ),
            "x_NaCl_v": slice(
                v_dofs + p_dof + z_dof + h_dof + t_dof + sv_dof + x_H2O_l_dof + x_H2O_v_dof + x_NaCl_l_dof,
                v_dofs + p_dof + z_dof + h_dof + t_dof + sv_dof + x_H2O_l_dof + x_H2O_v_dof + x_NaCl_l_dof + x_NaCl_v_dof
            ),
        }

        n_dof = v_dofs + 9 * n_u_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # Partial local vectorization
        du_h_star = det_jac * weights * du_h_tab[0, :, :, 0].T

        # constant directors
        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])

        with ad.AutoDiff(alpha_n) as alpha_n:
            el_form = np.zeros(n_dof)

            for i, omega in enumerate(weights):
                xv = x[i]

                # Functions and derivatives at integration point i
                dv_h = dv_h_tab[0, i, :, 0:dim]
                grad_dv_h = dv_h_tab[1 : dv_h_tab.shape[0] + 1, i, :, 0:dim]
                div_dv_h = np.array(
                    [
                        [
                            np.trace(grad_dv_h[:, j, :]) / det_jac[i]
                            for j in range(n_v_dof)
                        ]
                    ]
                )
                du_h = du_h_tab[0, i, :, 0:dim]

                dmd_h = dv_h
                dca_h = dv_h
                dqd_h = dv_h
                dqa_h = dv_h

                # Dof per field
                a_md_n = alpha_n[:, idx_dof["md"]]
                a_ca_n = alpha_n[:, idx_dof["ca"]]
                a_qd_n = alpha_n[:, idx_dof["qd"]]
                a_qa_n = alpha_n[:, idx_dof["qa"]]

                a_p_n = alpha_n[:,idx_dof["p"]]
                a_z_n = alpha_n[:,idx_dof["z"]]
                a_h_n = alpha_n[:,idx_dof["h"]]
                a_t_n = alpha_n[:,idx_dof["t"]]
                a_sv_n = alpha_n[:,idx_dof["sv"]]
                a_x_H2O_l_n = alpha_n[:, idx_dof["x_H2O_l"]]
                a_x_H2O_v_n = alpha_n[:, idx_dof["x_H2O_v"]]
                a_x_NaCl_l_n = alpha_n[:, idx_dof["x_NaCl_l"]]
                a_x_NaCl_v_n = alpha_n[:, idx_dof["x_NaCl_v"]]


                a_p = alpha[idx_dof["p"]]
                a_z = alpha[idx_dof["z"]]
                a_h = alpha[idx_dof["h"]]

                # FEM approximation
                md_h_n = a_md_n @ dmd_h
                ca_h_n = a_ca_n @ dmd_h
                qd_h_n = a_qd_n @ dqd_h
                qa_h_n = a_qa_n @ dmd_h

                p_h_n = a_p_n @ du_h
                z_h_n = a_z_n @ du_h
                h_h_n = a_h_n @ du_h
                t_h_n = a_t_n @ du_h
                sv_h_n = a_sv_n @ du_h
                x_H2O_l_h_n = a_x_H2O_l_n @ du_h
                x_H2O_v_h_n = a_x_H2O_v_n @ du_h
                x_NaCl_l_h_n = a_x_NaCl_l_n @ du_h
                x_NaCl_v_h_n = a_x_NaCl_v_n @ du_h

                p_h = a_p @ du_h
                z_h = a_z @ du_h
                h_h = a_h @ du_h

                md_h_n *= 1.0 / f_kappa(xv[0], xv[1], xv[2])
                qd_h_n *= 1.0 / f_K_thermal(xv[0], xv[1], xv[2])

                div_md_h = a_md_n @ div_dv_h.T
                div_ca_h = a_ca_n @ div_dv_h.T
                div_qd_h = a_qd_n @ div_dv_h.T
                div_qa_h = a_qa_n @ div_dv_h.T

                equ_1_integrand = (md_h_n @ dv_h.T) - (p_h_n @ div_dv_h)

                # delete 2
                equ_2_integrand = (ca_h_n @ dv_h.T) - (z_h_n @ div_dv_h)

                equ_3_integrand = (qd_h_n @ dv_h.T) - (t_h_n @ div_dv_h)
                # delete 4
                equ_4_integrand = (qa_h_n @ dv_h.T) - (h_h_n @ div_dv_h)

                equ_5_integrand = (div_md_h) @ du_h.T
                equ_6_integrand = (div_ca_h) @ du_h.T
                equ_7_integrand = (div_qd_h + div_qa_h) @ du_h.T
                equ_8_integrand = (t_h_n - h_h_n) @ du_h.T
                equ_9_integrand = (sv_h_n - 0.0) @ du_h.T
                equ_10_integrand = (x_H2O_l_h_n - (1.0 - z_h_n)) @ du_h.T
                equ_11_integrand = (x_H2O_v_h_n - (1.0 - z_h_n)) @ du_h.T
                equ_12_integrand = (x_NaCl_l_h_n - z_h_n) @ du_h.T
                equ_13_integrand = (x_NaCl_v_h_n - z_h_n) @ du_h.T


                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, idx_dof["md"]] = equ_1_integrand
                multiphysic_integrand[:, idx_dof["ca"]] = equ_2_integrand
                multiphysic_integrand[:, idx_dof["qd"]] = equ_3_integrand
                multiphysic_integrand[:, idx_dof["qa"]] = equ_4_integrand
                multiphysic_integrand[:, idx_dof["p"]] = equ_5_integrand
                multiphysic_integrand[:, idx_dof["z"]] = equ_6_integrand
                multiphysic_integrand[:, idx_dof["h"]] = equ_7_integrand
                multiphysic_integrand[:, idx_dof["t"]] = equ_8_integrand
                multiphysic_integrand[:, idx_dof["sv"]] = equ_9_integrand
                multiphysic_integrand[:, idx_dof["x_H2O_l"]] = equ_10_integrand
                multiphysic_integrand[:, idx_dof["x_H2O_v"]] = equ_11_integrand
                multiphysic_integrand[:, idx_dof["x_NaCl_l"]] = equ_12_integrand
                multiphysic_integrand[:, idx_dof["x_NaCl_v"]] = equ_13_integrand
                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        return r_el, j_el


class DiffusionWeakFormBCRobin(WeakForm):
    def evaluate_form(self, element_index, alpha, t):
        iel = element_index

        f_beta = self.functions["beta"]
        f_gamma = self.functions["gamma"]
        f_c = self.functions["c"]
        f_velocity = self.functions["velocity"]

        q_space = self.space.discrete_spaces["q"]
        m_space = self.space.discrete_spaces["m"]

        q_components = q_space.n_comp
        q_data: ElementData = q_space.bc_elements[iel].data

        m_components = m_space.n_comp
        m_data: ElementData = m_space.bc_elements[iel].data

        cell = q_data.cell
        points, weights = self.space.bc_quadrature
        dim = cell.dimension
        x, jac, det_jac, inv_jac = q_space.bc_elements[iel].evaluate_mapping(points)

        # Diffusive flux
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

        # Advective flux
        # find high-dimension neigh q space
        neigh_list = find_higher_dimension_neighs(cell, m_space.dof_map.mesh_topology)
        neigh_check_mp = len(neigh_list) > 0
        assert neigh_check_mp
        neigh_cell_id = neigh_list[0]
        neigh_cell_index = m_space.id_to_element[neigh_cell_id]
        neigh_element = m_space.elements[neigh_cell_index]
        neigh_cell = neigh_element.data.cell

        # compute trace q space
        mapped_points = transform_lower_to_higher(points, m_data, neigh_element.data)
        _, jac_c0, det_jac_c0, inv_jac_c0 = neigh_element.evaluate_mapping(
            mapped_points
        )
        dm_tr_phi_tab = neigh_element.evaluate_basis(
            mapped_points, jac_c0, det_jac_c0, inv_jac_c0
        )
        dm_facet_index = (
            neigh_cell.sub_cells_ids[cell.dimension].tolist().index(cell.id)
        )
        dm_dof_n_index = neigh_element.data.dof.entity_dofs[cell.dimension][
            dm_facet_index
        ]

        n_dm_phi = dm_tr_phi_tab[0, :, dm_dof_n_index, 0:dim].shape[0]
        n_m_dof = n_dm_phi * m_components

        idx_dof = {
            "q": slice(0, n_q_dof),
            "m": slice(n_q_dof, n_q_dof + n_m_dof),
        }

        n_dof = n_q_dof + n_m_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        dim = neigh_cell.dimension
        with ad.AutoDiff(alpha) as alpha:
            el_form = np.zeros(n_dof)
            alpha_q = alpha[:, idx_dof["q"]]
            alpha_m = alpha[:, idx_dof["m"]]

            # compute normal
            n = normal(q_data.mesh, neigh_cell, cell)

            for i, omega in enumerate(weights):
                beta_v = f_beta(x[i, 0], x[i, 1], x[i, 2])
                gamma_v = f_gamma(x[i, 0], x[i, 1], x[i, 2])
                c_v = f_c(x[i, 0], x[i, 1], x[i, 2])

                dq_h = dq_tr_phi_tab[0, i, dq_dof_n_index, 0:dim]  # @ n[0:dim]
                dm_h = dm_tr_phi_tab[0, i, dm_dof_n_index, 0:dim]  # @ n[0:dim]

                # This sign may be needed in 1d computations because BC orientation
                # However in pure Hdiv functions in 2d and 3d it is not needed
                bc_sign = np.sign(n[0:dim])[0]

                q_h_n = bc_sign * alpha_q @ dq_h
                m_h_n = bc_sign * alpha_m @ dm_h

                equ_1_integrand = (
                    (1.0 / beta_v)
                    * (q_h_n + m_h_n + beta_v * c_v - gamma_v * c_v)
                    * bc_sign
                    * dq_h.T
                )

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, idx_dof["q"]] = equ_1_integrand
                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

            r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        return r_el, j_el


class AdvectionWeakForm(WeakForm):
    def evaluate_form(self, cell_id, element_pair_index, alpha_pair):
        iel_p, iel_n = element_pair_index
        alpha_p, alpha_n = alpha_pair
        if self.space is None or self.functions is None:
            raise ValueError

        q_space = self.space.discrete_spaces["q"]
        m_space = self.space.discrete_spaces["m"]
        u_space = self.space.discrete_spaces["u"]

        f_velocity = self.functions["velocity"]

        q_components = q_space.n_comp
        m_components = m_space.n_comp
        u_components = u_space.n_comp

        q_data_p: ElementData = q_space.elements[iel_p].data
        m_data_p: ElementData = m_space.elements[iel_p].data
        u_data_p: ElementData = u_space.elements[iel_p].data

        q_data_n: ElementData = q_space.elements[iel_n].data
        m_data_n: ElementData = m_space.elements[iel_n].data
        u_data_n: ElementData = u_space.elements[iel_n].data

        dim = q_data_p.cell.dimension

        # trace of qh on both sides
        gmesh = q_space.mesh_topology.mesh
        cell_c1 = gmesh.cells[cell_id]
        element_c1_data = ElementData(cell_c1, m_space.mesh_topology.mesh)
        points, weights = self.space.bc_quadrature

        # compute trace u space
        trace_m_space = []
        trace_u_space = []
        trace_x = []
        normals = []
        dofs = []
        det_jacs = []
        for iel, c0_data in zip([iel_p, iel_n], [q_data_p, q_data_n]):
            mapped_points = transform_lower_to_higher(points, element_c1_data, c0_data)
            x_c0, jac_c0, det_jac_c0, inv_jac_c0 = m_space.elements[
                iel
            ].evaluate_mapping(mapped_points)
            dm_tr_phi_tab = m_space.elements[iel].evaluate_basis(
                mapped_points, jac_c0, det_jac_c0, inv_jac_c0
            )
            dm_facet_index = (
                c0_data.cell.sub_cells_ids[cell_c1.dimension].tolist().index(cell_c1.id)
            )
            dm_dof_n_index = c0_data.dof.entity_dofs[cell_c1.dimension][dm_facet_index]
            dm_h = dm_tr_phi_tab[0, :, dm_dof_n_index, 0:dim]

            du_tr_phi_tab = u_space.elements[iel].evaluate_basis(
                mapped_points, jac_c0, det_jac_c0, inv_jac_c0
            )
            n = normal(gmesh, c0_data.cell, cell_c1)
            du_h = du_tr_phi_tab[0, :, :, 0:dim]

            trace_m_space.append(dm_h)
            trace_u_space.append(du_h)
            trace_x.append(x_c0)
            normals.append(n)
            dofs.append(dm_dof_n_index)
            det_jac_c0[0] = 1.0
            det_jacs.append(det_jac_c0)

        assert q_data_p.dof.n_dof == q_data_n.dof.n_dof
        assert m_data_p.dof.n_dof == m_data_n.dof.n_dof
        assert u_data_p.dof.n_dof == u_data_n.dof.n_dof

        n_dq_h = q_data_p.dof.n_dof
        n_dm_h = m_data_p.dof.n_dof
        n_du_h = u_data_p.dof.n_dof
        n_q_dof = n_dq_h * q_components
        n_m_dof = n_dm_h * m_components
        n_u_dof = n_du_h * u_components

        idx_dof = {
            "q_p": slice(0, n_q_dof),
            "m_p": slice(n_q_dof, n_q_dof + n_m_dof),
            "u_p": slice(n_q_dof + n_m_dof, n_q_dof + n_m_dof + n_u_dof),
            "q_n": slice(n_q_dof + n_m_dof + n_u_dof, 2 * n_q_dof + n_m_dof + n_u_dof),
            "m_n": slice(
                2 * n_q_dof + n_m_dof + n_u_dof, 2 * n_q_dof + 2 * n_m_dof + n_u_dof
            ),
            "u_n": slice(
                2 * n_q_dof + 2 * n_m_dof + n_u_dof,
                2 * n_q_dof + 2 * n_m_dof + 2 * n_u_dof,
            ),
        }

        n_dof = 2 * n_q_dof + 2 * n_m_dof + 2 * n_u_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)
        alpha = np.concatenate((alpha_p, alpha_n))
        with ad.AutoDiff(alpha) as alpha:
            el_form = np.zeros(n_dof)

            for i, omega in enumerate(weights):
                alpha_m_p = alpha[:, idx_dof["m_p"]][:, dofs[0]]
                alpha_m_n = alpha[:, idx_dof["m_n"]][:, dofs[1]]
                alpha_u_p = alpha[:, idx_dof["u_p"]]
                alpha_u_n = alpha[:, idx_dof["u_n"]]

                assert np.all(np.isclose(alpha_m_p.val, alpha_m_n.val))

                x_p, x_n = trace_x
                assert np.all(np.isclose(x_p, x_n))
                xv = x_p = x_n

                dm_h_p, dm_h_n = trace_m_space[0][i], trace_m_space[1][i]
                m_h_p, m_h_n = alpha_m_p @ dm_h_p, alpha_m_n @ dm_h_n

                n_p, n_n = normals
                du_h_p, du_h_n = trace_u_space[0][i], trace_u_space[1][i]
                u_h_p, u_h_n = alpha_u_p @ du_h_p, alpha_u_n @ du_h_n

                v_n = f_velocity(xv[:, 0], xv[:, 1], xv[:, 2]) @ n_p
                beta_upwind = 0.0
                if v_n > 0.0 or np.isclose(v_n, 0.0):
                    beta_upwind = 1.0

                u_h_upwind = (1.0 - beta_upwind) * u_h_n + beta_upwind * u_h_p

                equ_1_integrand = 0.5 * (m_h_p - u_h_upwind * v_n) @ dm_h_p.T
                equ_2_integrand = 0.5 * (m_h_n - u_h_upwind * v_n) @ dm_h_n.T

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, idx_dof["m_p"]][:, dofs[0]] = equ_1_integrand
                multiphysic_integrand[:, idx_dof["m_n"]][:, dofs[1]] = equ_2_integrand
                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jacs[0][i] * omega * discrete_integrand

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        return r_el, j_el


class AdvectionWeakFormBC(WeakForm):
    def evaluate_form(self, cell_id, element_index, alpha):
        iel = element_index
        if self.space is None or self.functions is None:
            raise ValueError

        q_space = self.space.discrete_spaces["q"]
        m_space = self.space.discrete_spaces["m"]
        u_space = self.space.discrete_spaces["u"]

        f_u = self.functions["u"]
        f_velocity = self.functions["velocity"]

        q_components = q_space.n_comp
        m_components = m_space.n_comp
        u_components = u_space.n_comp

        q_data: ElementData = q_space.elements[iel].data
        m_data: ElementData = m_space.elements[iel].data
        u_data: ElementData = u_space.elements[iel].data

        dim = q_data.cell.dimension

        # trace of qh on both sides
        gmesh = q_space.mesh_topology.mesh
        cell_c1 = gmesh.cells[cell_id]
        element_c1_data = ElementData(cell_c1, m_space.mesh_topology.mesh)
        points, weights = self.space.bc_quadrature

        # compute trace u space
        trace_m_space = []
        trace_u_space = []
        trace_x = []
        normals = []
        det_jacs = []
        dofs = []
        for iel, c0_data in zip([iel], [q_data]):
            mapped_points = transform_lower_to_higher(points, element_c1_data, c0_data)
            x_c0, jac_c0, det_jac_c0, inv_jac_c0 = m_space.elements[
                iel
            ].evaluate_mapping(mapped_points)
            dm_tr_phi_tab = m_space.elements[iel].evaluate_basis(
                mapped_points, jac_c0, det_jac_c0, inv_jac_c0
            )
            dm_facet_index = (
                c0_data.cell.sub_cells_ids[cell_c1.dimension].tolist().index(cell_c1.id)
            )
            dm_dof_n_index = c0_data.dof.entity_dofs[cell_c1.dimension][dm_facet_index]
            dm_h = dm_tr_phi_tab[0, :, dm_dof_n_index, 0:dim]
            trace_m_space.append(dm_h)

            du_tr_phi_tab = u_space.elements[iel].evaluate_basis(
                mapped_points, jac_c0, det_jac_c0, inv_jac_c0
            )
            n = normal(gmesh, c0_data.cell, cell_c1)
            du_h = du_tr_phi_tab[0, :, :, 0:dim]
            trace_u_space.append(du_h)
            trace_x.append(x_c0)
            normals.append(n)
            dofs.append(dm_dof_n_index)
            det_jac_c0[0] = 1.0
            det_jacs.append(det_jac_c0)

        n_dq_h = q_data.dof.n_dof
        n_dm_h = m_data.dof.n_dof
        n_du_h = u_data.dof.n_dof
        n_q_dof = n_dq_h * q_components
        n_m_dof = n_dm_h * m_components
        n_u_dof = n_du_h * u_components

        idx_dof = {
            "q": slice(0, n_q_dof),
            "m": slice(n_q_dof, n_q_dof + n_m_dof),
            "u": slice(n_q_dof + n_m_dof, n_q_dof + n_m_dof + n_u_dof),
        }

        n_dof = n_q_dof + n_m_dof + n_u_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)
        with ad.AutoDiff(alpha) as alpha:
            el_form = np.zeros(n_dof)

            for i, omega in enumerate(weights):
                alpha_m = alpha[:, idx_dof["m"]][:, dofs[0]]
                alpha_u = alpha[:, idx_dof["u"]]
                xv = trace_x[i]
                n = normals[i]

                # This sign may be needed in 1d computations because BC orientation
                # However in pure Hdiv functions in 2d and 3d it is not needed
                bc_sign = np.sign(n[0:dim])[0]

                dm_h = trace_m_space[0][i]
                m_h = alpha_m @ dm_h

                du_h = trace_u_space[0][i]
                u_h = alpha_u @ du_h

                v_n = f_velocity(xv[:, 0], xv[:, 1], xv[:, 2]) @ n
                u_v = f_u(xv[:, 0], xv[:, 1], xv[:, 2])[0]

                beta_upwind = 0.0
                if v_n > 0.0 or np.isclose(v_n, 0.0):
                    beta_upwind = 1.0

                u_h_upwind = (1.0 - beta_upwind) * u_v + beta_upwind * u_h

                equ_1_integrand = (m_h - bc_sign * (u_h_upwind * v_n)) @ dm_h.T

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, idx_dof["m"]][:, dofs[0]] = equ_1_integrand
                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jacs[0][i] * omega * discrete_integrand

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        return r_el, j_el
