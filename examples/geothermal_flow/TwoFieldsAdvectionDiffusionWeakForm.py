import auto_diff as ad
import numpy as np

from basis.element_data import ElementData
from basis.parametric_transformation import transform_lower_to_higher
from geometry.compute_normal import normal
from topology.topological_queries import find_higher_dimension_neighs
from topology.topological_queries import sub_entity_by_co_dimension
from weak_forms.weak_from import WeakForm


class TwoFieldsDiffusionWeakForm(WeakForm):
    def evaluate_form(self, element_index, alpha_n_p_1, alpha_n, t):
        iel = element_index
        if self.space is None or self.functions is None:
            raise ValueError

        q_space = self.space.discrete_spaces["q"]
        m_space = self.space.discrete_spaces["m"]
        u_space = self.space.discrete_spaces["u"]

        f_rhs = self.functions["rhs"]
        f_kappa = self.functions["kappa"]
        f_velocity = self.functions["velocity"]

        q_components = q_space.n_comp
        m_components = m_space.n_comp
        u_components = u_space.n_comp

        q_data: ElementData = q_space.elements[iel].data
        m_data: ElementData = m_space.elements[iel].data
        u_data: ElementData = u_space.elements[iel].data

        cell = q_data.cell
        dim = cell.dimension
        points, weights = self.space.quadrature
        x, jac, det_jac, inv_jac = q_space.elements[iel].evaluate_mapping(points)

        # basis
        dq_h_tab = q_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        dm_h_tab = m_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        du_h_tab = u_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)

        n_dq_h = dq_h_tab.shape[2]
        n_dm_h = dm_h_tab.shape[2]
        n_du_h = du_h_tab.shape[2]

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

        # Partial local vectorization
        f_val_star = f_rhs(x[:, 0], x[:, 1], x[:, 2])
        du_h_star = det_jac * weights * du_h_tab[0, :, :, 0].T

        # constant directors
        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])

        # trace of qh
        codimension = 1
        sub_entities = sub_entity_by_co_dimension(cell, codimension)
        sub_entities_q = len(sub_entities) > 0
        assert sub_entities_q
        gmesh = q_space.mesh_topology.mesh
        el_datas = [
            ElementData(gmesh.cells[cell_id], q_space.mesh_topology.mesh)
            for cell_id in sub_entities
        ]
        c1_points, c1_weights = self.space.bc_quadrature

        # compute trace q space
        trace_q_space = []
        trace_u_space = []
        trace_q_dof = []
        trace_x = []
        normals = []
        for i, idx in enumerate(sub_entities):
            sub_cell = gmesh.cells[idx]
            mapped_points = transform_lower_to_higher(c1_points, el_datas[i], q_data)
            x_c0, jac_c0, det_jac_c0, inv_jac_c0 = q_space.elements[
                iel
            ].evaluate_mapping(mapped_points)
            dq_tr_phi_tab = q_space.elements[iel].evaluate_basis(
                mapped_points, jac_c0, det_jac_c0, inv_jac_c0
            )
            du_tr_phi_tab = u_space.elements[iel].evaluate_basis(
                mapped_points, jac_c0, det_jac_c0, inv_jac_c0
            )
            dq_facet_index = cell.sub_cells_ids[sub_cell.dimension].tolist().index(idx)
            dq_dof_n_index = np.array(
                q_space.elements[iel].data.dof.entity_dofs[sub_cell.dimension][
                    dq_facet_index
                ]
            )
            trace_q_dof.append(dq_dof_n_index)
            n = normal(gmesh, cell, sub_cell)
            dq_h = dq_tr_phi_tab[0, :, dq_dof_n_index, 0:dim] @ n[0:dim]
            du_h = du_tr_phi_tab[0, :, :, 0:dim]
            trace_q_space.append(dq_h)
            trace_u_space.append(du_h)
            trace_x.append(x_c0)
            normals.append(n)

        with ad.AutoDiff(alpha_n_p_1) as alpha_n_p_1:
            el_form = np.zeros(n_dof)
            for c in range(u_components):
                b = c + n_q_dof + n_m_dof
                e = b + n_u_dof
                el_form[b:e:u_components] -= du_h_star @ f_val_star[c].T

            for i, omega in enumerate(weights):
                xv = x[i]

                # Functions and derivatives at integration point i
                dq_h = dq_h_tab[0, i, :, 0:dim]
                dm_h = dm_h_tab[0, i, :, 0:dim]
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

                grad_dm_h = dm_h_tab[1: dm_h.shape[0] + 1, i, :, 0:dim]
                div_dm_h = np.array(
                    [
                        [
                            np.trace(grad_dm_h[:, j, :]) / det_jac[i]
                            for j in range(n_m_dof)
                        ]
                    ]
                )

                # Dof per field
                alpha_q_n_p_1 = alpha_n_p_1[:, idx_dof["q"]]
                alpha_m_n_p_1 = alpha_n_p_1[:, idx_dof["m"]]
                alpha_u_n_p_1 = alpha_n_p_1[:, idx_dof["u"]]
                alpha_u_n = alpha_n[idx_dof["u"]]

                # FEM approximation
                q_h_n_p_1 = alpha_q_n_p_1 @ dq_h
                u_h_n_p_1 = alpha_u_n_p_1 @ du_h
                u_h_n = alpha_u_n @ du_h

                div_q_h = alpha_q_n_p_1 @ div_dq_h.T
                q_h_n_p_1 *= 1.0 / f_kappa(xv[0], xv[1], xv[2])

                div_m_h = alpha_m_n_p_1 @ div_dm_h.T

                equ_1_integrand = (q_h_n_p_1 @ dq_h.T) - (u_h_n_p_1 @ div_dq_h)
                equ_3_integrand = (div_q_h + div_m_h) @ du_h.T

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, idx_dof["q"]] = equ_1_integrand
                multiphysic_integrand[:, idx_dof["u"]] = equ_3_integrand
                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        return r_el, j_el


class TwoFieldsDiffusionWeakFormBCRobin(WeakForm):
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
            alpha_q = alpha[:, idx_dof['q']]
            alpha_m = alpha[:, idx_dof['m']]

            # compute normal
            n = normal(q_data.mesh, neigh_cell, cell)


            for i, omega in enumerate(weights):
                beta_v = f_beta(x[i, 0], x[i, 1], x[i, 2])
                gamma_v = f_gamma(x[i, 0], x[i, 1], x[i, 2])
                c_v = f_c(x[i, 0], x[i, 1], x[i, 2])

                dq_h = dq_tr_phi_tab[0, i, dq_dof_n_index, 0:dim]# @ n[0:dim]
                dm_h = dm_tr_phi_tab[0, i, dm_dof_n_index, 0:dim]# @ n[0:dim]

                # This sign may be needed in 1d computations because BC orientation
                # However in pure Hdiv functions in 2d and 3d it is not needed
                bc_sign = np.sign(n[0:dim])[0]

                q_h_n = bc_sign * alpha_q @ dq_h
                m_h_n = bc_sign * alpha_m @ dm_h

                equ_1_integrand = (1.0 / beta_v) * (
                    q_h_n + m_h_n + beta_v * c_v - gamma_v * c_v
                ) * bc_sign * dq_h.T

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, idx_dof["q"]] = equ_1_integrand
                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

            r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        return r_el, j_el


class TwoFieldsAdvectionWeakForm(WeakForm):
    def evaluate_form(self, cell_c1, element_pair_index, alpha_pair):
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
            dm_dof_n_index = c0_data.dof.entity_dofs[cell_c1.dimension][
                dm_facet_index
            ]
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
            "q_n": slice(n_q_dof + n_m_dof + n_u_dof, 2*n_q_dof + n_m_dof + n_u_dof),
            "m_n": slice(2*n_q_dof + n_m_dof + n_u_dof, 2*n_q_dof + 2*n_m_dof + n_u_dof),
            "u_n": slice(2*n_q_dof + 2*n_m_dof + n_u_dof, 2*n_q_dof + 2*n_m_dof + 2*n_u_dof),
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


class TwoFieldsAdvectionWeakFormBC(WeakForm):
    def evaluate_form(self, cell_c1, element_index, alpha):
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
            dm_dof_n_index = c0_data.dof.entity_dofs[cell_c1.dimension][
                dm_facet_index
            ]
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
