import auto_diff as ad
import numpy as np

from weak_forms.weak_from import WeakForm
from basis.element_data import ElementData
from basis.finite_element import FiniteElement
from geometry.compute_normal import normal
from basis.basis_trace import trace_product_space

from mesh.topological_queries import find_higher_dimension_neighs
from basis.element_family import family_by_name


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
        f_rho_r = self.functions["rho_r"]
        f_cp_r = self.functions["cp_r"]
        f_kappa = self.functions["kappa"]
        f_phi = self.functions["porosity"]
        f_mu = self.functions["mu"]
        dt = self.functions["delta_t"]

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
                v_dofs + p_dof + z_dof + h_dof + t_dof,
                v_dofs + p_dof + z_dof + h_dof + t_dof + sv_dof,
            ),
            "x_H2O_l": slice(
                v_dofs + p_dof + z_dof + h_dof + t_dof + sv_dof,
                v_dofs + p_dof + z_dof + h_dof + t_dof + sv_dof + x_H2O_l_dof,
            ),
            "x_H2O_v": slice(
                v_dofs + p_dof + z_dof + h_dof + t_dof + sv_dof + x_H2O_l_dof,
                v_dofs
                + p_dof
                + z_dof
                + h_dof
                + t_dof
                + sv_dof
                + x_H2O_l_dof
                + x_H2O_v_dof,
            ),
            "x_NaCl_l": slice(
                v_dofs
                + p_dof
                + z_dof
                + h_dof
                + t_dof
                + sv_dof
                + x_H2O_l_dof
                + x_H2O_v_dof,
                v_dofs
                + p_dof
                + z_dof
                + h_dof
                + t_dof
                + sv_dof
                + x_H2O_l_dof
                + x_H2O_v_dof
                + x_NaCl_l_dof,
            ),
            "x_NaCl_v": slice(
                v_dofs
                + p_dof
                + z_dof
                + h_dof
                + t_dof
                + sv_dof
                + x_H2O_l_dof
                + x_H2O_v_dof
                + x_NaCl_l_dof,
                v_dofs
                + p_dof
                + z_dof
                + h_dof
                + t_dof
                + sv_dof
                + x_H2O_l_dof
                + x_H2O_v_dof
                + x_NaCl_l_dof
                + x_NaCl_v_dof,
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

                a_p_n = alpha_n[:, idx_dof["p"]]
                a_z_n = alpha_n[:, idx_dof["z"]]
                a_h_n = alpha_n[:, idx_dof["h"]]
                a_t_n = alpha_n[:, idx_dof["t"]]
                a_sv_n = alpha_n[:, idx_dof["sv"]]
                a_x_H2O_l_n = alpha_n[:, idx_dof["x_H2O_l"]]
                a_x_H2O_v_n = alpha_n[:, idx_dof["x_H2O_v"]]
                a_x_NaCl_l_n = alpha_n[:, idx_dof["x_NaCl_l"]]
                a_x_NaCl_v_n = alpha_n[:, idx_dof["x_NaCl_v"]]

                a_p = alpha[idx_dof["p"]]
                a_z = alpha[idx_dof["z"]]
                a_h = alpha[idx_dof["h"]]
                a_t = alpha[idx_dof["t"]]
                a_sv = alpha[idx_dof["sv"]]

                # FEM approximation
                md_h_n = a_md_n @ dmd_h
                ca_h_n = a_ca_n @ dca_h
                qd_h_n = a_qd_n @ dqd_h
                qa_h_n = a_qa_n @ dqa_h

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
                t_h = a_t @ du_h
                sv_h = a_sv @ du_h

                # fluid and rock data
                rho_l = 1000.0
                rho_v = 1.0
                mu_l = 0.001
                mu_v = 0.00001
                rho_n = sv_h_n * rho_v + (1.0 - sv_h_n) * rho_l
                rho = sv_h * rho_v + (1.0 - sv_h) * rho_l
                rho_r = f_rho_r(xv[0], xv[1], xv[2])
                cp_r = f_cp_r(xv[0], xv[1], xv[2])

                # mobilities and fractional flows
                lambda_H2O_l = (x_H2O_l_h_n * rho_l * (1.0 - sv_h_n) / mu_l)
                lambda_H2O_v = (x_H2O_v_h_n * rho_v * sv_h_n / mu_v)
                lambda_NaCl_l = (x_NaCl_l_h_n * rho_l * (1.0 - sv_h_n) / mu_l)
                lambda_NaCl_v = (x_NaCl_v_h_n * rho_v * sv_h_n / mu_v)
                lambda_H2O = (lambda_H2O_l + lambda_H2O_v)
                lambda_NaCl = (lambda_NaCl_l + lambda_NaCl_v)
                lambda_m = lambda_H2O + lambda_NaCl

                md_h_n *= 1.0 / (lambda_m * f_kappa(xv[0], xv[1], xv[2]))
                qd_h_n *= 1.0 / f_K_thermal(xv[0], xv[1], xv[2])
                phi_v = f_phi(xv[0], xv[1], xv[2])

                div_md_h = a_md_n @ div_dv_h.T
                div_ca_h = a_ca_n @ div_dv_h.T
                div_qd_h = a_qd_n @ div_dv_h.T
                div_qa_h = a_qa_n @ div_dv_h.T


                # accumulation terms
                overall_mass = (1.0/dt) * phi_v * (rho_n - rho)
                mass_z = (1.0/dt) * phi_v * (rho_n * z_h_n - rho * z_h)

                p_work = phi_v * (p_h_n-p_h)
                solid_energy = (1.0-phi_v) * rho_r * cp_r *(t_h_n - t_h)
                fluid_energy = phi_v * (rho_n * h_h_n - rho * h_h)
                energy = (1.0 / dt) * (fluid_energy + solid_energy - p_work)

                equ_1_integrand = (md_h_n @ dv_h.T) - (p_h_n @ div_dv_h)
                # equ_2_integrand = (ca_h_n @ dv_h.T) - (z_h_n @ div_dv_h)
                equ_3_integrand = (qd_h_n @ dv_h.T) - (t_h_n @ div_dv_h)
                # equ_4_integrand = (qa_h_n @ dv_h.T) - 0.0 * (h_h_n @ div_dv_h)
                equ_5_integrand = (div_md_h + overall_mass) @ du_h.T
                equ_6_integrand = (div_ca_h + mass_z) @ du_h.T
                equ_7_integrand = (div_qd_h + div_qa_h + energy) @ du_h.T
                equ_8_integrand = (t_h_n - 0.25 * h_h_n) @ du_h.T
                equ_9_integrand = (sv_h_n - 0.0) @ du_h.T
                equ_10_integrand = (x_H2O_l_h_n - (1.0 - z_h_n)) @ du_h.T
                equ_11_integrand = (x_H2O_v_h_n - (1.0 - z_h_n)) @ du_h.T
                equ_12_integrand = (x_NaCl_l_h_n - z_h_n) @ du_h.T
                equ_13_integrand = (x_NaCl_v_h_n - z_h_n) @ du_h.T

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, idx_dof["md"]] = equ_1_integrand
                # multiphysic_integrand[:, idx_dof["ca"]] = equ_2_integrand
                multiphysic_integrand[:, idx_dof["qd"]] = equ_3_integrand
                # multiphysic_integrand[:, idx_dof["qa"]] = equ_4_integrand
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

        f_beta_md = self.functions["beta_md"]
        f_gamma_md = self.functions["gamma_md"]
        f_p_D = self.functions["p_D"]

        f_beta_qd = self.functions["beta_qd"]
        f_gamma_qd = self.functions["gamma_qd"]
        f_t_D = self.functions["t_D"]

        md_space = self.space.discrete_spaces["md"]
        n_components = md_space.n_comp
        md_data: ElementData = md_space.bc_elements[iel].data

        cell = md_data.cell
        points, weights = self.space.bc_quadrature
        x, jac, det_jac, inv_jac = md_space.bc_elements[iel].evaluate_mapping(points)

        fields = ["md", "ca", "qd", "qa"]
        traces = trace_product_space(fields, self.space, points, md_data, False)

        n_md_dof = traces["md"].shape[2] * n_components
        n_cd_dof = traces["ca"].shape[2] * n_components
        n_qd_dof = traces["qd"].shape[2] * n_components
        n_qa_dof = traces["qa"].shape[2] * n_components

        idx_dof = {
            "md": slice(0, n_md_dof),
            "ca": slice(n_md_dof, n_md_dof + n_cd_dof),
            "qd": slice(n_md_dof + n_cd_dof, n_md_dof + n_cd_dof + n_qd_dof),
            "qa": slice(
                n_md_dof + n_cd_dof + n_qd_dof,
                n_md_dof + n_cd_dof + n_qd_dof + n_qa_dof,
            ),
        }

        n_dof = n_md_dof + n_cd_dof + n_qd_dof + n_qa_dof

        # compute normal
        neigh_list = find_higher_dimension_neighs(cell, md_space.dof_map.mesh_topology)
        neigh_check = len(neigh_list) > 0
        assert neigh_check
        # select neighbor
        neigh_cell = md_data.mesh.cells[neigh_list[0]]
        dim = neigh_cell.dimension
        n = normal(md_data.mesh, neigh_cell, cell)

        with ad.AutoDiff(alpha) as alpha:
            el_form = np.zeros(n_dof)

            alpha_md = alpha[:, idx_dof["md"]]
            # alpha_ca = alpha[:, idx_dof["ca"]]
            alpha_qd = alpha[:, idx_dof["qd"]]
            alpha_qa = alpha[:, idx_dof["qa"]]

            tr_dmq_h = traces["md"]
            # tr_dca_h = traces['ca']
            tr_dqd_h = traces["qd"]
            tr_dqa_h = traces["qa"]

            for i, omega in enumerate(weights):
                beta_md_v = f_beta_md(x[i, 0], x[i, 1], x[i, 2])
                gamma_md_v = f_gamma_md(x[i, 0], x[i, 1], x[i, 2])
                p_v = f_p_D(x[i, 0], x[i, 1], x[i, 2])

                beta_qd_v = f_beta_qd(x[i, 0], x[i, 1], x[i, 2])
                gamma_qd_v = f_gamma_qd(x[i, 0], x[i, 1], x[i, 2])
                t_v = f_t_D(x[i, 0], x[i, 1], x[i, 2])

                dmd_h = tr_dmq_h[0:1, i, :, :] @ n
                # dca_h = tr_dca_h[0:1, i, :, :] @ n
                dqd_h = tr_dqd_h[0:1, i, :, :] @ n
                dqa_h = tr_dqa_h[0:1, i, :, :] @ n

                md_h_n = alpha_md @ dmd_h
                qd_h_n = alpha_qd @ dqd_h
                qa_h_n = alpha_qa @ dqa_h

                equ_1_integrand = (
                    (1.0 / beta_md_v) * (md_h_n + beta_md_v * p_v - gamma_md_v)
                    * dmd_h.T
                )

                equ_2_integrand = (
                    (1.0 / beta_qd_v) * (qd_h_n + qa_h_n + beta_qd_v * t_v - gamma_qd_v)
                    * dqd_h.T
                )

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, idx_dof["md"]] = equ_1_integrand
                multiphysic_integrand[:, idx_dof["qd"]] = equ_2_integrand
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

        md_space = self.space.discrete_spaces["md"]
        ca_space = self.space.discrete_spaces["ca"]
        qd_space = self.space.discrete_spaces["qd"]
        qa_space = self.space.discrete_spaces["qa"]
        z_space = self.space.discrete_spaces["z"]
        h_space = self.space.discrete_spaces["h"]

        n_components = ca_space.n_comp

        md_data_p: ElementData = md_space.elements[iel_p].data
        ca_data_p: ElementData = ca_space.elements[iel_p].data
        qd_data_p: ElementData = qd_space.elements[iel_p].data
        qa_data_p: ElementData = qa_space.elements[iel_p].data
        z_data_p: ElementData = z_space.elements[iel_p].data
        h_data_p: ElementData = h_space.elements[iel_p].data

        md_data_n: ElementData = md_space.elements[iel_n].data
        ca_data_n: ElementData = ca_space.elements[iel_n].data
        qd_data_n: ElementData = qd_space.elements[iel_n].data
        qa_data_n: ElementData = qa_space.elements[iel_n].data
        z_data_n: ElementData = z_space.elements[iel_n].data
        h_data_n: ElementData = h_space.elements[iel_n].data

        dim = ca_data_p.cell.dimension

        # trace of qh on both sides
        gmesh = ca_data_p.mesh
        c1_cell = gmesh.cells[cell_id]
        element_c1_data = ElementData(c1_cell, gmesh)
        points, weights = self.space.bc_quadrature
        c1_element = FiniteElement(
            cell_id, family_by_name("Lagrange"), 0, gmesh, True, 0
        )
        x, _, det_jac, _ = c1_element.evaluate_mapping(points)

        # compute normal
        neigh_list = find_higher_dimension_neighs(
            c1_cell, md_space.dof_map.mesh_topology
        )
        neigh_check = len(neigh_list) > 0
        assert neigh_check
        # select neighbor
        neigh_cell = md_data_p.mesh.cells[neigh_list[0]]
        n_p = normal(md_data_p.mesh, neigh_cell, c1_cell)
        neigh_cell = md_data_p.mesh.cells[neigh_list[1]]
        n_n = normal(md_data_p.mesh, neigh_cell, c1_cell)

        # compute trace u space
        fields = ["md", "ca", "qd", "qa"]
        traces_v_p = trace_product_space(
            fields, self.space, points, c1_element.data, True, 0
        )
        traces_v_n = trace_product_space(
            fields, self.space, points, c1_element.data, True, 1
        )
        fields = ["h", "z"]
        traces_s_p = trace_product_space(
            fields, self.space, points, c1_element.data, True, 0
        )
        traces_s_n = trace_product_space(
            fields, self.space, points, c1_element.data, True, 1
        )

        n_hdiv_dof = md_data_p.dof.n_dof
        n_dg_dof = z_data_p.dof.n_dof

        n_dmd_dof, n_dca_dof, n_dqd_dof, n_dqa_dof = 4 * [n_hdiv_dof]
        n_dp_dof, n_dz_dof, n_dh_dof = 3 * [n_dg_dof]
        n_dof_p = 4 * n_hdiv_dof + 9 * n_dg_dof

        idx_dof = {
            "md_p": slice(0, n_dmd_dof),
            "ca_p": slice(n_dmd_dof, n_dmd_dof + n_dca_dof),
            "qd_p": slice(n_dmd_dof + n_dca_dof, n_dmd_dof + n_dca_dof + n_dqd_dof),
            "qa_p": slice(
                n_dmd_dof + n_dca_dof + n_dqd_dof,
                n_dmd_dof + n_dca_dof + n_dqd_dof + n_dqa_dof,
            ),
            "p_p": slice(
                n_dmd_dof + n_dca_dof + n_dqd_dof + n_dqa_dof,
                n_dmd_dof + n_dca_dof + n_dqd_dof + n_dqa_dof + n_dp_dof,
            ),
            "z_p": slice(
                n_dmd_dof + n_dca_dof + n_dqd_dof + n_dqa_dof + n_dp_dof,
                n_dmd_dof + n_dca_dof + n_dqd_dof + n_dqa_dof + n_dp_dof + n_dz_dof,
            ),
            "h_p": slice(
                n_dmd_dof + n_dca_dof + n_dqd_dof + n_dqa_dof + n_dp_dof + n_dz_dof,
                n_dmd_dof
                + n_dca_dof
                + n_dqd_dof
                + n_dqa_dof
                + n_dp_dof
                + n_dz_dof
                + n_dh_dof,
            ),
            "md_n": slice(n_dof_p, n_dof_p + n_dmd_dof),
            "ca_n": slice(n_dof_p + n_dmd_dof, n_dof_p + n_dmd_dof + n_dca_dof),
            "qd_n": slice(
                n_dof_p + n_dmd_dof + n_dca_dof,
                n_dof_p + n_dmd_dof + n_dca_dof + n_dqd_dof,
            ),
            "qa_n": slice(
                n_dof_p + n_dmd_dof + n_dca_dof + n_dqd_dof,
                n_dof_p + n_dmd_dof + n_dca_dof + n_dqd_dof + n_dqa_dof,
            ),
            "p_n": slice(
                n_dof_p + n_dmd_dof + n_dca_dof + n_dqd_dof + n_dqa_dof,
                n_dof_p + n_dmd_dof + n_dca_dof + n_dqd_dof + n_dqa_dof + n_dp_dof,
            ),
            "z_n": slice(
                n_dof_p + n_dmd_dof + n_dca_dof + n_dqd_dof + n_dqa_dof + n_dp_dof,
                n_dof_p
                + n_dmd_dof
                + n_dca_dof
                + n_dqd_dof
                + n_dqa_dof
                + n_dp_dof
                + n_dz_dof,
            ),
            "h_n": slice(
                n_dof_p
                + n_dmd_dof
                + n_dca_dof
                + n_dqd_dof
                + n_dqa_dof
                + n_dp_dof
                + n_dz_dof,
                n_dof_p
                + n_dmd_dof
                + n_dca_dof
                + n_dqd_dof
                + n_dqa_dof
                + n_dp_dof
                + n_dz_dof
                + n_dh_dof,
            ),
        }

        n_dof = 2 * n_dof_p
        alpha = np.concatenate((alpha_p, alpha_n))
        with ad.AutoDiff(alpha) as alpha:
            el_form = np.zeros(n_dof)

            for i, omega in enumerate(weights):
                alpha_md_p = alpha[:, idx_dof["md_p"]]
                alpha_ca_p = alpha[:, idx_dof["ca_p"]]
                alpha_qd_p = alpha[:, idx_dof["qd_p"]]
                alpha_qa_p = alpha[:, idx_dof["qa_p"]]
                alpha_md_n = alpha[:, idx_dof["md_n"]]
                alpha_ca_n = alpha[:, idx_dof["ca_n"]]
                alpha_qd_n = alpha[:, idx_dof["qd_n"]]
                alpha_qa_n = alpha[:, idx_dof["qa_n"]]
                alpha_z_p = alpha[:, idx_dof["z_p"]]
                alpha_z_n = alpha[:, idx_dof["z_n"]]
                alpha_h_p = alpha[:, idx_dof["h_p"]]
                alpha_h_n = alpha[:, idx_dof["h_n"]]

                dmd_p_h = traces_v_p["md"][0:1, i, :, :] @ n_p
                dca_p_h = traces_v_p["ca"][0:1, i, :, :] @ n_p
                dqd_p_h = traces_v_p["qd"][0:1, i, :, :] @ n_p
                dqa_p_h = traces_v_p["qa"][0:1, i, :, :] @ n_p

                dmd_n_h = traces_v_n["md"][0:1, i, :, :] @ n_n
                dca_n_h = traces_v_n["ca"][0:1, i, :, :] @ n_n
                dqd_n_h = traces_v_n["qd"][0:1, i, :, :] @ n_n
                dqa_n_h = traces_v_n["qa"][0:1, i, :, :] @ n_n

                md_p_h = alpha_md_p @ dmd_p_h.T
                ca_p_h = alpha_ca_p @ dca_p_h.T
                qd_p_h = alpha_qd_p @ dqd_p_h.T
                qa_p_h = alpha_qa_p @ dqa_p_h.T

                md_n_h = alpha_md_n @ dmd_n_h.T
                ca_n_h = alpha_ca_n @ dca_n_h.T
                qd_n_h = alpha_qd_n @ dqd_n_h.T
                qa_n_h = alpha_qa_n @ dqa_n_h.T

                dz_h_p, dz_h_n = (
                    traces_s_p["z"][0, i, :, :],
                    traces_s_n["z"][0, i, :, :],
                )
                dh_h_p, dh_h_n = (
                    traces_s_p["h"][0, i, :, :],
                    traces_s_n["h"][0, i, :, :],
                )
                z_h_p, z_h_n = alpha_z_p @ dz_h_p, alpha_z_n @ dz_h_n
                h_h_p, h_h_n = alpha_h_p @ dh_h_p, alpha_h_n @ dh_h_n

                md_n = md_p_h.val[0, 0]
                beta_upwind = 0.0
                if md_n > 0.0 or np.isclose(md_n, 0.0):
                    beta_upwind = 1.0

                z_h_upwind = (1.0 - beta_upwind) * z_h_n + beta_upwind * z_h_p
                h_h_upwind = (1.0 - beta_upwind) * h_h_n + beta_upwind * h_h_p

                equ_1_integrand = +0.5 * (ca_p_h - z_h_upwind * md_p_h) @ dca_p_h
                equ_2_integrand = +0.5 * (ca_n_h - z_h_upwind * md_n_h) @ dca_n_h
                equ_3_integrand = +0.5 * (qa_p_h - h_h_upwind * md_p_h) @ dqa_p_h
                equ_4_integrand = +0.5 * (qa_n_h - h_h_upwind * md_n_h) @ dqa_n_h

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, idx_dof["ca_p"]] = equ_1_integrand
                multiphysic_integrand[:, idx_dof["ca_n"]] = equ_2_integrand
                multiphysic_integrand[:, idx_dof["qa_p"]] = equ_3_integrand
                multiphysic_integrand[:, idx_dof["qa_n"]] = equ_4_integrand
                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        return r_el, j_el


class AdvectionWeakFormBC(WeakForm):
    def evaluate_form(self, cell_id, element_index, alpha):
        iel = element_index
        if self.space is None or self.functions is None:
            raise ValueError

        f_z = self.functions["z_inlet"]
        f_h = self.functions["h_inlet"]

        md_space = self.space.discrete_spaces["md"]
        ca_space = self.space.discrete_spaces["ca"]
        qd_space = self.space.discrete_spaces["qd"]
        qa_space = self.space.discrete_spaces["qa"]
        z_space = self.space.discrete_spaces["z"]
        h_space = self.space.discrete_spaces["h"]

        n_components = ca_space.n_comp

        md_data: ElementData = md_space.elements[iel].data
        ca_data: ElementData = ca_space.elements[iel].data
        qd_data: ElementData = qd_space.elements[iel].data
        qa_data: ElementData = qa_space.elements[iel].data
        z_data: ElementData = z_space.elements[iel].data
        h_data: ElementData = h_space.elements[iel].data

        dim = ca_data.cell.dimension

        # trace of qh on both sides
        gmesh = ca_data.mesh
        c1_cell = gmesh.cells[cell_id]
        element_c1_data = ElementData(c1_cell, gmesh)
        points, weights = self.space.bc_quadrature
        c1_element = FiniteElement(
            cell_id, family_by_name("Lagrange"), 0, gmesh, True, 0
        )
        x, _, det_jac, _ = c1_element.evaluate_mapping(points)

        # compute normal
        neigh_list = find_higher_dimension_neighs(
            c1_cell, md_space.dof_map.mesh_topology
        )
        neigh_check = len(neigh_list) > 0
        assert neigh_check
        # select neighbor
        neigh_cell = md_data.mesh.cells[neigh_list[0]]
        dim = neigh_cell.dimension
        n = normal(md_data.mesh, neigh_cell, c1_cell)

        # compute trace u space
        fields = ["md", "ca", "qd", "qa"]
        traces_v = trace_product_space(
            fields, self.space, points, c1_element.data, True
        )
        fields = ["h", "z"]
        traces_s = trace_product_space(
            fields, self.space, points, c1_element.data, True, 0
        )

        n_hdiv_dof = md_data.dof.n_dof
        n_dg_dof = z_data.dof.n_dof

        n_dmd_dof, n_dca_dof, n_dqd_dof, n_dqa_dof = 4 * [n_hdiv_dof]
        n_dp_dof, n_dz_dof, n_dh_dof = 3 * [n_dg_dof]

        idx_dof = {
            "md": slice(0, n_dmd_dof),
            "ca": slice(n_dmd_dof, n_dmd_dof + n_dca_dof),
            "qd": slice(n_dmd_dof + n_dca_dof, n_dmd_dof + n_dca_dof + n_dqd_dof),
            "qa": slice(
                n_dmd_dof + n_dca_dof + n_dqd_dof,
                n_dmd_dof + n_dca_dof + n_dqd_dof + n_dqa_dof,
            ),
            "p": slice(
                n_dmd_dof + n_dca_dof + n_dqd_dof + n_dqa_dof,
                n_dmd_dof + n_dca_dof + n_dqd_dof + n_dqa_dof + n_dp_dof,
            ),
            "z": slice(
                n_dmd_dof + n_dca_dof + n_dqd_dof + n_dqa_dof + n_dp_dof,
                n_dmd_dof + n_dca_dof + n_dqd_dof + n_dqa_dof + n_dp_dof + n_dz_dof,
            ),
            "h": slice(
                n_dmd_dof + n_dca_dof + n_dqd_dof + n_dqa_dof + n_dp_dof + n_dz_dof,
                n_dmd_dof
                + n_dca_dof
                + n_dqd_dof
                + n_dqa_dof
                + n_dp_dof
                + n_dz_dof
                + n_dh_dof,
            ),
        }

        n_dof = 4 * n_hdiv_dof + 9 * n_dg_dof

        with ad.AutoDiff(alpha) as alpha:
            el_form = np.zeros(n_dof)

            for i, omega in enumerate(weights):
                alpha_md = alpha[:, idx_dof["md"]]
                alpha_ca = alpha[:, idx_dof["ca"]]
                alpha_qd = alpha[:, idx_dof["qd"]]
                alpha_qa = alpha[:, idx_dof["qa"]]
                alpha_z = alpha[:, idx_dof["z"]]
                alpha_h = alpha[:, idx_dof["h"]]

                dmd_h = traces_v["md"][0:1, i, :, :] @ n
                dca_h = traces_v["ca"][0:1, i, :, :] @ n
                dqd_h = traces_v["qd"][0:1, i, :, :] @ n
                dqa_h = traces_v["qa"][0:1, i, :, :] @ n

                md_h = alpha_md @ dmd_h.T
                ca_h = alpha_ca @ dca_h.T
                qd_h = alpha_qd @ dqd_h.T
                qa_h = alpha_qa @ dqa_h.T

                dz_h = traces_s["z"][0, i, :, :]
                dh_h = traces_s["h"][0, i, :, :]
                z_h = alpha_z @ dz_h
                h_h = alpha_h @ dh_h

                md_n = md_h.val[0, 0]
                beta_upwind = 0.0
                if md_n > 0.0 or np.isclose(md_n, 0.0):
                    beta_upwind = 1.0

                z_v = f_z(x[i, 0], x[i, 1], x[i, 2])
                h_v = f_h(x[i, 0], x[i, 1], x[i, 2])

                z_h_upwind = (1.0 - beta_upwind) * z_v + beta_upwind * z_h
                h_h_upwind = (1.0 - beta_upwind) * h_v + beta_upwind * h_h

                equ_1_integrand = (ca_h - z_h_upwind * md_h) @ dca_h
                equ_2_integrand = (qa_h - h_h_upwind * md_h) @ dqa_h

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, idx_dof["ca"]] = equ_1_integrand
                multiphysic_integrand[:, idx_dof["qa"]] = equ_2_integrand
                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        return r_el, j_el
