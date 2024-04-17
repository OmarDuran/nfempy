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


class SundusDualWeakForm(WeakForm):
    def evaluate_form(self, element_index, alpha_n_p_1, alpha_n, t):
        iel = element_index
        if self.space is None or self.functions is None:
            raise ValueError

        mp_space = self.space.discrete_spaces["mp"] # flux for pressure
        mc_space = self.space.discrete_spaces["mc"] # flux  for concentration
        p_space = self.space.discrete_spaces["p"]
        c_space = self.space.discrete_spaces["c"]

        f_rhs_f = self.functions["rhs_f"]
        f_kappa = self.functions["kappa"]
        f_rhs_r = self.functions["rhs_r"]
        f_delta = self.functions["delta"]
        f_eta = self.functions["eta"]
        delta_t = self.functions["delta_t"]


        mp_components = mp_space.n_comp  # mass flux for pressure
        mc_components = mc_space.n_comp  # mass flux for concentration

        p_components = p_space.n_comp
        c_components = c_space.n_comp
        mp_data: ElementData = mp_space.elements[iel].data
        mc_data: ElementData = mc_space.elements[iel].data
        p_data: ElementData = p_space.elements[iel].data
        c_data: ElementData = c_space.elements[iel].data

        cell = mp_data.cell
        dim = mp_data.dimension
        points, weights = self.space.quadrature
        bc_points, bc_weights = self.space.bc_quadrature
        x, jac, det_jac, inv_jac = mp_space.elements[iel].evaluate_mapping(points)

        # basis
        vp_h_tab = mp_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        vc_h_tab = mc_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        wp_h_tab = p_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        wc_h_tab = c_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)

        n_vp_h = vp_h_tab.shape[2]
        n_vc_h = vc_h_tab.shape[2]
        n_wp_h = wp_h_tab.shape[2]
        n_wc_h = wc_h_tab.shape[2]

        n_mp_dof = n_vp_h * mp_components
        n_mc_dof = n_vc_h * mc_components

        n_p_dof = n_wp_h * p_components
        n_c_dof = n_wc_h * c_components

        idx_dof = {
            "mp" : slice(0,n_mp_dof),
            "mc" : slice(n_mp_dof, n_mp_dof + n_mc_dof),
            "p" : slice(n_mp_dof + n_mc_dof, n_mp_dof + n_mc_dof + n_p_dof),
            "c": slice(n_mp_dof + n_mc_dof + n_p_dof, n_mp_dof + n_mc_dof + n_p_dof + n_c_dof),
        }

        n_dof = n_mp_dof + n_mc_dof + n_p_dof + n_c_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # Partial local vectorization
        f_f_val_star = f_rhs_f(x[:, 0], x[:, 1], x[:, 2], t)
        f_r_val_star = f_rhs_r(x[:, 0], x[:, 1], x[:, 2],t)


        wp_h_star = det_jac * weights * wp_h_tab[0, :, :, 0].T
        wc_h_star = det_jac * weights * wc_h_tab[0, :, :, 0].T

        # information needed for the advection term
        # compute normals
        cells_co_dim_1 = mp_data.mesh.cells[cell.sub_cells_ids[dim - 1]]
        ns = [normal(mp_data.mesh, cell, cell_co_dim_1) for cell_co_dim_1 in
              cells_co_dim_1]
        # compute element data on boundary cells
        co_dim_1_data = [
            ElementData(cell_co_dim_1.dimension, cell_co_dim_1, mp_data.mesh) for
            cell_co_dim_1 in cells_co_dim_1]

        # map integration rule to boundary of omega
        mapped_points_on_facets = [
            transform_lower_to_higher(bc_points, data, mp_data) for data in
            co_dim_1_data]

        # compute traces per face
        vp_h_tabs = []
        wc_h_tabs = []
        mp_dof_n_idxs = []
        for facet_index, mapped_points in enumerate(mapped_points_on_facets):
            _, tr_jac, tr_det_jac, tr_inv_jac = mp_space.elements[iel].evaluate_mapping(
                mapped_points
            )
            tr_vp_h_tab = mp_space.elements[iel].evaluate_basis(mapped_points, tr_jac,
                                                             tr_det_jac, tr_inv_jac)
            vp_h_tabs.append(tr_vp_h_tab)
            mp_dof_n_idxs.append(
                np.array([mp_data.dof.entity_dofs[dim - 1][facet_index]]))
            tr_wc_h_tab = c_space.elements[iel].evaluate_basis(mapped_points, tr_jac,
                                                            tr_det_jac, tr_inv_jac)
            wc_h_tabs.append(tr_wc_h_tab)

        # constant directors
        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])
        with ad.AutoDiff(alpha_n_p_1) as alpha_n_p_1:
            el_form = np.zeros(n_dof)
            for c in range(p_components):
                b = c + n_mp_dof + n_mc_dof
                e = b + n_p_dof
                el_form[b:e:p_components] -= wp_h_star @ f_f_val_star[c].T

            for c in range(c_components):
                b = c + n_mp_dof + n_mc_dof + n_p_dof
                e = b + n_c_dof
                el_form[b:e:p_components] -= wc_h_star @ f_r_val_star[c].T

            # integrals over omega
            for i, omega in enumerate(weights):
                xv = x[i]
                if dim == 2:
                    inv_jac_m = np.vstack((inv_jac[i] @ e1, inv_jac[i] @ e2))
                else:
                    inv_jac_m = np.vstack(
                        (inv_jac[i] @ e1, inv_jac[i] @ e2, inv_jac[i] @ e3)
                    )

                # Functions and derivatives at integration point i
                vp_h = vp_h_tab[0, i, :, 0:dim]
                vc_h = vc_h_tab[0, i, :, 0:dim]
                wp_h = wp_h_tab[0, i, :, 0:dim]
                wc_h = wc_h_tab[0, i, :, 0:dim]
                grad_vp_h = vp_h_tab[1: vp_h_tab.shape[0] + 1, i, :, 0:dim]
                grad_vc_h = vc_h_tab[1: vc_h_tab.shape[0] + 1, i, :, 0:dim]
                div_vp_h = np.array(
                    [[np.trace(grad_vp_h[:, j, :]) / det_jac[i] for j in range(n_mp_dof)]]
                )
                div_vc_h = np.array(
                    [[np.trace(grad_vc_h[:, j, :]) / det_jac[i] for j in range(n_mc_dof)]]
                )

                # Dof per field
                alpha_mp_n_p_1 = alpha_n_p_1[:, idx_dof['mp']]
                alpha_mc_n_p_1 = alpha_n_p_1[:, idx_dof['mc']]
                alpha_p_n_p_1 = alpha_n_p_1[:, idx_dof['p']]
                alpha_c_n_p_1 = alpha_n_p_1[:, idx_dof['c']]

                alpha_p_n = alpha_n[idx_dof['p']]
                alpha_c_n = alpha_n[idx_dof['c']]


                # FEM approximation
                mp_h_n_p_1 = alpha_mp_n_p_1 @ vp_h
                mc_h_n_p_1 = alpha_mc_n_p_1 @ vc_h
                p_h_n_p_1 = alpha_p_n_p_1 @ wp_h
                c_h_n_p_1 = alpha_c_n_p_1 @ wc_h

                p_h_n = alpha_p_n @ wp_h
                c_h_n = alpha_c_n @ wc_h

                div_mp_h = alpha_mp_n_p_1 @ div_vp_h.T
                div_mc_h = alpha_mc_n_p_1 @ div_vc_h.T

                mp_h_n_p_1 *= 1.0 / f_kappa(xv[0], xv[1], xv[2])
                mc_h_n_p_1*= 1.0 / f_delta(xv[0], xv[1], xv[2])

                # Example of reaction term
                R_h = (1.0 + f_eta(xv[0], xv[1], xv[2]) * c_h_n_p_1 * c_h_n_p_1)
                dph_dt = (p_h_n_p_1 - p_h_n)/ delta_t
                dch_dt = (c_h_n_p_1 - c_h_n) / delta_t

                equ_1_integrand = (mp_h_n_p_1 @ vp_h.T) - (p_h_n_p_1 @ div_vp_h)
                equ_2_integrand = (mc_h_n_p_1 @ vc_h.T) - (c_h_n_p_1 @ div_vc_h)

                equ_3_integrand = dph_dt @ wp_h.T + div_mp_h @ wp_h.T
                equ_4_integrand = dch_dt @ wc_h.T + div_mc_h @ wc_h.T + R_h @ wc_h.T

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, idx_dof['mp']] = equ_1_integrand
                multiphysic_integrand[:, idx_dof['mc']] = equ_2_integrand
                multiphysic_integrand[:, idx_dof['p']] = equ_3_integrand
                multiphysic_integrand[:, idx_dof['c']] = equ_4_integrand
                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

            # Advection term
            # integrals over gamma
            for facet_index, mp_dof_n in enumerate(mp_dof_n_idxs):
                # Dof per field
                alpha_mp_n_p_1 = alpha_n_p_1[:, idx_dof['mp']][:, mp_dof_n.ravel()]
                alpha_c_n_p_1 = alpha_n_p_1[:, idx_dof['c']]
                vp_h_tab = vp_h_tabs[facet_index]
                wc_h_tab = wc_h_tabs[facet_index]
                n = ns[facet_index]
                for i, omega in enumerate(bc_weights):
                    # Functions and derivatives at integration point i
                    vp_n_h = vp_h_tab[0, i, mp_dof_n, 0:dim] @ n[0:dim]
                    wc_h = wc_h_tab[0, i, :, 0:dim]
                    mp_h_n_p_1 = alpha_mp_n_p_1 @ vp_n_h
                    c_h_n_p_1 = alpha_c_n_p_1 @ wc_h

                    beta = 0.0
                    dir_q = mp_h_n_p_1.val[0, 0] >= 0.0
                    if dir_q:
                        beta = 1.0
                    else:
                        beta = 0.0

                    # Example of nonlinear advection function
                    f_h_n_p_1 = c_h_n_p_1*c_h_n_p_1

                    equ_4_integrand = beta * (f_h_n_p_1 * mp_h_n_p_1) @ wc_h.T
                    multiphysic_integrand = np.zeros((1, n_dof))
                    multiphysic_integrand[:, idx_dof['c']] = equ_4_integrand
                    discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                    el_form += det_jac[i] * omega * discrete_integrand


        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))



        return r_el, j_el


class SundusDualWeakFormBCDirichlet(WeakForm):
    def evaluate_form(self, element_index, alpha,t):
        iel = element_index
        p_D = self.functions["p"]
        c_D = self.functions["c"]

        mp_space = self.space.discrete_spaces["mp"]
        mc_space = self.space.discrete_spaces["mc"]
        mp_components = mp_space.n_comp
        mp_data: ElementData = mp_space.bc_elements[iel].data
        mc_components = mc_space.n_comp
        mc_data: ElementData = mc_space.bc_elements[iel].data

        cell = mp_data.cell
        points, weights = self.space.bc_quadrature
        dim = mp_data.dimension
        x, jac, det_jac, inv_jac = mp_space.bc_elements[iel].evaluate_mapping(points)

        # find high-dimension neigh q space
        neigh_list = find_higher_dimension_neighs(cell, mp_space.dof_map.mesh_topology)
        neigh_check_mp = len(neigh_list) > 0
        assert neigh_check_mp
        neigh_cell_id = neigh_list[0]
        neigh_cell_index = mp_space.id_to_element[neigh_cell_id]
        neigh_element = mp_space.elements[neigh_cell_index]
        neigh_cell = neigh_element.data.cell

        # compute trace q space
        mapped_points = transform_lower_to_higher(points, mp_data, neigh_element.data)
        _, jac_c0, det_jac_c0, inv_jac_c0 = neigh_element.evaluate_mapping(
            mapped_points
        )
        mp_tr_phi_tab = neigh_element.evaluate_basis(
            mapped_points, jac_c0, det_jac_c0, inv_jac_c0
        )
        mp_facet_index = neigh_cell.sub_cells_ids[cell.dimension].tolist().index(cell.id)
        mp_dof_n_index = neigh_element.data.dof.entity_dofs[cell.dimension][
            mp_facet_index
        ]

        # find high-dimension neigh q space
        neigh_list = find_higher_dimension_neighs(cell, mc_space.dof_map.mesh_topology)
        neigh_check_mp = len(neigh_list) > 0
        assert neigh_check_mp
        neigh_cell_id = neigh_list[0]
        neigh_cell_index = mc_space.id_to_element[neigh_cell_id]
        neigh_element = mc_space.elements[neigh_cell_index]
        neigh_cell = neigh_element.data.cell

        # compute trace u space
        mapped_points = transform_lower_to_higher(points, mc_data, neigh_element.data)
        _, jac_c0, det_jac_c0, inv_jac_c0 = neigh_element.evaluate_mapping(
            mapped_points
        )
        mc_tr_phi_tab = neigh_element.evaluate_basis(
            mapped_points, jac_c0, det_jac_c0, inv_jac_c0
        )
        mc_facet_index = neigh_cell.sub_cells_ids[cell.dimension].tolist().index(cell.id)
        mc_dof_n_index = neigh_element.data.dof.entity_dofs[cell.dimension][
            mp_facet_index
        ]

        n_mp_phi = mp_tr_phi_tab[0, :, mp_dof_n_index, 0:dim].shape[0]
        n_mp_dof = n_mp_phi * mp_components
        n_mc_phi = mc_tr_phi_tab[0, :, mc_dof_n_index, 0:dim].shape[0]
        n_mc_dof = n_mc_phi * mc_components

        n_dof = n_mp_dof + n_mc_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # compute normal
        n = normal(mp_data.mesh, neigh_cell, cell)
        for c in range(mp_components):
            b = c
            e = b + n_mp_dof

            res_block_mp = np.zeros(n_mp_phi)
            dim = neigh_cell.dimension
            for i, omega in enumerate(weights):
                p_D_v = p_D(x[i, 0], x[i, 1], x[i, 2], t)
                phi = mp_tr_phi_tab[0, i, mp_dof_n_index, 0:dim] @ n[0:dim]
                res_block_mp += det_jac[i] * omega * p_D_v[c] * phi

            r_el[b:e:mp_components] += res_block_mp

        for c in range(mc_components):
            b = c + n_mp_dof
            e = b + n_mc_dof

            res_block_mc = np.zeros(n_mc_phi)
            dim = neigh_cell.dimension
            for i, omega in enumerate(weights):
                c_D_v = c_D( x[i, 0], x[i, 1], x[i, 2], t)
                phi = mc_tr_phi_tab[0, i, mc_dof_n_index, 0:dim] @ n[0:dim]
                res_block_mc += det_jac[i] * omega * c_D_v[c] * phi

            r_el[b:e:mp_components] += res_block_mc

        return r_el, j_el
