import auto_diff as ad
import numpy as np

from basis.element_data import ElementData
from basis.parametric_transformation import transform_lower_to_higher
from geometry.compute_normal import normal
from mesh.topological_queries import find_higher_dimension_neighs
from weak_forms.weak_from import WeakForm


class LMgWeakForm(WeakForm):
    def evaluate_form(self, index_c1, index_c0_n, index_c0_p, alpha):
        iel_c1, iel_c0n, iel_c0p = index_c1, index_c0_n, index_c0_p

        # from c0 object
        f_d_phi = self.functions["d_phi_c0n"]

        # from c1 object
        f_porosity = self.functions["porosity_c0n"]

        def compute_trace_space(iel_pair, field_pair, space_idx):

            iel_c0, iel_c1 = iel_pair
            field_c0, field_c1 = field_pair

            space_c0 = self.space[space_idx].discrete_spaces[field_c0]
            space_c1 = self.space[2].discrete_spaces[field_c1]
            f_data_c0: ElementData = space_c0.bc_elements[iel_c0].data

            cell_c0 = f_data_c0.cell
            dim = cell_c0.dimension
            points, weights = self.space[space_idx].bc_quadrature[dim]
            x, jac, det_jac, inv_jac = space_c0.bc_elements[iel_c0].evaluate_mapping(
                points
            )

            # Diffusive flux
            # find high-dimension neigh q space
            neigh_list = find_higher_dimension_neighs(
                cell_c0, space_c0.dof_map.mesh_topology
            )
            neigh_check_mp = len(neigh_list) > 0
            assert neigh_check_mp
            neigh_cell_id = neigh_list[0][1]
            neigh_cell_index = space_c0.id_to_element[neigh_cell_id]
            neigh_element = space_c0.elements[neigh_cell_index]
            neigh_cell = neigh_element.data.cell

            # compute trace q space
            mapped_points = transform_lower_to_higher(
                points, f_data_c0, neigh_element.data
            )
            _, jac_c0, det_jac_c0, inv_jac_c0 = neigh_element.evaluate_mapping(
                mapped_points
            )
            dphi_tr_phi_tab = neigh_element.evaluate_basis(
                mapped_points, jac_c0, det_jac_c0, inv_jac_c0
            )
            dphi_facet_index = (
                neigh_cell.sub_cells_ids[cell_c0.dimension].tolist().index(cell_c0.id)
            )
            dphi_dof_n_index = np.array(
                neigh_element.data.dof.entity_dofs[cell_c0.dimension][dphi_facet_index]
            )
            p_phi_tab = space_c1.elements[iel_c1].evaluate_basis(
                points, jac, det_jac, inv_jac
            )
            n = normal(f_data_c0.mesh, neigh_cell, cell_c0)
            map_data = (x, weights, det_jac)
            return dphi_tr_phi_tab, dphi_dof_n_index, n, p_phi_tab, map_data
        (
            dv_tr_phi_tab_n,
            dv_dof_n_index_n,
            n_n,
            q_phi_tab_n,
            map_data_n,
        ) = compute_trace_space((iel_c0n, iel_c1), ("v", "l"), space_idx=0)
        (
            dv_tr_phi_tab_p,
            dv_dof_n_index_p,
            n_p,
            q_phi_tab_p,
            map_data_p,
        ) = compute_trace_space((iel_c0p, iel_c1), ("u", "l"), space_idx=1)
        assert np.all(np.isclose(q_phi_tab_p, q_phi_tab_n))
        assert np.all(np.isclose(map_data_p[0], map_data_n[0]))
        assert np.all(np.isclose(map_data_p[1], map_data_n[1]))
        assert np.all(np.isclose(map_data_p[2], map_data_n[2]))
        q_phi_tab = q_phi_tab_p = q_phi_tab_n
        x, weights, det_jac = map_data_p = map_data_n

        dim = self.space[0].discrete_spaces["v"].dimension
        v_c0_components = self.space[0].discrete_spaces["v"].n_comp
        u_c0_components = self.space[1].discrete_spaces["u"].n_comp
        q_c1_components = self.space[2].discrete_spaces["l"].n_comp

        n_dv_phi_n = dv_tr_phi_tab_n[0, :, dv_dof_n_index_n, 0:dim].shape[0]
        n_dv_phi_p = dv_tr_phi_tab_p[0, :, dv_dof_n_index_p, 0:dim].shape[0]
        assert n_dv_phi_p == n_dv_phi_n

        n_q_phi = q_phi_tab_p.shape[2]
        n_v_dof_n = n_dv_phi_n * v_c0_components
        n_u_dof_p = n_dv_phi_p * u_c0_components
        n_q_dof = n_q_phi * q_c1_components

        idx_dof = {
            "v_n": slice(0, n_v_dof_n),
            "u_p": slice(n_v_dof_n, n_v_dof_n + n_u_dof_p),
            "l": slice(n_v_dof_n + n_u_dof_p, n_v_dof_n + n_u_dof_p + n_q_dof),
        }
        n_dof = n_v_dof_n + n_u_dof_p + n_q_dof

        # partial vectorization
        porosity_star = f_porosity(x[:, 0], x[:, 1], x[:, 2])
        d_phi_star = f_d_phi(x[:, 0], x[:, 1], x[:, 2])

        # compute normal
        with ad.AutoDiff(alpha) as alpha:
            el_form = np.zeros(n_dof)
            alpha_v_n = alpha[:, idx_dof["v_n"]]
            alpha_u_p = alpha[:, idx_dof["u_p"]]
            alpha_q = alpha[:, idx_dof["l"]]

            for i, omega in enumerate(weights):

                porosity_v = porosity_star[i]
                d_phi_v = d_phi_star[i]

                dv_h_n = (
                    dv_tr_phi_tab_n[0:1, i, dv_dof_n_index_n, 0:dim] @ n_n[0:dim]
                ).T
                du_h_p = (
                    dv_tr_phi_tab_p[0:1, i, dv_dof_n_index_p, 0:dim] @ n_p[0:dim]
                ).T

                phi_scale = np.sqrt(porosity_v)

                dq_h = q_phi_tab[0, i, :, 0:dim]
                v_h_n = alpha_v_n @ dv_h_n
                u_h_p = alpha_u_p @ du_h_p
                q_h_c1 = alpha_q @ dq_h

                equ_1_integrand = ( (d_phi_v) * q_h_c1) @ dv_h_n.T
                equ_2_integrand = ( (d_phi_v) * q_h_c1) @ du_h_p.T
                equ_3_integrand = (d_phi_v) * (v_h_n @ dq_h.T + u_h_p @ dq_h.T)

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, idx_dof["v_n"]] = equ_1_integrand
                multiphysic_integrand[:, idx_dof["u_p"]] = equ_2_integrand
                multiphysic_integrand[:, idx_dof["l"]] = equ_3_integrand
                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

            r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        return r_el, j_el
