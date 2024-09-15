import auto_diff as ad
import numpy as np

from basis.element_data import ElementData
from basis.parametric_transformation import transform_lower_to_higher
from geometry.compute_normal import normal
from mesh.topological_queries import find_higher_dimension_neighs
from weak_forms.weak_from import WeakForm


class CouplingWeakForm(WeakForm):

    def evaluate_form(self, index_c1, index_c0_p, index_c0_n, alpha):
        iel_c1, iel_c0p, iel_c0n = index_c1, index_c0_p, index_c0_n

        f_kappa = self.functions["kappa_normal"]
        f_delta = self.functions["delta"]

        def compute_trace_space(iel_pair, field_pair, md_space):

            iel_c0, iel_c1 = iel_pair
            field_c0, field_c1 = field_pair

            space_c0 = md_space[0].discrete_spaces[field_c0]
            space_c1 = md_space[1].discrete_spaces[field_c1]
            f_data_c0: ElementData = space_c0.bc_elements[iel_c0].data

            cell_c0 = f_data_c0.cell
            dim = cell_c0.dimension
            points, weights = md_space[0].bc_quadrature[dim]
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

        du_tr_phi_tab_p, du_dof_n_index_p, n_p, p_phi_tab_p, map_data_p = (
            compute_trace_space((iel_c0p, iel_c1), ("u", "p"), self.space)
        )
        du_tr_phi_tab_n, du_dof_n_index_n, n_n, p_phi_tab_n, map_data_n = (
            compute_trace_space((iel_c0n, iel_c1), ("u", "p"), self.space)
        )
        assert np.all(np.isclose(p_phi_tab_p, p_phi_tab_n))
        assert np.all(np.isclose(map_data_p[0], map_data_n[0]))
        assert np.all(np.isclose(map_data_p[1], map_data_n[1]))
        assert np.all(np.isclose(map_data_p[2], map_data_n[2]))
        p_phi_tab = p_phi_tab_p = p_phi_tab_n
        x, weights, det_jac = map_data_p = map_data_n

        dim = self.space[0].discrete_spaces["u"].dimension
        u_c0_components = self.space[0].discrete_spaces["u"].n_comp
        p_c1_components = self.space[1].discrete_spaces["p"].n_comp

        n_du_phi_p = du_tr_phi_tab_p[0, :, du_dof_n_index_p, 0:dim].shape[0]
        n_du_phi_n = du_tr_phi_tab_n[0, :, du_dof_n_index_n, 0:dim].shape[0]
        assert n_du_phi_p == n_du_phi_n

        n_p_phi = p_phi_tab_p.shape[2]
        n_u_dof_p = n_du_phi_p * u_c0_components
        n_u_dof_n = n_du_phi_n * u_c0_components
        n_p_dof = n_p_phi * p_c1_components

        idx_dof = {
            "u_p": slice(0, n_u_dof_p),
            "u_n": slice(n_u_dof_p, n_u_dof_p + n_u_dof_n),
            "p": slice(n_u_dof_p + n_u_dof_n, n_u_dof_p + n_u_dof_n + n_p_dof),
        }
        n_dof = n_u_dof_p + n_u_dof_n + n_p_dof

        # compute normal
        with ad.AutoDiff(alpha) as alpha:
            el_form = np.zeros(n_dof)
            alpha_u_p = alpha[:, idx_dof["u_p"]]
            alpha_u_n = alpha[:, idx_dof["u_n"]]
            alpha_p = alpha[:, idx_dof["p"]]

            for i, omega in enumerate(weights):
                kappa_n_v = f_kappa(x[i, 0], x[i, 1], x[i, 2])
                delta_v = f_delta(x[i, 0], x[i, 1], x[i, 2])

                du_h_p = (
                    du_tr_phi_tab_p[0:1, i, du_dof_n_index_p, 0:dim] @ n_p[0:dim]
                ).T
                du_h_n = (
                    du_tr_phi_tab_n[0:1, i, du_dof_n_index_n, 0:dim] @ n_n[0:dim]
                ).T

                dp_h = p_phi_tab[0, i, :, 0:dim]
                u_h_p = alpha_u_p @ du_h_p
                u_h_n = alpha_u_n @ du_h_n
                p_h_c1 = alpha_p @ dp_h

                alpha_v = kappa_n_v / (delta_v / 2.0)

                # Robin coupling
                equ_1_integrand = ((1.0 / alpha_v) * u_h_p + p_h_c1) @ du_h_p.T
                equ_2_integrand = ((1.0 / alpha_v) * u_h_n + p_h_c1) @ du_h_n.T

                # Robin coupling is a self-adjoint operator of c1 mass conservation
                equ_3_integrand = u_h_p @ dp_h.T + u_h_n @ dp_h.T

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, idx_dof["u_p"]] = equ_1_integrand
                multiphysic_integrand[:, idx_dof["u_n"]] = equ_2_integrand
                multiphysic_integrand[:, idx_dof["p"]] = equ_3_integrand
                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

            r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        return r_el, j_el
