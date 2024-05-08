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


class ArbogastDualWeakForm(WeakForm):
    def evaluate_form(self, element_index, alpha):
        iel = element_index
        if self.space is None or self.functions is None:
            raise ValueError

        v_space = self.space.discrete_spaces["v"]
        q_space = self.space.discrete_spaces["q"]

        f_rhs_f = self.functions["rhs_f"]
        f_kappa = self.functions["kappa"]
        f_porosity = self.functions["porosity"]
        f_d_phi = self.functions["d_phi"]
        f_grad_d_phi = self.functions["grad_d_phi"]

        v_components = v_space.n_comp
        q_components  = q_space.n_comp

        v_data: ElementData = v_space.elements[iel].data
        q_data: ElementData = q_space.elements[iel].data

        cell = v_data.cell
        dim = v_data.dimension
        points, weights = self.space.quadrature
        x, jac, det_jac, inv_jac = v_space.elements[iel].evaluate_mapping(points)

        # basis
        v_phi_tab = v_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        q_phi_tab = q_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)

        n_v_phi = v_phi_tab.shape[2]
        n_q_phi  = q_phi_tab.shape[2]

        n_v_dof = n_v_phi * v_components
        n_q_dof = n_q_phi * q_components

        idx_dof = {
            "v" : slice(0,n_v_dof),
            "q" : slice(n_v_dof, n_v_dof + n_q_dof),
        }

        n_dof = n_v_dof + n_q_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # Partial local vectorization

        phi_star = f_porosity(x[:, 0], x[:, 1], x[:, 2])
        # nick name for d_phi
        delta_star = f_d_phi(x[:, 0], x[:, 1], x[:, 2])
        grad_delta_star = f_grad_d_phi(x[:, 0], x[:, 1], x[:, 2])

        f_f_val_star = f_rhs_f(x[:, 0], x[:, 1], x[:, 2])
        phi_q_star = det_jac * weights * q_phi_tab[0, :, :, 0].T

        # constant directors
        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])
        with ad.AutoDiff(alpha) as alpha:
            el_form = np.zeros(n_dof)
            for c in range(q_components):
                b = c + n_v_dof
                e = b + n_q_dof
                el_form[b:e:q_components] -= phi_q_star @ f_f_val_star[c].T

            for i, omega in enumerate(weights):
                xv = x[i]

                phi = phi_star[i]
                delta = delta_star[i]
                grad_delta = grad_delta_star[: , i]

                # Functions and derivatives at integration point i
                psi_h = v_phi_tab[0, i, :, 0:dim]
                w_h = q_phi_tab[0, i, :, 0:dim]

                v_h = alpha[:, idx_dof['v']] @ psi_h
                q_h = alpha[:, idx_dof['q']] @ w_h

                grad_psi_h = v_phi_tab[1 : v_phi_tab.shape[0] + 1, i, :, 0:dim]
                div_psi_h = np.array(
                    [np.trace(grad_psi_h, axis1=0, axis2=2) / det_jac[i]]
                )

                grad_delta_dot_psi_h = np.array(
                    [
                        [
                            np.dot(
                                grad_delta[0:dim], psi_h[j,0:dim]
                            )
                            for j in range(n_v_phi)
                        ]
                    ]
                )
                div_phi_h_s = (delta * div_psi_h + grad_delta_dot_psi_h) / np.sqrt(phi)
                div_delta_v_h = alpha[:, idx_dof['v']] @ div_phi_h_s.T

                equ_1_integrand = (v_h @ psi_h.T) - (q_h @ div_phi_h_s)
                equ_2_integrand = div_delta_v_h @ w_h.T + q_h @ w_h.T

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, idx_dof['v']] = equ_1_integrand
                multiphysic_integrand[:, idx_dof['q']] = equ_2_integrand
                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        return r_el, j_el


class ArbogastDualWeakFormBCDirichlet(WeakForm):
    def evaluate_form(self, element_index, alpha):
        iel = element_index
        q_D = self.functions["q"]

        mp_space = self.space.discrete_spaces["v"]
        mp_components = mp_space.n_comp
        mp_data: ElementData = mp_space.bc_elements[iel].data

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

        n_mp_phi = mp_tr_phi_tab[0, :, mp_dof_n_index, 0:dim].shape[0]
        n_mp_dof = n_mp_phi * mp_components

        n_dof = n_mp_dof
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
                q_D_v = q_D(x[i, 0], x[i, 1], x[i, 2])
                phi = mp_tr_phi_tab[0, i, mp_dof_n_index, 0:dim] @ n[0:dim]
                res_block_mp += det_jac[i] * omega * q_D_v[c] * phi

            r_el[b:e:mp_components] += res_block_mp

        return r_el, j_el
