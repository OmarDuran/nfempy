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

        mp_space = self.space.discrete_spaces["mp"] #flux for pressure
        p_space = self.space.discrete_spaces["p"]

        f_rhs_f = self.functions["rhs_f"]
        f_kappa = self.functions["kappa"]
        f_porosity = self.functions["porosity"]
        f_d_phi = self.functions["d_phi"]
        


        mp_components = mp_space.n_comp # mass flux for pressure
        p_components  = p_space.n_comp

        mp_data: ElementData = mp_space.elements[iel].data
        p_data: ElementData = p_space.elements[iel].data

        cell = mp_data.cell
        dim = mp_data.dimension
        points, weights = self.space.quadrature
        x, jac, det_jac, inv_jac = mp_space.elements[iel].evaluate_mapping(points)

        # basis
        mp_phi_tab = mp_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        p_phi_tab = p_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)

        n_mp_phi = mp_phi_tab.shape[2]
        n_p_phi  = p_phi_tab.shape[2]

        n_mp_dof = n_mp_phi * mp_components
        n_p_dof  = n_p_phi * p_components

        n_dof = n_mp_dof + n_p_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # Partial local vectorization
        f_f_val_star = f_rhs_f(x[:, 0], x[:, 1], x[:, 2])

        phi_p_star = det_jac * weights * p_phi_tab[0, :, :, 0].T

        # constant directors
        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])
        with ad.AutoDiff(alpha) as alpha:
            el_form = np.zeros(n_dof)
            for c in range(p_components):
                b = c + n_mp_dof
                e = b + n_p_dof
                el_form[b:e:p_components] -= phi_p_star @ f_f_val_star[c].T

            for i, omega in enumerate(weights):
                xv = x[i]
                mp_h = alpha[:, 0:n_mp_dof:1] @ mp_phi_tab[0, i, :, 0:dim]
                #mp_h *=  1.0 / (f_d_phi(xv[0], xv[1], xv[2])**2)
                v = (1.0 / ((f_d_phi(xv[0], xv[1], xv[2])))) * mp_h

                p_h = (alpha[:, n_mp_dof : n_mp_dof + n_p_dof : 1] @ p_phi_tab[0, i, :, 0:dim])
                q = np.sqrt(f_porosity(xv[0], xv[1], xv[2])) * p_h

                grad_mp_h = mp_phi_tab[1 : mp_phi_tab.shape[0] + 1, i, :, 0:dim]
                div_vc_h = np.array(
                    [[np.trace(grad_mp_h[:, j, :]) / det_jac[i] for j in range(n_mp_dof)]]
                )

                div_mp_h = alpha[:, 0:n_mp_dof:1] @ div_vc_h.T
                div_v = div_mp_h / ((f_d_phi(xv[0], xv[1], xv[2])))


               # equ_1_integrand = (mp_h @ mp_phi_tab[0, i, :, 0:dim].T) - (p_h @ div_vc_h)
               # equ_2_integrand = div_mp_h @ p_phi_tab[0, i, :, 0:dim].T

                equ_1_integrand = (v @ mp_phi_tab[0, i, :, 0:dim].T) - \
                                  (q * (f_d_phi(xv[0], xv[1], xv[2])) @ div_vc_h) / np.sqrt(f_porosity(xv[0], xv[1], xv[2]))

                #equ_2_integrand = (div_mp_h @ p_phi_tab[0, i, :, 0:dim].T) / (np.sqrt(f_porosity(xv[0], xv[1], xv[2]))) \
                #                  + (np.sqrt(f_porosity(xv[0], xv[1], xv[2])) * p_h)

                equ_2_integrand = ((f_d_phi(xv[0], xv[1], xv[2])) * div_v @ p_phi_tab[0, i, :, 0:dim].T) / (np.sqrt(f_porosity(xv[0], xv[1], xv[2])))  + q @ p_phi_tab[0, i, :, 0:dim].T

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, 0:n_mp_dof:1] = equ_1_integrand
                multiphysic_integrand[:, n_mp_dof : n_mp_dof + n_p_dof: 1] = equ_2_integrand
                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        return r_el, j_el


class ArbogastDualWeakFormBCDirichlet(WeakForm):
    def evaluate_form(self, element_index, alpha):
        iel = element_index
        p_D = self.functions["p"]

        mp_space = self.space.discrete_spaces["mp"]
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
                p_D_v = p_D(x[i, 0], x[i, 1], x[i, 2])
                phi = mp_tr_phi_tab[0, i, mp_dof_n_index, 0:dim] @ n[0:dim]
                res_block_mp += det_jac[i] * omega * p_D_v[c] * phi

            r_el[b:e:mp_components] += res_block_mp

        return r_el, j_el
