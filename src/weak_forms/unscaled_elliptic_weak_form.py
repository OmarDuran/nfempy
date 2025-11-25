"""
Unscaled Elliptic Weak Form for Two-Phase Mixtures

This module implements the evaluation of the weak form of a non-degenerate elliptic
equation, which arises in models for two-phase mixtures presented in [1].

[1] Arbogast, T., Taicher, A.L. A cell-centered finite difference method for a degenerate
elliptic equation arising from two-phase mixtures. Comput Geosci 21, 701–712 (2017).
https://doi.org/10.1007/s10596-017-9649-9

Key concepts and components:
- **Element-level evaluation**: The weak form is evaluated on individual elements
  of the finite element mesh.
- **Auto-differentiation**: The solution vector `alpha` is differentiated using
  automatic differentiation (from the `auto_diff` module) to compute the Jacobian
  and residuals.
- **Porosity and Phase Transition**: Functions related to porosity and phase transition
  are included, ensuring the degenerate nature of the equation is accounted for.
- **Boundary Conditions**: Dirichlet boundary conditions are applied to the weak form
  through a separate class, ensuring proper handling of boundary elements.


Classes:
- `DegenerateEllipticWeakForm`: Main class for evaluating the weak form on
  each element of the mesh.
- `DegenerateEllipticWeakFormBCDirichlet`: Class for applying Dirichlet
  boundary conditions on the weak form.

"""

import auto_diff as ad
import numpy as np

from basis.element_data import ElementData
from basis.parametric_transformation import transform_lower_to_higher
from geometry.compute_normal import normal
from mesh.topological_queries import find_higher_dimension_neighs
from weak_forms.weak_from import WeakForm


class UnscaledEllipticWeakForm(WeakForm):
    def evaluate_form(self, element_index, alpha):
        iel = element_index
        if self.space is None or self.functions is None:
            raise ValueError

        u_space = self.space.discrete_spaces["u"]
        p_space = self.space.discrete_spaces["p"]

        f_rhs_f = self.functions["rhs"]
        f_porosity = self.functions["porosity"]
        f_d_phi = self.functions["d_phi"]
        f_grad_d_phi = self.functions["grad_d_phi"]

        u_components = u_space.n_comp
        p_components = p_space.n_comp

        u_data: ElementData = u_space.elements[iel].data

        cell = u_data.cell
        dim = cell.dimension
        points, weights = self.space.quadrature[dim]
        x, jac, det_jac, inv_jac = u_space.elements[iel].evaluate_mapping(points)

        # basis
        u_phi_tab = u_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        p_phi_tab = p_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)

        n_u_phi = u_phi_tab.shape[2]
        n_p_phi = p_phi_tab.shape[2]

        n_u_dof = n_u_phi * u_components
        n_p_dof = n_p_phi * p_components

        idx_dof = {
            "u": slice(0, n_u_dof),
            "p": slice(n_u_dof, n_u_dof + n_p_dof),
        }

        n_dof = n_u_dof + n_p_dof

        # Partial local vectorization
        phi_star = f_porosity(x[:, 0], x[:, 1], x[:, 2])

        sqrt_phi_star = np.sqrt(phi_star)

        # alternative name for d_phi
        delta_star = f_d_phi(x[:, 0], x[:, 1], x[:, 2])
        grad_delta_star = f_grad_d_phi(x[:, 0], x[:, 1], x[:, 2])

        f_f_val_star = f_rhs_f(x[:, 0], x[:, 1], x[:, 2])
        phi_q_star = det_jac * weights * p_phi_tab[0, :, :, 0].T

        with ad.AutoDiff(alpha) as alpha:
            el_form = np.zeros(n_dof)
            for c in range(p_components):
                b = c + n_u_dof
                e = b + n_p_dof
                el_form[b:e:p_components] -= (sqrt_phi_star * phi_q_star) @ f_f_val_star[c].T

            for i, omega in enumerate(weights):

                phi = phi_star[i]
                delta = delta_star[i]

                # Functions and derivatives at integration point i
                psi_h = u_phi_tab[0, i, :, 0:dim]
                w_h = p_phi_tab[0, i, :, 0:dim]

                u_h = alpha[:, idx_dof["u"]] @ psi_h
                p_h = alpha[:, idx_dof["p"]] @ w_h

                grad_psi_h = u_phi_tab[1 : u_phi_tab.shape[0] + 1, i, :, 0:dim]
                div_psi_h = np.array(
                    [np.trace(grad_psi_h, axis1=0, axis2=2) / det_jac[i]]
                )

                if  np.isclose(phi, 0.0) and phi < 0.0:
                    phi = 1e-16
                    print("UnscaledEllipticWeakForm:: porosity zero or negative.")

                div_u_h = alpha[:, idx_dof["u"]] @ div_psi_h.T

                equ_1_integrand = ((1/(delta*delta)) * u_h @ psi_h.T) - (p_h @ div_psi_h)
                equ_2_integrand = div_u_h @ w_h.T + phi * p_h @ w_h.T

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, idx_dof["u"]] = equ_1_integrand
                multiphysic_integrand[:, idx_dof["p"]] = equ_2_integrand
                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        return r_el, j_el


class UnscaledEllipticWeakFormBCDirichlet(WeakForm):
    def evaluate_form(self, element_index, alpha):
        iel = element_index
        p_D = self.functions["p"]
        f_d_phi = self.functions["d_phi"]
        f_porosity = self.functions["porosity"]

        mp_space = self.space.discrete_spaces["u"]
        mp_components = mp_space.n_comp
        mp_data: ElementData = mp_space.bc_elements[iel].data

        cell = mp_data.cell
        dim = cell.dimension
        points, weights = self.space.bc_quadrature[dim]
        dim = cell.dimension
        x, jac, det_jac, inv_jac = mp_space.bc_elements[iel].evaluate_mapping(points)

        # find high-dimension neigh q space
        neigh_list = find_higher_dimension_neighs(cell, mp_space.dof_map.mesh_topology)
        neigh_check_mp = len(neigh_list) > 0
        assert neigh_check_mp
        neigh_cell_id = neigh_list[0][1]
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
        mp_facet_index = (
            neigh_cell.sub_cells_ids[cell.dimension].tolist().index(cell.id)
        )
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
                # d_phi = f_d_phi(x[i, 0], x[i, 1], x[i, 2])
                # phi_v = f_porosity(x[i, 0], x[i, 1], x[i, 2])
                # if not np.isclose(phi_v, 0.0) and phi_v > 0.0:
                #     p_D_v *= d_phi / np.sqrt(phi_v)
                phi = mp_tr_phi_tab[0, i, mp_dof_n_index, 0:dim] @ n[0:dim]
                res_block_mp += det_jac[i] * omega * p_D_v[c] * phi

            r_el[b:e:mp_components] += res_block_mp

        return r_el, j_el
