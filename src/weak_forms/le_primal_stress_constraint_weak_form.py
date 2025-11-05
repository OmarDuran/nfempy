import auto_diff as ad
import numpy as np
from auto_diff.vecvalder import VecValDer

from basis.element_data import ElementData
from weak_forms.weak_from import WeakForm
from geometry.compute_normal import normal
from mesh.topological_queries import find_higher_dimension_neighs

class LEPrimalStressConstraintWeakForm(WeakForm):
    """Linear elastic primal weak form for Cauchy elasticity."""

    def evaluate_form(self, element_index: int, alpha: float):
        iel = element_index
        if self.space is None or self.functions is None:
            raise ValueError

        u_space = self.space.discrete_spaces["u"]
        u_components = u_space.n_comp

        f_s_target = self.functions["s_target"]
        f_lambda = self.functions["lambda"]
        f_mu = self.functions["mu"]

        beta = 1.0e6  # Penalization parameter

        u_data: ElementData = u_space.elements[iel].data

        cell = u_data.cell
        dim = cell.dimension
        points, weights = self.space.quadrature[dim]
        x, jac, det_jac, inv_jac = u_space.elements[iel].evaluate_mapping(points)

        u_phi_tab = u_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)

        n_u_phi = u_phi_tab.shape[2]
        n_u_dof = n_u_phi * u_components
        n_dof = n_u_dof

        # Partial local vectorization
        f_s_target_star = f_s_target(x[:, 0], x[:, 1], x[:, 2])
        f_lambda_star = f_lambda(x[:, 0], x[:, 1], x[:, 2])
        f_mu_star = f_mu(x[:, 0], x[:, 1], x[:, 2])

        # constant directors
        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])
        Imat = np.identity(dim)

        with ad.AutoDiff(alpha) as alpha:
            el_form = np.zeros(n_dof)

            for i, omega in enumerate(weights):
                if dim == 2:
                    inv_jac_m = np.vstack((inv_jac[i] @ e1, inv_jac[i] @ e2))
                    grad_phi_u = (
                        inv_jac_m @ u_phi_tab[1 : u_phi_tab.shape[0] + 1, i, :, 0]
                    ).T

                    c = 0
                    a_ux = alpha[:, c : n_u_dof + c : u_components]
                    c = 1
                    a_uy = alpha[:, c : n_u_dof + c : u_components]

                    grad_uh_x = a_ux @ grad_phi_u
                    grad_uh_y = a_uy @ grad_phi_u
                    grad_uh = VecValDer(
                        np.vstack((grad_uh_x.val, grad_uh_y.val)),
                        np.vstack((grad_uh_x.der, grad_uh_y.der)),
                    )

                    # Strain tensor (symmetric gradient)
                    eh = 0.5 * (grad_uh + grad_uh.T)
                    tr_eh = VecValDer(eh.val.trace(), eh.der.trace())

                    # Stress tensor (linear elastic constitutive law)
                    sh = (
                        2.0 * f_mu_star[i] * eh
                        + f_lambda_star[i] * tr_eh * Imat
                    )
                    st = f_s_target_star[:,:,i]
                    strain_energy_h = (grad_phi_u @ (beta * (sh.T - st.T))).reshape((n_u_dof,))

                else:  # dim == 3
                    inv_jac_m = np.vstack(
                        (inv_jac[i] @ e1, inv_jac[i] @ e2, inv_jac[i] @ e3)
                    )
                    grad_phi_u = (
                        inv_jac_m @ u_phi_tab[1 : u_phi_tab.shape[0] + 1, i, :, 0]
                    ).T

                    c = 0
                    a_ux = alpha[:, c : n_u_dof + c : u_components]
                    c = 1
                    a_uy = alpha[:, c : n_u_dof + c : u_components]
                    c = 2
                    a_uz = alpha[:, c : n_u_dof + c : u_components]

                    grad_uh_x = a_ux @ grad_phi_u
                    grad_uh_y = a_uy @ grad_phi_u
                    grad_uh_z = a_uz @ grad_phi_u
                    grad_uh = VecValDer(
                        np.vstack((grad_uh_x.val, grad_uh_y.val, grad_uh_z.val)),
                        np.vstack((grad_uh_x.der, grad_uh_y.der, grad_uh_z.der)),
                    )

                    # Strain tensor (symmetric gradient)
                    eh = 0.5 * (grad_uh + grad_uh.T)
                    tr_eh = VecValDer(eh.val.trace(), eh.der.trace())

                    # Stress tensor (linear elastic constitutive law)
                    sh = (
                        2.0 * f_mu_star[i] * eh
                        + f_lambda_star[i] * tr_eh * Imat
                    )

                    st = f_s_target_star[:,:,i]
                    strain_energy_h = (grad_phi_u @ (beta * (sh.T - st.T))).reshape((n_u_dof,))

                discrete_integrand = strain_energy_h
                el_form += det_jac[i] * omega * discrete_integrand

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))
        return r_el, j_el

