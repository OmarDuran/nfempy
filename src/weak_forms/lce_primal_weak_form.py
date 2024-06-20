import auto_diff as ad
import numpy as np
from auto_diff.vecvalder import VecValDer

from basis.element_data import ElementData
from weak_forms.weak_from import WeakForm


class LCEPrimalWeakForm(WeakForm):
    def evaluate_form(self, element_index, alpha):
        iel = element_index
        if self.space is None or self.functions is None:
            raise ValueError

        u_space = self.space.discrete_spaces["u"]
        t_space = self.space.discrete_spaces["t"]

        f_rhs = self.functions["rhs"]
        f_lambda = self.functions["lambda"]
        f_mu = self.functions["mu"]
        f_kappa = self.functions["kappa"]
        f_gamma = self.functions["gamma"]

        u_components = u_space.n_comp
        t_components = t_space.n_comp

        u_data: ElementData = u_space.elements[iel].data
        t_data: ElementData = t_space.elements[iel].data

        cell = u_data.cell
        dim = cell.dimension
        points = u_data.quadrature.points
        weights = u_data.quadrature.weights
        x = u_data.mapping.x
        det_jac = u_data.mapping.det_jac
        inv_jac = u_data.mapping.inv_jac

        u_phi_tab = u_data.basis.phi
        t_phi_tab = t_data.basis.phi

        n_u_phi = u_phi_tab.shape[2]
        n_t_phi = t_phi_tab.shape[2]

        n_u_dof = n_u_phi * u_components
        n_t_dof = n_t_phi * t_components

        n_dof = n_u_dof + n_t_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)
        alpha = np.zeros(n_dof)

        # Partial local vectorization
        f_val_star = f_rhs(x[:, 0], x[:, 1], x[:, 2])

        # constant directors
        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])
        Imat = np.identity(dim)
        with ad.AutoDiff(alpha) as alpha:
            el_form = np.zeros(n_dof)
            for c in range(u_components):
                b = c
                e = b + n_u_dof
                el_form[b:e:u_components] += (
                    det_jac * weights * u_phi_tab[0, :, :, 0].T @ f_val_star[c]
                )
            for c in range(t_components):
                b = c + n_u_dof
                e = b + n_t_dof
                el_form[b:e:t_components] += (
                    det_jac
                    * weights
                    * t_phi_tab[0, :, :, 0].T
                    @ f_val_star[c + u_components]
                )

            for i, omega in enumerate(weights):
                xv = x[i]
                if dim == 2:
                    inv_jac_m = np.vstack((inv_jac[i] @ e1, inv_jac[i] @ e2))
                    grad_phi_u = (
                        inv_jac_m @ u_phi_tab[1 : u_phi_tab.shape[0] + 1, i, :, 0]
                    ).T
                    grad_phi_t = (
                        inv_jac_m @ t_phi_tab[1 : t_phi_tab.shape[0] + 1, i, :, 0]
                    ).T

                    c = 0
                    a_ux = alpha[:, c : n_u_dof + c : u_components]
                    c = 1
                    a_uy = alpha[:, c : n_u_dof + c : u_components]
                    a_t = alpha[:, n_u_dof : n_t_dof + n_u_dof : t_components]

                    ux_h = a_ux @ u_phi_tab[0, i, :, 0:dim]
                    uy_h = a_uy @ u_phi_tab[0, i, :, 0:dim]
                    grad_uh_x = a_ux @ grad_phi_u
                    grad_uh_y = a_uy @ grad_phi_u
                    grad_uh = VecValDer(
                        np.vstack((grad_uh_x.val, grad_uh_y.val)),
                        np.vstack((grad_uh_x.der, grad_uh_y.der)),
                    )

                    grad_th = a_t @ grad_phi_t
                    th = a_t @ t_phi_tab[0, i, :, 0:t_components]

                    Theta_outer = th * np.array([[0.0, -1.0], [1.0, 0.0]])
                    eh = grad_uh + Theta_outer
                    # Stress decomposition
                    Symm_eh = 0.5 * (eh + eh.T)
                    Skew_eh = 0.5 * (eh - eh.T)
                    tr_eh = VecValDer(eh.val.trace(), eh.der.trace())
                    sh = (
                        2.0 * f_mu(xv[0], xv[1], xv[2]) * Symm_eh
                        + 2.0 * f_kappa(xv[0], xv[1], xv[2]) * Skew_eh
                        + f_lambda(xv[0], xv[1], xv[2]) * tr_eh * Imat
                    )

                    Skew_sh = 0.5 * (sh - sh.T)
                    S_cross = np.array([[Skew_sh[1, 0] - Skew_sh[0, 1]]])
                    k = f_gamma(xv[0], xv[1], xv[2]) * grad_th
                    strain_energy_h = (grad_phi_u @ sh.T).reshape((n_u_dof,))
                    curvature_energy_h = (
                        grad_phi_t @ k.T + t_phi_tab[0, i, :, 0:t_components] @ S_cross
                    ).reshape((n_t_dof,))

                else:
                    inv_jac_m = np.vstack(
                        (inv_jac[i] @ e1, inv_jac[i] @ e2, inv_jac[i] @ e3)
                    )
                    grad_phi_u = (
                        inv_jac_m @ u_phi_tab[1 : u_phi_tab.shape[0] + 1, i, :, 0]
                    ).T
                    grad_phi_t = (
                        inv_jac_m @ t_phi_tab[1 : t_phi_tab.shape[0] + 1, i, :, 0]
                    ).T

                    c = 0
                    a_ux = alpha[:, c : n_u_dof + c : u_components]
                    a_tx = alpha[:, c + n_u_dof : n_u_dof + n_t_dof + c : t_components]
                    c = 1
                    a_uy = alpha[:, c : n_u_dof + c : u_components]
                    a_ty = alpha[:, c + n_u_dof : n_u_dof + n_t_dof + c : t_components]
                    c = 2
                    a_uz = alpha[:, c : n_u_dof + c : u_components]
                    a_tz = alpha[:, c + n_u_dof : n_u_dof + n_t_dof + c : t_components]

                    ux_h = a_ux @ u_phi_tab[0, i, :, 0:dim]
                    uy_h = a_uy @ u_phi_tab[0, i, :, 0:dim]
                    uz_h = a_uz @ u_phi_tab[0, i, :, 0:dim]
                    grad_uh_x = a_ux @ grad_phi_u
                    grad_uh_y = a_uy @ grad_phi_u
                    grad_uh_z = a_uz @ grad_phi_u
                    grad_uh = VecValDer(
                        np.vstack((grad_uh_x.val, grad_uh_y.val, grad_uh_z.val)),
                        np.vstack((grad_uh_x.der, grad_uh_y.der, grad_uh_z.der)),
                    )

                    tx_h = a_tx @ t_phi_tab[0, i, :, 0:dim]
                    ty_h = a_ty @ t_phi_tab[0, i, :, 0:dim]
                    tz_h = a_tz @ t_phi_tab[0, i, :, 0:dim]
                    grad_th_x = a_tx @ grad_phi_t
                    grad_th_y = a_ty @ grad_phi_t
                    grad_th_z = a_tz @ grad_phi_t
                    grad_th = VecValDer(
                        np.vstack((grad_th_x.val, grad_th_y.val, grad_th_z.val)),
                        np.vstack((grad_th_x.der, grad_th_y.der, grad_th_z.der)),
                    )

                    uh = VecValDer(
                        np.hstack((ux_h.val, uy_h.val, uz_h.val)),
                        np.hstack((ux_h.der, uy_h.der, uz_h.der)),
                    )

                    th = VecValDer(
                        np.hstack((tx_h.val, ty_h.val, tz_h.val)),
                        np.hstack((tx_h.der, ty_h.der, tz_h.der)),
                    )

                    Theta_outer = np.array(
                        [
                            [0.0 * th[0, 0], -th[0, 2], +th[0, 1]],
                            [+th[0, 2], 0.0 * th[0, 0], -th[0, 0]],
                            [-th[0, 1], +th[0, 0], 0.0 * th[0, 0]],
                        ]
                    )

                    eh = grad_uh + Theta_outer
                    # Stress decomposition
                    Symm_eh = 0.5 * (eh + eh.T)
                    Skew_eh = 0.5 * (eh - eh.T)
                    tr_eh = VecValDer(eh.val.trace(), eh.der.trace())
                    sh = (
                        2.0 * f_mu(xv[0], xv[1], xv[2]) * Symm_eh
                        + 2.0 * f_kappa(xv[0], xv[1], xv[2]) * Skew_eh
                        + f_lambda(xv[0], xv[1], xv[2]) * tr_eh * Imat
                    )

                    Skew_sh = 0.5 * (sh - sh.T)
                    S_cross = np.array(
                        [
                            [
                                Skew_sh[2, 1] - Skew_sh[1, 2],
                                Skew_sh[0, 2] - Skew_sh[2, 0],
                                Skew_sh[1, 0] - Skew_sh[0, 1],
                            ]
                        ]
                    )

                    k = f_gamma(xv[0], xv[1], xv[2]) * grad_th
                    strain_energy_h = (grad_phi_u @ sh.T).reshape((n_u_dof,))
                    curvature_energy_h = (
                        grad_phi_t @ k.T + t_phi_tab[0, i, :, 0:t_components] @ S_cross
                    ).reshape((n_t_dof,))

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, 0:n_u_dof:1] = strain_energy_h
                multiphysic_integrand[:, n_u_dof : n_u_dof + n_t_dof : 1] = (
                    curvature_energy_h
                )

                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        return r_el, j_el


class LCEPrimalWeakFormBCDirichlet(WeakForm):
    def evaluate_form(self, element_index, alpha):
        u_D = self.functions["u"]
        t_D = self.functions["t"]

        iel = element_index
        u_space = self.space.discrete_spaces["u"]
        t_space = self.space.discrete_spaces["t"]
        u_components = u_space.n_comp
        t_components = t_space.n_comp

        u_data: ElementData = u_space.bc_elements[iel].data
        t_data: ElementData = t_space.bc_elements[iel].data

        cell = u_data.cell
        points = u_data.quadrature.points
        weights = u_data.quadrature.weights
        x = u_data.mapping.x
        det_jac = u_data.mapping.det_jac
        inv_jac = u_data.mapping.inv_jac

        u_phi_tab = u_data.basis.phi
        t_phi_tab = t_data.basis.phi

        n_u_phi = u_phi_tab.shape[2]
        n_t_phi = t_phi_tab.shape[2]

        n_u_dof = n_u_phi * u_components
        n_t_dof = n_t_phi * t_components

        n_dof = n_u_dof + n_t_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # local blocks
        beta = 1.0e12
        jac_block_u = np.zeros((n_u_phi, n_u_phi))
        for i, omega in enumerate(weights):
            phi = u_phi_tab[0, i, :, 0]
            jac_block_u += beta * det_jac[i] * omega * np.outer(phi, phi)

        jac_block_t = np.zeros((n_t_phi, n_t_phi))
        for i, omega in enumerate(weights):
            phi = t_phi_tab[0, i, :, 0]
            jac_block_t += beta * det_jac[i] * omega * np.outer(phi, phi)

        for c in range(u_components):
            b = c
            e = b + n_u_dof

            res_block_u = np.zeros(n_u_phi)
            for i, omega in enumerate(weights):
                phi = u_phi_tab[0, i, :, 0]
                u_D_v = u_D(x[i, 0], x[i, 1], x[i, 2])
                res_block_u -= beta * det_jac[i] * omega * u_D_v[c] * phi

            r_el[b:e:u_components] += res_block_u
            j_el[b:e:u_components, b:e:u_components] += jac_block_u

        for c in range(t_components):
            b = c + n_u_dof
            e = b + n_t_dof

            res_block_t = np.zeros(n_t_phi)
            for i, omega in enumerate(weights):
                phi = t_phi_tab[0, i, :, 0]
                t_D_v = t_D(x[i, 0], x[i, 1], x[i, 2])
                res_block_t -= beta * det_jac[i] * omega * t_D_v[c] * phi

            r_el[b:e:t_components] += res_block_t
            j_el[b:e:t_components, b:e:t_components] += jac_block_t

        return r_el, j_el
