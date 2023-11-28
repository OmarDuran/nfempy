import auto_diff as ad
import numpy as np
from auto_diff.vecvalder import VecValDer

from basis.element_data import ElementData
from weak_forms.weak_from import WeakForm

class LCEDualWeakForm(WeakForm):

    def evaluate_form(self, element_index, alpha):

        i = element_index
        if self.space is None or self.functions is None:
            raise ValueError

        s_space = self.space.discrete_spaces["s"]
        m_space = self.space.discrete_spaces["m"]
        u_space = self.space.discrete_spaces["u"]
        t_space = self.space.discrete_spaces["t"]

        f_rhs = self.functions["rhs"]
        f_lambda = self.functions["lambda"]
        f_mu = self.functions["mu"]
        f_kappa = self.functions["kappa"]
        f_gamma = self.functions["gamma"]

        s_components = s_space.n_comp
        m_components = m_space.n_comp
        u_components = u_space.n_comp
        t_components = t_space.n_comp

        s_data: ElementData = s_space.elements[i].data
        m_data: ElementData = m_space.elements[i].data
        u_data: ElementData = u_space.elements[i].data
        t_data: ElementData = t_space.elements[i].data

        cell = s_data.cell
        dim = s_data.dimension
        points = s_data.quadrature.points
        weights = s_data.quadrature.weights
        x = s_data.mapping.x
        det_jac = s_data.mapping.det_jac
        inv_jac = s_data.mapping.inv_jac

        # basis
        s_phi_tab = s_data.basis.phi
        m_phi_tab = m_data.basis.phi
        u_phi_tab = u_data.basis.phi
        t_phi_tab = t_data.basis.phi
        
        n_s_phi = s_phi_tab.shape[2]
        n_m_phi = m_phi_tab.shape[2]
        n_u_phi = u_phi_tab.shape[2]
        n_t_phi = t_phi_tab.shape[2]

        n_s_dof = n_s_phi * s_components
        n_m_dof = n_m_phi * m_components
        n_u_dof = n_u_phi * u_components
        n_t_dof = n_t_phi * t_components

        n_dof = n_s_dof + n_m_dof + n_u_dof + n_t_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)
        alpha = np.zeros(n_dof)

        # Partial local vectorization
        f_val_star = f_rhs(x[:, 0], x[:, 1], x[:, 2])
        u_phi_s_star = det_jac * weights * u_phi_tab[0, :, :, 0].T
        t_phi_s_star = det_jac * weights * t_phi_tab[0, :, :, 0].T

        # constant directors
        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])
        Imat = np.identity(dim)
        with ad.AutoDiff(alpha) as alpha:
            el_form = np.zeros(n_dof)
            for c in range(u_components):
                b = c + n_s_dof + n_m_dof
                e = b + n_u_dof
                el_form[b:e:u_components] += -1.0 * u_phi_s_star @ f_val_star[c]
            for c in range(t_components):
                b = c + n_s_dof + n_m_dof + n_u_dof
                e = b + n_t_dof
                el_form[b:e:t_components] += (
                        -1.0 * t_phi_s_star @ f_val_star[c + u_components]
                )

            for i, omega in enumerate(weights):
                xv = x[i]
                if dim == 2:
                    c = 0
                    a_sx = alpha[:, c: n_s_dof + c: s_components]
                    a_ux = alpha[
                           :,
                           n_s_dof
                           + n_m_dof
                           + c: n_s_dof
                                + n_m_dof
                                + n_u_dof
                                + c: u_components,
                           ]

                    a_m = alpha[:, n_s_dof + c: n_s_dof + n_m_dof + c: m_components]
                    a_t = alpha[
                          :,
                          n_s_dof
                          + n_m_dof
                          + n_u_dof
                          + c: n_s_dof
                               + n_m_dof
                               + n_u_dof
                               + n_t_dof
                               + c: t_components,
                          ]

                    c = 1
                    a_sy = alpha[:, c: n_s_dof + c: s_components]
                    a_uy = alpha[
                           :,
                           n_s_dof
                           + n_m_dof
                           + c: n_s_dof
                                + n_m_dof
                                + n_u_dof
                                + c: u_components,
                           ]

                    sx_h = a_sx @ s_phi_tab[0, i, :, 0:dim]
                    sy_h = a_sy @ s_phi_tab[0, i, :, 0:dim]
                    ux_h = a_ux @ u_phi_tab[0, i, :, 0:dim]
                    uy_h = a_uy @ u_phi_tab[0, i, :, 0:dim]

                    mh = a_m @ m_phi_tab[0, i, :, 0:dim]
                    th = a_t @ t_phi_tab[0, i, :, 0:dim]

                    uh = VecValDer(
                        np.hstack((ux_h.val, uy_h.val)), np.hstack((ux_h.der, uy_h.der))
                    )

                    sh = VecValDer(
                        np.vstack((sx_h.val, sy_h.val)), np.vstack((sx_h.der, sy_h.der))
                    )

                    # Stress decomposition
                    Symm_sh = 0.5 * (sh + sh.T)
                    Skew_sh = 0.5 * (sh - sh.T)

                    tr_s_h = VecValDer(sh.val.trace(), sh.der.trace())
                    A_sh = (1.0 / 2.0 * f_mu(xv[0], xv[1], xv[2])) * (
                            Symm_sh
                            - (f_lambda(xv[0], xv[1], xv[2]) / (2.0 * f_mu(xv[0], xv[1], xv[2]) + dim * f_lambda(xv[0], xv[1], xv[2]))) * tr_s_h * Imat
                    ) + (1.0 / 2.0 * f_kappa(xv[0], xv[1], xv[2])) * Skew_sh

                    A_mh = (1.0 / f_gamma(xv[0], xv[1], xv[2])) * mh

                    grad_s_phi = s_phi_tab[1: s_phi_tab.shape[0] + 1, i, :, 0:dim]
                    div_tau = np.array(
                        [
                            [
                                np.trace(grad_s_phi[:, j, :]) / det_jac[i]
                                for j in range(n_s_phi)
                            ]
                        ]
                    )

                    grad_m_phi = m_phi_tab[1: s_phi_tab.shape[0] + 1, i, :, 0:dim]
                    div_v = np.array(
                        [
                            [
                                np.trace(grad_m_phi[:, j, :]) / det_jac[i]
                                for j in range(n_m_phi)
                            ]
                        ]
                    )

                    div_sh_x = a_sx @ div_tau.T
                    div_sh_y = a_sy @ div_tau.T
                    div_sh = VecValDer(
                        np.hstack((div_sh_x.val, div_sh_y.val)),
                        np.hstack((div_sh_x.der, div_sh_y.der)),
                    )

                    div_mh = a_m @ div_v.T

                    Gamma_outer = th * np.array([[0.0, -1.0], [1.0, 0.0]])
                    S_cross = np.array([[sh[1, 0] - sh[0, 1]]])

                else:
                    c = 0
                    a_sx = alpha[:, c: n_s_dof + c: s_components]
                    a_ux = alpha[
                           :,
                           n_s_dof
                           + n_m_dof
                           + c: n_s_dof
                                + n_m_dof
                                + n_u_dof
                                + c: u_components,
                           ]
                    a_mx = alpha[:, n_s_dof + c: n_s_dof + n_m_dof + c: m_components]
                    a_tx = alpha[
                           :,
                           n_s_dof
                           + n_m_dof
                           + n_u_dof
                           + c: n_s_dof
                                + n_m_dof
                                + n_u_dof
                                + n_t_dof
                                + c: t_components,
                           ]

                    c = 1
                    a_sy = alpha[:, c: n_s_dof + c: s_components]
                    a_uy = alpha[
                           :,
                           n_s_dof
                           + n_m_dof
                           + c: n_s_dof
                                + n_m_dof
                                + n_u_dof
                                + c: u_components,
                           ]
                    a_my = alpha[:, n_s_dof + c: n_s_dof + n_m_dof + c: m_components]
                    a_ty = alpha[
                           :,
                           n_s_dof
                           + n_m_dof
                           + n_u_dof
                           + c: n_s_dof
                                + n_m_dof
                                + n_u_dof
                                + n_t_dof
                                + c: t_components,
                           ]

                    c = 2
                    a_sz = alpha[:, c: n_s_dof + c: s_components]
                    a_uz = alpha[
                           :,
                           n_s_dof
                           + n_m_dof
                           + c: n_s_dof
                                + n_m_dof
                                + n_u_dof
                                + c: u_components,
                           ]
                    a_mz = alpha[:, n_s_dof + c: n_s_dof + n_m_dof + c: m_components]
                    a_tz = alpha[
                           :,
                           n_s_dof
                           + n_m_dof
                           + n_u_dof
                           + c: n_s_dof
                                + n_m_dof
                                + n_u_dof
                                + n_t_dof
                                + c: t_components,
                           ]

                    sx_h = a_sx @ s_phi_tab[0, i, :, 0:dim]
                    sy_h = a_sy @ s_phi_tab[0, i, :, 0:dim]
                    sz_h = a_sz @ s_phi_tab[0, i, :, 0:dim]

                    mx_h = a_mx @ m_phi_tab[0, i, :, 0:dim]
                    my_h = a_my @ m_phi_tab[0, i, :, 0:dim]
                    mz_h = a_mz @ m_phi_tab[0, i, :, 0:dim]

                    ux_h = a_ux @ u_phi_tab[0, i, :, 0:dim]
                    uy_h = a_uy @ u_phi_tab[0, i, :, 0:dim]
                    uz_h = a_uz @ u_phi_tab[0, i, :, 0:dim]

                    tx_h = a_tx @ t_phi_tab[0, i, :, 0:dim]
                    ty_h = a_ty @ t_phi_tab[0, i, :, 0:dim]
                    tz_h = a_tz @ t_phi_tab[0, i, :, 0:dim]

                    uh = VecValDer(
                        np.hstack((ux_h.val, uy_h.val, uz_h.val)),
                        np.hstack((ux_h.der, uy_h.der, uz_h.der)),
                    )

                    th = VecValDer(
                        np.hstack((tx_h.val, ty_h.val, tz_h.val)),
                        np.hstack((tx_h.der, ty_h.der, tz_h.der)),
                    )

                    sh = VecValDer(
                        np.vstack((sx_h.val, sy_h.val, sz_h.val)),
                        np.vstack((sx_h.der, sy_h.der, sz_h.der)),
                    )

                    mh = VecValDer(
                        np.vstack((mx_h.val, my_h.val, mz_h.val)),
                        np.vstack((mx_h.der, my_h.der, mz_h.der)),
                    )

                    # Stress decomposition
                    Symm_sh = 0.5 * (sh + sh.T)
                    Skew_sh = 0.5 * (sh - sh.T)

                    tr_s_h = VecValDer(sh.val.trace(), sh.der.trace())
                    A_sh = (1.0 / 2.0 * f_mu(xv[0], xv[1], xv[2])) * (
                            Symm_sh
                            - (f_lambda(xv[0], xv[1], xv[2]) / (2.0 * f_mu(xv[0], xv[1], xv[2]) + dim * f_lambda(xv[0], xv[1], xv[2]))) * tr_s_h * Imat
                    ) + (1.0 / 2.0 * f_kappa(xv[0], xv[1], xv[2])) * Skew_sh

                    A_mh = (1.0 / f_gamma(xv[0], xv[1], xv[2])) * mh

                    grad_s_phi = s_phi_tab[1: s_phi_tab.shape[0] + 1, i, :, 0:dim]
                    div_tau = np.array(
                        [
                            [
                                np.trace(grad_s_phi[:, j, :]) / det_jac[i]
                                for j in range(n_s_phi)
                            ]
                        ]
                    )

                    div_sh_x = a_sx @ div_tau.T
                    div_sh_y = a_sy @ div_tau.T
                    div_sh_z = a_sz @ div_tau.T

                    div_sh = VecValDer(
                        np.hstack((div_sh_x.val, div_sh_y.val, div_sh_z.val)),
                        np.hstack((div_sh_x.der, div_sh_y.der, div_sh_z.der)),
                    )

                    grad_m_phi = m_phi_tab[1: m_phi_tab.shape[0] + 1, i, :, 0:dim]
                    div_v = np.array(
                        [
                            [
                                np.trace(grad_m_phi[:, j, :]) / det_jac[i]
                                for j in range(n_m_phi)
                            ]
                        ]
                    )

                    div_mh_x = a_mx @ div_v.T
                    div_mh_y = a_my @ div_v.T
                    div_mh_z = a_mz @ div_v.T

                    div_mh = VecValDer(
                        np.hstack((div_mh_x.val, div_mh_y.val, div_mh_z.val)),
                        np.hstack((div_mh_x.der, div_mh_y.der, div_mh_z.der)),
                    )

                    Gamma_outer = np.array(
                        [
                            [0.0 * th[0, 0], -th[0, 2], +th[0, 1]],
                            [+th[0, 2], 0.0 * th[0, 0], -th[0, 0]],
                            [-th[0, 1], +th[0, 0], 0.0 * th[0, 0]],
                        ]
                    )

                    S_cross = np.array(
                        [
                            [
                                sh[2, 1] - sh[1, 2],
                                sh[0, 2] - sh[2, 0],
                                sh[1, 0] - sh[0, 1],
                            ]
                        ]
                    )

                equ_1_integrand = (
                        (s_phi_tab[0, i, :, 0:dim] @ A_sh.T)
                        + (div_tau.T @ uh)
                        + (s_phi_tab[0, i, :, 0:dim] @ Gamma_outer)
                )
                equ_2_integrand = (m_phi_tab[0, i, :, 0:dim] @ A_mh.T) + (div_v.T @ th)
                equ_3_integrand = u_phi_tab[0, i, :, 0:dim] @ div_sh
                equ_4_integrand = (t_phi_tab[0, i, :, 0:dim] @ div_mh) - (
                        t_phi_tab[0, i, :, 0:dim] @ S_cross
                )

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, 0:n_s_dof:1] = (equ_1_integrand).reshape(
                    (n_s_dof,)
                )
                multiphysic_integrand[:, n_s_dof: n_s_dof + n_m_dof: 1] = (
                    equ_2_integrand
                ).reshape((n_m_dof,))
                multiphysic_integrand[
                :, n_s_dof + n_m_dof: n_s_dof + n_m_dof + n_u_dof: 1
                ] = (equ_3_integrand).reshape((n_u_dof,))
                multiphysic_integrand[
                :,
                n_s_dof
                + n_m_dof
                + n_u_dof: n_s_dof
                           + n_m_dof
                           + n_u_dof
                           + n_t_dof: 1,
                ] = (equ_4_integrand).reshape((n_t_dof,))

                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        return r_el, j_el