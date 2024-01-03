import auto_diff as ad
import numpy as np
from auto_diff.vecvalder import VecValDer

from basis.element_data import ElementData
from basis.parametric_transformation import transform_lower_to_higher
from geometry.compute_normal import normal
from topology.topological_queries import find_higher_dimension_neighs
from weak_forms.weak_from import WeakForm


class LCEScaledRieszMapWeakForm(WeakForm):
    def evaluate_form(self, element_index, alpha):
        iel = element_index
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
        f_grad_gamma = self.functions["grad_gamma"]

        s_components = s_space.n_comp
        m_components = m_space.n_comp
        u_components = u_space.n_comp
        t_components = t_space.n_comp

        s_data: ElementData = s_space.elements[iel].data
        m_data: ElementData = m_space.elements[iel].data
        u_data: ElementData = u_space.elements[iel].data
        t_data: ElementData = t_space.elements[iel].data

        points, weights = self.space.quadrature
        dim = s_data.dimension
        x, jac, det_jac, inv_jac = s_space.elements[iel].evaluate_mapping(points)

        # basis
        s_phi_tab = s_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        m_phi_tab = m_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        u_phi_tab = u_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        t_phi_tab = t_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)

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

        # Partial local vectorization
        f_val_star = f_rhs(x[:, 0], x[:, 1], x[:, 2])
        u_phi_s_star = det_jac * weights * u_phi_tab[0, :, :, 0].T
        t_phi_s_star = det_jac * weights * t_phi_tab[0, :, :, 0].T

        # constant directors
        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])

        gamma_scale_v = f_gamma(x[:, 0], x[:, 1], x[:, 2])
        grad_gamma_v = f_grad_gamma(x[:, 0], x[:, 1], x[:, 2])

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

                gamma_scale = gamma_scale_v[i]
                grad_gamma_scale = grad_gamma_v[:, i]

                aka = 0
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
                            - (
                                    f_lambda(xv[0], xv[1], xv[2])
                                    / (
                                            2.0 * f_mu(xv[0], xv[1], xv[2])
                                            + dim * f_lambda(xv[0], xv[1], xv[2])
                                    )
                            )
                            * tr_s_h
                            * Imat
                    ) + (1.0 / 2.0 * f_kappa(xv[0], xv[1], xv[2])) * Skew_sh

                    A_mh = mh

                    grad_s_phi = s_phi_tab[1: s_phi_tab.shape[0] + 1, i, :, 0:dim]
                    div_tau = np.array(
                        [np.trace(grad_s_phi, axis1=0, axis2=2) / det_jac[i]]
                    )

                    grad_m_phi = m_phi_tab[1: m_phi_tab.shape[0] + 1, i, :, 0:dim]
                    div_v = np.array(
                        [np.trace(grad_m_phi, axis1=0, axis2=2) / det_jac[i]]
                    )

                    tr_grad_eps_otimes_v = np.array(
                        [
                            [
                                np.trace(
                                    np.outer(
                                        grad_gamma_scale, m_phi_tab[0, i, j, 0:dim]
                                    )
                                )
                                for j in range(n_m_phi)
                            ]
                        ]
                    )
                    div_v_s = gamma_scale * div_v + tr_grad_eps_otimes_v

                    div_sh_x = a_sx @ div_tau.T
                    div_sh_y = a_sy @ div_tau.T
                    div_sh = VecValDer(
                        np.hstack((div_sh_x.val, div_sh_y.val)),
                        np.hstack((div_sh_x.der, div_sh_y.der)),
                    )

                    div_mh = a_m @ div_v_s.T

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
                            - (
                                    f_lambda(xv[0], xv[1], xv[2])
                                    / (
                                            2.0 * f_mu(xv[0], xv[1], xv[2])
                                            + dim * f_lambda(xv[0], xv[1], xv[2])
                                    )
                            )
                            * tr_s_h
                            * Imat
                    ) + (1.0 / 2.0 * f_kappa(xv[0], xv[1], xv[2])) * Skew_sh

                    A_mh = mh

                    grad_s_phi = s_phi_tab[1: s_phi_tab.shape[0] + 1, i, :, 0:dim]
                    div_tau = np.array(
                        [np.trace(grad_s_phi, axis1=0, axis2=2) / det_jac[i]]
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
                        [np.trace(grad_m_phi, axis1=0, axis2=2) / det_jac[i]]
                    )

                    tr_grad_eps_otimes_v = np.array(
                        [
                            [
                                np.trace(
                                    np.outer(
                                        grad_gamma_scale, m_phi_tab[0, i, j, 0:dim]
                                    )
                                )
                                for j in range(n_m_phi)
                            ]
                        ]
                    )
                    div_v_s = gamma_scale * div_v + tr_grad_eps_otimes_v

                    div_mh_x = a_mx @ div_v_s.T
                    div_mh_y = a_my @ div_v_s.T
                    div_mh_z = a_mz @ div_v_s.T

                    div_mh = VecValDer(
                        np.hstack((div_mh_x.val, div_mh_y.val, div_mh_z.val)),
                        np.hstack((div_mh_x.der, div_mh_y.der, div_mh_z.der)),
                    )


                equ_1_integrand = s_phi_tab[0, i, :, 0:dim] @ A_sh.T + div_tau.T @ div_sh
                equ_2_integrand = m_phi_tab[0, i, :, 0:dim] @ A_mh.T + div_v_s.T @ div_mh
                equ_3_integrand = u_phi_tab[0, i, :, 0:dim] @ uh
                equ_4_integrand = t_phi_tab[0, i, :, 0:dim] @ th

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, 0:n_s_dof:1] = (equ_1_integrand).reshape(
                    (n_s_dof,)
                )
                multiphysic_integrand[:, n_s_dof : n_s_dof + n_m_dof : 1] = (
                    equ_2_integrand
                ).reshape((n_m_dof,))
                multiphysic_integrand[
                    :, n_s_dof + n_m_dof : n_s_dof + n_m_dof + n_u_dof : 1
                ] = (equ_3_integrand).reshape((n_u_dof,))
                multiphysic_integrand[
                    :,
                    n_s_dof
                    + n_m_dof
                    + n_u_dof : n_s_dof
                    + n_m_dof
                    + n_u_dof
                    + n_t_dof : 1,
                ] = (equ_4_integrand).reshape((n_t_dof,))

                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        return r_el, j_el

    def evaluate_form_vectorized(self, element_index, alpha):
        iel = element_index
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
        f_grad_gamma = self.functions["grad_gamma"]

        s_components = s_space.n_comp
        m_components = m_space.n_comp
        u_components = u_space.n_comp
        t_components = t_space.n_comp

        s_data: ElementData = s_space.elements[iel].data
        m_data: ElementData = m_space.elements[iel].data
        u_data: ElementData = u_space.elements[iel].data
        t_data: ElementData = t_space.elements[iel].data

        points, weights = self.space.quadrature
        dim = s_data.dimension
        x, jac, det_jac, inv_jac = s_space.elements[iel].evaluate_mapping(points)

        # basis
        s_phi_tab = s_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        m_phi_tab = m_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        u_phi_tab = u_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        t_phi_tab = t_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)

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

        f_val_star = f_rhs(x[:, 0], x[:, 1], x[:, 2])

        lambda_v = f_lambda(x[:, 0], x[:, 1], x[:, 2])
        mu_v = f_mu(x[:, 0], x[:, 1], x[:, 2])
        kappa_v = f_kappa(x[:, 0], x[:, 1], x[:, 2])

        # Vectorized contributions
        # s : stress
        # m : coupled stress
        # u : displacement
        # t : rotation

        gamma_scale_v = f_gamma(x[:, 0], x[:, 1], x[:, 2])
        grad_gamma_v = f_grad_gamma(x[:, 0], x[:, 1], x[:, 2])

        # (s,s) block
        s_phi_star = s_phi_tab[0, :, :, 0:dim]
        s_j_el = np.array([0.5 * np.dot(phi, phi.T) for phi in s_phi_star]).T @ (
                det_jac * weights
        )
        for c in range(s_components):
            b = c
            e = b + n_s_dof
            j_el[b:e:s_components, b:e:s_components] += (1 / (2.0 * mu_v)) * s_j_el + (
                    1 / (2.0 * kappa_v)
            ) * s_j_el

        # transpose_s_j_el = np.zeros((n_s_dof, n_s_dof))
        #
        # def phi_outer(phi):
        #     n_data = phi.shape[0]
        #     return np.array(
        #         [
        #             np.outer(phi[i], phi[j]).T
        #             for i in range(n_data)
        #             for j in range(n_data)
        #         ]
        #     )
        #
        # dest_idx = lambda idx: np.unravel_index(idx, (n_s_phi, n_s_phi))
        #
        # def insert_prod(k, prod, array):
        #     ip, jp = dest_idx(k)
        #     array[
        #         ip * s_components : (ip + 1) * s_components,
        #         jp * s_components : (jp + 1) * s_components,
        #     ] += prod
        #
        # trans_outer_prods = (
        #     np.array([phi_outer(phi) for phi in s_phi_star]).T @ (det_jac * weights)
        # ).T
        # [
        #     insert_prod(
        #         k,
        #         0.5 * ((1 / (2.0 * mu_v)) - (1 / (2.0 * kappa_v))) * oprod,
        #         transpose_s_j_el,
        #     )
        #     for k, oprod in enumerate(trans_outer_prods)
        # ]
        # j_el[0:n_s_dof, 0:n_s_dof] += transpose_s_j_el

        vol_factor = (1.0 / (2.0 * mu_v)) * (lambda_v / (2.0 * mu_v + dim * lambda_v))
        vol_s_j_el = (
                -1.0
                * np.array([np.outer(phi, phi) for phi in s_phi_tab[0, :, :, 0:dim]]).T
                @ (vol_factor * det_jac * weights)
        )
        j_el[0:n_s_dof, 0:n_s_dof] += vol_s_j_el

        # (m,m) block
        m_j_el = np.array(
            [np.dot(phi, phi.T) for phi in m_phi_tab[0, :, :, 0:dim]]
        ).T @ (det_jac * weights)
        for c in range(m_components):
            b = c + n_s_dof
            e = b + n_m_dof
            j_el[b:e:m_components, b:e:m_components] += m_j_el

        # (div s, div tau) block
        grad_s_phi_star = s_phi_tab[1: s_phi_tab.shape[0] + 1, :, :, 0:dim]
        div_tau_star = np.trace(grad_s_phi_star, axis1=0, axis2=3).T * (1/det_jac)
        div_tau_block_outer = np.array([np.outer(div_phi, div_phi) for div_phi in div_tau_star.T]).T @ (det_jac * weights)
        for sc in range(s_components):
            sb = sc
            se = sb + n_s_dof
            j_el[sb:se:s_components, sb:se:s_components] += div_tau_block_outer

        # (div s, div tau) block
        grad_m_phi_star = m_phi_tab[1: m_phi_tab.shape[0] + 1, :, :, 0:dim]
        div_v_star = (
                np.array(
                    [
                        [
                            np.trace(
                                np.outer(grad_gamma_v[:, i], m_phi_tab[0, i, j, 0:dim])
                            )
                            for i in range(len(points))
                        ]
                        for j in range(n_m_phi)
                    ]
                )
        )

        div_v_star += np.trace(grad_m_phi_star, axis1=0, axis2=3).T * (gamma_scale_v/det_jac)
        div_v_block_outer = np.array([np.outer(div_phi, div_phi) for div_phi in div_v_star.T]).T @ (det_jac * weights)
        for mc in range(m_components):
            mb = mc + n_s_dof
            me = mb + n_m_dof
            j_el[mb:me:m_components, mb:me:m_components] += div_v_block_outer

        # (u,u) block
        u_block_outer = np.array([np.outer(phi, phi) for phi in u_phi_tab[0, :, :, 0]]).T @ (det_jac * weights)
        for c in range(u_components):
            b = c + n_s_dof + n_m_dof
            e = b + n_u_dof
            j_el[b:e:u_components,b:e:u_components] += u_block_outer

        # (t, t) block
        t_block_outer = np.array([np.outer(phi, phi) for phi in t_phi_tab[0, :, :, 0]]).T @ (det_jac * weights)
        for c in range(t_components):
            b = c + n_s_dof + n_m_dof + n_u_dof
            e = b + n_t_dof
            j_el[b:e:t_components,b:e:t_components] += t_block_outer

        return r_el, j_el


