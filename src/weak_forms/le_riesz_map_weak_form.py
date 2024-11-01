import auto_diff as ad
import numpy as np
from auto_diff.vecvalder import VecValDer

from basis.element_data import ElementData
from weak_forms.weak_from import WeakForm


class LERieszMapWeakForm(WeakForm):
    def evaluate_form(self, element_index, alpha):
        iel = element_index
        if self.space is None or self.functions is None:
            raise ValueError

        s_space = self.space.discrete_spaces["s"]
        u_space = self.space.discrete_spaces["u"]
        t_space = self.space.discrete_spaces["t"]

        f_lambda = self.functions["lambda"]
        f_mu = self.functions["mu"]

        s_components = s_space.n_comp
        u_components = u_space.n_comp
        t_components = t_space.n_comp

        s_data: ElementData = s_space.elements[iel].data
        u_data: ElementData = u_space.elements[iel].data
        t_data: ElementData = t_space.elements[iel].data

        points, weights = self.space.quadrature
        dim = s_data.cell.dimension
        x, jac, det_jac, inv_jac = s_space.elements[iel].evaluate_mapping(points)

        # basis
        s_phi_tab = s_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        u_phi_tab = u_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        t_phi_tab = t_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)

        n_s_phi = s_phi_tab.shape[2]
        n_u_phi = u_phi_tab.shape[2]
        n_t_phi = t_phi_tab.shape[2]

        n_s_dof = n_s_phi * s_components
        n_u_dof = n_u_phi * u_components
        n_t_dof = n_t_phi * t_components

        n_dof = n_s_dof + n_u_dof + n_t_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

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
                xv = x[i]

                if dim == 2:
                    c = 0
                    a_sx = alpha[:, c : n_s_dof + c : s_components]
                    a_ux = alpha[
                        :,
                        n_s_dof + c : n_s_dof + n_u_dof + c : u_components,
                    ]
                    a_t = alpha[
                        :,
                        n_s_dof
                        + n_u_dof
                        + c : n_s_dof
                        + n_u_dof
                        + n_t_dof
                        + c : t_components,
                    ]

                    c = 1
                    a_sy = alpha[:, c : n_s_dof + c : s_components]
                    a_uy = alpha[
                        :,
                        n_s_dof + c : n_s_dof + n_u_dof + c : u_components,
                    ]

                    sx_h = a_sx @ s_phi_tab[0, i, :, 0:dim]
                    sy_h = a_sy @ s_phi_tab[0, i, :, 0:dim]
                    ux_h = a_ux @ u_phi_tab[0, i, :, 0:dim]
                    uy_h = a_uy @ u_phi_tab[0, i, :, 0:dim]

                    th = a_t @ t_phi_tab[0, i, :, 0:dim]

                    uh = VecValDer(
                        np.hstack((ux_h.val, uy_h.val)), np.hstack((ux_h.der, uy_h.der))
                    )

                    sh = VecValDer(
                        np.vstack((sx_h.val, sy_h.val)), np.vstack((sx_h.der, sy_h.der))
                    )

                    tr_s_h = VecValDer(sh.val.trace(), sh.der.trace())
                    A_sh = (1.0 / (2.0 * f_mu_star[i])) * (
                        sh
                        - (
                            f_lambda_star[i]
                            / (2.0 * f_mu_star[i] + dim * f_lambda_star[i])
                        )
                        * tr_s_h
                        * Imat
                    )

                    grad_s_phi = s_phi_tab[1 : s_phi_tab.shape[0] + 1, i, :, 0:dim]
                    div_tau = np.array(
                        [np.trace(grad_s_phi, axis1=0, axis2=2) / det_jac[i]]
                    )

                    div_sh_x = a_sx @ div_tau.T
                    div_sh_y = a_sy @ div_tau.T
                    div_sh = VecValDer(
                        np.hstack((div_sh_x.val, div_sh_y.val)),
                        np.hstack((div_sh_x.der, div_sh_y.der)),
                    )

                else:
                    c = 0
                    a_sx = alpha[:, c : n_s_dof + c : s_components]
                    a_ux = alpha[
                        :,
                        n_s_dof + c : n_s_dof + n_u_dof + c : u_components,
                    ]
                    a_tx = alpha[
                        :,
                        n_s_dof
                        + n_u_dof
                        + c : n_s_dof
                        + n_u_dof
                        + n_t_dof
                        + c : t_components,
                    ]

                    c = 1
                    a_sy = alpha[:, c : n_s_dof + c : s_components]
                    a_uy = alpha[
                        :,
                        n_s_dof + c : n_s_dof + n_u_dof + c : u_components,
                    ]
                    a_ty = alpha[
                        :,
                        n_s_dof
                        + n_u_dof
                        + c : n_s_dof
                        + n_u_dof
                        + n_t_dof
                        + c : t_components,
                    ]

                    c = 2
                    a_sz = alpha[:, c : n_s_dof + c : s_components]
                    a_uz = alpha[
                        :,
                        n_s_dof + c : n_s_dof + n_u_dof + c : u_components,
                    ]
                    a_tz = alpha[
                        :,
                        n_s_dof
                        + n_u_dof
                        + c : n_s_dof
                        + n_u_dof
                        + n_t_dof
                        + c : t_components,
                    ]

                    sx_h = a_sx @ s_phi_tab[0, i, :, 0:dim]
                    sy_h = a_sy @ s_phi_tab[0, i, :, 0:dim]
                    sz_h = a_sz @ s_phi_tab[0, i, :, 0:dim]

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

                    tr_s_h = VecValDer(sh.val.trace(), sh.der.trace())
                    A_sh = (1.0 / (2.0 * f_mu(xv[0], xv[1], xv[2]))) * (
                        sh
                        - (
                            f_lambda(xv[0], xv[1], xv[2])
                            / (
                                2.0 * f_mu(xv[0], xv[1], xv[2])
                                + dim * f_lambda(xv[0], xv[1], xv[2])
                            )
                        )
                        * tr_s_h
                        * Imat
                    )

                    grad_s_phi = s_phi_tab[1 : s_phi_tab.shape[0] + 1, i, :, 0:dim]
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

                equ_1_integrand = (
                    s_phi_tab[0, i, :, 0:dim] @ A_sh.T + div_tau.T @ div_sh
                )
                equ_2_integrand = u_phi_tab[0, i, :, 0:dim] @ uh
                equ_3_integrand = t_phi_tab[0, i, :, 0:dim] @ th

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, 0:n_s_dof:1] = (equ_1_integrand).reshape(
                    (n_s_dof,)
                )
                multiphysic_integrand[:, n_s_dof : n_s_dof + n_u_dof : 1] = (
                    equ_2_integrand
                ).reshape((n_u_dof,))
                multiphysic_integrand[
                    :, n_s_dof + n_u_dof : n_s_dof + n_u_dof + n_t_dof : 1
                ] = (equ_3_integrand).reshape((n_t_dof,))

                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        return r_el, j_el

    def evaluate_form_vectorized(self, element_index, alpha):
        iel = element_index
        if self.space is None or self.functions is None:
            raise ValueError

        s_space = self.space.discrete_spaces["s"]
        u_space = self.space.discrete_spaces["u"]
        t_space = self.space.discrete_spaces["t"]

        f_rhs = self.functions["rhs"]
        f_lambda = self.functions["lambda"]
        f_mu = self.functions["mu"]

        s_components = s_space.n_comp
        u_components = u_space.n_comp
        t_components = t_space.n_comp

        s_data: ElementData = s_space.elements[iel].data
        u_data: ElementData = u_space.elements[iel].data
        t_data: ElementData = t_space.elements[iel].data

        points, weights = self.space.quadrature
        dim = s_data.cell.dimension
        x, jac, det_jac, inv_jac = s_space.elements[iel].evaluate_mapping(points)

        # basis
        s_phi_tab = s_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        u_phi_tab = u_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        t_phi_tab = t_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)

        n_s_phi = s_phi_tab.shape[2]
        n_u_phi = u_phi_tab.shape[2]
        n_t_phi = t_phi_tab.shape[2]

        n_s_dof = n_s_phi * s_components
        n_u_dof = n_u_phi * u_components
        n_t_dof = n_t_phi * t_components

        n_dof = n_s_dof + n_u_dof + n_t_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        f_val_star = f_rhs(x[:, 0], x[:, 1], x[:, 2])

        lambda_v = f_lambda(x[:, 0], x[:, 1], x[:, 2])
        mu_v = f_mu(x[:, 0], x[:, 1], x[:, 2])

        # Vectorized contributions
        # s : stress
        # m : coupled stress
        # u : displacement
        # t : rotation

        # (s,s) block
        s_phi_star = s_phi_tab[0, :, :, 0:dim]
        s_j_el = np.array([np.dot(phi, phi.T) for phi in s_phi_star]).T @ (
            (1.0 / (2.0 * mu_v)) * det_jac * weights
        )
        for c in range(s_components):
            b = c
            e = b + n_s_dof
            j_el[b:e:s_components, b:e:s_components] += s_j_el

        vol_factor = (1.0 / (2.0 * mu_v)) * (lambda_v / (2.0 * mu_v + dim * lambda_v))
        vol_s_j_el = (
            -1.0
            * np.array([np.outer(phi, phi) for phi in s_phi_tab[0, :, :, 0:dim]]).T
            @ (vol_factor * det_jac * weights)
        )
        j_el[0:n_s_dof, 0:n_s_dof] += vol_s_j_el

        # (div s, div tau) block
        grad_s_phi_star = s_phi_tab[1 : s_phi_tab.shape[0] + 1, :, :, 0:dim]
        div_tau_star = np.trace(grad_s_phi_star, axis1=0, axis2=3).T * (1 / det_jac)
        div_tau_block_outer = np.array(
            [np.outer(div_phi, div_phi) for div_phi in div_tau_star.T]
        ).T @ (det_jac * weights)
        for sc in range(s_components):
            sb = sc
            se = sb + n_s_dof
            j_el[sb:se:s_components, sb:se:s_components] += div_tau_block_outer

        # (u,u) block
        u_block_outer = np.array(
            [np.outer(phi, phi) for phi in u_phi_tab[0, :, :, 0]]
        ).T @ (det_jac * weights)
        for c in range(u_components):
            b = c + n_s_dof
            e = b + n_u_dof
            j_el[b:e:u_components, b:e:u_components] += u_block_outer

        # (t, t) block
        t_block_outer = np.array(
            [np.outer(phi, phi) for phi in t_phi_tab[0, :, :, 0]]
        ).T @ (det_jac * weights)
        for c in range(t_components):
            b = c + n_s_dof + n_u_dof
            e = b + n_t_dof
            j_el[b:e:t_components, b:e:t_components] += t_block_outer

        return r_el, j_el
