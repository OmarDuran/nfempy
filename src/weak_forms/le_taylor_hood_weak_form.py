import auto_diff as ad
import numpy as np
from auto_diff.vecvalder import VecValDer

from basis.element_data import ElementData
from weak_forms.weak_from import WeakForm


class LETaylorHoodWeakForm(WeakForm):
    """Taylor-Hood weak form for linear elasticity using P2-P1 elements."""

    def evaluate_form(self, element_index: int, alpha: float):
        iel = element_index
        if self.space is None or self.functions is None:
            raise ValueError

        # Get spaces
        u_space = self.space.discrete_spaces["u"]  # displacement (vector)
        p_space = self.space.discrete_spaces["p"]  # pressure (scalar)
        u_components = u_space.n_comp

        # Get functions
        f_rhs = self.functions["rhs"]
        f_lambda = self.functions["lambda"]
        f_mu = self.functions["mu"]

        # Get element data
        u_data: ElementData = u_space.elements[iel].data
        p_data: ElementData = p_space.elements[iel].data

        # Get quadrature data
        cell = u_data.cell
        dim = cell.dimension
        points, weights = self.space.quadrature[dim]
        x, jac, det_jac, inv_jac = u_space.elements[iel].evaluate_mapping(points)

        # Get basis functions and their derivatives
        u_phi_tab = u_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        p_phi_tab = p_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)

        n_u_phi = u_phi_tab.shape[2]
        n_p_phi = p_phi_tab.shape[2]
        n_u_dof = n_u_phi * u_components
        n_p_dof = n_p_phi
        n_dof = n_u_dof + n_p_dof

        # Partial local vectorization
        f_val_star = f_rhs(x[:, 0], x[:, 1], x[:, 2])
        f_lambda_star = f_lambda(x[:, 0], x[:, 1], x[:, 2])
        f_mu_star = f_mu(x[:, 0], x[:, 1], x[:, 2])

        # constant directors
        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])
        Imat = np.identity(dim)

        with ad.AutoDiff(alpha) as alpha:
            # Initialize element form
            el_form = np.zeros(n_dof)

            # Add right-hand side term
            for c in range(u_components):
                b = c
                e = b + n_u_dof
                el_form[b:e:u_components] += (
                    det_jac * weights * u_phi_tab[0, :, :, 0].T @ f_val_star[c]
                )

            # Main loop over quadrature points
            for i, omega in enumerate(weights):
                xv = x[i]
                if dim == 2:
                    inv_jac_m = np.vstack((inv_jac[i] @ e1, inv_jac[i] @ e2))
                else:  # dim == 3
                    inv_jac_m = np.vstack((inv_jac[i] @ e1, inv_jac[i] @ e2, inv_jac[i] @ e3))

                grad_phi_u = (inv_jac_m @ u_phi_tab[1:u_phi_tab.shape[0]+1, i, :, 0]).T
                phi_p = p_phi_tab[0, i, :, 0:dim]

                # Get displacement gradients
                if dim == 2:
                    c = 0
                    a_ux = alpha[:,c:n_u_dof: u_components]
                    c = 1
                    a_uy = alpha[:,c:n_u_dof + c:u_components]

                    grad_uh_x = a_ux @ grad_phi_u
                    grad_uh_y = a_uy @ grad_phi_u
                    grad_uh = VecValDer(
                        np.vstack((grad_uh_x.val, grad_uh_y.val)),
                        np.vstack((grad_uh_x.der, grad_uh_y.der)),
                    )

                    # Compute strain tensor (symmetric gradient)
                    eh = 0.5 * (grad_uh + grad_uh.T)
                    tr_eh = VecValDer(eh.val.trace(), eh.der.trace())

                    # Compute divergence of displacement
                    grad_u_phi = u_phi_tab[1:u_phi_tab.shape[0], i, :, 0:dim]
                    div_u = np.array([np.trace(grad_u_phi, axis1=0, axis2=2)])
                    trace_grad_u_phi = np.array([np.trace(grad_u_phi, axis1=0, axis2=2), np.trace(grad_u_phi, axis1=0, axis2=2)])

                    div_uh = (a_ux @ div_u.T, a_uy @ div_u.T)
                    div_uh = VecValDer(div_uh[0].val, div_uh[0].der)

                    # Compute pressure field
                    alpha_p = alpha[:, n_u_dof:n_u_dof + n_p_dof]
                    ph = alpha_p @ phi_p

                    # Material parameters at quadrature point
                    lambda_v = f_lambda_star[i]
                    mu_v = f_mu_star[i]

                    # Compute deviatoric stress tensor
                    sh_dev = 2.0 * mu_v * (eh - (1.0/dim) * tr_eh * Imat)

                    # Replace the loops with equation integrands
                    equ_1_integrand = grad_phi_u @ sh_dev.T - ph * trace_grad_u_phi.T
                    equ_2_integrand = (1.0/lambda_v) * ph * phi_p - div_uh * phi_p

                    # Assemble multiphysic integrand
                    multiphysic_integrand = np.zeros((1, n_dof))
                    multiphysic_integrand[:, 0:n_u_dof] = (equ_1_integrand).reshape((1, n_u_dof))
                    multiphysic_integrand[:, n_u_dof:n_u_dof + n_p_dof] = equ_2_integrand.reshape((1, n_p_dof))

                    discrete_integrand = multiphysic_integrand.reshape((n_dof,))
                    el_form += det_jac[i] * omega * discrete_integrand
                else:  # dim == 3
                    c = 0
                    a_ux = alpha[:,c:n_u_dof + c:u_components]
                    c = 1
                    a_uy = alpha[:,c:n_u_dof + c:u_components]
                    c = 2
                    a_uz = alpha[:,c:n_u_dof + c:u_components]

                    grad_uh_x = a_ux @ grad_phi_u
                    grad_uh_y = a_uy @ grad_phi_u
                    grad_uh_z = a_uz @ grad_phi_u
                    grad_uh = VecValDer(
                        np.vstack((grad_uh_x.val, grad_uh_y.val, grad_uh_z.val)),
                        np.vstack((grad_uh_x.der, grad_uh_y.der, grad_uh_z.der)),
                    )

                    # Compute strain tensor (symmetric gradient)
                    eh = 0.5 * (grad_uh + grad_uh.T)
                    tr_eh = VecValDer(eh.val.trace(), eh.der.trace())

                    # Compute divergence of displacement
                    grad_u_phi = u_phi_tab[1:u_phi_tab.shape[0], i, :, 0:dim]
                    div_u = np.array([np.trace(grad_u_phi, axis1=0, axis2=2) / det_jac[i]])
                    div_uh = (a_ux @ div_u.T, a_uy @ div_u.T, a_uz @ div_u.T)
                    div_uh = VecValDer(div_uh[0].val, div_uh[0].der)

                    # Compute pressure field
                    alpha_p = alpha[:, n_u_dof:n_u_dof + n_p_dof]
                    ph = alpha_p @ phi_p

                    # Material parameters at quadrature point
                    lambda_v = f_lambda_star[i]
                    mu_v = f_mu_star[i]

                    # Compute deviatoric stress tensor
                    sh_dev = 2.0 * mu_v * (eh - (1.0/dim) * tr_eh * Imat)

                    # Replace the loops with equation integrands
                    equ_1_integrand = (grad_phi_u @ sh_dev.T).reshape(n_u_dof,)
                    equ_2_integrand = -ph * (grad_phi_u @ np.ones(dim)).reshape(n_u_dof,)
                    equ_3_integrand = (1.0/lambda_v) * ph * phi_p - div_uh * phi_p

                    # Assemble multiphysic integrand
                    multiphysic_integrand = np.zeros((1, n_dof))
                    multiphysic_integrand[:, 0:n_u_dof] = (equ_1_integrand + equ_2_integrand).reshape((1, n_u_dof))
                    multiphysic_integrand[:, n_u_dof:n_u_dof + n_p_dof] = equ_3_integrand.reshape((1, n_p_dof))

                    discrete_integrand = multiphysic_integrand.reshape((n_dof,))
                    el_form += det_jac[i] * omega * discrete_integrand

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))
        return r_el, j_el


class LETaylorHoodWeakFormBCDirichlet(WeakForm):
    """Dirichlet boundary conditions for Taylor-Hood formulation."""

    def evaluate_form(self, element_index, alpha):
        u_D = self.functions["u"]

        iel = element_index
        u_space = self.space.discrete_spaces["u"]
        u_components = u_space.n_comp

        u_data: ElementData = u_space.bc_elements[iel].data

        cell = u_data.cell
        dim = cell.dimension
        points, weights = self.space.bc_quadrature[dim]
        x, jac, det_jac, inv_jac = u_space.bc_elements[iel].evaluate_mapping(points)

        u_phi_tab = u_space.bc_elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)

        n_u_phi = u_phi_tab.shape[2]
        n_u_dof = n_u_phi * u_components
        n_dof = n_u_dof

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

        return r_el, j_el
