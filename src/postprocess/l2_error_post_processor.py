import numpy as np

from basis.element_family import family_by_name
from spaces.product_space import ProductSpace


def l2_error(dim, fe_space, functions, alpha):
    l2_errors = []
    for item in fe_space.discrete_spaces.items():
        name, space = item
        l2_error = 0.0
        indexes = [
            i
            for i, element in enumerate(space.elements)
            if element.data.dimension == dim
        ]
        for i in indexes:
            n_components = space.n_comp
            el_data = space.elements[i].data
            cell = el_data.cell
            points = el_data.quadrature.points
            weights = el_data.quadrature.weights
            phi_tab = space.elements[i].evaluate_basis(points)

            x = el_data.mapping.x
            det_jac = el_data.mapping.det_jac
            inv_jac = el_data.mapping.inv_jac

            # scattering dof
            dest = fe_space.discrete_spaces_destination_indexes(i)[name]
            alpha_l = alpha[dest]

            # vectorization
            exact = functions[name]
            n_phi = phi_tab.shape[2]
            alpha_star = np.array(np.split(alpha_l, n_phi))
            if space.family is family_by_name("Lagrange"):
                f_e_s = exact(x[:, 0], x[:, 1], x[:, 2])[0:dim, :]
                f_h_s = (phi_tab[0, :, :, 0] @ alpha_star[:, 0:dim]).T
                l2_error += np.sum(
                    det_jac * weights * (f_e_s - f_h_s) * (f_e_s - f_h_s)
                )
            else:
                f_e = np.array([exact(xv[0], xv[1], xv[2]) for xv in x])
                f_h = np.array(
                    [
                        np.vstack(
                            tuple(
                                [
                                    phi_tab[0, k, :, 0:dim].T @ alpha_star[:, c]
                                    for c in range(n_components)
                                ]
                            )
                        )
                        for k in range(len(points))
                    ]
                )
                diff_f = f_e - f_h
                l2_error += np.sum(
                    det_jac * weights * np.array([np.trace(e.T @ e) for e in diff_f])
                )

        l2_errors.append(np.sqrt(l2_error))

    return l2_errors


def grad_error(dim, fe_space, functions, alpha):
    # constant directors
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])

    grad_errors = []
    for item in fe_space.discrete_spaces.items():
        name, space = item
        grad_error = 0.0
        indexes = [
            i
            for i, element in enumerate(space.elements)
            if element.data.dimension == dim
        ]
        for idx in indexes:
            n_components = space.n_comp
            el_data = space.elements[idx].data
            cell = el_data.cell
            points = el_data.quadrature.points
            weights = el_data.quadrature.weights
            phi_tab = space.elements[idx].evaluate_basis(points)

            x = el_data.mapping.x
            det_jac = el_data.mapping.det_jac
            inv_jac = el_data.mapping.inv_jac

            # scattering dof
            dest = fe_space.discrete_spaces_destination_indexes(idx)[name]
            alpha_l = alpha[dest]

            # vectorization
            n_phi = phi_tab.shape[2]
            alpha_star = np.array(np.split(alpha_l, n_phi))
            if space.family is family_by_name("Lagrange"):
                grad_name = "grad_" + name
                exact = functions[grad_name]
                for i, omega in enumerate(weights):
                    if dim == 2:
                        inv_jac_m = np.vstack((inv_jac[i] @ e1, inv_jac[i] @ e2))
                        grad_phi = (
                            inv_jac_m @ phi_tab[1 : phi_tab.shape[0] + 1, i, :, 0]
                        ).T
                    else:
                        inv_jac_m = np.vstack(
                            (inv_jac[i] @ e1, inv_jac[i] @ e2, inv_jac[i] @ e3)
                        )
                        grad_phi = (
                            inv_jac_m @ phi_tab[1 : phi_tab.shape[0] + 1, i, :, 0]
                        ).T

                    f_e = exact(x[i, 0], x[i, 1], x[i, 2])
                    f_h = np.vstack(
                        tuple(
                            [alpha_star[:, c] @ grad_phi for c in range(n_components)]
                        )
                    )
                    diff_f = f_e - f_h
                    grad_error += det_jac[i] * weights[i] * np.trace(diff_f.T @ diff_f)

        grad_errors.append(np.sqrt(grad_error))

    return grad_errors


def div_error(dim, fe_space, functions, alpha, skip_fields = []):
    vec_families = [
        family_by_name("RT"),
        family_by_name("BDM"),
    ]

    def compute_div_error(idx):
        n_components = space.n_comp
        el_data = space.elements[idx].data
        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = space.elements[idx].evaluate_basis(points)

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # scattering dof
        dest = fe_space.discrete_spaces_destination_indexes(idx)[name]
        alpha_l = alpha[dest]

        # vectorization
        n_phi = phi_tab.shape[2]
        alpha_star = np.array(np.split(alpha_l, n_phi))

        grad_m_phi_star = phi_tab[1: phi_tab.shape[0] + 1, :, :, 0:dim]
        div_v_star = np.trace(grad_m_phi_star, axis1=0, axis2=3).T * (1 / det_jac)
        f_e = np.array([exact(xv[0], xv[1], xv[2]) for xv in x])
        f_h = div_v_star.T @ alpha_star
        div_error = np.sum(((f_e - f_h) * (f_e - f_h)).T @ (det_jac * weights))
        return div_error

    div_errors = []
    for item in fe_space.discrete_spaces.items():
        name, space = item
        if space.family not in vec_families:
            continue
        if name in skip_fields:
            continue
        indexes = [
            i
            for i, element in enumerate(space.elements)
            if element.data.dimension == dim
        ]

        div_name = "div_" + name
        exact = functions[div_name]
        div_error = np.sum([compute_div_error(idx) for idx in indexes])
        div_errors.append(np.sqrt(div_error))

    return div_errors


def div_scaled_error(dim, fe_space, functions, alpha, skip_fields = []):
    vec_families = [
        family_by_name("RT"),
        family_by_name("BDM"),
    ]

    def compute_div_error(idx):
        n_components = space.n_comp
        el_data = space.elements[idx].data
        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = space.elements[idx].evaluate_basis(points)

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # scattering dof
        dest = fe_space.discrete_spaces_destination_indexes(idx)[name]
        alpha_l = alpha[dest]

        # vectorization
        n_phi = phi_tab.shape[2]
        alpha_star = np.array(np.split(alpha_l, n_phi))

        gamma = scale(x[:, 0], x[:, 1], x[:, 2])
        grad_gamma = grad_scale(x[:, 0], x[:, 1], x[:, 2])
        grad_m_phi_star = phi_tab[1: phi_tab.shape[0] + 1, :, :, 0:dim]

        div_v_star = (
                np.array(
                    [
                        [
                            np.trace(
                                np.outer(grad_gamma[:, i], phi_tab[0, i, j, 0:dim])
                            )
                            for i in range(len(points))
                        ]
                        for j in range(n_phi)
                    ]
                )
        )

        div_v_star += np.trace(grad_m_phi_star, axis1=0, axis2=3).T * (
                gamma / det_jac
        )

        f_e = np.array([exact(xv[0], xv[1], xv[2]) for xv in x])
        f_h = div_v_star.T @ alpha_star
        div_error = np.sum(((f_e - f_h) * (f_e - f_h)).T @ (det_jac * weights))
        return div_error

    div_errors = []
    for item in fe_space.discrete_spaces.items():
        name, space = item
        if space.family not in vec_families:
            continue
        if name in skip_fields:
            continue
        indexes = [
            i
            for i, element in enumerate(space.elements)
            if element.data.dimension == dim
        ]

        div_name = "div_" + name
        exact = functions[div_name]
        scale = functions.get("gamma", None)
        grad_scale = functions.get("grad_gamma", None)
        if scale is None or grad_scale is None:
            continue
        div_error = np.sum([compute_div_error(idx) for idx in indexes])
        div_errors.append(np.sqrt(div_error))

    return div_errors


def devia_l2_error(dim, fe_space, functions, alpha):
    vec_families = [
        family_by_name("RT"),
        family_by_name("BDM"),
    ]

    # Compute volumetric integral
    tr_T_avgs = {}
    for item in fe_space.discrete_spaces.items():
        name, space = item
        tr_T_e_avg = 0.0
        tr_T_h_avg = 0.0
        if space.family not in vec_families:
            continue
        indexes = [
            i
            for i, element in enumerate(space.elements)
            if element.data.dimension == dim
        ]
        for i in indexes:
            n_components = space.n_comp
            el_data = space.elements[i].data
            cell = el_data.cell
            points = el_data.quadrature.points
            weights = el_data.quadrature.weights
            phi_tab = space.elements[i].evaluate_basis(points)

            x = el_data.mapping.x
            det_jac = el_data.mapping.det_jac
            inv_jac = el_data.mapping.inv_jac

            # scattering dof
            dest = fe_space.discrete_spaces_destination_indexes(i)[name]
            alpha_l = alpha[dest]

            # vectorization
            exact = functions[name]
            n_phi = phi_tab.shape[2]
            alpha_star = np.array(np.split(alpha_l, n_phi))
            for i, omega in enumerate(weights):
                f_e = exact(x[i, 0], x[i, 1], x[i, 2])
                f_h = np.vstack(
                    tuple(
                        [
                            phi_tab[0, i, :, 0:dim].T @ alpha_star[:, c]
                            for c in range(n_components)
                        ]
                    )
                )
                tr_T_e_avg += det_jac[i] * weights[i] * np.trace(f_e) / float(dim)
                tr_T_h_avg += det_jac[i] * weights[i] * np.trace(f_h) / float(dim)

        tr_T_avgs.__setitem__(name, (tr_T_e_avg, tr_T_h_avg))

    l2_errors = []
    for item in fe_space.discrete_spaces.items():
        name, space = item
        l2_error = 0.0
        if space.family not in vec_families:
            continue
        if space.n_comp != dim:
            l2_errors.append(np.sqrt(l2_error))
            continue
        indexes = [
            i
            for i, element in enumerate(space.elements)
            if element.data.dimension == dim
        ]
        tr_T_e_avg, tr_T_h_avg = tr_T_avgs[name]
        for i in indexes:
            n_components = space.n_comp
            el_data = space.elements[i].data
            cell = el_data.cell
            points = el_data.quadrature.points
            weights = el_data.quadrature.weights
            phi_tab = space.elements[i].evaluate_basis(points)

            x = el_data.mapping.x
            det_jac = el_data.mapping.det_jac
            inv_jac = el_data.mapping.inv_jac

            # scattering dof
            dest = fe_space.discrete_spaces_destination_indexes(i)[name]
            alpha_l = alpha[dest]

            # vectorization
            exact = functions[name]
            n_phi = phi_tab.shape[2]
            alpha_star = np.array(np.split(alpha_l, n_phi))
            for i, omega in enumerate(weights):
                f_e = exact(x[i, 0], x[i, 1], x[i, 2])
                f_h = np.vstack(
                    tuple(
                        [
                            phi_tab[0, i, :, 0:dim].T @ alpha_star[:, c]
                            for c in range(n_components)
                        ]
                    )
                )
                f_e -= tr_T_e_avg * np.eye(dim)
                f_h -= tr_T_h_avg * np.eye(dim)
                diff_f = f_e - f_h
                l2_error += det_jac[i] * weights[i] * np.trace(diff_f.T @ diff_f)

        l2_errors.append(np.sqrt(l2_error))

    return l2_errors
