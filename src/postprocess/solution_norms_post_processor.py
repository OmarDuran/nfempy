import numpy as np

from basis.element_family import family_by_name
from spaces.product_space import ProductSpace


def l2_norm(dim, fe_space, functions, skip_fields=[]):
    l2_norms = []
    points, weights = fe_space.quadrature
    for item in fe_space.discrete_spaces.items():
        name, space = item
        if name in skip_fields:
            continue
        l2_norm = 0.0
        indexes = [
            i
            for i, element in enumerate(space.elements)
            if element.data.dimension == dim
        ]
        for i in indexes:
            n_components = space.n_comp
            el_data = space.elements[i].data
            cell = el_data.cell
            x, jac, det_jac, inv_jac = space.elements[i].evaluate_mapping(points)

            # vectorization
            exact = functions[name]
            if space.family is family_by_name("Lagrange"):
                f_e_s = exact(x[:, 0], x[:, 1], x[:, 2])[0:dim, :]
                l2_norm += np.sum(
                    det_jac * weights * (f_e_s) * (f_e_s)
                )
            else:
                f_e = np.array([exact(xv[0], xv[1], xv[2]) for xv in x])
                l2_norm += np.sum(
                    det_jac * weights * np.array([np.trace(e.T @ e) for e in f_e])
                )

        l2_norms.append(np.sqrt(l2_norm))

    return l2_norms

def div_norm(dim, fe_space, functions, skip_fields=[]):
    vec_families = [
        family_by_name("RT"),
        family_by_name("BDM"),
    ]
    points, weights = fe_space.quadrature

    def compute_div_norm(idx):

        n_components = space.n_comp
        el_data = space.elements[idx].data
        cell = el_data.cell
        x, jac, det_jac, inv_jac = space.elements[idx].evaluate_mapping(points)

        # vectorization
        f_e = np.array([exact(xv[0], xv[1], xv[2]) for xv in x])
        div_error = np.sum((f_e * f_e).T @ (det_jac * weights))
        return div_error

    div_norms = []
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
        div_norm = np.sum([compute_div_norm(idx) for idx in indexes])
        div_norms.append(np.sqrt(div_norm))

    return div_norms


# def div_scaled_norm(dim, fe_space, functions, skip_fields=[]):
#     vec_families = [
#         family_by_name("RT"),
#         family_by_name("BDM"),
#     ]
#     points, weights = fe_space.quadrature
#
#     def compute_div_norm(idx):
#         n_components = space.n_comp
#         el_data = space.elements[idx].data
#         cell = el_data.cell
#         x, jac, det_jac, inv_jac = space.elements[idx].evaluate_mapping(points)
#
#         # vectorization
#         f_e = np.array([exact(xv[0], xv[1], xv[2]) for xv in x])
#         div_error = np.sum(((f_e - f_h) * (f_e - f_h)).T @ (det_jac * weights))
#         return div_error
#
#     div_errors = []
#     for item in fe_space.discrete_spaces.items():
#         name, space = item
#         if space.family not in vec_families:
#             continue
#         if name in skip_fields:
#             continue
#         indexes = [
#             i
#             for i, element in enumerate(space.elements)
#             if element.data.dimension == dim
#         ]
#
#         div_name = "div_" + name
#         exact = functions[div_name]
#         scale = functions.get("gamma", None)
#         grad_scale = functions.get("grad_gamma", None)
#         if scale is None or grad_scale is None:
#             continue
#         div_error = np.sum([compute_div_error(idx) for idx in indexes])
#         div_errors.append(np.sqrt(div_error))
#
#     return div_errors


def devia_l2_error(dim, fe_space, functions, alpha, skip_fields=[]):
    vec_families = [
        family_by_name("RT"),
        family_by_name("BDM"),
    ]

    points, weights = fe_space.quadrature

    # Compute volumetric integral
    tr_T_avgs = {}
    for item in fe_space.discrete_spaces.items():
        name, space = item
        if space.family not in vec_families:
            continue
        if name in skip_fields:
            continue
        tr_T_e_avg = 0.0
        tr_T_h_avg = 0.0
        exact = functions[name]
        indexes = [
            i
            for i, element in enumerate(space.elements)
            if element.data.dimension == dim
        ]
        for idx in indexes:
            n_components = space.n_comp
            el_data = space.elements[idx].data
            cell = el_data.cell
            x, jac, det_jac, inv_jac = space.elements[idx].evaluate_mapping(points)
            phi_tab = space.elements[idx].evaluate_basis(points, jac, det_jac, inv_jac)

            # scattering dof
            dest = fe_space.discrete_spaces_destination_indexes(idx)[name]
            alpha_l = alpha[dest]

            # vectorization
            n_phi = phi_tab.shape[2]
            alpha_star = np.array(np.split(alpha_l, n_phi))
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
            tr_T_e_avg = np.sum(
                (det_jac * weights * np.trace(f_e, axis1=1, axis2=2) / float(dim))
            )
            tr_T_h_avg = np.sum(
                (det_jac * weights * np.trace(f_h, axis1=1, axis2=2) / float(dim))
            )

        tr_T_avgs.__setitem__(name, (tr_T_e_avg, tr_T_h_avg))

    l2_errors = []
    for item in fe_space.discrete_spaces.items():
        name, space = item
        l2_error = 0.0
        if space.family not in vec_families:
            continue
        if name in skip_fields:
            continue
        indexes = [
            i
            for i, element in enumerate(space.elements)
            if element.data.dimension == dim
        ]
        tr_T_e_avg, tr_T_h_avg = tr_T_avgs[name]
        for idx in indexes:
            n_components = space.n_comp
            el_data = space.elements[idx].data
            cell = el_data.cell
            x, jac, det_jac, inv_jac = space.elements[idx].evaluate_mapping(points)
            phi_tab = space.elements[idx].evaluate_basis(points, jac, det_jac, inv_jac)

            # scattering dof
            dest = fe_space.discrete_spaces_destination_indexes(idx)[name]
            alpha_l = alpha[dest]

            # vectorization
            exact = functions[name]
            n_phi = phi_tab.shape[2]
            alpha_star = np.array(np.split(alpha_l, n_phi))
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

            f_e = np.array([T_e - tr_T_e_avg * np.eye(dim) for T_e in f_e])
            f_h = np.array([T_h - tr_T_h_avg * np.eye(dim) for T_h in f_h])
            diff_f = f_e - f_h
            l2_error += np.sum(
                det_jac * weights * np.array([np.trace(e.T @ e) for e in diff_f])
            )
        l2_errors.append(np.sqrt(l2_error))

    return l2_errors
