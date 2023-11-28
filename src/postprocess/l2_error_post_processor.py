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
            phi_tab = el_data.basis.phi

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
                    diff_f = f_e - f_h
                    l2_error += det_jac[i] * weights[i] * np.trace(diff_f.T @ diff_f)

        l2_errors.append(np.sqrt(l2_error))

    return l2_errors
