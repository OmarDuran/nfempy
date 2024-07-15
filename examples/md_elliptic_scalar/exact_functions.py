import numpy as np
from functools import partial


def f_kappa(x, y, z, m_kappa):
    return m_kappa


def f_delta(x, y, z, m_delta):
    return m_delta


def p_f_exact(x, y, z):
    return np.array([np.sin(2 * np.pi * y)])


def grad_p_f_exact(x, y, z):
    return np.array([2 * np.pi * np.cos(2 * np.pi * y)])


def u_f_exact(x, y, z, m_c, m_kappa, m_delta):
    return np.array(
        [
            -f_kappa(x, y, z, m_kappa)
            * f_delta(x, y, z, m_delta)
            * grad_p_f_exact(x, y, z)
        ]
    )


def p_exact(x, y, z, m_c, m_kappa, m_delta):
    return np.array([(m_c * x + f_delta(x, y, z, m_delta)) * p_f_exact(x, y, z)])


def u_exact(x, y, z, m_c, m_kappa, m_delta):
    return np.array(
        [
            [
                -m_c * p_f_exact(x, y, z)[0],
                -(m_c * x + f_delta(x, y, z, m_delta)) * grad_p_f_exact(x, y, z)[0],
            ]
        ]
    )


def f_c0_rhs(x, y, z, m_c, m_kappa, m_delta):
    return np.array(
        [[4 * (np.pi**2) * (m_c * x + f_delta(x, y, z, m_delta)) * p_f_exact(x, y, z)]]
    )


def f_c1_rhs(x, y, z, m_c, m_kappa, m_delta):
    return np.array(
        [
            [
                4
                * (np.pi**2)
                * f_kappa(x, y, z, m_kappa)
                * f_delta(x, y, z, m_delta)
                * np.sin(2 * np.pi * y)
            ]
        ]
    )


def get_exact_functions_by_co_dimension(
    co_dimension, flux_name, potential_name, m_c, m_kappa, m_delta
):
    if co_dimension == 0:
        exact_functions = {
            flux_name: partial(u_exact, m_c=m_c, m_kappa=m_kappa, m_delta=m_delta),
            potential_name: partial(p_exact, m_c=m_c, m_kappa=m_kappa, m_delta=m_delta),
        }
    elif co_dimension == 1:
        exact_functions = {
            flux_name: partial(u_f_exact, m_c=m_c, m_kappa=m_kappa, m_delta=m_delta),
            potential_name: p_f_exact,
        }
    else:
        raise ValueError("Case not available.")

    return exact_functions
