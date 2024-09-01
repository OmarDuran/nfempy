import numpy as np
from functools import partial


def p_f_exact(x, y, z):
    return np.array([np.sin(2 * np.pi * x)])


def grad_p_f_exact(x, y, z):
    return np.array([2 * np.pi * np.cos(2 * np.pi * x)])


def u_f_exact(x, y, z, m_c1, m_c2, m_kappa_c0, m_kappa_c1, m_delta):
    return np.array([-m_kappa_c1 * m_delta * grad_p_f_exact(x, y, z)])


def p_exact(x, y, z, m_c1, m_c2, m_kappa_c0, m_kappa_c1, m_delta):
    p_val = np.where(
        y < 0.5,
        np.array([(m_c1 * (0.5 - y) + 1.0) * p_f_exact(x, y, z)]),
        np.array([(m_c2 * (y - 0.5) + 1.0) * p_f_exact(x, y, z)]),
    )
    return p_val


def u_exact(x, y, z, m_c1, m_c2, m_kappa_c0, m_kappa_c1, m_delta):
    u_1_val = np.array(
        [
            [
                -m_kappa_c0 * (m_c1 * (0.5 - y) + 1.0) * grad_p_f_exact(x, y, z)[0],
                m_kappa_c0 * m_c1 * p_f_exact(x, y, z)[0],
            ]
        ]
    )
    u_2_val = np.array(
        [
            [
                -m_kappa_c0 * (m_c2 * (y - 0.5) + 1.0) * grad_p_f_exact(x, y, z)[0],
                -m_kappa_c0 * m_c2 * p_f_exact(x, y, z)[0],
            ]
        ]
    )
    return np.where(y < 0.5, u_1_val, u_2_val)


def f_c0_rhs(x, y, z, m_c1, m_c2, m_kappa_c0, m_kappa_c1, m_delta):
    f_1 = np.array(
        [4 * (np.pi**2) * m_kappa_c0 * (m_c1 * (0.5 - y) + 1.0) * p_f_exact(x, y, z)]
    )
    f_2 = np.array(
        [4 * (np.pi**2) * m_kappa_c0 * (m_c2 * (y - 0.5) + 1.0) * p_f_exact(x, y, z)]
    )
    f_c0_val = np.where(y < 0.5, f_1, f_2)
    return f_c0_val


def f_c1_rhs(x, y, z, m_c1, m_c2, m_kappa_c0, m_kappa_c1, m_delta):
    return np.array(
        [
            4 * (np.pi**2) * m_kappa_c1 * m_delta * np.sin(2 * np.pi * x)
            + m_kappa_c0 * (m_c2 + m_c1) * p_f_exact(x, y, z)
        ]
    )


def get_exact_functions_by_co_dimension(
    co_dimension, flux_name, potential_name, m_c1, m_c2, m_kappa_c0, m_kappa_c1, m_delta
):
    if co_dimension == 0:
        exact_functions = {
            flux_name: partial(
                u_exact,
                m_c1=m_c1,
                m_c2=m_c2,
                m_kappa_c0=m_kappa_c0,
                m_kappa_c1=m_kappa_c1,
                m_delta=m_delta,
            ),
            potential_name: partial(
                p_exact,
                m_c1=m_c1,
                m_c2=m_c2,
                m_kappa_c0=m_kappa_c0,
                m_kappa_c1=m_kappa_c1,
                m_delta=m_delta,
            ),
        }
    elif co_dimension == 1:
        exact_functions = {
            flux_name: partial(
                u_f_exact,
                m_c1=m_c1,
                m_c2=m_c2,
                m_kappa_c0=m_kappa_c0,
                m_kappa_c1=m_kappa_c1,
                m_delta=m_delta,
            ),
            potential_name: p_f_exact,
        }
    else:
        raise ValueError("Case not available.")

    return exact_functions


def get_rhs_by_co_dimension(
    co_dimension, rhs_name, m_c1, m_c2, m_kappa_c0, m_kappa_c1, m_delta
):
    if co_dimension == 0:
        rhs_functions = {
            rhs_name: partial(
                f_c0_rhs,
                m_c1=m_c1,
                m_c2=m_c2,
                m_kappa_c0=m_kappa_c0,
                m_kappa_c1=m_kappa_c1,
                m_delta=m_delta,
            ),
        }
    elif co_dimension == 1:
        rhs_functions = {
            rhs_name: partial(
                f_c1_rhs,
                m_c1=m_c1,
                m_c2=m_c2,
                m_kappa_c0=m_kappa_c0,
                m_kappa_c1=m_kappa_c1,
                m_delta=m_delta,
            ),
        }
    else:
        raise ValueError("Case not available.")

    return rhs_functions
