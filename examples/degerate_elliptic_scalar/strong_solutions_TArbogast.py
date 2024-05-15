import numpy as np

def f_porosity(x, y, z, m_par, dim):
    if dim == 1:
        return np.array(x ** 2)
    elif dim == 2:
        gamma = m_par
        return np.where(np.logical_or(x <= -3/4, y <= -3/4),
                np.zeros_like(x),
                ((0.75 + x)**gamma)*((0.75 + y)**(2*gamma)),
            )
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")

def f_grad_porosity(x, y, z, m_par, dim):
    if dim == 1:
        return np.array([2 * x, y * 0.0, z * 0.0])
    elif dim == 2:
        gamma = m_par
        return np.where(np.logical_or(x <= -3 / 4, y <= -3 / 4),
                        np.array([x * 0.0, y * 0.0]),
                        np.array(
                            [
                                ((0.75 + x) ** (-1 + gamma)) * (
                                            (0.75 + y) ** (2 * gamma)) * gamma
                                ,
                                2 * ((0.75 + x) ** gamma) * (
                                            (0.75 + y) ** (-1 + 2 * gamma)) * gamma
                            ]
                        ),
                        )
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")

def f_kappa(x, y, z, m_par, dim):
    return f_porosity(x, y, z, m_par, dim) ** 2

def f_d_phi(x, y, z, m_par, m_mu, dim):
    return np.sqrt(f_kappa(x, y, z, m_par, dim) / m_mu)

def f_grad_d_phi(x, y, z, m_par, m_mu, dim):
    if dim == 1:
        scalar_part = f_porosity(x, y, z, m_par, dim) / (m_mu * f_d_phi(x, y, z, m_par, m_mu, dim))
        vector_part = f_grad_porosity(x, y, z, m_par, dim)
        return scalar_part * vector_part
    elif dim == 2:
        scalar_part = np.where(np.logical_or(x <= -3 / 4, y <= -3 / 4), np.zeros_like(x),
                               f_porosity(x, y, z, m_par, dim) / (m_mu * f_d_phi(x, y, z, m_par, m_mu, dim)))
        vector_part = f_grad_porosity(x, y, z, m_par, dim)
        return scalar_part * vector_part
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")

def p_exact(x, y, z, m_par, dim):
    if dim == 1:
        beta  = m_par
        return np.where(
            x < 0.0,
            np.array([x * 0.0]),
            np.array(
                [
                    (
                            -(np.abs(x) ** beta)
                            + 0.5
                            * (3 + np.sqrt(13))
                            * np.abs(x) ** (0.5 * (-3 + np.sqrt(13)))
                            * beta
                    )
                    / (-1 + beta * (3 + beta)),
                ]
            ),
        )
    elif dim == 2:
        return np.where(np.logical_or(x <= -3/4, y <= -3/4),
            np.array([np.zeros_like(x)]),
            np.array([np.cos(6 * x * (y ** 2))]),
        )
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")

def u_exact(x, y, z, m_par, dim):
    if dim == 1:
        beta = m_par
        return np.where(
            x < 0.0,
            np.array([x * 0.0]),
            np.array(
                [
                    (
                        (
                            -np.abs(x) ** (0.5 * (3 + np.sqrt(13)))
                            + np.abs(x) ** (3 + beta)
                        )
                        * beta
                    )
                    / (-1 + beta * (3 + beta)),
                ]
            ),
        )
    elif dim == 2:
        gamma = m_par
        return np.where(np.logical_or(x <= -3 / 4, y <= -3 / 4),
                        np.array([[x * 0.0, y * 0.0]]),
                        np.array(
                            [
                                [
                                    6 * ((0.75 + x) ** (2 * gamma)) * (y ** 2) * (
                                            (0.75 + y) ** (4 * gamma)) * np.sin(
                                        6 * x * (y ** 2))
                                    ,
                                    12 * x * ((0.75 + x) ** (2 * gamma)) * y * (
                                            (0.75 + y) ** (4 * gamma)) * np.sin(
                                        6 * x * (y ** 2))
                                ]
                            ]
                        ),
        )
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")

def q_exact(x, y, z, m_par, dim):
    if dim == 1:
        return np.where(
            x < 0.0,
            np.zeros_like(x),
            np.sqrt(f_porosity(x, y, z, m_par, dim)) * p_exact(x, y, z, m_par, dim),
        )
    elif dim == 2:
        gamma = m_par
        return np.where(np.logical_or(x <= -3 / 4, y <= -3 / 4),
                        np.array([np.zeros_like(x)]),
                        np.array([np.sqrt(
                            ((0.75 + x) ** gamma) * ((0.75 + y) ** (2 * gamma))) * np.cos(
                            6 * x * (y ** 2))]),
                        )
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")

def v_exact(x, y, z, m_par, m_mu, dim):
    if dim == 1:
        return np.where(
            x < 0.0, np.array([x * 0.0]), u_exact(x, y, z, m_par, dim) / f_d_phi(x, y, z, m_par, m_mu, dim)
        )
    elif dim == 2:
        gamma = m_par
        return np.where(np.logical_or(x <= -3 / 4, y <= -3 / 4),
                        np.array([[x * 0.0, y * 0.0]]),
                        np.array(
                            [
                                [
                                    6 * (y ** 2) * np.sqrt(((0.75 + x) ** (2 * gamma)) * (
                                                (0.75 + y) ** (4 * gamma))) * np.sin(
                                        6 * x * (y ** 2))
                                    ,
                                    12 * x * y * np.sqrt(((0.75 + x) ** (2 * gamma)) * (
                                                (0.75 + y) ** (4 * gamma))) * np.sin(
                                        6 * x * (y ** 2))
                                ]
                            ]
                        ),
                        )
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")

def f_rhs(x, y, z, m_par, dim):
    if dim == 1:
        beta = m_par
        return np.where(
            x < 0.0,
            np.array([[x * 0.0]]),
            np.array([[(np.abs(x) ** beta) * np.sqrt(f_porosity(x, y, z, m_par, dim))]]),
        )
    elif dim == 2:
        gamma = m_par
        return np.where(np.logical_or(x <= -3 / 4, y <= -3 / 4),
                        np.array([[x * 0.0]]),
                        np.array([[np.sqrt(f_porosity(x, y, z, m_par, dim)) *
                                   ((1 + 36 * ((0.75 + x) ** gamma) * (y ** 2) * (
                                               (0.75 + y) ** (2 * gamma)) * (
                                             4 * (x ** 2) + (y ** 2))) *
                                    np.cos(6 * x * (y ** 2)) + (
                                            12 * ((0.75 + x) ** gamma) * (
                                                (0.75 + y) ** (2 * gamma)) *
                                            (x * (3 + 4 * x) * (3 + 4 * y) + 4 * y * (
                                                    4 * x * (3 + 4 * x) + y * (
                                                    3 + 4 * y)) * gamma) * np.sin(
                                        6 * x * (y ** 2)))
                                    / ((3 + 4 * x) * (3 + 4 * y)))]]),
                        )
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")

