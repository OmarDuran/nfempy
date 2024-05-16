import numpy as np


def f_porosity(x, y, z, m_par, dim):
    if dim == 1:
        val = np.array(x**2)
    elif dim == 2:
        gamma = m_par
        mask = np.logical_or(x <= -3 / 4, y <= -3 / 4)
        val = np.empty_like(x)
        val[~mask] = ((0.75 + x[~mask]) ** gamma) * ((0.75 + y[~mask]) ** (2 * gamma))
        val[mask] = np.zeros_like(x[mask])
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")
    return val


def f_grad_porosity(x, y, z, m_par, dim):
    if dim == 1:
        val = np.array([2 * x, y * 0.0, z * 0.0])
    elif dim == 2:
        gamma = m_par
        mask = np.logical_or(x <= -3 / 4, y <= -3 / 4)
        val = np.empty_like([x * 0.0, y * 0.0])
        val[:, mask] = np.array([x[mask] * 0.0, y[mask] * 0.0])
        val[:, ~mask] = np.array(
            [
                ((0.75 + x[~mask]) ** (-1 + gamma))
                * ((0.75 + y[~mask]) ** (2 * gamma))
                * gamma,
                2
                * ((0.75 + x[~mask]) ** gamma)
                * ((0.75 + y[~mask]) ** (-1 + 2 * gamma))
                * gamma,
            ]
        )
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")
    return val


def f_kappa(x, y, z, m_par, dim):
    return f_porosity(x, y, z, m_par, dim) ** 2


def f_d_phi(x, y, z, m_par, m_mu, dim):
    return np.sqrt(f_kappa(x, y, z, m_par, dim) / m_mu)


def f_grad_d_phi(x, y, z, m_par, m_mu, dim):
    if dim == 1:
        mask = x < 0.0
    elif dim == 2:
        mask = np.logical_or(x <= -3 / 4, y <= -3 / 4)
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")
    scalar_part = np.empty_like(x)
    scalar_part[mask] = np.zeros_like(x[mask])
    scalar_part[~mask] = f_porosity(x[~mask], y[~mask], z[~mask], m_par, dim) / (
        m_mu * f_d_phi(x[~mask], y[~mask], z[~mask], m_par, m_mu, dim)
    )
    vector_part = f_grad_porosity(x, y, z, m_par, dim)
    return scalar_part * vector_part


def p_exact(x, y, z, m_par, dim):
    if dim == 1:
        beta = m_par
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
        return np.where(
            np.logical_or(x <= -3 / 4, y <= -3 / 4),
            np.array([np.zeros_like(x)]),
            np.array([np.cos(6 * x * (y**2))]),
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
        return np.where(
            np.logical_or(x <= -3 / 4, y <= -3 / 4),
            np.array([[x * 0.0, y * 0.0]]),
            np.array(
                [
                    [
                        6
                        * ((0.75 + x) ** (2 * gamma))
                        * (y**2)
                        * ((0.75 + y) ** (4 * gamma))
                        * np.sin(6 * x * (y**2)),
                        12
                        * x
                        * ((0.75 + x) ** (2 * gamma))
                        * y
                        * ((0.75 + y) ** (4 * gamma))
                        * np.sin(6 * x * (y**2)),
                    ]
                ]
            ),
        )
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")


def q_exact(x, y, z, m_par, dim):
    if dim == 1:
        mask = x < 0.0
        val = np.empty_like([x * 0.0])
        val[:, mask] = np.zeros_like([x[mask] * 0.0])
        val[:, ~mask] = np.sqrt(
            f_porosity(x[~mask], y[~mask], z[~mask], m_par, dim)
        ) * p_exact(x[~mask], y[~mask], z[~mask], m_par, dim)
    elif dim == 2:
        gamma = m_par
        mask = np.logical_or(x <= -3 / 4, y <= -3 / 4)
        val = np.empty_like([x * 0.0])
        val[:, mask] = np.zeros_like([x[mask] * 0.0])
        val[:, ~mask] = np.array(
            [
                np.sqrt(
                    ((0.75 + x[~mask]) ** gamma) * ((0.75 + y[~mask]) ** (2 * gamma))
                )
                * np.cos(6 * x[~mask] * (y[~mask] ** 2))
            ]
        )
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")
    return val


def v_exact(x, y, z, m_par, m_mu, dim):
    if dim == 1:
        mask = x < 0.0
        val = np.empty_like([x * 0.0])
        val[:, mask] = np.zeros_like([x[mask] * 0.0])
        val[:, ~mask] = u_exact(x[~mask], y[~mask], z[~mask], m_par, dim) / f_d_phi(
            x[~mask], y[~mask], z[~mask], m_par, m_mu, dim
        )
    elif dim == 2:
        gamma = m_par
        mask = np.logical_or(x <= -3 / 4, y <= -3 / 4)
        val = np.empty_like([[x * 0.0, y * 0.0]])
        val[:, :, mask] = np.zeros_like([[x[mask] * 0.0, y[mask] * 0.0]])
        val[:, :, ~mask] = np.array(
            [
                [
                    6
                    * (y[~mask] ** 2)
                    * np.sqrt(
                        ((0.75 + x[~mask]) ** (2 * gamma))
                        * ((0.75 + y[~mask]) ** (4 * gamma))
                    )
                    * np.sin(6 * x[~mask] * (y[~mask] ** 2)),
                    12
                    * x[~mask]
                    * y[~mask]
                    * np.sqrt(
                        ((0.75 + x[~mask]) ** (2 * gamma))
                        * ((0.75 + y[~mask]) ** (4 * gamma))
                    )
                    * np.sin(6 * x[~mask] * (y[~mask] ** 2)),
                ]
            ]
        )
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")
    return val


def f_rhs(x, y, z, m_par, dim):
    if dim == 1:
        beta = m_par
        mask = x < 0.0
        val = np.empty_like([[x * 0.0]])
        val[:, :, mask] = np.zeros_like([x[mask] * 0.0])
        val[:, :, ~mask] = np.array(
            [
                [
                    (x[~mask] ** beta)
                    * np.sqrt(f_porosity(x[~mask], y[~mask], z[~mask], m_par, dim))
                ]
            ]
        )
    elif dim == 2:
        gamma = m_par
        mask = np.logical_or(x <= -3 / 4, y <= -3 / 4)
        val = np.empty_like([[x * 0.0]])
        val[:, :, mask] = np.zeros_like([x[mask] * 0.0])

        val[:, :, ~mask] = np.array(
            [
                [
                    np.sqrt(f_porosity(x[~mask], y[~mask], z[~mask], m_par, dim))
                    * (
                        (
                            1
                            + 36
                            * ((0.75 + x[~mask]) ** gamma)
                            * (y[~mask] ** 2)
                            * ((0.75 + y[~mask]) ** (2 * gamma))
                            * (4 * (x[~mask] ** 2) + (y[~mask] ** 2))
                        )
                        * np.cos(6 * x[~mask] * (y[~mask] ** 2))
                        + (
                            12
                            * ((0.75 + x[~mask]) ** gamma)
                            * ((0.75 + y[~mask]) ** (2 * gamma))
                            * (
                                x[~mask] * (3 + 4 * x[~mask]) * (3 + 4 * y[~mask])
                                + 4
                                * y[~mask]
                                * (
                                    4 * x[~mask] * (3 + 4 * x[~mask])
                                    + y[~mask] * (3 + 4 * y[~mask])
                                )
                                * gamma
                            )
                            * np.sin(6 * x[~mask] * (y[~mask] ** 2))
                        )
                        / ((3 + 4 * x[~mask]) * (3 + 4 * y[~mask]))
                    )
                ]
            ]
        )
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")
    return val


def test_degeneracy(m_par, m_mu, dim):
    x = np.random.uniform(-1.0, +1.0, (10, 3))
    try:
        phi = f_porosity(x[:, 0], x[:, 1], x[:, 2], m_par, dim)
        grad_phi = f_grad_porosity(x[:, 0], x[:, 1], x[:, 2], m_par, dim)
        d_phi = f_d_phi(x[:, 0], x[:, 1], x[:, 2], m_par, m_mu, dim)
        grad_d_phi = f_grad_d_phi(x[:, 0], x[:, 1], x[:, 2], m_par, m_mu, dim)
        kappa = f_kappa(x[:, 0], x[:, 1], x[:, 2], m_par, dim)

        # exact functions
        u = u_exact(x[:, 0], x[:, 1], x[:, 2], m_par, dim)
        p = p_exact(x[:, 0], x[:, 1], x[:, 2], m_par, dim)
        v = v_exact(x[:, 0], x[:, 1], x[:, 2], m_par, m_mu, dim)
        q = q_exact(x[:, 0], x[:, 1], x[:, 2], m_par, dim)
        rhs = f_rhs(x[:, 0], x[:, 1], x[:, 2], m_par, dim)
    except Exception:
        return False

    return True
