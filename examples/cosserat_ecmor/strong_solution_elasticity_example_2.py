import numpy as np


def chi(x, y, z, dim):
    if dim == 2:
        return np.where(
            np.min(np.array([x, y]), axis=0) > 0.5, np.ones_like(x), np.zeros_like(x)
        )
    else:
        raise ValueError("Dimension not implemented")


def xi(x, y, z, m_kappa, dim):
    return (1.0 - chi(x, y, z, dim)) + m_kappa * chi(x, y, z, dim)


def grad_xi(x, y, z, dim):
    if dim == 2:
        return np.zeros_like(np.array([x, y]))
    else:
        return np.zeros_like(np.array([x, y, z]))


def displacement(m_lambda, m_mu, m_kappa, dim):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                ((1 - y) * y * np.sin(2 * np.pi * x)) / xi(x, y, z, m_kappa, dim),
                ((1 - x) * x * np.sin(2 * np.pi * y)) / xi(x, y, z, m_kappa, dim),
            ]
        )
    else:
        raise ValueError("Dimension not implemented")


def rotation(m_lambda, m_mu, m_kappa, dim):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                -(((1 - y) * np.sin(2 * np.pi * x)) / xi(x, y, z, m_kappa, dim))
                + (y * np.sin(2 * np.pi * x)) / xi(x, y, z, m_kappa, dim)
                + ((1 - x) * np.sin(2 * np.pi * y)) / xi(x, y, z, m_kappa, dim)
                - (x * np.sin(2 * np.pi * y)) / xi(x, y, z, m_kappa, dim),
            ]
        )
    else:
        raise ValueError("Dimension not implemented")


def stress(m_lambda, m_mu, m_kappa, dim):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                [
                    (
                        -2
                        * np.pi
                        * (-1 + y)
                        * y
                        * (m_lambda + 2 * m_mu)
                        * np.cos(2 * np.pi * x)
                        - 2 * np.pi * (-1 + x) * x * m_lambda * np.cos(2 * np.pi * y)
                    )
                    / xi(x, y, z, m_kappa, dim),
                    (
                        -2
                        * m_mu
                        * (
                            (-2 + 4 * y) * np.sin(2 * np.pi * x)
                            + (1 - 2 * x) * np.sin(2 * np.pi * y)
                        )
                    )
                    / xi(x, y, z, m_kappa, dim),
                ],
                [
                    (
                        2
                        * m_mu
                        * (
                            (-1 + 2 * y) * np.sin(2 * np.pi * x)
                            + 2 * (1 - 2 * x) * np.sin(2 * np.pi * y)
                        )
                    )
                    / xi(x, y, z, m_kappa, dim),
                    (
                        -2 * np.pi * (-1 + y) * y * m_lambda * np.cos(2 * np.pi * x)
                        - 2
                        * np.pi
                        * (-1 + x)
                        * x
                        * (m_lambda + 2 * m_mu)
                        * np.cos(2 * np.pi * y)
                    )
                    / xi(x, y, z, m_kappa, dim),
                ],
            ]
        )
    else:
        raise ValueError("Dimension not implemented")


def rhs(m_lambda, m_mu, m_kappa, dim):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                (
                    -2
                    * np.pi
                    * (-1 + 2 * x)
                    * (m_lambda - 2 * m_mu)
                    * np.cos(2 * np.pi * y)
                    + 4
                    * (-2 * m_mu + (np.pi**2) * (-1 + y) * y * (m_lambda + 2 * m_mu))
                    * np.sin(2 * np.pi * x)
                )
                / xi(x, y, z, m_kappa, dim),
                (
                    -2
                    * np.pi
                    * (-1 + 2 * y)
                    * (m_lambda - 2 * m_mu)
                    * np.cos(2 * np.pi * x)
                    + 4
                    * (-2 * m_mu + (np.pi**2) * (-1 + x) * x * (m_lambda + 2 * m_mu))
                    * np.sin(2 * np.pi * y)
                )
                / xi(x, y, z, m_kappa, dim),
                (
                    -6
                    * m_mu
                    * (
                        (-1 + 2 * y) * np.sin(2 * np.pi * x)
                        + (1 - 2 * x) * np.sin(2 * np.pi * y)
                    )
                )
                / xi(x, y, z, m_kappa, dim),
            ]
        )
    else:
        raise ValueError("Dimension not implemented")


def stress_divergence(m_lambda, m_mu, m_kappa, dim):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                (
                    -2
                    * np.pi
                    * (-1 + 2 * x)
                    * (m_lambda - 2 * m_mu)
                    * np.cos(2 * np.pi * y)
                    + 4
                    * (-2 * m_mu + (np.pi**2) * (-1 + y) * y * (m_lambda + 2 * m_mu))
                    * np.sin(2 * np.pi * x)
                )
                / xi(x, y, z, m_kappa, dim),
                (
                    -2
                    * np.pi
                    * (-1 + 2 * y)
                    * (m_lambda - 2 * m_mu)
                    * np.cos(2 * np.pi * x)
                    + 4
                    * (-2 * m_mu + (np.pi**2) * (-1 + x) * x * (m_lambda + 2 * m_mu))
                    * np.sin(2 * np.pi * y)
                )
                / xi(x, y, z, m_kappa, dim),
            ]
        )
    else:
        raise ValueError("Dimension not implemented")
