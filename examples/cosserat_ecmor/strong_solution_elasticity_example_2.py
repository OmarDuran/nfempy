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
                ((0.5 - y) * (1 - y) * y * np.sin(2 * np.pi * x))
                / xi(x, y, z, m_kappa, dim),
                ((0.5 - x) * (1 - x) * x * np.sin(2 * np.pi * y))
                / xi(x, y, z, m_kappa, dim),
            ]
        )
    else:
        raise ValueError("Dimension not implemented")


def rotation(m_lambda, m_mu, m_kappa, dim):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                (
                    (1 + 6 * (-1 + y) * y) * np.sin(2 * np.pi * x)
                    + (-1 - 6 * (-1 + x) * x) * np.sin(2 * np.pi * y)
                )
                / (4.0 * xi(x, y, z, m_kappa, dim)),
            ]
        )
    else:
        raise ValueError("Dimension not implemented")


def stress(m_lambda, m_mu, m_kappa, dim):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                [
                    3 * np.pi * (-1 + y) * y * (-1 + 2 * y) * np.cos(2 * np.pi * x)
                    + np.pi * (-1 + x) * x * (-1 + 2 * x) * np.cos(2 * np.pi * y),
                    (
                        (1 + 6 * (-1 + y) * y) * np.sin(2 * np.pi * x)
                        + (1 + 6 * (-1 + x) * x) * np.sin(2 * np.pi * y)
                    )
                    / 2.0,
                ],
                [
                    (
                        (1 + 6 * (-1 + y) * y) * np.sin(2 * np.pi * x)
                        + (1 + 6 * (-1 + x) * x) * np.sin(2 * np.pi * y)
                    )
                    / 2.0,
                    np.pi * (-1 + y) * y * (-1 + 2 * y) * np.cos(2 * np.pi * x)
                    + 3 * np.pi * (-1 + x) * x * (-1 + 2 * x) * np.cos(2 * np.pi * y),
                ],
            ]
        )
    else:
        raise ValueError("Dimension not implemented")


def rhs(m_lambda, m_mu, m_kappa, dim):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                2 * np.pi * (1 - 6 * x + 6 * (x**2)) * np.cos(2 * np.pi * y)
                - 3
                * (-1 + 2 * y)
                * (-1 + 2 * (np.pi**2) * (-1 + y) * y)
                * np.sin(2 * np.pi * x),
                2 * np.pi * (1 - 6 * y + 6 * (y**2)) * np.cos(2 * np.pi * x)
                - 3
                * (-1 + 2 * x)
                * (-1 + 2 * (np.pi**2) * (-1 + x) * x)
                * np.sin(2 * np.pi * y),
                np.zeros_like(x),
            ]
        )
    else:
        raise ValueError("Dimension not implemented")


def stress_divergence(m_lambda, m_mu, m_kappa, dim):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                2 * np.pi * (1 - 6 * x + 6 * (x**2)) * np.cos(2 * np.pi * y)
                - 3
                * (-1 + 2 * y)
                * (-1 + 2 * (np.pi**2) * (-1 + y) * y)
                * np.sin(2 * np.pi * x),
                2 * np.pi * (1 - 6 * y + 6 * (y**2)) * np.cos(2 * np.pi * x)
                - 3
                * (-1 + 2 * x)
                * (-1 + 2 * (np.pi**2) * (-1 + x) * x)
                * np.sin(2 * np.pi * y),
            ]
        )
    else:
        raise ValueError("Dimension not implemented")
