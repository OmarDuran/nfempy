import numpy as np


def displacement(m_lambda, m_mu, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                4
                * np.pi
                * np.cos(2 * np.pi * y)
                * (np.sin(2 * np.pi * x) ** 2)
                * np.sin(2 * np.pi * y),
                -4
                * np.pi
                * np.cos(2 * np.pi * x)
                * np.sin(2 * np.pi * x)
                * (np.sin(2 * np.pi * y) ** 2),
            ]
        )
    else:
        raise ValueError("Dimension not implemented")


def rotation(m_lambda, m_mu, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                (
                    8
                    * (np.pi**2)
                    * (np.cos(2 * np.pi * y) ** 2)
                    * (np.sin(2 * np.pi * x) ** 2)
                    + 8
                    * (np.pi**2)
                    * (np.cos(2 * np.pi * x) ** 2)
                    * (np.sin(2 * np.pi * y) ** 2)
                    - 16
                    * (np.pi**2)
                    * (np.sin(2 * np.pi * x) ** 2)
                    * (np.sin(2 * np.pi * y) ** 2)
                )
                / 2.0,
            ]
        )
    else:
        raise ValueError("Dimension not implemented")


def stress(m_lambda, m_mu, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                [
                    8
                    * (np.pi**2)
                    * m_mu
                    * np.sin(4 * np.pi * x)
                    * np.sin(4 * np.pi * y),
                    4
                    * (np.pi**2)
                    * m_mu
                    * (-np.cos(4 * np.pi * x) + np.cos(4 * np.pi * y)),
                ],
                [
                    4
                    * (np.pi**2)
                    * m_mu
                    * (-np.cos(4 * np.pi * x) + np.cos(4 * np.pi * y)),
                    -8
                    * (np.pi**2)
                    * m_mu
                    * np.sin(4 * np.pi * x)
                    * np.sin(4 * np.pi * y),
                ],
            ]
        )
    else:
        raise ValueError("Dimension not implemented")


def rhs(m_lambda, m_mu, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                16
                * (np.pi**3)
                * m_mu
                * (-1 + 2 * np.cos(4 * np.pi * x))
                * np.sin(4 * np.pi * y),
                16
                * (np.pi**3)
                * m_mu
                * (1 - 2 * np.cos(4 * np.pi * y))
                * np.sin(4 * np.pi * x),
                np.zeros_like(x),
            ]
        )
    else:
        raise ValueError("Dimension not implemented")


def stress_divergence(m_lambda, m_mu, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                16
                * (np.pi**3)
                * m_mu
                * (-1 + 2 * np.cos(4 * np.pi * x))
                * np.sin(4 * np.pi * y),
                16
                * (np.pi**3)
                * m_mu
                * (1 - 2 * np.cos(4 * np.pi * y))
                * np.sin(4 * np.pi * x),
            ]
        )
    else:
        raise ValueError("Dimension not implemented")
