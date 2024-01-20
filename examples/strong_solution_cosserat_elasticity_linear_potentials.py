import numpy as np


def displacement(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                x,
                y,
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                y,
                z,
                x,
            ]
        )


def displacement_gradient(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                [
                    np.ones_like(x),
                    np.zeros_like(x),
                ],
                [
                    np.zeros_like(x),
                    np.ones_like(x),
                ],
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                [
                    np.zeros_like(x),
                    np.ones_like(x),
                    np.zeros_like(x),
                ],
                [
                    np.zeros_like(x),
                    np.zeros_like(x),
                    np.ones_like(x),
                ],
                [
                    np.ones_like(x),
                    np.zeros_like(x),
                    np.zeros_like(x),
                ],
            ]
        )


def rotation(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                x + y,
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                z,
                x,
                y,
            ]
        )


def rotation_gradient(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                np.ones_like(x),
                np.ones_like(x),
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                [
                    np.zeros_like(x),
                    np.zeros_like(x),
                    np.ones_like(x),
                ],
                [
                    np.ones_like(x),
                    np.zeros_like(x),
                    np.zeros_like(x),
                ],
                [
                    np.zeros_like(x),
                    np.ones_like(x),
                    np.zeros_like(x),
                ],
            ]
        )


def stress(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                [
                    2 * (m_lambda + m_mu) * np.ones_like(x),
                    -2 * (x + y) * m_kappa * np.ones_like(x),
                ],
                [
                    2 * (x + y) * m_kappa * np.ones_like(x),
                    2 * (m_lambda + m_mu) * np.ones_like(x),
                ],
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                [
                    np.zeros_like(x),
                    m_kappa - 2 * y * m_kappa + m_mu,
                    (-1 + 2 * x) * m_kappa + m_mu,
                ],
                [
                    (-1 + 2 * y) * m_kappa + m_mu,
                    np.zeros_like(x),
                    m_kappa - 2 * z * m_kappa + m_mu,
                ],
                [
                    m_kappa - 2 * x * m_kappa + m_mu,
                    (-1 + 2 * z) * m_kappa + m_mu,
                    np.zeros_like(x),
                ],
            ]
        )


def couple_stress(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                [
                    m_gamma * np.ones_like(x),
                    m_gamma * np.ones_like(x),
                ],
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                [
                    np.zeros_like(x),
                    np.zeros_like(x),
                    m_gamma * np.ones_like(x),
                ],
                [
                    m_gamma * np.ones_like(x),
                    np.zeros_like(x),
                    np.zeros_like(x),
                ],
                [
                    np.zeros_like(x),
                    m_gamma * np.ones_like(x),
                    np.zeros_like(x),
                ],
            ]
        )


def couple_stress_scaled(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.sqrt(m_gamma) * np.array(
            [
                np.ones_like(x),
                np.ones_like(x),
            ]
        )
    else:
        return lambda x, y, z: np.sqrt(m_gamma) * np.array(
            [
                [
                    np.zeros_like(x),
                    np.zeros_like(x),
                    np.ones_like(x),
                ],
                [
                    np.ones_like(x),
                    np.zeros_like(x),
                    np.zeros_like(x),
                ],
                [
                    np.zeros_like(x),
                    np.ones_like(x),
                    np.zeros_like(x),
                ],
            ]
        )


def rhs(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                -2 * m_kappa * np.ones_like(x),
                2 * m_kappa * np.ones_like(x),
                -4 * (x + y) * m_kappa * np.ones_like(x),
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                -2 * m_kappa * np.ones_like(x),
                -2 * m_kappa * np.ones_like(x),
                -2 * m_kappa * np.ones_like(x),
                (2 - 4 * z) * m_kappa * np.ones_like(x),
                (2 - 4 * x) * m_kappa * np.ones_like(x),
                (2 - 4 * y) * m_kappa * np.ones_like(x),
            ]
        )


def rhs_scaled(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                -2 * m_kappa * np.ones_like(x),
                2 * m_kappa * np.ones_like(x),
                -4 * (x + y) * m_kappa * np.ones_like(x),
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                -2 * m_kappa * np.ones_like(x),
                -2 * m_kappa * np.ones_like(x),
                -2 * m_kappa * np.ones_like(x),
                (2 - 4 * z) * m_kappa * np.ones_like(x),
                (2 - 4 * x) * m_kappa * np.ones_like(x),
                (2 - 4 * y) * m_kappa * np.ones_like(x),
            ]
        )


def stress_divergence(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                -2 * m_kappa,
                2 * m_kappa,
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                -2 * m_kappa,
                -2 * m_kappa,
                -2 * m_kappa,
            ]
        )


def couple_stress_divergence(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                np.zeros_like(x),
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                np.zeros_like(x),
                np.zeros_like(x),
                np.zeros_like(x),
            ]
        )


def couple_stress_divergence_scaled(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                np.zeros_like(x),
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                np.zeros_like(x),
                np.zeros_like(x),
                np.zeros_like(x),
            ]
        )
