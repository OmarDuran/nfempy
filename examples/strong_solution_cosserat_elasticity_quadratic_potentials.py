import numpy as np


def displacement(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                (x**2) + y,
                x + (y**2),
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                x + (y**2) + z,
                x + y + (z**2),
                (x**2) + y + z,
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
                    np.ones_like(x),
                    2 * y,
                    np.ones_like(x),
                ],
                [
                    np.ones_like(x),
                    np.ones_like(x),
                    2 * z,
                ],
                [
                    2 * x,
                    np.ones_like(x),
                    np.ones_like(x),
                ],
            ]
        )


def rotation(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                (x**2) + (y**2),
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                x + y + (z**2),
                (x**2) + y + z,
                x + (y**2) + z,
            ]
        )


def rotation_gradient(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                2 * x,
                2 * y,
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                [
                    np.ones_like(x),
                    np.ones_like(x),
                    2 * z,
                ],
                [
                    2 * x,
                    np.ones_like(x),
                    np.ones_like(x),
                ],
                [
                    np.ones_like(x),
                    2 * y,
                    np.ones_like(x),
                ],
            ]
        )


def stress(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                [
                    2 * (y * m_lambda + x * (m_lambda + 2 * m_mu)),
                    -2 * ((x**2) * m_kappa + (y**2) * m_kappa - m_mu),
                ],
                [
                    2 * ((x**2) * m_kappa + (y**2) * m_kappa + m_mu),
                    2 * (x * m_lambda + y * (m_lambda + 2 * m_mu)),
                ],
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                [
                    3 * m_lambda + 2 * m_mu,
                    -((1 + 2 * x - 2 * y + 2 * (y**2) + 2 * z) * m_kappa)
                    + m_mu
                    + 2 * y * m_mu,
                    (1 - 2 * x + 2 * (x**2) + 2 * y + 2 * z) * m_kappa
                    + m_mu
                    + 2 * x * m_mu,
                ],
                [
                    (1 + 2 * x - 2 * y + 2 * (y**2) + 2 * z) * m_kappa
                    + m_mu
                    + 2 * y * m_mu,
                    3 * m_lambda + 2 * m_mu,
                    -((1 + 2 * x + 2 * y - 2 * z + 2 * (z**2)) * m_kappa)
                    + m_mu
                    + 2 * z * m_mu,
                ],
                [
                    -((1 - 2 * x + 2 * (x**2) + 2 * y + 2 * z) * m_kappa)
                    + m_mu
                    + 2 * x * m_mu,
                    (1 + 2 * x + 2 * y - 2 * z + 2 * (z**2)) * m_kappa
                    + m_mu
                    + 2 * z * m_mu,
                    3 * m_lambda + 2 * m_mu,
                ],
            ]
        )


def couple_stress(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                [
                    2 * x * m_gamma,
                    2 * y * m_gamma,
                ],
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                [
                    m_gamma * np.ones_like(x),
                    m_gamma * np.ones_like(x),
                    2 * z * m_gamma,
                ],
                [
                    2 * x * m_gamma,
                    m_gamma * np.ones_like(x),
                    m_gamma * np.ones_like(x),
                ],
                [
                    m_gamma * np.ones_like(x),
                    2 * y * m_gamma,
                    m_gamma * np.ones_like(x),
                ],
            ]
        )

def couple_stress_scaled(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.sqrt(m_gamma) * np.array(
            [
                2 * x,
                2 * y,
            ]
        )
    else:
        return lambda x, y, z: np.sqrt(m_gamma) * np.array(
            [
                [
                    np.ones_like(x),
                    np.ones_like(x),
                    2 * z,
                ],
                [
                    2 * x,
                    np.ones_like(x),
                    np.ones_like(x),
                ],
                [
                    np.ones_like(x),
                    2 * y,
                    np.ones_like(x),
                ],
            ]
        )

def rhs(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                -4 * y * m_kappa + 2 * m_lambda + 4 * m_mu,
                2 * (2 * x * m_kappa + m_lambda + 2 * m_mu),
                4 * m_gamma - 4 * ((x**2) + (y**2)) * m_kappa,
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                -4 * (-1 + y) * m_kappa + 2 * m_mu,
                -4 * (-1 + z) * m_kappa + 2 * m_mu,
                -4 * (-1 + x) * m_kappa + 2 * m_mu,
                2 * m_gamma - 2 * (1 + 2 * x + 2 * y - 2 * z + 2 * (z**2)) * m_kappa,
                2 * m_gamma - 2 * (1 - 2 * x + 2 * (x**2) + 2 * y + 2 * z) * m_kappa,
                2 * m_gamma - 2 * (1 + 2 * x - 2 * y + 2 * (y**2) + 2 * z) * m_kappa,
            ]
        )


def stress_divergence(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                -4 * y * m_kappa + 2 * m_lambda + 4 * m_mu,
                2 * (2 * x * m_kappa + m_lambda + 2 * m_mu),
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                -4 * (-1 + y) * m_kappa + 2 * m_mu,
                -4 * (-1 + z) * m_kappa + 2 * m_mu,
                -4 * (-1 + x) * m_kappa + 2 * m_mu,
            ]
        )


def couple_stress_divergence(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                4.0 * m_gamma * np.ones_like(x),
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                2 * m_gamma * np.ones_like(x),
                2 * m_gamma * np.ones_like(x),
                2 * m_gamma * np.ones_like(x),
            ]
        )
