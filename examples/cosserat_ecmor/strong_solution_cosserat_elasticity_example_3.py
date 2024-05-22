import numpy as np


def gamma_s(dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.min(
            [
                np.ones_like(x),
                np.max(
                    [
                        np.zeros_like(x),
                        np.max(
                            [3 * x - np.ones_like(x), 3 * y - np.ones_like(y)], axis=0
                        ),
                    ],
                    axis=0,
                ),
            ],
            axis=0,
        )
    else:
        raise ValueError("Dimension not implemented")


def grad_gamma_s(dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                np.where(
                    np.logical_or(
                        np.logical_and(x - y >= 0, x >= 2.0 / 3.0),
                        np.logical_and(x - y < 0, y >= 2.0 / 3.0),
                    ),
                    np.zeros_like(x),
                    np.where(
                        np.logical_and(
                            np.logical_and(1.0 / 3.0 < x, x < 2.0 / 3.0), x - y >= 0.0
                        ),
                        3.0 * np.ones_like(x),
                        np.zeros_like(x),
                    ),
                ),
                np.where(
                    np.logical_or.reduce(
                        (
                            np.logical_and(x - y >= 0, x >= 2.0 / 3.0),
                            np.logical_and(x - y < 0.0, y >= 2.0 / 3.0),
                            np.logical_and(
                                np.logical_and(1.0 / 3.0 < x, x < 2.0 / 3.0),
                                x - y >= 0.0,
                            ),
                        )
                    ),
                    np.zeros_like(x),
                    np.where(
                        np.logical_and(
                            np.logical_and(1.0 / 3.0 < y, y < 2.0 / 3.0), x - y < 0.0
                        ),
                        3.0 * np.ones_like(x),
                        np.zeros_like(x),
                    ),
                ),
            ]
        )
    else:
        raise ValueError("Dimension not implemented")


def gamma_eval(x, y, z, dim: int = 2):
    if dim == 2:
        return np.min(
            [
                np.ones_like(x),
                np.max(
                    [
                        np.zeros_like(x),
                        np.max(
                            [3 * x - np.ones_like(x), 3 * y - np.ones_like(y)], axis=0
                        ),
                    ],
                    axis=0,
                ),
            ],
            axis=0,
        )
    else:
        raise ValueError("Dimension not implemented")


def grad_gamma_eval(x, y, z, dim: int = 2):
    if dim == 2:
        return np.array(
            [
                np.where(
                    np.logical_or(
                        np.logical_and(x - y >= 0, x >= 2.0 / 3.0),
                        np.logical_and(x - y < 0, y >= 2.0 / 3.0),
                    ),
                    np.zeros_like(x),
                    np.where(
                        np.logical_and(
                            np.logical_and(1.0 / 3.0 < x, x < 2.0 / 3.0), x - y >= 0.0
                        ),
                        3.0 * np.ones_like(x),
                        np.zeros_like(x),
                    ),
                ),
                np.where(
                    np.logical_or.reduce(
                        (
                            np.logical_and(x - y >= 0, x >= 2.0 / 3.0),
                            np.logical_and(x - y < 0.0, y >= 2.0 / 3.0),
                            np.logical_and(
                                np.logical_and(1.0 / 3.0 < x, x < 2.0 / 3.0),
                                x - y >= 0.0,
                            ),
                        )
                    ),
                    np.zeros_like(x),
                    np.where(
                        np.logical_and(
                            np.logical_and(1.0 / 3.0 < y, y < 2.0 / 3.0), x - y < 0.0
                        ),
                        3.0 * np.ones_like(x),
                        np.zeros_like(x),
                    ),
                ),
            ]
        )
    else:
        raise ValueError("Dimension not implemented")


def displacement(m_lambda, m_mu, m_kappa, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                (1 - y) * y * np.sin(2 * np.pi * x),
                (1 - x) * x * np.sin(2 * np.pi * y),
            ]
        )
    else:
        raise ValueError("Dimension not implemented")


def rotation(m_lambda, m_mu, m_kappa, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                (1 - x) * x * (1 - y) * y,
            ]
        )
    else:
        raise ValueError("Dimension not implemented")


def stress(m_lambda, m_mu, m_kappa, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                [
                    -2
                    * np.pi
                    * (-1 + y)
                    * y
                    * (m_lambda + 2 * m_mu)
                    * np.cos(2 * np.pi * x)
                    - 2 * np.pi * (-1 + x) * x * m_lambda * np.cos(2 * np.pi * y),
                    2 * x * y * (-1 + x + y - x * y) * m_kappa
                    - (-1 + 2 * y) * (m_kappa + m_mu) * np.sin(2 * np.pi * x)
                    + (-1 + 2 * x) * (m_kappa - m_mu) * np.sin(2 * np.pi * y),
                ],
                [
                    2 * (-1 + x) * x * (-1 + y) * y * m_kappa
                    + (-1 + 2 * y) * (m_kappa - m_mu) * np.sin(2 * np.pi * x)
                    - (-1 + 2 * x) * (m_kappa + m_mu) * np.sin(2 * np.pi * y),
                    -2 * np.pi * (-1 + y) * y * m_lambda * np.cos(2 * np.pi * x)
                    - 2
                    * np.pi
                    * (-1 + x)
                    * x
                    * (m_lambda + 2 * m_mu)
                    * np.cos(2 * np.pi * y),
                ],
            ]
        )
    else:
        raise ValueError("Dimension not implemented")


def couple_stress_scaled(m_lambda, m_mu, m_kappa, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                [
                    ((1 - x) * (1 - y) * y - x * (1 - y) * y)
                    * gamma_eval(x, y, z, dim),
                    ((1 - x) * x * (1 - y) - (1 - x) * x * y)
                    * gamma_eval(x, y, z, dim),
                ]
            ]
        )
    else:
        raise ValueError("Dimension not implemented")


def rhs_scaled(m_lambda, m_mu, m_kappa, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                2
                * (
                    x * (-1 + x + 2 * y - 2 * x * y) * m_kappa
                    + np.pi
                    * (-1 + 2 * x)
                    * (m_kappa - m_lambda - m_mu)
                    * np.cos(2 * np.pi * y)
                    - (
                        m_kappa
                        + m_mu
                        - 2 * (np.pi**2) * (-1 + y) * y * (m_lambda + 2 * m_mu)
                    )
                    * np.sin(2 * np.pi * x)
                ),
                2
                * (
                    (-1 + 2 * x) * (-1 + y) * y * m_kappa
                    + np.pi
                    * (-1 + 2 * y)
                    * (m_kappa - m_lambda - m_mu)
                    * np.cos(2 * np.pi * x)
                    - (
                        m_kappa
                        + m_mu
                        - 2 * (np.pi**2) * (-1 + x) * x * (m_lambda + 2 * m_mu)
                    )
                    * np.sin(2 * np.pi * y)
                ),
                2
                * (
                    m_kappa
                    * (
                        2 * x * y * (-1 + x + y - x * y)
                        + (1 - 2 * y) * np.sin(2 * np.pi * x)
                        + (-1 + 2 * x) * np.sin(2 * np.pi * y)
                    )
                    + (-x + (x**2) + (-1 + y) * y) * (gamma_eval(x, y, z, dim) ** 2)
                    + gamma_eval(x, y, z, dim)
                    * (
                        (-1 + x) * x * (-1 + 2 * y) * grad_gamma_eval(x, y, z, dim)[1]
                        + (-1 + 2 * x) * (-1 + y) * y * grad_gamma_eval(x, y, z, dim)[0]
                    )
                ),
            ]
        )
    else:
        raise ValueError("Dimension not implemented")


def stress_divergence(m_lambda, m_mu, m_kappa, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                2
                * (
                    x * (-1 + x + 2 * y - 2 * x * y) * m_kappa
                    + np.pi
                    * (-1 + 2 * x)
                    * (m_kappa - m_lambda - m_mu)
                    * np.cos(2 * np.pi * y)
                    - (
                        m_kappa
                        + m_mu
                        - 2 * (np.pi**2) * (-1 + y) * y * (m_lambda + 2 * m_mu)
                    )
                    * np.sin(2 * np.pi * x)
                ),
                2
                * (
                    (-1 + 2 * x) * (-1 + y) * y * m_kappa
                    + np.pi
                    * (-1 + 2 * y)
                    * (m_kappa - m_lambda - m_mu)
                    * np.cos(2 * np.pi * x)
                    - (
                        m_kappa
                        + m_mu
                        - 2 * (np.pi**2) * (-1 + x) * x * (m_lambda + 2 * m_mu)
                    )
                    * np.sin(2 * np.pi * y)
                ),
            ]
        )
    else:
        raise ValueError("Dimension not implemented")


def couple_stress_divergence_scaled(m_lambda, m_mu, m_kappa, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                2
                * gamma_eval(x, y, z, dim)
                * (
                    (-1 + x) * x * gamma_eval(x, y, z, dim)
                    + (-1 + y) * y * gamma_eval(x, y, z, dim)
                    + (-1 + x) * x * (-1 + 2 * y) * grad_gamma_eval(x, y, z, dim)[1]
                    + (-1 + 2 * x) * (-1 + y) * y * grad_gamma_eval(x, y, z, dim)[0]
                ),
            ]
        )
    else:
        raise ValueError("Dimension not implemented")
