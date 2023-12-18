import numpy as np


# def zeta(xi):
#     hs = 0.5
#     hi = 0.1
#     conditions = [
#         xi <= -hs,
#         np.logical_and(-hs < xi, xi < -hi),
#         np.logical_and(-hi < xi, xi < hi),
#         np.logical_and(hi < xi, xi < hs),
#         xi >= hs,
#     ]
#     functions = [
#         lambda xi: np.ones_like(xi),
#         lambda xi: -0.25 - xi / (-hi + hs),
#         lambda xi: np.zeros_like(xi),
#         lambda xi: -0.25 + xi / (-hi + hs),
#         lambda xi: np.ones_like(xi),
#     ]
#     return np.piecewise(xi, conditions, functions)
#
#
# def dzeta(xi):
#     hs = 0.5
#     hi = 0.1
#     conditions = [
#         xi <= -hs,
#         np.logical_and(-hs < xi, xi < -hi),
#         np.logical_and(-hi < xi, xi < hi),
#         np.logical_and(hi < xi, xi < hs),
#         xi >= hs,
#     ]
#     functions = [
#         lambda xi: np.zeros_like(xi),
#         lambda xi: -np.ones_like(xi) / (-hi + hs),
#         lambda xi: np.zeros_like(xi),
#         lambda xi: np.ones_like(xi) / (-hi + hs),
#         lambda xi: np.zeros_like(xi),
#     ]
#     return np.piecewise(xi, conditions, functions)
#
#
# def gamma_s(dim: int = 2):
#     if dim == 2:
#         return lambda x1, x2, x3: zeta(x1) * zeta(x2)
#     else:
#         return lambda x1, x2, x3: zeta(x1) * zeta(x2) * zeta(x3)
#
#
# def grad_gamma_s(dim: int = 2):
#     if dim == 2:
#         return lambda x1, x2, x3: np.array([dzeta(x1) * zeta(x2), zeta(x1) * dzeta(x2)])
#     else:
#         return lambda x1, x2, x3: np.array(
#             [
#                 dzeta(x1) * zeta(x2) * zeta(x3),
#                 zeta(x1) * dzeta(x2) * zeta(x3),
#                 zeta(x1) * zeta(x2) * dzeta(x3),
#             ]
#         )
#
#
# def gamma_eval(x1, x2, x3, dim: int = 2):
#     if dim == 2:
#         return zeta(x1) * zeta(x2)
#     else:
#         return zeta(x1) * zeta(x2) * zeta(x3)
#
#
# def grad_gamma_eval(x1, x2, x3, dim: int = 2):
#     if dim == 2:
#         return np.array([dzeta(x1) * zeta(x2), zeta(x1) * dzeta(x2)])
#     else:
#         return np.array(
#             [
#                 dzeta(x1) * zeta(x2) * zeta(x3),
#                 zeta(x1) * dzeta(x2) * zeta(x3),
#                 zeta(x1) * zeta(x2) * dzeta(x3),
#             ]
#         )

# def gamma_s(dim: int = 2):
#     if dim == 2:
#         return lambda x1, x2, x3: zeta(x1) * zeta(x2)
#     else:
#         return lambda x1, x2, x3: zeta(x1) * zeta(x2) * zeta(x3)
#
#


def gamma_s(dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.min(
            [
                np.ones_like(x),
                np.max(
                    [
                        np.zeros_like(x),
                        np.max([3 * x, 3 * y], axis=0) - np.ones_like(x),
                    ],
                    axis=0,
                ),
            ],
            axis=0,
        )
    else:
        return lambda x, y, z: np.min(
            [
                np.ones_like(x),
                np.max(
                    [
                        np.zeros_like(x),
                        np.max([3 * x, 3 * y, 3 * z], axis=0) - np.ones_like(x),
                    ],
                    axis=0,
                ),
            ],
            axis=0,
        )


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
                    np.logical_or(
                        np.logical_and(x - y >= 0, x >= 2.0 / 3.0),
                        np.logical_or(
                            np.logical_and(x - y < 0, y >= 2.0 / 3.0),
                            np.logical_and(1.0 / 3.0 < x, x < 2.0 / 3.0),
                            x - y >= 0.0,
                        ),
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


def gamma_eval(x, y, z, dim: int = 2):
    if dim == 2:
        return np.min(
            [
                np.ones_like(x),
                np.max(
                    [
                        np.zeros_like(x),
                        np.max([3 * x, 3 * y], axis=0) - np.ones_like(x),
                    ],
                    axis=0,
                ),
            ],
            axis=0,
        )
    else:
        return np.min(
            [
                np.ones_like(x),
                np.max(
                    [
                        np.zeros_like(x),
                        np.max([3 * x, 3 * y, 3 * z], axis=0) - np.ones_like(x),
                    ],
                    axis=0,
                ),
            ],
            axis=0,
        )


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
                    np.logical_or(
                        np.logical_and(x - y >= 0, x >= 2.0 / 3.0),
                        np.logical_or(
                            np.logical_and(x - y < 0, y >= 2.0 / 3.0),
                            np.logical_and(1.0 / 3.0 < x, x < 2.0 / 3.0),
                            x - y >= 0.0,
                        ),
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


def displacement(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                (1 - y) * (1 + y) * np.sin(np.pi * x),
                (1 - x) * (1 + x) * np.sin(np.pi * y),
            ]
        )
    else:
        assert False
        return lambda x, y, z: np.array(
            [
                (1 - y) * y * (1 - z) * z * np.sin(np.pi * x),
                (1 - x) * x * (1 - z) * z * np.sin(np.pi * y),
                (1 - x) * x * (1 - y) * y * np.sin(np.pi * z),
            ]
        )


def displacement_gradient(m_lambda, m_mu, m_kappa, f_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                [
                    np.pi * (1 - y) * (1 + y) * np.cos(np.pi * x),
                    (1 - y) * np.sin(np.pi * x) - (1 + y) * np.sin(np.pi * x),
                ],
                [
                    (1 - x) * np.sin(np.pi * y) - (1 + x) * np.sin(np.pi * y),
                    np.pi * (1 - x) * (1 + x) * np.cos(np.pi * y),
                ],
            ]
        )
    else:
        assert False
        return lambda x, y, z: np.array(
            [
                [
                    np.pi * (1 - y) * y * (1 - z) * z * np.cos(np.pi * x),
                    (1 - y) * (1 - z) * z * np.sin(np.pi * x)
                    - y * (1 - z) * z * np.sin(np.pi * x),
                    (1 - y) * y * (1 - z) * np.sin(np.pi * x)
                    - (1 - y) * y * z * np.sin(np.pi * x),
                ],
                [
                    (1 - x) * (1 - z) * z * np.sin(np.pi * y)
                    - x * (1 - z) * z * np.sin(np.pi * y),
                    np.pi * (1 - x) * x * (1 - z) * z * np.cos(np.pi * y),
                    (1 - x) * x * (1 - z) * np.sin(np.pi * y)
                    - (1 - x) * x * z * np.sin(np.pi * y),
                ],
                [
                    (1 - x) * (1 - y) * y * np.sin(np.pi * z)
                    - x * (1 - y) * y * np.sin(np.pi * z),
                    (1 - x) * x * (1 - y) * np.sin(np.pi * z)
                    - (1 - x) * x * y * np.sin(np.pi * z),
                    np.pi * (1 - x) * x * (1 - y) * y * np.cos(np.pi * z),
                ],
            ]
        )


def rotation(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                np.sin(np.pi * x) * np.sin(np.pi * y),
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                (1 - x) * x * np.sin(np.pi * y) * np.sin(np.pi * z),
                (1 - y) * y * np.sin(np.pi * x) * np.sin(np.pi * z),
                (1 - z) * z * np.sin(np.pi * x) * np.sin(np.pi * y),
            ]
        )


def rotation_gradient(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                np.pi * np.cos(np.pi * x) * np.sin(np.pi * y),
                np.pi * np.cos(np.pi * y) * np.sin(np.pi * x),
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                [
                    (1 - x) * np.sin(np.pi * y) * np.sin(np.pi * z)
                    - x * np.sin(np.pi * y) * np.sin(np.pi * z),
                    np.pi * (1 - x) * x * np.cos(np.pi * y) * np.sin(np.pi * z),
                    np.pi * (1 - x) * x * np.cos(np.pi * z) * np.sin(np.pi * y),
                ],
                [
                    np.pi * (1 - y) * y * np.cos(np.pi * x) * np.sin(np.pi * z),
                    (1 - y) * np.sin(np.pi * x) * np.sin(np.pi * z)
                    - y * np.sin(np.pi * x) * np.sin(np.pi * z),
                    np.pi * (1 - y) * y * np.cos(np.pi * z) * np.sin(np.pi * x),
                ],
                [
                    np.pi * (1 - z) * z * np.cos(np.pi * x) * np.sin(np.pi * y),
                    np.pi * (1 - z) * z * np.cos(np.pi * y) * np.sin(np.pi * x),
                    (1 - z) * np.sin(np.pi * x) * np.sin(np.pi * y)
                    - z * np.sin(np.pi * x) * np.sin(np.pi * y),
                ],
            ]
        )


def stress(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                [
                    np.pi
                    * (
                        -((-1 + (y**2)) * (m_lambda + 2 * m_mu) * np.cos(np.pi * x))
                        - (-1 + (x**2)) * m_lambda * np.cos(np.pi * y)
                    ),
                    -2
                    * (
                        x * (-m_kappa + m_mu) * np.sin(np.pi * y)
                        + np.sin(np.pi * x)
                        * (y * (m_kappa + m_mu) + m_kappa * np.sin(np.pi * y))
                    ),
                ],
                [
                    -2 * x * (m_kappa + m_mu) * np.sin(np.pi * y)
                    + 2
                    * np.sin(np.pi * x)
                    * (y * (m_kappa - m_mu) + m_kappa * np.sin(np.pi * y)),
                    np.pi
                    * (
                        -((-1 + (y**2)) * m_lambda * np.cos(np.pi * x))
                        - (-1 + (x**2)) * (m_lambda + 2 * m_mu) * np.cos(np.pi * y)
                    ),
                ],
            ]
        )
    else:
        assert False
        return lambda x, y, z: np.array(
            [
                [
                    2 * np.pi * (-1 + y) * y * (-1 + z) * z * m_mu * np.cos(np.pi * x)
                    + np.pi
                    * m_lambda
                    * (
                        (-1 + y) * y * (-1 + z) * z * np.cos(np.pi * x)
                        + (-1 + x) * x * (-1 + z) * z * np.cos(np.pi * y)
                        + (-1 + x) * x * (-1 + y) * y * np.cos(np.pi * z)
                    ),
                    (-1 + z)
                    * z
                    * (
                        -((-1 + 2 * x) * (m_kappa - m_mu) * np.sin(np.pi * y))
                        + np.sin(np.pi * x)
                        * (
                            (-1 + 2 * y) * (m_kappa + m_mu)
                            + 2 * m_kappa * np.sin(np.pi * y)
                        )
                    ),
                    (-1 + y)
                    * y
                    * (
                        (-1 + 2 * z) * (m_kappa + m_mu) * np.sin(np.pi * x)
                        - (-1 + 2 * x) * (m_kappa - m_mu) * np.sin(np.pi * z)
                        - 2 * m_kappa * np.sin(np.pi * x) * np.sin(np.pi * z)
                    ),
                ],
                [
                    -(
                        (-1 + z)
                        * z
                        * (
                            -((-1 + 2 * x) * (m_kappa + m_mu) * np.sin(np.pi * y))
                            + np.sin(np.pi * x)
                            * (
                                (-1 + 2 * y) * (m_kappa - m_mu)
                                + 2 * m_kappa * np.sin(np.pi * y)
                            )
                        )
                    ),
                    2 * np.pi * (-1 + x) * x * (-1 + z) * z * m_mu * np.cos(np.pi * y)
                    + np.pi
                    * m_lambda
                    * (
                        (-1 + y) * y * (-1 + z) * z * np.cos(np.pi * x)
                        + (-1 + x) * x * (-1 + z) * z * np.cos(np.pi * y)
                        + (-1 + x) * x * (-1 + y) * y * np.cos(np.pi * z)
                    ),
                    (-1 + x)
                    * x
                    * (
                        -((-1 + 2 * y) * (m_kappa - m_mu) * np.sin(np.pi * z))
                        + np.sin(np.pi * y)
                        * (
                            (-1 + 2 * z) * (m_kappa + m_mu)
                            + 2 * m_kappa * np.sin(np.pi * z)
                        )
                    ),
                ],
                [
                    (-1 + y)
                    * y
                    * (
                        (-1 + 2 * x) * (m_kappa + m_mu) * np.sin(np.pi * z)
                        + np.sin(np.pi * x)
                        * (
                            m_kappa
                            - 2 * z * m_kappa
                            - m_mu
                            + 2 * z * m_mu
                            + 2 * m_kappa * np.sin(np.pi * z)
                        )
                    ),
                    -(
                        (-1 + x)
                        * x
                        * (
                            -((-1 + 2 * y) * (m_kappa + m_mu) * np.sin(np.pi * z))
                            + np.sin(np.pi * y)
                            * (
                                (-1 + 2 * z) * (m_kappa - m_mu)
                                + 2 * m_kappa * np.sin(np.pi * z)
                            )
                        )
                    ),
                    2 * np.pi * (-1 + x) * x * (-1 + y) * y * m_mu * np.cos(np.pi * z)
                    + np.pi
                    * m_lambda
                    * (
                        (-1 + y) * y * (-1 + z) * z * np.cos(np.pi * x)
                        + (-1 + x) * x * (-1 + z) * z * np.cos(np.pi * y)
                        + (-1 + x) * x * (-1 + y) * y * np.cos(np.pi * z)
                    ),
                ],
            ]
        )


def couple_stress(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                [
                    np.pi * m_gamma * np.cos(np.pi * x) * np.sin(np.pi * y),
                    np.pi * m_gamma * np.cos(np.pi * y) * np.sin(np.pi * x),
                ],
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                [
                    (1 - 2 * x) * m_gamma * np.sin(np.pi * y) * np.sin(np.pi * z),
                    -(
                        np.pi
                        * (-1 + x)
                        * x
                        * m_gamma
                        * np.cos(np.pi * y)
                        * np.sin(np.pi * z)
                    ),
                    -(
                        np.pi
                        * (-1 + x)
                        * x
                        * m_gamma
                        * np.cos(np.pi * z)
                        * np.sin(np.pi * y)
                    ),
                ],
                [
                    -(
                        np.pi
                        * (-1 + y)
                        * y
                        * m_gamma
                        * np.cos(np.pi * x)
                        * np.sin(np.pi * z)
                    ),
                    (1 - 2 * y) * m_gamma * np.sin(np.pi * x) * np.sin(np.pi * z),
                    -(
                        np.pi
                        * (-1 + y)
                        * y
                        * m_gamma
                        * np.cos(np.pi * z)
                        * np.sin(np.pi * x)
                    ),
                ],
                [
                    -(
                        np.pi
                        * (-1 + z)
                        * z
                        * m_gamma
                        * np.cos(np.pi * x)
                        * np.sin(np.pi * y)
                    ),
                    -(
                        np.pi
                        * (-1 + z)
                        * z
                        * m_gamma
                        * np.cos(np.pi * y)
                        * np.sin(np.pi * x)
                    ),
                    (1 - 2 * z) * m_gamma * np.sin(np.pi * x) * np.sin(np.pi * y),
                ],
            ]
        )


def couple_stress_scaled(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: gamma_eval(x, y, z, dim) * np.array(
            [
                np.pi * np.cos(np.pi * x) * np.sin(np.pi * y),
                np.pi * np.cos(np.pi * y) * np.sin(np.pi * x),
            ]
        )
    else:
        return lambda x, y, z: gamma_eval(x, y, z, dim) * np.array(
            [
                [
                    (1 - x) * np.sin(np.pi * y) * np.sin(np.pi * z)
                    - x * np.sin(np.pi * y) * np.sin(np.pi * z),
                    np.pi * (1 - x) * x * np.cos(np.pi * y) * np.sin(np.pi * z),
                    np.pi * (1 - x) * x * np.cos(np.pi * z) * np.sin(np.pi * y),
                ],
                [
                    np.pi * (1 - y) * y * np.cos(np.pi * x) * np.sin(np.pi * z),
                    (1 - y) * np.sin(np.pi * x) * np.sin(np.pi * z)
                    - y * np.sin(np.pi * x) * np.sin(np.pi * z),
                    np.pi * (1 - y) * y * np.cos(np.pi * z) * np.sin(np.pi * x),
                ],
                [
                    np.pi * (1 - z) * z * np.cos(np.pi * x) * np.sin(np.pi * y),
                    np.pi * (1 - z) * z * np.cos(np.pi * y) * np.sin(np.pi * x),
                    (1 - z) * np.sin(np.pi * x) * np.sin(np.pi * y)
                    - z * np.sin(np.pi * x) * np.sin(np.pi * y),
                ],
            ]
        )


def rhs(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                np.pi * (-1 + 2 * x) * (m_kappa - m_mu) * np.cos(np.pi * y)
                - 2
                * (m_kappa + m_mu + np.pi * m_kappa * np.cos(np.pi * y))
                * np.sin(np.pi * x)
                + np.pi
                * (
                    (m_lambda - 2 * x * m_lambda) * np.cos(np.pi * y)
                    + np.pi * (-1 + y) * y * (m_lambda + 2 * m_mu) * np.sin(np.pi * x)
                ),
                -2 * (m_kappa + m_mu) * np.sin(np.pi * y)
                + np.pi
                * np.cos(np.pi * x)
                * ((-1 + 2 * y) * (m_kappa - m_mu) + 2 * m_kappa * np.sin(np.pi * y))
                + np.pi
                * (
                    (m_lambda - 2 * y * m_lambda) * np.cos(np.pi * x)
                    + np.pi * (-1 + x) * x * (m_lambda + 2 * m_mu) * np.sin(np.pi * y)
                ),
                -2
                * (
                    (1 - 2 * x) * m_kappa * np.sin(np.pi * y)
                    + np.sin(np.pi * x)
                    * (
                        (-1 + 2 * y) * m_kappa
                        + ((np.pi**2) * m_gamma + 2 * m_kappa) * np.sin(np.pi * y)
                    )
                ),
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                -2
                * (np.pi**2)
                * (-1 + y)
                * y
                * (-1 + z)
                * z
                * m_mu
                * np.sin(np.pi * x)
                + (-1 + z)
                * z
                * (
                    -(np.pi * (-1 + 2 * x) * (m_kappa - m_mu) * np.cos(np.pi * y))
                    + 2
                    * (m_kappa + m_mu + np.pi * m_kappa * np.cos(np.pi * y))
                    * np.sin(np.pi * x)
                )
                + np.pi
                * m_lambda
                * (
                    (-1 + 2 * x) * (-1 + z) * z * np.cos(np.pi * y)
                    + (-1 + y)
                    * y
                    * (
                        (-1 + 2 * x) * np.cos(np.pi * z)
                        - np.pi * (-1 + z) * z * np.sin(np.pi * x)
                    )
                )
                + (-1 + y)
                * y
                * (
                    2 * (m_kappa + m_mu) * np.sin(np.pi * x)
                    - np.pi
                    * np.cos(np.pi * z)
                    * (
                        (-1 + 2 * x) * (m_kappa - m_mu)
                        + 2 * m_kappa * np.sin(np.pi * x)
                    )
                ),
                -2
                * (np.pi**2)
                * (-1 + x)
                * x
                * (-1 + z)
                * z
                * m_mu
                * np.sin(np.pi * y)
                + (-1 + x)
                * x
                * (
                    -(np.pi * (-1 + 2 * y) * (m_kappa - m_mu) * np.cos(np.pi * z))
                    + 2
                    * (m_kappa + m_mu + np.pi * m_kappa * np.cos(np.pi * z))
                    * np.sin(np.pi * y)
                )
                + np.pi
                * m_lambda
                * (
                    (-1 + 2 * y) * (-1 + z) * z * np.cos(np.pi * x)
                    + (-1 + x)
                    * x
                    * (
                        (-1 + 2 * y) * np.cos(np.pi * z)
                        - np.pi * (-1 + z) * z * np.sin(np.pi * y)
                    )
                )
                - (-1 + z)
                * z
                * (
                    -2 * (m_kappa + m_mu) * np.sin(np.pi * y)
                    + np.pi
                    * np.cos(np.pi * x)
                    * (
                        (-1 + 2 * y) * (m_kappa - m_mu)
                        + 2 * m_kappa * np.sin(np.pi * y)
                    )
                ),
                -2
                * (np.pi**2)
                * (-1 + x)
                * x
                * (-1 + y)
                * y
                * m_mu
                * np.sin(np.pi * z)
                + np.pi
                * m_lambda
                * (
                    (-1 + y) * y * (-1 + 2 * z) * np.cos(np.pi * x)
                    + (-1 + x)
                    * x
                    * (
                        (-1 + 2 * z) * np.cos(np.pi * y)
                        - np.pi * (-1 + y) * y * np.sin(np.pi * z)
                    )
                )
                - (-1 + x)
                * x
                * (
                    -2 * (m_kappa + m_mu) * np.sin(np.pi * z)
                    + np.pi
                    * np.cos(np.pi * y)
                    * (
                        (-1 + 2 * z) * (m_kappa - m_mu)
                        + 2 * m_kappa * np.sin(np.pi * z)
                    )
                )
                + (-1 + y)
                * y
                * (
                    2 * (m_kappa + m_mu) * np.sin(np.pi * z)
                    + np.pi
                    * np.cos(np.pi * x)
                    * (
                        m_kappa
                        - 2 * z * m_kappa
                        - m_mu
                        + 2 * z * m_mu
                        + 2 * m_kappa * np.sin(np.pi * z)
                    )
                ),
                -2 * (-1 + x) * x * (-1 + 2 * y) * m_kappa * np.sin(np.pi * z)
                + 2
                * np.sin(np.pi * y)
                * (
                    (-1 + x) * x * (-1 + 2 * z) * m_kappa
                    + (
                        (-1 + (np.pi**2) * (-1 + x) * x) * m_gamma
                        + 2 * (-1 + x) * x * m_kappa
                    )
                    * np.sin(np.pi * z)
                ),
                2
                * (
                    (-1 + 2 * x) * (-1 + y) * y * m_kappa * np.sin(np.pi * z)
                    + np.sin(np.pi * x)
                    * (
                        y * (-1 + y + 2 * z - 2 * y * z) * m_kappa
                        + (
                            (-1 + (np.pi**2) * (-1 + y) * y) * m_gamma
                            + 2 * (-1 + y) * y * m_kappa
                        )
                        * np.sin(np.pi * z)
                    )
                ),
                -2 * (-1 + 2 * x) * (-1 + z) * z * m_kappa * np.sin(np.pi * y)
                + 2
                * np.sin(np.pi * x)
                * (
                    (-1 + 2 * y) * (-1 + z) * z * m_kappa
                    + (
                        (-1 + (np.pi**2) * (-1 + z) * z) * m_gamma
                        + 2 * (-1 + z) * z * m_kappa
                    )
                    * np.sin(np.pi * y)
                ),
            ]
        )


def rhs_scaled(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                np.pi
                * (
                    -2 * x * m_lambda * np.cos(np.pi * y)
                    + np.pi
                    * (-1 + (y**2))
                    * (m_lambda + 2 * m_mu)
                    * np.sin(np.pi * x)
                )
                - 2
                * (
                    np.pi * x * (-m_kappa + m_mu) * np.cos(np.pi * y)
                    + (m_kappa + m_mu + np.pi * m_kappa * np.cos(np.pi * y))
                    * np.sin(np.pi * x)
                ),
                -2 * (m_kappa + m_mu) * np.sin(np.pi * y)
                + 2
                * np.pi
                * np.cos(np.pi * x)
                * (y * (m_kappa - m_mu) + m_kappa * np.sin(np.pi * y))
                + np.pi
                * (
                    -2 * y * m_lambda * np.cos(np.pi * x)
                    + np.pi
                    * (-1 + (x**2))
                    * (m_lambda + 2 * m_mu)
                    * np.sin(np.pi * y)
                ),
                4 * x * m_kappa * np.sin(np.pi * y)
                - 4 * m_kappa * np.sin(np.pi * x) * (y + np.sin(np.pi * y))
                - 2
                * (np.pi**2)
                * np.sin(np.pi * x)
                * np.sin(np.pi * y)
                * (gamma_eval(x, y, z, dim) ** 2)
                + 2
                * np.pi
                * gamma_eval(x, y, z, dim)
                * (
                    np.cos(np.pi * y)
                    * np.sin(np.pi * x)
                    * grad_gamma_eval(x, y, z, dim)[1]
                    + np.cos(np.pi * x)
                    * np.sin(np.pi * y)
                    * grad_gamma_eval(x, y, z, dim)[0]
                ),
            ]
        )
    else:
        assert False
        return lambda x, y, z: np.array(
            [
                -2
                * (np.pi**2)
                * (-1 + y)
                * y
                * (-1 + z)
                * z
                * m_mu
                * np.sin(np.pi * x)
                + (-1 + z)
                * z
                * (
                    -(np.pi * (-1 + 2 * x) * (m_kappa - m_mu) * np.cos(np.pi * y))
                    + 2
                    * (m_kappa + m_mu + np.pi * m_kappa * np.cos(np.pi * y))
                    * np.sin(np.pi * x)
                )
                + np.pi
                * m_lambda
                * (
                    (-1 + 2 * x) * (-1 + z) * z * np.cos(np.pi * y)
                    + (-1 + y)
                    * y
                    * (
                        (-1 + 2 * x) * np.cos(np.pi * z)
                        - np.pi * (-1 + z) * z * np.sin(np.pi * x)
                    )
                )
                + (-1 + y)
                * y
                * (
                    2 * (m_kappa + m_mu) * np.sin(np.pi * x)
                    - np.pi
                    * np.cos(np.pi * z)
                    * (
                        (-1 + 2 * x) * (m_kappa - m_mu)
                        + 2 * m_kappa * np.sin(np.pi * x)
                    )
                ),
                -2
                * (np.pi**2)
                * (-1 + x)
                * x
                * (-1 + z)
                * z
                * m_mu
                * np.sin(np.pi * y)
                + (-1 + x)
                * x
                * (
                    -(np.pi * (-1 + 2 * y) * (m_kappa - m_mu) * np.cos(np.pi * z))
                    + 2
                    * (m_kappa + m_mu + np.pi * m_kappa * np.cos(np.pi * z))
                    * np.sin(np.pi * y)
                )
                + np.pi
                * m_lambda
                * (
                    (-1 + 2 * y) * (-1 + z) * z * np.cos(np.pi * x)
                    + (-1 + x)
                    * x
                    * (
                        (-1 + 2 * y) * np.cos(np.pi * z)
                        - np.pi * (-1 + z) * z * np.sin(np.pi * y)
                    )
                )
                - (-1 + z)
                * z
                * (
                    -2 * (m_kappa + m_mu) * np.sin(np.pi * y)
                    + np.pi
                    * np.cos(np.pi * x)
                    * (
                        (-1 + 2 * y) * (m_kappa - m_mu)
                        + 2 * m_kappa * np.sin(np.pi * y)
                    )
                ),
                -2
                * (np.pi**2)
                * (-1 + x)
                * x
                * (-1 + y)
                * y
                * m_mu
                * np.sin(np.pi * z)
                + np.pi
                * m_lambda
                * (
                    (-1 + y) * y * (-1 + 2 * z) * np.cos(np.pi * x)
                    + (-1 + x)
                    * x
                    * (
                        (-1 + 2 * z) * np.cos(np.pi * y)
                        - np.pi * (-1 + y) * y * np.sin(np.pi * z)
                    )
                )
                - (-1 + x)
                * x
                * (
                    -2 * (m_kappa + m_mu) * np.sin(np.pi * z)
                    + np.pi
                    * np.cos(np.pi * y)
                    * (
                        (-1 + 2 * z) * (m_kappa - m_mu)
                        + 2 * m_kappa * np.sin(np.pi * z)
                    )
                )
                + (-1 + y)
                * y
                * (
                    2 * (m_kappa + m_mu) * np.sin(np.pi * z)
                    + np.pi
                    * np.cos(np.pi * x)
                    * (
                        m_kappa
                        - 2 * z * m_kappa
                        - m_mu
                        + 2 * z * m_mu
                        + 2 * m_kappa * np.sin(np.pi * z)
                    )
                ),
                -2 * (-1 + x) * x * (-1 + 2 * y) * m_kappa * np.sin(np.pi * z)
                + 2
                * np.sin(np.pi * y)
                * (
                    (-1 + x) * x * (-1 + 2 * z) * m_kappa
                    + (
                        (-1 + (np.pi**2) * (-1 + x) * x) * (m_gamma)
                        + 2 * (-1 + x) * x * m_kappa
                    )
                    * np.sin(np.pi * z)
                ),
                2
                * (
                    (-1 + 2 * x) * (-1 + y) * y * m_kappa * np.sin(np.pi * z)
                    + np.sin(np.pi * x)
                    * (
                        y * (-1 + y + 2 * z - 2 * y * z) * m_kappa
                        + (
                            (-1 + (np.pi**2) * (-1 + y) * y) * (m_gamma)
                            + 2 * (-1 + y) * y * m_kappa
                        )
                        * np.sin(np.pi * z)
                    )
                ),
                -2 * (-1 + 2 * x) * (-1 + z) * z * m_kappa * np.sin(np.pi * y)
                + 2
                * np.sin(np.pi * x)
                * (
                    (-1 + 2 * y) * (-1 + z) * z * m_kappa
                    + (
                        (-1 + (np.pi**2) * (-1 + z) * z) * (m_gamma)
                        + 2 * (-1 + z) * z * m_kappa
                    )
                    * np.sin(np.pi * y)
                ),
            ]
        )


def stress_divergence(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                np.pi
                * (
                    -2 * x * m_lambda * np.cos(np.pi * y)
                    + np.pi
                    * (-1 + (y**2))
                    * (m_lambda + 2 * m_mu)
                    * np.sin(np.pi * x)
                )
                - 2
                * (
                    np.pi * x * (-m_kappa + m_mu) * np.cos(np.pi * y)
                    + (m_kappa + m_mu + np.pi * m_kappa * np.cos(np.pi * y))
                    * np.sin(np.pi * x)
                ),
                -2 * (m_kappa + m_mu) * np.sin(np.pi * y)
                + 2
                * np.pi
                * np.cos(np.pi * x)
                * (y * (m_kappa - m_mu) + m_kappa * np.sin(np.pi * y))
                + np.pi
                * (
                    -2 * y * m_lambda * np.cos(np.pi * x)
                    + np.pi
                    * (-1 + (x**2))
                    * (m_lambda + 2 * m_mu)
                    * np.sin(np.pi * y)
                ),
            ]
        )
    else:
        assert False
        return lambda x, y, z: np.array(
            [
                -2
                * (np.pi**2)
                * (-1 + y)
                * y
                * (-1 + z)
                * z
                * m_mu
                * np.sin(np.pi * x)
                + (-1 + z)
                * z
                * (
                    -(np.pi * (-1 + 2 * x) * (m_kappa - m_mu) * np.cos(np.pi * y))
                    + 2
                    * (m_kappa + m_mu + np.pi * m_kappa * np.cos(np.pi * y))
                    * np.sin(np.pi * x)
                )
                + np.pi
                * m_lambda
                * (
                    (-1 + 2 * x) * (-1 + z) * z * np.cos(np.pi * y)
                    + (-1 + y)
                    * y
                    * (
                        (-1 + 2 * x) * np.cos(np.pi * z)
                        - np.pi * (-1 + z) * z * np.sin(np.pi * x)
                    )
                )
                + (-1 + y)
                * y
                * (
                    2 * (m_kappa + m_mu) * np.sin(np.pi * x)
                    - np.pi
                    * np.cos(np.pi * z)
                    * (
                        (-1 + 2 * x) * (m_kappa - m_mu)
                        + 2 * m_kappa * np.sin(np.pi * x)
                    )
                ),
                -2
                * (np.pi**2)
                * (-1 + x)
                * x
                * (-1 + z)
                * z
                * m_mu
                * np.sin(np.pi * y)
                + (-1 + x)
                * x
                * (
                    -(np.pi * (-1 + 2 * y) * (m_kappa - m_mu) * np.cos(np.pi * z))
                    + 2
                    * (m_kappa + m_mu + np.pi * m_kappa * np.cos(np.pi * z))
                    * np.sin(np.pi * y)
                )
                + np.pi
                * m_lambda
                * (
                    (-1 + 2 * y) * (-1 + z) * z * np.cos(np.pi * x)
                    + (-1 + x)
                    * x
                    * (
                        (-1 + 2 * y) * np.cos(np.pi * z)
                        - np.pi * (-1 + z) * z * np.sin(np.pi * y)
                    )
                )
                - (-1 + z)
                * z
                * (
                    -2 * (m_kappa + m_mu) * np.sin(np.pi * y)
                    + np.pi
                    * np.cos(np.pi * x)
                    * (
                        (-1 + 2 * y) * (m_kappa - m_mu)
                        + 2 * m_kappa * np.sin(np.pi * y)
                    )
                ),
                -2
                * (np.pi**2)
                * (-1 + x)
                * x
                * (-1 + y)
                * y
                * m_mu
                * np.sin(np.pi * z)
                + np.pi
                * m_lambda
                * (
                    (-1 + y) * y * (-1 + 2 * z) * np.cos(np.pi * x)
                    + (-1 + x)
                    * x
                    * (
                        (-1 + 2 * z) * np.cos(np.pi * y)
                        - np.pi * (-1 + y) * y * np.sin(np.pi * z)
                    )
                )
                - (-1 + x)
                * x
                * (
                    -2 * (m_kappa + m_mu) * np.sin(np.pi * z)
                    + np.pi
                    * np.cos(np.pi * y)
                    * (
                        (-1 + 2 * z) * (m_kappa - m_mu)
                        + 2 * m_kappa * np.sin(np.pi * z)
                    )
                )
                + (-1 + y)
                * y
                * (
                    2 * (m_kappa + m_mu) * np.sin(np.pi * z)
                    + np.pi
                    * np.cos(np.pi * x)
                    * (
                        m_kappa
                        - 2 * z * m_kappa
                        - m_mu
                        + 2 * z * m_mu
                        + 2 * m_kappa * np.sin(np.pi * z)
                    )
                ),
            ]
        )


def couple_stress_divergence(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                -2 * (np.pi**2) * m_gamma * np.sin(np.pi * x) * np.sin(np.pi * y),
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                2
                * (-1 + (np.pi**2) * (-1 + x) * x)
                * m_gamma
                * np.sin(np.pi * y)
                * np.sin(np.pi * z),
                2
                * (-1 + (np.pi**2) * (-1 + y) * y)
                * m_gamma
                * np.sin(np.pi * x)
                * np.sin(np.pi * z),
                2
                * (-1 + (np.pi**2) * (-1 + z) * z)
                * m_gamma
                * np.sin(np.pi * x)
                * np.sin(np.pi * y),
            ]
        )


def couple_stress_divergence_scaled(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                2
                * np.pi
                * gamma_eval(x, y, z, dim)
                * (
                    -(
                        np.pi
                        * np.sin(np.pi * x)
                        * np.sin(np.pi * y)
                        * gamma_eval(x, y, z, dim)
                    )
                    + np.cos(np.pi * y)
                    * np.sin(np.pi * x)
                    * grad_gamma_eval(x, y, z, dim)[1]
                    + np.cos(np.pi * x)
                    * np.sin(np.pi * y)
                    * grad_gamma_eval(x, y, z, dim)[0]
                ),
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                2
                * (-1 + (np.pi**2) * (-1 + x) * x)
                * (m_gamma)
                * np.sin(np.pi * y)
                * np.sin(np.pi * z),
                2
                * (-1 + (np.pi**2) * (-1 + y) * y)
                * (m_gamma)
                * np.sin(np.pi * x)
                * np.sin(np.pi * z),
                2
                * (-1 + (np.pi**2) * (-1 + z) * z)
                * (m_gamma)
                * np.sin(np.pi * x)
                * np.sin(np.pi * y),
            ]
        )
