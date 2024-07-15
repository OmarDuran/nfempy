import numpy as np


def displacement(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                np.pi * np.cos(np.pi * y) * (np.sin(np.pi * x) ** 2),
                -2 * np.pi * np.cos(np.pi * x) * np.sin(np.pi * x) * np.sin(np.pi * y),
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                -2
                * np.pi
                * (np.sin(np.pi * x) ** 2)
                * np.sin(np.pi * y)
                * np.sin(np.pi * (y - z))
                * np.sin(np.pi * z),
                -(
                    np.pi
                    * np.sin(2 * np.pi * x)
                    * (np.sin(np.pi * y) ** 2)
                    * (np.sin(np.pi * z) ** 2)
                ),
                np.pi
                * np.sin(2 * np.pi * x)
                * (np.sin(np.pi * y) ** 2)
                * (np.sin(np.pi * z) ** 2),
            ]
        )


def displacement_gradient(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                [
                    2
                    * (np.pi**2)
                    * np.cos(np.pi * x)
                    * np.cos(np.pi * y)
                    * np.sin(np.pi * x),
                    -((np.pi**2) * (np.sin(np.pi * x) ** 2) * np.sin(np.pi * y)),
                ],
                [
                    -2 * (np.pi**2) * (np.cos(np.pi * x) ** 2) * np.sin(np.pi * y)
                    + 2 * (np.pi**2) * (np.sin(np.pi * x) ** 2) * np.sin(np.pi * y),
                    -2
                    * (np.pi**2)
                    * np.cos(np.pi * x)
                    * np.cos(np.pi * y)
                    * np.sin(np.pi * x),
                ],
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                [
                    -4
                    * (np.pi**2)
                    * np.cos(np.pi * x)
                    * np.sin(np.pi * x)
                    * np.sin(np.pi * y)
                    * np.sin(np.pi * (y - z))
                    * np.sin(np.pi * z),
                    -2
                    * (np.pi**2)
                    * np.cos(np.pi * (y - z))
                    * (np.sin(np.pi * x) ** 2)
                    * np.sin(np.pi * y)
                    * np.sin(np.pi * z)
                    - 2
                    * (np.pi**2)
                    * np.cos(np.pi * y)
                    * (np.sin(np.pi * x) ** 2)
                    * np.sin(np.pi * (y - z))
                    * np.sin(np.pi * z),
                    -2
                    * (np.pi**2)
                    * np.cos(np.pi * z)
                    * (np.sin(np.pi * x) ** 2)
                    * np.sin(np.pi * y)
                    * np.sin(np.pi * (y - z))
                    + 2
                    * (np.pi**2)
                    * np.cos(np.pi * (y - z))
                    * (np.sin(np.pi * x) ** 2)
                    * np.sin(np.pi * y)
                    * np.sin(np.pi * z),
                ],
                [
                    -2
                    * (np.pi**2)
                    * np.cos(2 * np.pi * x)
                    * (np.sin(np.pi * y) ** 2)
                    * (np.sin(np.pi * z) ** 2),
                    -2
                    * (np.pi**2)
                    * np.cos(np.pi * y)
                    * np.sin(2 * np.pi * x)
                    * np.sin(np.pi * y)
                    * (np.sin(np.pi * z) ** 2),
                    -2
                    * (np.pi**2)
                    * np.cos(np.pi * z)
                    * np.sin(2 * np.pi * x)
                    * (np.sin(np.pi * y) ** 2)
                    * np.sin(np.pi * z),
                ],
                [
                    2
                    * (np.pi**2)
                    * np.cos(2 * np.pi * x)
                    * (np.sin(np.pi * y) ** 2)
                    * (np.sin(np.pi * z) ** 2),
                    2
                    * (np.pi**2)
                    * np.cos(np.pi * y)
                    * np.sin(2 * np.pi * x)
                    * np.sin(np.pi * y)
                    * (np.sin(np.pi * z) ** 2),
                    2
                    * (np.pi**2)
                    * np.cos(np.pi * z)
                    * np.sin(2 * np.pi * x)
                    * (np.sin(np.pi * y) ** 2)
                    * np.sin(np.pi * z),
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
                    2 * (np.pi**2) * m_mu * np.cos(np.pi * y) * np.sin(2 * np.pi * x),
                    (
                        (
                            -((np.pi**2) * (m_kappa + m_mu))
                            + (np.pi**2)
                            * (5 * m_kappa - 3 * m_mu)
                            * np.cos(2 * np.pi * x)
                            - 4 * m_kappa * np.sin(np.pi * x)
                        )
                        * np.sin(np.pi * y)
                    )
                    / 2.0,
                ],
                [
                    -0.5
                    * (
                        (
                            (np.pi**2) * (-m_kappa + m_mu)
                            + (np.pi**2)
                            * (5 * m_kappa + 3 * m_mu)
                            * np.cos(2 * np.pi * x)
                            - 4 * m_kappa * np.sin(np.pi * x)
                        )
                        * np.sin(np.pi * y)
                    ),
                    -2 * (np.pi**2) * m_mu * np.cos(np.pi * y) * np.sin(2 * np.pi * x),
                ],
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                [
                    -4
                    * (np.pi**2)
                    * m_mu
                    * np.sin(2 * np.pi * x)
                    * np.sin(np.pi * y)
                    * np.sin(np.pi * (y - z))
                    * np.sin(np.pi * z),
                    -2
                    * (
                        -(
                            (-1 + z)
                            * z
                            * m_kappa
                            * np.sin(np.pi * x)
                            * np.sin(np.pi * y)
                        )
                        + (np.pi**2)
                        * (m_kappa + m_mu)
                        * (np.sin(np.pi * x) ** 2)
                        * np.sin(np.pi * (2 * y - z))
                        * np.sin(np.pi * z)
                        + (np.pi**2)
                        * (-m_kappa + m_mu)
                        * np.cos(2 * np.pi * x)
                        * (np.sin(np.pi * y) ** 2)
                        * (np.sin(np.pi * z) ** 2)
                    ),
                    2
                    * (
                        -(
                            (np.pi**2)
                            * m_mu
                            * (np.cos(np.pi * z) ** 2)
                            * (np.sin(np.pi * x) ** 2)
                            * (np.sin(np.pi * y) ** 2)
                        )
                        - (np.pi**2)
                        * m_kappa
                        * np.cos(np.pi * z)
                        * (np.sin(np.pi * x) ** 2)
                        * np.sin(np.pi * y)
                        * np.sin(np.pi * (y - z))
                        - (-1 + y) * y * m_kappa * np.sin(np.pi * x) * np.sin(np.pi * z)
                        + (np.pi**2)
                        * (
                            m_mu * (np.cos(np.pi * x) ** 2)
                            - m_kappa * np.cos(2 * np.pi * x)
                        )
                        * (np.sin(np.pi * y) ** 2)
                        * (np.sin(np.pi * z) ** 2)
                        + (np.pi**2)
                        * (np.sin(np.pi * x) ** 2)
                        * np.sin(np.pi * y)
                        * (
                            m_kappa * np.cos(np.pi * (y - z)) * np.sin(np.pi * z)
                            + m_mu * np.cos(np.pi * y) * np.sin(2 * np.pi * z)
                        )
                    ),
                ],
                [
                    -2
                    * (
                        (-1 + z) * z * m_kappa * np.sin(np.pi * x) * np.sin(np.pi * y)
                        - (np.pi**2)
                        * (m_kappa - m_mu)
                        * (np.sin(np.pi * x) ** 2)
                        * np.sin(np.pi * (2 * y - z))
                        * np.sin(np.pi * z)
                        + (np.pi**2)
                        * (m_kappa + m_mu)
                        * np.cos(2 * np.pi * x)
                        * (np.sin(np.pi * y) ** 2)
                        * (np.sin(np.pi * z) ** 2)
                    ),
                    -2
                    * (np.pi**2)
                    * m_mu
                    * np.sin(2 * np.pi * x)
                    * np.sin(2 * np.pi * y)
                    * (np.sin(np.pi * z) ** 2),
                    -2
                    * np.sin(np.pi * y)
                    * np.sin(np.pi * z)
                    * (
                        -((-1 + x) * x * m_kappa)
                        + (np.pi**2)
                        * (m_kappa + m_mu)
                        * np.cos(np.pi * z)
                        * np.sin(2 * np.pi * x)
                        * np.sin(np.pi * y)
                        + (np.pi**2)
                        * (m_kappa - m_mu)
                        * np.cos(np.pi * y)
                        * np.sin(2 * np.pi * x)
                        * np.sin(np.pi * z)
                    ),
                ],
                [
                    2
                    * (
                        (np.pi**2)
                        * (m_kappa - m_mu)
                        * np.cos(np.pi * z)
                        * (np.sin(np.pi * x) ** 2)
                        * np.sin(np.pi * y)
                        * np.sin(np.pi * (y - z))
                        + np.sin(np.pi * z)
                        * (
                            (-1 + y) * y * m_kappa * np.sin(np.pi * x)
                            - (np.pi**2)
                            * (m_kappa - m_mu)
                            * np.cos(np.pi * (y - z))
                            * (np.sin(np.pi * x) ** 2)
                            * np.sin(np.pi * y)
                            + (np.pi**2)
                            * (m_kappa + m_mu)
                            * np.cos(2 * np.pi * x)
                            * (np.sin(np.pi * y) ** 2)
                            * np.sin(np.pi * z)
                        )
                    ),
                    2
                    * np.sin(np.pi * y)
                    * np.sin(np.pi * z)
                    * (
                        -((-1 + x) * x * m_kappa)
                        + (np.pi**2)
                        * (m_kappa - m_mu)
                        * np.cos(np.pi * z)
                        * np.sin(2 * np.pi * x)
                        * np.sin(np.pi * y)
                        + (np.pi**2)
                        * (m_kappa + m_mu)
                        * np.cos(np.pi * y)
                        * np.sin(2 * np.pi * x)
                        * np.sin(np.pi * z)
                    ),
                    2
                    * (np.pi**2)
                    * m_mu
                    * np.sin(2 * np.pi * x)
                    * (np.sin(np.pi * y) ** 2)
                    * np.sin(2 * np.pi * z),
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


def rhs(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                (
                    np.pi
                    * np.cos(np.pi * y)
                    * (
                        -((np.pi**2) * (m_kappa + m_mu))
                        + 5 * (np.pi**2) * (m_kappa + m_mu) * np.cos(2 * np.pi * x)
                        - 4 * m_kappa * np.sin(np.pi * x)
                    )
                )
                / 2.0,
                2
                * np.pi
                * np.cos(np.pi * x)
                * (m_kappa + 5 * (np.pi**2) * (m_kappa + m_mu) * np.sin(np.pi * x))
                * np.sin(np.pi * y),
                (
                    -((np.pi**2) * m_kappa)
                    + 5 * (np.pi**2) * m_kappa * np.cos(2 * np.pi * x)
                    - 2 * ((np.pi**2) * m_gamma + 2 * m_kappa) * np.sin(np.pi * x)
                )
                * np.sin(np.pi * y),
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                2
                * np.pi
                * (
                    (-1 + z) * z * m_kappa * np.cos(np.pi * y) * np.sin(np.pi * x)
                    - (-1 + y) * y * m_kappa * np.cos(np.pi * z) * np.sin(np.pi * x)
                    + (np.pi**2)
                    * (m_kappa + m_mu)
                    * (np.cos(np.pi * z) ** 2)
                    * (np.sin(np.pi * x) ** 2)
                    * np.sin(2 * np.pi * y)
                    + (
                        (np.pi**2)
                        * (m_kappa + m_mu)
                        * (
                            -8 * np.cos(np.pi * (2 * x - z))
                            + np.cos(np.pi * (2 * x - 2 * y - z))
                            - 14 * np.cos(np.pi * (2 * y - z))
                            + 11 * np.cos(np.pi * (2 * x + 2 * y - z))
                            + 8 * np.cos(np.pi * z)
                            - 8 * np.cos(np.pi * (2 * x + z))
                            + 11 * np.cos(np.pi * (2 * x - 2 * y + z))
                            - 2 * np.cos(np.pi * (2 * y + z))
                            + np.cos(np.pi * (2 * x + 2 * y + z))
                        )
                        * np.sin(np.pi * z)
                    )
                    / 8.0
                ),
                -2
                * np.pi
                * (-1 + z)
                * z
                * m_kappa
                * np.cos(np.pi * x)
                * np.sin(np.pi * y)
                + 2
                * np.pi
                * (-1 + x)
                * x
                * m_kappa
                * np.cos(np.pi * z)
                * np.sin(np.pi * y)
                - 2
                * (np.pi**3)
                * (m_kappa + m_mu)
                * (np.cos(np.pi * z) ** 2)
                * np.sin(2 * np.pi * x)
                * (np.sin(np.pi * y) ** 2)
                - (np.pi**3)
                * (m_kappa + m_mu)
                * (-3 + 5 * np.cos(2 * np.pi * y))
                * np.sin(2 * np.pi * x)
                * (np.sin(np.pi * z) ** 2),
                2
                * (
                    (np.pi**3)
                    * (m_kappa - m_mu)
                    * (np.cos(np.pi * z) ** 2)
                    * np.sin(2 * np.pi * x)
                    * (np.sin(np.pi * y) ** 2)
                    + 2
                    * (np.pi**3)
                    * m_mu
                    * np.cos(2 * np.pi * z)
                    * np.sin(2 * np.pi * x)
                    * (np.sin(np.pi * y) ** 2)
                    + np.pi
                    * np.sin(np.pi * z)
                    * (
                        (-1 + y) * y * m_kappa * np.cos(np.pi * x)
                        - (-1 + x) * x * m_kappa * np.cos(np.pi * y)
                        + (np.pi**2)
                        * (m_kappa + m_mu)
                        * (np.cos(np.pi * y) ** 2)
                        * np.sin(2 * np.pi * x)
                        * np.sin(np.pi * z)
                        - 2
                        * (np.pi**2)
                        * (2 * m_kappa + m_mu)
                        * np.sin(2 * np.pi * x)
                        * (np.sin(np.pi * y) ** 2)
                        * np.sin(np.pi * z)
                    )
                ),
                2
                * np.sin(np.pi * y)
                * np.sin(np.pi * z)
                * (
                    -m_gamma
                    + (np.pi**2) * (-1 + x) * x * m_gamma
                    - 2 * x * m_kappa
                    + 2 * (x**2) * m_kappa
                    - 2
                    * (np.pi**2)
                    * m_kappa
                    * np.cos(np.pi * z)
                    * np.sin(2 * np.pi * x)
                    * np.sin(np.pi * y)
                    - 2
                    * (np.pi**2)
                    * m_kappa
                    * np.cos(np.pi * y)
                    * np.sin(2 * np.pi * x)
                    * np.sin(np.pi * z)
                ),
                4
                * (np.pi**2)
                * m_kappa
                * (np.cos(np.pi * z) ** 2)
                * (np.sin(np.pi * x) ** 2)
                * (np.sin(np.pi * y) ** 2)
                + 2
                * np.sin(np.pi * z)
                * (
                    (
                        (-1 + (np.pi**2) * (-1 + y) * y) * m_gamma
                        + 2 * (-1 + y) * y * m_kappa
                    )
                    * np.sin(np.pi * x)
                    - 4
                    * (np.pi**2)
                    * m_kappa
                    * np.cos(np.pi * (y - z))
                    * (np.sin(np.pi * x) ** 2)
                    * np.sin(np.pi * y)
                    + 2
                    * (np.pi**2)
                    * m_kappa
                    * (np.cos(np.pi * x) ** 2)
                    * (np.sin(np.pi * y) ** 2)
                    * np.sin(np.pi * z)
                ),
                2
                * (
                    (-1 + (np.pi**2) * (-1 + z) * z) * m_gamma
                    + 2 * (-1 + z) * z * m_kappa
                )
                * np.sin(np.pi * x)
                * np.sin(np.pi * y)
                - 4
                * (np.pi**2)
                * m_kappa
                * (np.sin(np.pi * x) ** 2)
                * np.sin(np.pi * (2 * y - z))
                * np.sin(np.pi * z)
                + 4
                * (np.pi**2)
                * m_kappa
                * np.cos(2 * np.pi * x)
                * (np.sin(np.pi * y) ** 2)
                * (np.sin(np.pi * z) ** 2),
            ]
        )


def stress_divergence(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                (
                    np.pi
                    * np.cos(np.pi * y)
                    * (
                        -((np.pi**2) * (m_kappa + m_mu))
                        + 5 * (np.pi**2) * (m_kappa + m_mu) * np.cos(2 * np.pi * x)
                        - 4 * m_kappa * np.sin(np.pi * x)
                    )
                )
                / 2.0,
                2
                * np.pi
                * np.cos(np.pi * x)
                * (m_kappa + 5 * (np.pi**2) * (m_kappa + m_mu) * np.sin(np.pi * x))
                * np.sin(np.pi * y),
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                2
                * np.pi
                * (
                    (-1 + z) * z * m_kappa * np.cos(np.pi * y) * np.sin(np.pi * x)
                    - (-1 + y) * y * m_kappa * np.cos(np.pi * z) * np.sin(np.pi * x)
                    + (np.pi**2)
                    * (m_kappa + m_mu)
                    * (np.cos(np.pi * z) ** 2)
                    * (np.sin(np.pi * x) ** 2)
                    * np.sin(2 * np.pi * y)
                    + (
                        (np.pi**2)
                        * (m_kappa + m_mu)
                        * (
                            -8 * np.cos(np.pi * (2 * x - z))
                            + np.cos(np.pi * (2 * x - 2 * y - z))
                            - 14 * np.cos(np.pi * (2 * y - z))
                            + 11 * np.cos(np.pi * (2 * x + 2 * y - z))
                            + 8 * np.cos(np.pi * z)
                            - 8 * np.cos(np.pi * (2 * x + z))
                            + 11 * np.cos(np.pi * (2 * x - 2 * y + z))
                            - 2 * np.cos(np.pi * (2 * y + z))
                            + np.cos(np.pi * (2 * x + 2 * y + z))
                        )
                        * np.sin(np.pi * z)
                    )
                    / 8.0
                ),
                -2
                * np.pi
                * (-1 + z)
                * z
                * m_kappa
                * np.cos(np.pi * x)
                * np.sin(np.pi * y)
                + 2
                * np.pi
                * (-1 + x)
                * x
                * m_kappa
                * np.cos(np.pi * z)
                * np.sin(np.pi * y)
                - 2
                * (np.pi**3)
                * (m_kappa + m_mu)
                * (np.cos(np.pi * z) ** 2)
                * np.sin(2 * np.pi * x)
                * (np.sin(np.pi * y) ** 2)
                - (np.pi**3)
                * (m_kappa + m_mu)
                * (-3 + 5 * np.cos(2 * np.pi * y))
                * np.sin(2 * np.pi * x)
                * (np.sin(np.pi * z) ** 2),
                2
                * (
                    (np.pi**3)
                    * (m_kappa - m_mu)
                    * (np.cos(np.pi * z) ** 2)
                    * np.sin(2 * np.pi * x)
                    * (np.sin(np.pi * y) ** 2)
                    + 2
                    * (np.pi**3)
                    * m_mu
                    * np.cos(2 * np.pi * z)
                    * np.sin(2 * np.pi * x)
                    * (np.sin(np.pi * y) ** 2)
                    + np.pi
                    * np.sin(np.pi * z)
                    * (
                        (-1 + y) * y * m_kappa * np.cos(np.pi * x)
                        - (-1 + x) * x * m_kappa * np.cos(np.pi * y)
                        + (np.pi**2)
                        * (m_kappa + m_mu)
                        * (np.cos(np.pi * y) ** 2)
                        * np.sin(2 * np.pi * x)
                        * np.sin(np.pi * z)
                        - 2
                        * (np.pi**2)
                        * (2 * m_kappa + m_mu)
                        * np.sin(2 * np.pi * x)
                        * (np.sin(np.pi * y) ** 2)
                        * np.sin(np.pi * z)
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
                -2
                * (np.pi**2)
                * m_gamma
                * m_gamma
                * np.sin(np.pi * x)
                * np.sin(np.pi * y),
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                2
                * (-1 + (np.pi**2) * (-1 + x) * x)
                * m_gamma
                * m_gamma
                * np.sin(np.pi * y)
                * np.sin(np.pi * z),
                2
                * (-1 + (np.pi**2) * (-1 + y) * y)
                * m_gamma
                * m_gamma
                * np.sin(np.pi * x)
                * np.sin(np.pi * z),
                2
                * (-1 + (np.pi**2) * (-1 + z) * z)
                * m_gamma
                * m_gamma
                * np.sin(np.pi * x)
                * np.sin(np.pi * y),
            ]
        )
