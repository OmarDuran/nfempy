import numpy as np


def generalized_displacement(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                (1 - y) * y * np.sin(np.pi * x),
                (1 - x) * x * np.sin(np.pi * y),
                np.sin(np.pi * x) * np.sin(np.pi * y),
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                (1 - y) * y * (1 - z) * z * np.sin(np.pi * x),
                (1 - x) * x * (1 - z) * z * np.sin(np.pi * y),
                (1 - x) * x * (1 - y) * y * np.sin(np.pi * z),
                (1 - x) * x * np.sin(np.pi * y) * np.sin(np.pi * z),
                (1 - y) * y * np.sin(np.pi * x) * np.sin(np.pi * z),
                (1 - z) * z * np.sin(np.pi * x) * np.sin(np.pi * y),
            ]
        )


def displacement(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                (1 - y) * y * np.sin(np.pi * x),
                (1 - x) * x * np.sin(np.pi * y),
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                (1 - y) * y * (1 - z) * z * np.sin(np.pi * x),
                (1 - x) * x * (1 - z) * z * np.sin(np.pi * y),
                (1 - x) * x * (1 - y) * y * np.sin(np.pi * z),
            ]
        )


def displacement_gradient(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                [
                    np.pi * (1 - y) * y * np.cos(np.pi * x),
                    (1 - y) * np.sin(np.pi * x) - y * np.sin(np.pi * x),
                ],
                [
                    (1 - x) * np.sin(np.pi * y) - x * np.sin(np.pi * y),
                    np.pi * (1 - x) * x * np.cos(np.pi * y),
                ],
            ]
        )
    else:
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
                        -((-1 + y) * y * (m_lambda + 2 * m_mu) * np.cos(np.pi * x))
                        - (-1 + x) * x * m_lambda * np.cos(np.pi * y)
                    ),
                    (-1 + 2 * x) * (m_kappa - m_mu) * np.sin(np.pi * y)
                    + np.sin(np.pi * x)
                    * (
                        -((-1 + 2 * y) * (m_kappa + m_mu))
                        - 2 * m_kappa * np.sin(np.pi * y)
                    ),
                ],
                [
                    -((-1 + 2 * x) * (m_kappa + m_mu) * np.sin(np.pi * y))
                    + np.sin(np.pi * x)
                    * (
                        (-1 + 2 * y) * (m_kappa - m_mu)
                        + 2 * m_kappa * np.sin(np.pi * y)
                    ),
                    np.pi
                    * (
                        -((-1 + y) * y * m_lambda * np.cos(np.pi * x))
                        - (-1 + x) * x * (m_lambda + 2 * m_mu) * np.cos(np.pi * y)
                    ),
                ],
            ]
        )
    else:
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
        return lambda x, y, z: np.sqrt(m_gamma) * np.array(
            [
                np.pi * np.cos(np.pi * x) * np.sin(np.pi * y),
                np.pi * np.cos(np.pi * y) * np.sin(np.pi * x),
            ]
        )
    else:
        return lambda x, y, z: np.sqrt(m_gamma) * np.array(
            [
                [
                    (1 - 2 * x) * np.sin(np.pi * y) * np.sin(np.pi * z),
                    -(np.pi * (-1 + x) * x * np.cos(np.pi * y) * np.sin(np.pi * z)),
                    -(np.pi * (-1 + x) * x * np.cos(np.pi * z) * np.sin(np.pi * y)),
                ],
                [
                    -(np.pi * (-1 + y) * y * np.cos(np.pi * x) * np.sin(np.pi * z)),
                    (1 - 2 * y) * np.sin(np.pi * x) * np.sin(np.pi * z),
                    -(np.pi * (-1 + y) * y * np.cos(np.pi * z) * np.sin(np.pi * x)),
                ],
                [
                    -(np.pi * (-1 + z) * z * np.cos(np.pi * x) * np.sin(np.pi * y)),
                    -(np.pi * (-1 + z) * z * np.cos(np.pi * y) * np.sin(np.pi * x)),
                    (1 - 2 * z) * np.sin(np.pi * x) * np.sin(np.pi * y),
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


def stress_divergence(m_lambda, m_mu, m_kappa, m_gamma, dim: int = 2):
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
