import numpy as np


def displacement(m_lambda, m_mu, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [np.sin(np.pi * x) * y * (1 - y), np.sin(np.pi * y) * x * (1 - x)]
        )
    else:
        return lambda x, y, z: np.array(
            [
                (1 - y) * y * (1 - z) * z * np.sin(np.pi * x),
                (1 - x) * x * (1 - z) * z * np.sin(np.pi * y),
                (1 - x) * x * (1 - y) * y * np.sin(np.pi * z),
            ]
        )


def stress(m_lambda, m_mu, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                [
                    2 * np.pi * (1 - y) * y * m_mu * np.cos(np.pi * x)
                    + m_lambda
                    * (
                        np.pi * (1 - y) * y * np.cos(np.pi * x)
                        + np.pi * (1 - x) * x * np.cos(np.pi * y)
                    ),
                    m_mu
                    * (
                        (1 - y) * np.sin(np.pi * x)
                        - y * np.sin(np.pi * x)
                        + (1 - x) * np.sin(np.pi * y)
                        - x * np.sin(np.pi * y)
                    ),
                ],
                [
                    m_mu
                    * (
                        (1 - y) * np.sin(np.pi * x)
                        - y * np.sin(np.pi * x)
                        + (1 - x) * np.sin(np.pi * y)
                        - x * np.sin(np.pi * y)
                    ),
                    2 * np.pi * (1 - x) * x * m_mu * np.cos(np.pi * y)
                    + m_lambda
                    * (
                        np.pi * (1 - y) * y * np.cos(np.pi * x)
                        + np.pi * (1 - x) * x * np.cos(np.pi * y)
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
                    * m_mu
                    * (
                        (-1 + 2 * y) * np.sin(np.pi * x)
                        + (-1 + 2 * x) * np.sin(np.pi * y)
                    ),
                    (-1 + y)
                    * y
                    * m_mu
                    * (
                        (-1 + 2 * z) * np.sin(np.pi * x)
                        + (-1 + 2 * x) * np.sin(np.pi * z)
                    ),
                ],
                [
                    (-1 + z)
                    * z
                    * m_mu
                    * (
                        (-1 + 2 * y) * np.sin(np.pi * x)
                        + (-1 + 2 * x) * np.sin(np.pi * y)
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
                    * m_mu
                    * (
                        (-1 + 2 * z) * np.sin(np.pi * y)
                        + (-1 + 2 * y) * np.sin(np.pi * z)
                    ),
                ],
                [
                    (-1 + y)
                    * y
                    * m_mu
                    * (
                        (-1 + 2 * z) * np.sin(np.pi * x)
                        + (-1 + 2 * x) * np.sin(np.pi * z)
                    ),
                    (-1 + x)
                    * x
                    * m_mu
                    * (
                        (-1 + 2 * z) * np.sin(np.pi * y)
                        + (-1 + 2 * y) * np.sin(np.pi * z)
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


def rotations(m_lambda, m_mu, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                [
                    0.0 * x,
                    (
                        (1 - y) * np.sin(np.pi * x)
                        - y * np.sin(np.pi * x)
                        - (1 - x) * np.sin(np.pi * y)
                        + x * np.sin(np.pi * y)
                    )
                    / 2.0,
                ],
                [
                    (
                        -((1 - y) * np.sin(np.pi * x))
                        + y * np.sin(np.pi * x)
                        + (1 - x) * np.sin(np.pi * y)
                        - x * np.sin(np.pi * y)
                    )
                    / 2.0,
                    0.0 * y,
                ],
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                [
                    0.0 * x,
                    (
                        (-1 + z)
                        * z
                        * (
                            (-1 + 2 * y) * np.sin(np.pi * x)
                            + (1 - 2 * x) * np.sin(np.pi * y)
                        )
                    )
                    / 2.0,
                    (
                        (-1 + y)
                        * y
                        * (
                            (-1 + 2 * z) * np.sin(np.pi * x)
                            + (1 - 2 * x) * np.sin(np.pi * z)
                        )
                    )
                    / 2.0,
                ],
                [
                    -(
                        (-1 + z)
                        * z
                        * (
                            (-1 + 2 * y) * np.sin(np.pi * x)
                            + (1 - 2 * x) * np.sin(np.pi * y)
                        )
                    )
                    / 2.0,
                    0.0 * y,
                    (
                        (-1 + x)
                        * x
                        * (
                            (-1 + 2 * z) * np.sin(np.pi * y)
                            + (1 - 2 * y) * np.sin(np.pi * z)
                        )
                    )
                    / 2.0,
                ],
                [
                    -(
                        (-1 + y)
                        * y
                        * (
                            (-1 + 2 * z) * np.sin(np.pi * x)
                            + (1 - 2 * x) * np.sin(np.pi * z)
                        )
                    )
                    / 2.0,
                    -(
                        (-1 + x)
                        * x
                        * (
                            (-1 + 2 * z) * np.sin(np.pi * y)
                            + (1 - 2 * y) * np.sin(np.pi * z)
                        )
                    )
                    / 2.0,
                    0.0 * z,
                ],
            ]
        )


def rhs(m_lambda, m_mu, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                -(np.pi * (-1 + 2 * x) * (m_lambda + m_mu) * np.cos(np.pi * y))
                + (-2 * m_mu + (np.pi**2) * (-1 + y) * y * (m_lambda + 2 * m_mu))
                * np.sin(np.pi * x),
                -(np.pi * (-1 + 2 * y) * (m_lambda + m_mu) * np.cos(np.pi * x))
                + (-2 * m_mu + (np.pi**2) * (-1 + x) * x * (m_lambda + 2 * m_mu))
                * np.sin(np.pi * y),
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
                * m_mu
                * (np.pi * (-1 + 2 * x) * np.cos(np.pi * y) + 2 * np.sin(np.pi * x))
                + (-1 + y)
                * y
                * m_mu
                * (np.pi * (-1 + 2 * x) * np.cos(np.pi * z) + 2 * np.sin(np.pi * x))
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
                ),
                -2
                * (np.pi**2)
                * (-1 + x)
                * x
                * (-1 + z)
                * z
                * m_mu
                * np.sin(np.pi * y)
                + (-1 + z)
                * z
                * m_mu
                * (np.pi * (-1 + 2 * y) * np.cos(np.pi * x) + 2 * np.sin(np.pi * y))
                + (-1 + x)
                * x
                * m_mu
                * (np.pi * (-1 + 2 * y) * np.cos(np.pi * z) + 2 * np.sin(np.pi * y))
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
                ),
                -2
                * (np.pi**2)
                * (-1 + x)
                * x
                * (-1 + y)
                * y
                * m_mu
                * np.sin(np.pi * z)
                + (-1 + y)
                * y
                * m_mu
                * (np.pi * (-1 + 2 * z) * np.cos(np.pi * x) + 2 * np.sin(np.pi * z))
                + (-1 + x)
                * x
                * m_mu
                * (np.pi * (-1 + 2 * z) * np.cos(np.pi * y) + 2 * np.sin(np.pi * z))
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
                ),
            ]
        )
