import numpy as np


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
                    np.logical_or.reduce(
                        (
                            np.logical_and.reduce((x - y >= 0, x - z >= 0, x >= 2 / 3)),
                            np.logical_and.reduce((x - y < 0, y - z >= 0, y >= 2 / 3)),
                            np.logical_and.reduce((y - z < 0, x - z < 0, z >= 2 / 3)),
                        )
                    ),
                    np.zeros_like(x),
                    np.where(
                        np.logical_and.reduce(
                            (1 / 3 < x, x < 2 / 3, x - y >= 0, x - z >= 0)
                        ),
                        3.0 * np.ones_like(x),
                        np.zeros_like(x),
                    ),
                ),
                np.where(
                    np.logical_or.reduce(
                        (
                            np.logical_and.reduce((x - y >= 0, x - z >= 0, x >= 2 / 3)),
                            np.logical_and.reduce((x - y < 0, y - z >= 0, y >= 2 / 3)),
                            np.logical_and.reduce((y - z < 0, x - z < 0, z >= 2 / 3)),
                            np.logical_and.reduce(
                                (1 / 3 < x, x < 2 / 3, x - y >= 0, x - z >= 0)
                            ),
                        )
                    ),
                    np.zeros_like(y),
                    np.where(
                        np.logical_and.reduce(
                            (x - y < 0, 1 / 3 < y, y < 2 / 3, y - z >= 0)
                        ),
                        3.0 * np.ones_like(y),
                        np.zeros_like(y),
                    ),
                ),
                np.where(
                    np.logical_or.reduce(
                        (
                            np.logical_and.reduce((x - y >= 0, x - z >= 0, x >= 2 / 3)),
                            np.logical_and.reduce((x - y < 0, y - z >= 0, y >= 2 / 3)),
                            np.logical_and.reduce((y - z < 0, x - z < 0, z >= 2 / 3)),
                            np.logical_and.reduce(
                                (1 / 3 < x, x < 2 / 3, x - y >= 0, x - z >= 0)
                            ),
                            np.logical_and.reduce(
                                (x - y < 0, 1 / 3 < y, y < 2 / 3, y - z >= 0)
                            ),
                        )
                    ),
                    np.zeros_like(z),
                    np.where(
                        np.logical_and.reduce(
                            (x - z < 0, y - z < 0, 1 / 3 < z, z < 2 / 3)
                        ),
                        3.0 * np.ones_like(z),
                        np.zeros_like(z),
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
                    np.logical_or.reduce(
                        (
                            np.logical_and.reduce((x - y >= 0, x - z >= 0, x >= 2 / 3)),
                            np.logical_and.reduce((x - y < 0, y - z >= 0, y >= 2 / 3)),
                            np.logical_and.reduce((y - z < 0, x - z < 0, z >= 2 / 3)),
                        )
                    ),
                    np.zeros_like(x),
                    np.where(
                        np.logical_and.reduce(
                            (1 / 3 < x, x < 2 / 3, x - y >= 0, x - z >= 0)
                        ),
                        3.0 * np.ones_like(x),
                        np.zeros_like(x),
                    ),
                ),
                np.where(
                    np.logical_or.reduce(
                        (
                            np.logical_and.reduce((x - y >= 0, x - z >= 0, x >= 2 / 3)),
                            np.logical_and.reduce((x - y < 0, y - z >= 0, y >= 2 / 3)),
                            np.logical_and.reduce((y - z < 0, x - z < 0, z >= 2 / 3)),
                            np.logical_and.reduce(
                                (1 / 3 < x, x < 2 / 3, x - y >= 0, x - z >= 0)
                            ),
                        )
                    ),
                    np.zeros_like(y),
                    np.where(
                        np.logical_and.reduce(
                            (x - y < 0, 1 / 3 < y, y < 2 / 3, y - z >= 0)
                        ),
                        3.0 * np.ones_like(y),
                        np.zeros_like(y),
                    ),
                ),
                np.where(
                    np.logical_or.reduce(
                        (
                            np.logical_and.reduce((x - y >= 0, x - z >= 0, x >= 2 / 3)),
                            np.logical_and.reduce((x - y < 0, y - z >= 0, y >= 2 / 3)),
                            np.logical_and.reduce((y - z < 0, x - z < 0, z >= 2 / 3)),
                            np.logical_and.reduce(
                                (1 / 3 < x, x < 2 / 3, x - y >= 0, x - z >= 0)
                            ),
                            np.logical_and.reduce(
                                (x - y < 0, 1 / 3 < y, y < 2 / 3, y - z >= 0)
                            ),
                        )
                    ),
                    np.zeros_like(z),
                    np.where(
                        np.logical_and.reduce(
                            (x - z < 0, y - z < 0, 1 / 3 < z, z < 2 / 3)
                        ),
                        3.0 * np.ones_like(z),
                        np.zeros_like(z),
                    ),
                ),
            ]
        )


def displacement(material_data, dim: int = 2):
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


def displacement_gradient(material_data, dim: int = 2):
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


def rotation(material_data, dim: int = 2):
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


def rotation_gradient(material_data, dim: int = 2):
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


def stress(material_data, dim: int = 2):
    m_lambda_s = material_data["lambda_s"]
    m_mu_s = material_data["mu_s"]
    m_kappa_s = material_data["kappa_s"]
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                [
                    np.pi
                    * (
                        -((-1 + y) * y * (m_lambda_s + 2 * m_mu_s) * np.cos(np.pi * x))
                        - (-1 + x) * x * m_lambda_s * np.cos(np.pi * y)
                    ),
                    (-1 + 2 * x) * (m_kappa_s - m_mu_s) * np.sin(np.pi * y)
                    + np.sin(np.pi * x)
                    * (
                        -((-1 + 2 * y) * (m_kappa_s + m_mu_s))
                        - 2 * m_kappa_s * np.sin(np.pi * y)
                    ),
                ],
                [
                    -((-1 + 2 * x) * (m_kappa_s + m_mu_s) * np.sin(np.pi * y))
                    + np.sin(np.pi * x)
                    * (
                        (-1 + 2 * y) * (m_kappa_s - m_mu_s)
                        + 2 * m_kappa_s * np.sin(np.pi * y)
                    ),
                    np.pi
                    * (
                        -((-1 + y) * y * m_lambda_s * np.cos(np.pi * x))
                        - (-1 + x) * x * (m_lambda_s + 2 * m_mu_s) * np.cos(np.pi * y)
                    ),
                ],
            ]
        )
    else:
        return lambda x, y, z: np.array(
            [
                [
                    2 * np.pi * (-1 + y) * y * (-1 + z) * z * m_mu_s * np.cos(np.pi * x)
                    + np.pi
                    * m_lambda_s
                    * (
                        (-1 + y) * y * (-1 + z) * z * np.cos(np.pi * x)
                        + (-1 + x) * x * (-1 + z) * z * np.cos(np.pi * y)
                        + (-1 + x) * x * (-1 + y) * y * np.cos(np.pi * z)
                    ),
                    (-1 + z)
                    * z
                    * (
                        -((-1 + 2 * x) * (m_kappa_s - m_mu_s) * np.sin(np.pi * y))
                        + np.sin(np.pi * x)
                        * (
                            (-1 + 2 * y) * (m_kappa_s + m_mu_s)
                            + 2 * m_kappa_s * np.sin(np.pi * y)
                        )
                    ),
                    (-1 + y)
                    * y
                    * (
                        (-1 + 2 * z) * (m_kappa_s + m_mu_s) * np.sin(np.pi * x)
                        - (-1 + 2 * x) * (m_kappa_s - m_mu_s) * np.sin(np.pi * z)
                        - 2 * m_kappa_s * np.sin(np.pi * x) * np.sin(np.pi * z)
                    ),
                ],
                [
                    -(
                        (-1 + z)
                        * z
                        * (
                            -((-1 + 2 * x) * (m_kappa_s + m_mu_s) * np.sin(np.pi * y))
                            + np.sin(np.pi * x)
                            * (
                                (-1 + 2 * y) * (m_kappa_s - m_mu_s)
                                + 2 * m_kappa_s * np.sin(np.pi * y)
                            )
                        )
                    ),
                    2 * np.pi * (-1 + x) * x * (-1 + z) * z * m_mu_s * np.cos(np.pi * y)
                    + np.pi
                    * m_lambda_s
                    * (
                        (-1 + y) * y * (-1 + z) * z * np.cos(np.pi * x)
                        + (-1 + x) * x * (-1 + z) * z * np.cos(np.pi * y)
                        + (-1 + x) * x * (-1 + y) * y * np.cos(np.pi * z)
                    ),
                    (-1 + x)
                    * x
                    * (
                        -((-1 + 2 * y) * (m_kappa_s - m_mu_s) * np.sin(np.pi * z))
                        + np.sin(np.pi * y)
                        * (
                            (-1 + 2 * z) * (m_kappa_s + m_mu_s)
                            + 2 * m_kappa_s * np.sin(np.pi * z)
                        )
                    ),
                ],
                [
                    (-1 + y)
                    * y
                    * (
                        (-1 + 2 * x) * (m_kappa_s + m_mu_s) * np.sin(np.pi * z)
                        + np.sin(np.pi * x)
                        * (
                            m_kappa_s
                            - 2 * z * m_kappa_s
                            - m_mu_s
                            + 2 * z * m_mu_s
                            + 2 * m_kappa_s * np.sin(np.pi * z)
                        )
                    ),
                    -(
                        (-1 + x)
                        * x
                        * (
                            -((-1 + 2 * y) * (m_kappa_s + m_mu_s) * np.sin(np.pi * z))
                            + np.sin(np.pi * y)
                            * (
                                (-1 + 2 * z) * (m_kappa_s - m_mu_s)
                                + 2 * m_kappa_s * np.sin(np.pi * z)
                            )
                        )
                    ),
                    2 * np.pi * (-1 + x) * x * (-1 + y) * y * m_mu_s * np.cos(np.pi * z)
                    + np.pi
                    * m_lambda_s
                    * (
                        (-1 + y) * y * (-1 + z) * z * np.cos(np.pi * x)
                        + (-1 + x) * x * (-1 + z) * z * np.cos(np.pi * y)
                        + (-1 + x) * x * (-1 + y) * y * np.cos(np.pi * z)
                    ),
                ],
            ]
        )


def couple_stress_scaled(material_data, dim: int = 2):
    m_lambda_o = material_data["lambda_o"]
    m_mu_o = material_data["mu_o"]
    m_kappa_o = material_data["kappa_o"]
    if dim == 2:
        return lambda x, y, z: gamma_eval(x, y, z, dim) * np.array(
            [
                [
                    np.pi
                    * (m_mu_o + m_kappa_o)
                    * np.cos(np.pi * x)
                    * np.sin(np.pi * y),
                    np.pi
                    * (m_mu_o + m_kappa_o)
                    * np.cos(np.pi * y)
                    * np.sin(np.pi * x),
                ]
            ]
        )
    else:
        return lambda x, y, z: gamma_eval(x, y, z, dim) * np.array(
            [
                [
                    -(
                        (-1 + 2 * x)
                        * (m_lambda_o + 2 * m_mu_o)
                        * np.sin(np.pi * y)
                        * np.sin(np.pi * z)
                    )
                    + m_lambda_o
                    * np.sin(np.pi * x)
                    * (
                        (1 - 2 * z) * np.sin(np.pi * y)
                        + (1 - 2 * y) * np.sin(np.pi * z)
                    ),
                    np.pi
                    * (
                        (-1 + y) * y * (m_kappa_o - m_mu_o) * np.cos(np.pi * x)
                        - (-1 + x) * x * (m_kappa_o + m_mu_o) * np.cos(np.pi * y)
                    )
                    * np.sin(np.pi * z),
                    np.pi
                    * (
                        (-1 + z) * z * (m_kappa_o - m_mu_o) * np.cos(np.pi * x)
                        - (-1 + x) * x * (m_kappa_o + m_mu_o) * np.cos(np.pi * z)
                    )
                    * np.sin(np.pi * y),
                ],
                [
                    np.pi
                    * (
                        -((-1 + y) * y * (m_kappa_o + m_mu_o) * np.cos(np.pi * x))
                        + (-1 + x) * x * (m_kappa_o - m_mu_o) * np.cos(np.pi * y)
                    )
                    * np.sin(np.pi * z),
                    (1 - 2 * x) * m_lambda_o * np.sin(np.pi * y) * np.sin(np.pi * z)
                    + np.sin(np.pi * x)
                    * (
                        (m_lambda_o - 2 * z * m_lambda_o) * np.sin(np.pi * y)
                        - (-1 + 2 * y) * (m_lambda_o + 2 * m_mu_o) * np.sin(np.pi * z)
                    ),
                    np.pi
                    * (
                        (-1 + z) * z * (m_kappa_o - m_mu_o) * np.cos(np.pi * y)
                        - (-1 + y) * y * (m_kappa_o + m_mu_o) * np.cos(np.pi * z)
                    )
                    * np.sin(np.pi * x),
                ],
                [
                    np.pi
                    * (
                        -((-1 + z) * z * (m_kappa_o + m_mu_o) * np.cos(np.pi * x))
                        + (-1 + x) * x * (m_kappa_o - m_mu_o) * np.cos(np.pi * z)
                    )
                    * np.sin(np.pi * y),
                    np.pi
                    * (
                        -((-1 + z) * z * (m_kappa_o + m_mu_o) * np.cos(np.pi * y))
                        + (-1 + y) * y * (m_kappa_o - m_mu_o) * np.cos(np.pi * z)
                    )
                    * np.sin(np.pi * x),
                    (1 - 2 * x) * m_lambda_o * np.sin(np.pi * y) * np.sin(np.pi * z)
                    + np.sin(np.pi * x)
                    * (
                        -((-1 + 2 * z) * (m_lambda_o + 2 * m_mu_o) * np.sin(np.pi * y))
                        + (1 - 2 * y) * m_lambda_o * np.sin(np.pi * z)
                    ),
                ],
            ]
        )


def rhs_scaled(material_data, dim: int = 2):

    m_lambda_s = material_data["lambda_s"]
    m_mu_s = material_data["mu_s"]
    m_kappa_s = material_data["kappa_s"]

    m_lambda_o = material_data["lambda_o"]
    m_mu_o = material_data["mu_o"]
    m_kappa_o = material_data["kappa_o"]

    if dim == 2:
        return lambda x, y, z: np.array(
            [
                np.pi * (-1 + 2 * x) * (m_kappa_s - m_mu_s) * np.cos(np.pi * y)
                - 2
                * (m_kappa_s + m_mu_s + np.pi * m_kappa_s * np.cos(np.pi * y))
                * np.sin(np.pi * x)
                + np.pi
                * (
                    (m_lambda_s - 2 * x * m_lambda_s) * np.cos(np.pi * y)
                    + np.pi
                    * (-1 + y)
                    * y
                    * (m_lambda_s + 2 * m_mu_s)
                    * np.sin(np.pi * x)
                ),
                -2 * (m_kappa_s + m_mu_s) * np.sin(np.pi * y)
                + np.pi
                * np.cos(np.pi * x)
                * (
                    (-1 + 2 * y) * (m_kappa_s - m_mu_s)
                    + 2 * m_kappa_s * np.sin(np.pi * y)
                )
                + np.pi
                * (
                    (m_lambda_s - 2 * y * m_lambda_s) * np.cos(np.pi * x)
                    + np.pi
                    * (-1 + x)
                    * x
                    * (m_lambda_s + 2 * m_mu_s)
                    * np.sin(np.pi * y)
                ),
                -2
                * m_kappa_s
                * (
                    (1 - 2 * x) * np.sin(np.pi * y)
                    + np.sin(np.pi * x) * (-1 + 2 * y + 2 * np.sin(np.pi * y))
                )
                - 2
                * (np.pi**2)
                * (m_kappa_o + m_mu_o)
                * np.sin(np.pi * x)
                * np.sin(np.pi * y)
                * (gamma_eval(x, y, z, dim) ** 2)
                + 2
                * np.pi
                * (m_kappa_o + m_mu_o)
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
        return lambda x, y, z: np.array(
            [
                -2
                * (np.pi**2)
                * (-1 + y)
                * y
                * (-1 + z)
                * z
                * m_mu_s
                * np.sin(np.pi * x)
                + (-1 + z)
                * z
                * (
                    -(np.pi * (-1 + 2 * x) * (m_kappa_s - m_mu_s) * np.cos(np.pi * y))
                    + 2
                    * (m_kappa_s + m_mu_s + np.pi * m_kappa_s * np.cos(np.pi * y))
                    * np.sin(np.pi * x)
                )
                + np.pi
                * m_lambda_s
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
                    2 * (m_kappa_s + m_mu_s) * np.sin(np.pi * x)
                    - np.pi
                    * np.cos(np.pi * z)
                    * (
                        (-1 + 2 * x) * (m_kappa_s - m_mu_s)
                        + 2 * m_kappa_s * np.sin(np.pi * x)
                    )
                ),
                -2
                * (np.pi**2)
                * (-1 + x)
                * x
                * (-1 + z)
                * z
                * m_mu_s
                * np.sin(np.pi * y)
                + (-1 + x)
                * x
                * (
                    -(np.pi * (-1 + 2 * y) * (m_kappa_s - m_mu_s) * np.cos(np.pi * z))
                    + 2
                    * (m_kappa_s + m_mu_s + np.pi * m_kappa_s * np.cos(np.pi * z))
                    * np.sin(np.pi * y)
                )
                + np.pi
                * m_lambda_s
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
                    -2 * (m_kappa_s + m_mu_s) * np.sin(np.pi * y)
                    + np.pi
                    * np.cos(np.pi * x)
                    * (
                        (-1 + 2 * y) * (m_kappa_s - m_mu_s)
                        + 2 * m_kappa_s * np.sin(np.pi * y)
                    )
                ),
                -2
                * (np.pi**2)
                * (-1 + x)
                * x
                * (-1 + y)
                * y
                * m_mu_s
                * np.sin(np.pi * z)
                + np.pi
                * m_lambda_s
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
                    -2 * (m_kappa_s + m_mu_s) * np.sin(np.pi * z)
                    + np.pi
                    * np.cos(np.pi * y)
                    * (
                        (-1 + 2 * z) * (m_kappa_s - m_mu_s)
                        + 2 * m_kappa_s * np.sin(np.pi * z)
                    )
                )
                + (-1 + y)
                * y
                * (
                    2 * (m_kappa_s + m_mu_s) * np.sin(np.pi * z)
                    + np.pi
                    * np.cos(np.pi * x)
                    * (
                        m_kappa_s
                        - 2 * z * m_kappa_s
                        - m_mu_s
                        + 2 * z * m_mu_s
                        + 2 * m_kappa_s * np.sin(np.pi * z)
                    )
                ),
                2 * x * m_kappa_s * np.sin(np.pi * y)
                - 2 * (x**2) * m_kappa_s * np.sin(np.pi * y)
                - 4 * x * z * m_kappa_s * np.sin(np.pi * y)
                + 4 * (x**2) * z * m_kappa_s * np.sin(np.pi * y)
                - 2 * x * m_kappa_s * np.sin(np.pi * z)
                + 2 * (x**2) * m_kappa_s * np.sin(np.pi * z)
                + 4 * x * y * m_kappa_s * np.sin(np.pi * z)
                - 4 * (x**2) * y * m_kappa_s * np.sin(np.pi * z)
                - 4 * x * m_kappa_s * np.sin(np.pi * y) * np.sin(np.pi * z)
                + 4 * (x**2) * m_kappa_s * np.sin(np.pi * y) * np.sin(np.pi * z)
                + gamma_eval(x, y, z, dim)
                * (
                    np.pi
                    * (
                        (-1 + 2 * y) * (m_kappa_o - m_mu_o) * np.cos(np.pi * x)
                        + np.pi
                        * (-1 + x)
                        * x
                        * (m_kappa_o + m_mu_o)
                        * np.sin(np.pi * y)
                    )
                    * np.sin(np.pi * z)
                    * gamma_eval(x, y, z, dim)
                    + np.pi
                    * np.sin(np.pi * y)
                    * (
                        (-1 + 2 * z) * (m_kappa_o - m_mu_o) * np.cos(np.pi * x)
                        + np.pi
                        * (-1 + x)
                        * x
                        * (m_kappa_o + m_mu_o)
                        * np.sin(np.pi * z)
                    )
                    * gamma_eval(x, y, z, dim)
                    - (
                        2
                        * (m_lambda_o + 2 * m_mu_o)
                        * np.sin(np.pi * y)
                        * np.sin(np.pi * z)
                        + np.pi
                        * m_lambda_o
                        * np.cos(np.pi * x)
                        * (
                            (-1 + 2 * z) * np.sin(np.pi * y)
                            + (-1 + 2 * y) * np.sin(np.pi * z)
                        )
                    )
                    * gamma_eval(x, y, z, dim)
                    + 2
                    * np.pi
                    * (
                        (-1 + z) * z * (m_kappa_o - m_mu_o) * np.cos(np.pi * x)
                        - (-1 + x) * x * (m_kappa_o + m_mu_o) * np.cos(np.pi * z)
                    )
                    * np.sin(np.pi * y)
                    * grad_gamma_eval(x, y, z, dim)[2]
                    + 2
                    * np.pi
                    * (
                        (-1 + y) * y * (m_kappa_o - m_mu_o) * np.cos(np.pi * x)
                        - (-1 + x) * x * (m_kappa_o + m_mu_o) * np.cos(np.pi * y)
                    )
                    * np.sin(np.pi * z)
                    * grad_gamma_eval(x, y, z, dim)[1]
                    - 2
                    * (
                        (-1 + 2 * x)
                        * (m_lambda_o + 2 * m_mu_o)
                        * np.sin(np.pi * y)
                        * np.sin(np.pi * z)
                        + m_lambda_o
                        * np.sin(np.pi * x)
                        * (
                            (-1 + 2 * z) * np.sin(np.pi * y)
                            + (-1 + 2 * y) * np.sin(np.pi * z)
                        )
                    )
                    * grad_gamma_eval(x, y, z, dim)[0]
                ),
                -2 * y * m_kappa_s * np.sin(np.pi * x)
                + 2 * (y**2) * m_kappa_s * np.sin(np.pi * x)
                + 4 * y * z * m_kappa_s * np.sin(np.pi * x)
                - 4 * (y**2) * z * m_kappa_s * np.sin(np.pi * x)
                + 2 * y * m_kappa_s * np.sin(np.pi * z)
                - 4 * x * y * m_kappa_s * np.sin(np.pi * z)
                - 2 * (y**2) * m_kappa_s * np.sin(np.pi * z)
                + 4 * x * (y**2) * m_kappa_s * np.sin(np.pi * z)
                - 4 * y * m_kappa_s * np.sin(np.pi * x) * np.sin(np.pi * z)
                + 4 * (y**2) * m_kappa_s * np.sin(np.pi * x) * np.sin(np.pi * z)
                + gamma_eval(x, y, z, dim)
                * (
                    np.pi
                    * (
                        (-1 + 2 * x) * (m_kappa_o - m_mu_o) * np.cos(np.pi * y)
                        + np.pi
                        * (-1 + y)
                        * y
                        * (m_kappa_o + m_mu_o)
                        * np.sin(np.pi * x)
                    )
                    * np.sin(np.pi * z)
                    * gamma_eval(x, y, z, dim)
                    + np.pi
                    * np.sin(np.pi * x)
                    * (
                        (-1 + 2 * z) * (m_kappa_o - m_mu_o) * np.cos(np.pi * y)
                        + np.pi
                        * (-1 + y)
                        * y
                        * (m_kappa_o + m_mu_o)
                        * np.sin(np.pi * z)
                    )
                    * gamma_eval(x, y, z, dim)
                    - (
                        2
                        * (m_lambda_o + 2 * m_mu_o)
                        * np.sin(np.pi * x)
                        * np.sin(np.pi * z)
                        + np.pi
                        * m_lambda_o
                        * np.cos(np.pi * y)
                        * (
                            (-1 + 2 * z) * np.sin(np.pi * x)
                            + (-1 + 2 * x) * np.sin(np.pi * z)
                        )
                    )
                    * gamma_eval(x, y, z, dim)
                    + 2
                    * np.pi
                    * (
                        (-1 + z) * z * (m_kappa_o - m_mu_o) * np.cos(np.pi * y)
                        - (-1 + y) * y * (m_kappa_o + m_mu_o) * np.cos(np.pi * z)
                    )
                    * np.sin(np.pi * x)
                    * grad_gamma_eval(x, y, z, dim)[2]
                    - 2
                    * (
                        (-1 + 2 * x)
                        * m_lambda_o
                        * np.sin(np.pi * y)
                        * np.sin(np.pi * z)
                        + np.sin(np.pi * x)
                        * (
                            (-1 + 2 * z) * m_lambda_o * np.sin(np.pi * y)
                            + (-1 + 2 * y)
                            * (m_lambda_o + 2 * m_mu_o)
                            * np.sin(np.pi * z)
                        )
                    )
                    * grad_gamma_eval(x, y, z, dim)[1]
                    + 2
                    * np.pi
                    * (
                        -((-1 + y) * y * (m_kappa_o + m_mu_o) * np.cos(np.pi * x))
                        + (-1 + x) * x * (m_kappa_o - m_mu_o) * np.cos(np.pi * y)
                    )
                    * np.sin(np.pi * z)
                    * grad_gamma_eval(x, y, z, dim)[0]
                ),
                2 * z * m_kappa_s * np.sin(np.pi * x)
                - 4 * y * z * m_kappa_s * np.sin(np.pi * x)
                - 2 * (z**2) * m_kappa_s * np.sin(np.pi * x)
                + 4 * y * (z**2) * m_kappa_s * np.sin(np.pi * x)
                - 2 * z * m_kappa_s * np.sin(np.pi * y)
                + 4 * x * z * m_kappa_s * np.sin(np.pi * y)
                + 2 * (z**2) * m_kappa_s * np.sin(np.pi * y)
                - 4 * x * (z**2) * m_kappa_s * np.sin(np.pi * y)
                - 4 * z * m_kappa_s * np.sin(np.pi * x) * np.sin(np.pi * y)
                + 4 * (z**2) * m_kappa_s * np.sin(np.pi * x) * np.sin(np.pi * y)
                + gamma_eval(x, y, z, dim)
                * (
                    np.pi
                    * (
                        (-1 + 2 * x) * (m_kappa_o - m_mu_o) * np.cos(np.pi * z)
                        + np.pi
                        * (-1 + z)
                        * z
                        * (m_kappa_o + m_mu_o)
                        * np.sin(np.pi * x)
                    )
                    * np.sin(np.pi * y)
                    * gamma_eval(x, y, z, dim)
                    + np.pi
                    * np.sin(np.pi * x)
                    * (
                        (-1 + 2 * y) * (m_kappa_o - m_mu_o) * np.cos(np.pi * z)
                        + np.pi
                        * (-1 + z)
                        * z
                        * (m_kappa_o + m_mu_o)
                        * np.sin(np.pi * y)
                    )
                    * gamma_eval(x, y, z, dim)
                    - (
                        2
                        * (m_lambda_o + 2 * m_mu_o)
                        * np.sin(np.pi * x)
                        * np.sin(np.pi * y)
                        + np.pi
                        * m_lambda_o
                        * np.cos(np.pi * z)
                        * (
                            (-1 + 2 * y) * np.sin(np.pi * x)
                            + (-1 + 2 * x) * np.sin(np.pi * y)
                        )
                    )
                    * gamma_eval(x, y, z, dim)
                    - 2
                    * (
                        (-1 + 2 * x)
                        * m_lambda_o
                        * np.sin(np.pi * y)
                        * np.sin(np.pi * z)
                        + np.sin(np.pi * x)
                        * (
                            (-1 + 2 * z) * (m_lambda_o + 2 * m_mu_o) * np.sin(np.pi * y)
                            + (-1 + 2 * y) * m_lambda_o * np.sin(np.pi * z)
                        )
                    )
                    * grad_gamma_eval(x, y, z, dim)[2]
                    + 2
                    * np.pi
                    * (
                        -((-1 + z) * z * (m_kappa_o + m_mu_o) * np.cos(np.pi * y))
                        + (-1 + y) * y * (m_kappa_o - m_mu_o) * np.cos(np.pi * z)
                    )
                    * np.sin(np.pi * x)
                    * grad_gamma_eval(x, y, z, dim)[1]
                    + 2
                    * np.pi
                    * (
                        -((-1 + z) * z * (m_kappa_o + m_mu_o) * np.cos(np.pi * x))
                        + (-1 + x) * x * (m_kappa_o - m_mu_o) * np.cos(np.pi * z)
                    )
                    * np.sin(np.pi * y)
                    * grad_gamma_eval(x, y, z, dim)[0]
                ),
            ]
        )


def stress_divergence(material_data, dim: int = 2):
    m_lambda_s = material_data["lambda_s"]
    m_mu_s = material_data["mu_s"]
    m_kappa_s = material_data["kappa_s"]
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                np.pi * (-1 + 2 * x) * (m_kappa_s - m_mu_s) * np.cos(np.pi * y)
                - 2
                * (m_kappa_s + m_mu_s + np.pi * m_kappa_s * np.cos(np.pi * y))
                * np.sin(np.pi * x)
                + np.pi
                * (
                    (m_lambda_s - 2 * x * m_lambda_s) * np.cos(np.pi * y)
                    + np.pi
                    * (-1 + y)
                    * y
                    * (m_lambda_s + 2 * m_mu_s)
                    * np.sin(np.pi * x)
                ),
                -2 * (m_kappa_s + m_mu_s) * np.sin(np.pi * y)
                + np.pi
                * np.cos(np.pi * x)
                * (
                    (-1 + 2 * y) * (m_kappa_s - m_mu_s)
                    + 2 * m_kappa_s * np.sin(np.pi * y)
                )
                + np.pi
                * (
                    (m_lambda_s - 2 * y * m_lambda_s) * np.cos(np.pi * x)
                    + np.pi
                    * (-1 + x)
                    * x
                    * (m_lambda_s + 2 * m_mu_s)
                    * np.sin(np.pi * y)
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
                * m_mu_s
                * np.sin(np.pi * x)
                + (-1 + z)
                * z
                * (
                    -(np.pi * (-1 + 2 * x) * (m_kappa_s - m_mu_s) * np.cos(np.pi * y))
                    + 2
                    * (m_kappa_s + m_mu_s + np.pi * m_kappa_s * np.cos(np.pi * y))
                    * np.sin(np.pi * x)
                )
                + np.pi
                * m_lambda_s
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
                    2 * (m_kappa_s + m_mu_s) * np.sin(np.pi * x)
                    - np.pi
                    * np.cos(np.pi * z)
                    * (
                        (-1 + 2 * x) * (m_kappa_s - m_mu_s)
                        + 2 * m_kappa_s * np.sin(np.pi * x)
                    )
                ),
                -2
                * (np.pi**2)
                * (-1 + x)
                * x
                * (-1 + z)
                * z
                * m_mu_s
                * np.sin(np.pi * y)
                + (-1 + x)
                * x
                * (
                    -(np.pi * (-1 + 2 * y) * (m_kappa_s - m_mu_s) * np.cos(np.pi * z))
                    + 2
                    * (m_kappa_s + m_mu_s + np.pi * m_kappa_s * np.cos(np.pi * z))
                    * np.sin(np.pi * y)
                )
                + np.pi
                * m_lambda_s
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
                    -2 * (m_kappa_s + m_mu_s) * np.sin(np.pi * y)
                    + np.pi
                    * np.cos(np.pi * x)
                    * (
                        (-1 + 2 * y) * (m_kappa_s - m_mu_s)
                        + 2 * m_kappa_s * np.sin(np.pi * y)
                    )
                ),
                -2
                * (np.pi**2)
                * (-1 + x)
                * x
                * (-1 + y)
                * y
                * m_mu_s
                * np.sin(np.pi * z)
                + np.pi
                * m_lambda_s
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
                    -2 * (m_kappa_s + m_mu_s) * np.sin(np.pi * z)
                    + np.pi
                    * np.cos(np.pi * y)
                    * (
                        (-1 + 2 * z) * (m_kappa_s - m_mu_s)
                        + 2 * m_kappa_s * np.sin(np.pi * z)
                    )
                )
                + (-1 + y)
                * y
                * (
                    2 * (m_kappa_s + m_mu_s) * np.sin(np.pi * z)
                    + np.pi
                    * np.cos(np.pi * x)
                    * (
                        m_kappa_s
                        - 2 * z * m_kappa_s
                        - m_mu_s
                        + 2 * z * m_mu_s
                        + 2 * m_kappa_s * np.sin(np.pi * z)
                    )
                ),
            ]
        )


def couple_stress_divergence_scaled(material_data, dim: int = 2):
    m_lambda_o = material_data["lambda_o"]
    m_mu_o = material_data["mu_o"]
    m_kappa_o = material_data["kappa_o"]
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                2
                * np.pi
                * (m_mu_o + m_kappa_o)
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
                gamma_eval(x, y, z, dim)
                * (
                    np.pi
                    * (
                        (-1 + 2 * y) * (m_kappa_o - m_mu_o) * np.cos(np.pi * x)
                        + np.pi
                        * (-1 + x)
                        * x
                        * (m_kappa_o + m_mu_o)
                        * np.sin(np.pi * y)
                    )
                    * np.sin(np.pi * z)
                    * gamma_eval(x, y, z, dim)
                    + np.pi
                    * np.sin(np.pi * y)
                    * (
                        (-1 + 2 * z) * (m_kappa_o - m_mu_o) * np.cos(np.pi * x)
                        + np.pi
                        * (-1 + x)
                        * x
                        * (m_kappa_o + m_mu_o)
                        * np.sin(np.pi * z)
                    )
                    * gamma_eval(x, y, z, dim)
                    - (
                        2
                        * (m_lambda_o + 2 * m_mu_o)
                        * np.sin(np.pi * y)
                        * np.sin(np.pi * z)
                        + np.pi
                        * m_lambda_o
                        * np.cos(np.pi * x)
                        * (
                            (-1 + 2 * z) * np.sin(np.pi * y)
                            + (-1 + 2 * y) * np.sin(np.pi * z)
                        )
                    )
                    * gamma_eval(x, y, z, dim)
                    + 2
                    * np.pi
                    * (
                        (-1 + z) * z * (m_kappa_o - m_mu_o) * np.cos(np.pi * x)
                        - (-1 + x) * x * (m_kappa_o + m_mu_o) * np.cos(np.pi * z)
                    )
                    * np.sin(np.pi * y)
                    * grad_gamma_eval(x, y, z, dim)[2]
                    + 2
                    * np.pi
                    * (
                        (-1 + y) * y * (m_kappa_o - m_mu_o) * np.cos(np.pi * x)
                        - (-1 + x) * x * (m_kappa_o + m_mu_o) * np.cos(np.pi * y)
                    )
                    * np.sin(np.pi * z)
                    * grad_gamma_eval(x, y, z, dim)[1]
                    - 2
                    * (
                        (-1 + 2 * x)
                        * (m_lambda_o + 2 * m_mu_o)
                        * np.sin(np.pi * y)
                        * np.sin(np.pi * z)
                        + m_lambda_o
                        * np.sin(np.pi * x)
                        * (
                            (-1 + 2 * z) * np.sin(np.pi * y)
                            + (-1 + 2 * y) * np.sin(np.pi * z)
                        )
                    )
                    * grad_gamma_eval(x, y, z, dim)[0]
                ),
                gamma_eval(x, y, z, dim)
                * (
                    np.pi
                    * (
                        (-1 + 2 * x) * (m_kappa_o - m_mu_o) * np.cos(np.pi * y)
                        + np.pi
                        * (-1 + y)
                        * y
                        * (m_kappa_o + m_mu_o)
                        * np.sin(np.pi * x)
                    )
                    * np.sin(np.pi * z)
                    * gamma_eval(x, y, z, dim)
                    + np.pi
                    * np.sin(np.pi * x)
                    * (
                        (-1 + 2 * z) * (m_kappa_o - m_mu_o) * np.cos(np.pi * y)
                        + np.pi
                        * (-1 + y)
                        * y
                        * (m_kappa_o + m_mu_o)
                        * np.sin(np.pi * z)
                    )
                    * gamma_eval(x, y, z, dim)
                    - (
                        2
                        * (m_lambda_o + 2 * m_mu_o)
                        * np.sin(np.pi * x)
                        * np.sin(np.pi * z)
                        + np.pi
                        * m_lambda_o
                        * np.cos(np.pi * y)
                        * (
                            (-1 + 2 * z) * np.sin(np.pi * x)
                            + (-1 + 2 * x) * np.sin(np.pi * z)
                        )
                    )
                    * gamma_eval(x, y, z, dim)
                    + 2
                    * np.pi
                    * (
                        (-1 + z) * z * (m_kappa_o - m_mu_o) * np.cos(np.pi * y)
                        - (-1 + y) * y * (m_kappa_o + m_mu_o) * np.cos(np.pi * z)
                    )
                    * np.sin(np.pi * x)
                    * grad_gamma_eval(x, y, z, dim)[2]
                    - 2
                    * (
                        (-1 + 2 * x)
                        * m_lambda_o
                        * np.sin(np.pi * y)
                        * np.sin(np.pi * z)
                        + np.sin(np.pi * x)
                        * (
                            (-1 + 2 * z) * m_lambda_o * np.sin(np.pi * y)
                            + (-1 + 2 * y)
                            * (m_lambda_o + 2 * m_mu_o)
                            * np.sin(np.pi * z)
                        )
                    )
                    * grad_gamma_eval(x, y, z, dim)[1]
                    + 2
                    * np.pi
                    * (
                        -((-1 + y) * y * (m_kappa_o + m_mu_o) * np.cos(np.pi * x))
                        + (-1 + x) * x * (m_kappa_o - m_mu_o) * np.cos(np.pi * y)
                    )
                    * np.sin(np.pi * z)
                    * grad_gamma_eval(x, y, z, dim)[0]
                ),
                gamma_eval(x, y, z, dim)
                * (
                    np.pi
                    * (
                        (-1 + 2 * x) * (m_kappa_o - m_mu_o) * np.cos(np.pi * z)
                        + np.pi
                        * (-1 + z)
                        * z
                        * (m_kappa_o + m_mu_o)
                        * np.sin(np.pi * x)
                    )
                    * np.sin(np.pi * y)
                    * gamma_eval(x, y, z, dim)
                    + np.pi
                    * np.sin(np.pi * x)
                    * (
                        (-1 + 2 * y) * (m_kappa_o - m_mu_o) * np.cos(np.pi * z)
                        + np.pi
                        * (-1 + z)
                        * z
                        * (m_kappa_o + m_mu_o)
                        * np.sin(np.pi * y)
                    )
                    * gamma_eval(x, y, z, dim)
                    - (
                        2
                        * (m_lambda_o + 2 * m_mu_o)
                        * np.sin(np.pi * x)
                        * np.sin(np.pi * y)
                        + np.pi
                        * m_lambda_o
                        * np.cos(np.pi * z)
                        * (
                            (-1 + 2 * y) * np.sin(np.pi * x)
                            + (-1 + 2 * x) * np.sin(np.pi * y)
                        )
                    )
                    * gamma_eval(x, y, z, dim)
                    - 2
                    * (
                        (-1 + 2 * x)
                        * m_lambda_o
                        * np.sin(np.pi * y)
                        * np.sin(np.pi * z)
                        + np.sin(np.pi * x)
                        * (
                            (-1 + 2 * z) * (m_lambda_o + 2 * m_mu_o) * np.sin(np.pi * y)
                            + (-1 + 2 * y) * m_lambda_o * np.sin(np.pi * z)
                        )
                    )
                    * grad_gamma_eval(x, y, z, dim)[2]
                    + 2
                    * np.pi
                    * (
                        -((-1 + z) * z * (m_kappa_o + m_mu_o) * np.cos(np.pi * y))
                        + (-1 + y) * y * (m_kappa_o - m_mu_o) * np.cos(np.pi * z)
                    )
                    * np.sin(np.pi * x)
                    * grad_gamma_eval(x, y, z, dim)[1]
                    + 2
                    * np.pi
                    * (
                        -((-1 + z) * z * (m_kappa_o + m_mu_o) * np.cos(np.pi * x))
                        + (-1 + x) * x * (m_kappa_o - m_mu_o) * np.cos(np.pi * z)
                    )
                    * np.sin(np.pi * y)
                    * grad_gamma_eval(x, y, z, dim)[0]
                ),
            ]
        )


def get_material_functions(material_data, dim: int = 2):

    m_lambda_s = material_data["lambda_s"]
    m_mu_s = material_data["mu_s"]
    m_kappa_s = material_data["kappa_s"]

    m_lambda_o = material_data["lambda_o"]
    m_mu_o = material_data["mu_o"]
    m_kappa_o = material_data["kappa_o"]

    def f_lambda_s(x, y, z):
        return m_lambda_s * np.ones_like(x)

    def f_mu_s(x, y, z):
        return m_mu_s * np.ones_like(x)

    def f_kappa_s(x, y, z):
        return m_kappa_s * np.ones_like(x)

    def f_lambda_o(x, y, z):
        return m_lambda_o * np.ones_like(x)

    def f_mu_o(x, y, z):
        return m_mu_o * np.ones_like(x)

    def f_kappa_o(x, y, z):
        return m_kappa_o * np.ones_like(x)

    m_functions = {
        "lambda_s": f_lambda_s,
        "mu_s": f_mu_s,
        "kappa_s": f_kappa_s,
        "lambda_o": f_lambda_o,
        "mu_o": f_mu_o,
        "kappa_o": f_kappa_o,
        "l": gamma_s(dim),
        "grad_l": grad_gamma_s(dim),
    }

    return m_functions
