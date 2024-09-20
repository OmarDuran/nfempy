import numpy as np


def displacement(material_data, dim: int = 2):
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


def displacement_gradient(material_data, dim: int = 2):
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
                    2*(np.pi**2)*m_mu_s*np.cos(np.pi*y)*np.sin(2*np.pi*x),
                    ((-((np.pi**2)*(m_kappa_s + m_mu_s)) + (np.pi**2)*(5*m_kappa_s - 3*m_mu_s)*np.cos(2*np.pi*x) - 4*m_kappa_s*np.sin(np.pi*x))*np.sin(np.pi*y))/2.,
                ],
                [
                    -0.5*(((np.pi**2)*(-m_kappa_s + m_mu_s) + (np.pi**2)*(5*m_kappa_s + 3*m_mu_s)*np.cos(2*np.pi*x) - 4*m_kappa_s*np.sin(np.pi*x))*np.sin(np.pi*y)),
                    -2*(np.pi**2)*m_mu_s*np.cos(np.pi*y)*np.sin(2*np.pi*x),
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


def couple_stress(material_data, dim: int = 2):
    m_lambda_o = material_data["lambda_o"]
    m_mu_o = material_data["mu_o"]
    m_kappa_o = material_data["kappa_o"]

    m_l = material_data["l"]
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                [
                    m_l*np.pi*(m_kappa_o + m_mu_o)*np.cos(np.pi*x)*np.sin(np.pi*y),
                    m_l*np.pi*(m_kappa_o + m_mu_o)*np.cos(np.pi*y)*np.sin(np.pi*x),
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


def rhs(material_data, dim: int = 2):
    m_lambda_s = material_data["lambda_s"]
    m_mu_s = material_data["mu_s"]
    m_kappa_s = material_data["kappa_s"]

    m_lambda_o = material_data["lambda_o"]
    m_mu_o = material_data["mu_o"]
    m_kappa_o = material_data["kappa_o"]

    m_l = material_data["l"]
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                (np.pi*np.cos(np.pi*y)*(-((np.pi**2)*(m_kappa_s + m_mu_s)) + 5*(np.pi**2)*(m_kappa_s + m_mu_s)*np.cos(2*np.pi*x) - 4*m_kappa_s*np.sin(np.pi*x)))/2.,
                2*np.pi*np.cos(np.pi*x)*(m_kappa_s + 5*(np.pi**2)*(m_kappa_s + m_mu_s)*np.sin(np.pi*x))*np.sin(np.pi*y),
                (-((np.pi**2)*m_kappa_s) + 5*(np.pi**2)*m_kappa_s*np.cos(2*np.pi*x) - 2*(2*m_kappa_s + m_l*(np.pi**2)*(m_kappa_o + m_mu_o))*np.sin(np.pi*x))*np.sin(np.pi*y),
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


def stress_divergence(material_data, dim: int = 2):
    m_lambda_s = material_data["lambda_s"]
    m_mu_s = material_data["mu_s"]
    m_kappa_s = material_data["kappa_s"]
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                (np.pi * np.cos(np.pi * y) * (
                            -((np.pi ** 2) * (m_kappa_s + m_mu_s)) + 5 * (np.pi ** 2) * (
                                m_kappa_s + m_mu_s) * np.cos(
                        2 * np.pi * x) - 4 * m_kappa_s * np.sin(np.pi * x))) / 2.,
                2 * np.pi * np.cos(np.pi * x) * (
                            m_kappa_s + 5 * (np.pi ** 2) * (m_kappa_s + m_mu_s) * np.sin(
                        np.pi * x)) * np.sin(np.pi * y),
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


def couple_stress_divergence(material_data, dim: int = 2):

    m_lambda_o = material_data["lambda_o"]
    m_mu_o = material_data["mu_o"]
    m_kappa_o = material_data["kappa_o"]

    m_l = material_data["l"]
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                -2*m_l*(np.pi**2)*(m_kappa_o + m_mu_o)*np.sin(np.pi*x)*np.sin(np.pi*y),
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

def get_material_functions(material_data, dim: int = 2):

    m_lambda_s = material_data["lambda_s"]
    m_mu_s = material_data["mu_s"]
    m_kappa_s = material_data["kappa_s"]

    m_lambda_o = material_data["lambda_o"]
    m_mu_o = material_data["mu_o"]
    m_kappa_o = material_data["kappa_o"]

    m_l = material_data["l"]

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

    def f_l(x, y, z):
        return m_l * np.ones_like(x)

    if dim == 2:

        def f_grad_l(x, y, z):
            d_l_x = 0.0 * x
            d_l_y = 0.0 * y
            return np.array([d_l_x, d_l_y])

    elif dim == 3:

        def f_grad_l(x, y, z):
            d_l_x = 0.0 * x
            d_l_y = 0.0 * y
            d_l_z = 0.0 * z
            return np.array([d_l_x, d_l_y, d_l_z])

    else:
        raise ValueError("Dimension not implemented: ", dim)

    m_functions = {
        "lambda_s": f_lambda_s,
        "mu_s": f_mu_s,
        "kappa_s": f_kappa_s,
        "lambda_o": f_lambda_o,
        "mu_o": f_mu_o,
        "kappa_o": f_kappa_o,
        "l": f_l,
        "grad_l": f_grad_l,
    }

    return m_functions