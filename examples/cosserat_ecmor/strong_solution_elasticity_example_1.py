import numpy as np


def displacement(m_lambda, m_mu, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                4*np.pi*np.cos(2*np.pi*y)*(np.sin(2*np.pi*x)**2)*np.sin(2*np.pi*y),
                -4*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*x)*(np.sin(2*np.pi*y)**2),
            ]
        )
    else:
        raise ValueError("Dimension not implemented")

def rotation(m_lambda, m_mu, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                -8 * (np.pi ** 2) * (np.cos(2 * np.pi * y) ** 2) * (
                            np.sin(2 * np.pi * x) ** 2) - 8 * (np.pi ** 2) * (
                            np.cos(2 * np.pi * x) ** 2) * (np.sin(2 * np.pi * y) ** 2) +
                16 * (np.pi ** 2) * (np.sin(2 * np.pi * x) ** 2) * (
                            np.sin(2 * np.pi * y) ** 2),
            ]
        )
    else:
        raise ValueError("Dimension not implemented")


def stress(m_lambda, m_mu, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                [
                    8*(np.pi**2)*m_mu*np.sin(4*np.pi*x)*np.sin(4*np.pi*y),
                    4*(np.pi**2)*m_mu*(2*np.cos(4*np.pi*x) - 3*np.cos(4*np.pi*(x - y)) + 4*np.cos(4*np.pi*y) - 3*np.cos(4*np.pi*(x + y))),
                ],
                [
                    4*(np.pi**2)*m_mu*(-4*np.cos(4*np.pi*x) + 3*np.cos(4*np.pi*(x - y)) - 2*np.cos(4*np.pi*y) + 3*np.cos(4*np.pi*(x + y))),
                    -8*(np.pi**2)*m_mu*np.sin(4*np.pi*x)*np.sin(4*np.pi*y),
                ],
            ]
        )
    else:
        raise ValueError("Dimension not implemented")

def rhs(m_lambda, m_mu, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                -16*(np.pi**3)*m_mu*(3*np.sin(4*np.pi*(x - y)) - 2*(-2 + np.cos(4*np.pi*x))*np.sin(4*np.pi*y) - 3*np.sin(4*np.pi*(x + y))),
                -16*(np.pi**3)*m_mu*(2*(-2 + np.cos(4*np.pi*y))*np.sin(4*np.pi*x) + 3*(np.sin(4*np.pi*(x - y)) + np.sin(4*np.pi*(x + y)))),
                24*(np.pi**2)*m_mu*(np.cos(4*np.pi*x) - np.cos(4*np.pi*(x - y)) + np.cos(4*np.pi*y) - np.cos(4*np.pi*(x + y))),
            ]
        )
    else:
        raise ValueError("Dimension not implemented")

def stress_divergence(m_lambda, m_mu, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                -16*(np.pi**3)*m_mu*(3*np.sin(4*np.pi*(x - y)) - 2*(-2 + np.cos(4*np.pi*x))*np.sin(4*np.pi*y) - 3*np.sin(4*np.pi*(x + y))),
                -16*(np.pi**3)*m_mu*(2*(-2 + np.cos(4*np.pi*y))*np.sin(4*np.pi*x) + 3*(np.sin(4*np.pi*(x - y)) + np.sin(4*np.pi*(x + y)))),
            ]
        )
    else:
        raise ValueError("Dimension not implemented")



# def displacement(m_lambda, m_mu, dim: int = 2):
#     if dim == 2:
#         return lambda x, y, z: np.array(
#             [
#                 np.sin(2*np.pi*x),
#                 np.sin(2*np.pi*y),
#             ]
#         )
#     else:
#         raise ValueError("Dimension not implemented")
#
# def rotation(m_lambda, m_mu, dim: int = 2):
#     if dim == 2:
#         return lambda x, y, z: np.array(
#             [
#                 np.sin(2*np.pi*x)*np.sin(2*np.pi*y),
#             ]
#         )
#     else:
#         raise ValueError("Dimension not implemented")
#
#
# def stress(m_lambda, m_mu, dim: int = 2):
#     if dim == 2:
#         return lambda x, y, z: np.array(
#             [
#                 [
#                     2*np.pi*((m_lambda + 2*m_mu)*np.cos(2*np.pi*x) + m_lambda*np.cos(2*np.pi*y)),
#                     -2*m_mu*np.sin(2*np.pi*x)*np.sin(2*np.pi*y),
#                 ],
#                 [
#                     2*m_mu*np.sin(2*np.pi*x)*np.sin(2*np.pi*y),
#                     2*np.pi*(m_lambda*np.cos(2*np.pi*x) + (m_lambda + 2*m_mu)*np.cos(2*np.pi*y)),
#                 ],
#             ]
#         )
#     else:
#         raise ValueError("Dimension not implemented")
#
# def rhs(m_lambda, m_mu, dim: int = 2):
#     if dim == 2:
#         return lambda x, y, z: np.array(
#             [
#                 -4*np.pi*(np.pi*(m_lambda + 2*m_mu) + m_mu*np.cos(2*np.pi*y))*np.sin(2*np.pi*x),
#                 4*np.pi*(-(np.pi*(m_lambda + 2*m_mu)) + m_mu*np.cos(2*np.pi*x))*np.sin(2*np.pi*y),
#                 -4 * m_mu * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y),
#             ]
#         )
#     else:
#         raise ValueError("Dimension not implemented")
#
# def stress_divergence(m_lambda, m_mu, dim: int = 2):
#     if dim == 2:
#         return lambda x, y, z: np.array(
#             [
#                 -4*np.pi*(np.pi*(m_lambda + 2*m_mu) + m_mu*np.cos(2*np.pi*y))*np.sin(2*np.pi*x),
#                 4*np.pi*(-(np.pi*(m_lambda + 2*m_mu)) + m_mu*np.cos(2*np.pi*x))*np.sin(2*np.pi*y),
#             ]
#         )
#     else:
#         raise ValueError("Dimension not implemented")

