import numpy as np


def displacement(m_lambda, m_mu, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                np.sin(2*np.pi*x),
                np.sin(2*np.pi*y),
            ]
        )
    else:
        raise ValueError("Dimension not implemented")

def rotation(m_lambda, m_mu, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                np.sin(2*np.pi*x)*np.sin(2*np.pi*y),
            ]
        )
    else:
        raise ValueError("Dimension not implemented")


def stress(m_lambda, m_mu, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                [
                    2*np.pi*((m_lambda + 2*m_mu)*np.cos(2*np.pi*x) + m_lambda*np.cos(2*np.pi*y)),
                    -2*m_mu*np.sin(2*np.pi*x)*np.sin(2*np.pi*y),
                ],
                [
                    2*m_mu*np.sin(2*np.pi*x)*np.sin(2*np.pi*y),
                    2*np.pi*(m_lambda*np.cos(2*np.pi*x) + (m_lambda + 2*m_mu)*np.cos(2*np.pi*y)),
                ],
            ]
        )
    else:
        raise ValueError("Dimension not implemented")

def rhs(m_lambda, m_mu, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                -4*np.pi*(np.pi*(m_lambda + 2*m_mu) + m_mu*np.cos(2*np.pi*y))*np.sin(2*np.pi*x),
                4*np.pi*(-(np.pi*(m_lambda + 2*m_mu)) + m_mu*np.cos(2*np.pi*x))*np.sin(2*np.pi*y),
                4 * m_mu * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y),
            ]
        )
    else:
        raise ValueError("Dimension not implemented")

def stress_divergence(m_lambda, m_mu, dim: int = 2):
    if dim == 2:
        return lambda x, y, z: np.array(
            [
                -4*np.pi*(np.pi*(m_lambda + 2*m_mu) + m_mu*np.cos(2*np.pi*y))*np.sin(2*np.pi*x),
                4*np.pi*(-(np.pi*(m_lambda + 2*m_mu)) + m_mu*np.cos(2*np.pi*x))*np.sin(2*np.pi*y),
            ]
        )
    else:
        raise ValueError("Dimension not implemented")

