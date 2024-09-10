import numpy as np

# lifting functions
def _xs(x, y, z, ):
    vals = np.ones_like(x) + np.exp(np.pi * (x-1.0))
    return vals
def _dxsdx(x, y, z, ):
    vals = np.pi * np.exp(np.pi * (x-1.0))
    return vals
def _dxsdx2(x, y, z, ):
    vals = np.pi * np.pi * np.exp(np.pi * (x-1.0))
    return vals

def _ys(x, y, z, ):
    vals = np.piecewise(y, [y < 0.0, y >= 0.0], [lambda y: 1.0-y, lambda y: (1.0+y)**2])
    return vals
def _dysdy(x, y, z, ):
    vals = np.piecewise(y, [y < 0.0, y >= 0.0], [lambda y: -1.0, lambda y: 2.0*(1.0+y)])
    return vals
def _dysdy2(x, y, z, ):
    vals = np.piecewise(y, [y < 0.0, y >= 0.0], [lambda y: 0.0, lambda y: 2.0])
    return vals

def f_porosity(x, y, z, m_data, co_dim):
    m_rho_1 = m_data['rho_1']
    m_rho_2 = m_data['rho_2']
    if co_dim == 0:
        val = (m_rho_1 * np.ones_like(x) + m_rho_2 * np.array(x**2)) * _ys(x, y, z)
    elif co_dim == 1:
        val = m_rho_1 * np.ones_like(x) + m_rho_2 * np.array(x**2)
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")
    return val

def f_grad_porosity(x, y, z, m_data, co_dim):
    m_rho_2 = m_data['rho_2']
    if co_dim == 0:
        val = np.array([2.0 * x * m_rho_2 * _ys(x,y,z), f_porosity(x, y, z, m_data, co_dim) * _dysdy(x,y,z), z * 0.0])
    elif co_dim == 1:
        val = np.array([2.0 * x * m_rho_2, y * 0.0, z * 0.0])
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")
    return val

def f_kappa(x, y, z, m_data, co_dim):
    m_kappa = m_data['kappa']
    val = m_kappa * f_porosity(x, y, z, m_data, co_dim) ** 2
    return val

def f_d_phi(x, y, z, m_data, co_dim):
    m_mu = m_data['mu']
    return np.sqrt(f_kappa(x, y, z, m_data, co_dim) / m_mu)

def f_grad_d_phi(x, y, z, m_data, co_dim):
    m_mu = m_data['mu']
    scalar_part = f_porosity(x, y, z, m_data, co_dim) / (
        m_mu * f_d_phi(x, y, z, m_data, co_dim)
    )
    vector_part = f_grad_porosity(x, y, z, m_data, co_dim)
    return scalar_part * vector_part

# The construction stem from:

# The pressure in the fracture;
def pf(x, y, z, m_data):
    bubble = 0.5 * (np.ones_like(x) - x) * 0.5 * (np.ones_like(x) + x)
    return bubble * (x**2) * np.sin(2.0 * np.pi * x)

# The pressure gradient;
def dpfdx(x, y, z, m_data):
    term_1 = 0.25 * (-np.ones_like(x) - x) +  0.25 * (np.ones_like(x) - x)
    term_2 = -2.0 * np.pi * (x**2) * np.cos(2.0 * np.pi * x)
    term_3 = -2.0 * x * np.sin(2.0 * np.pi * x)
    return term_1 + term_2 + term_3

# The pressure laplacian.
def dpfdx2(x, y, z, m_data):
    term_1 = -0.5 * np.ones_like(x)
    term_2 = -8.0 * np.pi * x * np.cos(2.0 * np.pi * x)
    term_3 = -2.0 * np.sin(2.0 * np.pi * x)
    term_4 = +4.0 * (np.pi**2) * (x**2) * np.sin(2.0 * np.pi * x)
    return term_1 + term_2 + term_3 + term_4

def p_exact(x, y, z, m_data, co_dim):
    if co_dim == 0:
        val = _xs(x,y,z) * _ys(x,y,z) * pf(x, y, z, m_data)
    elif co_dim == 1:
        val = pf(x, y, z, m_data)
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")
    return val

def u_exact(x, y, z, m_data, co_dim):
    if co_dim == 0:
        val = -np.array([
            _xs(x, y, z) * _ys(x, y, z) * dpfdx(x, y, z, m_data) + _dxsdx(x, y, z) * _ys(x, y, z) * pf(x, y, z, m_data),
            _xs(x, y, z) * _dysdy(x, y, z) * pf(x, y, z, m_data),
            z * 0.0
            ])
        val *= f_d_phi(x, y, z, m_data, co_dim)**2
    elif co_dim == 1:
        val = -dpfdx(x, y, z, m_data)
        val *= f_d_phi(x, y, z, m_data, co_dim)**2
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")
    return val


def q_exact(x, y, z, m_par, dim):
    if dim == 1:
        mask = x < 0.0
        val = np.empty_like([x * 0.0])
        val[:, mask] = np.zeros_like([x[mask] * 0.0])
        val[:, ~mask] = np.sqrt(
            f_porosity(x[~mask], y[~mask], z[~mask], m_par, dim)
        ) * p_exact(x[~mask], y[~mask], z[~mask], m_par, dim)
    elif dim == 2:
        gamma = m_par
        mask = np.logical_or(x <= -3 / 4, y <= -3 / 4)
        val = np.empty_like([x * 0.0])
        val[:, mask] = np.zeros_like([x[mask] * 0.0])
        val[:, ~mask] = np.array(
            [
                np.sqrt(
                    ((0.75 + x[~mask]) ** gamma) * ((0.75 + y[~mask]) ** (2 * gamma))
                )
                * np.cos(6 * x[~mask] * (y[~mask] ** 2))
            ]
        )
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")
    return val


def v_exact(x, y, z, m_par, m_mu, dim):
    if dim == 1:
        mask = x < 0.0
        val = np.empty_like([x * 0.0])
        val[:, mask] = np.zeros_like([x[mask] * 0.0])
        val[:, ~mask] = u_exact(x[~mask], y[~mask], z[~mask], m_par, dim) / f_d_phi(
            x[~mask], y[~mask], z[~mask], m_par, m_mu, dim
        )
    elif dim == 2:
        gamma = m_par
        mask = np.logical_or(x <= -3 / 4, y <= -3 / 4)
        val = np.empty_like([[x * 0.0, y * 0.0]])
        val[:, :, mask] = np.zeros_like([[x[mask] * 0.0, y[mask] * 0.0]])
        val[:, :, ~mask] = np.array(
            [
                [
                    6
                    * (y[~mask] ** 2)
                    * np.sqrt(
                        ((0.75 + x[~mask]) ** (2 * gamma))
                        * ((0.75 + y[~mask]) ** (4 * gamma))
                    )
                    * np.sin(6 * x[~mask] * (y[~mask] ** 2)),
                    12
                    * x[~mask]
                    * y[~mask]
                    * np.sqrt(
                        ((0.75 + x[~mask]) ** (2 * gamma))
                        * ((0.75 + y[~mask]) ** (4 * gamma))
                    )
                    * np.sin(6 * x[~mask] * (y[~mask] ** 2)),
                ]
            ]
        )
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")
    return val


def f_rhs(x, y, z, m_par, dim):
    if dim == 1:
        beta = m_par
        mask = x < 0.0
        val = np.empty_like([[x * 0.0]])
        val[:, :, mask] = np.zeros_like([x[mask] * 0.0])
        val[:, :, ~mask] = np.array(
            [
                [
                    (x[~mask] ** beta)
                    * np.sqrt(f_porosity(x[~mask], y[~mask], z[~mask], m_par, dim))
                ]
            ]
        )
    elif dim == 2:
        gamma = m_par
        mask = np.logical_or(x <= -3 / 4, y <= -3 / 4)
        val = np.empty_like([[x * 0.0]])
        val[:, :, mask] = np.zeros_like([x[mask] * 0.0])

        val[:, :, ~mask] = np.array(
            [
                [
                    np.sqrt(f_porosity(x[~mask], y[~mask], z[~mask], m_par, dim))
                    * (
                        (
                            1
                            + 36
                            * ((0.75 + x[~mask]) ** gamma)
                            * (y[~mask] ** 2)
                            * ((0.75 + y[~mask]) ** (2 * gamma))
                            * (4 * (x[~mask] ** 2) + (y[~mask] ** 2))
                        )
                        * np.cos(6 * x[~mask] * (y[~mask] ** 2))
                        + (
                            12
                            * ((0.75 + x[~mask]) ** gamma)
                            * ((0.75 + y[~mask]) ** (2 * gamma))
                            * (
                                x[~mask] * (3 + 4 * x[~mask]) * (3 + 4 * y[~mask])
                                + 4
                                * y[~mask]
                                * (
                                    4 * x[~mask] * (3 + 4 * x[~mask])
                                    + y[~mask] * (3 + 4 * y[~mask])
                                )
                                * gamma
                            )
                            * np.sin(6 * x[~mask] * (y[~mask] ** 2))
                        )
                        / ((3 + 4 * x[~mask]) * (3 + 4 * y[~mask]))
                    )
                ]
            ]
        )
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")
    return val

def grad_p_exact(x, y, z, m_par, dim):
    if dim == 1:
        beta = m_par
        return np.where(
            x < 0.0,
            np.array([x * 0.0]),
            np.array(
                [
                    (((x ** (np.sqrt(13) / 2.0)) - (x ** (1.5 + beta))) * beta) / (
                            (x ** 2.5) * (-1 + beta * (3 + beta))),
                ]
            ),
        )
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")

def laplacian_p_exact(x, y, z, m_par, dim):
    if dim == 1:
        beta = m_par
        return np.where(
            x < 0.0,
            np.array([x * 0.0]),
            np.array(
                [
                    (((-5 + np.sqrt(13))*(x**(np.sqrt(13)/2.0)) - 2*(x**(1.5 + beta))*(-1 + beta))*beta)/ (2.*(x**3.5)*(-1 + beta*(3 + beta))),
                ]
            ),
        )
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")

def test_degeneracy(m_par, m_mu, dim):
    x = np.random.uniform(-1.0, +1.0, (10, 3))
    try:
        phi = f_porosity(x[:, 0], x[:, 1], x[:, 2], m_par, dim)
        grad_phi = f_grad_porosity(x[:, 0], x[:, 1], x[:, 2], m_par, dim)
        d_phi = f_d_phi(x[:, 0], x[:, 1], x[:, 2], m_par, m_mu, dim)
        grad_d_phi = f_grad_d_phi(x[:, 0], x[:, 1], x[:, 2], m_par, m_mu, dim)
        kappa = f_kappa(x[:, 0], x[:, 1], x[:, 2], m_par, dim)

        # exact functions
        u = u_exact(x[:, 0], x[:, 1], x[:, 2], m_par, dim)
        p = p_exact(x[:, 0], x[:, 1], x[:, 2], m_par, dim)
        v = v_exact(x[:, 0], x[:, 1], x[:, 2], m_par, m_mu, dim)
        q = q_exact(x[:, 0], x[:, 1], x[:, 2], m_par, dim)
        rhs = f_rhs(x[:, 0], x[:, 1], x[:, 2], m_par, dim)

        # exact functions for md cases
        grad_p = grad_p_exact(x[:, 0], x[:, 1], x[:, 2], m_par, dim)
        laplacian_p = laplacian_p_exact(x[:, 0], x[:, 1], x[:, 2], m_par, dim)
    except Exception:
        return False

    return True