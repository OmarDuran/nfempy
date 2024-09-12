import numpy as np
from functools import partial


import numpy as np


# lifting functions for pressure
def _xs(
    x,
    y,
    z,
    m_data,
):
    vals = np.ones_like(x) + np.exp(np.pi * (x - 1.0))
    return vals


def _dxsdx(
    x,
    y,
    z,
    m_data,
):
    vals = np.pi * np.exp(np.pi * (x - 1.0))
    return vals


def _dxsdx2(
    x,
    y,
    z,
    m_data,
):
    vals = np.pi * np.pi * np.exp(np.pi * (x - 1.0))
    return vals


def _ys(
    x,
    y,
    z,
    m_data,
):
    m_xi = m_data["xi"]
    m_eta = m_data["eta"]
    vals = np.piecewise(
        y,
        [y < 0.0, y >= 0.0],
        [lambda y: (1.0 - y) ** m_xi, lambda y: (1.0 + y) ** m_eta],
    )
    return vals


def _dysdy(
    x,
    y,
    z,
    m_data,
):
    m_xi = m_data["xi"]
    m_eta = m_data["eta"]
    vals = np.piecewise(
        y,
        [y < 0.0, y >= 0.0],
        [
            lambda y: -m_xi * (1.0 - y) ** (m_xi - 1.0),
            lambda y: +m_eta * (1.0 + y) ** (m_eta - 1.0),
        ],
    )
    return vals


def _dysdy2(
    x,
    y,
    z,
    m_data,
):
    m_xi = m_data["xi"]
    m_eta = m_data["eta"]
    vals = np.piecewise(
        y,
        [y < 0.0, y >= 0.0],
        [
            lambda y: m_xi * (m_xi - 1.0) * (1.0 - y) ** (m_xi - 2.0),
            lambda y: m_eta * (m_eta - 1.0) * (1.0 + y) ** (m_eta - 2.0),
        ],
    )
    return vals


# lifting functions for porosity


def phi_ys(
    x,
    y,
    z,
    m_data,
):
    m_chi = m_data["chi"]
    vals = np.piecewise(
        y,
        [y < 0.0, y >= 0.0],
        [lambda y: (1.0 - y) ** m_chi, lambda y: (1.0 + y) ** m_chi],
    )
    return vals


def phi_dysdy(
    x,
    y,
    z,
    m_data,
):
    m_chi = m_data["chi"]
    vals = np.piecewise(
        y,
        [y < 0.0, y >= 0.0],
        [
            lambda y: -m_chi * (1.0 - y) ** (m_chi - 1.0),
            lambda y: +m_chi * (1.0 + y) ** (m_chi - 1.0),
        ],
    )
    return vals


def phi_dysdy2(
    x,
    y,
    z,
    m_data,
):
    m_chi = m_data["chi"]
    vals = np.piecewise(
        y,
        [y < 0.0, y >= 0.0],
        [
            lambda y: m_chi * (m_chi - 1.0) * (1.0 - y) ** (m_chi - 2.0),
            lambda y: m_chi * (m_chi - 1.0) * (1.0 + y) ** (m_chi - 2.0),
        ],
    )
    return vals


def f_porosity(x, y, z, m_data, co_dim):
    m_rho_1 = m_data["rho_1"]
    m_rho_2 = m_data["rho_2"]
    if co_dim == 0:
        val = (m_rho_1 * np.ones_like(x) + m_rho_2 * np.array(x**2)) * phi_ys(
            x, y, z, m_data
        )
    elif co_dim == 1:
        val = m_rho_1 * np.ones_like(x) + m_rho_2 * np.array(x**2)
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")
    return val


def f_grad_porosity(x, y, z, m_data, co_dim):
    m_rho_2 = m_data["rho_2"]
    if co_dim == 0:
        val = np.array(
            [
                2.0 * x * m_rho_2 * phi_ys(x, y, z, m_data),
                f_porosity(x, y, z, m_data, co_dim=1) * phi_dysdy(x, y, z, m_data),
                z * 0.0,
            ]
        )
    elif co_dim == 1:
        val = np.array([2.0 * x * m_rho_2, y * 0.0, z * 0.0])
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")
    return val


def f_kappa(x, y, z, m_data, co_dim):
    if co_dim == 0:
        m_kappa_c0 = m_data["kappa_c0"]
        val = m_kappa_c0 * f_porosity(x, y, z, m_data, co_dim) ** 2
    elif co_dim == 1:
        m_kappa_c1 = m_data["kappa_c1"]
        val = m_kappa_c1 * f_porosity(x, y, z, m_data, co_dim) ** 2
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")
    return val


def f_d_phi(x, y, z, m_data, co_dim):
    m_mu = m_data["mu"]
    if co_dim == 0:
        val = np.sqrt(f_kappa(x, y, z, m_data, co_dim) / m_mu)
    elif co_dim == 1:
        m_delta = m_data["delta"]
        val = np.sqrt(m_delta * f_kappa(x, y, z, m_data, co_dim) / m_mu)
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")

    return val


def f_grad_d_phi(x, y, z, m_data, co_dim):
    m_mu = m_data["mu"]
    if co_dim == 0:
        m_kappa_c0 = m_data["kappa_c0"]
        scalar_part = (
            m_kappa_c0
            * f_porosity(x, y, z, m_data, co_dim)
            / (m_mu * f_d_phi(x, y, z, m_data, co_dim))
        )
        vector_part = f_grad_porosity(x, y, z, m_data, co_dim)
    elif co_dim == 1:
        m_kappa_c1 = m_data["kappa_c1"]
        m_delta = m_data["delta"]
        scalar_part = (
            m_delta
            * m_kappa_c1
            * f_porosity(x, y, z, m_data, co_dim)
            / (m_mu * f_d_phi(x, y, z, m_data, co_dim))
        )
        vector_part = f_grad_porosity(x, y, z, m_data, co_dim)
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")
    return scalar_part * vector_part


# The construction stem from:
# The pressure in the fracture;
def pf(x, y, z, m_data):
    bubble = 0.5 * (np.ones_like(x) - x) * 0.5 * (np.ones_like(x) + x)
    return bubble - (x**2) * np.sin(2.0 * np.pi * x)


# The pressure gradient;
def dpfdx(x, y, z, m_data):
    term_1 = 0.25 * (-np.ones_like(x) - x) + 0.25 * (np.ones_like(x) - x)
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
        val = _xs(x, y, z, m_data) * _ys(x, y, z, m_data) * pf(x, y, z, m_data)
    elif co_dim == 1:
        val = pf(x, y, z, m_data)
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")
    return np.array([val])


def u_exact(x, y, z, m_data, co_dim):
    if co_dim == 0:
        val = -(f_d_phi(x, y, z, m_data, co_dim) ** 2) * np.array(
            [
                [
                    _xs(x, y, z, m_data) * _ys(x, y, z, m_data) * dpfdx(x, y, z, m_data)
                    + _dxsdx(x, y, z, m_data)
                    * _ys(x, y, z, m_data)
                    * pf(x, y, z, m_data),
                    _xs(x, y, z, m_data)
                    * _dysdy(x, y, z, m_data)
                    * pf(x, y, z, m_data),
                ]
            ]
        )
    elif co_dim == 1:
        val = -(f_d_phi(x, y, z, m_data, co_dim) ** 2) * np.array(
            [dpfdx(x, y, z, m_data)]
        )
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")
    return val


def q_exact(x, y, z, m_data, co_dim):
    val = np.sqrt(f_porosity(x, y, z, m_data, co_dim)) * p_exact(
        x, y, z, m_data, co_dim
    )
    return val


def v_exact(x, y, z, m_data, co_dim):
    val = (1.0 / f_d_phi(x, y, z, m_data, co_dim)) * u_exact(x, y, z, m_data, co_dim)
    return val


def f_rhs(x, y, z, m_data, co_dim):
    if co_dim == 0:
        term_1 = -(f_d_phi(x, y, z, m_data, co_dim) ** 2) * (
            2.0
            * _ys(x, y, z, m_data)
            * dpfdx(x, y, z, m_data)
            * _dxsdx(x, y, z, m_data)
            + _xs(x, y, z, m_data) * _ys(x, y, z, m_data) * dpfdx2(x, y, z, m_data)
            + _dxsdx2(x, y, z, m_data) * _ys(x, y, z, m_data) * pf(x, y, z, m_data)
        )
        term_2 = (
            -(f_d_phi(x, y, z, m_data, co_dim) ** 2)
            * _xs(x, y, z, m_data)
            * _dysdy2(x, y, z, m_data)
            * pf(x, y, z, m_data)
            - 2.0
            * pf(x, y, z, m_data)
            * _xs(x, y, z, m_data)
            * f_d_phi(x, y, z, m_data, co_dim)
            * _dysdy(x, y, z, m_data)
            * f_grad_d_phi(x, y, z, m_data, co_dim)[1]
        )
        term_3 = (
            -2.0
            * f_d_phi(x, y, z, m_data, co_dim)
            * (
                _xs(x, y, z, m_data) * _ys(x, y, z, m_data) * dpfdx(x, y, z, m_data)
                + _dxsdx(x, y, z, m_data) * _ys(x, y, z, m_data) * pf(x, y, z, m_data)
            )
            * f_grad_d_phi(x, y, z, m_data, co_dim)[0]
        )
        div_u = term_1 + term_2 + term_3
        val = div_u + f_porosity(x, y, z, m_data, co_dim) * p_exact(
            x, y, z, m_data, co_dim
        )
        val *= 1.0 / np.sqrt(f_porosity(x, y, z, m_data, co_dim))
    elif co_dim == 1:
        div_u = (
            -2.0
            * f_d_phi(x, y, z, m_data, co_dim)
            * dpfdx(x, y, z, m_data)
            * f_grad_d_phi(x, y, z, m_data, co_dim)[0]
        )
        div_u += -(f_d_phi(x, y, z, m_data, co_dim) ** 2) * dpfdx2(x, y, z, m_data)
        val = div_u + f_porosity(x, y, z, m_data, co_dim) * p_exact(
            x, y, z, m_data, co_dim
        )
        val *= 1.0 / np.sqrt(f_porosity(x, y, z, m_data, co_dim))
        n_p = np.array([0.0, -1.0])
        n_n = np.array([0.0, 1.0])
        un_p = u_exact(x, (+1.0e-13) * np.ones_like(y), z, m_data, co_dim=0)[0].T @ n_p
        un_n = u_exact(x, (-1.0e-13) * np.ones_like(y), z, m_data, co_dim=0)[0].T @ n_n
        val += np.array([un_p + un_n]) / np.sqrt(f_porosity(x, y, z, m_data, co_dim))

    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")
    return np.array([val])


def test_evaluation(m_data, co_dim):
    x = np.random.uniform(-1.0, +1.0, (10, 3))
    try:
        phi = f_porosity(x[:, 0], x[:, 1], x[:, 2], m_data, co_dim)
        grad_phi = f_grad_porosity(x[:, 0], x[:, 1], x[:, 2], m_data, co_dim)
        d_phi = f_d_phi(x[:, 0], x[:, 1], x[:, 2], m_data, co_dim)
        grad_d_phi = f_grad_d_phi(x[:, 0], x[:, 1], x[:, 2], m_data, co_dim)
        kappa = f_kappa(x[:, 0], x[:, 1], x[:, 2], m_data, co_dim)

        # exact functions
        u = u_exact(x[:, 0], x[:, 1], x[:, 2], m_data, co_dim)
        p = p_exact(x[:, 0], x[:, 1], x[:, 2], m_data, co_dim)
        v = v_exact(x[:, 0], x[:, 1], x[:, 2], m_data, co_dim)
        q = q_exact(x[:, 0], x[:, 1], x[:, 2], m_data, co_dim)
        rhs = f_rhs(x[:, 0], x[:, 1], x[:, 2], m_data, co_dim)

    except Exception:
        return False

    return True


def get_exact_functions_by_co_dimension(co_dim, flux_name, potential_name, m_data):
    if co_dim not in [0, 1]:
        raise ValueError("Case not available.")
    exact_functions = {
        flux_name: partial(
            u_exact,
            m_data=m_data,
            co_dim=co_dim,
        ),
        potential_name: partial(
            p_exact,
            m_data=m_data,
            co_dim=co_dim,
        ),
    }
    return exact_functions


def get_scaled_exact_functions_by_co_dimension(
    co_dim, flux_name, potential_name, m_data
):
    if co_dim not in [0, 1]:
        raise ValueError("Case not available.")
    exact_functions = {
        flux_name: partial(
            v_exact,
            m_data=m_data,
            co_dim=co_dim,
        ),
        potential_name: partial(
            q_exact,
            m_data=m_data,
            co_dim=co_dim,
        ),
    }
    return exact_functions


def get_problem_functions_by_co_dimension(
    co_dim, rhs_name, porosity_name, d_phi_name, grad_d_phi_name, m_data
):
    if co_dim not in [0, 1]:
        raise ValueError("Case not available.")
    problem_functions = {
        rhs_name: partial(
            f_rhs,
            m_data=m_data,
            co_dim=co_dim,
        ),
        porosity_name: partial(
            f_porosity,
            m_data=m_data,
            co_dim=co_dim,
        ),
        d_phi_name: partial(
            f_d_phi,
            m_data=m_data,
            co_dim=co_dim,
        ),
        grad_d_phi_name: partial(
            f_grad_d_phi,
            m_data=m_data,
            co_dim=co_dim,
        ),
    }
    return problem_functions
