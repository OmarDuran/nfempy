import auto_diff as ad
import basix
import numpy as np
from auto_diff.vecvalder import VecValDer
from basix import CellType

from basis.element_data import ElementData
from basis.parametric_transformation import transform_lower_to_higher
from topology.topological_queries import find_higher_dimension_neighs
from weak_forms.weak_from import WeakForm

class OdenDualMixedWeakForm(WeakForm):
   def evaluate_form(self, element_index, alpha):
    iel = element_index
    if self.space is None or self.functions is None:
        raise ValueError

    q_space = self.space.discrete_spaces["q"]
    u_space = self.space.discrete_spaces["u"]

    f_rhs = self.functions["rhs"]


    q_components = q_space.n_comp
    u_components = u_space.n_comp

    q_data: ElementData = q_space.elements[iel].data
    u_data: ElementData = u_space.elements[iel].data

    cell = q_data.cell
    dim = q_data.dimension
    points, weights = self.space.quadrature
    x, jac, det_jac, inv_jac = q_space.elements[iel].evaluate_mapping(points)

    # basis
    q_phi_tab = q_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
    u_phi_tab = u_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
    #
    n_q_phi = q_phi_tab.shape[2]
    n_u_phi = u_phi_tab.shape[2]
    n_q_dof = n_q_phi * q_components
    n_u_dof = n_u_phi * u_components
    n_dof = n_q_dof + n_u_dof
    js = (n_dof, n_dof)
    rs = n_dof
    j_el = np.zeros(js)
    r_el = np.zeros(rs)
    alpha = np.zeros(n_dof)

    #R.H.S
    f_val_star = f_rhs(x[:, 0], x[:, 1], x[:, 2])
    phi_s_star = det_jac * weights * u_phi_tab[0, :, :, 0].T

    # constant directors
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])
    with ad.AutoDiff(alpha) as alpha:
        el_form = np.zeros(n_dof)
        for c in range(u_components):
            b = c + n_q_dof
            e = b + n_u_dof
            el_form[b:e:u_components] -= phi_s_star @ f_val_star[c].T

            for i, omega in enumerate(weights):
                xv = x[i]
                qh = alpha[:, 0:n_q_dof:1] @ q_phi_tab[0, i, :, 0:dim]
                #qh *= 1.0 / f_kappa(xv[0], xv[1], xv[2])
                uh = alpha[:, n_q_dof:n_dof:1] @ u_phi_tab[0, i, :, 0:dim]
                grad_qh = q_phi_tab[1: q_phi_tab.shape[0] + 1, i, :, 0:dim]
                div_vh = np.array(
                    [[np.trace(grad_qh[:, j, :]) / det_jac[i] for j in range(n_q_dof)]]
                )
                div_qh = alpha[:, 0:n_q_dof:1] @ div_vh.T

                equation_1 = (qh @ q_phi_tab[0, i, :, 0:dim].T) - (uh @ div_vh)
                equation_2 = div_qh @ u_phi_tab[0, i, :, 0:dim].T + uh @ u_phi_tab[0, i, :, 0:dim].T
                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, 0:n_q_dof:1] = equation_1
                multiphysic_integrand[:, n_q_dof:n_dof:1] = equation_2
                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        return r_el, j_el
