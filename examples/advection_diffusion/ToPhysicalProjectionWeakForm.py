import auto_diff as ad
import basix
import numpy as np
from auto_diff.vecvalder import VecValDer
from basix import CellType

from basis.element_data import ElementData
from basis.parametric_transformation import transform_lower_to_higher
from geometry.compute_normal import normal
from topology.topological_queries import find_higher_dimension_neighs
from weak_forms.weak_from import WeakForm


class ToPhysicalProjectionWeakForm(WeakForm):
    def evaluate_form(self, element_index, alpha, alpha_scaled):
        iel = element_index
        if self.space is None or self.functions is None:
            raise ValueError

        v_space = self.space.discrete_spaces["v"]
        q_space = self.space.discrete_spaces["q"]

        f_porosity = self.functions["porosity"]
        f_d_phi = self.functions["d_phi"]

        v_components = v_space.n_comp
        q_components = q_space.n_comp

        v_data: ElementData = v_space.elements[iel].data
        q_data: ElementData = q_space.elements[iel].data

        cell = v_data.cell
        dim = v_data.dimension
        points, weights = self.space.quadrature
        x, jac, det_jac, inv_jac = v_space.elements[iel].evaluate_mapping(points)

        # basis
        v_phi_tab = v_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)
        q_phi_tab = q_space.elements[iel].evaluate_basis(points, jac, det_jac, inv_jac)

        n_v_phi = v_phi_tab.shape[2]
        n_q_phi = q_phi_tab.shape[2]

        n_v_dof = n_v_phi * v_components
        n_q_dof = n_q_phi * q_components

        idx_dof = {
            "v": slice(0, n_v_dof),
            "q": slice(n_v_dof, n_v_dof + n_q_dof),
        }

        n_dof = n_v_dof + n_q_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # Partial local vectorization

        phi_star = f_porosity(x[:, 0], x[:, 1], x[:, 2])
        # nick name for d_phi
        delta_star = f_d_phi(x[:, 0], x[:, 1], x[:, 2])
        phi_q_star = det_jac * weights * q_phi_tab[0, :, :, 0].T

        # constant directors
        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])
        with ad.AutoDiff(alpha) as alpha:
            el_form = np.zeros(n_dof)

            for i, omega in enumerate(weights):
                xv = x[i]

                phi = phi_star[i]
                delta = delta_star[i]

                # Functions and derivatives at integration point i
                psi_h = v_phi_tab[0, i, :, 0:dim]
                w_h = q_phi_tab[0, i, :, 0:dim]

                u_h = alpha[:, idx_dof["v"]] @ psi_h
                v_h = alpha_scaled[idx_dof["v"]] @ psi_h

                p_h = alpha[:, idx_dof["q"]] @ w_h
                q_h = alpha_scaled[idx_dof["q"]] @ w_h

                if phi > 0.0:
                    v_h *= delta
                    q_h *= 1.0 / np.sqrt(phi)

                equ_1_integrand = (u_h - v_h) @ psi_h.T
                equ_2_integrand = (p_h - q_h) @ w_h.T

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, idx_dof["v"]] = equ_1_integrand
                multiphysic_integrand[:, idx_dof["q"]] = equ_2_integrand
                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        return r_el, j_el


