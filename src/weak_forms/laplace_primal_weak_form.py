import auto_diff as ad
import basix
import numpy as np
from auto_diff.vecvalder import VecValDer
from basix import CellType

from basis.element_data import ElementData
from weak_forms.weak_from import WeakForm


class LaplacePrimalWeakForm(WeakForm):
    def evaluate_form(self, element_index, alpha):
        i = element_index
        if self.space is None or self.functions is None:
            raise ValueError

        p_space = self.space.discrete_spaces["p"]

        f_rhs = self.functions["rhs"]
        f_kappa = self.functions["kappa"]

        p_components = p_space.n_comp
        p_data: ElementData = p_space.elements[i].data

        cell = p_data.cell
        dim = p_data.dimension
        points = p_data.quadrature.points
        weights = p_data.quadrature.weights
        x = p_data.mapping.x
        det_jac = p_data.mapping.det_jac
        inv_jac = p_data.mapping.inv_jac
        p_phi_tab = p_data.basis.phi

        n_p_phi = p_phi_tab.shape[2]
        n_p_dof = n_p_phi * p_components

        n_dof = n_p_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # Partial local vectorization
        f_val_star = f_rhs(x[:, 0], x[:, 1], x[:, 2])
        phi_s_star = det_jac * weights * p_phi_tab[0, :, :, 0].T

        # constant directors
        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])
        with ad.AutoDiff(alpha) as alpha:
            el_form = np.zeros(n_dof)
            for c in range(p_components):
                b = c
                e = b + n_dof
                el_form[b:e:p_components] -= phi_s_star @ f_val_star[c]

            for i, omega in enumerate(weights):
                xv = x[i]
                if dim == 2:
                    inv_jac_m = np.vstack((inv_jac[i] @ e1, inv_jac[i] @ e2))
                else:
                    inv_jac_m = np.vstack(
                        (inv_jac[i] @ e1, inv_jac[i] @ e2, inv_jac[i] @ e3)
                    )
                grad_phi = (
                    inv_jac_m @ p_phi_tab[1 : p_phi_tab.shape[0] + 1, i, :, 0]
                ).T
                grad_uh = alpha @ grad_phi
                grad_uh *= f_kappa(xv[0], xv[1], xv[2])
                energy_h = (grad_phi @ grad_uh.T).reshape((n_dof,))
                el_form += det_jac[i] * omega * energy_h

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        return r_el, j_el


class LaplacePrimalWeakFormBCDirichlet(WeakForm):
    def evaluate_form(self, element_index, alpha):
        p_D = self.functions["p"]

        i = element_index
        p_space = self.space.discrete_spaces["p"]
        p_components = p_space.n_comp
        p_data: ElementData = p_space.bc_elements[i].data

        cell = p_data.cell
        points = p_data.quadrature.points
        weights = p_data.quadrature.weights
        x = p_data.mapping.x
        det_jac = p_data.mapping.det_jac
        inv_jac = p_data.mapping.inv_jac

        p_phi_tab = p_data.basis.phi
        n_p_phi = p_phi_tab.shape[2]
        n_p_dof = n_p_phi * p_components

        n_dof = n_p_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # local blocks
        beta = 1.0e12
        jac_block_p = np.zeros((n_p_phi, n_p_phi))
        for i, omega in enumerate(weights):
            phi = p_phi_tab[0, i, :, 0]
            jac_block_p += beta * det_jac[i] * omega * np.outer(phi, phi)

        for c in range(p_components):
            b = c
            e = b + n_p_dof

            res_block_p = np.zeros(n_p_phi)
            for i, omega in enumerate(weights):
                phi = p_phi_tab[0, i, :, 0]
                p_D_v = p_D(x[i, 0], x[i, 1], x[i, 2])
                res_block_p -= beta * det_jac[i] * omega * p_D_v[c] * phi

            r_el[b:e:p_components] += res_block_p
            j_el[b:e:p_components, b:e:p_components] += jac_block_p

        return r_el, j_el
