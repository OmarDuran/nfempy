import numpy as np
from basis.element_data import ElementData
from weak_forms.weak_from import WeakForm


class L2ProjectorWeakForm(WeakForm):

    def evaluate_form(self, element_index, alpha):

        i = element_index
        if self.space is None or self.functions is None:
            raise ValueError

        n_dofs = []
        j_els = []
        r_els = []
        for item in self.space.discrete_spaces.items():
            name, space = item
            f_rhs = self.functions[name]

            n_comp = space.n_comp
            data: ElementData = space.elements[i].data

            cell = data.cell
            dim = data.dimension
            points = data.quadrature.points
            weights = data.quadrature.weights
            x = data.mapping.x
            det_jac = data.mapping.det_jac
            inv_jac = data.mapping.inv_jac
            phi_tab = data.basis.phi

            n_phi = phi_tab.shape[2]
            n_dof = n_phi * n_comp
            n_dofs.append(n_dof)

            js = (n_dof, n_dof)
            rs = n_dof
            j_el = np.zeros(js)
            r_el = np.zeros(rs)

            # Partial local vectorization
            f_val_star = f_rhs(x[:, 0], x[:, 1], x[:, 2])
            phi_s_star = det_jac * weights * phi_tab[0, :, :, 0].T

            for c in range(n_comp):
                b = c
                e = b + n_dof
                r_el[b:e:n_comp] -= phi_s_star @ f_val_star[c]

            for i, omega in enumerate(weights):
                phi = phi_tab[0, i, :, 0]
                j_el += det_jac[i] * omega * np.outer(phi, phi)

            r_els.append(r_el)
            j_els.append(j_el)

        n_dofs.insert(0,0)
        n_accumulated_dofs = np.add.accumulate(n_dofs)

        n_dof = n_accumulated_dofs[-1]

        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        for i, stride_dof in enumerate(n_accumulated_dofs[0:-1]):
            b = stride_dof
            e = b + n_dofs[i+1]
            r_el[b:e] += r_els[i]
            j_el[b:e,b:e] += j_els[i]


        return r_el, j_el