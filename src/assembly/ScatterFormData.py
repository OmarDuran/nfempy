
import numpy as np

def __scatter_generic_lhs_data(jac_g, jac_l, destination_idx):
    data = jac_l.ravel()
    row = np.repeat(destination_idx, len(destination_idx))
    col = np.tile(destination_idx, len(destination_idx))
    nnz = data.shape[0]
    for k in range(nnz):
        jac_g.setValue(row=row[k], col=col[k], value=data[k], addv=True)

def __scatter_generic_rhs_data(res_g, res_l, destination_idx):
    # The following statement res_g[destination_idx] += res_l does not account for
    # multiple repeated index in destination_idx
    for k, sequ in enumerate(destination_idx):
        res_g[sequ] += res_l[k]

def __scatter_generic_form_data(res_g, jac_g, res_l, jac_l, destination_idx):
    # contribute rhs
    __scatter_generic_rhs_data(res_g, res_l, destination_idx)
    # contribute lhs
    __scatter_generic_lhs_data(jac_g, jac_l, destination_idx)

def scatter_time_dependent_form_data(idx, weak_form, res_g, jac_g, alpha_n, alpha, t):
    destination_idx = weak_form.space.destination_indexes(idx)
    alpha_l = alpha[destination_idx]
    alpha_l_n = alpha_n[destination_idx]
    res_l, jac_l = weak_form.evaluate_form(idx, alpha_l_n, alpha_l, t)
    __scatter_generic_form_data(res_g, jac_g, res_l, jac_l, destination_idx)

def scatter_form_data(idx, weak_form, res_g, jac_g, alpha):
    destination_idx = weak_form.space.destination_indexes(idx)
    alpha_l = alpha[destination_idx]
    res_l, jac_l = weak_form.evaluate_form(idx, alpha_l)
    __scatter_generic_form_data(res_g, jac_g, res_l, jac_l, destination_idx)

def scatter_time_dependent_bc_form_data(idx, weak_form, res_g, jac_g, alpha_n, alpha, t):
    destination_idx = weak_form.space.bc_destination_indexes(idx)
    alpha_l = alpha_n[destination_idx]
    res_l, jac_l = weak_form.evaluate_form(idx, alpha_l, t)
    __scatter_generic_form_data(res_g, jac_g, res_l, jac_l, destination_idx)

def scatter_bc_form_data(idx, weak_form, res_g, jac_g, alpha):
    destination_idx = weak_form.space.bc_destination_indexes(idx)
    alpha_l = alpha[destination_idx]
    res_l, jac_l = weak_form.evaluate_form(idx, alpha_l)
    __scatter_generic_form_data(res_g, jac_g, res_l, jac_l, destination_idx)

def scatter_interface_form_data(triplet_idx, weak_form, res_g, jac_g, alpha):
    idx, idx_pair = triplet_idx
    idx_p, idx_n = idx_pair
    destination_idx_p = weak_form.space.destination_indexes(idx_p)
    destination_idx_n = weak_form.space.destination_indexes(idx_n)
    destination_idx = np.concatenate((destination_idx_p, destination_idx_n))
    alpha_pair = (alpha[destination_idx_p], alpha[destination_idx_n])

    res_l, jac_l = weak_form.evaluate_form(idx, idx_pair, alpha_pair)
    __scatter_generic_form_data(res_g, jac_g, res_l, jac_l, destination_idx)

def scatter_bc_interface_form_data(pair_idx, weak_form, res_g, jac_g, alpha):
    idx, idx_p = pair_idx
    destination_idx = weak_form.space.destination_indexes(idx_p)
    alpha_l = alpha[destination_idx]
    r_el, j_el = weak_form.evaluate_form(idx, idx_p, alpha_l)
    __scatter_generic_form_data(res_g, jac_g, r_el, j_el, destination_idx)

