import time

import numpy as np
from petsc4py import PETSc

from basis.element_family import family_by_name
from spaces.product_space import ProductSpace
from weak_forms.l2_projector_weak_form import L2ProjectorWeakForm


def l2_projector(fe_space, functions, symmetric_solver_q=True):
    n_dof_g = fe_space.n_dof
    rg = np.zeros(n_dof_g)
    alpha = np.zeros(n_dof_g)

    # Assembler
    st = time.time()
    A = PETSc.Mat()
    A.createAIJ([n_dof_g, n_dof_g])
    if symmetric_solver_q:
        A.setType("sbaij")

    st = time.time()

    weak_form = L2ProjectorWeakForm(fe_space)
    weak_form.functions = functions

    def scatter_form_data(A, i, weak_form):
        # destination indexes
        dest = weak_form.space.destination_indexes(i)
        alpha_l = alpha[dest]
        r_el, j_el = weak_form.evaluate_form(i, alpha_l)

        # contribute rhs
        rg[dest] += r_el

        # contribute lhs
        data = j_el.ravel()
        row = np.repeat(dest, len(dest))
        col = np.tile(dest, len(dest))
        nnz = data.shape[0]
        nnz_idx = np.where(np.logical_not(np.isclose(data, 1.0e-16)))[0]
        if symmetric_solver_q:
            [
                A.setValue(row=row[idx], col=col[idx], value=data[idx], addv=True)
                for idx in nnz_idx
                if row[idx] <= col[idx]
            ]
        else:
            [
                A.setValue(row=row[idx], col=col[idx], value=data[idx], addv=True)
                for idx in nnz_idx
            ]

    name = list(fe_space.discrete_spaces.keys())[0]
    n_els = len(fe_space.discrete_spaces[name].elements)
    [scatter_form_data(A, i, weak_form) for i in range(n_els)]

    A.assemble()

    et = time.time()
    elapsed_time = et - st
    print("Assembly time:", elapsed_time, "seconds")

    # solving ls
    st = time.time()
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    b = A.createVecLeft()
    b.array[:] = -rg
    x = A.createVecRight()

    ksp.setType("preonly")
    if symmetric_solver_q:
        ksp.getPC().setType("cholesky")
    else:
        ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setConvergenceHistory()

    ksp.solve(b, x)
    alpha = x.array

    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    return alpha
