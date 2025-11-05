import resource
import time
import numpy as np
import meshio

from mesh.mesh import Mesh
from mesh.mesh_metrics import cell_centroid
from petsc4py import PETSc
from basis.element_family import family_by_name
from postprocess.solution_post_processor import cell_centered_quatity, node_average_quatity
from spaces.product_space import ProductSpace
from weak_forms.le_primal_weak_form import LEPrimalWeakForm, LEPrimalWeakFormBCNormalDirichlet, LEPrimalWeakFormBCNeumann
from weak_forms.le_primal_stress_constraint_weak_form import LEPrimalStressConstraintWeakForm


def create_product_space(method, gmesh, mat_ids):
    # FESpace: data
    u_k_order = method[1]["u"][1]
    u_components = 2 if gmesh.dimension == 2 else 3
    u_family = method[1]["u"][0]

    discrete_spaces_data = {
        "u": (gmesh.dimension, u_components, u_family, u_k_order, gmesh),
    }

    u_disc_Q = False  # We want continuous elements for CG
    discrete_spaces_disc = {
        "u": u_disc_Q,
    }

    physical_tags = {
        "u": mat_ids['under'] + mat_ids['reservoir'] + mat_ids['over'],  # domain tags
    }
    # Boundary tags: 5=left, 6=right, 7=bottom, 8=top (assumed from mesh)
    b_physical_tags = {
        "u": mat_ids['bc_bottom'] + mat_ids['bc_top'] + mat_ids['bc_west'] + mat_ids['bc_east'],
    }

    space = ProductSpace(discrete_spaces_data)
    space.make_subspaces_discontinuous(discrete_spaces_disc)
    space.build_structures(physical_tags, b_physical_tags)
    return space


def primal_approximation_with_load(material_data, fe_space, gmesh, mat_ids):
    n_dof_g = fe_space.n_dof

    rg = np.zeros(n_dof_g)
    alpha = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    memory_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    st = time.time()

    A = PETSc.Mat()
    A.createAIJ([n_dof_g, n_dof_g])
    A.setType("sbaij")

    m_lambda = material_data["lambda"]
    m_mu = material_data["mu"]

    # Zero body force
    def f_rhs(x, y, z):
        return np.array(
            [
                np.zeros_like(x),
                material_data['rho'] * np.ones_like(y),
            ]
        )

    def f_lambda(x, y, z):
        return m_lambda * np.ones_like(x)

    def f_mu(x, y, z):
        return m_mu * np.ones_like(x)

    def s_target(x, y, z):
        return np.array(
            [
                [
                    -3.0e6 * np.ones_like(x),
                    np.zeros_like(x),
                ],
                [
                    np.zeros_like(x),
                    -11.0e6 * np.ones_like(x),
                ]
            ]
        )

    m_functions = {
        "rhs": f_rhs,
        "lambda": f_lambda,
        "mu": f_mu,
        "s_target": s_target,
    }

    weak_form = LEPrimalWeakForm(fe_space)
    weak_form.functions = m_functions

    # Dirichlet BC: zero displacement on left, right, bottom (tags 5, 6, 7)
    def zero_disp(x, y, z):
        return np.array(
            [
                np.zeros_like(x),
                np.zeros_like(y),
            ]
        )

    bc_dirichlet = LEPrimalWeakFormBCNormalDirichlet(fe_space)
    bc_dirichlet.functions = {"u": zero_disp}

    # Neumann BC: vertical load on top (tag 8)
    def vertical_load(x, y, z):
        # Apply a vertical load of magnitude 1.0 in y direction
        return np.array([np.zeros_like(x), 10.0e6 * np.ones_like(x)])

    bc_neumann = LEPrimalWeakFormBCNeumann(fe_space)
    bc_neumann.functions = {"t": vertical_load}


    constraint_weak_form = LEPrimalStressConstraintWeakForm(fe_space)
    constraint_weak_form.functions = m_functions

    def scatter_form_data(A, i, weak_form, n_els):
        dest = weak_form.space.destination_indexes(i)
        alpha_l = alpha[dest]
        r_el, j_el = weak_form.evaluate_form(i, alpha_l)

        # contribute rhs
        rg[dest] += r_el

        # contribute lhs
        data = j_el.ravel()
        row = np.repeat(dest, len(dest))
        col = np.tile(dest, len(dest))
        nnz_idx = np.where(np.logical_not(np.isclose(data, 1.0e-16)))[0]
        [
            A.setValue(row=row[idx], col=col[idx], value=data[idx], addv=True)
            for idx in nnz_idx
            if row[idx] <= col[idx]
        ]
        # Diagonal zeros for PETSc ILU
        [A.setValue(row=idx, col=idx, value=0.0, addv=True) for idx in dest]

        check_points = [(int(k * n_els / 10)) for k in range(11)]
        if i in check_points or i == n_els - 1:
            if i == n_els - 1:
                print("Assembly: progress [%]: ", 100)
            else:
                print("Assembly: progress [%]: ", check_points.index(i) * 10)
            print(
                "Assembly: Memory used [Byte] :",
                (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - memory_start),
            )

    def scatter_bc_dirichlet(A, i, bc_weak_form):
        dest = fe_space.bc_destination_indexes(i)
        alpha_l = alpha[dest]
        r_el, j_el = bc_weak_form.evaluate_form(i, alpha_l)
        rg[dest] += r_el

        # contribute lhs
        data = j_el.ravel()
        row = np.repeat(dest, len(dest))
        col = np.tile(dest, len(dest))
        nnz_idx = np.where(np.logical_not(np.isclose(data, 1.0e-16)))[0]
        [
            A.setValue(row=row[idx], col=col[idx], value=data[idx], addv=True)
            for idx in nnz_idx
            if row[idx] <= col[idx]
        ]

    def scatter_bc_neumann(i, bc_weak_form):
        dest = fe_space.bc_destination_indexes(i)
        alpha_l = alpha[dest]
        r_el, _ = bc_weak_form.evaluate_form(i, alpha_l)
        rg[dest] += r_el

    n_els = len(fe_space.discrete_spaces["u"].elements)
    [scatter_form_data(A, i, weak_form, n_els) for i in range(n_els)]

    # get bounday elements
    bc_elements = fe_space.discrete_spaces["u"].bc_elements
    bc_top_idx = [i for i, bel in enumerate(bc_elements) if bel.data.cell.material_id in mat_ids['bc_top']]
    bc_bottom_idx = [i for i, bel in enumerate(bc_elements) if bel.data.cell.material_id in mat_ids['bc_bottom']]
    bc_laterals_idx = [i for i, bel in enumerate(bc_elements) if bel.data.cell.material_id in mat_ids['bc_west']+mat_ids['bc_east']]

    [scatter_bc_dirichlet(A, i, bc_dirichlet) for i in bc_bottom_idx]
    [scatter_bc_dirichlet(A, i, bc_dirichlet) for i in bc_laterals_idx]
    [scatter_bc_neumann(i, bc_neumann) for i in bc_top_idx]

    # apply constraint on selected elements
    elements = fe_space.discrete_spaces["u"].elements

    gd2c1 = gmesh.build_graph(2,1)
    faccets_well_1 = [cell for cell in gmesh.cells if cell.material_id == mat_ids["well_1"]]
    faccets_well_2 = [cell for cell in gmesh.cells if cell.material_id == mat_ids["well_2"]]
    faccets_fault = [cell for cell in gmesh.cells if cell.material_id == mat_ids["fault"]]

    # target_idx = [i for i, el in enumerate(elements) if el.data.cell.material_id in mat_ids['reservoir']]

    cell_centroids = [cell_centroid(el.data.cell, gmesh) for el in elements]
    centroids = np.asarray(cell_centroids)
    xc = np.array([[500.0, 277, 0.0]])
    r = 100.0  # Radius of the sphere in meters
    # Squared distances to avoid unnecessary sqrt
    diff = centroids - xc
    dist2 = np.einsum("ij,ij->i", diff, diff)
    r2 = r * r
    # Optional numerical tolerance
    tol = 1e-12
    mask = dist2 <= r2 + tol
    # Indices of elements whose centroids lie inside / on the sphere
    inside_idx = np.nonzero(mask)[0]
    # Subset centroids and elements if needed
    inside_centroids = centroids[inside_idx]
    inside_elements = [elements[i] for i in inside_idx]
    print(f"Selected {inside_idx.size} elements inside sphere.")
    [scatter_form_data(A, i, constraint_weak_form, len(inside_idx)) for i in inside_idx]

    A.assemble()
    print("Assembly: nz_allocated:", int(A.getInfo()["nz_allocated"]))
    print("Assembly: nz_used:", int(A.getInfo()["nz_used"]))
    print("Assembly: nz_unneeded:", int(A.getInfo()["nz_unneeded"]))

    et = time.time()
    print("Assembly: Time:", et - st, "seconds")

    st = time.time()
    ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
    ksp.setOperators(A)
    b = A.createVecLeft()
    b.array[:] = -rg
    x = A.createVecRight()

    ksp.setType("cg")
    ksp.getPC().setType("icc")
    ksp.setTolerances(rtol=0.0, atol=1e-10, divtol=5000, max_it=1000)
    ksp.setFromOptions()

    ksp.solve(b, x)
    alpha = x.array

    PETSc.KSP.destroy(ksp)
    PETSc.Mat.destroy(A)
    PETSc.Vec.destroy(b)
    PETSc.Vec.destroy(x)

    et = time.time()
    print("Linear solver: Time:", et - st, "seconds")

    return alpha

def write_vtk_file_with_stress(file_name, gmesh, fe_space, alpha, material_data, cell_centered=[]):
    """
    Write VTK including:
      - Nodal displacement field(s) (unchanged logic)
      - Nodal Cauchy stress tensor (volume–weighted average of adjacent element stresses)
    Stress tensor is written as a flattened array:
        2D -> [s_xx, s_xy, s_yx, s_yy]
        3D -> [s_xx, s_xy, s_xz, s_yx, s_yy, s_yz, s_zx, s_zy, s_zz]
    """
    dim = gmesh.dimension
    lam = material_data["lambda"]
    mu = material_data["mu"]

    vec_families = [
        family_by_name("RT"),
        family_by_name("BDM"),
        family_by_name("N1E"),
        family_by_name("N2E"),
    ]

    p_data_dict = {}
    c_data_dict = {}

    # Displacements (and other primal fields) as before
    for name, space in fe_space.discrete_spaces.items():
        n_comp = space.n_comp
        n_data = n_comp
        if space.family in vec_families:
            n_data *= dim

        if name in cell_centered:
            cells_dim = [cell for cell in gmesh.cells if cell.dimension == dim]
            fh_data = np.zeros((len(cells_dim), n_data))
            for cell_idx, cell in enumerate(cells_dim):
                f_h = cell_centered_quatity(cell, fe_space, name, alpha)
                fh_data[cell_idx] = f_h.ravel()
            c_data_dict[name + "_h"] = [fh_data]
        else:
            fh_data = np.zeros((len(gmesh.points), n_data))
            vertices = space.mesh_topology.entities_by_dimension(0)
            cell_vertex_map = space.mesh_topology.entity_map_by_dimension(0)
            for vertex_idx in vertices:
                vertex_g_index = (0, vertex_idx)
                if not cell_vertex_map.has_node(vertex_g_index):
                    continue
                node_idx = gmesh.cells[vertex_idx].node_tags[0]
                cell_idxs = list(cell_vertex_map.predecessors(vertex_g_index))
                f_h = node_average_quatity(node_idx, cell_idxs, fe_space, name, alpha)
                fh_data[node_idx] = f_h.ravel()
            p_data_dict[name + "_h"] = fh_data

    # --- Nodal stress tensor recovery ---
    u_space = fe_space.discrete_spaces["u"]
    n_el = len(u_space.elements)

    # Accumulators
    n_points = len(gmesh.points)
    stress_nodes = np.zeros((n_points, dim, dim))
    weight_nodes = np.zeros(n_points)

    # Constants for gradient reconstruction
    e1 = np.zeros(3); e1[0] = 1.0
    e2 = np.zeros(3); e2[1] = 1.0
    e3 = np.zeros(3); e3[2] = 1.0

    points_q, weights_q = fe_space.quadrature[dim]

    for iel in range(n_el):
        el = u_space.elements[iel]
        dest = fe_space.destination_indexes(iel)

        x_q, jac_q, det_jac_q, inv_jac_q = el.evaluate_mapping(points_q)
        u_phi_tab = el.evaluate_basis(points_q, jac_q, det_jac_q, inv_jac_q)  # [val/derivs, nq, nphi, 0/1]

        n_phi = u_phi_tab.shape[2]
        u_comp = u_space.n_comp

        # Local coefficients grouped by component (assuming stride = u_comp)
        a_comp = [alpha[dest[c::u_comp]] for c in range(u_comp)]

        stress_acc = np.zeros((dim, dim))
        vol_acc = 0.0

        for iq, omega in enumerate(weights_q):
            detJ = det_jac_q[iq]
            w = detJ * omega

            if dim == 2:
                inv_jac_m = np.vstack((inv_jac_q[iq] @ e1, inv_jac_q[iq] @ e2))  # (2,2)
                grad_phi_u = (inv_jac_m @ u_phi_tab[1:u_phi_tab.shape[0] + 1, iq, :, 0]).T  # (n_phi,2)
            else:
                inv_jac_m = np.vstack((inv_jac_q[iq] @ e1, inv_jac_q[iq] @ e2, inv_jac_q[iq] @ e3))  # (3,3)
                grad_phi_u = (inv_jac_m @ u_phi_tab[1:u_phi_tab.shape[0] + 1, iq, :, 0]).T  # (n_phi,3)

            grad_u = np.zeros((dim, dim))
            for c in range(u_comp):
                grad_u[c, :] = a_comp[c] @ grad_phi_u

            eps = 0.5 * (grad_u + grad_u.T)
            tr_eps = np.trace(eps)
            sigma = lam * tr_eps * np.eye(dim) + 2.0 * mu * eps

            stress_acc += w * sigma
            vol_acc += w

        if vol_acc <= 0.0:
            continue
        sigma_el = stress_acc / vol_acc  # element-averaged stress

        # Distribute to element vertices (volume-weighted)
        node_tags = el.data.cell.node_tags
        for nt in node_tags:
            stress_nodes[nt, :, :] += sigma_el * vol_acc
            weight_nodes[nt] += vol_acc

    # Finalize nodal averaging
    nonzero = weight_nodes > 0
    stress_nodes[nonzero] /= weight_nodes[nonzero][:, None, None]

    # Enforce symmetry explicitly (numerical safety)
    for k in np.where(nonzero)[0]:
        stress_nodes[k] = 0.5 * (stress_nodes[k] + stress_nodes[k].T)

    # Flatten tensor for VTK
    sigma_flat = stress_nodes.reshape(n_points, dim * dim)
    p_data_dict["s_h"] = sigma_flat

    # Build connectivity (top-dim cells)
    cells_dim = [cell for cell in gmesh.cells if cell.dimension == dim]
    con_d = np.array([cell.node_tags for cell in cells_dim])
    meshio_cell_types = {0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
    cells_dict = {meshio_cell_types[dim]: con_d}

    mesh = meshio.Mesh(
        points=gmesh.points,
        cells=cells_dict,
        point_data=p_data_dict,
        cell_data=c_data_dict,
    )
    mesh.write(file_name)
    print(f"VTK (with nodal tensor stress) written to {file_name}")

def compute_lame(E, nu):
    """
    Convert (E, nu) to Lamé parameters (lambda, mu) in SI units.
    E: Young's modulus [Pa]
    nu: Poisson ratio [-]
    """
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu

def main():

    mat_ids = {
        'under': [1],
        'reservoir': [2],
        'over': [3],
        'bc_bottom': [4],
        'bc_top': [5],
        'bc_west': [6],
        'bc_east': [7],
        'well_1': [8],
        'well_2': [9],
        'fault': [10],
    }

    dimension = 2
    mesh_file = "gmsh_files/ex_2/example_2_2d.msh"
    gmesh = Mesh(dimension=dimension, file_name=mesh_file)
    gmesh.build_conformal_mesh()
    gmesh.write_vtk()

    # Representative elastic properties (SI units) for a layered reservoir model
    # Sources: typical literature values (order-of-magnitude examples)
    # underburden shale:   E ≈ 25 GPa, nu ≈ 0.25
    # reservoir sandstone: E ≈ 12 GPa, nu ≈ 0.22
    # overburden shale:    E ≈ 20 GPa, nu ≈ 0.26
    materials = {
        'under':     {'E': 25e9, 'nu': 0.25, 'rho': 2500.0},
        'reservoir': {'E': 25e9, 'nu': 0.22, 'rho': 2300.0},
        'over':      {'E': 20e9, 'nu': 0.26, 'rho': 2400.0},
    }

    # Current example: use reservoir layer properties globally (infrastructure not yet per-cell)
    lam, mu = compute_lame(materials['reservoir']['E'], materials['reservoir']['nu'])
    material_data = {"lambda": lam, "mu": mu, "rho": materials['reservoir']['rho']}

    method = ("FEM", {"u": ("Lagrange", 2)})

    fe_space = create_product_space(method, gmesh, mat_ids)
    alpha = primal_approximation_with_load(material_data, fe_space, gmesh, mat_ids)

    write_vtk_file_with_stress("constrained_le_ex_2.vtk", gmesh, fe_space, alpha, material_data)


if __name__ == "__main__":
    main()
