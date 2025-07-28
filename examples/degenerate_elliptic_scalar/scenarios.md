# Scenario 1: Demonstrating the Need for Transformed Variables

Scenario 1 is a two-materials domain numerical experiment designed to show the failure of the standard Darcy formulation in low porosity and low permeability conditions.

**Formulation Used:** This scenario uses the Non-Transformed Model in a single domain. The porosity (ϕ) is not set to zero but is assigned a small positive "cutoff" value. The equations are:

$$
\nabla \cdot \mathbf{u} + \phi p = \phi^{1/2} f
$$

$$
\mathbf{u} = -d^2(\phi) \nabla p
$$

**Goal of the Numerical Results:** The goal is to demonstrate that the standard approach becomes numerically unstable as the porosity gets smaller. This is shown by measuring how the following metrics worsen as the porosity cutoff value is reduced:

- **Condition number:** It is expected to become much larger.
- **Error rates:** They are expected to degrade.
- **Absolute error:** It is expected to increase.

The expected outcome is that the **transformed model** would not suffer from these issues, thereby motivating its use in the domain decomposition setup.

---

# Scenario 2: Testing the Domain Decomposition Method with Stationary Degeneracy

Scenario 2 tests the proposed domain decomposition method where one part of the domain has a fixed (stationary) degenerate porosity field.

**Formulation Used:** This scenario uses the Transformed Model (Model 2), which couples a standard Darcy domain (Ω₁) with a degenerate domain (Ω₂) using transformed variables. The setup is:

**In subdomain Ω₁ (standard):**

$$
\nabla \cdot \mathbf{u}_1 + \phi_1 p_1 = \phi_1^{1/2} f_1
$$

$$
\mathbf{u}_1 = -d_1^2(\phi_1) \nabla p_1
$$

**In subdomain Ω₂ (degenerate):**

$$
\phi_2^{1/2} \nabla \cdot (d_2(\phi_2) \mathbf{v}_2) + q_2 = f_2
$$

$$
\mathbf{v}_2 = -d_2(\phi_2) \nabla(\phi_2^{-1/2} q_2)
$$

**On the interface:**

$$
p_1 = \phi_2^{-1/2} q_2
$$

$$
\mathbf{u}_1 \cdot \mathbf{n}_1 = -d_2(\phi_2) \mathbf{v}_2 \cdot \mathbf{n}_2
$$

For the simulation, the porosity in the degenerate domain is defined as:

$$
\phi = x^5 y^{10}
$$

**Goal of the Numerical Results:** The objective is to validate the domain decomposition method and show that it is both stable and accurate. The specific goal is to compute the error rates for the numerical solution and demonstrate that they achieve the expected linear convergence as the mesh is refined. This confirms the method's reliability for coupling standard and degenerate regions.