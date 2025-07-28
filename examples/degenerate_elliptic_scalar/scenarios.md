# Scenario 1: Need for Transformed Variables

Scenario 1 is a two-materials domain numerical experiment designed to show the failure of the standard Darcy formulation in low porosity and low permeability conditions.

**Formulation Used:** This scenario uses the Non-Transformed Model in a single domain. The porosity (ϕ) is not set to zero but is assigned a small positive "cutoff" value. The equations are:

$$
\nabla \cdot \mathbf{u} + \phi p = \phi^{1/2} f
$$

$$
\mathbf{u} = -d^2(\phi) \nabla p
$$

**Goal:** The objetive is to showcase that the standard approach becomes numerically unstable as the porosity gets smaller. This can be shown by sampling how the following metrics worsen as the porosity cutoff value is reduced:

- **Condition number:** expected to increase as porosity decrease.
- **Error rates:** Expected to degrade.
- **Absolute error:** Expected to increase.


# Scenario 2: Stationary Degeneracy

Scenario 2 is a two-materials domain and tests the proposed method where one part of the domain has a fixed (stationary) degenerate porosity field. The interace separating the two domains is not matching the level set (ϕ=0).

**Formulation Used:** This scenario uses the Transformed Model (Model 2), which couples a standard Darcy domain (Ω₁) with a degenerate domain (Ω₂) using transformed variables.

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

Pressure continuity is enforced weakly, while flux continuity is enforced using piecewise constant Lagrange multipliers.

$$
p_1 = \phi_2^{-1/2} q_2
$$

$$
\mathbf{u}_1 \cdot \mathbf{n}_1 = -d_2(\phi_2) \mathbf{v}_2 \cdot \mathbf{n}_2
$$

For the simulation, the porosity in the degenerate domain is defined from [1]. There in, see Sections **See 4.1 A simple Euler equation in one dimension** and **4.2 A smooth solution test in two dimensions** for the 1D and 2D cases, respectively.


**Goal:** The objective is to validate the method show that it is stable and accurate.

## References

[1] Arbogast, T., & Taicher, A. L. (2017). A cell-centered finite difference method for a degenerate elliptic equation arising from two-phase mixtures. Computational Geosciences, 21(4), 701–712. https://doi.org/10.1007/s10596-017-9649-9