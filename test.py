import daft

def build_consequence_model_daft():
    # observed_style="shaded" -> filled circles for observed nodes
    pgm = daft.PGM(shape=[9, 6], origin=[0, 0], observed_style="shaded")

    # ------------------------------------------------------------
    # Scales
    # ------------------------------------------------------------
    latent_scale = 1.0   # θ, A
    obs_scale    = 1.0   # X, Y (shaded)
    hyper_scale  = 0.6   # a, b, β, γ, C, s, u

    # ------------------------------------------------------------
    # Hyperparameters / fixed values (smaller circles)
    # ------------------------------------------------------------

    # Hyperparameters for theta (above tumor plate, middle-top)
    pgm.add_node("a_theta",  r"$a_\theta$", 3.5, 5.0, scale=hyper_scale)
    pgm.add_node("b_theta",  r"$b_\theta$", 4.5, 5.0, scale=hyper_scale)

    # Hyperparameters for A (above A plate, right-top)
    pgm.add_node("beta_A",   r"$\beta_A$",  7.0, 5.0, scale=hyper_scale)
    pgm.add_node("gamma_A",  r"$\gamma_A$", 8.0, 5.0, scale=hyper_scale)

    # Fixed COSMIC matrix C_kj (left)
    pgm.add_node("C_kj", r"$C_{kj}$", 1.0, 3.0, scale=hyper_scale)

    # Fixed tumor-level scales s_i, u_i (inside tumor plate)
    pgm.add_node("s_i",  r"$s_i$", 3.0, 3.0, scale=hyper_scale)
    pgm.add_node("u_i",  r"$u_i$", 5.0, 3.0, scale=hyper_scale)

    # ------------------------------------------------------------
    # Latent variables (regular circles)
    # ------------------------------------------------------------

    # Shared tumor–signature exposures (center of tumor plate)
    pgm.add_node("theta_ik", r"$\theta_{ik}$", 4.0, 3.5, scale=latent_scale)

    # Signature–consequence rates (right)
    pgm.add_node("A_kl", r"$A_{k\ell}$", 7.5, 3.5, scale=latent_scale)

    # ------------------------------------------------------------
    # Observed variables (shaded circles)
    # ------------------------------------------------------------

    # SBS96 counts (bottom-left of tumor plate)
    pgm.add_node("X_ij", r"$X_{ij}$", 3.5, 1.5, scale=obs_scale, observed=True)

    # Consequence counts (bottom-right of tumor plate)
    pgm.add_node("Y_il", r"$Y_{i\ell}$", 4.5, 1.5, scale=obs_scale, observed=True)

    # ------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------

    # Priors: hyperparameters -> Gamma latent variables
    pgm.add_edge("a_theta", "theta_ik")
    pgm.add_edge("b_theta", "theta_ik")

    pgm.add_edge("beta_A",  "A_kl")
    pgm.add_edge("gamma_A", "A_kl")

    # SBS96 channel:
    #   X_ij ~ Poisson(s_i * Σ_k θ_ik C_kj)
    # We show the key parents: s_i, θ_ik, C_kj
    pgm.add_edge("s_i",      "X_ij")
    pgm.add_edge("theta_ik", "X_ij")
    pgm.add_edge("C_kj",     "X_ij")

    # Consequence channel:
    #   Y_il ~ Poisson(u_i * Σ_k θ_ik A_kl)
    pgm.add_edge("u_i",      "Y_il")
    pgm.add_edge("theta_ik", "Y_il")
    pgm.add_edge("A_kl",     "Y_il")

    # ------------------------------------------------------------
    # Plates
    # ------------------------------------------------------------

    # Tumor plate over i (center)
    # Contains: s_i, u_i, theta_ik, X_ij, Y_il
    pgm.add_plate(
        [2.5, 1.0, 3.5, 3.0],
        label=r"$N$",
        shift=-0.1,
    )

    # Inside tumors: signatures k for θ_ik
    pgm.add_plate(
        [3.5, 3.2, 1.0, 0.8],
        label=r"$K$",
        shift=-0.1,
    )

    # Inside tumors: contexts j for X_ij
    pgm.add_plate(
        [3.0, 1.2, 1.2, 1.0],
        label=r"$J$",
        shift=-0.1,
    )

    # Inside tumors: consequences ℓ for Y_il
    pgm.add_plate(
        [4.0, 1.2, 1.2, 1.0],
        label=r"$L$",
        shift=-0.1,
    )

    # Left plate: (k, j) for C_kj
    pgm.add_plate(
        [0.5, 2.5, 1.5, 1.0],
        label=r"$KxJ$",
        shift=-0.1,
    )

    # Right plate: (k, ℓ) for A_kl
    pgm.add_plate(
        [7.0, 3.0, 1.5, 1.0],
        label=r"$KxL$",
        shift=-0.1,
    )

    return pgm


if __name__ == "__main__":
    pgm = build_consequence_model_daft()
    pgm.render()
    pgm.savefig("consequence_model_daft.pdf")
    pgm.savefig("consequence_model_daft.png", dpi=200)
