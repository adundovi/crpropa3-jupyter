import numpy as np
import scipy.special
from crpropa import c_light

# Slab turbulence

def C_slab(nu: "spectral_index") -> float:
    return (
        1
        / (2 * np.sqrt(np.pi))
        * scipy.special.gamma(nu / 2)
        / scipy.special.gamma((nu - 1) / 2.0)
    )

def C_slab_finite(
    nu: "spectral_index", l_bo: "bendover_scale", kmin: "k_min", kmax: "k_max"
) -> float:
    return 1.0 / (
        4
        * l_bo
        * (
            kmax * scipy.special.hyp2f1(1 / 2.0, nu / 2, 3 / 2.0, -(kmax * l_bo) ** 2)
            - kmin * scipy.special.hyp2f1(1 / 2.0, nu / 2, 3 / 2.0, -(kmin * l_bo) ** 2)
        )
    )

def S_slab(Brms, l_bo, k, nu):
    return (
        C_slab(nu)
        / (2 * np.pi)
        * Brms ** 2
        * l_bo
        / (1 + (k * l_bo) ** 2) ** (nu / 2.0)
    )


def S_slab_finite(Brms, l_bo, k, nu, kmin, kmax):
    return (
        C_slab_finite(nu, l_bo, kmin, kmax)
        / (2 * np.pi)
        * Brms ** 2
        * l_bo
        / (1 + (k * l_bo) ** 2) ** (nu / 2.0)
    )

def corr_length_slab(nu: "spectral_index", l_slab: "bendover_scale") -> float:
    return 2 * np.pi * C_slab(nu) * l_slab

# Isotropic turbulence

def corr_length_isotropic_finite(
    n: "spectral index", Lbo: "bendover scale", Lmax: "L max scale", Lmin: "L min scale"
) -> float:
    return (
        (2 * np.pi) ** n
        * (n - 1)
        / (2 * n * (n + 2))
        * (
            Lmax ** n
            / Lbo ** (n - 1)
            * (2 * np.pi ** 2 * (n + 2) + Lmax ** 2 / Lbo ** 2)
            / (4 * np.pi ** 2 + Lmax ** 2 / Lbo ** 2) ** (n / 2 + 1)
            - Lmin ** n
            / Lbo ** (n - 1)
            * (2 * np.pi ** 2 * (n + 2) + Lmin ** 2 / Lbo ** 2)
            / (4 * np.pi ** 2 + Lmin ** 2 / Lbo ** 2) ** (n / 2 + 1)
        )
        / (
            (Lmax / Lbo) ** (n - 1)
            * scipy.special.hyp2f1(
                (n - 1) / 2.0,
                (n + 4) / 2.0,
                (n + 1) / 2.0,
                -Lmax ** 2 / (4 * np.pi ** 2 * Lbo ** 2),
            )
            - (Lmin / Lbo) ** (n - 1)
            * scipy.special.hyp2f1(
                (n - 1) / 2.0,
                (n + 4) / 2.0,
                (n + 1) / 2.0,
                -Lmin ** 2 / (4 * np.pi ** 2 * Lbo ** 2),
            )
        )
    )


corr_length_finite_with_grid_size = lambda n, Lbo, Lmax, N: corr_lenght_finite(
    n, Lbo, Lmax, 2 * Lmax / N
)


def corr_length_isotropic(n: "spectral index", Lbo: "bendover scale") -> float:
    return (
        8
        * np.sqrt(np.pi)
        * Lbo
        / (3 * (n ** 2 + 2 * n))
        * scipy.special.gamma(n / 2.0 + 2)
        / scipy.special.gamma(n / 2.0 - 1 / 2)
    )
