import os
import glob
import numpy as np
import scipy.special
from crpropa import ParticleCollector

def addTwoAccumulators(acs1 : "accumulator", acs2 : "accumulator"):
    """ Combine outputs of two accumulators """
    total = []
    for a1, a2 in zip(acs1, acs2):
        accumulator = {"dist": a1["dist"], "output": ParticleCollector()}
        accumulator["output"].setClone(True)

        for c in a1["output"]:
            accumulator["output"].process(c)
        for c in a2["output"]:
            accumulator["output"].process(c)

        total.append(accumulator)
    return total

def sumAccumulators(list_of_acc : "list_of_accumulators"):
    """ Join outputs of a list accumulators """
    if len(list_of_acc) > 1:
        return addTwoAccumulators(sumAccumulators(list_of_acc[1:]), list_of_acc[0])
    return list_of_acc[0]

def saveAccumulators(accumulators : "accumulators", dirname : "path"):
    if not os.path.exists(dirname):
        os.makedirs(dirname, 0o700)

    for a in accumulators:
        a["output"].dump("{}/{}.txt.gz".format(dirname, a["dist"]))

def loadAccumulators(dirname : "path"):
    accumulators = []

    files = glob.glob(dirname+"/*.txt.gz")
    for f in files:
        accumulator = {
            "dist": float(".".join(os.path.basename(f).split(".")[0:2]))*meter,
            "output": ParticleCollector()
        }
        accumulator["output"].load(f)
        accumulator["output"].setClone(True)
        accumulators.append(accumulator)

    return sorted(accumulators, key=lambda k: k['dist'])

# QLT slab theory

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
        C_slab_exact(nu)
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

def diff_coeff_slab_QLT(nu: "spectral_index",
        l_slab: "correlation_length",
        B0: "mean_mfield",
        Brms: "rms_mfield",
        ratio: "R_g/L_c") -> float:
    return (
        c_light
        * l_slab
        / (8 * np.pi * C_slab_exact(nu))
        * (B0 / Brms) ** 2
        * ratio ** (2 - nu)
        * (
            1
            / (1 - nu / 2)
            * scipy.special.hyp2f1(1 - nu / 2, -nu / 2, 2 - nu / 2, -ratio ** 2)
            - 1
            / (2 - nu / 2)
            * scipy.special.hyp2f1(2 - nu / 2, -nu / 2, 3 - nu / 2, -ratio ** 2)
        )
    )

def corr_length_slab(nu: "spectral_index", l_slab: "bendover_scale") -> float:
    return 2 * np.pi * C_slab_exact(nu) * l_slab

def diff_coeff_slab_QLT_approx(nu: "spectral_index",
        l_slab: "correlation_length",
        B0: "mean_mfield",
        Brms: "rms_mfield",
        ratio: "R_g/L_c") -> float:
    return (
        c_light
        * l_slab
        / (8 * np.pi * C_slab_exact(nu))
        * (B0 / Brms) ** 2
        * ratio ** (2 - nu)
        * (1 / (1 - nu / 2) - 1 / (2 - nu / 2))
    )

