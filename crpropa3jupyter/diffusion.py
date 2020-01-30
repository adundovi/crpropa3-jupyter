import os
import glob
import numpy as np
import scipy.special
from crpropa import ParticleCollector, c_light

from .turbulence import *

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

def diff_coeff_slab_QLT(nu: "spectral_index",
        l_slab: "correlation_length",
        B0: "mean_mfield",
        Brms: "rms_mfield",
        ratio: "R_g/L_c") -> float:
    return (
        c_light
        * l_slab
        / (8 * np.pi * C_slab(nu))
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

def diff_coeff_slab_QLT_approx(nu: "spectral_index",
        l_slab: "correlation_length",
        B0: "mean_mfield",
        Brms: "rms_mfield",
        ratio: "R_g/L_c") -> float:
    return (
        c_light
        * l_slab
        / (8 * np.pi * C_slab(nu))
        * (B0 / Brms) ** 2
        * ratio ** (2 - nu)
        * (1 / (1 - nu / 2) - 1 / (2 - nu / 2))
    )

def diff_coeff_slab_QLT_perp(nu, l_slab, B0, Brms) -> float:
    return np.pi/2 * c_light * C_slab(nu) * l_slab * (Brms/B0)**2

## SOQLT for slab

def D_mumu_slab_SOQLT(nu: "spectral_index",
        l_slab: "correlation_length",
        B0: "mean_mfield",
        Brms: "rms_mfield",
        ratio: "R_g/L_c",
        mu: "pitch_angle") -> float:

    h_spectrum = lambda nu, x: (1 + x ** 2) ** (-1*nu / 2)
    gamma_sub = lambda ratio, B0, Brms: ratio * (Brms / B0)

    D_mumu_Int = (
        lambda nu, B0, Brms, ratio, mu, x: x ** (-1)
        * h_spectrum(nu, x)
        * (
            np.exp(-(mu * ratio * x + 1) ** 2 / (gamma_sub(ratio, B0, Brms) * x) ** 2)
            + np.exp(-(mu * ratio * x - 1) ** 2 / (gamma_sub(ratio, B0, Brms) * x) ** 2)
        )
    )
    return ( np.sqrt(np.pi)
        * C_slab(nu) / l_slab * c_light
        * (1 - mu ** 2)
        * ratio ** (-2)
        * (Brms / B0)
        * scipy.integrate.quad(lambda x: D_mumu_Int(nu, B0, Brms, ratio, mu, x), 0, np.inf, limit=100)[0]
    )

def diff_coeff_slab_SOQLT(nu: "spectral_index",
        l_slab: "correlation_length",
        B0: "mean_mfield",
        Brms: "rms_mfield",
        ratio: "R_g/L_c") -> float:

    return (c_light ** 2 / 8
        * scipy.integrate.quad(
            lambda mu: (1 - mu ** 2) ** 2 / D_mumu_slab_SOQLT(nu, l_slab, B0, Brms, ratio, mu),
            -1,
            1,
        )[0]
    )

## NLGC for slab/2D

def D_perp_D_par_ratio_slab2D_NLGC(a, nu, B0, dB_slab, dB_2D, l_slab, l_2D, D_parallel):

    lambda_parallel = 3 * D_parallel / c_light

    beta_f = (
        lambda x, y: scipy.special.gamma(x)
        * scipy.special.gamma(y)
        / scipy.special.gamma(x + y)
    )

    E_integral = (
        lambda nu, a: 1.0
        / (2 * a)
        * beta_f(1.0 / 2, 1.0 / 2 + nu / 2)
        * scipy.special.hyp2f1(1, 1.0 / 2, nu / 2 + 1, (a - 1.0) / a)
    )

    D_perp_D_par_ratio = (
        lambda a, nu, B0, dB_slab, dB_2D, epsilon_slab, epsilon_2D: 2
        * a ** 2
        * C_slab(nu)
        * (
            (dB_slab / B0) ** 2 * epsilon_slab * E_integral(nu, epsilon_slab)
            + (dB_2D / B0) ** 2 * epsilon_2D * E_integral(nu, epsilon_2D)
        )
    )

    epsilon_2D_f = (
        lambda l_2D, lambda_parallel, lambda_perp: 3
        * l_2D ** 2
        / (lambda_parallel * lambda_perp)
    )
    epsilon_slab_f = lambda l_slab, lambda_parallel: epsilon_2D_f(
        l_slab, lambda_parallel, lambda_parallel
    )

    ratio = 1
    new_ratio = 1e99
    e = 1e-5 # precision
    while np.fabs(ratio - new_ratio) > e:
        ratio = new_ratio
        new_ratio = D_perp_D_par_ratio(
            a,
            nu,
            B0,
            dB_slab,
            dB_2D,
            epsilon_slab_f(l_slab, lambda_parallel),
            epsilon_2D_f(l_2D, lambda_parallel, ratio * lambda_parallel),
        )

    return ratio

## NLGC for generalized slab/2D

def D_perp_D_par_ratio_generalized_slab2D_NLGC(
        a, 
        q, s, 
        B0, dB_slab, dB_2D,
        l_slab, l_2D,
        D_parallel):

    lambda_parallel = 3 * D_parallel / c_light

    O_integral = (
        lambda q, s, alpha:
        (s - 1) / (s + q)
        * scipy.special.hyp2f1(1.0, (q + 1)/2, (s + q) / 2 + 1, 1.0 - alpha)
    )

    D_perp_D_par_ratio = (
        lambda a, q, s, B0, dB_slab, dB_2D, alpha_slab, alpha_2D:
        a ** 2 / 2.0
        * (
            (dB_slab / B0) ** 2 * O_integral(q, s, alpha_slab)
            + 2*(dB_2D / B0) ** 2 * O_integral(q, s, alpha_2D)
        )
    )

    alpha_2D_f = (
        lambda l_2D, lambda_parallel, lambda_perp:
        (lambda_parallel * lambda_perp)
        / (3 * l_2D**2)
    )
    alpha_slab_f = lambda l_slab, lambda_parallel: alpha_2D_f(
        l_slab, lambda_parallel, lambda_parallel
    )

    ratio = 1
    new_ratio = 1e99
    e = 1e-5 # precision
    while np.fabs(ratio - new_ratio) > e:
        ratio = new_ratio
        new_ratio = D_perp_D_par_ratio(
            a,
            q,
            s,
            B0,
            dB_slab,
            dB_2D,
            alpha_slab_f(l_slab, lambda_parallel),
            alpha_2D_f(l_2D, lambda_parallel, ratio * lambda_parallel),
        )

    return ratio
