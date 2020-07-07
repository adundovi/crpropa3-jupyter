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

def calc_diffusion_from_accumulators(accumulators):
    """ calc correlators D_ij_t
    returns dict("t" : time, "x": Dxx,"y": Dyy, "z": Dzz) """
    list_of_t = np.zeros(len(accumulators))
    list_of_D_xx = np.zeros(len(accumulators))
    list_of_D_yy = np.zeros(len(accumulators))
    list_of_D_zz = np.zeros(len(accumulators))

    for i, a in enumerate(accumulators):
        if len(a["output"]) == 0:
            continue

        time = a["dist"]/c_light              
        list_of_deltaDist = [
            c.source.getPosition() - c.current.getPosition()
            for c in a["output"]]

        calc_diff = lambda axis, dXs=list_of_deltaDist, t=time:\
            np.mean(np.array([getattr(dX, axis)**2 for dX in dXs]))/(2*t)    
                
        list_of_t[i] = time
        list_of_D_xx[i] = calc_diff('x')
        list_of_D_yy[i] = calc_diff('y')
        list_of_D_zz[i] = calc_diff('z')

    return {
        "t": list_of_t,
        "x": list_of_D_xx,
        "y": list_of_D_yy,
        "z": list_of_D_zz
    }

def find_D_by_convergence_in_half(list_of_D):
    n_half = int(len(list_of_D)/2)
    
    if n_half == 1:
        raise ValueError("Running diffusion coefficient does not converge on in the given range.")
    
    first_mean = np.mean(list_of_D[0:n_half])
    second_mean = np.mean(list_of_D[n_half:])
    second_std = np.std(list_of_D[n_half:])
    
    if second_mean - 2*second_std > first_mean:
        return find_D_by_convergence_in_half(list_of_D[n_half:])
    return second_mean, second_std

# Diffusion: general expressions

def meanfreepath_from_D(D, v = c_light):
    return 3 * D / v

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
        l_slab: "length_slab",
        B0: "mean_mfield",
        Brms: "rms_mfield",
        ratio: "R_g/L_slab") -> float:

    return (c_light ** 2 / 8
        * scipy.integrate.quad(
            lambda mu: (1 - mu ** 2) ** 2 / D_mumu_slab_SOQLT(nu, l_slab, B0, Brms, ratio, mu),
            -1,
            1,
        )[0]
    )


# SECOND METHOD
# EQ. 6.61 Shalchi - INTEGRAND
def D_mumu_Int_2(s, B0, dB, ratio, mu, x):
    # Dmumuint = h_spectrum(s,x)/x*np.exp(-1./(gamma_sub(ratio, B0, dB) *x) ** 2)*np.cosh(2.*mu*ratio/(x*gamma_sub(ratio,B0,dB)**2) )
    
    h_spectrum = lambda nu, x: (1 + x ** 2) ** (-1*nu / 2)
    gamma_sub = lambda ratio, B0, Brms: ratio * (Brms / B0)

    Dmumuint = (
        h_spectrum(s, x)
        / x
        * (
            np.exp(
                -1.0 / (gamma_sub(ratio, B0, dB) * x) ** 2
                - 2.0 * mu * ratio / (x * gamma_sub(ratio, B0, dB) ** 2)
            )
            + np.exp(
                -1.0 / (gamma_sub(ratio, B0, dB) * x) ** 2
                + 2.0 * mu * ratio / (x * gamma_sub(ratio, B0, dB) ** 2)
            )
        )
    )
    return Dmumuint


# Eq. 6.61 Shalchi
def D_mumu_SOQLT_2(s, l_slab, B0, dB, ratio, mu, lim1, lim2):
    muexp = np.exp(-(mu * B0 / dB) ** 2)
    muint = scipy.integrate.quad(
        lambda x: D_mumu_Int_2(s, B0, dB, ratio, mu, x), lim1, lim2, limit=100
    )[0]
    # print(muexp,muint)
    return (
        np.sqrt(np.pi)
        * C_slab(s)
        / l_slab
        * c_light
        * (1 - mu ** 2)
        * ratio ** (-2)
        * (dB / B0)
        * muexp
        * muint
    )


def diff_coeff_SOQLT_slab_2(
    s: "spectral_index",
    l_slab: "slab_length",
    B0: "mean_mfield",
    dB: "rms_mfield",
    ratio: "R_g/L_c",
    lim1,
    lim2,
):
    #  print(c_light,s,l_slab,B0,dB,ratio)
    return (
        c_light ** 2
        / 8
        * scipy.integrate.quad(
            lambda mu: (1 - mu ** 2) ** 2
            / D_mumu_SOQLT_2(s, l_slab, B0, dB, ratio, mu, lim1, lim2),
            -1,
            1,
            limit=100,
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

## NLGC for isotropic case

def D_perp_D_par_ratio_isotropic_NLGC(
        a, s, 
        B0, dB_iso,
        l_iso,
        D_parallel):

    lambda_parallel = 3 * D_parallel / c_light

    D_perp_D_par_ratio = (
        lambda a, s, B0, dB_iso, R, alpha_parallel, alpha_perp:
        a ** 2 / 2.0
        * C_iso(s)
        * (dB_iso / B0) ** 2
        * (
            int_I(s, alpha_parallel*R, alpha_parallel) + 
            0 #2*R * int_J(s, alpha_parallel*R, alpha_perp)
        )
    )

    int_I = (
        lambda s, A1, A2:
        scipy.integrate.quad(lambda x_par:
            scipy.integrate.quad(lambda x_per:
                x_per * (2*x_par**2 + x_per**2) *
                (1 + x_par**2 + x_per**2)**(-s/2.-2) /
                (1 + A1*x_per**2 + A2*x_par**2),
                0, np.inf)[0],
                -np.inf, np.inf)[0]
    )
    
    int_J = (
        lambda s, A1, A2:
        scipy.integrate.quad(lambda x_par:
            scipy.integrate.quad(lambda x_per:
                x_per**3 *
                (1 + x_par**2 + x_per**2)**(-s/2.-2) /
                (1 + A1*x_par**2 + A2*x_per**2),
                0, np.inf)[0],
                -np.inf, np.inf)[0]
    )
       
    alpha_f = (
        lambda l_iso, lambda_:
        (lambda_**2)
        / (3 * l_iso**2)
    )

    ratio = 1
    new_ratio = 1e99
    e = 1e-2 # precision
    while np.fabs(ratio - new_ratio)/ratio > e:
        ratio = new_ratio
        new_ratio = D_perp_D_par_ratio(
            a,
            s,
            B0,
            dB_iso,
            ratio,
            alpha_f(l_iso, lambda_parallel),
            alpha_f(l_iso, ratio * lambda_parallel),
        )

    return ratio

## UNLT for isotropic case

def D_perp_D_par_ratio_isotropic_UNLT(
        a, s, 
        B0, dB_iso,
        l_iso,
        D_parallel):

    lambda_parallel = 3 * D_parallel / c_light

    D_perp_D_par_ratio = (
        lambda a, s, B0, dB_iso, R, alpha_parallel:
        a ** 2 / 2.0
        * C_iso(s)
        * (dB_iso / B0) ** 2
        * (
            int_I(s, alpha_parallel, R) 
        )
    )

    int_I = (
        lambda s, A1, R:
        scipy.integrate.quad(lambda x_per:
            scipy.integrate.quad(lambda x_par:
                x_per**3 * R * (2*x_par**2 + x_per**2) *
                (1 + x_par**2 + x_per**2)**(-s/2.-2) /
                (x_per**2 * R * (1 + 4/3.* A1 * R * x_per**2) + (x_par)**2),
                -np.inf, np.inf, limit=200)[0],
                0, np.inf, limit=200)[0]
    )
    
    alpha_f = (
        lambda l_iso, lambda_:
        (lambda_**2)
        / (3 * l_iso**2)
    )

    ratio = 1
    new_ratio = 1e20
    e = 1e-2 # precision
    while np.fabs(ratio - new_ratio)/ratio > e:
        ratio = new_ratio
        new_ratio = D_perp_D_par_ratio(
            a,
            s,
            B0,
            dB_iso,
            ratio,
            alpha_f(l_iso, lambda_parallel)
        )

    return ratio


def diff_coeff_Subeid_HE(L_c, r_g):
    return r_g**2 * c_light / (2 * L_c)

def diff_coeff_Subedi_QLT(s, l_bo, r_g):
    A_factor = lambda s: (2/3.)**(s/2.) * s*(s+2)/(s+1) * \
        scipy.special.gamma((s-1)/2.) / (scipy.special.gamma(s/2+2) * scipy.special.gamma((3-s)/2))
    return c_light * l_bo / 8 * A_factor(s) * (r_g / l_bo)**(2-s)

def diff_coeff_Subedi_NLGC(s, l_bo, r_g):
    A_factor = lambda s: (2/3.)**(s/2.) * s*(s+2)/(s+1) * \
        scipy.special.gamma((s-1)/2.) / (scipy.special.gamma((s+4)/2) * scipy.special.gamma((3-s)/2))
    B_factor = lambda s: 3**2 / 2**10 * 1 / ((s-2)*(s+1)) * \
        (2**5 * s*(4*s-5) - 221 + 3**(1-s) * (13+8*s) * scipy.special.hyp2f1(-1/2.,(3-s)/2, 1/2., -8))
    return c_light * l_bo / 8 * A_factor(s) / B_factor(s) * (r_g / l_bo)**(2-s)

def diff_coeff_QLT_generic(s, Bz, Brms, l_bo, l_c, r_g):
    if Bz == 0:
        if r_g > l_bo:
            return diff_coeff_Subeid_HE(l_c, r_g)
        else:
            return diff_coeff_Subedi_QLT(s, l_bo, r_g)
    else:
        return diff_coeff_slab_QLT(s, l_bo, Bz, Brms, r_g / l_bo)

def k_perp2k_xx_iso(k_perp):
    return k_perp/3.
