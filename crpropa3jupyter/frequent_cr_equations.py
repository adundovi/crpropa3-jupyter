from crpropa import *

def get_rigidity(E : "energy", Z: "charge" = 1) -> float: return E / (Z * eplus)
def get_energy_from_rigidity(R: "rigidity", Z: "charge" = 1) -> float: return R * Z * eplus
def get_gyroradius(R: "rigidity", B: "magnetic field") -> float: return R / (c_light * B)
def get_rigidity_from_gyroradius(r_g: "gyroradius", B: "magnetic field") -> float: return r_g * (c_light * B)
def get_Lorenz_factor(E: "energy", m: "mass" = mass_proton) -> float:
    if E < m * c_squared: raise Exception("Error in input arguments: E_tot < mass")
    else: return (E / (m * c_squared))
def get_gyrofrequency(B: "magnetic field",  gamma: "Lorentz factor" = 1,
        Z: "charge" = 1, m: "mass" = mass_proton) -> float: return B * Z * eplus / (m * gamma)
def get_mean_free_path(D: "diffusion coefficient", v: "speed" = c_light) -> float: return  3 * D / v
