from crpropa import *

def get_rigidity(E : "energy", Z: "charge" = 1) -> float: return E / (Z * eplus)
def get_energy_from_rigidity(R: "rigidity", Z: "charge" = 1) -> float: return R * Z * eplus
def get_gyroradius(R: "rigidity", B: "magnetic field") -> float: return R / (c_light * B)
def get_rigidity_from_gyroradius(r_g: "gyroradius", B: "magnetic field") -> float: return r_g * (c_light * B)
#def get_gyrofrequency( m: "mass" = mass_proton : return (p_B0*eplus/(ma))
