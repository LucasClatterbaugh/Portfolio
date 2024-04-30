########################################################################
# Team Flat Earth Society: Brock Wallin, Lucas Clatterbaugh, Daniel Weatherspoon, Connor Gloden
# AST 304, Fall 2022
# Michigan State University
########################################################################
"""
Values of physical constants imported from astropy
"""

import astropy.constants as _ac
import astropy.units as _au

# mathematical constants
from numpy import pi
fourpi = 4.0*pi

# solar mass, radius, luminosity
Msun = _ac.M_sun.value
Rsun = _ac.R_sun.value
Lsun = _ac.L_sun.value

# physical constants from astropy, all in MKS units
G = _ac.G.value
h = _ac.h.value
hbar = _ac.hbar.value
m_e = _ac.m_e.value
m_p = _ac.m_p.value
m_n = _ac.m_n.value
m_u = _ac.u.value
c = _ac.c.value
kB = _ac.k_B.value
pc = _ac.pc.value
au = _ac.au.value
year = _au.year.to(_au.second)
sigmaSB = _ac.sigma_sb.value

# new constant used for Pc_guess function in structure.py
Ke = (1/5)*(3/(8*pi))**(2/3)*(h**2/(m_e*m_u**(5/3)))

if __name__ == "__main__":
    
    constants = [
        ("pi",pi,""),
        ("4*pi",fourpi,""),
        ("solar mass",Msun,"kg"),
        ("solar radius",Rsun,"m"),
        ("solar luminosity",Lsun,"W"),
        ("gravitational constant",G,"m**3 s**-2 kg**-1"),
        ("Planck constant",h,"J s"),
        ("Planck constant, reduced",hbar,"J s"),
        ("electron mass",m_e,"kg"),
        ("proton mass",m_p,"kg"),
        ("neutron mass",m_n,"kg"),
        ("atomic mass unit",m_u,"kg"),
        ("speed of light",c,"m s**-1"),
        ("Boltzmann constant",kB,"J K**-1"),
        ("parsec",pc,"m"),
        ("astronomical unit",au,"m"),
        ("year",year,"s"),
        ("Stefan-Boltzmann constant",sigmaSB,"W m**-2 K**-4")
    ]
    
    
    for const in constants:
        print('{0[0]:28} = {0[1]:11.4e} {0[2]}'.format(const))
