########################################################################
# Team Flat Earth Society: Brock Wallin, Lucas Clatterbaugh, Daniel Weatherspoon, Connor Gloden
# AST 304, Fall 2022
# Michigan State University
########################################################################
"""
Module for computing rate of nuclear reaction in star given properties of star.
"""
# imports
import numpy as np

def pp_rate(T,rho,XH,pp_factor):
    """
    Specific heating rate from pp chain hydrogen burning. Approximate rate 
    taken from Hansen, Kawaler, & Trimble.
    
    Arguments
        T, rho
            temperature [K] and density [kg/m**3]
        XH
            mass fraction hydrogen
        pp_factor
            multiplicative factor for rate
    Returns
        heating rate from the pp-reaction chain [W/kg]
    """
    
    # solving for rate of nuclear reaction
    Tg = T/1e9
    eps = ((2.4e-3*rho*XH**2)/(Tg**(2/3))) * np.exp(-3.380/(Tg**(1/3)))
    rate = pp_factor * eps
    return rate
