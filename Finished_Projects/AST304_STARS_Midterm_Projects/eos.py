########################################################################
# Team Flat Earth Society: Brock Wallin, Lucas Clatterbaugh, Daniel Weatherspoon, Connor Gloden
# AST 304, Fall 2022
# Michigan State University
########################################################################
"""
Routines to compute an adiabatic equation of state.
"""
# imports
import numpy as np

def mean_molecular_weight(Z,A,X):
    """Computes the mean molecular weight for a fully ionized plasma with an 
    arbitrary mixture of species
    
    Arguments
        Z, A, X (either scaler or array)
            charge numbers, atomic numbers, and mass fractions
            The mass fractions must sum to 1
    """
    # initialize arrays
    Zs = np.array(Z)
    As = np.array(A)
    Xs = np.array(X)
    assert np.sum(Xs) == 1.0
    
    # compute value of mean molecular weight
    mu = (sum((Xs/As)*(Zs+1)))**-1 
    return mu
    
def get_rho_and_T(P,P_c,rho_c,T_c):
    """
    Compute density and temperature along an adiabat of index gamma given a 
    pressure and a reference point (central pressure, density, and temperature).
    
    Arguments
        P (either scalar or array-like)
            value of pressure
    
        P_c, rho_c, T_c
            reference values; these values should be consistent with an ideal 
            gas EOS [units: Pa, kg/m^3, K]
    
    Returns
        density, temperature [units: kg/m^3, K]
    """

    # adiabatic constant
    gamma = 5/3
   
    # solving for current density and temperature
    rho = rho_c*((P/P_c))**(1/gamma)
    T = T_c*((P/P_c))**(1-(1/gamma))
    return rho, T
