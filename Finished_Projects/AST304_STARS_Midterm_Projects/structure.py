########################################################################
# Team Flat Earth Society: Brock Wallin, Lucas Clatterbaugh, Daniel Weatherspoon, Connor Gloden
# AST 304, Fall 2022
# Michigan State University
########################################################################
""" 
Routines for computing structure of fully convective star of a given mass and 
radius.
"""
# imports
import numpy as np
from eos import get_rho_and_T, mean_molecular_weight
from ode import rk4
from astro_const import G, Msun, Rsun, Lsun, kB, m_u, fourpi
from reactions import pp_rate

def central_thermal(m, r, mu):
    """ 
    Computes the central pressure, density, and temperature from the polytropic
    relations for n = 3/2.

    Arguments
        m
            final mass in solar units
        r
            final radius is solar units
        mu
            mean molecular weight
    Returns
        Pc, rhoc, Tc
            central pressure, density, and temperature in solar units
    """
    # converting to SI
    m = m * Msun
    r = r * Rsun
    
    # Solving for central values given ratios from Project 2
    Pc = 0.77*((G*(m**2))/(r**4))
    rhoc = 5.99*((3*m)/(fourpi*(r**3)))
    Tc = 0.54*((mu*m_u)/kB)*((G*m)/r)
    
    return Pc, rhoc, Tc

def stellar_derivatives(m, z, mu, Pc, rhoc, Tc, XH, pp_factor):
    """
    RHS of Lagrangian differential equations for radius and pressure
    
    Arguments
        m
            current value of the mass [kg]
        z (array)
            current values of (radius, pressure, luminosity) [kg, Pa, W]
        Pc
            central pressure [Pa]
        rhoc
            central density [kg/m^2]
        Tc
            central temperature [K]
        XH
            mass fraction hydrogen
        pp_factor
            scaling factor to change nuclear reaction rate     
        
    Returns
        dzdm (array)
            Lagrangian derivatives dr/dm, dP/dm, dL/dm
    """
    # initialize output array
    dzdm = np.zeros_like(z)
    
    # solve for density and temperature at step
    rho, T = get_rho_and_T(z[1],Pc,rhoc,Tc)

    #dr/dm
    dzdm[0] = (fourpi*z[0]**2*rho)**(-1)

    #dP/dm
    dzdm[1] = (-G*m/(fourpi*z[0]**4))

    #dL/dm
    dzdm[2] = pp_rate(T,rho,XH,pp_factor)
    
    return dzdm

def central_values(Pc, rhoc, Tc, delta_m, mu, XH, pp_factor):
    """
    Constructs the boundary conditions at the edge of a small, constant density 
    core of mass delta_m with central pressure P_c
    
    Arguments
        Pc
            central pressure [Pa]
        rhoc
            central density [kg/m^2]
        Tc
            central temperature [K]
        delta_m
            core mass (units = Kilograms)
        mu
            mean molecular weight
        XH
            mass fraction hydrogen
        pp_factor
            scaling factor to change nuclear reaction rate   
    
    Returns
        z = array([ r, p, L ])
            central values of radius and pressure and luminosity [kg, Pa, W]
    """
    # initialize z
    z = np.zeros(3)
    
    # compute initial values of z = [ r, p, L ]
    z[0] = ((3*delta_m*Msun)/(fourpi*rhoc))**(1/3)    
    z[1] = Pc
    z[2] = pp_rate(Tc,rhoc,XH,pp_factor) * delta_m*Msun
    
    return z

def lengthscales(m, z, mu, Pc, rhoc, Tc, XH, pp_factor):
    """
    Computes the radial length scale H_r and the pressure length H_P
    
    Arguments
        m
            current mass coordinate (units = Kilograms)
        z (array)
           [ r, p, L ] (units = Meters and Pascals)
        Pc
            central pressure [Pa]
        rhoc
            central density [kg/m^2]
        Tc
            central temperature [K]
        delta_m
            core mass (units = Kilograms)
        mu
            mean molecular weight
        XH
            mass fraction hydrogen
        pp_factor
            scaling factor to change nuclear reaction rate
    
    Returns
        z/|dzdm| (units = Kilograms and Kilograms)
    """
    
    # solving for dzdm using function
    dzdm = (stellar_derivatives(m,z,mu,Pc, rhoc, Tc, XH,pp_factor))
    
    # solving for lengthscales
    return z/np.abs(dzdm)

def integrate(Pc, rhoc, Tc, delta_m, eta, xi, mu, XH, pp_factor, max_steps=10000):
    """
    Integrates the scaled stellar structure equations

    Arguments
        Pc
            central pressure [Pa]
        rhoc
            central density [kg/m^2]
        Tc
            central temperature [K]
        delta_m
            initial offset from center (units = Kilograms)
        eta
            The integration stops when P < eta * Pc
        xi
            The stepsize is set to be xi*min(p/|dp/dm|, r/|dr/dm|)
        mu
            mean molecular weight       
        XH
            mass fraction hydrogen
        pp_factor
            scaling factor to change nuclear reaction rate
        max_steps
            solver will quit and throw error if this more than max_steps are 
            required (default is 10000)
                        
    Returns
        m_step, r_step, p_step, L_step
            arrays containing mass coordinates, radii and pressures during 
            integration [kg, m, Pa, W]
    """
    # initializing arrays of m, r, and p    
    m_step = np.zeros(max_steps)
    r_step = np.zeros(max_steps)
    p_step = np.zeros(max_steps)
    L_step = np.zeros(max_steps)
    
    # set starting conditions using central values
    m = delta_m*Msun
    
    # solve for central values
    z = central_values(Pc,rhoc,Tc, delta_m, mu, XH, pp_factor)
    
    # beginning integration loop
    Nsteps = 0
    for step in range(max_steps):
        # saving values
        radius = z[0]
        pressure = z[1]
        Luminosity = z[2]

        # are we at the surface?
        if (pressure < eta*Pc):
            break
        
        # store the step of current values
        m_step[step] = m 
        r_step[step] = radius
        p_step[step] = pressure
        L_step[step] = Luminosity
        
        # set the stepsize
        h = xi*min(lengthscales(m,z,mu,Pc,rhoc,Tc,XH,pp_factor))
        
        # take a step using RK4 and adding mass change
        z = rk4(stellar_derivatives,m,z,h,args=(mu,Pc,rhoc,Tc,XH,pp_factor))
        m = m + h

        # increment the counter
        Nsteps += 1
    # if the loop runs to max_steps, then signal an error
    else:
        raise Exception('too many iterations')
        
    return m_step[0:Nsteps],r_step[0:Nsteps],p_step[0:Nsteps],L_step[0:Nsteps]
