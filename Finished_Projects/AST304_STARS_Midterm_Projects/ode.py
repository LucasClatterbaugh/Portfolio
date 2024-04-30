########################################################################
# Team Flat Earth Society: Brock Wallin, Lucas Clatterbaugh, Daniel Weatherspoon, Connor Gloden
# AST 304, Fall 2022
# Michigan State University
########################################################################

"""
ode.py contains the three different routines for numerical analysis used for the project. They are
- fEuler: Forward Euler method
- rk2: Runge-Kutta Second Order method
- rk4: Runge-Kutta Fourth Order method

This file is used in kepler.py where all the methods below are imported to be used in the numerical
analysis function created there. This file only has to be in the same folder as kepler.py and the
script created to utilize kepler.py.
"""

def fEuler(f,t,z,h,args=()):
    """
    Uses the Forward Euler numerical method to approximate the motion dictated by the input function
    f. This function is called in kepler.py when looping through the analysis time over the time steps
    h in order to numerically solve the function f. To use this function, it must be called and all of
    the arguments listed below must be included.
    
    Arguments:
        f(t,z,...)
            Function that contains the RHS of the equation dz/dt = f(t,z,...).
    
        t (scalar)
            The current time value of the analyis.
            
        z (array like)
            Array of length four that contains the current time value:
            [x position,  y position,  x velocity,  y velocity]
            
        h (scalar)
            Time step size for the analysis.
    
        args (tuple, optional)
            Additional arguments to pass to f.
    
    Returns
        z_step = z(t+h)
    """
    
    # Converting arguments into a tuple
    if not isinstance(args,tuple):
        args = (args,)
    
    # Following Euler's Method
    z_step = z + h*f(t, z, *args)
    
    return z_step

def rk2(f,t,z,h,args=()):
    """
    Uses the Runge-Kutta Second Order numerical method to approximate the motion dictated by the input 
    function f. The method usse the slope at the midpoint of the step to better approximate the value of
    z for the next step. This function is called in kepler.py when looping through the analysis time over 
    the time steps h in order to numerically solve the function f. To use this function, it must be called 
    and all of the arguments listed below must be included.
    
    Arguments
        f(t,z,...)
            Function that contains the RHS of the equation dz/dt = f(t,z,...).
    
        t (scalar)
            The current time value of the analyis.
            
        z (array like)
            Array of length four that contains the current time value:
            [x position,  y position,  x velocity,  y velocity]
            
        h (scalar)
            Time step size for the analysis.
    
        args (tuple, optional)
            Additional arguments to pass to f.
    
    Returns
        z_step = z(t+h)
    """
    
    # Converting arguments into tuple if not already in form
    if not isinstance(args,tuple):
        args = (args,)

    # Solving for the predicted value of our function at the midpoint of the step
    z_p = z + (h/2)*f(t, z, *args)
    
    # Solving for the predicted value of our function after a full time step
    z_step = z + h*f(t+(h/2), z_p, *args)


    return z_step

def rk4(f,t,z,h,args=()):
    """
    Uses the Runge-Kutta Fourth Order numerical method to approximate the motion dictated by the input 
    function f. This is a higher accuracy version of RK2 as more points and slopes are estimated in order
    to approximate z for the next step. This function is called in kepler.py when looping through the 
    analysis time over the time steps h in order to numerically solve the function f. To use this function,
    it must be called and all of the arguments listed below must be included.
    
    Arguments:
        f(t,z,...)
            Function that contains the RHS of the equation dz/dt = f(t,z,...).
    
        t (scalar)
            The current time value of the analyis.
            
        z (array like)
            Array of length four that contains the current time value:
            [x position,  y position,  x velocity,  y velocity]
            
        h (scalar)
            Time step size for the analysis.
    
        args (tuple, optional)
            Additional arguments to pass to f.
    
    Returns
        znew = z(t+h)
    """
    
   # Converting arguments into a tuple
    if not isinstance(args,tuple):
        args = (args,)
    
    # Following the RK4 method
    k1 = f(t, z, *args)
    k2 = f(t+(h/2), z+(h/2)*k1, *args)
    k3 = f(t+(h/2), z+(h/2)*k2, *args)
    k4 = f(t+h, z+h*k3, *args)
    
    #Using the four approximations above to estimate the z array after the step
    z_step = z + (h/6)*(k1+2*k2+2*k3+k4)
    
    return z_step
