import numpy as np
from kepler_solver import solve_kepler_equation, true_anomaly

def rv_model(t, P, tp, e, omega, K, gamma):
    """
    Synthetic RV model (in m/s).
    
    Parameters:
    t: time (days, array)
    P: period (days)
    tp: time of periastron (days)
    e: eccentricity
    omega: argument of periastron (radians)
    K: semi-amplitude (m/s)
    gamma: systemic velocity (m/s)
    
    Returns:
    rv: radial velocity (m/s, array)
    """
    M = 2 * np.pi * (t - tp) / P
    E = solve_kepler_equation(M, e)
    f = true_anomaly(E, e)
    rv = K * (np.cos(f + omega) + e * np.cos(omega)) + gamma
    return rv

def sky_projected_orbit(t, P, tp, e, omega, Omega, i, a, K, gamma, x0, y0):
    """
    Sky-projected orbit model (RV in km/s).
    
    Parameters:
    t: time (years, array)
    P: period (years)
    tp: time of periastron (years)
    e: eccentricity
    omega: argument of periastron (rad)
    Omega: longitude of ascending node (rad)
    i: inclination (rad)
    a: semi-major axis (arcsec)
    K: semi-amplitude (km/s)
    gamma: systemic velocity (km/s)
    x0, y0: offsets (arcsec)
    
    Returns:
    ra, dec: right ascension, declination (arcsec)
    rv: radial velocity (km/s)
    """
    M = 2 * np.pi * (t - tp) / P
    E = solve_kepler_equation(M, e)
    f = true_anomaly(E, e)
    
    X = np.cos(E) - e
    Y = np.sqrt(1 - e**2) * np.sin(E)
    
    A = a * (np.cos(omega) * np.cos(Omega) - np.sin(omega) * np.sin(Omega) * np.cos(i))
    B = a * (np.cos(omega) * np.sin(Omega) + np.sin(omega) * np.cos(Omega) * np.cos(i))
    F = a * (-np.sin(omega) * np.cos(Omega) - np.cos(omega) * np.sin(Omega) * np.cos(i))
    G = a * (-np.sin(omega) * np.sin(Omega) + np.cos(omega) * np.cos(Omega) * np.cos(i))
    
    ra = A * X + F * Y + x0
    dec = B * X + G * Y + y0
    
    rv = K * (np.cos(f + omega) + e * np.cos(omega)) + gamma
    
    return ra, dec, rv
