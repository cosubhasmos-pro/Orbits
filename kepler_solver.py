import numpy as np

def solve_kepler_equation(M, e, tol=1e-12, max_iter=100):
    """
    Solve Kepler's equation E - e*sin(E) = M using Newton-Raphson.
    Handles array inputs for M.
    
    Parameters:
    M: mean anomaly (radians, scalar or array)
    e: eccentricity (scalar)
    tol: tolerance
    max_iter: max iterations
    
    Returns:
    E: eccentric anomaly (radians, same shape as M)
    """
    M = np.asarray(M)
    sin_M = np.sin(M)
    sin_M_plus_e = np.sin(M + e)
    E = M + (e * sin_M) / (1 - sin_M_plus_e + sin_M)
    
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        delta = f / f_prime
        E -= delta
        if np.all(np.abs(delta) < tol):
            break
    return E

def true_anomaly(E, e):
    """
    Compute true anomaly from eccentric anomaly.
    
    Parameters:
    E: eccentric anomaly (radians)
    e: eccentricity
    
    Returns:
    f: true anomaly (radians)
    """
    sqrt_term_num = np.sqrt(1 + e)
    sqrt_term_den = np.sqrt(1 - e)
    return 2 * np.arctan2(sqrt_term_num * np.sin(E / 2), sqrt_term_den * np.cos(E / 2))
