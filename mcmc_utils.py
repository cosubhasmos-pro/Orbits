import numpy as np

def log_prior(theta):
    """
    Log prior for S2 joint fit parameters.
    
    Parameters:
    theta: array [P, tp, e, omega, Omega, i, a, K, gamma, x0, y0]
    
    Returns:
    log_prior: float (0 or -inf)
    """
    P, tp, e, omega, Omega, i, a, K, gamma, x0, y0 = theta
    if not (14 < P < 18 and 2017 < tp < 2020 and 0.8 < e < 0.95 and
            0 < omega < 2*np.pi and 0 < Omega < 2*np.pi and 0 < i < np.pi and
            0.1 < a < 0.15 and 1000 < K < 5000 and -100 < gamma < 100 and
            -0.01 < x0 < 0.01 and -0.01 < y0 < 0.01):
        return -np.inf
    return 0.0

def log_likelihood(theta, t_ast, ra_obs, dec_obs, ra_err, dec_err, t_rv, rv_obs, rv_err):
    """
    Log likelihood for S2 joint astrometric + RV fit.
    
    Parameters:
    theta: array [P, tp, e, omega, Omega, i, a, K, gamma, x0, y0]
    t_ast, ra_obs, dec_obs, ra_err, dec_err: astrometry data
    t_rv, rv_obs, rv_err: RV data
    
    Returns:
    log_likelihood: float
    """
    from orbital_models import sky_projected_orbit
    P, tp, e, omega, Omega, i, a, K, gamma, x0, y0 = theta
    ra_mod, dec_mod, _ = sky_projected_orbit(t_ast, P, tp, e, omega, Omega, i, a, K, gamma, x0, y0)
    _, _, rv_mod = sky_projected_orbit(t_rv, P, tp, e, omega, Omega, i, a, K, gamma, x0, y0)
    
    chi2 = np.sum(((ra_obs - ra_mod) / ra_err)**2) + \
           np.sum(((dec_obs - dec_mod) / dec_err)**2) + \
           np.sum(((rv_obs - rv_mod) / rv_err)**2)
    return -0.5 * chi2

def log_posterior(theta, t_ast, ra_obs, dec_obs, ra_err, dec_err, t_rv, rv_obs, rv_err):
    """
    Log posterior for MCMC.
    
    Parameters: same as log_likelihood
    
    Returns:
    log_posterior: float
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, t_ast, ra_obs, dec_obs, ra_err, dec_err, t_rv, rv_obs, rv_err)
    return lp + ll
