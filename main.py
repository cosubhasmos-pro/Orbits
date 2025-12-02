import numpy as np
import emcee
import pandas as pd
import matplotlib.pyplot as plt

from orbital_models import rv_model
from data_loader import S2DataLoader
from mcmc_utils import log_posterior
from plot_results import plot_rv_curve, plot_corner, plot_orbit

# Synthetic data generation
np.random.seed(42)
t = np.linspace(0, 10, 50)  # days
true_params = {
    'P': 5.0,
    'tp': 0.5,
    'e': 0.2,
    'omega': 0.5,
    'K': 20.0,
    'gamma': 3.0
}

rv_true = rv_model(t, **true_params)
sigma = 2.0  # m/s
rv_obs_syn = rv_true + sigma * np.random.randn(len(t))
rv_err_syn = sigma * np.ones_like(t)

# Plot synthetic RV
plot_rv_curve(t, rv_obs_syn, rv_err_syn, rv_true)

# Load real S2 data
s2_data = S2DataLoader()
astro_df, vel_df = s2_data.load_data()
print(astro_df.head(3))
print(vel_df.head(3))

# Plot S2 data
fig = s2_data.plot_all()
fig.savefig('s2_data.pdf')
plt.show()

data = s2_data.get_data()
t_ast = data['t_ast']
ra_obs = data['ra_obs']
dec_obs = data['dec_obs']
ra_err = data['ra_err']
dec_err = data['dec_err']
t_rv = data['t_rv']
rv_obs = data['rv_obs']  # km/s
rv_err = data['rv_err']  # km/s

# Initial guess for S2
p0_guess = np.array([
    16.0,      # P (yr)
    2018.379,  # tp (yr)
    0.884,     # e
    1.16,      # omega (rad)
    4.0,       # Omega (rad)
    2.35,      # i (rad)
    0.125,     # a (arcsec)
    2630.0,    # K (km/s)
    0.0,       # gamma (km/s)
    0.0,       # x0 (arcsec)
    0.0        # y0 (arcsec)
])

# Run MCMC for S2
ndim = len(p0_guess)
nwalkers = 264
nsteps = 20000

pos = p0_guess + 1e-4 * np.random.randn(nwalkers, ndim)
pos[:, 2] = np.clip(pos[:, 2], 0.8, 0.95)  # e
pos[:, 3:6] %= 2*np.pi  # angles

print("Starting joint astrometric + RV fit of S2 — this will take 5–10 minutes...")
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_posterior,
    args=(t_ast, ra_obs, dec_obs, ra_err, dec_err, t_rv, rv_obs, rv_err)
)
sampler.run_mcmc(pos, nsteps, progress=True)

# Results
samples = sampler.get_chain(discard=2000, thin=80, flat=True)
print(f"Final samples: {samples.shape}")

labels = [r"$P$ (yr)", r"$t_p$ (yr)", r"$e$", r"$\omega$ (rad)", 
          r"$\Omega$ (rad)", r"$i$ (rad)", r"$a$ (arcsec)", r"$K$ (km/s)", r"$\gamma$ (km/s)",
          r"$x_0$ (arcsec)", r"$y_0$ (arcsec)"]

# Plot corner
plot_corner(samples, labels, p0_guess)

# Best-fit model and plot
med = np.median(samples, axis=0)
t_dense = np.linspace(min(t_ast.min(), t_rv.min()) -1, max(t_ast.max(), t_rv.max()) +2, 2000)
plot_orbit(ra_obs, dec_obs, ra_err, dec_err, t_rv, rv_obs, rv_err, med, t_dense)

print("\nJOINT FIT RESULTS (S2 around Sgr A*):")
print(f"Period           = {med[0]:.4f} yr")
print(f"Periastron       = {med[1]:.4f}")
print(f"Eccentricity     = {med[2]:.4f}")
print(f"a (arcsec)       = {med[6]:.5f}")
print(f"K (km/s)         = {med[7]:.1f}")

# Mass estimate (assuming D=8000 pc)
D_pc = 8000
a_AU = med[6] * D_pc
M_bh = a_AU**3 / med[0]**2
print(f"Mass of SMBH     = {M_bh:.1e} M_sun")
