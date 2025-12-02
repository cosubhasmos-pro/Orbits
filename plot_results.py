import matplotlib.pyplot as plt
import corner
import numpy as np
from orbital_models import rv_model, sky_projected_orbit

def plot_rv_curve(t, rv_obs, rv_err, rv_true, filename='Rv_curve.pdf'):
    """
    Plot synthetic RV curve.
    """
    plt.figure(figsize=(8, 5))
    plt.errorbar(t, rv_obs, yerr=rv_err, fmt='o', label='Observed RV')
    plt.plot(t, rv_true, 'k-', label='True model')
    plt.xlabel('Time (days)')
    plt.ylabel('RV (m/s)')
    plt.legend()
    plt.title('Synthetic RV Curve')
    plt.savefig(filename)
    plt.show()
'''
def plot_corner(samples, labels, truths, filename='corner_s2_joint.pdf'):
    # Wrap angles (indices 3: omega, 4: Omega, 5: i)
    samples[:, 3:6] %= 2 * np.pi  # Modulo 2pi for radians
    
    # Define ranges to clip outliers (example; adjust per parameter)
    ranges = [(15, 17), (2017, 2020), (0.8, 0.95), (0, 2*np.pi), (0, 2*np.pi), (0, np.pi),
              (0.1, 0.15), (1000, 5000), (-100, 100), (-0.01, 0.01), (-0.01, 0.01)]
    
    fig = corner.corner(
        samples, labels=labels, truths=truths, range=ranges,
        color='dodgerblue', truth_color='crimson', quantiles=[0.16, 0.5, 0.84],
        show_titles=True, title_fmt='.3f', title_kwargs={'fontsize': 14},
        label_kwargs={'fontsize': 16}, hist_kwargs={'linewidth': 1.5},
        smooth=1.5, smooth1d=1.5, fill_contours=True, plot_datapoints=False,
        levels=[0.39, 0.68, 0.95], contour_kwargs={'linewidths': 1.0}
    )
    fig.set_size_inches(14, 14)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()
'''
def plot_corner(samples, labels, truths, filename='corner_s2_joint.pdf'):
    plt.rcParams['text.usetex'] = True  # Enable LaTeX for proper superscript/subscript rendering
    
    # Wrap angles (indices 3: omega, 4: Omega, 5: i)
    samples[:, 3:6] %= 2 * np.pi  # Modulo 2pi for radians
    
    # Define ranges to clip outliers (example; adjust per parameter)
    ranges = [(15.6, 16.7), (2017.6, 2019), (0.84, 0.93), (0.5, 1.7), (3.5, 6.5), (1.6, np.pi),
              (0.11, 0.14), (2000, 3600), (-1, 60), (-0.007, 0.002), (-0.01, 0.01)]
    
    fig = corner.corner(
        samples, labels=labels, truths=truths, range=ranges,
        color='dodgerblue', truth_color='crimson', quantiles=[0.16, 0.5, 0.84],
        show_titles=True, title_fmt='.3f', title_kwargs={'fontsize': 12},  # Reduced fontsize for cleaner look
        label_kwargs={'fontsize': 12}, hist_kwargs={'linewidth': 1.5},  # Reduced fontsize
        smooth=1.5, smooth1d=1.5, fill_contours=True, plot_datapoints=False,
        levels=[0.39, 0.68, 0.95], contour_kwargs={'linewidths': 1.0}
    )
    fig.set_size_inches(16, 16)
    plt.savefig(filename, bbox_inches='tight', dpi=400)
    plt.show()

def plot_orbit(ra_obs, dec_obs, ra_err, dec_err, t_rv, rv_obs, rv_err, med, t_dense, filename='s2_joint_orbit.pdf'):
    """
    Plot best-fit orbit and RV.
    """
    ra_fit, dec_fit, rv_fit = sky_projected_orbit(t_dense, *med)
    
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    plt.errorbar(ra_obs, dec_obs, xerr=ra_err, yerr=dec_err, fmt='o', alpha=0.6, label='S2 Astrometry')
    plt.plot(ra_fit, dec_fit, 'r-', lw=2, label='Best-fit orbit')
    plt.xlabel('RA (arcsec)')
    plt.ylabel('Dec (arcsec)')
    plt.legend()
    plt.title('S2 On-Sky Orbit')
    plt.gca().invert_xaxis()
    plt.axis('equal')
    
    plt.subplot(1,2,2)
    plt.errorbar(t_rv, rv_obs, yerr=rv_err, fmt='o', alpha=0.7, label='RV data')
    plt.plot(t_dense, rv_fit, 'r-', lw=2, label='Best-fit RV')
    plt.axvline(med[1], color='gray', ls='--', label=f'Periastron {med[1]:.3f}')
    plt.xlabel('Time (years)')
    plt.ylabel('RV (km/s)')
    plt.legend()
    plt.title('S2 Radial Velocity')
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()
