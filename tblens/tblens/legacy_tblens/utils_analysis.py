import numpy as np
from astropy.cosmology import Planck15 as cosmo
from scipy.constants import speed_of_light as c
import glob


def get_parmtracks(folder):
    """Useful for Analysis."""
    parmtracklist = np.sort(glob.glob(folder + '*_parmtrack.npy'))
    parmtrack = parmtracklist[0]
    parm = np.load(parmtrack)
    parmtracks = np.zeros(((len(parmtracklist),) + parm.shape))
    for ind, pt in enumerate(parmtracklist):
        parmtracks[ind] = np.load(pt)
        return parmtracks


def einstein_radius(sig_sis_km_s, z_source, z_deflector):
    D_ds = cosmo.angular_diameter_distance_z1z2(z_deflector, z_source).value
    D_s = cosmo.angular_diameter_distance(z_source).value
    return 4 * np.pi * (sig_sis_km_s * 1e3 / c) ** 2 * D_ds / D_s * 180 / np.pi * 3600