import numpy as np
from astropy.cosmology import Planck15 as cosmo
from scipy.constants import speed_of_light as c
import glob


def mass_sampling(pdf='uniform', uniform_lower_bound=8.5, uniform_width=2.5, mass_mean_log10=9, mass_sig_log10=1,
                  lower_limit=6., upper_limit=11.):
    if pdf == 'uniform':
        mhi = np.random.rand()*uniform_width + uniform_lower_bound
    elif pdf == 'normal':
        mhi = np.log10(np.random.lognormal(np.log(10 ** mass_mean_log10), np.log(10 ** mass_sig_log10)))
        if mhi < lower_limit:
            mhi = lower_limit
        elif mhi > upper_limit:
            mhi = upper_limit
    elif pdf == 'mean':
        mhi = mass_mean_log10
    return mhi


def sample_z(zgrid=None, pz=None, zmean=None, u_z=None, sampling='mean', zspec=None, zbandedge=1.45, zcluster=None):
    nz = 0
    if zspec > 0:
        z = zspec
    elif sampling == 'mean':
        z = zmean
    elif sampling == 'normal':
        z = 0.0
        while np.logical_or(z <= zcluster, z >= zbandedge):
            z = np.random.normal(zmean, u_z)
            nz += 1
    elif sampling == 'pz':
        z = 0.0
        while np.logical_or(z <= zcluster, z >= zbandedge):
            z = np.random.choice(zgrid, p=pz/pz.sum())
            nz += 1

    return z, nz


def sample_inclination_deg():
    success = 0
    while success == 0:
        inclination = np.random.rand()*90.
        trial = np.random.rand()
        if np.sin(inclination*np.pi/180.) > trial:
            success = 1

    if inclination > 89.:
        inclination = 89.
    return inclination


def rand_signs(size=None):
    r = np.random.rand(size)
    signs = np.ones(size)
    negs = r <= 0.5
    signs[negs] = -1
    return signs


def rand_sign():
    r = np.random.rand()
    if r > 0.5:
        return 1
    else:
        return -1

def turning_points(array):
    ''' turning_points(array) -> min_indices, max_indices
    Finds the turning points within an 1D array and returns the indices of the minimum and
    maximum turning points in two separate lists.
    '''
    idx_max, idx_min = [], []

    NEUTRAL, RISING, FALLING = range(3)
    def get_state(a, b):
        if a < b: return RISING
        if a > b: return FALLING
        return NEUTRAL

    ps = get_state(array[0], array[1])
    begin = 1
    for i in range(2, len(array)):
        s = get_state(array[i - 1], array[i])
        if s != NEUTRAL:
            if ps != NEUTRAL and ps != s:
                if s == FALLING:
                    idx_max.append((begin + i - 1) // 2)
                else:
                    idx_min.append((begin + i - 1) // 2)
            begin = i
            ps = s
    return idx_min, idx_max

def get_parmtracks(folder):
    parmtracklist = np.sort(glob.glob(folder+'*_parmtrack.npy'))
    parmtrack = parmtracklist[0]
    parm = np.load(parmtrack)
    parmtracks = np.zeros(((len(parmtracklist),) + parm.shape))
    for ind, pt in enumerate(parmtracklist):
        parmtracks[ind] = np.load(pt)
        return parmtracks


def einstein_radius(sig_sis_km_s, z_source, z_deflector):
    D_ds = cosmo.angular_diameter_distance_z1z2(z_deflector, z_source).value
    D_s = cosmo.angular_diameter_distance(z_source).value
    return 4*np.pi * (sig_sis_km_s*1e3 / c) ** 2 * D_ds / D_s * 180 / np.pi * 3600