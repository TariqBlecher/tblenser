import numpy as np
from astropy.cosmology import Planck15 as cosmo


def HI_mass_size_relation(log10_mhi):
    """HI Mass-size relation - Jing Wang et al. 2016"""
    r_1_kpc = 0.5 * 10 ** (0.506 * log10_mhi - 3.293)
    return r_1_kpc


def set_borders_to_zero(array_2d):
    array_2d[:1, :] = 0
    array_2d[:, :1] = 0
    array_2d[-1:, :] = 0
    array_2d[:, -1:] = 0
    return array_2d


def convert_pc_to_arcsec(length_pc, z):
    kpc_per_arcsec_src = cosmo.kpc_proper_per_arcmin(z).value / 60.
    length_arcsec = length_pc * 1e-3 / kpc_per_arcsec_src
    return length_arcsec


def mass_sampling(pdf='uniform', uniform_lower_bound=8.5, uniform_width=2.5, mass_mean_log10=9, mass_sig_log10=1,
                  lower_limit=6., upper_limit=11.):
    if pdf == 'uniform':
        mhi = np.random.rand() * uniform_width + uniform_lower_bound
    elif pdf == 'normal':
        mhi = np.log10(np.random.lognormal(np.log(10 ** mass_mean_log10), np.log(10 ** mass_sig_log10)))
        if mhi < lower_limit:
            mhi = lower_limit
        elif mhi > upper_limit:
            mhi = upper_limit
    elif pdf == 'mean':
        mhi = mass_mean_log10
    return mhi


def sample_z(zmean=None, zspec=None):
    if zspec > 0:
        z = zspec
    else:
        z = zmean
    return z


def sample_inclination_deg():
    success = 0
    while success == 0:
        inclination = np.random.rand() * 90.
        trial = np.random.rand()
        if np.sin(inclination * np.pi / 180.) > trial:
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
    idx_maxima, idx_minima = [], []

    NEUTRAL, RISING, FALLING = range(3)

    def get_state(initial_position, final_position):
        if initial_position < final_position:
            return RISING
        if initial_position > final_position:
            return FALLING
        return NEUTRAL

    pre_state = get_state(array[0], array[1])
    start_index = 1
    for array_index in range(2, len(array)):
        state = get_state(array[array_index - 1], array[array_index])
        if state != NEUTRAL:
            if pre_state != NEUTRAL and pre_state != state:
                if state == FALLING:
                    idx_maxima.append((start_index + array_index - 1) // 2)
                else:
                    idx_minima.append((start_index + array_index - 1) // 2)
            start_index = array_index
            pre_state = state
    return idx_minima, idx_maxima
