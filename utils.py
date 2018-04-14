import numpy as np


def sample_inclination_deg():
    success = 0
    while success == 0:
        inclination = np.random.rand()*90.
        trial = np.random.rand()
        if np.sin(inclination*np.pi/180.) > trial:
            success = 1
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