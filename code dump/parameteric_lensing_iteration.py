from tblenser.lens_parametric_srcs import LensInstance
import numpy as np
from copy import deepcopy
from collections import OrderedDict

#Default parameters
lens_dict = {
    'v_sigma_kms': 262,
    'ellipticity': 0.24,
    'position_angle_deg_eastofnorth': 57.3,
    'zl': 0.095,
    'xmin_arcsec': -3,
    'xmax_arcsec': 3,
    'ymin_arcsec': -3,
    'ymax_arcsec': 3,
    'prefix': 'J1106+5228',
    'pix_res': 0.01,
}
source_dict = {
    'source_type': 'sersic',
    'z': 0.407,
    'density_normalisation': 1,
    'ellipticity': 0,
    'position_angle': 0,
    'r_e': 0.11,
    'sersic_index': 0.2,
}


#varying parameters
N = 2
x_offset = np.linspace(-2, 2, N)
y_offset = np.linspace(-2, 2, N)

mag = np.ones((N,N))
#iteration
for indx, off_x in enumerate(x_offset):
    for indy, off_y in enumerate(y_offset):
        source_dict["x_off"] = off_x
        source_dict["y_off"] = off_y
        lens_dict['prefix'] = 'test_sie_%.2f_x_%.2f_y' % (off_x, off_y)
        lens = LensInstance(lens_dict, source_dict, 'slacs_%.2f_x_%.2f_y' % (off_x, off_y)+'.input')
        mag[indx, indy] = lens.write_and_run()

