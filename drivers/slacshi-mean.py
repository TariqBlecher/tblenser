import sys
import time
import numpy as np
from tblens.HiDisk import HiDisk
from tblens.lens_nonparametric_src import LensPoints


start = time.time()
log = file('log.txt', 'w'); sys.stdout = log
parm_file = 'sim_param_test.txt'
src_names = np.loadtxt(parm_file, skiprows=1, dtype=str, usecols=0)
mhi_log_solar_mass, z_src, l_z, l_vsig, l_e, l_theta_e, impact_parameter = np.swapaxes(np.loadtxt(parm_file, skiprows=1, dtype=float,
                                                                                usecols=(1, 2, 3, 4, 5, 6, 7)), 0, 1)

grid_length = 10
pix_res_as = 0.1
src_threshold = 0.005
smoothing_height_pix = 3
n_samples_per_src = 1
rcmol_log_mean_sig = [0, 1]

for ind, name in enumerate(src_names):
    rcmol = 10**rcmol_log_mean_sig[0]
    mhi = mhi_log_solar_mass[ind]
    x_off = 0
    y_off = 0
    theta_10 = 45.
    theta_20 = 90.

    # noinspection PyTypeChecker,PyTypeChecker
    hidisk = HiDisk(grid_length_arcsec=grid_length, n_pix=int(grid_length / pix_res_as),
                    rcmol=rcmol, smoothing_height_pix=smoothing_height_pix,
                    theta_2_0=theta_20, theta_1_0=theta_10,
                    x_off_arcsec=np.int(x_off / pix_res_as), y_off_arcsec=np.int(y_off / pix_res_as), log10_mhi=mhi, z_src=z_src[ind])

    nonparametric_sim = LensPoints(input_file_name=name + '.input', prefix=name, length_arcsec=grid_length,
                                   pix_res=pix_res_as, lens_v_sigma_kms=l_vsig[ind], lens_ellipticity=l_e[ind],
                                   lens_position_angle_deg_eastofnorth=l_theta_e[ind],
                                   lens_z=l_z[ind], source_data=hidisk.twod_disk,
                                   src_threshold=src_threshold, z_src=z_src[ind])

print 'time taken', time.time()-start
log.close()
