import multiprocessing as mp
import sys
import time
import numpy as np
from HiDisk import HiDisk
from lens_nonparametric_src import LensPoints
from utils import rand_sign, sample_inclination_deg
import pyfits as pf
start = time.time()
log = file('log.txt', 'w'); sys.stdout = log
parm_file = 'sim_param.txt'
src_names = np.loadtxt(parm_file, skiprows=1, dtype=str, usecols=0)
mhi_log_solar_mass, z_src, l_z, l_vsig, l_e, l_theta_e = np.swapaxes(np.loadtxt(parm_file, skiprows=1, dtype=float,
                                                                                usecols=(1, 2, 3, 4, 5, 6)), 0, 1)

grid_length = 10
pix_res_as = 0.1
src_threshold = 0.005
smoothing_height_pix = 3
n_samples_per_src = 100
rcmol_log_mean_sig = [0, 1]
log_mhi_uncertainty = 0.4
offset_bound_arcsec = 0.5
inclination_bound_deg = 90.
position_angle_bound_deg = 180.


def run_sim(src_ind):
    rcmol = np.random.lognormal(rcmol_log_mean_sig[0], rcmol_log_mean_sig[1])
    mhi = np.log10(np.random.lognormal(np.log(10 ** mhi_log_solar_mass[src_ind]), np.log(10 ** log_mhi_uncertainty)))
    x_off = rand_sign()*np.random.rand()*offset_bound_arcsec
    y_off = rand_sign()*np.random.rand()*offset_bound_arcsec
    theta_10 = np.random.rand()*position_angle_bound_deg
    theta_20 = sample_inclination_deg()

    # noinspection PyTypeChecker,PyTypeChecker
    hidisk = HiDisk(grid_length_arcsec=grid_length, n_pix=int(grid_length / pix_res_as),
                    rcmol=rcmol, smoothing_height_pix=smoothing_height_pix,
                    theta_2_0=theta_20, theta_1_0=theta_10,
                    x_off=np.int(x_off/pix_res_as), y_off=np.int(y_off/pix_res_as), log10_mhi=mhi, z_src=z_src[src_ind])

    nonparametric_sim = LensPoints(input_file_name=name + '.input', prefix=name, length_arcsec=grid_length,
                                   pix_res=pix_res_as, lens_v_sigma_kms=l_vsig[src_ind], lens_ellipticity=l_e[src_ind],
                                   lens_position_angle_deg_eastofnorth=l_theta_e[src_ind],
                                   lens_z=l_z[src_ind], source_data=hidisk.twod_disk,
                                   src_threshold=src_threshold, z_src=z_src[src_ind])
    sampling_error = hidisk.twod_disk-pf.getdata(name+'_source.fits')
    orig_total_flux = hidisk.twod_disk.sum()
    sampled_unlensed_flux = pf.getdata(name+'_source.fits').sum()
    summed_error = sampling_error.sum()
    mag = nonparametric_sim.calc_mag()
    return [mag, rcmol, mhi, x_off, y_off, theta_20, theta_10, hidisk.rdisk_arcsec,
            orig_total_flux, sampled_unlensed_flux, summed_error]


for ind, name in enumerate(src_names):
    pool = mp.Pool(processes=16)
    parameter_tracking = np.array([pool.apply(run_sim, args=(ind,)) for sample in range(n_samples_per_src)])
    np.save('%s_parmtrack' % name, parameter_tracking)

print 'time taken', time.time()-start
log.close()
