import sys
import time
import numpy as np
from tblens.HiDisk import HiDisk
from tblens.lens_nonparametric_src_rxj import LensPoints
from tblens.utils import rand_sign
import pyfits as pf


start = time.time()
log = file('log.txt', 'w'); sys.stdout = log
parm_file = 'rx_param.txt'
src_names = np.loadtxt(parm_file, skiprows=2, dtype=str, usecols=0)
mhi_log, z_src, l_z, r_ein, l_e, l_theta_e, impact_parameter_min, impact_parameter_max, radial_slope = \
    np.loadtxt(parm_file, skiprows=2, dtype=float, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9))

mass_sampling = 'uniform'
npix = 100
src_threshold = 0.01 # the outer edges won't be magnified much anyway
smoothing_height_pix = False  # # Without smoothness, gaussian sampler doesn't work as nicely.
n_samples_per_src = 10000
rcmol_log_mean_sig = [-0.1, 0.3]
position_angle_bound_deg = 180.
parameter_tracking = np.zeros((n_samples_per_src, 8))
flux_error_tracking = np.zeros((n_samples_per_src, 3))
for sample in range(n_samples_per_src):
    rcmol = np.random.lognormal(rcmol_log_mean_sig[0], rcmol_log_mean_sig[1])
    mhi = np.random.rand()*2.5 + 8

    x_off = rand_sign()/np.sqrt(2) * (impact_parameter_min +
                                      np.random.rand() * (impact_parameter_max - impact_parameter_min))
    y_off = rand_sign()/np.sqrt(2) * (impact_parameter_min +
                                      np.random.rand() * (impact_parameter_max - impact_parameter_min))
    theta_10 = np.random.rand()*position_angle_bound_deg
    theta_20 = np.random.rand()*20+45.
    grid_size_min = 4*r_ein
    hidisk = HiDisk(n_pix=npix,
                    rcmol=rcmol, smoothing_height_pix=smoothing_height_pix,
                    theta_2_0=theta_20, theta_1_0=theta_10,
                    x_off_arcsec=x_off, y_off_arcsec=y_off, log10_mhi=mhi, z_src=z_src,
                    grid_size_min_arcsec=grid_size_min)
    np.save('hidisk_twodisk_RXJ_%s' % sample, hidisk.twod_disk)

    prefix = '_%s' % sample
    nonparametric_sim = LensPoints(input_file_name=prefix + '.input', prefix=prefix,
                                   length_arcsec=hidisk.grid_length_arcsec,
                                   pix_res=hidisk.pixel_scale_arcsec, einstein_radius=r_ein,
                                   lens_ellipticity=l_e,
                                   lens_position_angle_deg_eastofnorth=l_theta_e,
                                   lens_z=l_z, source_data=hidisk.twod_disk,
                                   src_threshold=src_threshold, z_src=z_src)
    sampling_error = hidisk.twod_disk-pf.getdata(prefix+'_source.fits')
    orig_total_flux = hidisk.twod_disk.sum()
    sampled_unlensed_flux = pf.getdata(prefix+'_source.fits').sum()
    summed_error = sampling_error.sum()

    mag = nonparametric_sim.calc_mag()
    parameter_tracking[sample] = [mag, rcmol, mhi, x_off, y_off, theta_20, theta_10, hidisk.rdisk_arcsec]
    flux_error_tracking[sample] = [orig_total_flux, sampled_unlensed_flux, summed_error]

np.save('RXJ_parmtrack', parameter_tracking)
np.save('RXJ_errortrack', flux_error_tracking)

print 'time taken', time.time()-start
log.close()