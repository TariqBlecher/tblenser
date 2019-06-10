import sys
import time
import numpy as np
from tblens.HiDisk import HiDisk
from tblens.lens_nonparametric_src import LensPoints
from tblens.utils import rand_sign, sample_inclination_deg, einstein_radius
from astropy.io import fits


start = time.time()
log = file('log.txt', 'w'); sys.stdout = log
parm_file = 'sim_param.txt'
src_names = np.loadtxt(parm_file, skiprows=1, dtype=str, usecols=0)
mhi_log, z_src, l_z, l_vsig, l_e, l_theta_e, impact_parameter_min, impact_parameter_max, u_l_vsig, u_mhi = \
    np.swapaxes(np.loadtxt(parm_file, skiprows=1, dtype=float,
                           usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)), 0, 1)

mass_sampling = 'uniform'
npix = 100
src_threshold = 0.01 # the outer edges won't be magnified much anyway
smoothing_height_pix = False  # # Without smoothness, gaussian sampler doesn't work as nicely.
n_samples_per_src = 10000
rcmol_log_mean_sig = [-0.1, 0.3]
inclination_bound_deg = 90.
position_angle_bound_deg = 180.
for ind, name in enumerate(src_names):
    parameter_tracking = np.zeros((n_samples_per_src, 8))
    flux_error_tracking = np.zeros((n_samples_per_src, 3))
    for sample in range(n_samples_per_src):
        lens_v_sig = np.random.normal(l_vsig[ind], u_l_vsig[ind])
        rcmol = np.random.lognormal(rcmol_log_mean_sig[0], rcmol_log_mean_sig[1])

        if mass_sampling == 'uniform':
            mhi = np.random.rand()*2.5 + 8.5
        elif mass_sampling == 'normal':
            mhi = np.log10(np.random.lognormal(np.log(10 ** mhi_log[ind]), np.log(10 ** u_mhi[ind])))

        x_off = rand_sign()/np.sqrt(2) * (impact_parameter_min[ind] +
                                          np.random.rand() * (impact_parameter_max[ind] - impact_parameter_min[ind]))
        y_off = rand_sign()/np.sqrt(2) * (impact_parameter_min[ind] +
                                          np.random.rand() * (impact_parameter_max[ind] - impact_parameter_min[ind]))
        theta_10 = np.random.rand()*position_angle_bound_deg
        theta_20 = sample_inclination_deg()
        grid_size_min = 4*einstein_radius(lens_v_sig, z_src[ind], l_z[ind])
        # noinspection PyTypeChecker,PyTypeChecker
        hidisk = HiDisk(n_pix=npix,
                        rcmol=rcmol, smoothing_height_pix=smoothing_height_pix,
                        theta_2_0=theta_20, theta_1_0=theta_10,
                        x_off_arcsec=x_off, y_off_arcsec=y_off, log10_mhi=mhi, z_src=z_src[ind], grid_size_min_arcsec=grid_size_min)

        prefix = name+'_%s' % sample
        nonparametric_sim = LensPoints(input_file_name=name + '.input', prefix=prefix,
                                       length_arcsec=hidisk.grid_length_arcsec,
                                       pix_res=hidisk.pixel_scale_arcsec, lens_v_sigma_kms=lens_v_sig,
                                       lens_ellipticity=l_e[ind],
                                       lens_position_angle_deg_eastofnorth=l_theta_e[ind],
                                       lens_z=l_z[ind], source_data=hidisk.twod_disk,
                                       src_threshold=src_threshold, z_src=z_src[ind])
        sampling_error = hidisk.twod_disk-fits.getdata(prefix+'_source.fits')
        orig_total_flux = hidisk.twod_disk.sum()
        sampled_unlensed_flux = fits.getdata(prefix+'_source.fits').sum()
        summed_error = sampling_error.sum()

        mag = nonparametric_sim.calc_mag()
        parameter_tracking[sample] = [mag, rcmol, mhi, x_off, y_off, theta_20, theta_10, hidisk.rdisk_arcsec]
        flux_error_tracking[sample] = [orig_total_flux, sampled_unlensed_flux, summed_error]
        np.save('hidisk_twodisk_%s_%s' % (name, sample), hidisk.twod_disk)

    np.save('%s_parmtrack' % name, parameter_tracking)
    np.save('%s_errortrack' % name, flux_error_tracking)

print 'time taken', time.time()-start
log.close()
