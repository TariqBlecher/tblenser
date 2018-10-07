from map_utils import *
from HiDiskFrontier import *
from utils import sample_inclination_deg
import time

log = file('log.txt', 'w'); sys.stdout = log
start = time.time()

parm_file = 'sim_catalog.txt'
xdeflectfits = 'hlsp_frontier_model_abell2744_cats_v4.1_x-arcsec-deflect.fits'
ydeflectfits = 'hlsp_frontier_model_abell2744_cats_v4.1_y-arcsec-deflect.fits'
ra_image_mean, dec_image_mean, z_src, u_zsrc, mhi_log, u_mhi, ra_err_deg, dec_err_deg = \
    np.swapaxes(np.loadtxt(parm_file, skiprows=1, dtype=float, usecols=(0, 1, 2, 3, 4, 5, 6, 7)), 0, 1)
src_indices = np.arange(mhi_log.shape[0])

defmap = DeflectionMap(xdeflect_fits=xdeflectfits, ydeflect_fits=ydeflectfits, center=False)
n_samples_per_src = 1000
rcmol_log_mean_sig = [-0.1, 0.3]
mass_sampling = 'uniform'

for src_ind in src_indices:
    parameter_tracking = np.zeros((n_samples_per_src, 9))
    for sample in range(n_samples_per_src):
        if mass_sampling == 'uniform':
            mhi = np.random.rand()*2.5 + 8.5
        elif mass_sampling == 'normal':
            mhi = np.log10(np.random.lognormal(np.log(10 ** mhi_log[src_ind]), np.log(10 ** u_mhi[src_ind])))
        rcmol = np.random.lognormal(rcmol_log_mean_sig[0], rcmol_log_mean_sig[1])
        theta_10 = np.random.rand()*180.
        theta_20 = sample_inclination_deg()
        z = np.random.normal(z_src[src_ind], u_zsrc[src_ind])

        ra_image_arcsec = np.random.normal(loc=ra_image_mean[src_ind], scale=ra_err_deg[src_ind]) * 3600.
        dec_image_arcsec = np.random.normal(loc=dec_image_mean[src_ind], scale=dec_err_deg[src_ind]) * 3600.
        ra_image_arcsec_rel_to_defmap = ra_image_arcsec - defmap.center[0]
        dec_image_arcsec_rel_to_defmap = dec_image_arcsec - defmap.center[1]
        im_coord = np.array([ra_image_arcsec_rel_to_defmap, dec_image_arcsec_rel_to_defmap]).reshape((2, 1))
        source_coord_arcsec_rel = defmap.calc_source_positions(im_coord, z)[:, 0]
        source_coord_pix = source_coord_arcsec_rel / defmap.pix_scale_arcsec * np.array([-1, 1])

        hidisk = HiDiskCluster(n_pix=defmap.npix, pix_res=defmap.pix_scale_arcsec, rcmol=rcmol, smoothing_height_pix=False,
                               theta_2_0=theta_20, theta_1_0=theta_10, x_off=source_coord_pix[0], y_off=source_coord_pix[1],
                               log10_mhi=mhi, z_src=z)

        mag = defmap.calc_magnification(hidisk.twod_disk, z, write_image=True, imagename='image_%s_%s' % (src_ind, sample))
        parameter_tracking[sample] = [mag, rcmol, mhi, ra_image_arcsec_rel_to_defmap, dec_image_arcsec_rel_to_defmap,
                                      theta_20, theta_10, hidisk.rdisk_arcsec, z]
        np.save('hidisk_twodisk_%s_%s' % (src_ind, sample), hidisk.twod_disk)
    np.save('%s_parmtrack' % src_ind, parameter_tracking)

print 'time taken', time.time()-start
log.close()


