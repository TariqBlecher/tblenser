from map_utils import *
from HiDisk import *
from utils import sample_inclination_deg, sample_check_z, mass_sampling
import time

# log = file('log.txt', 'w'); sys.stdout = log
start = time.time()

parm_file = 'sim_catalog.npy'
xdeflectfits = 'hlsp_frontier_model_abell2744_cats_v4.1_x-arcsec-deflect-zoom.fits'
ydeflectfits = 'hlsp_frontier_model_abell2744_cats_v4.1_y-arcsec-deflect-zoom.fits'
ra_image_mean, dec_image_mean, z_src, u_zsrc, mhi_log, u_mhi, ra_err_deg, dec_err_deg = np.load(parm_file)
src_indices = np.arange(mhi_log.shape[0])
coords_deg = np.vstack((ra_image_mean, dec_image_mean))
coords_err_deg = np.vstack((ra_err_deg, dec_err_deg))
n_samples_per_src = 2
rcmol_log_mean_sig = [-0.1, 0.3]
mass_sampling_pdf = 'uniform'
zcluster = 0.308

defmap = DeflectionMap(xdeflect_fits=xdeflectfits, ydeflect_fits=ydeflectfits, center=False)

for src_ind in src_indices:
    parameter_tracking = np.zeros((n_samples_per_src, 10))

    for sample in range(n_samples_per_src):
        # # Sampling
        mhi = 8.5
        rcmol = 1
        theta_10 = 45.
        theta_20 = 45.
        z, nz = 0.45, 1
        recentered_image_coordinate = defmap.recenter_im_coord(coords_deg[:, src_ind], coords_err_deg[:, src_ind])
        print recentered_image_coordinate
        source_coord_arcsec_rel = defmap.calc_source_positions(recentered_image_coordinate, z)[:, 0]
        print source_coord_arcsec_rel
        # # HI Disc
        hidisk = HiDisk(rcmol=rcmol, smoothing_height_pix=False, theta_2_0=theta_20, theta_1_0=theta_10,
                        log10_mhi=mhi, z_src=z, scale_by_rdisk=12, grid_size_min_arcsec=6, minpixelsizecheck=False)

        hidisk.writeto_fits('hidisk_twodisk_%s_%s.fits' % (src_ind, sample), defmap.header, source_coord_arcsec_rel)

        mag = defmap.calc_magnification(hidisk.fitsname, z, write_image=True,
                                        imagename='image_%s_%s' % (src_ind, sample))

        parameter_tracking[sample] = [mag, rcmol, mhi, source_coord_arcsec_rel[0], source_coord_arcsec_rel[1],
                                      theta_20, theta_10, hidisk.rdisk_arcsec, z, nz]

    # # Save parameter tracker
    np.save('%s_parmtrack' % src_ind, parameter_tracking)

# log.close()
print 'time taken', time.time()-start


