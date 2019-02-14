from map_utils import *
from HiDisk import *
from utils import sample_inclination_deg, sample_check_z, mass_sampling
import time

# log = file('log.txt', 'w'); sys.stdout = log
start = time.time()

# Parameter setup
parm_file = 'sim_catalog_29.npy'
xdeflectfits = 'hlsp_frontier_model_abell2744_glafic_v4_x-arcsec-deflect-recentered.fits'#'hlsp_frontier_model_abell2744_cats_v4.1_x-arcsec-deflect-zoom.fits'
ydeflectfits = 'hlsp_frontier_model_abell2744_glafic_v4_y-arcsec-deflect-recentered.fits'#'hlsp_frontier_model_abell2744_cats_v4.1_y-arcsec-deflect-zoom.fits'
ra_image_mean, dec_image_mean, z_src, u_zsrc, mhi_log, u_mhi_log, ra_err_deg, dec_err_deg = np.load(parm_file)
src_indices = np.arange(mhi_log.shape[0])
coords_deg = np.vstack((ra_image_mean, dec_image_mean))
coords_err_deg = np.vstack((ra_err_deg, dec_err_deg))
n_samples_per_src = 1 # 100
rcmol_log_mean_sig = [-0.1, 0.3]
mass_sampling_pdf = 'uniform' #'normal'
zcluster = 0.308
srctype = 'gauss' ##gauss for gauss testing

defmap = DeflectionMap(xdeflect_fits=xdeflectfits, ydeflect_fits=ydeflectfits, center=False, z_lens=zcluster)

for src_ind in src_indices:
    parameter_tracking = np.zeros((n_samples_per_src, 10))

    for sample in range(n_samples_per_src):
        # # Sampling
        mhi = mass_sampling(pdf=mass_sampling_pdf, uniform_lower_bound=8., uniform_width=2.5,
                            mass_mean_log10=mhi_log[src_ind], mass_sig_log10=u_mhi_log[src_ind])

        rcmol = np.random.lognormal(rcmol_log_mean_sig[0], rcmol_log_mean_sig[1])
        theta_10 = np.random.rand()*180.
        theta_20 = sample_inclination_deg()
        z, nz = sample_check_z(z=z_src[src_ind], u_z=u_zsrc[src_ind])
        recentered_image_coordinate = defmap.recenter_im_coord(coords_deg[:, src_ind], coords_err_deg[:, src_ind])
        source_coord_arcsec_rel = defmap.calc_source_positions(recentered_image_coordinate, z)[:, 0]
        print recentered_image_coordinate, source_coord_arcsec_rel

        # # HI Disc
        hidisk = HiDisk(rcmol=rcmol, smoothing_height_pix=False, theta_2_0=theta_20, theta_1_0=theta_10,
                        log10_mhi=mhi, z_src=z, scale_by_rdisk=12, grid_size_min_arcsec=3, minpixelsizecheck=False)

        hidisk.writeto_fits('hidisk_twodisk_%s_%s.fits' % (src_ind, sample), defmap.header,
                            source_coord_arcsec_rel, src=srctype)

        mag = defmap.calc_magnification(hidisk.fitsname, write_image=True, z_src=z,
                                        imagename='image_%s_%s' % (src_ind, sample), zoom_factor=2,
                                        cutout_margin_factor=2)

        parameter_tracking[sample] = [mag, rcmol, mhi, source_coord_arcsec_rel[0], source_coord_arcsec_rel[1],
                                      theta_20, theta_10, hidisk.rdisk_arcsec, z, nz]

    # # Save parameter tracker
    np.save('%02d_parmtrack' % src_ind, parameter_tracking)

# log.close()
print 'time taken', time.time()-start


