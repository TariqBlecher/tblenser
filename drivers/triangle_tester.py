from tblens.map_utils_ext_tri import *
from tblens.HiDisk import *
import numpy as np
import time
import sys
from tblens.utils import get_parmtracks

log = file('log.txt', 'w'); sys.stdout = log
start = time.time()

# Parameter setup
xdeflectfits = 'hlsp_frontier_model_abell2744_cats_v4.1_x-arcsec-deflect-cut.fits'
ydeflectfits = 'hlsp_frontier_model_abell2744_cats_v4.1_y-arcsec-deflect-cut.fits'
folder_to_test = '../fullimage9/'
parmtracks = get_parmtracks(folder_to_test)
mag = parmtracks[:, :, 0]
rcmol_parm = parmtracks[:, :, 1]
mhi_parm = parmtracks[:, :, 2]
ra = parmtracks[:, :, 3]
dec = parmtracks[:, :, 4]
inclination = parmtracks[:, :, 5]
posangle = parmtracks[:, :, 6]
z_parm = parmtracks[:, :, 8]

src_indices = np.arange(parmtracks.shape[0])
n_samples_per_src = parmtracks.shape[1]
zcluster = 0.308
nulltest = False
writeimages = True
defmapext = DeflectionMapExt(xdeflect_fits=xdeflectfits, ydeflect_fits=ydeflectfits, center=False,
                             z_lens=zcluster)
for src_ind in src_indices:
    parameter_tracking = np.zeros((n_samples_per_src, 2))
    for sample in range(n_samples_per_src):
        # # Sampling
        mhi = mhi_parm[src_ind, sample]
        rcmol = rcmol_parm[src_ind, sample]
        theta_10 = posangle[src_ind, sample]
        theta_20 = inclination[src_ind, sample]
        z = z_parm[src_ind, sample]
        source_coord_arcsec_rel = np.array([ra[src_ind, sample], dec[src_ind, sample]])
        # # HI Disc
        hidisk = HiDisk(rcmol=rcmol, smoothing_height_pix=False, theta_2_0=theta_20, theta_1_0=theta_10,
                        log10_mhi=mhi, z_src=z, scale_by_rdisk=12, grid_size_min_arcsec=3, minpixelsizecheck=False)

        hidisk.writeto_fits('hidisk_twodisk_%04d_%04d.fits' % (src_ind, sample), defmapext.header, source_coord_arcsec_rel,
                            flux_norm=True)
        if src_ind == 0 and sample == 0:
            defmapext.weight_table(source_fits='hidisk_twodisk_%04d_%04d.fits' % (src_ind, sample), z_src=z,
                                   suffix='')
        mag = defmapext.calc_mag_triangle(source_fits='hidisk_twodisk_%04d_%04d.fits' % (src_ind, sample), z_src=z,
                                          image_suffix='%04d_%04d' % (src_ind, sample), write_image=writeimages)

        parameter_tracking[sample] = [mag, hidisk.rdisk_arcsec]

    np.save('%04d_parmtrack' % src_ind, parameter_tracking)

print 'time taken', time.time()-start
log.close()


