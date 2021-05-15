"""
A high level script to simulate lensed HI disks behind the HFF clusters. 

The first part of this script reads in the relevant parameters of the foreground clusters and background sources.

The second part of the script, consisting of the double for loop, handles the actual lensing simulation and magnification calculation.

The outer for-loop runs over all background sources while the inner for-loop runs over the nsamples of each source. Sources are sampled multiple times in order to marginalise over uncertain or nuisance parameters.
"""
import logging
import numpy as np
from tblens.map_utils_core import DeflectionMap
from tblens.HiDisk import HiDisk
from tblens.utils import sample_inclination_deg, sample_z, mass_sampling
import time
import os
from astropy.table import Table

logging.basicConfig(filename='image_full_cluster.log')
start = time.time()

# # Configuration
field_prefix = 'abell2744'
n_samples_per_src = 100
print(field_prefix, 'nsamples=', n_samples_per_src)
writeimages = True

# # Foreground Cluster Inputs
field_info = Table.read('hfftab', format='ascii')
field_ind = np.where(field_info['field'] == field_prefix)[0]
parm_file = 'table_catalog_deepspace%s' % field_prefix
xdeflectfits = field_info[field_ind]['xdeflectfits'][0]
ydeflectfits = field_info[field_ind]['ydeflectfits'][0]
zcluster = field_info[field_ind]['zcluster'][0]

# # Background Galaxy Inputs
data_tab = Table.read(parm_file, format='ascii')
ra_image = data_tab['ra']
dec_image = data_tab['dec']
z_spec = data_tab['z_spec']
mhi_log = data_tab['hi_mass']
u_mhi_log = data_tab['uhi']
z_mean = data_tab['z']
src_indices = np.arange(mhi_log.shape[0])
coords_deg = np.vstack((ra_image, dec_image))
rcmol_log_mean_sig = [-0.1, 0.3]
mass_sampling_pdf = 'normal'

# # Run Simulations

defmap = DeflectionMap(xdeflect_fits=xdeflectfits, ydeflect_fits=ydeflectfits, z_lens=zcluster)

for src_ind in src_indices:
    parameter_tracking = Table(names=('mag', 'rcmol', 'mhi', 'relative_ra', 'relative_dec', 'inclination',
                                      'pos_angle', 'rdisk', 'z',))
    for sample in range(n_samples_per_src):
        # # Parameter sampling
        mhi = mass_sampling(pdf=mass_sampling_pdf, uniform_lower_bound=8., uniform_width=2.5,
                            mass_mean_log10=mhi_log[src_ind], mass_sig_log10=u_mhi_log[src_ind])

        rcmol = np.random.lognormal(rcmol_log_mean_sig[0], rcmol_log_mean_sig[1])
        position_angle_deg = np.random.rand() * 180.
        inclination_angle_deg = sample_inclination_deg()
        z = sample_z(zspec=z_spec[src_ind], zmean=z_mean[src_ind])

        # # HI disk creation
        source_coord_deg = defmap.calc_source_position(coords_deg[:, src_ind], z)

        hidisk = HiDisk(rcmol=rcmol, smoothing_height_pix=False, inclination_angle_degrees=inclination_angle_deg, position_angle_degrees=position_angle_deg, log10_mhi=mhi, z_src=z, grid_scaling_factor=12)

        hidisk.writeto_fits('hidisk_twodisk_%04d_%04d.fits' % (src_ind, sample), defmap.header, source_coord_deg,
                            flux_norm=True)
        # # Lensing
        mag = defmap.calc_magnification(hidisk.fitsname, write_image=writeimages, z_src=z,
                                        imagename='image_%04d_%04d' % (src_ind, sample))

        # # Write outputs
        parameter_tracking.add_row((mag, rcmol, mhi, source_coord_deg[0], source_coord_deg[1],
                                    inclination_angle_deg, position_angle_deg, hidisk.rdisk_arcsec, z))

    parameter_tracking.write('%04d_parmtrack' % src_ind, format='ascii', overwrite=True)

    if not writeimages:
        os.system('rm hidisk_twodisk_%04d*.fits' % src_ind)

logging.warning('time taken = %s', time.time() - start)
