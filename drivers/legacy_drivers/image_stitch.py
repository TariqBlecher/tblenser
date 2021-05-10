from tblens.map_utils_core import *
from tblens.HiDisk import *
from tblens.utils import sample_inclination_deg, sample_z, mass_sampling
import time
import os
import sys
from astropy.table import Table

start = time.time()
# Parameter setup
field_prefix = 'abell370'
n_samples_per_src = 200
print(field_prefix, 'nsamples=', n_samples_per_src)
field_info = Table.read('hfftab', format='ascii')
field_ind = np.where(field_info['field'] == field_prefix)[0]
parm_file = 'table_catalog_deepspace%s' % field_prefix
xdeflectfits = field_info[field_ind]['xdeflectfits'][0]
ydeflectfits = field_info[field_ind]['ydeflectfits'][0]
zcluster = field_info[field_ind]['zcluster'][0]
pzs = np.load('eazy_pz'+field_prefix+'.npy')
zgrid = np.load('eazy_zgrid'+field_prefix+'.npy')

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
mass_sampling_pdf = 'uniform'
nulltest = False
zsampling = 'mean'
writeimages = True
defmap = DeflectionMap(xdeflect_fits=xdeflectfits, ydeflect_fits=ydeflectfits, z_lens=zcluster)
masksrc = np.load('GreatArcMask.npy')
parameter_tracking = Table(names=('rcmol', 'mhi', 'inclination',
                                  'pos_angle', 'rdisk', 'z', 'nz'))
z=0.725
for sample in range(n_samples_per_src):
    # # Sampling
    mhi = mass_sampling(pdf=mass_sampling_pdf, uniform_lower_bound=8., uniform_width=2.5,
                        mass_mean_log10=mhi_log[56], mass_sig_log10=u_mhi_log[56])

    rcmol = np.random.lognormal(rcmol_log_mean_sig[0], rcmol_log_mean_sig[1])
    theta_10 = 135.
    theta_20 = 75.
    z, nz = [0.725,1]
    hidisk = HiDisk(rcmol=rcmol, smoothing_height_pix=False, theta_2_0=theta_20, theta_1_0=theta_10,
                    log10_mhi=mhi, z_src=z, scale_by_rdisk=12, grid_size_min_arcsec=3)
    for src_ind in [49, 56]:

        source_coord_deg = defmap.calc_source_position(coords_deg[:, src_ind], z)
        hidisk.writeto_fits('hidisk_twodisk_%04d_%04d.fits' % (src_ind, sample), defmap.header, source_coord_deg,
                            flux_norm=True)
        defmap.calc_image_pixel_center(hidisk.fitsname, z, write_image=writeimages, imagename='image_%04d_%04d' % (src_ind, sample))
    parameter_tracking.add_row((rcmol, mhi, theta_20, theta_10, hidisk.rdisk_arcsec, z, nz))
parameter_tracking.write('parmtrack', format='ascii')

print('time taken', time.time()-start)


