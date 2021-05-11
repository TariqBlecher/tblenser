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
n_samples_per_src = 100
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
# rcmol_log_mean_sig = [-0.1, 0.3]
mass_sampling_pdf = 'mean'
nulltest = False
zsampling = 'mean'
writeimages = True
defmap = DeflectionMap(xdeflect_fits=xdeflectfits, ydeflect_fits=ydeflectfits, z_lens=zcluster)
mhisamples = 10
inclination_samples = 6
pos_angle_samples = 8
mhi_array = np.linspace(8, 10, mhisamples)
theta_10_array = np.linspace(0, 180, pos_angle_samples)
theta_20_array = np.linspace(20, 85, inclination_samples)

sample = 0
for src_ind in [227]:
    parameter_tracking = Table(names=('mag', 'rcmol', 'mhi', 'relative_ra', 'relative_dec', 'inclination',
                                      'pos_angle', 'rdisk', 'z', 'nz'))
    for mhi_sample in range(mhisamples):
        for inclination_sample in range(inclination_samples):
            for pos_angle_sample in range(pos_angle_samples):
                # # Sampling
                mhi = mhi_array[mhi_sample]
                rcmol = 1.
                theta_10 = theta_10_array[pos_angle_sample]
                theta_20 = theta_20_array[inclination_sample]
                z, nz = sample_z(zspec=z_spec[src_ind], pz=pzs[src_ind], zgrid=zgrid, sampling=zsampling, zmean=z_mean[src_ind],
                                 zcluster=zcluster)
                source_coord_deg = defmap.calc_source_position(coords_deg[:, src_ind], z)
                # # HI Disc
                hidisk = HiDisk(rcmol=rcmol, smoothing_height_pix=False, inclination_degrees=theta_20, position_angle_degrees=theta_10,
                                log10_mhi=mhi, z_src=z, grid_scaling_factor=12, grid_size_min_arcsec=3)

                hidisk.writeto_fits('hidisk_twodisk_%04d_%04d.fits' % (src_ind, sample), defmap.header, source_coord_deg,
                                    flux_norm=True)
                mag = defmap.calc_magnification(hidisk.fitsname, write_image=writeimages, z_src=z,
                                                imagename='image_%04d_%04d' % (src_ind, sample), nulltest=nulltest)
                parameter_tracking.add_row((mag, rcmol, mhi, source_coord_deg[0], source_coord_deg[1],
                                            theta_20, theta_10, hidisk.rdisk_arcsec, z, nz))
                sample += 1
    parameter_tracking.write('%04d_parmtrack' % src_ind, format='ascii')


print('time taken', time.time()-start)


