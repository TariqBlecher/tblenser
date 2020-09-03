from tblens.map_utils_core import *
from tblens.HiDisk import *
from tblens.utils import sample_inclination_deg, sample_z, mass_sampling
import time
import os
import sys
from astropy.table import Table

start = time.time()
field_prefix = 'abell370'
field_info = Table.read('hfftab', format='ascii')
field_ind = np.where(field_info['field'] == field_prefix)[0]
parm_file = 'table_catalog_deepspace%s' % field_prefix

lens_models_table = Table.read('lens_model_table_a370', format='ascii')
xdeflectfits_list = lens_models_table['xdeflectfits']
ydeflectfits_list = lens_models_table['ydeflectfits']
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
mass_sampling_pdf = 'mean'
nulltest = False
zsampling = 'mean'
writeimages = True
mhisamples = 8
inclination_samples = 4
pos_angle_samples = 4
mhi_array = np.linspace(8.5, 10, mhisamples)
pos_angle_array = np.linspace(10, 80, pos_angle_samples)
incl_angle_array = np.linspace(70, 80, inclination_samples)

for defmap_ind in range(len(xdeflectfits_list)):
    defmap = DeflectionMap(xdeflect_fits=xdeflectfits_list[defmap_ind],
                           ydeflect_fits=ydeflectfits_list[defmap_ind], z_lens=zcluster)

    sample = 0
    for src_ind in [79]:
        parameter_tracking = Table(names=('mag', 'rcmol', 'mhi', 'relative_ra', 'relative_dec', 'inclination',
                                          'pos_angle', 'rdisk', 'z', 'nz'))
        for mhi_sample in range(mhisamples):
            for inclination_sample in range(inclination_samples):
                for pos_angle_sample in range(pos_angle_samples):
                    # # Sampling
                    mhi = mhi_array[mhi_sample]
                    rcmol = 1.#np.random.lognormal(rcmol_log_mean_sig[0], rcmol_log_mean_sig[1])

                    theta_10 = pos_angle_array[pos_angle_sample]
                    theta_20 = incl_angle_array[inclination_sample]
                    z, nz = sample_z(zspec=z_spec[src_ind], pz=pzs[src_ind], zgrid=zgrid, sampling=zsampling, zmean=z_mean[src_ind],
                                     zcluster=zcluster)
                    source_coord_deg = defmap.calc_source_position(coords_deg[:, src_ind], z)
                    # # HI Disc
                    hidisk = HiDisk(rcmol=rcmol, smoothing_height_pix=False, theta_2_0=theta_20, theta_1_0=theta_10,
                                    log10_mhi=mhi, z_src=z, scale_by_rdisk=12, grid_size_min_arcsec=3)

                    hidisk.writeto_fits('hidisk_twodisk_%04d_%04d_%04d.fits' % (defmap_ind, src_ind, sample), defmap.header, source_coord_deg,
                                        flux_norm=True)
                    mag = defmap.calc_magnification(hidisk.fitsname, write_image=writeimages, z_src=z,
                                                    imagename='image_%04d_%04d_%04d' % (defmap_ind, src_ind, sample), nulltest=nulltest)
                    parameter_tracking.add_row((mag, rcmol, mhi, source_coord_deg[0], source_coord_deg[1],
                                                theta_20, theta_10, hidisk.rdisk_arcsec, z, nz))
                    sample += 1
        parameter_tracking.write('%04d_%04d_parmtrack' % (defmap_ind, src_ind), format='ascii')


print('time taken', time.time()-start)


