from astropy.io import fits
from astropy.constants.astropyconst20 import c, G
from astropy import units
from tblens.lens_creator import *
from tblens.map_utils_core import *
from tblens.grid_creators import *
from astropy.cosmology import Planck15 as cosmo
import numpy as np

# # Check that the flux is contained within Einstein Ring

# # Set up test

z_src = 0.4
z_lens = 0.1
D_s = cosmo.angular_diameter_distance(z_src)
D_ds = cosmo.angular_diameter_distance_z1z2(z_lens, z_src)
D_d = cosmo.angular_diameter_distance(z_lens)

npix = 100
pixscale = 0.1
pix_area = pixscale**2
center = npix / 2

mass_density = np.zeros((npix, npix))
mass_density[center, center] = 1e11/pix_area
write_defmap(mass=mass_density, pixscale=pixscale, z_lens=z_lens, sample_fits='fitsexample.fits')
defmap = DeflectionMap(xdeflect_fits='deflectionx.fits', ydeflect_fits='deflectiony.fits')

src = np.zeros((npix, npix))
src[center, center] = 1
fits.writeto('src.fits', src, header=fits.getheader('deflectionx.fits'), overwrite=True)
lim = defmap.calc_image_pixel_center('src.fits', z_src)
R_einstein = np.sqrt(mass_density.sum()*pix_area/10**11.09) * (D_d * D_s * 1e-3/D_ds)**(-0.5)

# # check all flux lies within a few pixels of Einstein Ring
y, x = setup_coordinate_grid('deflectionx.fits')
r = np.sqrt(y**2 + x**2)
einstein_ring_args = np.abs(r-R_einstein.value) < 3*pixscale
total_sum = np.sum(lim)
sum_einstein_ring = np.sum(lim[einstein_ring_args])
assert total_sum > 1e-1, 'no flux in image'
assert np.abs(sum_einstein_ring-total_sum)/total_sum < 0.05, 'more than 1% of flux outside R Einstein'
