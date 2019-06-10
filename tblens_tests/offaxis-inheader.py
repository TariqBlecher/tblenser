from astropy.io import fits
from astropy.constants.astropyconst20 import c, G
from astropy import units
from tblens.lens_creator import *
from tblens.map_utils_core import *
from tblens.grid_creators import *
from astropy.cosmology import Planck15 as cosmo
import numpy as np
# # Check that off axis header works properly

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

srcheader = fits.getheader('deflectionx.fits')
yoff = -1.
xoff = -2.
srcheader['CRVAL1'] = xoff/3600.
srcheader['CRVAL2'] = yoff/3600.

fits.writeto('src.fits', src, header=srcheader, overwrite=True)
lim = defmap.calc_image_pixel_center('src.fits', z_src)
R_einstein = np.sqrt(mass_density.sum()*pix_area/10**11.09) * (D_d * D_s * 1e-3/D_ds)**(-0.5)
R_einstein = R_einstein.value

y, x = setup_coordinate_grid('deflectionx.fits')

max_pos = np.where(lim == lim.max())
xsign = np.sign(xoff)
xoff = np.abs(xoff)
ysign = np.sign(yoff)
yoff = np.abs(yoff)
expected_x = xsign*0.5*(xoff+np.sqrt(xoff**2+4*R_einstein**2))  # # Equation 24 narayan lecture notes
expected_y = ysign*0.5*(yoff+np.sqrt(yoff**2+4*R_einstein**2))

print 'y error is %.1f in pixels' % ((y[max_pos]-expected_y)/pixscale)
print 'x error is %.1f in pixels' % ((x[max_pos]-expected_x)/pixscale)

assert np.abs(y[max_pos]-expected_y) < 5*pixscale, 'y off'
assert np.abs(x[max_pos]-expected_x) < 5*pixscale, 'x off'

# # This test is untested