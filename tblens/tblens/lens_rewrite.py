from astropy.io import fits
from astropy.constants.astropyconst20 import c, G
from astropy import units
from astropy.cosmology import Planck15 as cosmo
import numpy as np


def rewrite_defmap(mass, z_lens, sample_fits,xname='xdeflect.fits',yname='ydeflect.fits'):
    """
    Write down the deflection maps given a mass sheet
    """
    header = fits.getheader(sample_fits)
    pixscale = header['CDELT2'] * 3600
    D_d = cosmo.angular_diameter_distance(z_lens)
    npix = header['NAXIS1']
    pix_area = pixscale**2
    center = int(npix / 2)
    image_radius = center * pixscale
    y, x = np.mgrid[-1 * image_radius:image_radius:1j * npix, -1 * image_radius:image_radius:1j * npix]
    deflection_prefactor = units.rad.to('arcsec')**2* 4*G*pix_area * units.M_sun.to('kg') /(c**2 * D_d*units.Mpc.to('m'))

    ##LENSING EFFICIENCY IS BUILT INTO MAP_UTILS
    #4G/c^2 is the usual prefactor. pix_area is to get mass density to mass. "/kpc is to get 1/r to kpc

    defy, defx = np.zeros((2, npix, npix))
    for indx in range(npix):
        for indy in range(npix):
            r_2 = (y[indy, indx]-y)**2 + (x[indy, indx]-x)**2
            r_2[r_2 == 0.0] = np.nan
            defy[indy, indx] = np.nansum((y[indy, indx]-y) * mass/r_2)
            defx[indy, indx] = np.nansum((x[indy, indx]-x) * mass/r_2)

    defy *= deflection_prefactor.value
    defx *= deflection_prefactor.value

    fits.writeto(xname, defx, header=header, overwrite=True)
    fits.writeto(yname, defy, header=header, overwrite=True)
