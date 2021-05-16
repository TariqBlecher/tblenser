from astropy.io import fits
from astropy.constants.astropyconst20 import c, G
from astropy import units
from astropy.cosmology import Planck15 as cosmo
import numpy as np


def write_defmap(mass, pixscale, z_lens, sample_fits, ra_off=0, dec_off=0):
    """
    Calculate and write deflection maps given a mass sheet. Note that the lensing efficiency is built into map_utils and so is not implemented here.

    Parameters
    ----------
    mass : ndarray
        Two dimensional array representing mass sheet
    pixscale : float
        Pixel size in arcsec
    z_lens : float
        redshift of lens
    sample_fits : str
        path to template fits file for header generation
    ra_off : float
        ra offset in arcsec from template fits file ra reference
    dec_off : float
        dec offset in arcsec from template fits file dec reference

    Returns -> None 

    Writes 'deflectionx.fits' and 'deflectiony.fits' 
    """
    D_d = cosmo.angular_diameter_distance(z_lens)
    npix = mass.shape[0]
    pix_area = pixscale**2
    center = int(npix / 2)
    image_radius = center * pixscale
    y, x = np.mgrid[-1 * image_radius:image_radius:1j * npix, -1 * image_radius:image_radius:1j * npix]
    # # 4G/c^2 is the usual prefactor. pix_area is to get mass density to mass. "/kpc is to get 1/r to kpc
    deflection_prefactor = units.rad.to('arcsec')**2 * 4 * G * pix_area * units.M_sun.to('kg') / (c**2 * D_d * units.Mpc.to('m'))

    defy, defx = np.zeros((2, npix, npix))
    for indx in range(npix):
        for indy in range(npix):
            r_2 = (y[indy, indx] - y)**2 + (x[indy, indx] - x)**2
            r_2[r_2 == 0.0] = np.nan
            defy[indy, indx] = np.nansum((y[indy, indx] - y) * mass / r_2)
            defx[indy, indx] = np.nansum((x[indy, indx] - x) * mass / r_2)

    defy *= deflection_prefactor.value
    defx *= deflection_prefactor.value

    header = fits.getheader(sample_fits)
    header['CDELT2'] = pixscale / 3600.
    header['CDELT1'] = -1 * pixscale / 3600.
    header['NAXIS1'] = npix
    header['NAXIS2'] = npix
    header['CRVAL1'] = ra_off
    header['CRVAL2'] = dec_off
    header['CRPIX1'] = int(center)
    header['CRPIX2'] = int(center)
    fits.writeto('deflectionx.fits', defx, header=header, overwrite=True)
    fits.writeto('deflectiony.fits', defy, header=header, overwrite=True)
