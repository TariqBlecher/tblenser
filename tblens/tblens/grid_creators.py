import numpy as np
from astropy.io import fits
from astropy import wcs


def setup_coordinate_grid(fitsfile, ndim=2):
    """
    2D coordinate grid for x(RA) & y(DEC). x(RA) increases to the left.
    :param fitsfile: file from which grid is calculated
    :param ndim: dimension of output coordinates
    :return: coordinate grid
    """
    header = fits.getheader(fitsfile)
    wrad = wcs.WCS(fitsfile)
    x = np.arange(header['NAXIS1'])
    y = np.arange(header['NAXIS2'])
    if ndim == 1:
        ra_deg, dec_deg = wrad.all_pix2world(x, y, 0)
    elif ndim == 2:
        x_array, y_array = np.meshgrid(x, y)
        ra_deg, dec_deg = wrad.all_pix2world(x_array, y_array, 0)
    return ra_deg, dec_deg


class PositionGrid(object):
    """
    Useful class for managing 2D grids
    """
    def __init__(self, fitsfile):
        self.wcax = wcs.WCS(fitsfile)
        self.original_fits = fitsfile
        self.x_deg, self.y_deg = setup_coordinate_grid(fitsfile)
        self.npix = self.x_deg.shape[0]
        self.header = fits.getheader(fitsfile)
        self.pix_scale_deg = self.header['CDELT2']
        self.extent = self.npix * self.pix_scale_deg
        self.center = np.zeros(2)
        self.center[0] = self.header['CRVAL1']
        self.center[1] = self.header['CRVAL2']

