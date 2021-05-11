import numpy as np
from astropy.io import fits
from astropy import wcs


def setup_coordinate_grid(fitsfile, ndim=2):
    """
    Grid creator to be used for PositionGrid and DeflectionMap classes

    :param fitsfile: input file from which coordinate system is based
    :param ndim: dimension of output coordinates
    :return: 2D coordinate grid for x(RA) & y(DEC). x(RA) increases to the left.
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
    A useful class for managing 2D grids

    Attributes
    ----------
    wcax : WCS
        World Coordinate System derived from input fits file
    original_fits : str
        Input fits file
    x_deg : ndarray
        One or two dimensional array describing RA coordinate in units of degrees
    y_deg : ndarray
        One or two dimensional array describing DEC coordinate in units of degrees
    npix : int
        Number of pixels along length of coordinate axes
    header : str
        header of input fits file
    pix_scale_deg : float
        pixel size in degrees
    extent : float
        Length of coordinate grid
    center : ndarray
        Two element array consisting of coordinates of RA/DEC fits header reference values 
    """
    def __init__(self, fitsfile):
        """:param: fitsfile : str : file path to fits file which coordinate system is based on"""
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

