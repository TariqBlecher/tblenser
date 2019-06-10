import numpy as np
from astropy.io import fits


def setup_coordinate_grid(fitsfile):
    """
    2D coordinate grid for x(RA) & y(DEC). x(RA) increases to the left.
    :param fitsfile: file from which grid is calculated
    :return: coordinate grid
    """
    header = fits.getheader(fitsfile)
    pix_scale_arcsec = header['CDELT2']*3600.
    npix = np.array([header['NAXIS1'], header['NAXIS2']])
    image_radius = pix_scale_arcsec * npix / 2.
    dec_arcsec, ra_arcsec = np.mgrid[-1 * image_radius[0]:image_radius[0]:1j * npix[0],
                                     -1 * image_radius[1]:image_radius[1]:1j * npix[1]]
    ra_arcsec = ra_arcsec.flatten()[::-1].reshape(ra_arcsec.shape)

    return dec_arcsec, ra_arcsec


class PositionGrid(object):
    """
    Contains a 2D coordinate grid.
     Given a coordinate, can calculats the index of the grid corresponding to that coordinate
    """
    def __init__(self, fitsfile, center=True):
        self.original_fits = fitsfile
        self.y_arcsec, self.x_arcsec = setup_coordinate_grid(fitsfile)
        self.npix = self.x_arcsec.shape[0]
        self.header = fits.getheader(fitsfile)
        self.pix_scale_arcsec = self.header['CDELT2']*3600.
        self.extent = self.npix * self.pix_scale_arcsec
        self.center = np.zeros(2)
        self.center[0] = self.header['CRVAL1']*3600.
        self.center[1] = self.header['CRVAL2']*3600.
        if center:
            self.recenter_grid(self.center)

        print 'Position Grid init'

    def recenter_grid(self, offset):
        self.y_arcsec += offset[1]
        self.x_arcsec += offset[0]

    def position_index(self, coordinate):
        dist = (self.x_arcsec - coordinate[0])**2. + (self.y_arcsec - coordinate[1])**2.
        pos_ind = np.where(dist == dist.min())
        return pos_ind


def oned_coordinate_grid(fitsfile, translate=False):
    """
    :param fitsfile: file from which grid is calculated
    :param translate: shift grid center
    :return: 1D x,y arrays
    """
    fits_hdr = fits.getheader(fitsfile)
    npix = fits_hdr['NAXIS1']
    dx = fits_hdr['CDELT2'] * 3600.
    radius = npix * dx / 2.
    x = np.linspace(-1 * radius, radius, npix)
    y = np.linspace(-1 * radius, radius, npix)
    if translate:
        x += fits_hdr['CRVAL1'] * 3600.
        y += fits_hdr['CRVAL2'] * 3600,
    return x, y
