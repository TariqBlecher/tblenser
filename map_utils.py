import numpy as np
import pyfits as pf
from scipy.interpolate import RectBivariateSpline
from astropy.cosmology import Planck15 as cosmo

def setup_coordinate_grid(fits, inverse_ra_convention=False):
    header = pf.getheader(fits)
    pix_scale_arcsec = header['CDELT2']*3600.
    npix = np.array([header['NAXIS1'], header['NAXIS2']])
    image_radius = pix_scale_arcsec * npix / 2.
    if inverse_ra_convention:
        y_arcsec, x_arcsec = np.mgrid[-1 * image_radius[0]:image_radius[0]:1j * npix[0],
                                      -1 * image_radius[1]:image_radius[1]:1j * npix[1]]
    else:
        y_arcsec, x_arcsec = np.mgrid[-1 * image_radius[0]:image_radius[0]:1j * npix[0],
                                      image_radius[1]:-1*image_radius[1]:1j * npix[1]]

    return y_arcsec, x_arcsec


def oned_coordinate_grid(fits):
        fits_hdr = pf.getheader(fits)
        npix = fits_hdr['NAXIS1']
        dx = fits_hdr['CDELT2'] * 3600.
        radius = npix * dx / 2.
        x = np.linspace(-1 * radius, radius, npix)
        y = np.linspace(-1 * radius, radius, npix)
        return x, y


def lens_efficiency(z, z_cluster=0.308):
    dist_lens_source = cosmo.angular_diameter_distance_z1z2(z_cluster, z).value
    dist_observer_source = cosmo.angular_diameter_distance(z).value
    return dist_lens_source/dist_observer_source


class PositionGrid(object):
    def __init__(self, fits, center=True):
        self.original_fits = fits
        self.y_arcsec, self.x_arcsec = setup_coordinate_grid(fits)
        self.npix = self.x_arcsec.shape[0]
        self.header = pf.getheader(fits)
        self.pix_scale_arcsec = self.header['CDELT2']*3600.
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


class DeflectionMap(PositionGrid):
    def __init__(self, xdeflect_fits, ydeflect_fits, center=True):
        PositionGrid.__init__(self, xdeflect_fits, center=center)
        self.xdeflect = pf.getdata(xdeflect_fits)
        self.ydeflect = pf.getdata(ydeflect_fits)
        print 'Deflection map init'

    def deflect_at_position(self, coordinate):
        """
        :param coordinate: Image coordinate, depending on whether the map was centered at initialisation
        will determine if coordinate needs to be centered. Units are in arcsec
        :return: value of deflection map at position in image plane
        """
        pos_ind = self.position_index(coordinate)
        return np.hstack((self.xdeflect[pos_ind], self.ydeflect[pos_ind]))

    def deflect_at_positions(self, coordinates):
        """
        :param coordinates: array of coordinates in arcsec
        :return: deflections in arcsec at coordinate positions
        """
        deflections = np.zeros(coordinates.shape)

        for ind in range(coordinates.shape[1]):
            deflections[:, ind] = self.deflect_at_position(coordinates[:, ind])
        return deflections

    def calc_source_positions(self, coordinates, z, ra_convention=-1, dec_convention=1):
        """Calculate source positions for many coordinates. coordinates have to be in array format"""
        source_coordinates = np.zeros(coordinates.shape)
        deflections = self.deflect_at_positions(coordinates) * lens_efficiency(z)
        source_coordinates[0, :] = coordinates[0, :] - ra_convention * deflections[0, :]
        source_coordinates[1, :] = coordinates[1, :] - dec_convention * deflections[1, :]
        return source_coordinates

    def regrid(self, nfits):
        """
        Useful function to compare maps on different grids
        Have to change the RA convention to RA=-RA for this because RectBivariatSpline needs strictly ascending arrays
        But this doesn't affect anything else in the class"""
        x, y = oned_coordinate_grid(self.original_fits)
        interpolatefunc_xdeflect = RectBivariateSpline(x, y, self.xdeflect)
        interpolatefunc_ydeflect = RectBivariateSpline(x, y, self.ydeflect)

        ny, nx = setup_coordinate_grid(nfits, inverse_ra_convention=True)
        interpxdeflect = interpolatefunc_xdeflect.ev(nx.flatten(), ny.flatten()).reshape(nx.shape, order='F')
        interpydeflect = interpolatefunc_ydeflect.ev(nx.flatten(), ny.flatten()).reshape(nx.shape, order='F')
        return interpxdeflect, interpydeflect

    def calc_image(self, source_fits, z):
        lens_eff = lens_efficiency(z)
        xmap = self.x_arcsec + self.xdeflect*lens_eff
        ymap = self.y_arcsec - self.ydeflect*lens_eff
        x, y = oned_coordinate_grid(source_fits)
        source_map_interp = RectBivariateSpline(x, y, pf.getdata(source_fits))
        image = source_map_interp.ev(xmap.flatten()[::-1], ymap.flatten()).reshape(self.x_arcsec.shape, order='F')
        return image

    def calc_image_same_grid(self, sourcedata, z, write_image=True, imagename='image.npy'):
        lens_eff = lens_efficiency(z)
        xmap = self.x_arcsec + self.xdeflect*lens_eff
        ymap = self.y_arcsec - self.ydeflect*lens_eff
        x, y = oned_coordinate_grid(self.original_fits)
        source_map_interp = RectBivariateSpline(x, y, sourcedata)
        image = source_map_interp.ev(xmap.flatten()[::-1], ymap.flatten()).reshape(self.x_arcsec.shape, order='F')
        if write_image:
            np.save(imagename, image)
        return image

    def calc_magnification(self, sourcedata, z, write_image=False, imagename='test'):
        image = self.calc_image_same_grid(sourcedata, z, write_image=write_image, imagename=imagename)
        return image.sum()/sourcedata.sum()








