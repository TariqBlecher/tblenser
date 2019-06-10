import numpy as np
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline
from astropy.cosmology import Planck15 as cosmo
from scipy.ndimage import zoom
from grid_creators import oned_coordinate_grid, PositionGrid


class DeflectionMap(PositionGrid):
    """
    MasterClass for deflections
    TAKES IN NORMALISED DEFLECTION MAPS
    """
    def __init__(self, xdeflect_fits, ydeflect_fits, center=True, z_lens=0.1):
        PositionGrid.__init__(self, xdeflect_fits, center=center)
        self.xdeflect = fits.getdata(xdeflect_fits)
        self.ydeflect = fits.getdata(ydeflect_fits)
        self.z_lens = z_lens
        print 'Deflection map init'

    def recenter_im_coord(self, coord_deg, coord_err_deg=None, sampling='mean'):
        if sampling == 'normal':
            coord_deg = np.random.normal(loc=coord_deg, scale=coord_err_deg)\

        coord_image_arcsec = coord_deg * 3600. - self.center
        test_in_map = (np.abs(coord_image_arcsec) > self.extent).sum()
        if test_in_map:
            print 'IMAGE OUTSIDE OF MAP'
        return coord_image_arcsec.reshape((2, 1))

    def get_deflection_at_image_position(self, coordinate):
        """
        :param coordinate: Image coordinate, depending on whether the map was centered at initialisation
        will determine if coordinate needs to be centered. Units are in arcsec
        :return: value of deflection map at position in image plane
        """
        pos_ind = self.position_index(coordinate)
        return np.hstack((self.xdeflect[pos_ind], self.ydeflect[pos_ind]))

    def get_deflection_at_image_positions(self, coordinates):
        """
        :param coordinates: array of coordinates in arcsec
        :return: deflections in arcsec at coordinate positions
        """
        deflections = np.zeros(coordinates.shape)

        for ind in range(coordinates.shape[1]):
            deflections[:, ind] = self.get_deflection_at_image_position(coordinates[:, ind])
        return deflections

    def calc_source_positions(self, coordinates, z_src, ra_convention=-1, dec_convention=1):
        """Lens equation. Calculate source positions for many coordinates. coordinates have to be in array format"""
        source_coordinates = np.zeros(coordinates.shape)
        deflections = self.get_deflection_at_image_positions(coordinates) * lens_efficiency(z_lens=self.z_lens,
                                                                                            z_src=z_src)
        source_coordinates[0, :] = coordinates[0, :] - ra_convention * deflections[0, :]
        source_coordinates[1, :] = coordinates[1, :] - dec_convention * deflections[1, :]
        return source_coordinates

    def calc_image_pixel_center(self, source_fits, z_src, write_image=True, imagename='image.npy', nulltest=False):
        """
        Lens equation at each point in image map
        """
        if nulltest:
            lens_eff = 0
        else:
            lens_eff = lens_efficiency(z_lens=self.z_lens, z_src=z_src)

        xmap = self.x_arcsec + self.xdeflect*lens_eff
        ymap = self.y_arcsec - self.ydeflect*lens_eff
        x, y = oned_coordinate_grid(source_fits, translate=True)
        source_map_interp = RectBivariateSpline(y, x, np.flip(fits.getdata(source_fits), 1))
        image = source_map_interp.ev(ymap, xmap)
        if write_image:
            np.save(imagename, image)
        return image

    def calc_magnification(self, sourcefits, z_src, write_image=False, imagename='test',
                           zoom_factor=1, cutout_margin_factor=2, nulltest=False, cutout=False):

        # #Source fits
        srcsum = fits.getdata(sourcefits).sum()
        srchdr = fits.getheader(sourcefits)
        srcpixelscale_arcsec = srchdr['CDELT2'] * 3600
        srcextent_arcsec = srcpixelscale_arcsec * srchdr['NAXIS1']
        npix_src = srcextent_arcsec/self.pix_scale_arcsec

        # # Interpolation is necessary to smooth pixelated features
        image = self.calc_image_pixel_center(sourcefits, z_src, write_image=write_image,
                                             imagename=imagename, nulltest=nulltest)

        if cutout:
            cutout_margin = np.int(npix_src * cutout_margin_factor)

            ind1, ind2 = np.concatenate(np.where(image == image.max()))
            if (ind1-cutout_margin) <= 0 or (ind2-cutout_margin) <= 0 or \
               (ind1+cutout_margin) >= image.shape[0] or (ind2+cutout_margin) >= image.shape[0]:
                cutout_margin_factor = 1
                cutout_margin = np.int(npix_src * cutout_margin_factor)
            imzoomed = zoom(image[ind1-cutout_margin:ind1+cutout_margin,
                            ind2-cutout_margin:ind2+cutout_margin], zoom_factor)/zoom_factor**2
            image = imzoomed
        return image.sum()/srcsum * (self.pix_scale_arcsec / srcpixelscale_arcsec)**2


def lens_efficiency(z_lens, z_src):
    dist_lens_source = cosmo.angular_diameter_distance_z1z2(z_lens, z_src).value
    dist_observer_source = cosmo.angular_diameter_distance(z_src).value
    return dist_lens_source/dist_observer_source







