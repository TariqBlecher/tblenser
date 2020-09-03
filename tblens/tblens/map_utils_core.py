import numpy as np
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline
from astropy.cosmology import Planck15 as cosmo
from tblens.grid_creators import setup_coordinate_grid, PositionGrid
from astropy.coordinates import SkyCoord
import astropy.units as u


class DeflectionMap(PositionGrid):
    """
    MasterClass for deflections
    TAKES IN NORMALISED DEFLECTION MAPS
    """
    def __init__(self, xdeflect_fits, ydeflect_fits, z_lens=0.1, interpolate_position=False):
        PositionGrid.__init__(self, xdeflect_fits)
        self.xdeflect = fits.getdata(xdeflect_fits) / 3600
        self.ydeflect = fits.getdata(ydeflect_fits) / 3600
        self.z_lens = z_lens
        x = np.arange(self.x_deg.shape[0])
        y = np.arange(self.y_deg.shape[0])
        if interpolate_position:
            self.interpolatedxdeflect = RectBivariateSpline(x, y, self.xdeflect)
            self.interpolatedydeflect = RectBivariateSpline(x, y, self.ydeflect)

    def get_deflection_at_image_position(self, coordinate):
        """
        :return: value of deflection map at position in image plane
        """
        coord = SkyCoord(ra=coordinate[0]*u.deg, dec=coordinate[1]*u.deg)
        rapix, decpix = coord.to_pixel(self.wcax)
        pos_ind = tuple(np.hstack((decpix, rapix)).astype(int))
        return np.hstack((self.xdeflect[pos_ind], self.ydeflect[pos_ind]))

    def get_deflection_at_image_position_interpolation(self, coordinate):
        """
        :return: value of deflection map at position in image plane
        """
        coord = SkyCoord(ra=coordinate[0]*u.deg, dec=coordinate[1]*u.deg)
        rapix, decpix = coord.to_pixel(self.wcax)
        return np.hstack((self.interpolatedxdeflect.ev(decpix, rapix), self.interpolatedydeflect.ev(decpix, rapix)))

    def calc_source_position(self, coordinate, z_src, use_interpolation=False):
        """Lens equation. Calculate source positions for many coordinates. coordinates have to be in array format"""
        if use_interpolation:
            deflection = self.get_deflection_at_image_position_interpolation(coordinate) * lens_efficiency(z_lens=self.z_lens,
                                                                                             z_src=z_src)
        else:
            deflection = self.get_deflection_at_image_position(coordinate) * lens_efficiency(z_lens=self.z_lens,
                                                                                          z_src=z_src)
        source_dec = coordinate[1] - deflection[1]
        source_ra = coordinate[0] + deflection[0] * np.cos(self.center[1] * np.pi/180)
        return np.hstack((source_ra, source_dec))

    def calc_image_pixel_center(self, source_fits, z_src, write_image=True, imagename='image.npy', nulltest=False):
        """
        Lens equation at each point in image map
        """
        if nulltest:
            lens_eff = 0
        else:
            lens_eff = lens_efficiency(z_lens=self.z_lens, z_src=z_src)
        xmap = self.x_deg + self.xdeflect * lens_eff * np.cos(self.center[1] * np.pi/180)
        ymap = self.y_deg - self.ydeflect * lens_eff
        x, y = setup_coordinate_grid(source_fits, ndim=1)
        x = np.flip(x)
        source_map_interp = RectBivariateSpline(y, x, np.flip(fits.getdata(source_fits), 1))
        image = source_map_interp.ev(ymap, xmap)
        if write_image:
            np.save(imagename, image)
        return image

    def calc_magnification(self, sourcefits, z_src, write_image=False, imagename='test', nulltest=False):

        # #Source fits
        srcsum = fits.getdata(sourcefits).sum()
        srchdr = fits.getheader(sourcefits)
        srcpixelscale_deg = srchdr['CDELT2']

        # # Interpolation is necessary to smooth pixelated features
        image = self.calc_image_pixel_center(sourcefits, z_src, write_image=write_image,
                                             imagename=imagename, nulltest=nulltest)

        return image.sum()/srcsum * (self.pix_scale_deg / srcpixelscale_deg) ** 2


def lens_efficiency(z_lens, z_src):
    dist_lens_source = cosmo.angular_diameter_distance_z1z2(z_lens, z_src).value
    dist_observer_source = cosmo.angular_diameter_distance(z_src).value
    return dist_lens_source/dist_observer_source







