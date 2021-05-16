"""This module contains the class 'DeflectionMap' which simulates gravitational lensing using the lens equation."""

import logging
import numpy as np
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline
from astropy.cosmology import Planck15 as cosmo
from tblens.grid_creators import setup_coordinate_grid, PositionGrid
from astropy.coordinates import SkyCoord
import astropy.units as u


class DeflectionMap(PositionGrid):
    """
    Class for implementing gravitational lensing. Inherits from PositionGrid which sets up the coordinate system.

    Attributes
    ----------
    xdeflect : ndarray
        2D deflection map in RA direction in units of degrees
    ydeflect : ndarray
        2D deflection map in DEC direction in units of degrees
    z_lens : float
        redshift of lens

    Methods
    -------
    calc_source_position(coordinate, z_src)
        Calculates source positions for an image plane coordinate
    get_deflection_at_image_position(coordinate)
        returns deflection angle of an image plane coordinate
    calc_image_pixel_center(source_fits, z_src, write_image=True, imagename='image.npy')
        Computes lensed image using vectorized lens equation
    calc_magnification(sourcefits, z_src, write_image=False, imagename='test')
        Calculates total magnification using calc_image_pixel_center 
    """
    def __init__(self, xdeflect_fits, ydeflect_fits, z_lens=0.1):
        """
        Parameters
        ----------
        xdeflect_fits : str
            input fits file specifying the deflections in RA. The coordinate system is also derived from this file header as it is passed to PositionGrid instance.
        ydeflect_fits : str    
            input fits file specifying the deflections in DEC
        z_lens : float
            redshift of lens
        """
        PositionGrid.__init__(self, xdeflect_fits)
        self.logger = logging.getLogger('tblens.map_utils_core.DeflectionMap')
        self.initialise_deflection_angles(xdeflect_fits=xdeflect_fits, ydeflect_fits=ydeflect_fits)
        self.z_lens = z_lens

    def initialise_deflection_angles(self, xdeflect_fits, ydeflect_fits):
        """Reads in deflection maps and converts values from arcseconds to degrees"""
        arcseconds_to_degrees = 3600
        self.xdeflect = fits.getdata(xdeflect_fits) / arcseconds_to_degrees
        self.ydeflect = fits.getdata(ydeflect_fits) / arcseconds_to_degrees
        self.logger.debug('We assume that the deflection angles are in units of arcseconds')

    def calc_source_position(self, coordinate, z_src):
        """Calculates source positions for an image plane coordinate using the lens equation.

        Parameters
        ----------
        coordinate : list
            image plane coordinate in degrees (ra, dec)
        z_src : float
             background galaxy redshift
        Returns: ndarray
            source plane coordinate
        """
        deflection = self.get_deflection_at_image_position(coordinate) * lens_efficiency(z_lens=self.z_lens, z_src=z_src)
        source_dec = coordinate[1] - deflection[1]
        source_ra = coordinate[0] + deflection[0] * np.cos(self.center[1] * np.pi / 180)
        return np.hstack((source_ra, source_dec))

    def get_deflection_at_image_position(self, coordinate):
        """Returns deflection angle at image plane coordinate. """
        coord = SkyCoord(ra=coordinate[0] * u.deg, dec=coordinate[1] * u.deg)
        rapix, decpix = coord.to_pixel(self.wcax)
        pos_ind = tuple(np.hstack((decpix, rapix)).astype(int))
        return np.hstack((self.xdeflect[pos_ind], self.ydeflect[pos_ind]))

    def calc_image_pixel_center(self, source_fits, z_src, write_image=True, imagename='image.npy'):
        """
        Computes lensed image using vectorized lens equation

        Parameters
        ----------
        source_fits : str
            filepath of background source fitsfile
        z_src : float
            redshift of source
        write_image : bool
            flag to write lensed image to file
        imagename : str
            name of lensed image file

        Returns
        -------
        ndarray : lensed image
        """

        lens_eff = lens_efficiency(z_lens=self.z_lens, z_src=z_src)
        xmap = self.x_deg + self.xdeflect * lens_eff * np.cos(self.center[1] * np.pi / 180)
        ymap = self.y_deg - self.ydeflect * lens_eff
        x, y = setup_coordinate_grid(source_fits, ndim=1)
        x = np.flip(x)
        source_map_interp = RectBivariateSpline(y, x, np.flip(fits.getdata(source_fits), 1))
        image = source_map_interp.ev(ymap, xmap)
        if write_image:
            np.save(imagename, image)
        return image

    def calc_magnification(self, sourcefits, z_src, write_image=False, imagename='test'):
        """ Returns total magnification. Shares parameters with calc_image_pixel_center"""
        srcsum = fits.getdata(sourcefits).sum()
        srchdr = fits.getheader(sourcefits)
        srcpixelscale_deg = srchdr['CDELT2']

        image = self.calc_image_pixel_center(sourcefits, z_src, write_image=write_image,imagename=imagename)

        return image.sum() / srcsum * (self.pix_scale_deg / srcpixelscale_deg) ** 2


def lens_efficiency(z_lens, z_src):
    """Calculates the efficiency of the lensing system given source and lens redshifts."""
    dist_lens_source = cosmo.angular_diameter_distance_z1z2(z_lens, z_src).value
    dist_observer_source = cosmo.angular_diameter_distance(z_src).value
    return dist_lens_source / dist_observer_source
