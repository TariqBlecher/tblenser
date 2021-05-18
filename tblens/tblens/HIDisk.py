"""This module defines only one object, HIDisk, which creates a neutral atomic hydrogen (HI) gas disk. See documentation of HIDisk.HIDisk for details."""

import logging
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage.interpolation import rotate
from astropy.cosmology import Planck15 as cosmo
from astropy.io import fits
from tblens.utils import find_turning_points, convert_pc_to_arcsec, set_borders_to_zero, HI_mass_size_relation


class HIDisk(object):
    """
    This class creates an axisymmetric neutral atomic hydrogen (HI) disk. 
    
    The radial mass density profile is based on the Obreschkow et al. (2009) model, however in constrast to Obreschkow (2009), the exponential scale radius (rdisk) is set under the constraint of the HI mass-size relation, see Blecher et al. (2019) for details.
    
    Note that the angular length of the coordinate grid is set dynamically based on the size of the HI disk. 

    Attributes
    ----------
    mh : float
        Total hydrogen mass of galaxy (atomic plus molecular)
    n_pix : int
        Number of pixels spanning the length of the grid
    log10_mhi : float
        log (base 10) of the total HI mass
    rcmol : float
        A quantity which determines the ratio of H2/HI in the Obreschkow (2009) model
    r1_pc : float
        Radius at which HI density reaches 1 Msun pc^(-2)
    rdisk_arcsec : float
        Exponential scale length of Hydrogen disk in units of arcsec
    flux_JyHz : float
        Total flux of source in JyHz
    grid_length_arcsec : float
        Length of coordinate grid in arcsec
    pixel_scale_arcsec : float
        pixel length in arcsec
    z_src : float
        redshift of source
    x : np.ndarray
        2D numpy array representing RA coordinates of each pixel of the grid. Centered at zero.
    y : np.ndarray
        2D numpy array representing DEC coordinates of each pixel of the grid. Centered at zero.
    r: np.ndarray
        2D numpy array representing radius from center across the grid. Centered at zero.
    disk_3d : np.ndarray
        3D numpy array of HI disk
    disk_2d : np.ndarray
        2D numpy array of HI disk
    disk_2d_flux_normalised : np.ndarray
        2D numpy array of flux-normalised HI disk in units of JyHz arcsec^(-2)

    Methods
    -------
    set_grid_length
        Sets grid length relative to the size of the HI disk
    normalise_flux_to_units_of_JyHz_per_arcsec_squared
        Flux normalisation of twod_disk attribute
    HI_flux
        Calculates HI flux in JyHz, ref: Meyer (2017)
    total_hydrogen_mass
        Calculates total Hydrogen (HI+H2) mass from rcmol and mhi
    solve_for_rdisk(log_rdisk_pc_range=(2, 6))
        Estimate rdisk by constraining the Obreschkow 2009 model with the HI mass-size relation
    create_grid()
        Create 2D coordinate systems corresponding to x, y and r
    create_face_on_disk()
        Create face-on 3D HI Disk
    update_fits_header(hdr, ra_dec_deg)
        Create fits header for HI disk from template, which is usually the lens map fits header
    writeto_fits(name, hdr, ra_dec_deg, flux_norm=False)
        Writes HI disk to fits file
    """
    def __init__(self, n_pix=100, log10_mhi=9, z_src=0.4, rcmol=1., inclination_angle_degrees=0, position_angle_degrees=0,
                 smoothing_height_pix=0, grid_scaling_factor=10):
        """
        Parameters
        ----------
        n_pix : int
            Number of pixels spanning the length of the grid
        log10_mhi : float
            log (base 10) of the total HI mass
        z_src : float
            redshift of source
        rcmol : float
            A quantity which determines the ratio of H2/HI in the Obreschkow (2009) model
        inclination_angle_degrees : float
            HI disk inclination angle in degrees [0, 90]
        position_angle_degrees : float
            HI disk position angle in degrees [0, 180]
        smoothing_height_pix : float 
            Gaussian smoothing height of disk (0. for thin disk, i.e. no smoothing)
        grid_scaling_factor : float
            Sets grid_length via grid_length = grid_scaling_factor * rdisk
        """
        self._logger = logging.getLogger('tblens.HIDisk.HIDisk')
        self.rcmol = rcmol
        self.log10_mhi = log10_mhi
        self.z_src = z_src
        self.n_pix = n_pix

        self.mh = self.total_hydrogen_mass() 
        self.r1_pc = HI_mass_size_relation(log10_mhi) 
        self.rdisk_arcsec = self.solve_for_rdisk()
        self.flux_JyHz = self.HI_flux()

        self.grid_length_arcsec = self.set_grid_length(grid_scaling_factor)
        self.pixel_scale_arcsec = self.grid_length_arcsec / self.n_pix

        self.create_grid()
        self.disk_3d = self.create_face_on_disk(smoothing_height_pix)
        self.disk_3d = rotate(self.disk_3d, inclination_angle_degrees, axes=(2, 0), reshape=False)
        self.disk_3d = rotate(self.disk_3d, position_angle_degrees, axes=(1,0),reshape=False)
        self.disk_2d = np.sum(self.disk_3d, axis=2)
        self.disk_2d_flux_normalised = self.normalise_flux_to_units_of_JyHz_per_arcsec_squared()

    def set_grid_length(self, grid_scaling_factor):
        """Sets grid length relative to the size of the HI disk"""
        grid_length_arcsec = np.ceil(grid_scaling_factor * self.rdisk_arcsec)
        if grid_scaling_factor < 6:
            self._logger.warning('grid_scaling_factor < 6, the full HI distribution may not be captured.')
        elif grid_scaling_factor > 20:
            self._logger.warning('grid_scaling_factor > 20, the HI distribution may suffer from pixelisation errors.')

        return grid_length_arcsec

    def normalise_flux_to_units_of_JyHz_per_arcsec_squared(self):
        """Flux normalisation of twod_disk attribute"""
        return self.disk_2d * self.flux_JyHz / (np.sum(self.disk_2d) * self.pixel_scale_arcsec**2)

    def HI_flux(self):
        """Calculates HI flux in JyHz, ref: Meyer (2017)"""
        return 10**self.log10_mhi / (49.7 * cosmo.luminosity_distance(self.z_src).value ** 2)

    def total_hydrogen_mass(self):
        """Calculates total Hydrogen (HI+H2) mass, ref: Obreschkow 2009"""
        return 10 ** self.log10_mhi * (1 + (3.44 * self.rcmol**(-0.506) + 4.82 * self.rcmol**(-1.054))**-1)

    def solve_for_rdisk(self, log_rdisk_pc_range=(2, 6)):
        """This function finds a value of rdisk which satisfies the HI-mass size relation.

            Note that in the Obreschkow (2009) model, there can be two numerical solutions for rdisk. The first solution puts most of the HI mass at r<R1 while the second solution puts most of the HI mass at r>R1. As the second scenario is unphysical, we choose the first solution to rdisk.

            Parameters
            ----------
            log_rdisk_pc_range : tuple
                lower and upper bounds to search for rdisk solution, syntax : (lower, upper). The current default values are (100 pc, 100 kpc) which should be fine for all physically possible HI disks. However, if there is a crash with 'IndexError: list index out of range', it means that a minimum for rdisk is not found within bounds.

            Returns
            -------
            float
                solution for rdisk in parsec
        """

        def residual_density_at_r1_squared(r_disk):
            """This function is minimised in order to solve for rdisk. density equation ref : Obreschkow (2009)"""
            density_at_r1 = self.mh / (2 * np.pi * r_disk ** 2) * np.exp(-self.r1_pc / r_disk) / (1 + self.rcmol * np.exp(-1.6 * self.r1_pc / r_disk))
            return (density_at_r1 - 1)**2

        rdisk_search_space = np.logspace(log_rdisk_pc_range[0], log_rdisk_pc_range[1], 4000)
        turning_points = find_turning_points(residual_density_at_r1_squared(rdisk_search_space))
        rdisk_pc = rdisk_search_space[turning_points[0][0]]    # # Picks the first minima
        rdisk_arcsec = convert_pc_to_arcsec(rdisk_pc, self.z_src)

        return rdisk_arcsec

    def create_grid(self):
        """Creates standard 2D coordinate arrays"""
        self.y, self.x = np.mgrid[-self.grid_length_arcsec / 2:self.grid_length_arcsec / 2:1j * self.n_pix,
                                  -self.grid_length_arcsec / 2:self.grid_length_arcsec / 2:1j * self.n_pix]
        self.r = np.sqrt(self.x * self.x + self.y * self.y)

    def create_face_on_disk(self, smoothing_height_pix):
        """Creates face-on 3D HI disk, density ref : Obreschkow (2009)"""
        twod_density = np.exp(- self.r / self.rdisk_arcsec) / (1 + self.rcmol * np.exp(-1.6 * self.r / self.rdisk_arcsec))
        disk_3d = np.zeros((self.n_pix, self.n_pix, self.n_pix))
        disk_3d[:, :, int(self.n_pix / 2)] = twod_density
        if smoothing_height_pix:
            disk_3d = gaussian_filter1d(disk_3d, smoothing_height_pix, axis=2)
        return disk_3d

    def update_fits_header(self, hdr, ra_dec_deg):
        """Create fits header for HI disk from template, usually lens fits header"""
        hdr['CDELT1'], hdr['CDELT2'] = np.array([-1, 1]) * self.pixel_scale_arcsec / 3600.
        hdr['CD1_1'], hdr['CD2_2'] = np.array([-1, 1]) * self.pixel_scale_arcsec / 3600.
        hdr['NAXIS1'], hdr['NAXIS2'] = np.ones(2) * self.n_pix
        hdr['CRPIX1'], hdr['CRPIX2'] = np.ones(2) * self.n_pix / 2.
        hdr['CRVAL1'], hdr['CRVAL2'] = ra_dec_deg
        return hdr

    def writeto_fits(self, name, hdr, ra_dec_deg, flux_norm=False):
        """Writes HI disk to fits file"""
        hdr = self.update_fits_header(hdr, ra_dec_deg)
        if flux_norm:
            src = self.disk_2d_flux_normalised
        else:
            src = self.disk_2d
        src = set_borders_to_zero(src)  # Avoids interpolation errors in lensing
        fits.writeto(name, src, hdr, overwrite=True)
        self.fitsname = name
