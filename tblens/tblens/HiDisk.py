import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage.interpolation import rotate
from astropy.cosmology import Planck15 as cosmo
from astropy.io import fits
from utils import turning_points, convert_pc_to_arcsec, set_borders_to_zero, HI_mass_size_relation


class HiDisk(object):
    """
    This class creates a 3D HI disk based on the Obreschkow 2009 model.

    Attributes
    ----------
    mh : float
        Total hydrogen mass of galaxy (atomic and molecular)
    n_pix : int
        Number of pixels spanning the length of the 3D grid
    log10_mhi : float
        log (base 10) of the total HI mass
    rcmol : float
        A quantity is related to the ratio of H2/HI in the Obreschkow 2009 model
    smoothing_height_pix : float / False
        Gaussian smoothing height of disk (False for no smoothing)
    r1_pc : float
        Radius at which density reaches 1 Msun pc^(-2)
    rdisk_arcsec : float
        Exponential scale length of Hydrogen disk
    flux_JyHz : float
        Total flux of source in JyHz
    grid_length_arcsec : float
        Length of coordinate grid in arcsec
    pixel_scale_arcsec : float
        pixel size in arcsec
    x : np.ndarray
        2D numpy array of RA across the grid. Centered at zero.
    y : np.ndarray
        2D numpy array of DEC across the grid. Centered at zero.
    r: np.ndarray
        2D numpy array of radius from center across the grid. Centered at zero.
    disk : np.ndarray
        3D numpy array of HI disk
    twod_disk : np.ndarray
        2D numpy array of HI disk
    twod_disk_normed : np.ndarray
        2D numpy array of flux-normalised HI disk

    Methods
    -------
    solve_for_rdisk(log_rdisk_pc_range=[2, 6])
        Estimate rdisk based on R1, MHI, MTotal within Obreschkow 2009 model
    create_grid()
        Create coordinate grid corresponding to x,y,r
    create_face_on_disk()
        Create face-on 3D HI Disk
    rotate_disk(theta_deg=2, plane_of_rotation=(2, 0), reshape=False)
        Rotate 3D HI disk
    roll_disk_pix(shiftxy_tuple)
        Shift disk in RA/DEC by pixel tuple
    update_fits_header(hdr, ra_dec_deg)
        Create fits header for HI disk from template, usually lens fits header
    writeto_fits(name, hdr, ra_dec_deg, flux_norm=False)
        Writes HI disk to fits file
    """
    def __init__(self, n_pix=100, x_off_arcsec=0, y_off_arcsec=0, log10_mhi=9,
                 z_src=0.4, rcmol=1., inclination_degrees=0, position_angle_degrees=0,
                 smoothing_height_pix=False, grid_scaling_factor=10, grid_size_min_arcsec=3.):
        """
        Parameters
        ----------
        n_pix : int
            Number of pixels spanning the length of the 3D grid
        x_off_arcsec : float
            HI centroid offset in right ascension from lens centroid
        y_off_arcsec : float
            HI centroid offset in declination from lens centroid
        log10_mhi : float
            log (base 10) of the total HI mass
        z_src : float
            redshift of HI disk
        rcmol : float
            A quantity is related to the ratio of H2/HI in the Obreschkow 2009 model
        inclination_degrees : float
            HI disk inclination angle
        position_angle_degrees : float
            HI disk position angle
        smoothing_height_pix : float / False
            Gaussian smoothing height of disk (False for no smoothing)
        grid_scaling_factor : float
            Sets grid_length via, grid_length = grid_scaling_factor * rdisk
        grid_size_min_arcsec : float
            Grid is restricted to be larger than this amount
        """

        # # Fundamental physical parameters
        self.Rcmol = rcmol
        self.smoothing_height_pix = smoothing_height_pix
        self.log10_mhi = log10_mhi
        self.mh = 10 ** log10_mhi * (1 + (3.44 * rcmol**(-0.506) + 4.82 * rcmol**(-1.054))**-1)  # # ref : Obreschkow 2009
        self.r1_pc = HI_mass_size_relation(log10_mhi) * 1e3
        self.rdisk_arcsec = self.solve_for_rdisk(log10_mhi, z_src)
        self.flux_JyHz = 10**log10_mhi / (49.7 * cosmo.luminosity_distance(z_src).value ** 2)

        # # Grid properties
        self.n_pix = n_pix
        self.grid_length_arcsec = np.ceil(grid_scaling_factor * self.rdisk_arcsec)
        if self.grid_length_arcsec < grid_size_min_arcsec:
            self.grid_length_arcsec = grid_size_min_arcsec
        self.pixel_scale_arcsec = self.grid_length_arcsec / self.n_pix

        # # Disk creation
        self.create_grid()
        self.create_face_on_disk()
        self.rotate_disk(theta_deg=inclination_degrees)
        self.rotate_disk(theta_deg=position_angle_degrees, plane_of_rotation=(1, 0))
        self.roll_disk_pix((int(y_off_arcsec / self.pixel_scale_arcsec), int(x_off_arcsec / self.pixel_scale_arcsec)))
        self.twod_disk = np.sum(self.disk, axis=2)
        # # Convert to units of JyHz arcsec^(-2)
        self.twod_disk_normed = self.twod_disk * self.flux_JyHz / (np.sum(self.twod_disk) * self.pixel_scale_arcsec**2)

    def solve_for_rdisk(self, log_rdisk_pc_range=[2, 6]):
        """This function finds a value of rdisk which satisfies the condition that the HI mass density at r=R1 is 1 Msun/pc^2.

            Note that in this particular model, there are two numerical solutions for rdisk. The first puts most of the HI mass
            before R1 while the second puts most of the HI mass after R1. As the second scenario is unphysical,
            we pick the first solution to rdisk.

            Parameters
            ----------
            log_rdisk_pc_range : list
                lower and upper bounds to search for rdisk solution

            Returns
            -------
            float
                solution for rdisk
        """

        def residual_density_at_r1_squared(r_disk):
            """This function is minimised in order to solve for rdisk"""
            density_at_r1 = self.mh / (2 * np.pi * r_disk ** 2) * np.exp(-self.r1_pc / r_disk) / (1 + self.Rcmol * np.exp(-1.6 * self.r1_pc / r_disk))
            return (density_at_r1 - 1)**2

        rdisk_search_space = np.logspace(log_rdisk_pc_range[0], log_rdisk_pc_range[1], 4000)
        turn_points = turning_points(residual_density_at_r1_squared(rdisk_search_space))
        rdisk_pc = rdisk_search_space[turn_points[0][0]]    # # Picks the first minima
        rdisk_arcsec = convert_pc_to_arcsec(rdisk_pc, self.z_src)

        return rdisk_arcsec

    def create_grid(self):
        """Creates basic 2D coordinate arrays"""
        self.y, self.x = np.mgrid[-self.grid_length_arcsec / 2:self.grid_length_arcsec / 2:1j * self.n_pix,
                                  -self.grid_length_arcsec / 2:self.grid_length_arcsec / 2:1j * self.n_pix]
        self.r = np.sqrt(self.x * self.x + self.y * self.y)

    def create_face_on_disk(self):
        """Creates face-on 3D HI disk"""
        twod_density = np.exp(- self.r / self.rdisk_arcsec) / (
            1 + self.Rcmol * np.exp(-1.6 * self.r / self.rdisk_arcsec))
        self.disk = np.zeros((self.n_pix, self.n_pix, self.n_pix))
        self.disk[:, :, int(self.n_pix / 2)] = twod_density
        if self.smoothing_height_pix:
            self.disk = gaussian_filter1d(self.disk, self.smoothing_height_pix, axis=2)

    def rotate_disk(self, theta_deg=2, plane_of_rotation=(2, 0), reshape=False):
        """Rotates 3D HI disk"""
        rotated_disk = rotate(self.disk, theta_deg, axes=plane_of_rotation, reshape=reshape)
        self.disk = rotated_disk

    def roll_disk_pix(self, shiftxy_tuple):
        """Shifts HI disk by in (x,y) directions."""
        rolled_disk = np.roll(self.disk, shiftxy_tuple, axis=(1, 0))
        self.disk = rolled_disk

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
            src = self.twod_disk_normed
        else:
            src = self.twod_disk
        src = set_borders_to_zero(src)  # Avoids interpolation errors in lensing
        fits.writeto(name, src, hdr, overwrite=True)
        self.fitsname = name
