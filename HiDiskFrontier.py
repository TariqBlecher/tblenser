import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage.interpolation import rotate
from astropy.cosmology import Planck15 as cosmo
from utils import turning_points
import sys


class HiDisk(object):
    def __init__(self, n_pix=100, rcmol=1.,
                 smoothing_height_pix=False, theta_2_0=0,
                 theta_1_0=0, x_off=0, y_off=0, log10_mhi=9, z_src=0.4, physical_normalisation=False,
                 scaling_size=10, grid_size_min_arcsec=3., minpixelsizecheck=True):

        # sys.exit('Grid length must be even!')
        self.Rcmol = rcmol
        self.smoothing_height = smoothing_height_pix
        self.mh = 10 ** log10_mhi * (1+(3.44*rcmol**(-0.506)+4.82*rcmol**(-1.054))**-1)
        self.rdisk_arcsec = self.solve_for_rdisk(log10_mhi, z_src)
        self.grid_length_arcsec = np.ceil(scaling_size*self.rdisk_arcsec)

        if self.grid_length_arcsec < grid_size_min_arcsec:
            self.grid_length_arcsec = grid_size_min_arcsec
        self.n_pix = n_pix
        self.pix_res = self.grid_length_arcsec/float(self.n_pix)
        if minpixelsizecheck:         # Stop at small pixel size..for unknown reason glafic crashes below this pixel size
            if self.pix_res < 0.03:
                self.pix_res = 0.03
                self.grid_length_arcsec = self.pix_res*self.n_pix
        if physical_normalisation:
            self.normalisation = self.mh / (2 * np.pi * self.rdisk_arcsec ** 2)
        else:
            self.normalisation = 1

        self.disk = self.face_on_disk()
        self.rotate_disk(theta_deg=theta_2_0)
        self.rotate_disk(theta_deg=theta_1_0, plane_of_rotation=(1, 0))
        self.roll_disk_pix((int(y_off/self.pix_res), int(x_off/self.pix_res)))
        self.twod_disk = np.sum(self.disk, axis=2)

    def solve_for_rdisk(self, log10_mhi, z_src=0.407, log_rdisk_pc_range=[2, 5]):
        def calc_r1(m_hi):
            """Jing Wang et al. 2016"""
            return 0.5 * 10**(0.506 * m_hi - 3.293)

        r1 = calc_r1(log10_mhi) * 1e3

        def function_to_minimise(r_disk, r_1):
            return np.power(self.mh / (2 * np.pi * r_disk ** 2) * np.exp(-r_1 / r_disk) /
                            (1 + self.Rcmol * np.exp(-1.6 * r1 / r_disk)) - 1, 2)

        rdisk_array = np.logspace(log_rdisk_pc_range[0], log_rdisk_pc_range[1], 1000)
        turn_points = turning_points(function_to_minimise(rdisk_array, r1))
        rdisk_soln_pc = rdisk_array[turn_points[0][0]]
        kpc_per_arcsec_src = cosmo.kpc_proper_per_arcmin(z_src).value / 60.
        rdisk_arcsec = rdisk_soln_pc*1e-3/kpc_per_arcsec_src
        return rdisk_arcsec

    def face_on_disk(self):
        """Creates a three dimensional face on HI disk. i.e. disk lives in plane of the first two dimensions"""
        y, x = np.mgrid[-self.grid_length_arcsec / 2:self.grid_length_arcsec / 2:1j * self.n_pix,
                        -self.grid_length_arcsec / 2:self.grid_length_arcsec / 2:1j * self.n_pix]
        r = np.sqrt(x*x+y*y)
        twod_density = self.normalisation*np.exp(-r / self.rdisk_arcsec) / (1 + self.Rcmol * np.exp(-1.6 * r / self.rdisk_arcsec))
        den3 = np.zeros((self.n_pix, self.n_pix, self.n_pix))
        den3[:, :, self.n_pix / 2] = twod_density
        if self.smoothing_height:
            den3 = gaussian_filter1d(den3, self.smoothing_height, axis=2)
        np.save('face_on_density', den3)
        return den3

    def rotate_disk(self, theta_deg=2, plane_of_rotation=(2, 0), reshape=False):
        rotated_disk = rotate(self.disk, theta_deg, axes=plane_of_rotation, reshape=reshape)
        np.save('rotated_disk', rotated_disk)
        self.disk = rotated_disk

    def roll_disk_pix(self, shiftxy_tuple):
        rolled_disk = np.roll(self.disk, shiftxy_tuple, axis=(1, 0))
        np.save('rolled_disk', rolled_disk)
        self.disk = rolled_disk




