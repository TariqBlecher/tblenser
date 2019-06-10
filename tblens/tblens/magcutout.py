from map_utils_core import *
from grid_creators import *
import astropy.units as u


class MagCutTool(PositionGrid):

    def __init__(self, gamma_fits, kappa_fits, center=True, z_lens=0.1):
        PositionGrid.__init__(self, gamma_fits, center=center)
        self.gamma = fits.getdata(gamma_fits)
        self.kappa = fits.getdata(kappa_fits)
        self.z_lens = z_lens

    def calc_mag(self, coord, flux_radius_as, z, return_array=False):
        cutout_ind = self.get_cutout_ind(coord, flux_radius_as)
        eff = lens_efficiency(self.z_lens, z)
        gamma_cut = self.gamma[cutout_ind] * eff
        kappa_cut = self.kappa[cutout_ind] * eff
        mag = 1/((1-kappa_cut)**2-gamma_cut**2)
        if return_array:
            return mag
        else:
            return np.mean(np.abs(mag))

    def get_cutout_ind(self, coord, flux_radius_as):
        xnew = self.x_arcsec-coord[0]
        ynew = self.y_arcsec-coord[1]
        twodsum = np.sqrt(xnew**2 + ynew**2)
        cutout_ind = twodsum <= flux_radius_as
        return cutout_ind


