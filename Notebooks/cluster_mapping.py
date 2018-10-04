from cluster_playground import *
import os


class LensMapper(ClusterLens):
    def __init__(self, input_file_name='slacs.input', length_arcsec=80., pix_res=0.5, lens_z=0.375, prefix='test'):
        ClusterLens.__init__(self, input_file_name, length_arcsec, pix_res, lens_z, prefix)

    def magnification_map(self, zsrc, source_sigma, pix_buffer, pix_step):
        """Assume Lenses have all been set up but no sources"""
        out_folder = './'+self.prefix
        os.mkdir(out_folder)
        y_arcsec, x_arcsec = np.mgrid[-1 * self.length_arcsec / 2:self.length_arcsec / 2:1j * self.num_pix,
                             -1 * self.length_arcsec / 2:self.length_arcsec / 2:1j * self.num_pix]
        shapely = (x_arcsec.shape[0]-2*pix_buffer)/pix_step
        mag = np.zeros((shapely, shapely)) ###MESSYNESS!! probably clashes later on
        for xind in range(pix_buffer, (x_arcsec.shape[0]-pix_buffer), pix_step):
            for yind in range(pix_buffer, (x_arcsec.shape[1]-pix_buffer), pix_step):
                xoff = x_arcsec[xind, yind]
                yoff = y_arcsec[xind, yind]
                new_prefix = self.prefix+'%04i_%04i' % (xind, yind)
                self.add_gaussian_src(z=zsrc, sigma_arcsec=source_sigma, x_off=xoff, y_off=yoff)
                self.calc_lensed_image(new_prefix=new_prefix)
                mag[xind, yind] = self.calc_mag(new_prefix=new_prefix)
                self.clear_sources()
        os.system('mv ./*.fits ' + out_folder)
        np.save(out_folder+'/mu_map.npy', mag)
        return mag
