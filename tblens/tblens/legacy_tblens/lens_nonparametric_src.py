import numpy as np
import subprocess
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
import sys


class LensPoints(object):
    """Original class for SIE lens"""
    def __init__(self, input_file_name='slacs.input', prefix='test', length_arcsec=6., pix_res=0.1,
                 lens_v_sigma_kms=220, lens_ellipticity=0., lens_position_angle_deg_eastofnorth=0., lens_z=0.1,
                 source_data=None, src_threshold=0.1, z_src=0.407):
        self.input_file = input_file_name
        self.prefix = prefix
        general_parameters = self.primary_parameter_dict(lens_z, length_arcsec, pix_res)
        sie_lens_string = self.sie_string(lens_v_sigma_kms, lens_ellipticity, lens_position_angle_deg_eastofnorth)
        self.write_srcfile(source_data, src_threshold, length_arcsec, pix_res)
        self.write_input_file(general_parameters, sie_lens_string, z_src)
        self.run_glafic()

    def primary_parameter_dict(self, zl, length_arcsec, pix_res):
        return {"omega": cosmo.Om0, "lambda": cosmo.Ode0, "hubble": (cosmo.H0.value / 100.),
                "weos": -1.0,
                "zl": zl, "prefix": self.prefix,
                "xmin": -1 * length_arcsec / 2, "xmax": length_arcsec / 2,
                "ymin": -1 * length_arcsec / 2, "ymax": length_arcsec / 2,
                "pix_ext": pix_res, "srcfile": self.prefix + 'srcs.dat', 'flag_extnorm': 1,
                'flag_srcsbin': 0, 'srcsbinsize': 0.2
                }

    def sie_string(self, v_sigma_kms, ellipticity, position_angle_deg_eastofnorth):
        """lens   sie sigma_kms[1] x[2] y[3] e[4]=0 spherical  theta_e_deg_eastofnorth[5] rcore[6] n/a[7]
        Assume lens is at centre for now."""
        return "lens sie %f 0 0 %f %f 0 0" % (v_sigma_kms,
                                              ellipticity,
                                              position_angle_deg_eastofnorth)

    def write_srcfile(self, src_image, threshold, length_arcsec, pix_res):
        num_pix = src_image.shape[0]
        y_arcsec, x_arcsec = np.mgrid[-1 * length_arcsec / 2:length_arcsec / 2:1j * num_pix,
                                      -1 * length_arcsec / 2:length_arcsec / 2:1j * num_pix]

        x_ind, y_ind = np.where(src_image/src_image.max() >= threshold)

        lines = []
        for src_ind in range(x_ind.shape[0]):
            # # GLAFIC parameters : flux, x pos, y pos, e, theta_e, r_e, n
            # # x pos, y pos, relative to lens which is at the center.
            # # e, theta_e set to zero. r_e set so that all the flux will be in 1 pixel
            lines.append('%f %f %f 0 0 %f 0.5 \n' % (src_image[x_ind[src_ind], y_ind[src_ind]],
                                                     x_arcsec[x_ind[src_ind], y_ind[src_ind]],
                                                     y_arcsec[x_ind[src_ind], y_ind[src_ind]],
                                                     pix_res))
        if len(lines) == 10000:
            sys.exit('source sampling at GLAFIC limit of 10000 sources. Sources may be undersampled.')
        srcfile = open(self.prefix + 'srcs.dat', 'w')
        srcfile.writelines(lines)
        srcfile.close()

    def write_input_file(self, primary_parameter_lines, sie_string, z_src):
        """Writes the lens component of the input file"""

        main_lines = ('\n', 'startup 1 1 0', '\n',
                      sie_string, '\n',
                      'extend srcs %.3f 1.0 0 0 0 0 0 0' % z_src, '\n',
                      'end_startup', '\n \n',
                      'start_command', '\n',
                      'writeimage', '\n',
                      'writeimage_ori', '\n \n'
                                        'quit')

        inputfile = open(self.input_file, 'w')
        for key in primary_parameter_lines.keys():
            inputfile.write('%s %s \n' % (key, primary_parameter_lines[key]))
        inputfile.writelines(main_lines)
        inputfile.close()

    def run_glafic(self):
        outfile = file(self.prefix + '_output.txt', 'w')
        subprocess.call(['./glafic', self.input_file], stdout=outfile, stderr=outfile)
        outfile.close()

    def calc_mag(self):
        original_image = fits.getdata(self.prefix + '_source.fits')
        lensed_image = fits.getdata(self.prefix + '_image.fits')
        return lensed_image.sum() / float(original_image.sum())

    # def check_lensed_flux_within_image_bounds(self):
    #     lensed_image = fits.getdata(self.prefix + '_image.fits')
    #     peak_flux = lensed_image.max()
    #     border_pixels = np.vstack((lensed_image[0, :], lensed_image[-1, :], lensed_image[0, :], lensed_image[-1, :]))
    #     border_flux_fraction = border_pixels/peak_flux
    #     if border_flux_fraction.max()>0.05:
