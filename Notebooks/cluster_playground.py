import numpy as np
import sys
from astLib import astCoords
import subprocess
import os
import pyfits as pf
from astropy.cosmology import Planck15 as cosmo
from copy import deepcopy
import time


class ClusterLens(object):
    """An object which controls for the lensing of a source. In the working directory, there should be a
    glafic binary. Assume lens is at centre for now.
    """
    def __init__(self, input_file_name='slacs.input', length_arcsec=100., pix_res=0.1, lens_z=0.375, prefix='test'):
        self.input_file = input_file_name
        self.prefix = prefix
        self.length_arcsec = length_arcsec
        self.num_pix = int(self.length_arcsec/float(pix_res))
        self.primary_parameters = self.primary_parameter_dict(lens_z, length_arcsec, pix_res)
        self.num_lenses = 0
        self.num_srcs = 0
        self.lens_string = []
        self.src_string = []
        self.point_src_string = []
        self.num_point_srcs = 0

    def primary_parameter_dict(self, zl, length_arcsec, pix_res):
        return {"omega": cosmo.Om0, "lambda": cosmo.Ode0, "hubble": (cosmo.H0.value / 100.),
                "weos": -1.0,
                "zl": zl, "prefix": self.prefix,
                "xmin": -1 * length_arcsec / 2, "xmax": length_arcsec / 2,
                "ymin": -1 * length_arcsec / 2, "ymax": length_arcsec / 2,
                "pix_ext": pix_res, "pix_poi":pix_res
                }

    def add_sie(self, v_sigma_kms=600, x_offset=0, y_offset=0, ellipticity=0, position_angle_deg_eastofnorth=0):
        """#lens   sie sigma_kms[1] x[2] y[3] e[4]=0 spherical  theta_e_deg_eastofnorth[5] rcore[6] n/a[7]
        Assume lens is at centre for now."""
        sie_string = "lens sie %f %f %f %f %f 0 0" % (v_sigma_kms, x_offset, y_offset, ellipticity,
                                                      position_angle_deg_eastofnorth)
        self.num_lenses += 1
        self.lens_string.append(sie_string)
        self.lens_string.append('\n')

    def add_jaffe(self, v_sigma_kms=600, x_offset=0, y_offset=0, ellipticity=0,
                  position_angle_deg_eastofnorth=0, r_trunc=100., r_core=5.):
        """#lens   jaffe sigma_kms[1] x[2] y[3] e[4]=0 spherical  theta_e_deg_eastofnorth[5] rtrunc[6] rcore[7]
        Assume lens is at centre for now."""
        jaffe_string = "lens jaffe %f %f %f %f %f %f %f" % (v_sigma_kms, x_offset, y_offset, ellipticity,
                                                      position_angle_deg_eastofnorth, r_trunc, r_core)
        self.num_lenses += 1
        self.lens_string.append(jaffe_string)
        self.lens_string.append('\n')

    def add_gaussian_src(self, z=1, norm=1, sigma_arcsec=1, x_off=0, y_off=0, ellipticity=0, position_angle=0):
        """extend gauss redshift[1] density_normalisation[2] x[3] y[4] e[5] theta_e[6] sigma[7] n/a [8]"""
        source_string = "extend gauss %f %f %f %f %f %f %f 0" % (z, norm, x_off, y_off, ellipticity,
                                                                 position_angle, sigma_arcsec)
        self.num_srcs += 1
        self.src_string.append(source_string)
        self.src_string.append('\n')

    def write_input_file(self, new_prefix=False):
        """Writes the lens component of the input file"""

        first_line = ('\n', 'startup %i %i 0' % (self.num_lenses, self.num_srcs), '\n')
        final_lines = ('end_startup', '\n \n',
                       'start_command', '\n',
                       'writeimage', '\n',
                       'writeimage_ori', '\n \n'
                                         'quit')
        inputfile = open(self.input_file, 'w')
        if new_prefix:
            self.primary_parameters["prefix"] = new_prefix
        for key in self.primary_parameters.keys():
            inputfile.write('%s %s \n' % (key, self.primary_parameters[key]))
        inputfile.writelines(first_line)
        inputfile.writelines(self.lens_string)
        inputfile.writelines(self.src_string)
        inputfile.writelines(final_lines)
        inputfile.close()

    def calc_lensed_image(self, new_prefix=False):
        self.write_input_file(new_prefix=new_prefix)
        self.run_glafic()

    def run_glafic(self):
        outfile = file(self.prefix + '_output.txt', 'w')
        subprocess.call(['./glafic', self.input_file], stdout=outfile, stderr=outfile)
        outfile.close()

    def clear_sources(self):
        self.num_srcs = 0
        self.num_point_srcs = 0
        self.src_string = []
        self.point_src_string = []

    def widget_sim(self, zsrc=1, sigma_arcsec=1, x_off=0, y_off=0):
        self.clear_sources()
        self.add_gaussian_src(z=zsrc, sigma_arcsec=sigma_arcsec, x_off=x_off, y_off=y_off)
        self.calc_lensed_image()
        print self.calc_mag()

    def write_critical_curves(self, zsrc):
        """For some reason, this only works if r_trun and r_core are integers"""

        first_line = ('\n', 'startup %i 0 0' % (self.num_lenses), '\n')
        final_lines = ('end_startup', '\n \n',
                       'start_command', '\n',
                       'writecrit %f'%zsrc, '\n',
                       'quit')
        inputfile = open(self.input_file, 'w')
        for key in self.primary_parameters.keys():
            inputfile.write('%s %s \n' % (key, self.primary_parameters[key]))
        inputfile.writelines(first_line)
        inputfile.writelines(self.lens_string)
        inputfile.writelines(final_lines)
        inputfile.close()
        self.run_glafic()

    def calc_mag(self, new_prefix=False):
        if new_prefix:
            prefix = new_prefix
        else:
            prefix = self.prefix
        original_image = pf.getdata(prefix + '_source.fits')
        lensed_image = pf.getdata(prefix + '_image.fits')
        return lensed_image.sum() / float(original_image.sum())





