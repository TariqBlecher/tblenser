import numpy as np
import pylab as pl
import sys
from astLib import astCoords
import subprocess
import os
import pyfits as pf
from astropy.cosmology import Planck15 as cosmo
import yt
from yt.analysis_modules.ppv_cube.api import PPVCube
from scipy.ndimage.interpolation import rotate as imrotate
from copy import deepcopy


class LensInstance(object):
    """An object which controls for the lensing of a source. In the working directory, there should be a
    glafic binary. Assume lens is at centre for now.

    """
    def __init__(self, input_file, source_dictionary, index):
        self.index = str(index)
        self.source_dictionary = deepcopy(source_dictionary)
        newname = input_file.split('.input')[0]+self.index+'.input'
        os.system('cp %s %s' % (input_file, newname))
        self.input_file = newname
        self.prefix = 'abell2744'+self.index

    def gaussian_source_string(self):
        """extend gauss redshift[1] density_normalisation[2] x[3] y[4] e[5] theta_e[6] sigma[7] n/a [8]"""
        source_string = "extend gauss %f %f %f %f %f %f %f 0" % (self.source_dictionary['z'],
                                                                 self.source_dictionary['density_normalisation'],
                                                                 self.source_dictionary['x_off'],
                                                                 self.source_dictionary['y_off'],
                                                                 self.source_dictionary['ellipticity'],
                                                                 self.source_dictionary['position_angle'],
                                                                 self.source_dictionary['sigma_arcsec'])
        return source_string

    def write_input_file(self):
        src_string = self.gaussian_source_string()
        inputfile = open(self.input_file, 'r+')
        morelines = (src_string, '\n',
                     'end_startup', '\n',
                     'start_command', '\n',
                     'writeimage', '\n',
                     'writeimage_ori', '\n',
                     'quit', '\n')
        inputfile.writelines(morelines)

    def run_glafic(self):
        outfile = file(self.prefix+'_output.txt', 'w')
        subprocess.call(['./glafic', self.input_file], stdout=outfile, stderr=outfile)
        outfile.close()

    def calc_mag(self):
        original_image = pf.getdata(self.prefix+'_source.fits')
        lensed_image = pf.getdata(self.prefix+'_image.fits')
        return lensed_image.sum()/float(original_image.sum())

    def write_and_run(self):
        self.write_input_file()
        self.run_glafic()
        mag=self.calc_mag()
        return mag




