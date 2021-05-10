import subprocess
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
from copy import deepcopy


class LensInstance(object):
    """An object which controls for the lensing of a source. In the working directory, there should be a
    glafic binary. Assume lens is at centre for now.

    """
    def __init__(self, lens_dictionary, source_dictionary, input_file_name):

        self.lens_dictionary = deepcopy(lens_dictionary)
        self.source_dictionary = deepcopy(source_dictionary)
        self.input_file = input_file_name

    def lens_string(self):
        if self.lens_dictionary['type']=='sie':
            """#lens   sie sigma_kms[1] x[2] y[3] e[4]=0 spherical  theta_e_deg_eastofnorth[5] rcore[6] n/a[7]
            Assume lens is at centre for now."""
            lens_string = "lens sie %f 0 0 %f %f %f 0" % (self.lens_dictionary['v_sigma_kms'],
                                                         self.lens_dictionary['ellipticity'],
                                                         self.lens_dictionary['position_angle_deg_eastofnorth'],
                                                          self.lens_dictionary['r_core'])
        else:
            lens_string = 'lens point %e 0 0 0 0 0 0' % (self.lens_dictionary['mass']*(cosmo.H0.value/100.))
        return lens_string

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

    def sersic_source_string(self):
        """extend gauss redshift[1] density_normalisation[2] x[3] y[4] e[5] theta_e[6] r_effective[7] power index x [8]"""
        source_string = "extend sersic %f %f %f %f %f %f %f %s" % (self.source_dictionary['z'],
                                                                   self.source_dictionary['density_normalisation'],
                                                                   self.source_dictionary['x_off'],
                                                                   self.source_dictionary['y_off'],
                                                                   self.source_dictionary['ellipticity'],
                                                                   self.source_dictionary['position_angle'],
                                                                   self.source_dictionary['r_e'],
                                                                   self.source_dictionary['sersic_index'])
        return source_string

    def write_input_file(self):
        """Writes the lens component of the input file"""
        primary_parameter_lines = {"omega": cosmo.Om0, "lambda": cosmo.Ode0, "hubble": (cosmo.H0.value/100.),
                                   "weos": -1.0,
                                   "zl": self.lens_dictionary['zl'], "prefix": self.lens_dictionary['prefix'],
                                   "xmin": self.lens_dictionary['xmin_arcsec'], "xmax": self.lens_dictionary['xmax_arcsec'],
                                   "ymin": self.lens_dictionary['ymin_arcsec'], "ymax": self.lens_dictionary['ymax_arcsec'],
                                   "pix_ext": self.lens_dictionary['pix_res']
                                   }
        if self.source_dictionary['source_type'] == 'sersic':
            src_string = self.sersic_source_string()
        else:
            src_string = self.gaussian_source_string()
        main_lines = ('\n','startup 1 1 0', '\n',
                      self.lens_string(), '\n',
                      src_string, '\n',
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
        outfile = open(self.lens_dictionary['prefix']+'_output.txt', 'w')
        subprocess.call(['./glafic', self.input_file], stdout=outfile, stderr=outfile)
        outfile.close()

    def calc_mag(self):
        original_image = fits.getdata(self.lens_dictionary['prefix']+'_source.fits')
        lensed_image = fits.getdata(self.lens_dictionary['prefix']+'_image.fits')
        return lensed_image.sum()/float(original_image.sum())

    def write_and_run(self):
        self.write_input_file()
        self.run_glafic()
        mag=self.calc_mag()
        return mag




