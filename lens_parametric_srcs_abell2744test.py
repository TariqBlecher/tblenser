import pyfits as pf
import subprocess
from copy import deepcopy


class LensInstance(object):
    """An object which controls for the lensing of a source. In the working directory, there should be a
    glafic binary. Assume lens is at centre for now.

    """
    def __init__(self, source_dictionary, index):
        self.index = str(index)
        self.source_dictionary = deepcopy(source_dictionary)
        self.input_file = self.index+'.input'
        self.prefix = 'abell2744'+self.index
        self.setup_lines = ('omega     0.3075\n',
               'lambda\t  0.691\n',
               'weos\t  -1.0\n',
               'hubble\t  67.74\n',
               'zl\t  0.308\n',
               'prefix\t  %s\n'%self.prefix,
               'xmin\t  -80.0\n',
               'ymin\t  -85.0\n',
               'xmax\t  81.0\n',
               'ymax\t  98.0\n',
               'pix_ext   0.2\n',
               'pix_poi   0.2\n',
               'maxlev\t  4\n',
               '\n',
               '## some examples of secondary parameters\n',
               'galfile        galfile_abell2744.dat\n',
               'addwcs         0\n',
               '\n',
               '## define lenses and sources\n',
               'startup 7 1 0\n',
               'lens   nfw      3.960613e+14 -3.397677e-01  2.959851e+00  3.410440e-01 -1.299874e+01  4.183545e+00  0.000000e+00 \n',
               'lens   nfw      1.438768e+14 -1.864913e+01 -1.797662e+01  3.746067e-01 -5.144515e+01  9.565266e+00  0.000000e+00 \n',
               'lens   nfw      2.485944e+13 -2.697214e+01  3.090671e+01  7.999937e-01 -7.311822e+01  1.000000e+01  0.000000e+00 \n',
               'lens   gals     1.868273e+02  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  1.018051e+02  1.499830e+00 \n',
               'lens   pert     2.000000e+00  0.000000e+00  0.000000e+00  9.859616e-02  1.403274e+02  0.000000e+00  0.000000e+00 \n',
               'lens   mpole    2.000000e+00  0.000000e+00  0.000000e+00  4.217461e-03  7.253089e+01  3.000000e+00  2.000000e+00 \n',
               'lens   mpole    2.000000e+00  0.000000e+00  0.000000e+00  1.154270e-02  9.272872e+01  4.000000e+00  2.000000e+00 \n',
                self.gaussian_source_string(),
               'end_startup\n',
               '\n',
               '\n',
               '## execute commands\n',
               'start_command\n',
               'writeimage\n',
               'writeimage_ori\n',
               '\n',
 'quit\n')
        self.write_input_file()
        self.run_glafic()
        self.mag = self.calc_mag()

    def gaussian_source_string(self):
        """extend gauss redshift[1] density_normalisation[2] x[3] y[4] e[5] theta_e[6] sigma[7] n/a [8]"""
        source_string = "extend gauss %f %f %f %f %f %f %f 0 \n" % (self.source_dictionary['z'],
                                                                    1,
                                                                    self.source_dictionary['x_off'],
                                                                    self.source_dictionary['y_off'],
                                                                    0,
                                                                    0,
                                                                    self.source_dictionary['sigma_arcsec'])
        return source_string

    def write_input_file(self):
        inputfile = open(self.input_file, 'w')
        inputfile.writelines(self.setup_lines)
        inputfile.close()

    def run_glafic(self):
        outfile = file(self.prefix+'_output.txt', 'w')
        subprocess.call(['./glafic', self.input_file], stdout=outfile, stderr=outfile)
        outfile.close()

    def calc_mag(self):
        original_image = pf.getdata(self.prefix+'_source.fits')
        lensed_image = pf.getdata(self.prefix+'_image.fits')
        return lensed_image.sum()/float(original_image.sum())




