import sys
import time
import numpy as np
from HiDisk import HiDisk
from lens_nonparametric_src import LensPoints

z_src = 0.4
l_z = 0.1

grid_length = 10
npix = 100
src_threshold = 0.01
smoothing_height_pix = False
rcmol = 1
mhi = 9
x_off = 0
y_off = 0
theta_10 = 45
theta_20 = 90

# noinspection PyTypeChecker,PyTypeChecker
hidisk = HiDisk(grid_length_arcsec=grid_length, n_pix=npix,
                rcmol=rcmol, smoothing_height_pix=smoothing_height_pix,
                theta_2_0=theta_20, theta_1_0=theta_10,
                x_off=x_off, y_off=y_off, log10_mhi=mhi, z_src=z_src)

nonparametric_sim = LensPoints(input_file_name='single.input', prefix='single',
                               length_arcsec=hidisk.grid_length_arcsec,
                               pix_res=hidisk.pix_res,
                               lens_z=l_z, source_data=hidisk.twod_disk,
                               src_threshold=src_threshold, z_src=z_src)


