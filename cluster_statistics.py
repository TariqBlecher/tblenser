from map_utils import *
from HiDisk import *
from utils import rand_sign, sample_inclination_deg
import time

# Setup timer and log
start = time.time()
log = file('log.txt', 'w'); sys.stdout = log

#Read catalog
parm_file = 'sim_param.txt'
src_names = np.loadtxt(parm_file, skiprows=1, dtype=str, usecols=0)
mhi_log, u_mhi, rcmol_mean, rcmol_sig, z_src, u_zsrc, x_center, y_center,\
impact_parameter_min, impact_parameter_max, u_l_vsig = \
    np.swapaxes(np.loadtxt(parm_file, skiprows=1, dtype=float, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)), 0, 1)

#Initialise deflection map
xdeflectfits = ''
ydeflectfits = ''
defmap = DeflectionMap(xdeflect_fits=xdeflectfits, ydeflect_fits=ydeflectfits)

#Sampling parameters ?? -> dont think this needs to be here
inclination_bound_deg = 90.
position_angle_bound_deg = 180.
src_threshold = 0.01  #  the outer edges won't be magnified much anyway
mass_sampling = 'uniform'


for ind, source in enumerate(src_names):

    # More HI DISC sampling
    mhi = np.random.rand()*2.5 + 8.5
    rcmol = np.random.lognormal(rcmol_mean[ind], rcmol_sig[ind])
    x_off = x_center + rand_sign()/np.sqrt(2) * (impact_parameter_min[ind] +
                                                 np.random.rand() * (impact_parameter_max[ind] - impact_parameter_min[ind]))
    y_off = y_center + rand_sign()/np.sqrt(2) * (impact_parameter_min[ind] +
                                                 np.random.rand() * (impact_parameter_max[ind] - impact_parameter_min[ind]))
    theta_10 = np.random.rand()*position_angle_bound_deg
    theta_20 = sample_inclination_deg()

    hidisk = HiDisk(n_pix=defmap.npix, rcmol=rcmol, smoothing_height_pix=False, theta_2_0=theta_20,
                    theta_1_0=theta_10, x_off=x_off, y_off=y_off, log10_mhi=mhi, z_src=z_src[ind], scaling_size=10,
                    physical_normalisation=False)