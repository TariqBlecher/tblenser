import numpy as np
import glob
from lens_parametric_srcs_abell2744test import LensInstance



parm_file='sim_catalog_29.npy'
ra_image_mean, dec_image_mean, z_src, u_zsrc, mhi_log, u_mhi_log, ra_err_deg, dec_err_deg = np.load(parm_file)
parmtracklist = np.sort(glob.glob('../glafic_test/lens_dir/*_parmtrack.npy'))
parmtrack = parmtracklist[0]
parm = np.load(parmtrack)

parmtracks = np.zeros(((len(parmtracklist),) + parm.shape))
for ind, pt in enumerate(parmtracklist):
    parmtracks[ind] = np.load(pt)

ra = parmtracks[:,:,3].flatten()
dec = parmtracks[:,:,4].flatten()

mhi = parmtracks[:,:,2].flatten()
mag = parmtracks[:,:,0].flatten()
rcmol =parmtracks[:,:,1].flatten()

inclination = parmtracks[:,:,5].flatten()
posangle = parmtracks[:,:,6].flatten()
rdisk = parmtracks[:,:,7].flatten()
z = parmtracks[:,:,8].flatten()
nz = parmtracks[:,:,9].flatten()
#Not sure if the ra/dec will work out
mag_glafic = zeros(mhi.shape[0])
for index in range(mhi.shape[0]):
    source_dict = {
        "z":z[index],
        "x_off":ra[index],
        "y_off":dec[index],
        'sigma_arcsec':rdisk[index]/2.
    }
    lensinstance = LensInstance(source_dict,index)
    mag_glafic[index] = lensinstance.mag
np.save('glafic_magnifications',mag_glafic)
