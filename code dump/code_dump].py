 from numpy import cross, eye, dot
from scipy.linalg import expm3, norm
from scipy.interpolate import griddata

    def rotate_disk(self, axis, theta):
        """ For each coordinate in the rest frame, i.e. with a face on disk, what is the coordinate in the observation
        frame which maps to the coordinate in the rest frame"""
        def rotation_matrix(axis, inverse_rotation_deg):
            """generates a 3X3 rotation matrix"""
            return expm3(cross(eye(3), axis / norm(axis) * inverse_rotation_deg))


        axis, theta = [0,0,1],45.* np.pi/180. # inverse rotation = theta
        M0 = rotation_matrix(axis, theta)

        x_rest, y_rest, z_rest = np.mgrid[-self.grid_length_arcsec / 2:self.grid_length_arcsec / 2:1j * self.n_pix,
                                          -self.grid_length_arcsec / 2:self.grid_length_arcsec / 2:1j * self.n_pix,
                                          -self.grid_length_arcsec / 2:self.grid_length_arcsec / 2:1j * self.n_pix]
        interpolated_coords = np.zeros((self.n_pix, self.n_pix, self.n_pix, 3))
        for ind0 in range(self.n_pix):
            for ind1 in range(self.n_pix):
                for ind2 in range(self.n_pix):
                    vec = np.array([x_rest[ind0, ind1, ind2], y_rest[ind0, ind1, ind2], z_rest[ind0, ind1, ind2]])
                    interpolated_coords[ind0, ind1, ind2, :] = dot(M0, vec)

        stacked_interp = np.vstack((interpolated_coords[:, :, :, 0].flatten(),
                                    interpolated_coords[:, :, :, 1].flatten(),
                                    interpolated_coords[:, :, :, 2].flatten()))

rotated_mass = griddata(np.swapaxes(stacked_interp,0,1),den3_conv.flatten(), (obs_x,obs_y,obs_z), method='nearest')

imshow(nansum(rotated_mass,axis=2),origin='lower')