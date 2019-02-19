import numpy as np
import pyfits as pf
from scipy.interpolate import RectBivariateSpline
from astropy.cosmology import Planck15 as cosmo
from scipy.ndimage import zoom
import itertools


def setup_coordinate_grid(fits, inverse_ra_convention=False, extra_row_column=False):
    """
    2D coordinate grid for x & y
    :param fits: file from which grid is calculated
    :param inverse_ra_convention: Hack to get around a numpy ascending-only constraint
    :return: coordinate grid
    """
    header = pf.getheader(fits)
    pix_scale_arcsec = header['CDELT2']*3600.
    npix = np.array([header['NAXIS1'], header['NAXIS2']])
    if extra_row_column:
        npix += 1
    image_radius = pix_scale_arcsec * npix / 2.
    if inverse_ra_convention:
        y_arcsec, x_arcsec = np.mgrid[-1 * image_radius[0]:image_radius[0]:1j * npix[0],
                                      -1 * image_radius[1]:image_radius[1]:1j * npix[1]]
    else:
        y_arcsec, x_arcsec = np.mgrid[-1 * image_radius[0]:image_radius[0]:1j * npix[0],
                                      image_radius[1]:-1*image_radius[1]:1j * npix[1]]

    return y_arcsec, x_arcsec


def oned_coordinate_grid(fits, translate=False):
    """
    :param fits: file from which grid is calculated
    :param translate: shift grid center
    :return: 1D x,y arrays
    """
    fits_hdr = pf.getheader(fits)
    npix = fits_hdr['NAXIS1']
    dx = fits_hdr['CDELT2'] * 3600.
    radius = npix * dx / 2.
    x = np.linspace(-1 * radius, radius, npix)
    y = np.linspace(-1 * radius, radius, npix)
    if translate:
        x += fits_hdr['CRVAL1'] * 3600.
        y += fits_hdr['CRVAL2'] * 3600,
    return x, y


def lens_efficiency(z_lens, z_src):
    dist_lens_source = cosmo.angular_diameter_distance_z1z2(z_lens, z_src).value
    dist_observer_source = cosmo.angular_diameter_distance(z_src).value
    return dist_lens_source/dist_observer_source

def line_equation(a, b):
    if b[0] == a[0]: # to protect against infinities
        b[0] = a[0]+1e-5
    slope = (b[1] - a[1])/(b[0] - a[0])
    yintercept = b[1] - slope*b[0]
    return slope, yintercept

def area_polygon(x, y):
    area = 0
    npoints = x.shape[0]
    j = npoints - 1
    for i in range(npoints):
        area = area + (x[j] + x[i]) * (y[j] - y[i])
        j = i
        i += 1
    return area/2.


def Sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def IsPointInTri(pt, v1, v2, v3):
    b1 = Sign(pt, v1, v2) < 0.0
    b2 = Sign(pt, v2, v3) < 0.0
    b3 = Sign(pt, v3, v1) < 0.0

    return ((b1 == b2) & (b2 == b3));


class PositionGrid(object):
    """
    Contains a 2D coordinate grid.
     Given a coordinate, can calculats the index of the grid corresponding to that coordinate
    """
    def __init__(self, fits, center=True):
        self.original_fits = fits
        self.y_arcsec, self.x_arcsec = setup_coordinate_grid(fits)
        self.npix = self.x_arcsec.shape[0]
        self.header = pf.getheader(fits)
        self.pix_scale_arcsec = self.header['CDELT2']*3600.
        self.extent = self.npix * self.pix_scale_arcsec
        self.center = np.zeros(2)
        self.center[0] = self.header['CRVAL1']*3600.
        self.center[1] = self.header['CRVAL2']*3600.
        if center:
            self.recenter_grid(self.center)


        print 'Position Grid init'

    def recenter_grid(self, offset):
        self.y_arcsec += offset[1]
        self.x_arcsec += offset[0]

    def position_index(self, coordinate):
        dist = (self.x_arcsec - coordinate[0])**2. + (self.y_arcsec - coordinate[1])**2.
        pos_ind = np.where(dist == dist.min())
        return pos_ind


class SrcGrid(PositionGrid):
    def __init__(self, fits, center=True):
        PositionGrid.__init__(self, fits, center)
        self.minsrcgrid = np.array([self.x_arcsec.min()-self.pix_scale_arcsec/2.,
                                    self.y_arcsec.min()-self.pix_scale_arcsec/2.])


        def find_intersections_of_grid_and_line(self, vertices_arcsec):
                extra_vertices = []
                vertices_arcsec -= self.minsrcgrid   # #shift vertices with respect to bottom of src grid

                n_grid_lines_above_bottom = np.floor(vertices_arcsec/self.pix_scale_arcsec)
                n_gridlines_between_points = np.abs(n_grid_lines_above_bottom[1, :] - n_grid_lines_above_bottom[0, :])
                slope, y_intercept = line_equation(vertices_arcsec[1], vertices_arcsec[0])

                if n_gridlines_between_points[0] > 0:
                    for i in range(1, n_gridlines_between_points[0]+1):
                        x = (n_grid_lines_above_bottom[np.argmin(n_grid_lines_above_bottom[:, 0]), 0]
                             + i) * self.pix_scale_arcsec
                        y = slope*x + y_intercept
                        extra_vertices.append([x, y])

                if n_gridlines_between_points[1] > 0:
                    for i in range(1, n_gridlines_between_points[1]+1):
                        y = (n_grid_lines_above_bottom[np.argmin(n_grid_lines_above_bottom[:, 1]), 1]
                             + i) * self.pix_scale_arcsec
                        x = (y-y_intercept)/slope
                        extra_vertices.append([x, y])
                return extra_vertices


        def find_vertices_triangle(self, main_vertices_arcsec):
            # #Find points where lines between main vertices hit the grid lines
            line_inds = list(itertools.combinations(range(3), 2))  # #[(0, 1), (0, 2), (1, 2)] each line
            all_vertices = main_vertices_arcsec.tolist()
            for line_ind in line_inds:
                # # restack vertices shape is (point, coordinate)
                vertices_arcsec = np.vstack((main_vertices_arcsec[line_ind[0]], main_vertices_arcsec[line_ind[1]]))
                all_vertices.append(self.find_intersections_of_grid_and_line(vertices_arcsec))

            # #Find relevant corners

            allowable_source_region = np.vstack((np.floor((np.min(main_vertices_arcsec, axis=0)-self.minsrcgrid) /
                                                          self.pix_scale_arcsec),
                                                 np.ceil((np.max(main_vertices_arcsec, axis=0)-self.minsrcgrid) /
                                                         self.pix_scale_arcsec)))   # # shape min/max, coord
            # # Looping over the source grid
            for src_ind_x in range(allowable_source_region[0, 0], allowable_source_region[1, 0]):
                for src_ind_y in range(allowable_source_region[0, 1], allowable_source_region[1, 1]):

                    dist = np.sqrt((all_vertices[1, :]-self.y_arcsec[src_ind_x, src_ind_y])**2 +
                                   (all_vertices[0, :]-self.x_arcsec[src_ind_x, src_ind_y])**2)
                    IsPointInTri() # check corners






class DeflectionMap(PositionGrid):
    """
    MasterClass for deflections
    """
    def __init__(self, xdeflect_fits, ydeflect_fits, center=True, z_lens=0.1):
        PositionGrid.__init__(self, xdeflect_fits, center=center)
        self.xdeflect = pf.getdata(xdeflect_fits)
        self.ydeflect = pf.getdata(ydeflect_fits)
        self.z_lens = z_lens
        print 'Deflection map init'


    def get_deflection_at_image_position(self, coordinate):
        """
        :param coordinate: Image coordinate, depending on whether the map was centered at initialisation
        will determine if coordinate needs to be centered. Units are in arcsec
        :return: value of deflection map at position in image plane
        """
        pos_ind = self.position_index(coordinate)
        return np.hstack((self.xdeflect[pos_ind], self.ydeflect[pos_ind]))

    def get_deflection_at_image_positions(self, coordinates):
        """
        :param coordinates: array of coordinates in arcsec
        :return: deflections in arcsec at coordinate positions
        """
        deflections = np.zeros(coordinates.shape)

        for ind in range(coordinates.shape[1]):
            deflections[:, ind] = self.get_deflection_at_image_position(coordinates[:, ind])
        return deflections

    def calc_source_positions(self, coordinates, z_src, ra_convention=-1, dec_convention=1):
        """Lens equation. Calculate source positions for many coordinates. coordinates have to be in array format"""
        source_coordinates = np.zeros(coordinates.shape)
        deflections = self.get_deflection_at_image_positions(coordinates) * lens_efficiency(z_lens=self.z_lens,
                                                                                            z_src=z_src)
        source_coordinates[0, :] = coordinates[0, :] - ra_convention * deflections[0, :]
        source_coordinates[1, :] = coordinates[1, :] - dec_convention * deflections[1, :]
        return source_coordinates

    def recenter_im_coord(self, coord_deg, coord_err_deg):
        coord_image_arcsec = np.random.normal(loc=coord_deg, scale=coord_err_deg) * 3600. - self.center
        test_in_map = (np.abs(coord_image_arcsec) > self.extent).sum()
        if test_in_map:
            print 'IMAGE OUTSIDE OF MAP'
        return coord_image_arcsec.reshape((2, 1))

    def regrid(self, nfits):
        """
        Useful function to compare maps on different grids
        Have to change the RA convention to RA=-RA for this because RectBivariatSpline needs strictly ascending arrays
        But this doesn't affect anything else in the class"""
        x, y = oned_coordinate_grid(self.original_fits)
        interpolatefunc_xdeflect = RectBivariateSpline(x, y, self.xdeflect)
        interpolatefunc_ydeflect = RectBivariateSpline(x, y, self.ydeflect)

        ny, nx = setup_coordinate_grid(nfits, inverse_ra_convention=True)
        interpxdeflect = interpolatefunc_xdeflect.ev(nx.flatten(), ny.flatten()).reshape(nx.shape, order='F')
        interpydeflect = interpolatefunc_ydeflect.ev(nx.flatten(), ny.flatten()).reshape(nx.shape, order='F')
        return interpxdeflect, interpydeflect

    def calc_image(self, source_fits, z_src, write_image=True, imagename='image.npy'):
        """
        Lens equation at each point in image map
        """
        lens_eff = lens_efficiency(self.z_lens, z_src)
        xmap = self.x_arcsec + self.xdeflect*lens_eff
        ymap = self.y_arcsec - self.ydeflect*lens_eff
        x, y = oned_coordinate_grid(source_fits, translate=True)
        source_map_interp = RectBivariateSpline(x, y, pf.getdata(source_fits))
        image = source_map_interp.ev(xmap.flatten()[::-1], ymap.flatten()).reshape(self.x_arcsec.shape, order='F')
        if write_image:
            np.save(imagename, image)
        return image

    def calc_magnification(self, sourcefits, z_src, write_image=False, imagename='test',
                           zoom_factor=1, cutout_margin_factor=2):
        # #Source fits
        srcsum = pf.getdata(sourcefits).sum()
        srchdr = pf.getheader(sourcefits)
        srcpixelscale_arcsec = srchdr['CDELT2'] * 3600
        srcextent_arcsec = srcpixelscale_arcsec * srchdr['NAXIS1']
        npix_src = srcextent_arcsec/self.pix_scale_arcsec

        # # Interpolation is necessary to smooth pixelated features
        image = self.calc_image(sourcefits, z_src, write_image=write_image, imagename=imagename)
        cutout_margin = np.int(npix_src * cutout_margin_factor)

        ind1, ind2 = np.concatenate(np.where(image == image.max()))
        if (ind1-cutout_margin) <= 0 or (ind2-cutout_margin) <= 0 or \
           (ind1+cutout_margin) >= image.shape[0] or (ind2+cutout_margin) >= image.shape[0]:
            cutout_margin_factor = 1
            cutout_margin = np.int(npix_src * cutout_margin_factor)
        imzoomed = zoom(image[ind1-cutout_margin:ind1+cutout_margin,
                        ind2-cutout_margin:ind2+cutout_margin], zoom_factor) / zoom_factor**2
        image = imzoomed
        return image.sum()/srcsum * (self.pix_scale_arcsec / srcpixelscale_arcsec)**2

    def map_image_coord_to_src(self, z_src):
        lens_eff = lens_efficiency(self.z_lens, z_src)
        xymap = np.zeros((2,)+self.x_arcsec.shape)  # shape is (coordinate, axis_ind1, axis_ind2)
        xymap[0] = self.x_arcsec + self.xdeflect*lens_eff
        xymap[1] = self.y_arcsec - self.ydeflect*lens_eff
        return xymap

    def calc_lines(self, xy_ray_traced):
        #  # Lines shape (horizontal/vertical/diagonal, m/c, x , y)
        lines = np.zeros((3, 2, self.npix, self.npix))
        # #slopes
        lines[0, 0, :-1, :] = (xy_ray_traced[1, 1:, :] - xy_ray_traced[1, :-1, :])/(xy_ray_traced[0, 1:, :] - xy_ray_traced[0, :-1, :])
        lines[1, 0, :, :-1] = (xy_ray_traced[1, :, 1:] - xy_ray_traced[1, :, :-1])/(xy_ray_traced[0, :, 1:] - xy_ray_traced[0, :, :-1])
        lines[2, 0, :-1, :-1] = (xy_ray_traced[1, :-1, 1:] - xy_ray_traced[1, 1:, :-1])/(xy_ray_traced[0, :-1, 1:] - xy_ray_traced[0, 1:, :-1])
        # # intercepts
        lines[0, 1, :-1, :] = xy_ray_traced[1, 1:, :] - xy_ray_traced[0, 1:, :] * lines[0, 0, :, :]
        lines[1, 1, :, :-1] = xy_ray_traced[1, :, 1:] - xy_ray_traced[0, :, 1:] * lines[1, 0, :, :]
        lines[2, 1, :-1, :-1] = xy_ray_traced[2, :-1, 1:] - xy_ray_traced[0, :-1, 1:]*lines[2, 0, :, :]
        return lines

    def calc_n_gridlines_between_points(self, xy_ray_traced):
        n_grid_lines_above_bottom = np.floor(xy_ray_traced/self.pix_scale_arcsec)
        #3 lines x 2 directions x N x N
        n_gridlines_between_points = np.zeros((3, 2, self.npix, self.npix))
        n_gridlines_between_points[0, 0, :, :-1] = np.abs(n_grid_lines_above_bottom[1, 1:, :] - n_grid_lines_above_bottom[0, :])





    def weight_matrix(self, source_fits, z_src):
        xy_ray_traced = self.map_image_coord_to_src(z_src=z_src)
        srcgrid = SrcGrid(source_fits)
        weights = np.zeros((srcgrid.npix, srcgrid.npix, self.npix-1, self.npix-1))
        lines = self.calc_lines(xy_ray_traced)  #   ##Lines shape (horizontal/vertical/diagonal, m/c, x , y)

        xy_ray_traced -= srcgrid.minsrcgrid.reshape(2,1,1)

        # # Loop over image pixels
        for x_ind in range((self.npix-1)):
            for y_ind in range((self.npix-1)):
                for triangle in range(2):
                    if triangle == 0:
                        # #lower triangle, shape (point, coordinate) i.e. (3,2)
                        main_vertices_arcsec = np.array([xy_ray_traced[:, x_ind, y_ind], xy_ray_traced[:, x_ind, y_ind+1],
                                                         xy_ray_traced[:, x_ind+1, y_ind]])
                    else:
                        # #upper triangle
                        main_vertices_arcsec = np.array([xy_ray_traced[:, x_ind+1, y_ind+1],
                                                         xy_ray_traced[:, x_ind, y_ind+1],
                                                         xy_ray_traced[:, x_ind+1, y_ind]])

                line_inds = list(itertools.combinations(range(3), 2))  # #[(0, 1), (0, 2), (1, 2)] each line
                extra_vertices_x = []; extra_vertices_y = []

                # #Works out the extra vertices (except for corners)
                for line_ind in line_inds:
                    # # restack vertices shape is (point, coordinate)
                    vertices_arcsec = np.vstack((main_vertices_arcsec[line_ind[0]], main_vertices_arcsec[line_ind[1]]))
                    # #shift vertices with respect to bottom of src grid
                    vertices_arcsec -= SrcGrid.minsrcgrid
                    n_grid_lines_above_bottom = np.floor(vertices_arcsec/srcgrid.pix_scale_arcsec)
                    n_vertices_between_points = np.abs(n_grid_lines_above_bottom[1, :] - n_grid_lines_above_bottom[0, :])
                    slope, y_intercept = line_equation(vertices_arcsec[1], vertices_arcsec[0])

                    if n_vertices_between_points[0] > 0:
                        for i in range(1, n_vertices_between_points[0]+1):
                            x = (n_grid_lines_above_bottom[np.argmin(n_grid_lines_above_bottom[:, 0]), 0]
                                 + i) * srcgrid.pix_scale_arcsec
                            y = slope*x + y_intercept
                            extra_vertices_x.append(x); extra_vertices_y.append(y)

                    if n_vertices_between_points[1] > 0:
                        for i in range(1, n_vertices_between_points[1]+1):
                            y = (n_grid_lines_above_bottom[np.argmin(n_grid_lines_above_bottom[:, 1]), 1]
                                 + i) * srcgrid.pix_scale_arcsec
                            x = (y-y_intercept)/slope
                            extra_vertices_x.append(x); extra_vertices_y.append(y)

                nex = len(extra_vertices_x)
                all_vertices = np.zeros((3+nex, 2))
                all_vertices[:3, :] = main_vertices_arcsec
                all_vertices[3:, :] = extra_vertices_x, extra_vertices_x

                allowable_source_region = np.vstack((np.floor((np.min(main_vertices_arcsec, axis=0)-SrcGrid.minsrcgrid) /
                                                              srcgrid.pix_scale_arcsec),
                                                     np.ceil((np.max(main_vertices_arcsec, axis=0)-SrcGrid.minsrcgrid) /
                                                             srcgrid.pix_scale_arcsec)))   # # shape min/max, coord
                # # Looping over the source grid
            for src_ind_x in range(allowable_source_region[0, 0], allowable_source_region[1, 0]):
                for src_ind_y in range(allowable_source_region[0,1], allowable_source_region[1, 1]):

                    dist = np.sqrt((all_vertices[1, :]-srcgrid.y_arcsec[src_ind_x, src_ind_y])**2 +
                                   (all_vertices[0, :]-srcgrid.x_arcsec[src_ind_x, src_ind_y])**2)
                    IsPointInTri() # check corners

                    relevant_vertices = all_vertices[dist <= srcgrid.pix_scale_arcsec]
                    weights[src_ind_x, src_ind_y, x_ind, y_ind] = area_polygon(relevant_vertices[0, :],
                                                                               relevant_vertices[1, :])

        return weights









