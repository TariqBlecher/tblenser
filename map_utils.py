import numpy as np
import pyfits as pf
from scipy.interpolate import RectBivariateSpline
from astropy.cosmology import Planck15 as cosmo
from scipy.ndimage import zoom
import itertools
from shapely.geometry import box, Polygon
from rtree import index
import pandas as pd
import pickle



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
    def __init__(self, source_fits, center=True):
        PositionGrid.__init__(self, source_fits, center=center)
        self.flat_x = self.x_arcsec.flatten()
        self.flat_y = self.y_arcsec.flatten()
        self.gridcells = np.zeros(self.flat_x.shape[0], dtype=object)
        self.idx = index.Index()
        for i in range(self.flat_x.shape[0]):
            cellbounds = (self.flat_x[i]-self.pix_scale_arcsec/2., self.flat_y[i]-self.pix_scale_arcsec/2.,
                          self.flat_x[i]+self.pix_scale_arcsec/2., self.flat_y[i]+self.pix_scale_arcsec/2.)
            self.gridcells[i] = box(*cellbounds)
            self.idx.insert(i, cellbounds)


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

    def weight_matrix(self, source_fits, z_src):
        xy_ray_traced = self.map_image_coord_to_src(z_src=z_src)
        srcgrid = SrcGrid(source_fits)

        areas = []
        src_inds = []
        # # Loop over image pixels
        for x_ind in range((self.npix-1)):
            for y_ind in range((self.npix-1)):
                for triangle in range(2):
                    if triangle == 0:
                        # #lower triangle, shape (point, coordinate) i.e. (3,2)
                        main_vertices_arcsec = np.array([xy_ray_traced[:, x_ind, y_ind],
                                                         xy_ray_traced[:, x_ind, y_ind+1],
                                                         xy_ray_traced[:, x_ind+1, y_ind]])
                    else:
                        # #upper triangle
                        main_vertices_arcsec = np.array([xy_ray_traced[:, x_ind+1, y_ind+1],
                                                         xy_ray_traced[:, x_ind, y_ind+1],
                                                         xy_ray_traced[:, x_ind+1, y_ind]])

                    polygon_shape = Polygon(main_vertices_arcsec)
                    indices_of_srcpix_intersection = np.array([src_pix_ind for src_pix_ind
                                                               in srcgrid.idx.intersection(polygon_shape.bounds)])
                    src_inds.append(indices_of_srcpix_intersection)
                    area_temp = []
                    for src_pix_ind in indices_of_srcpix_intersection:
                        area_temp.append(polygon_shape.intersection(srcgrid.gridcells[src_pix_ind]).area)
                    area_temp = np.array(area_temp)
                    if area_temp.sum() == 0.:
                        areas.append(np.zeros(len(indices_of_srcpix_intersection)))
                    else:
                        areas.append(area_temp/(2*area_temp.sum()))
        with open('areas.p', 'wb') as fp:
            pickle.dump(areas, fp)
        with open('srcind.p', 'wb') as fp:
            pickle.dump(src_inds, fp)
        return areas, src_inds

    def calc_image2(self, source_fits, z_src, write_image=True, imagename='image.npy'):
        areas, src_inds = self.weight_matrix(source_fits, z_src)
        srcfits = pf.getdata(source_fits).flatten()
        image = np.zeros((self.npix-1, self.npix-1))
        i = 0
        for x_ind, x in enumerate(np.arange(self.npix-1)):
            for y_ind, y in enumerate(np.arange(self.npix-1)):
                if len(list(src_inds[i])) == 0:
                    lower_triangle = 0
                else:
                    lower_triangle = np.sum(srcfits[src_inds[i]]*areas[i])
                if len(list(src_inds[i+1])) == 0:
                    upper_triangle = 0
                else:
                    upper_triangle = np.sum(srcfits[src_inds[i+1]]*areas[i+1])
                image[x_ind, y_ind] = lower_triangle+upper_triangle
                i += 2
        if write_image:
            np.save(imagename, image)
        return image







