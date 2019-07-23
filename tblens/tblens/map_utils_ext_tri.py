from map_utils_core import *
from shapely.geometry import box, Polygon
from rtree import index
import numpy as np
from astropy.io import fits


class SrcGrid(PositionGrid):
    def __init__(self, source_fits, center=True):
        PositionGrid.__init__(self, source_fits, center=center)
        self.flat_x = self.x_deg.flatten()
        self.flat_y = self.y_deg.flatten()
        self.gridcells = np.zeros(self.flat_x.shape[0], dtype=object)
        self.idx = index.Index()

        self.srchdr = fits.getheader(source_fits)
        self.srcpixelscale_arcsec = self.srchdr['CDELT2'] * 3600
        self.srcextent_arcsec = self.srcpixelscale_arcsec * self.srchdr['NAXIS1']

        for i in range(self.flat_x.shape[0]):
            cellbounds = (self.flat_x[i] - self.pix_scale_deg / 2., self.flat_y[i] - self.pix_scale_deg / 2.,
                          self.flat_x[i] + self.pix_scale_deg / 2., self.flat_y[i] + self.pix_scale_deg / 2.)
            self.gridcells[i] = box(*cellbounds)
            self.idx.insert(i, cellbounds)


class DeflectionMapExt(DeflectionMap):
    """
    MasterClass for deflections
    TAKES IN NORMALISED DEFLECTION MAPS
    """
    def __init__(self, xdeflect_fits, ydeflect_fits, center=True, z_lens=0.1):
        DeflectionMap.__init__(self, xdeflect_fits, ydeflect_fits, center=center, z_lens=z_lens)
        print 'Deflection map extension init'

    def map_image_coord_to_src(self, z_src):
        lens_eff = lens_efficiency(self.z_lens, z_src)
        xymap = np.zeros((2,) + self.x_deg.shape)  # shape is (coordinate, axis_ind1, axis_ind2)
        xymap[0] = self.x_deg + self.xdeflect * lens_eff
        xymap[1] = self.y_deg - self.ydeflect * lens_eff
        return xymap

    def weight_table(self, source_fits='', z_src=None, suffix=''):
        # # Just have to test and check if the factor extend guess is fine
        xy_ray_traced = self.map_image_coord_to_src(z_src=z_src)
        srcgrid = SrcGrid(source_fits)
        factorextend = 10  # #Guess
        areas, src_pix_inds, im_pix_inds = np.zeros((3, factorextend*self.npix**2))
        table_ind = 0
        im_pix_ind = 0

        for y_ind in range((self.npix-1)):
            for x_ind in range((self.npix-1)):
                for triangle in range(2):
                    if triangle == 0:
                        # #lower triangle, shape (point, coordinate) i.e. (3,2)
                        main_vertices_arcsec = np.array([xy_ray_traced[:, y_ind, x_ind],
                                                         xy_ray_traced[:, y_ind, x_ind+1],
                                                         xy_ray_traced[:, y_ind+1, x_ind]])
                    else:
                        # #upper triangle
                        main_vertices_arcsec = np.array([xy_ray_traced[:, y_ind+1, x_ind+1],
                                                         xy_ray_traced[:, y_ind, x_ind+1],
                                                         xy_ray_traced[:, y_ind+1, x_ind]])

                    polygon_shape = Polygon(main_vertices_arcsec)
                    indices_of_srcpix_intersection = np.array([src_pix_ind for src_pix_ind
                                                               in srcgrid.idx.intersection(polygon_shape.bounds)])

                    for i, src_pix_ind in enumerate(indices_of_srcpix_intersection):
                        if polygon_shape.intersection(srcgrid.gridcells[src_pix_ind]).geom_type == 'Polygon':
                            areas[table_ind] = polygon_shape.intersection(srcgrid.gridcells[src_pix_ind]).area
                            src_pix_inds[table_ind] = src_pix_ind
                            im_pix_inds[table_ind] = im_pix_ind
                            table_ind += 1
                boolim = im_pix_inds == im_pix_ind
                weightnorm = np.sum(areas[boolim])
                if weightnorm != 0:
                    areas[boolim] *= 1/weightnorm

                im_pix_ind += 1
        np.save('weightmatrix' + suffix, np.vstack((im_pix_inds, src_pix_inds, areas)))

    def calc_image_triangle(self, source_fits='', write_image=True, weight_suffix='', image_suffix=''):
        im_pix_inds, src_pix_inds, areas = np.load('weightmatrix' + weight_suffix + '.npy')
        im_pix_inds = im_pix_inds.astype('int')
        src_pix_inds = src_pix_inds.astype('int')
        srcfits = fits.getdata(source_fits).flatten()
        image = np.zeros((self.npix-1, self.npix-1)).flatten()
        for ind, im_pix in enumerate(im_pix_inds):
            image[im_pix] += srcfits[src_pix_inds[ind]] * areas[ind]
        image = image.reshape((self.npix-1, self.npix-1))
        if write_image:
            np.save('image' + image_suffix, image)
        return image

    def calc_mag_triangle(self, source_fits='', write_image=True, image_suffix='', z_src=None):
        srcpixelscale_arcsec = fits.getheader(source_fits)['CDELT2'] * 3600
        image = self.calc_image_triangle(source_fits=source_fits, write_image=write_image, image_suffix=image_suffix)
        return image.sum()/fits.getdata(source_fits).sum() * (self.pix_scale_deg / srcpixelscale_arcsec) ** 2









