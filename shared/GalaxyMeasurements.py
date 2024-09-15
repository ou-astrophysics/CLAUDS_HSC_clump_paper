import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import geopandas as gpd
from shapely import geometry
from shapely.ops import unary_union
from PIL import Image, ImageDraw

from astropy.io import fits
from astropy import coordinates as coords
from astropy.wcs import WCS
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats, SigmaClip
from astropy.convolution import convolve, convolve_fft, Box2DKernel, Gaussian2DKernel, Tophat2DKernel
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

import photutils
from photutils.background import Background2D, MedianBackground
from photutils.utils import circular_footprint
from photutils.segmentation import make_2dgaussian_kernel, detect_sources, deblend_sources, SourceCatalog
from photutils.utils.exceptions import NoDetectionsWarning

from skimage.filters.rank import majority
from skimage.morphology import square


def merge_clumps(coords, fwhm, close_by_threshold):
    """
    If two clumps are extremely close, merges them into one 
    and averages the position of two clumps.
    
    Args:
      coords: list of [x, y]
      fwhm: radius
      close_by_threshold: multiplicator for radius
      
    Returns:
      Returns array with new [x, y]
    """
    
    pts = [geometry.Point((p[0], p[1])) for p in coords]
    unioned_buffered_poly = unary_union([p.buffer(fwhm * close_by_threshold) for p in pts])
    
    if unioned_buffered_poly.geom_type == 'MultiPolygon':
        return np.array([[u.centroid.x, u.centroid.y] for u in unioned_buffered_poly])
    elif unioned_buffered_poly.geom_type == 'Polygon':
        return np.array([[unioned_buffered_poly.centroid.x, unioned_buffered_poly.centroid.y]])


def merge_clumps_df(df, coordx, coordy, fwhm, close_by_threshold, first_cols, max_cols):
    """
    If two clumps are extremely close, merges them into one 
    and averages the position of two clumps.
    If one clump is "odd" and the other "normal", assigns the clumps the class "odd"
    
    Creates a geopandas dataframe from the input-df, creates buffers around the centroids,
    dissolves the geometries and aggregates by geometry.
    
    Args:
      df: dataframe as input
      coordx, coordy: name of the cols in df to be used as x,y coordinates
      fwhm: radius
      close_by_threshold: multiplicator for radius
      cols: columns of the dataframe df to return
      
    Returns:
      Returns df with new [x, y], aggregated columns
    """
    
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[coordx], df[coordy]).buffer(0.5 * df[fwhm].iloc[0] * close_by_threshold)
    )

    agg_col_geo = {'geometry': 'first'}
    agg_first_cols = dict.fromkeys(first_cols, 'first')
    agg_max_cols = dict.fromkeys(max_cols, 'max')

    agg_col = agg_col_geo | agg_first_cols | agg_max_cols

    intersects = (
        gdf[['geometry']]
        .dissolve()
        .explode(index_parts=True)
        .sjoin(gdf, how='inner', predicate='intersects')
        .reset_index()
        .groupby('level_1')
        .agg(agg_col)
        .set_geometry('geometry')
        .assign(px_x = lambda df_: df_.centroid.x)
        .assign(px_y = lambda df_: df_.centroid.y)
        .reset_index()
    )
    
    return intersects # [['px_x', 'px_y', label, fwhm]]


def create_circular_mask(h, w, center=None, radius=None, invert=False):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    if invert:
        mask = dist_from_center > radius
    else:
        mask = dist_from_center <= radius
    return mask


def create_galaxy_mask(science_image, petroR90_asec, pixel_scale):
    """
    Takes a FITS-image and masks the galaxy.
    The mask can then be applied to the clump centroids.
    
    Args:
      science_image: image-data as array from FITS
      petroR90_asec: 90% Petrosian radius in arcsec
      pixel_scale: conversion factor arcec/pixel
      
    Returns:
      Returns mask-array, segment_map, and other intermediate steps for plotting
    """

    petroR90 = petroR90_asec / pixel_scale
    tophat_2D_kernel = Tophat2DKernel(5)
    _, _, std = sigma_clipped_stats(science_image)
    convolved_data = convolve_fft(science_image, tophat_2D_kernel)
    
    # Step 1: "hot" mode
    threshold = np.percentile(science_image.flatten(), 97.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=NoDetectionsWarning)
        segm_detect_step1 = detect_sources(convolved_data, threshold, npixels=1)
    # keep all segments outside the central segment
    if segm_detect_step1 is None:
        hot_mask = np.zeros_like(science_image).astype(bool)
    else:
        segm_detect_step1_orig = segm_detect_step1.copy()
        # segm_detect_step1_plot = segm_detect_step1.copy()
        circular_mask = create_circular_mask(h=science_image.shape[0], w=science_image.shape[1], radius=petroR90, invert=False)
        segm_detect_step1_orig.remove_masked_labels(mask=circular_mask, partial_overlap=False)
        hot_mask = segm_detect_step1_orig.make_source_mask()
    
    # Step 2: "cool" mode
    threshold = 1. * std
    npixels = int((1 / pixel_scale)**2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=NoDetectionsWarning)
        segm_detect_step2 = detect_sources(convolved_data, threshold, npixels=npixels)
        segm_deblend_step2 = deblend_sources(convolved_data, segm_detect_step2, npixels=npixels, nlevels=32, contrast=0.000001, progress_bar=False)
    
    # Step 3. Mask contaminants
    # keep all segments outside the central segment
    segm_deblend_step2_orig = segm_deblend_step2.copy()
    segm_deblend_step2_plot = segm_deblend_step2.copy()
    border = int(segm_deblend_step2.shape[0]/2 - 1)

    if border < min(segm_deblend_step2.shape) / 2:
        segm_deblend_step2.remove_border_labels(border_width=border, partial_overlap=False)
        label_central_galaxy = segm_deblend_step2.labels
        segm_deblend_step2_orig.remove_label(label_central_galaxy)
    
    circular_mask = create_circular_mask(h=science_image.shape[0], w=science_image.shape[1], radius=2*petroR90, invert=True)
    
    segm_deblend_step2_mask = np.logical_or(circular_mask, hot_mask)
    segm_deblend_step2_orig.remove_masked_labels(mask=~segm_deblend_step2_mask, partial_overlap=False)
    
    final_mask = segm_deblend_step2_orig.make_source_mask()
    
    # Step 4: "cold" mode
    threshold = 1. * std
    npixels = int(0.01 * (petroR90 / pixel_scale)**2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=NoDetectionsWarning)
        segm_detect_step4 = detect_sources(convolved_data, threshold, npixels=npixels, mask=final_mask)

    if segm_detect_step4 is None:
        final_seg_map = np.zeros_like(science_image).astype(bool)
    else:
        # keep only most central segmentation mask, presumably the main galaxy
        border = int(segm_detect_step4.shape[0]/2 - 1)
        if border < min(segm_detect_step4.shape) / 2:
            segm_detect_step4.remove_border_labels(border_width=border, partial_overlap=False, relabel=True)
        # smooth image
        kernel_size = int(0.1 * science_image.shape[0])
        final_seg_map = majority(segm_detect_step4.data.astype('uint8'), square(kernel_size))
        segm_detect_step4.data = final_seg_map

    # return final_seg_map, segm_detect_step4, segm_deblend_step2_plot, segm_detect_step1_plot, segm_deblend_step2_mask, final_mask
    return final_seg_map, segm_detect_step4, segm_deblend_step2_plot, segm_deblend_step2_mask, final_mask


def galaxy_mask(data, bkg_data, box_size=50, filter_size=3, kernel_size=3, detection_threshold=1.5, npixels=10, debug=False):
    """
    Takes a FITS-image and masks the galaxy.
    The mask can then be applied to the clump centroids.
    
    Args:
      data: image-data as array from FITS
      box_size: size in px of the box for background substraction
      filter_size: size in px for the background filter
      kernel_size: size in px of the kernel for convolve, should be FWHM
      detection_threshold: multiplier for a detection threshold image using the background RMS image (sigma per px)
      npixels: npixels settings for detect_sources
      
    Returns:
      Returns mask-array, segment_map, SourceCatalogue, Background2D
    """

    bkg_estimator = MedianBackground()

    bkg = Background2D(
        bkg_data, 
        (box_size, box_size), 
        filter_size=(filter_size, filter_size), 
        bkg_estimator=bkg_estimator
    )
    data -= bkg.background

    threshold = detection_threshold * bkg.background_rms

    # kernel = make_2dgaussian_kernel(kernel_size, size=5)
    kernel = Gaussian2DKernel(kernel_size)
    convolved_data = convolve_fft(data, kernel, boundary='wrap')

    segment_map = detect_sources(convolved_data, threshold, npixels)

    # keep only most central segmentation mask, presumably the main galaxy
    border = int(segment_map.shape[0]/2 - 1)
    if border < min(segment_map.shape) / 2:
        segment_map.remove_border_labels(border_width=border, partial_overlap=False, relabel=True)

    # plot segmentation map
    if debug:
        print(segment_map)
        norm = ImageNormalize(stretch=SqrtStretch())
        plt.imshow(data, cmap='Greys_r', norm=norm)
        segment_map.plot_patches(edgecolor='lime', lw=1, ls=':')
    
    # create mask (array)
    footprint = circular_footprint(radius=np.floor(kernel_size))
    mask = segment_map.make_source_mask(footprint=footprint)

    # create catalogue
    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)

    return mask, segment_map, cat, bkg


def clump_mask(px_x, px_y, radius, img_size, debug=False):
    """
    Takes centroid coordinates and FWHM-radius of clumps in px
    and creates a mask for background substraction.
    
    Args:
      px_x, px_y: clump centroids coords in px
      radius: radius of the circular mask to be applied, e.g. FWHM
      img_size: shape of the FITS-image
      
    Returns:
      Returns mask-array
    """

    mask = np.zeros(img_size).astype(bool)
    y0, x0 = np.indices(img_size)

    for x, y in zip(px_x, px_y):
        distance = np.sqrt((x-x0)**2 + (y-y0)**2)
        mask[distance <= radius] = True

    # plot segmentation map
    if debug:
        norm = ImageNormalize(stretch=SqrtStretch())
        plt.imshow(mask, cmap='Greys_r', norm=norm)
    
    return mask


def clump_mask_variable_radius(px_x, px_y, radius, img_size, debug=False):
    """
    Takes centroid coordinates and FWHM-radius of clumps in px
    and creates a mask for background substraction.
    
    Args:
      px_x, px_y: clump centroids coords in px
      radius: corresponding radius
      img_size: shape of the FITS-image
      
    Returns:
      Returns mask-array
    """

    mask = np.zeros(img_size).astype(bool)
    y0, x0 = np.indices(img_size)

    for x, y, r in zip(px_x, px_y, radius):
        distance = np.sqrt((x-x0)**2 + (y-y0)**2)
        mask[distance <= r] = True

    # plot segmentation map
    if debug:
        norm = ImageNormalize(stretch=SqrtStretch())
        plt.imshow(mask, cmap='Greys_r', norm=norm)
    
    return mask


def mask_image_with_bboxes(px_x1, px_x2, px_y1, px_y2, img_size, invert=True, list_of_boxes=True, debug=False):
    """
    Takes bbox coordinates and creates an image mask
    with the dimension of the original image.
    
    Args:
      px_x1, px_x2, px_y1, px_y2: bbox coords in px
      img_size: shape of the FITS-image
      invert: True for for box masks to contain a True value
      list_of_boxes: switch if a list of bbox coords are used
      debug: True if you want to see an image output of the mask
      
    Returns:
      Returns mask-array
    """

    img = Image.new('L', (img_size[1], img_size[0]), 0)
    if list_of_boxes:
        for x1, x2, y1, y2 in zip(px_x1, px_x2, px_y1, px_y2):
            ImageDraw.Draw(img).rectangle([x1, y1, x2, y2], outline=1, fill=1)
    else:
        ImageDraw.Draw(img).rectangle([px_x1, px_y1, px_x2, px_y2], outline=1, fill=1)
    
    if invert:
        mask = np.logical_not(np.array(img).astype(bool))
    else:
        mask = np.array(img).astype(bool)

    # plot segmentation map
    if debug:
        norm = ImageNormalize(stretch=SqrtStretch())
        plt.imshow(mask, cmap='Greys_r', norm=norm)
    
    return mask


def select_segments_by_mask(segement_map, mask):
    """
    Takes a segment map and corresponding catalogue and matches the segments with the FRCNN-predictions.
    Only those segments which partly or fully fall within a bbox of the predictions are selected

    Returns:
      segement_map_keep : segment map containing only labels which match requirements
      masks : numpy array of the masks
    """

    # keep central segment for galaxy
    border = int(segement_map.shape[0]/2 - 1)
    if border < min(segement_map.shape) / 2:
        segement_map.remove_border_labels(border_width=border, partial_overlap=True)

    # copy original segment_map so we can exclude the remaining segments after removing the masked segments
    segement_map_mask = segement_map.copy()

    # remove masked segments
    segement_map.remove_masked_labels(mask, partial_overlap=True, relabel=False)
    _labels_to_remove = segement_map.labels
    
    # remove the remaining segements from the original segment_map
    if _labels_to_remove.shape[0] > 0:
        segement_map_mask.remove_labels(labels=_labels_to_remove)
    masks = segement_map_mask.data.astype(bool)

    return segement_map_mask, masks


def get_ellipse_parameters(Mxx, Mxy, Myy, radius=1):
    """
    Takes the 2nd adaptive moments of an ellipse (e.g. SDSS or HSC)
    and returns the simple ellipse parameters for plotting.
    
    Args:
      Mxx, Mxy, Myy: 2nd order adaptive moments (arcsec^2)
      radius: radius for scaling (arcsec)
      
    Returns:
      Returns:
      a: semi-major axis
      b: semi-minor axis
      theta: angle of the semi-major axis with the positive x-axis
    """
    xx = Mxx * radius**2
    xy = Mxy * radius**2
    yy = Myy * radius**2

    Q = np.array([
        [xx, xy],
        [xy, yy]
    ])
    eig_val, eig_vec = np.linalg.eig(Q)
    
    a = np.sqrt(np.max(eig_val))  # major axis
    b = np.sqrt(np.min(eig_val))  # minor axis
    a_indx = np.argmax(eig_val)
    cos_t, sin_t = eig_vec[0, a_indx], -eig_vec[1, a_indx]
    theta = np.rad2deg(np.arctan2(sin_t, cos_t))  # position angle
    r_eff = np.sqrt(a * b)  # effective radius
    r_trace = np.sqrt((a*a + b*b)/2)  # trace radius
    q = b / a  # axial ratio
    r_det = (xx * yy - xy**2)**0.25
    
    return a, b, theta


def gauss2d(x, y, sig, x0=0, y0=0):
    return 0.5/sig**2/np.pi * np.exp(-0.5*((x-x0)**2+(y-y0)**2)/sig**2)


def fwhm_to_sigma(fwhm):
    return fwhm / (2 * np.sqrt(2*np.log(2.)))


def Moffat2D_xy(gamma, alpha, normed=True):
    if normed:
        norm = 1 / Moffat2D_integral(r=1e10,gamma=gamma,alpha=alpha)
    else:
        norm = 1.0

    return lambda x, y: norm * (1 + ((x**2+y**2)/gamma**2))**(-alpha)


def Moffat2D_integral(r, gamma, alpha):
    return np.pi * (gamma**2 - (gamma**2)**alpha * (gamma**2 + r**2)**(1-alpha)) / (alpha - 1)


def reff_from_gamma(gamma, alpha):
    norm = 1 / Moffat2D_integral(r=1e5, gamma=gamma, alpha=alpha)
    return np.sqrt(gamma**(-2*alpha/(1-alpha)) * (gamma**2 + (1-alpha)/(2*np.pi*norm))**(1/(1-alpha)) - gamma**2)


def fwhm_to_gamma(fwhm, alpha):
    reff_gauss = fwhm / 2
    return scipy.optimize.brentq(lambda g: reff_from_gamma(g, alpha=alpha) - reff_gauss, 0.1, 10)
    # return fwhm / 2 / np.sqrt(2**(1/alpha) - 1)


def get_aperture_correction(psf, aper, func='gauss'):
    dx = dy = 0.01
    x = np.arange(-10, 10, dx)
    y = np.arange(-10, 10, dy)

    yy, xx = np.meshgrid(y, x, sparse=True)

    if func=='gauss':
        g_see = gauss2d(xx, yy, sig=fwhm_to_sigma(psf))

    elif func=='moffat':
        alpha = 2.5
        MoffatFunc = Moffat2D_xy(gamma=fwhm_to_gamma(psf,alpha), alpha=alpha)
        g_see = MoffatFunc(xx,yy)

    else:
        raise Exception("Invalid function type.")

    cond_aper = np.sqrt(xx**2 + yy**2) <= aper
    f_see = np.sum(g_see[cond_aper]) * dx * dy

    aperture_correction = 1/f_see

    return aperture_correction


def calcFluxScale(zp0,zp1):
    """
    -2.5*log(f1) + zp1 = -2.5*log(f0) + zp0
    f1/f0 = 10**((zp1 - zp0) / 2.5)
    """
    fscale = 10**((zp1 - zp0) / 2.5)
    return fscale
