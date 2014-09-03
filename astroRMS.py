"""
Matt Mechtley - Dec 2011 - https://github.com/mmechtley
Python implementation of Mark Dickinson's acall iraf tasks for calculating
pixel-to-pixel RMS noise in astronomical images, taking into account
autocorrelation, noise variations in the weight map, and masking out
the objects themselves.

These functions are not intended to have exact parameter parity with the
medrms iraf package. Rather, they are intended to provide similar functionality
with a Pythonic interface

Example usage:
import astroRMS

sci_file = 'goodss_f160w_sci.fits'
wht_file = scifile.replace('sci', 'wht')
out_file = scifile.replace('sci', 'ivm')
astroRMS.create_error_map(sci_file, wht_file, out_file)
"""
# separate float division: / and floor division: //, See PEP 238
from __future__ import division
import datetime
import numpy as np
import scipy.ndimage as ndimg
from warnings import warn
from inspect import getargspec
from numpy.fft import fftn, ifftn, ifftshift
from astropy.io import fits
from astropy.stats import biweight_location, biweight_midvariance

_log_info = {
    'autocorr_rms': 'RMS calculated from autocorrelation function',
    'measured_rms': 'RMS measured from rms-filtered image',
    'corr_factor': 'RMS supression factor (correlation factor)',
    'weight_nominal': 'Correlation-corrected inverse variance. A weight map '
                      'can be created with this value',
    'weight_avg': 'Average value of supplied weight map',
    'weight_scale': 'Factor to scale supplied weight map by to create inverse '
                    'variance map',
    'frac_flagged': 'Fraction of pixels masked out of the input image when '
                    'calculating RMS (Object pixels, zero-weight pixels, etc.)'
}


def _pixel_stats(pixels, clip_sig=3, n_clip=10, min_pix=50):
    """
    Calculate mode and scale of pixel distribution, clipping outliers. Uses
    biweight as "robust" estimator of these quantities.

    :param pixels: Array to calculate statistics for
    :param clip_sig: Sigma value at which to clip outliers
    :param n_clip: Number of clipping iterations
    :param min_pix: Minimum number of retained pixels
    :return: 2-tuple of distribution mode, scale
    """
    clip_iter = 0
    mode, scale = 0, 1
    mask = np.ones(pixels.shape, dtype=bool)
    while True:
        mode = biweight_location(pixels[mask])
        scale = biweight_midvariance(pixels[mask])
        mask &= np.abs(pixels - mode) < clip_sig * scale
        clip_iter += 1
        if np.sum(mask) < min_pix or clip_iter >= n_clip:
            break
    return mode, scale


def _measure_autocorrelation(image_data, aper_radius=5,
                             bg_annulus_radii=(5, 7)):
    """
    Uses the autocorrelation peak to calculate pixelwise RMS and correlation
    factor. This is an aperture photometry-like algorithm.

    :param image_data: Image to compute autocorrelation of. Should be sky-
        subtracted and have objects and bad pixels set to 0.
    :param aper_radius: Radius of the aperture to use for finding total flux of
        the autocorrelation peak
    :param bg_annulus_radii: Inner and outer radius of sky annulus. NOT inner
        radius and annulus width as in IRAF.
    :return: 2-tuple of RMS and autocorrelation factor
    """
    # Compute 2D autocorrelation function
    fft_data = fftn(image_data)
    autocorr_image = ifftn(fft_data * np.conjugate(fft_data)).real
    autocorr_image = ifftshift(autocorr_image)
    # Get image shape and center coordinates
    im_shape = autocorr_image.shape
    center = np.array(im_shape) / 2 - 0.5

    # generate x,y coordinates of pixels, and square distance to center of each
    x, y = np.meshgrid(range(im_shape[1]), range(im_shape[0]))
    sq_dist = (x - center[1]) ** 2 + (y - center[0]) ** 2

    # 'sky' or background mask is an annulus around the center.
    # The annulus is also expanded by 1 pixel in both directions, with those
    # pixels contributing partial flux (grayscale masking)
    sky_mask = sq_dist > min(bg_annulus_radii) ** 2
    sky_mask &= sq_dist < max(bg_annulus_radii) ** 2
    sky_fix_mask = ~sky_mask
    sky_fix_mask &= sq_dist > (min(bg_annulus_radii) - 1) ** 2
    sky_fix_mask &= sq_dist < (max(bg_annulus_radii) + 1) ** 2

    # How much area is not accounted for in the original mask?
    sky_fix_area = (np.pi * (max(bg_annulus_radii) ** 2
                             - min(bg_annulus_radii) ** 2) - np.sum(sky_mask))
    # What fraction of the 1-pixel expanded ring is actually inside the annulus
    fix_pixels_weight = sky_fix_area / np.sum(sky_fix_mask)
    sky_wts = 1.0 * sky_mask + fix_pixels_weight * sky_fix_mask
    # 'Flux' or measurement mask is a circle around the center
    flux_mask = sq_dist < aper_radius ** 2
    flux_fix_mask = ~flux_mask & (sq_dist < (aper_radius + 1) ** 2)
    flux_fix_area = np.pi * aper_radius ** 2 - np.sum(flux_mask)
    fix_pixels_weight = flux_fix_area / np.sum(flux_fix_mask)
    flux_wts = 1.0 * flux_mask + fix_pixels_weight * flux_fix_mask

    # Calculate RMS and autocorrelation factor based on peak, background, and
    # integrated magnitude of the autocorrelation peak
    peak_val = np.max(autocorr_image[flux_mask])
    bg_val = np.average(autocorr_image, weights=sky_wts)
    total_corr = np.sum((autocorr_image - bg_val) * flux_wts)
    corr_rms = np.sqrt((peak_val - bg_val) / autocorr_image.size)
    corr_fac = np.sqrt(total_corr / (peak_val - bg_val))

    return corr_rms, corr_fac


def _rms_filter(image_data, filt_size=7):
    """
    Runs an 'RMS filter' on image data. Really this is the square root of the
    (appropriately scaled) difference of the squared uniform-filtered image and
    the uniform-filtered squared image. i.e. f*sqrt(boxcar(i**2) - boxcar(i)**2)

    :param image_data: Image data to be filtered
    :param filt_size: Size of uniform filter kernel to use
    :return: RMS-filtered version of input image
    """
    mean_sqr = ndimg.uniform_filter(image_data**2, size=filt_size)
    sqr_mean = ndimg.uniform_filter(image_data, size=filt_size)**2
    sqr_diff_scale = filt_size**2 / (filt_size**2 - 1)
    return np.sqrt(np.abs(sqr_diff_scale * (sqr_mean - mean_sqr)))


def _binned_median_filter(image_data, bin_size=4, filt_size=25):
    """
    Implements a median filter that block-averages the image data before
    applying the filter (to improve speed at the cost of accuracy).

    :param image_data: 2-dimensional numpy array containing image data
    :param bin_size: Binning factor to use before median-filtering.
        image_data.shape must be an integer multiple of bin_size in both
        directions.
    :param filt_size: Size of median filter kernel. Effective kernel size in
        the returned image will be filt_size*bin_size
    :return: Median-filtered version of input image
    """
    img_shape = image_data.shape
    if img_shape[0] % bin_size != 0 or img_shape[1] % bin_size != 0:
        raise ValueError('Size of image_data must be a multiple of bin_size '
                         'in both dimensions')
    if bin_size > 1:
        locs_y = np.arange(0, img_shape[0], bin_size)
        locs_x = np.arange(0, img_shape[1], bin_size)
        binned = np.add.reduceat(np.add.reduceat(image_data, locs_y, axis=0),
                                 locs_x, axis=1) / bin_size**2
    else:
        binned = image_data

    binned = ndimg.median_filter(binned, size=filt_size, mode='nearest')
    return np.repeat(np.repeat(binned, bin_size, axis=0), bin_size, axis=1)


def object_mask(input_data, sky_bin=4, sky_size=15, smooth_size=3,
                thresh_type='sigma', threshold=1.5, num_clips=10, grow_radius=0,
                out_file=None):
    """
    Creates a numpy array representing a mask of the object pixels in an input
    astronomical image. Uses sigma-based feature thresholding. Objects will be
    labeled with 1's, background pixels labeled with 0's.

    :param input_data: Input image data
    :param sky_bin: Binning factor to use before median-filtering to subtract
        the sky
    :param sky_size: Size of median filter kernel for subtracting local sky.
        Effective kernel size will be sky_size*sky_bin. If 0, image is assumed
        to already be sky-subtracted.
    :param smooth_size: Boxcar filter size for smoothing before thresholding
    :param thresh_type: Either 'sigma' or 'constant' for sigma or constant-
        value thresholding
    :param threshold: Threshold value, either in sigma above mean or constant
        value
    :param num_clips: Number of times to apply sigma clip when calculating sky
        statistics
    :param grow_radius: Radius of rings to grow around masked objects

    :return: numpy array with objects labeled with value '1', sky with '0'
    """
    # Raise error if boxcar smoothing parameters are not odd numbers
    if sky_size > 0 and sky_size % 2 != 1:
        raise ValueError('sky_size must be an odd integer, or 0')
    if smooth_size > 0 and smooth_size % 2 != 1:
        raise ValueError('smooth_size must be an odd integer, or 0')
    # Raise error if a bad keyword is given for thresh_type
    if thresh_type not in ('sigma', 'constant'):
        raise ValueError("thresh_type must be one of 'sigma' or 'constant'")

    # Create local copy of the image data in memory (don't muck with original)
    working_mask = input_data.copy()

    # If sky_size is supplied, create a median image of the local sky value,
    # and subtract that off
    if sky_size > 0:
        working_mask -= _binned_median_filter(working_mask, sky_bin, sky_size)

    # Calculate image statistics to determine median sky level and RMS noise
    sky, rms = _pixel_stats(working_mask, clip_sig=3, n_clip=num_clips)

    # Uniform filter before thresholding, if commanded
    if smooth_size > 0:
        working_mask = ndimg.uniform_filter(working_mask, size=smooth_size)

    # Do the thresholding
    if thresh_type == 'sigma':
        threshold *= rms
    working_mask = working_mask > sky + threshold

    # Dilate or Erode objects if commanded
    grow_kern = np.ones((2*grow_radius + 1, 2*grow_radius + 1))
    grow_kern[grow_radius, grow_radius] = 0
    grow_kern = ndimg.distance_transform_edt(grow_kern) <= grow_radius
    # TODO: Use min/max filter in place of binary_dilation/erosion? Time these.
    if grow_radius > 0:
        working_mask = ndimg.binary_dilation(working_mask, structure=grow_kern)
    elif grow_radius < 0:
        working_mask = ndimg.binary_erosion(working_mask, structure=grow_kern)

    # Write to file. Use 32-bit integer for widest support (iraf etc).
    if out_file is not None:
        fits.writeto(out_file, data=working_mask.astype('int32'),
                     clobber=True)
    return working_mask


def calc_rms(image_data, weight_data=None, sky_bin=4, sky_size=25,
             autocorr_aper=5, autocorr_bg_annulus_radii=(5, 7),
             mask_smooth_size=3, mask_sigma=1.5, mask_grow_size=3,
             use_rms_filt=False, rms_filt_size=7, log_file=None):
    """
    Calculate the pixel-to-pixel RMS noise in an astronomical image, taking
    into account autocorrelation, noise variations in the weight map, and
    masking out the objects themselves.

    :param image_data: 2-dimensional numpy array containing image data
    :param weight_data: 2-dimensional numpy array with weight map for image
        data. (e.g. weight map output from astrodrizzle)
    :param sky_bin: Binning factor to use before median-filtering when
        subtracting the sky
    :param sky_size: Size of median filter kernel for subtracting local sky.
        Effective kernel size will be sky_size*sky_bin
    :param autocorr_aper: Aperture used in measuring the autocorrelation
    :param autocorr_bg_annulus_radii: Inner and outer radii of background
        annulus for autocorrelation measurement
    :param mask_smooth_size: Size of median filter kernel for smoothing object
        mask
    :param mask_sigma: Sigma threshold for masking out objects
    :param mask_grow_size: Size of structuring element used to dilate masked
        objects
    :param use_rms_filt: Flag to enable using RMS filter to make noise map
    :param rms_filt_size: Size of RMS filter kernel for making noise map
    :param log_file: Name of log file to append output to
    :return: dictionary of string-keyed float values containing the following
        key : value pairs
        autocorr_rms   : RMS calculated from autocorrelation function
        corr_factor    : RMS supression factor (correlation factor)
        weight_nominal : Correlation-corrected inverse variance. A weight map
            can be created with this value.
        frac_flagged   : Fraction of pixels masked out of the input image when
            calculating RMS (Object pixels, zero-weight pixels, etc.)
        if use_rms_filt was specified,
        measured_rms   : RMS measured from rms-filtered image
        if weight_data was supplied,
        weight_avg     : Average value of supplied weight map
        weight_scale   : Factor to scale supplied weight map by to create
            inverse variance map
    """
    working_data = image_data.copy()
    bp_mask = ~np.isfinite(image_data)

    # Use weight map to normalize noise: drz_sci*sqrt(norm(drz_weight))
    if weight_data is not None:
        # Mask out pixels with zero weight
        bp_mask |= (weight_data <= 0) | ~np.isfinite(weight_data)
        weight_data[bp_mask] = 0.0
        weight_avg = np.mean(weight_data)
        working_data *= np.sqrt(weight_data / weight_avg)

    # Subtract the sky. Avoids doing it twice, once for object masking and
    # once here
    working_data -= _binned_median_filter(working_data, sky_bin, sky_size)
    working_data[bp_mask] = 0.0

    # Mask out objects using sigma thresholding
    if mask_sigma > 0:
        bp_mask |= object_mask(working_data, sky_size=0,
                               smooth_size=mask_smooth_size,
                               threshold=mask_sigma,
                               grow_radius=mask_grow_size)

    frac_flagged = bp_mask.sum() / image_data.size

    # Zero out non-sky pixels (including objects, zero-weight pixels)
    working_data[bp_mask] = 0.0

    # Measure autocorrelation function rms and correlation factor
    autocorr_rms, corr_factor = _measure_autocorrelation(
        working_data, aper_radius=autocorr_aper,
        bg_annulus_radii=autocorr_bg_annulus_radii)
    # Correct for masked-out pixels
    autocorr_rms /= np.sqrt(1.0 - frac_flagged)
    measured_rms = autocorr_rms

    # Use rms filter to make noise map, if commanded
    if use_rms_filt:
        rms_filtered = _rms_filter(image_data, filt_size=rms_filt_size)
        # dilate the mask by the structuring element used for median filter
        struct_elem = np.ones((rms_filt_size, rms_filt_size))
        bp_mask = ndimg.binary_dilation(bp_mask, structure=struct_elem)
        bp_mask |= rms_filtered <= 0.0  # Reject weights < 0
        # Find histogram peak of the masked, rms-filtered pixels
        med_rms, rms_rms = _pixel_stats(rms_filtered[~bp_mask], clip_sig=3,
                                        n_clip=10)
        measured_rms = med_rms

    # Compute weight map scaling
    weight_nominal = 1.0 / (measured_rms * corr_factor) ** 2

    # Construct output dictionary
    rms_info = {'frac_flagged': frac_flagged,
                'autocorr_rms': autocorr_rms,
                'corr_factor': corr_factor,
                'weight_nominal': weight_nominal}
    if use_rms_filt:
        rms_info['measured_rms'] = measured_rms
    if weight_data is not None:
        weight_scale = weight_nominal / weight_avg
        rms_info['weight_avg'] = weight_avg
        rms_info['weight_scale'] = weight_scale

    # Log output
    if log_file is not None:
        with open(log_file, 'a') as log:
            log.write('Python calc_rms run ended at {}\n'.format(
                str(datetime.now())))
            for key, value in rms_info.iteritems():
                log.write('{:0.5f}\t{}\n'.format(value, _log_info[key]))
    return rms_info


def select_region_slices(sci_data, badpix_mask, box_size=256, num_boxes=10):
    """
    Find the optimal regions for calculating the noise autocorrelation (areas
    with the fewest objects/least signal)

    :param sci_data: Science image data array
    :param badpix_mask: Boolean array that is True for bad pixels
    :param box_size: Size of the (square) boxes to select
    :param num_boxes: Number of boxes to select. If insufficient good pixels can
        be found, will return as many boxes as possible
    :return: List of 2D slices, tuples of (slice_y, slice_x)
    """
    # TODO: For large images this is pretty slow, due to 3 full-size filters
    img_boxcar = ndimg.uniform_filter(sci_data, size=box_size)

    # Smooth over mask with min filter, to ignore small areas of bad pixels
    # One tenth of box size in each dimension means ignoring bad pixel regions
    # comprising <1% of total box pixels
    smooth_size = box_size // 10
    badpix_mask = ndimg.minimum_filter(badpix_mask, size=smooth_size,
                                       mode='constant', cval=False)
    # Expand zone of avoidance of bad pixels, so we don't pick boxes that
    # contain them. mode=constant, cval=True means treat all borders as
    # if they were masked-out pixels
    badpix_mask = ndimg.maximum_filter(badpix_mask, size=smooth_size + box_size,
                                       mode='constant', cval=True)
    img_boxcar = np.ma.array(img_boxcar, mask=badpix_mask)

    box_slices = []
    for box in xrange(num_boxes):
        # Find the location of the minimum value of the boxcar image, excluding
        # masked areas. This will be a pixel with few nearby sources within one
        # box width
        min_loc = img_boxcar.argmin()
        min_loc = np.unravel_index(min_loc, img_boxcar.shape)
        lower_left = tuple(int(x - box_size / 2) for x in min_loc)
        # Negative values of lower_left mean argmin ran out of unmasked pixels
        if lower_left[0] < 0 or lower_left[1] < 0:
            warn('Ran out of good pixels when placing RMS calculation regions '
                 'for file {}. Only {:d} boxes selected.'.format(sci_data, box))
            break
        min_slice = tuple(slice(x, x + box_size) for x in lower_left)
        box_slices += [min_slice]

        # Zone of avoidance (for center) is twice as big, since we are picking
        # box centers. Use clip to ensure avoidance slice stays within array
        # bounds
        lower_left = tuple(int(x - box_size) for x in min_loc)
        avoid_slice = tuple(slice(np.clip(x, 0, extent),
                                  np.clip(x + 2 * box_size, 0, extent))
                            for x, extent in zip(lower_left, img_boxcar.shape))

        # Add this box to the mask
        img_boxcar[avoid_slice] = np.ma.masked
    return box_slices


def create_error_map(sci_file, weight_file, out_file, map_type='ivm',
                     bad_px_value=1e6, return_stats=False, **kwargs):
    """
    Create an autocorrelation-corrected error map for the given pair of science
    and weight files.

    :param sci_file: Science image file
    :param weight_file: Weight map file (for instance from astrodrizzle)
    :param out_file: Name of the output file
    :param map_type: Output error map type. One of ivm, var, or rms.
    :param bad_px_value: Output bad pixel value for map_type var or rms
    :param return_stats: If True, calculate and return statistics for the good
        pixels in the saved error map
    :param kwargs: Additional keyword arguments are passed to calc_rms() and
        select_region_slices(). There are many possible control parameters, see
        docstrings for those functions.
    :return: Filename of created error map, or dictionary of statistics if
        return_stats is True
    """
    # Split kwargs into those for calc_rms and those for select_region_slices
    calc_rms_kwargs = {k: v for k, v in kwargs.iteritems()
                       if k in getargspec(calc_rms).args}
    select_region_kwargs = {k: v for k, v in kwargs.iteritems()
                            if k in getargspec(select_region_slices).args}

    # Open drizzled image and weight image
    sci_data = fits.getdata(sci_file)
    weight_image = fits.open(weight_file, mode='readonly')
    weight_data = weight_image[0].data
    bpmask = weight_data <= 0
    bpmask |= ~np.isfinite(weight_data)
    bpmask |= ~np.isfinite(sci_data)

    # Find appropriate regions to calculate noise properties
    calc_slices = select_region_slices(sci_data, bpmask,
                                       **select_region_kwargs)

    weight_to_ivm_scale = 0.0
    # Calculate weightmap to rmsmap scaling for each slice, average
    print('Autocorrelating regions for {}:'.format(sci_file))
    for slice_y, slice_x in calc_slices:
        autocorr_dict = calc_rms(sci_data[slice_y, slice_x],
                                 weight_data[slice_y, slice_x],
                                 **calc_rms_kwargs)
        weight_to_ivm_scale += autocorr_dict['weight_scale'] / len(calc_slices)
        slice_str = '[{:d}:{:d},{:d}:{:d}]'.format(slice_x.start, slice_x.stop,
                                                   slice_y.start, slice_y.stop)
        print('RMS scale from region {}: {:.5f}'.format(
            slice_str, autocorr_dict['weight_scale']))

    # Bad pixels are those with 0 or negative weight.
    weight_data *= weight_to_ivm_scale
    weight_data[bpmask] = 0
    if map_type in ('rms',):
        weight_data = np.sqrt(weight_data)
    if map_type in ('var', 'rms'):
        weight_data = np.divide(1.0, weight_data)
        weight_data[bpmask] = bad_px_value

    # Save file
    fits.writeto(out_file, header=weight_image[0].header,
                 data=weight_data, clobber=True)
    weight_image.close()

    if return_stats:
        stats_pixels = weight_data[~bpmask]
        return {'image': out_file,
                'mean': np.mean(stats_pixels),
                'stddev': np.std(stats_pixels),
                'min': np.min(stats_pixels),
                'max': np.max(stats_pixels),
                'numpix': stats_pixels.size}
    else:
        return out_file
