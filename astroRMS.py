"""
Matt Mechtley - Dec 2011 - https://github.com/mmechtley
Python implementation of Mark Dickinson's iraf tasks for calculating
pixel-to-pixel RMS noise in astronomical images, taking into account
autocorrelation, noise variations in the weight map, and masking out
the objects themselves.

These functions are not intended to have exact parameter parity with the
MEDRMS iraf package. Rather, they are intended to provide similar
functionality with a Pythonic interface

Example usage:
import pyfits
import astroRMS
from numpy import sqrt

sciData = pyfits.getdata('myimage_drz_sci.fits')
weightData = pyfits.getdata('myimage_drz_weight.fits')
calcSection = (slice(400,600), slice(200,400))
rmsInfo = astroRMS.calc_RMS(sciData[calcSection], weightData[calcSection])

rmsData = 1/sqrt(rmsInfo['weightScale'] * weightData)
pyfits.writeto('myimage_drz_rms.fits', rmsData)
"""

## separate float division: / and floor division: //, See PEP 238
from __future__ import division
import pyfits
import numpy as np
from numpy.fft import fftn, ifftn, fftshift
from scipy.ndimage import median_filter, uniform_filter, binary_dilation, binary_erosion, minimum_filter, maximum_filter
import datetime

## TODO: Use minimum/maximum_filter in place of binary_dilation/erosion? Time these.

_log_info = {
	'autocorrRMS': 'RMS calculated from autocorrelation function',
	'measuredRMS': 'RMS measured from rms-filtered image',
	'corrFactor': 'RMS supression factor (correlation factor)',
	'weightNominal': 'Correlation-corrected inverse variance. A weight map can be created with this value',
	'weightAvg': 'Average value of supplied weight map',
	'weightScale': 'Factor to scale supplied weight map by to create inverse variance map',
	'fracFlagged': 'Fraction of pixels masked out of the input image when calculating RMS (Object pixels, zero-weight pixels, etc.)'
}

def object_mask(input_data, sky_bin=4, sky_size=15, smooth_size=3,
                thresh_type='sigma', threshold=1.5, num_clips=10, grow_size=0,
                out_file=None):
	"""
	Creates a numpy array representing a mask of the object pixels in an input
	astronomical image. Uses sigma-based feature thresholding. Objects will be
	labeled with 1's, background pixels labeled with 0's.

	Parameters
	----------
	input_data : numpy 2-D array
		Input image data
	sky_bin : integer
		Binning factor to use before median-filtering to subtract the sky
	sky_size : integer
		Size of median filter kernel for subtracting local sky. Effective
		kernel size will be sky_size*sky_bin. If 0, image must already be
		sky-subtracted.
	smooth_size : integer
		Boxcar filter size for smoothing before thresholding
	thresh_type : string
		Either 'sigma' or 'constant' for sigma or constant-value thresholding
	threshold : float
		Threshold value, either in sigma above mean or constant value
	num_clips : integer
		Number of times to apply histogram clip when calculating sky med/std
	grow_size : integer
		Width of rings to grow around masked objects

	Returns
	-------
	mask : integer numpy 2-D array
		numpy array with objects labeled with value '1', sky with '0'
	"""
	## TODO: Accept input bad pixel mask?

	## Raise error if boxcar smoothing parameters are not odd numbers
	if (sky_size > 0 and sky_size % 2 != 1) or (smooth_size > 0 and smooth_size % 2 != 1):
		raise ValueError('sky_size and smooth_size must both be odd integers, or 0')

	## Raise error if a bad keyword is given for thresh_type
	if thresh_type not in ['sigma', 'constant']:
		raise ValueError("thresh_type must be one of 'sigma' or 'constant'")

	## Create a local copy of the image data in memory (don't muck with original)
	workingData = input_data.copy()

	## If sky_size is supplied, create a median image of the local sky value, and subtract that off
	if sky_size > 0:
		workingData -= binned_median_filter(workingData, sky_bin, sky_size)

	## Calculate image statistics to determine median sky level and RMS noise
	statsMask = np.ones(workingData.shape, dtype=bool)
	for step in xrange(num_clips):
		sky, rms = (np.median(workingData[statsMask]), np.std(workingData[statsMask]))
		statsMask = (workingData < sky + 5*rms) & (workingData > sky - 5*rms)

	## Uniform filter before thresholding, if commanded
	if smooth_size > 0:
		workingData = uniform_filter(workingData, size=smooth_size)

	## Do the thresholding
	if thresh_type == 'sigma':
		threshold *= rms
	workingData = np.where(workingData < sky + threshold, 0, 1).astype(np.uint8)

	## Dilate or Erode objects if commanded
	growKern = np.ones((2*grow_size + 1, 2*grow_size + 1))
	x,y = np.meshgrid(range(2*grow_size + 1), range(2*grow_size + 1))
	growKern[(x-grow_size)**2 + (y-grow_size)**2 > grow_size**2] = 0
	if grow_size > 0:
		workingData = binary_dilation(workingData, structure=growKern)
	elif grow_size < 0:
		workingData = binary_erosion(workingData, structure=growKern)

	## Write to file if requested
	if out_file is not None:
		pyfits.writeto(out_file, data=workingData, clobber=True)

	return workingData

def autocorrelation_phot(autocorr_image, aper_radius=5, bg_annulus_radii=(5,7)):
	"""
	Uses the autocorrelation peak to calculate pixelwise RMS and correlation
	factor. This is an aperture photometry-like algorithm

	Parameters
	----------
	autocorr_image : numpy 2-D array
		An autocorrelation image, with 0th order in the center
	aper_radius : float
		Radius of the aperture to use for finding total flux of the
		autocorrelation peak
	bg_annulus_radii : float 2-tuple
		Inner and outer radius of sky annulus. NOT inner radius and annulus
		width as in IRAF.

	Returns
	-------
	(rms, corrFactor) : float 2-tuple
		pixelwise RMS and autocorrelation factor
	"""
	## TODO: Improve memory efficiency and numerical accuracy

	## Get image shape and center coordinates
	imShape = autocorr_image.shape
	center = tuple(dimSize/2 - 0.5 for dimSize in imShape)

	## generate x,y coordinates of pixels, and square distance to center of each
	x,y = np.meshgrid(range(imShape[1]), range(imShape[0]))
	sqDist = (x-center[1])**2 + (y-center[0])**2

	## 'sky' or background mask is an annulus around the center
	skyMask = (sqDist > bg_annulus_radii[0]**2) & (sqDist < bg_annulus_radii[1]**2)
	skyFixMask = ((sqDist > (bg_annulus_radii[0]-1)**2) &
	              (sqDist < (bg_annulus_radii[1]+1)**2)) & ~skyMask
	skyFixArea = (np.pi*(bg_annulus_radii[1]**2 - bg_annulus_radii[0]**2)
	              - autocorr_image[skyMask].size)
	skyWts = (np.where(skyMask, 1, 0)
	          + np.where(skyFixMask, skyFixArea/autocorr_image[skyFixMask].size, 0))
	## 'Flux' or measurement mask is a circle around the center
	fluxMask = (sqDist < aper_radius**2)
	fluxFixMask = (sqDist < (aper_radius+1)**2) & ~fluxMask
	fluxFixArea  = np.pi*aper_radius**2 - autocorr_image[fluxMask].size
	fluxWts = (np.where(fluxMask, 1, 0)
	           + np.where(fluxFixMask, fluxFixArea/autocorr_image[fluxFixMask].size, 0))

	## Calculate RMS and autocorrelation factor based on peak, background, and
	## integrated magnitude of the autocorrelation peak
	peakVal = np.max(autocorr_image[fluxMask])
	bgVal = np.average(autocorr_image, weights=skyWts)
	totalCorr = np.sum((autocorr_image - bgVal)*fluxWts)
	calcRMS = np.sqrt((peakVal - bgVal) / autocorr_image.size)
	corrFac = np.sqrt(totalCorr / (peakVal - bgVal))

	return calcRMS, corrFac

def autocorrelation(image_data, real_only=True):
	"""
	Computes the autocorrelation of input image data using fft.

	Parameters
	----------
	image_data : numpy n-D array
		n-dimensional image data to autocorrelate
	real_only : boolean
		return only the real part (amplitude) of autocorrelation

	Returns
	-------
	autoCorr : numpy n-D array
		autocorrelation of input image_data
	"""
	fftData = fftn(image_data)
	autoCorr = ifftn(fftData*np.conjugate(fftData))
	if real_only:
		autoCorr = autoCorr.real
	return fftshift(autoCorr)

def rms_filter(image_data, filt_size=7):
	"""
	Runs an 'RMS filter' on image data. Really this is the square root of the
	(appropriately scaled) difference of the squared uniform-filtered image and
	the uniform-filtered squared image. i.e. f*sqrt(boxcar(i**2) - boxcar(i)**2)

	Parameters
	----------
	image_data : numpy 2-D array
		Image data to be filtered
	filt_size : integer
		Size of uniform filter kernel to use

	Returns
	-------
	filtered : numpy 2-D array
		RMS-filtered version of input image
	"""
	medianSqr = uniform_filter(image_data**2, size=filt_size)
	sqrMedian = uniform_filter(image_data, size=filt_size)**2
	sqrDiffScale = filt_size**2 / (filt_size**2 - 1)
	return np.sqrt(np.abs(sqrDiffScale * (sqrMedian - medianSqr)))

def binned_median_filter(image_data, bin_size=4, filt_size=25):
	"""
	Implements a median filter that block-averages the image data before
	applying the filter.

	Parameters
	----------
	image_data : numpy 2-D array
		2-dimensional numpy array containing image data
	bin_size : integer
		Binning factor to use before median-filtering. image_data.shape must
		be an integer multiple of bin_size in both directions.
	filt_size : integer
		Size of median filter kernel. Effective kernel size in the returned
		image will be filt_size*bin_size

	Returns
	-------
	image_data_filtered : numpy 2-D array
		median-filtered version of input image
	"""
	imgShape = image_data.shape
	if (imgShape[0] % bin_size != 0 or imgShape[1] % bin_size != 0):
		raise ValueError('Size of image_data must be a multiple of bin_size '+
		                 'in both dimensions')
	if bin_size > 1:
		binned  = np.add.reduceat(
			np.add.reduceat(image_data, np.arange(0, imgShape[0], bin_size), axis=0),
			np.arange(0, imgShape[1], bin_size), axis=1) / bin_size**2
	else:
		binned = image_data

	binned = median_filter(binned, size=filt_size, mode='nearest')
	return np.repeat(np.repeat(binned, bin_size, axis=0), bin_size, axis=1)

def calc_RMS(image_data, weight_data=None, sky_bin=4, sky_size=25,
            autocorr_aper=5, autocorr_bg_annulus_radii=(5,7),
            mask_smooth_size=3, mask_sigma=1.5, mask_grow_size=3,
            use_rms_filt=False, rms_filt_size=7, log_file=None):
	"""
	Calculate the pixel-to-pixel RMS noise in an astronomical image, taking
	into account autocorrelation, noise variations in the weight map, and
	masking out the objects themselves.

	Parameters
	----------
	image_data : numpy 2-D array
		2-dimensional numpy array containing image data
	weight_data : numpy 2-D array
		2-dimensional numpy array with weight map for image data. (e.g.
		weight map output from Multidrizzle)
	sky_bin : integer
		Binning factor to use before median-filtering to subtract the sky
	sky_size : integer
		Size of median filter kernel for subtracting local sky. Effective
		kernel size will be sky_size*sky_bin
	autocorr_aper : float
		Aperture used in performing photometry on autocorrelation function
	autocorr_bg_annulus_radii : float 2-tuple
		Inner and outer radii of sky annulus for autocorrelation photometry
	mask_smooth_size : integer
		Size of median filter kernel for smoothing object mask
	mask_sigma : float
		sigma threshold for masking out objects
	mask_grow_size : integer
		Size of structuring element used to dilate masked objects
	use_rms_filt : boolean
		Flag to enable using RMS filter to make noise map
	rms_filt_size : integer
		Size of RMS filter kernel for making noise map
	log_file : string
		Name of log file to append output to

	Returns
	-------
	rms_info : dictionary of string-keyed float values
		Containing the following key : value pairs
	autocorrRMS   : RMS calculated from autocorrelation function
	corrFactor    : RMS supression factor (correlation factor)
	weightNominal : Correlation-corrected inverse variance. A weight map
		can be created with this value.
	fracFlagged   : Fraction of pixels masked out of the input image when
		calculating RMS (Object pixels, zero-weight pixels, etc.)
	if use_rms_filt was specified,
	measuredRMS   : RMS measured from rms-filtered image
	if weight_data was supplied,
	weightAvg     : Average value of supplied weight map
	weightScale   : Factor to scale supplied weight map by to create
		inverse variance map
	"""
	workingData = image_data.copy()
	bpMask = np.zeros(workingData.shape, dtype=bool)

	## Use weight map to normalize noise -- drz_sci*sqrt(norm(drz_weight))
	if weight_data is not None:
		weightAvg = np.mean(weight_data)
		workingData *= np.sqrt(weight_data / weightAvg)
		## Mask out pixels with zero weight
		bpMask |= (weight_data == 0)

	# Subtract the sky. Avoids doing it twice -- once for object masking & once here
	workingData -= binned_median_filter(workingData, sky_bin, sky_size)

	## Mask out objects using sigma thresholding
	if mask_sigma > 0:
		bpMask |= object_mask(workingData, sky_size=0, smooth_size=mask_smooth_size,
		                   threshold=mask_sigma, grow_size=mask_grow_size).astype(bool)

	fracFlagged = image_data[bpMask].size / image_data.size

	## Zero out non-sky pixels (including objects, zero-weight pixels)
	workingData[bpMask] = 0.0

	## Compute autocorrelation function
	autoCorr = autocorrelation(workingData)
	## Measure autocorrelation function rms and correlation factor
	autocorrRMS, corrFactor = autocorrelation_phot(
		autoCorr, aper_radius=autocorr_aper,
		bg_annulus_radii=autocorr_bg_annulus_radii)
	## Correct for masked-out pixels
	autocorrRMS /= np.sqrt(1.0 - fracFlagged)
	measuredRMS = autocorrRMS
	
	## Use rms filter to make noise map, if commanded
	if use_rms_filt:
		rmsFiltered = rms_filter(image_data, filt_size=rms_filt_size)
		## dilate the mask by the same structuring element used for median filter
		dilatedMask = binary_dilation(
			bpMask, structure=np.ones((rms_filt_size, rms_filt_size)))
		## Find median of the masked, rms-filtered image, using iterative clipping
		for step in xrange(10):
			dilatedMask |= (rmsFiltered <= 0.0)   ## Reject pixels < 0. Do these exist?
			medRMS, rmsRMS = (np.median(rmsFiltered[~dilatedMask]),
			                  np.std(rmsFiltered[~dilatedMask]))
			dilatedMask |= ((rmsFiltered > medRMS + 5*rmsRMS) |
			               (rmsFiltered < medRMS - 5*rmsRMS))
		measuredRMS = medRMS

	## Compute weight map scaling
	weightNominal = 1.0 / (measuredRMS * corrFactor)**2
	if weight_data is not None:
		weightScale = weightNominal / weightAvg

	## Construct output dictionary
	rms_info = dict({'fracFlagged': fracFlagged, 'autocorrRMS': autocorrRMS,
	                   'corrFactor': corrFactor, 'weightNominal': weightNominal})
	if use_rms_filt:
		rms_info['measuredRMS'] = measuredRMS
	if weight_data is not None:
		rms_info['weightAvg'] = weightAvg
		rms_info['weightScale'] = weightScale

	## Log output
	if log_file is not None:
		with open(log_file, 'a') as log:
			log.write('Python calc_RMS run ended at %s\n' % str(datetime.now()))
			for key, value in rms_info.iteritems():
				log.write('%.5f\t%s\n' % (value, _log_info[key]))
	return rms_info
