import multiprocessing
import numpy as np
from astropy import wcs
from astropy.io import fits
from scipy.optimize import leastsq
import math
from glob import glob

def powerlaw(x, amp, index):
    return amp * (x ** index)

# define our (line) fitting function
fitfunc = lambda p, x: p[0] + p[1] * x
errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err

def fit_spectrum(freq_array, flux_array, flux_errors):
    pinit = [0.0, -0.5]
    fit = leastsq(errfunc, pinit, args=(freq_array, flux_array, flux_errors), full_output=1)
    covar = fit[1]
    if covar is not None:
        P = fit[0]
        residual = errfunc(P, freq_array, flux_array, flux_errors)
        chi2red = sum(np.power(residual, 2)) / (len(freq_array) - len(pinit))
        alpha = P[1]
        amp = np.exp(P[0])
        # Errors
        err_alpha = np.sqrt(covar[1][1])
        err_amp = np.sqrt(covar[0][0])
    else:
        chi2red = None
        alpha = None
        amp = None
        err_alpha = None
        err_amp = None
    return alpha, err_alpha, amp, err_amp, chi2red

def process_pixel(args):
    x, y, cube, freq_array, flux_errors = args
    flux_array = np.log(cube[:, y, x])
    return fit_spectrum(freq_array, flux_array, flux_errors)

pattern = '../rgcv/J01445_Combined_C_split_channel_block*.*_rgcv.fits'
pattern_fixed = '../rgcv/J01445_Combined_*_split_channel_block*fixed.fits'
pattern2 = '../rgcv/J01445_Combined_X_split_channel_block*fixed_rgcv.fits'

images = glob(pattern) + glob(pattern_fixed) + glob(pattern2)

layers = []
freqs = []

for image in images:
    hdu = fits.open(image)
    layers.append(hdu[0].data)
    try:
        freq = hdu[0].header["RESTFREQ"]
    except KeyError:
        freq = hdu[0].header["RESTFRQ"]
    freqs.append(freq)

cube = np.stack(layers)
xmin = 0
xmax = hdu[0].data.shape[1]
ymin = 0
ymax = hdu[0].data.shape[0]

# array of frequencies
freq_array = np.log(freqs)
# flux density calibration accuracy = 2%
flux_errors = 0.02 * np.ones(freq_array.shape)

# Prepare arguments for parallel processing
args = [(x, y, cube, freq_array, flux_errors) for x in range(xmin, xmax) for y in range(ymin, ymax)]

# In parallel, fit the spectral indices
with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    results = pool.map(process_pixel, args)

# Unpack results
alpha, err_alpha, amp, err_amp, chi2red = map(list, zip(*results))

# Convert to numpy arrays
alpha = np.array(alpha, dtype="float32")
err_alpha = np.array(err_alpha, dtype="float32")
amp = np.array(amp, dtype="float32")
err_amp = np.array(err_amp, dtype="float32")
chi2red = np.array(chi2red, dtype="float32")

spixmap = alpha.reshape((xmax - xmin, ymax - ymin))

hdu[0].data = np.ones(hdu[0].data.shape)
hdu[0].data[ymin:ymax, xmin:xmax] = spixmap.T
hdu.writeto("alpha.fits", overwrite=True)

