import numpy as np
import argparse
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.mlab as mlab

def WLLS(S1, f1, Serr1):
    f = np.log10(np.array(f1))
    S = np.log10(np.array(S1))
    Serr = np.array(Serr1) / np.array(S1) * np.log(10)
    
    X = np.column_stack((np.ones_like(f), f))
    W = np.diag(1 / Serr**2)
    XtWX = np.linalg.inv(X.T @ W @ X)
    beta = XtWX @ X.T @ W @ S
    r = S - X @ beta
    df = len(f) - 2
    sumR = np.sum(r**2)
    sigma = sumR / df
    beta_var = sigma * XtWX
    stderr = np.sqrt(np.diag(beta_var))
    
    return beta[0], beta[1], stderr[0], stderr[1]

def create_spectral_index_map(fits_files, output_file):
    # Read FITS files and gather data
    data_list = []
    header = None
    for f in fits_files:
        with fits.open(f) as hdu:
            data = hdu[0].data
            data_list.append(data)
            if header is None:
                header = hdu[0].header

    data_stack = np.array(data_list)
    frequencies = [fits.getval(f, 'RESTFRQ') for f in fits_files]

    # Create arrays to store the spectral index and errors
    spectral_index_map = np.zeros_like(data_stack[0])
    spectral_index_error_map = np.zeros_like(data_stack[0])

    # Iterate over each pixel to fit the power law
    for i in range(data_stack.shape[1]):
        for j in range(data_stack.shape[2]):
            flux_array = data_stack[:, i, j]
            flux_errors = np.ones_like(flux_array)  # Assume uniform errors for simplicity
            if np.any(flux_array <= 0):
                spectral_index_map[i, j] = np.nan
                spectral_index_error_map[i, j] = np.nan
            else:
                _, alpha, _, err_alpha = WLLS(flux_array, frequencies, flux_errors)
                spectral_index_map[i, j] = alpha
                spectral_index_error_map[i, j] = err_alpha

    # Write spectral index map to a new FITS file
    hdu = fits.PrimaryHDU(spectral_index_map, header=header)
    hdul = fits.HDUList([hdu])
    hdul.writeto(output_file, overwrite=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Spectral Index Map")
    parser.add_argument('--infits', type=str, required=True, help="Text file containing list of FITS files")
    parser.add_argument('--outfits', type=str, required=True, help="Output FITS file for spectral index map")
    args = parser.parse_args()

    # Read in FITS files
    with open(args.infits, 'r') as file:
        fits_files = [line.strip() for line in file.readlines()]

    create_spectral_index_map(fits_files, args.outfits)

