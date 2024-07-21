#!/usr/bin/env python3
import numpy as np
import argparse
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u

import matplotlib.pyplot as plt
import multiprocessing as mp

#style
plt.style.use('seaborn-v0_8-bright')
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.grid"] = False

#TODO take sigma from highest frequency?? hgih freq has best res, but beam size is still the same so take contours from non-rgcv fits best

#output these from hexreg.py? OR have this high freq fits file as an input so that bkg rms can be calculated.

#TODO mask the spectral indices
def WLLS(S1, f1, Serr1):

    f = np.log10(f1)
    S = np.log10(S1)
    Serr = Serr1 / S1 * np.log(10)
    
    X = np.column_stack((np.ones_like(f), f))
    W = np.diag(1 / Serr**2)
    XtWX = np.linalg.inv(X.T @ W @ X)
    beta = XtWX @ X.T @ W @ S    # (XTWX)B = XTWy
    r = S - X @ beta
    df = len(f) - 2
    sumR = np.sum(r**2)
    sigma = sumR / df
    beta_var = sigma * XtWX
    stderr = np.sqrt(np.diag(beta_var))
    chi2red = (r.T @ W @ r) / df         
    #fit_fluxes = (10.0 **beta[0]) * f1 ** beta[1]
    #chi2 = np.sum(((S1 - fit_fluxes) / Serr1) ** 2)
    #chi2red = chi2 / (len(f1) -2)
    
    return beta[0], beta[1], stderr[0], stderr[1], chi2red

def process_pixel(args):
    i, j, data_stack, frequencies = args
    flux_array = data_stack[:, i, j]
    flux_errors = 0.02*flux_array        # Calib error
    if np.any(flux_array <= 0):
        return i, j, np.nan, np.nan, np.nan
    else:
        _, alpha, _, err_alpha, chi2red = WLLS(flux_array, frequencies, flux_errors)
        return i, j, alpha, err_alpha, chi2red

def create_spectral_index_map(fits_files, output_file):
    # Read FITS files and gather data
    data_list = []
    frequencies = []
    header = None
    for f in fits_files:
        with fits.open(f) as hdu:
            data = hdu[0].data
            data_list.append(data)
            #read frequencies
            try:
                freq = hdu[0].header['RESTFREQ']
            except KeyError:
                freq = hdu[0].header['RESTFRQ']
            frequencies.append(freq)

            if header is None:
                header = hdu[0].header

    data_stack = np.array(data_list)

    # Create arrays to store the spectral index and errors
    spectral_index_map = np.zeros_like(data_stack[0])
    spectral_index_error_map = np.zeros_like(data_stack[0])
    chi2red_map = np.zeros_like(data_stack[0])
    # Set up multiprocessing
    tasks = [(i, j, data_stack, frequencies) for i in range(data_stack.shape[1]) for j in range(data_stack.shape[2])]
    num_processes = mp.cpu_count()
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_pixel, tasks)

    # Collect results
    for i, j, alpha, err_alpha, chi2red in results:
        spectral_index_map[i, j] = alpha
        spectral_index_error_map[i, j] = err_alpha
        chi2red_map[i, j] = chi2red
    
    print(f"Min SEM: {np.nanmin(spectral_index_error_map)}, Max SEM: {np.nanmax(spectral_index_error_map)} ")
    print(f"Min chi2red: {np.nanmin(chi2red_map)}, Max chi2red: {np.nanmax(chi2red_map)}")
    print(f"mean chi2r: {np.nanmean(chi2red_map)}, mean SEM: {np.nanmean(spectral_index_error_map)}")
    # Add mask
    lower = spectral_index_map > -2.0
    upper = spectral_index_map < -0.5
    mask = upper & lower
    spectral_index_map = np.where(mask, spectral_index_map, np.nan)
    spectral_index_error_map = np.where(mask, spectral_index_error_map, np.nan)
    chi2red_map = np.where(mask, chi2red_map, np.nan)
    # Write spectral index map to a new FITS file
    hdu = fits.PrimaryHDU(spectral_index_map, header=header)
    hdul = fits.HDUList([hdu])
    hdul.writeto(output_file, overwrite=True)
    
    return spectral_index_map, spectral_index_error_map, chi2red_map, header

def plot_spectral_index_map(spectral_index_map, spectral_index_error_map, chi2red_map, header, cont_fits, host_coords):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), subplot_kw={'projection': WCS(header)})
    ax1, ax2, ax3 = axes
    
    with fits.open(cont_fits) as hdu:
        contour_data = hdu[0].data
    rms = 4.929831199803229e-05
    levels = [3*rms, 6*rms, 15*rms, 35*rms, 46*rms]

    im1 = ax1.imshow(spectral_index_map, cmap='gist_rainbow_r')
    ax1.contour(contour_data, levels=levels, colors='black', linewidths=1.0, transform=ax1.get_transform(WCS(header)),alpha = 0.5)
    ax1.set_title('Spectral Index Map')
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.75)
    cbar1.set_label(r'$\alpha_{6GHz}$')

    im2 = ax2.imshow(spectral_error_map, origin = 'lower', cmap='gist_rainbow_r')
    ax2.contour(contour_data, levels=levels, colors='black', linewidths=1.0, transform=ax2.get_transform(WCS(header)),alpha = 0.5)
    ax2.set_title('Spectral Index error Map')
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.75)
    cbar2.set_label('Error')
    
    im3 = ax3.imshow(chi2red_map, origin = 'lower', cmap = 'RdBu')
    ax3.contour(contour_data, levels=levels, colors='black', linewidths=1.0, transform=ax3.get_transform(WCS(header)), alpha = 0.5)
    ax3.set_title('Reduced Chi-square Map')
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.75)
    cbar3.set_label('Chi_square')

    for ax in axes:
        ax.set_xlabel('RA (J2000)')
        ax.set_ylabel('DEC (J2000)')
        #plot host galaxy
        if host_coords is not None:
            host_pixel_coords = WCS(header).world_to_pixel(host_coords)
            ax.plot(host_pixel_coords[0], host_pixel_coords[1], 'x', color='black', markersize=10)



    plt.tight_layout()
    plt.savefig('SIM.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Spectral Index Map")
    parser.add_argument('--infits', type=str, required=True, help="Text file containing list of FITS files")
    parser.add_argument('--outfits', type=str, required=True, help="Output FITS file for spectral index map")
    parser.add_argument('--contour', type=str, required=True, help="FITS file for contour overlay")
    parser.add_argument('--host_ra', type=float, help="RA of the host galaxy")
    parser.add_argument('--host_dec', type=float, help="DEC of the host galaxy")

    args = parser.parse_args()

    # Read in FITS files
    with open(args.infits, 'r') as file:
        fits_files = [line.strip() for line in file.readlines()]
    
    host_coords = None
    if args.host_ra is not None and args.host_dec is not None:
        host_coords = SkyCoord(ra=args.host_ra * u.deg, dec=args.host_dec * u.deg, frame='icrs')

    spectral_index_map, spectral_error_map, chi2red_map, header = create_spectral_index_map(fits_files, args.outfits)
    plot_spectral_index_map(spectral_index_map, spectral_error_map, chi2red_map, header, args.contour, host_coords)

