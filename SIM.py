#!/usr/bin/env python3

import numpy as np
import argparse
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from astropy.visualization import wcsaxes

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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

def create_spectral_index_map(fits_files, output_file, cont_fits, rms, host_coords, maxlvl,dex):
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
    lower = spectral_index_map > -5
    upper = spectral_index_map < -0.1
    with fits.open(cont_fits) as hdu:
        contour_data = hdu[0].data
    sigma3 = contour_data >= 3*rms
    mask = upper & lower & sigma3
    spectral_index_map = np.where(mask, spectral_index_map, np.nan)
    spectral_index_error_map = np.where(mask, spectral_index_error_map, np.nan)
    chi2red_map = np.where(mask, chi2red_map, np.nan)
    
    header['RMS'] = rms
    header['MAX_LVL']=maxlvl
    header['DEX']=dex
    # write host coordinates if provided

    coord_table_hdu = None
    if host_coords is not None:
        ra_vals = [c.ra.deg for c in host_coords]
        dec_vals = [c.dec.deg for c in host_coords]
        col1 = fits.Column(name='RA', format='E', array=ra_vals)
        col2 = fits.Column(name='DEC', format='E', array=dec_vals)
        coord_table_hdu = fits.BinTableHDU.from_columns([col1, col2], name='HOST_COORDS')

    maps_hdu = fits.PrimaryHDU(header=header)

    hdu_sim = fits.ImageHDU(spectral_index_map,header=header, name='INDEX_MAP')
    hdu_errm = fits.ImageHDU(spectral_index_error_map, header=header, name='ERROR_MAP')
    hdu_chi2redm = fits.ImageHDU(chi2red_map,header=header, name='CHI2RED_MAP')
    hdu_im = fits.ImageHDU(contour_data, header=header, name='IMAGE')

    hdul = [maps_hdu, hdu_sim, hdu_errm, hdu_chi2redm, hdu_im]
    if coord_table_hdu is not None:
        hdul.append(coord_table_hdu)
    fits.HDUList(hdul).writeto(output_file, overwrite=True)

    return spectral_index_map, spectral_index_error_map, chi2red_map, header

def plot_maps(spectral_index_map, spectral_index_error_map, chi2red_map, header, cont_fits, host_coords, rms, maxlvl, dex):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': WCS(header,naxis=2)}) #add naxis=2?
    ax1, ax2, ax3 = axes
    
    with fits.open(cont_fits) as hdu:
        contour_header = hdu[0].header
        contour_data = hdu[0].data
    #rms = 3.2323823378589636e-05                 #J01445 4.929831199803229e-05
    levels = np.logspace(np.log10(3), np.log10(maxlvl), num=int((np.log10(maxlvl) - np.log10(3)) / dex +1)) * rms

    im1 = ax1.imshow(spectral_index_map, cmap='inferno') #gist_rainbow_r
    ax1.contour(contour_data, levels=levels, colors='black', linewidths=1.0, transform=ax1.get_transform(WCS(header, naxis=2)),alpha = 0.9)
    ax1.set_title('Spectral Index Map')
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.89,pad=0.01,aspect=40)
    #cbar1.set_label(r'$\alpha_{4.5GHz}^{11.5GHz}$')
    wcsaxes.add_beam(ax1, header=contour_header,alpha=0.9,pad=0.65,frame=True, facecolor=None, edgecolor=None)
    
    im2 = ax2.imshow(spectral_index_error_map, origin = 'lower', cmap='inferno', norm=LogNorm())
    ax2.contour(contour_data, levels=levels, colors='black', linewidths=1.0, transform=ax2.get_transform(WCS(header,naxis=2)),alpha = 0.9)
    ax2.set_title('Spectral Index error Map')
    ax2.coords[1].set_ticklabel_visible(False)
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.89,pad=0.01,aspect=40)
    #cbar2.set_label('Error')
    wcsaxes.add_beam(ax2, header=contour_header,alpha=0.9,pad=0.65,frame=True, facecolor=None, edgecolor=None)
    
    im3 = ax3.imshow(chi2red_map, origin = 'lower', cmap = 'viridis', norm=LogNorm())
    ax3.contour(contour_data, levels=levels, colors='black', linewidths=1.0, transform=ax3.get_transform(WCS(header,naxis=2)), alpha = 0.9)
    ax3.set_title('Reduced Chi-square Map')
    ax3.coords[1].set_ticklabel_visible(False)
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.89, pad=0.01,aspect=40)
    #cbar3.set_label('Chi_square')
    wcsaxes.add_beam(ax3, header=contour_header,alpha=0.9,pad=0.65,frame=True, facecolor=None, edgecolor=None)
    
    for ax in axes:
        ax.set_xlabel('RA (J2000)')
        ax.set_ylabel('DEC (J2000)')
        #plot host galaxy
        if host_coords is not None:
            for i, c in enumerate(host_coords, start=1):
                #ax.plot(c.ra.deg, c.dec.deg, marker='o', color='red',markerfacecolor='none',transform=ax.get_transform('fk5'), markersize=8)
                ax.plot(c.ra.deg, c.dec.deg, marker='o', color='grey',markeredgecolor='k',transform=ax.get_transform('fk5'), markersize=6)
                ax.text(c.ra.deg, c.dec.deg, f'   {i}', color='blue', transform=ax.get_transform('fk5'), fontsize=8, ha='left', va='top')

    plt.tight_layout()
    plt.savefig('SIM.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Spectral Index Map")
    parser.add_argument('--infits', type=str, required=True, help="Text file containing list of FITS files")
    parser.add_argument('--outfits', type=str, required=True, help="Output FITS file for spectral index map")
    parser.add_argument('--contour', type=str, required=True, help="FITS file for contour overlay")
    parser.add_argument('--rms', type=float, required=True, help="Background RMS of contour image")
    parser.add_argument('--lvl_dex', type=str, help='Max contour level and dex value')
    parser.add_argument('--coords', type=str, required=True, help='Path to the text file containing host galaxy coordinates')
    args = parser.parse_args()

    # Read in FITS files
    with open(args.infits, 'r') as file:
        fits_files = [line.strip() for line in file.readlines()]
    

    host_coords = None
    with open(args.coords, 'r') as file:
        coords = [line.strip().split(',') for line in file.readlines()]
        ra_vals = [float(ra) for ra, dec in coords]
        dec_vals = [float(dec) for ra, dec in coords]
        host_coords = SkyCoord(ra=ra_vals, dec=dec_vals, unit='deg', frame='fk5')
    
    maxlvl, dex = map(float, args.lvl_dex.split(','))

    spectral_index_map, spectral_error_map, chi2red_map, header = create_spectral_index_map(
            fits_files, args.outfits, args.contour, args.rms, host_coords, maxlvl, dex)
    plot_maps(spectral_index_map, spectral_error_map, chi2red_map, header, args.contour, 
            host_coords, args.rms, maxlvl, dex)

