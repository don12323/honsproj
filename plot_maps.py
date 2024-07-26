import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.wcs import WCS
import glob

# Set the plot style and font
plt.style.use('seaborn-v0_8-bright')
plt.rcParams["font.family"] = "serif"

def plot_fits_files(fits_files_pattern, output_plot):
    fits_files = glob.glob(fits_files_pattern)
    
    # Filter out directories without 'a.fits' file
    fits_files = [f for f in fits_files if os.path.isfile(f)]
    
    # Calculate number of rows needed for the subplots (one row per source)
    num_sources = len(fits_files)
    
    if num_sources == 0:
        print("No valid FITS files found.")
        return

    # Define the figure size for A4 paper
    fig = plt.figure(figsize=(8.3, 11.7))  # A4 size in inches (landscape)
    
    # Create subplots
    num_cols = 3
    num_rows = num_sources
    plot_idx = 1
    row_idx = 1
    
    for i, fits_file in enumerate(fits_files):
        with fits.open(fits_file) as hdul:
            header = hdul['INDEX_MAP'].header
            wcs = WCS(header)
            spectral_index_map = hdul['INDEX_MAP'].data
            spectral_index_error_map = hdul['ERROR_MAP'].data
            chi2red_map = hdul['CHI2RED_MAP'].data
            original_image = hdul['IMAGE'].data
            rms = hdul[0].header['RMS']
        
        # Contour levels
        contour_levels = [3*rms, 6*rms, 10*rms, 15*rms, 25*rms]

        # Plot spectral index map
        ax1 = fig.add_subplot(num_rows, num_cols, plot_idx, projection=wcs)
        im1 = ax1.imshow(spectral_index_map, origin='lower', cmap='viridis', interpolation='none')
        ax1.contour(original_image, levels=contour_levels, colors='black', alpha=0.5, transform=ax1.get_transform(wcs))
        lon1 = ax1.coords[0]
        lat1 = ax1.coords[1]
        if i == 0:
            ax1.set_title('Spectral Index')
        #set xlabels
        if i < num_sources - 1:
            lon1.set_axislabel('')
        else: 
            ax1.set_xlabel('RA(J2000)')
        cbar1 = fig.colorbar(im1, ax=ax1, orientation='vertical', pad=0.05)
        plot_idx += 1

        # Plot spectral index error map
        ax2 = fig.add_subplot(num_rows, num_cols, plot_idx, projection=wcs)
        im2 = ax2.imshow(spectral_index_error_map, origin='lower', cmap='viridis', interpolation='none')
        ax2.contour(original_image, levels=contour_levels, colors='black', alpha=0.5, transform=ax2.get_transform(wcs))
        lon2 = ax2.coords[0]
        lat2 = ax2.coords[1]
        if i == 0:
            ax2.set_title('Spectral Index Error')
        #set xlabels
        if i < num_sources - 1:
            lon2.set_axislabel('')
        else:
            ax2.set_xlabel('RA(J2000)')
        cbar2 = fig.colorbar(im2, ax=ax2, orientation='vertical', pad=0.05)
        plot_idx += 1

        # Plot chi2red map
        ax3 = fig.add_subplot(num_rows, num_cols, plot_idx, projection=wcs)
        im3 = ax3.imshow(chi2red_map, origin='lower', cmap='viridis', interpolation='none', norm=LogNorm())
        ax3.contour(original_image, levels=contour_levels, colors='black', alpha=0.5, transform=ax3.get_transform(wcs))
        lon3 = ax3.coords[0]
        lat3 = ax3.coords[1]
        if i == 0:
            ax3.set_title('Reduced Chi-Square')
        #set xlabels
        if i < num_sources - 1:
            lon2.set_axislabel('')
        else:
            ax3.set_xlabel('RA(J2000)')
        cbar3 = fig.colorbar(im3, ax=ax3, orientation='vertical', pad=0.05)
        plot_idx += 1

    # Adjust layout to fit in A4
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.show()

# Example usage
fits_files_pattern = '/data/donwijesinghe/*_res/a.fits'
output_plot = '../results/spectral_index_maps.pdf'
plot_fits_files(fits_files_pattern, output_plot)

