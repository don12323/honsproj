import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import glob


plt.style.use('seaborn-v0_8-bright')
plt.rcParams["font.family"] = "serif"

def plot_first_four_sources(fits_files, output_file):
    fig = plt.figure(figsize=(15, 0.8*15 * np.sqrt(2)))  
    labels = ['(a)', '(b)', '(c)', '(d)']

    for i, fits_file in enumerate(fits_files[:4]):
        with fits.open(fits_file) as hdul:
            header = hdul['INDEX_MAP'].header
            wcs = WCS(header)
            spectral_index_map = hdul['INDEX_MAP'].data
            spectral_index_error_map = hdul['ERROR_MAP'].data
            chi2red_map = hdul['CHI2RED_MAP'].data
            image = hdul['IMAGE'].data
            rms = hdul[0].header['RMS']

            host_coords = None
            if 'HGRA' in header and 'HGDEC' in header:
                hgra = header['HGRA']
                hgdec = header['HGDEC']
                host_coords = SkyCoord(ra=hgra * u.deg, dec=hgdec * u.deg, frame='fk5')

        contour_levels = [3*rms, 6*rms, 10*rms, 15*rms, 25*rms]

        # Plot spectral index map
        ax1 = fig.add_subplot(4, 3, i * 3 + 1, projection=wcs)
        im1 = ax1.imshow(spectral_index_map, origin='lower', cmap='gist_rainbow_r', interpolation='none')
        ax1.contour(image, levels=contour_levels, colors='black', alpha=0.5, transform=ax1.get_transform(wcs))
        ax1.tick_params(direction='in') #width=2
        if i == 0:
            ax1.set_title('Spectral Index')
        ax1.coords[0].set_axislabel('RA(J2000)')
        ax1.coords[1].set_axislabel('Dec(J2000)')
        cbar1 = fig.colorbar(im1, ax=ax1, orientation='vertical', pad=0.05, shrink=0.91)
        # Label
        ax1.text(0.02, 0.95, labels[i], transform=ax1.transAxes, fontsize=14, fontweight='bold',va='top', ha='left')
        #TODO axis width doesnt work
        for spine in ax1.spines.values():
            spine.set_linewidth(3)

        # Spectral index error map
        ax2 = fig.add_subplot(4, 3, i * 3 + 2, projection=wcs)
        im2 = ax2.imshow(spectral_index_error_map, origin='lower', cmap='gist_rainbow_r', interpolation='none', norm=LogNorm())
        ax2.contour(image, levels=contour_levels, colors='black', alpha=0.5, transform=ax2.get_transform(wcs))
        ax2.tick_params(direction='in')
        if i == 0:
            ax2.set_title('Spectral Index Error')
        ax2.coords[0].set_axislabel('RA(J2000)')
        ax2.coords[1].set_ticklabel_visible(False)
        ax2.coords[1].set_axislabel('')
        cbar2 = fig.colorbar(im2, ax=ax2, orientation='vertical', pad=0.05, shrink=0.91)
        
        # Chi2red map
        ax3 = fig.add_subplot(4, 3, i * 3 + 3, projection=wcs)
        im3 = ax3.imshow(chi2red_map, origin='lower', cmap='inferno', interpolation='none', norm=LogNorm())
        ax3.contour(image, levels=contour_levels, colors='black', alpha=0.5, transform=ax3.get_transform(wcs))
        ax3.tick_params(direction='in')
        if i == 0:
            ax3.set_title('Reduced Chi-Square')
        ax3.coords[0].set_axislabel('RA(J2000)')
        ax3.coords[1].set_ticklabel_visible(False)
        ax3.coords[1].set_axislabel('')
        cbar3 = fig.colorbar(im3, ax=ax3, orientation='vertical', pad=0.05, shrink=0.91)
        
        # Plot hg coords
        axes = [ax1, ax2, ax3]
        for ax in axes:
            if host_coords is not None:
                host_pixel_coords = WCS(header).world_to_pixel(host_coords)
                ax.plot(host_pixel_coords[0], host_pixel_coords[1], 'x', color='black', markersize=10)
        
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)

def plot_remaining_sources(fits_files, output_file):
    fig = plt.figure(figsize=(15, 0.8*3/4*15 * np.sqrt(2)))  # A4 size in inches (landscape)
    labels = ['(e)', '(f)', '(g)']
    for i, fits_file in enumerate(fits_files[4:]):
        with fits.open(fits_file) as hdul:
            header = hdul['INDEX_MAP'].header
            wcs = WCS(header)
            spectral_index_map = hdul['INDEX_MAP'].data
            spectral_index_error_map = hdul['ERROR_MAP'].data
            chi2red_map = hdul['CHI2RED_MAP'].data
            image = hdul['IMAGE'].data
            rms = hdul[0].header['RMS']
        
            host_coords = None
            if 'HGRA' in header and 'HGDEC' in header:
                hgra = header['HGRA']
                hgdec = header['HGDEC']
                host_coords = SkyCoord(ra=hgra * u.deg, dec=hgdec * u.deg, frame='fk5')
        contour_levels = [3*rms, 6*rms, 10*rms, 15*rms, 25*rms]

        # spectral index map
        ax1 = fig.add_subplot(3, 3, i * 3 + 1, projection=wcs)
        im1 = ax1.imshow(spectral_index_map, origin='lower', cmap='gist_rainbow_r', interpolation='none')
        ax1.contour(image, levels=contour_levels, colors='black', alpha=0.5, transform=ax1.get_transform(wcs))
        ax1.tick_params(direction='in')
        if i == 0:
            ax1.set_title('Spectral Index')
        ax1.coords[0].set_axislabel('RA(J2000)')
        ax1.coords[1].set_axislabel('Dec(J2000)')
        cbar1 = fig.colorbar(im1, ax=ax1, orientation='vertical', pad=0.05, shrink=0.91)
        # Label
        ax1.text(0.02, 0.95, labels[i], transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
        # spectral index error map
        ax2 = fig.add_subplot(3, 3, i * 3 + 2, projection=wcs)
        im2 = ax2.imshow(spectral_index_error_map, origin='lower', cmap='gist_rainbow_r', interpolation='none', norm=LogNorm())
        ax2.contour(image, levels=contour_levels, colors='black', alpha=0.5, transform=ax2.get_transform(wcs))
        ax2.tick_params(direction='in')
        if i == 0:
            ax2.set_title('Spectral Index Error')
        ax2.coords[0].set_axislabel('RA(J2000)')
        ax2.coords[1].set_ticklabel_visible(False)
        ax2.coords[1].set_axislabel('')
        cbar2 = fig.colorbar(im2, ax=ax2, orientation='vertical', pad=0.05, shrink=0.91)
        
        # chi2red map
        ax3 = fig.add_subplot(3, 3, i * 3 + 3, projection=wcs)
        im3 = ax3.imshow(chi2red_map, origin='lower', cmap='inferno', interpolation='none', norm=LogNorm())
        ax3.contour(image, levels=contour_levels, colors='black', alpha=0.5, transform=ax3.get_transform(wcs))
        ax3.tick_params(direction='in')
        if i == 0:
            ax3.set_title('Reduced Chi-Square')
        ax3.coords[0].set_axislabel('RA(J2000)')
        ax3.coords[1].set_ticklabel_visible(False)
        ax3.coords[1].set_axislabel('')
        cbar3 = fig.colorbar(im3, ax=ax3, orientation='vertical', pad=0.05, shrink=0.91)

        # Plot hg coords
        axes = [ax1, ax2, ax3]
        for ax in axes:
            if host_coords is not None:
                host_pixel_coords = WCS(header).world_to_pixel(host_coords)
                ax.plot(host_pixel_coords[0], host_pixel_coords[1], 'x', color='black', markersize=10)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)

pattern1 = '/data/donwijesinghe/*_res/a.fits'
pattern2 = '/data/donwijesinghe/*_res/ma.fits'
fits_files1 = glob.glob(pattern1)
fits_files2 = glob.glob(pattern2)
fits_files = fits_files1 + fits_files2

output_file_1 = '../results/spectral_index_maps_1.pdf'
output_file_2 = '../results/spectral_index_maps_2.pdf'
plot_first_four_sources(fits_files, output_file_1)
plot_remaining_sources(fits_files, output_file_2)

